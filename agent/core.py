"""Цикл агента: LLM предлагает вызвать тул → выполняем → отдаём результат → … → ответ.

Один запрос пользователя = до agent_max_steps раундов. Обычно хватает двух:
вызвать тул и сформулировать ответ.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from agent.dialogue import direct_reply, starts_new_request
from agent.memory import ConversationMemory, HistoryMessage
from agent.policy import asks_to_read, explicitly_requests_change
from agent.prompt import build_system_prompt
from agent.tools import ALLOWED_TOOLS, MUTATING_TOOLS, McpToolbox, ToolResult
from common.display import format_calendar_time
from common.schemas import AgentResponse, AgentStatus, DraftAction, Event
from common.usage import record_usage

logger = logging.getLogger(__name__)

FALLBACK_ERROR = "Не получилось обработать запрос. Попробуйте ещё раз чуть позже."
TOO_MANY_STEPS = "Запутался в этом запросе. Попробуйте сформулировать проще."


class Agent:
    def __init__(
        self,
        llm: AsyncOpenAI,
        model: str,
        toolbox: McpToolbox,
        memory: ConversationMemory,
        tz: ZoneInfo,
        max_steps: int = 6,
    ) -> None:
        self._llm = llm
        self._model = model
        self._toolbox = toolbox
        self._memory = memory
        self._tz = tz
        self._max_steps = max_steps

    async def handle(
        self,
        user_id: str,
        text: str,
        now: datetime | None = None,
        *,
        confirm_changes: bool = False,
        request_id: UUID | None = None,
        trusted_edit_source: str | None = None,
    ) -> AgentResponse:
        now = now or datetime.now(self._tz)
        try:
            direct = direct_reply(text)
            if direct:
                reply, clear = direct
                if clear or starts_new_request(text):
                    await self._memory.reset(user_id)
                return AgentResponse(reply_text=reply, status=AgentStatus.DONE)
            history = await self._memory.load(user_id)
            if re.fullmatch(
                r"повтори\s+(?:сво[её]\s+)?последнее\s+сообщение[.!?]*", text.strip(), re.I
            ):
                previous = next(
                    (item["content"] for item in reversed(history) if item["role"] == "assistant"),
                    "Предыдущего ответа пока нет.",
                )
                return AgentResponse(
                    reply_text=_safe_model_reply(previous), status=AgentStatus.DONE
                )
            if starts_new_request(text) and history and history[-1].get("clarification_text"):
                history = []
            source_text = _user_source(history, text)
            if _keeps_conflicting_time(history, text):
                response = AgentResponse(
                    reply_text="Хорошо, оставляю события на выбранное время.",
                    status=AgentStatus.DONE,
                )
                await self._memory.append(
                    user_id,
                    HistoryMessage(role="user", content=text),
                    HistoryMessage(role="assistant", content=response.reply_text),
                )
                return response
            if trusted_edit_source:
                source_text = (trusted_edit_source + "\n" + text)[-6000:]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_system_prompt(now)},
                *({"role": item["role"], "content": item["content"]} for item in history),
                {"role": "user", "content": text},
            ]
            if trusted_edit_source:
                messages[-1]["content"] = (
                    "Исходная просьба для текущего черновика (данные):\n"
                    + trusted_edit_source
                    + "\nИсправление пользователя:\n"
                    + text
                )
            response = await self._run(
                user_id,
                messages,
                confirm_changes=confirm_changes,
                request_id=request_id,
                read_period=_read_period(text, now),
                source_text=source_text,
                deletion_period=_deletion_period(source_text, now),
            )
        except Exception:  # граница сервиса: бот должен получить ответ, а не 500
            logger.exception("LLM или MCP недоступны")
            return AgentResponse(reply_text=FALLBACK_ERROR, status=AgentStatus.ERROR)

        # В историю пишем только завершённые обмены: после ошибки повтор должен
        # начинаться с чистого листа, а не с половины старого запроса.
        if response.status != AgentStatus.ERROR:
            try:
                assistant = HistoryMessage(
                    role="assistant",
                    content=_with_event_refs(response),
                    clarification_text=(
                        source_text if response.status == AgentStatus.NEED_CLARIFICATION else ""
                    ),
                    conflict_warning=response._conflict_warning,
                )
                await self._memory.append(
                    user_id,
                    HistoryMessage(role="user", content=text),
                    assistant,
                )
            except Exception:
                # Запись события уже могла завершиться: не заставляем повторять её.
                logger.exception("Не удалось сохранить контекст диалога")
        return response

    async def reset(self, user_id: str) -> None:
        await self._memory.reset(user_id)

    async def execute_draft(self, user_id: str, draft: DraftAction) -> AgentResponse:
        if draft.tool not in MUTATING_TOOLS:
            return AgentResponse(
                reply_text="Не получилось выполнить: неизвестное действие.",
                status=AgentStatus.ERROR,
            )
        try:
            if draft.tool == "create_event":
                result = await self._toolbox.call(
                    user_id,
                    draft.tool,
                    draft.arguments,
                    idempotency_key=str(draft.id),
                    user_text=draft.arguments.get("user_text"),
                )
            elif draft.tool == "delete_events":
                result = await self._toolbox.call(
                    user_id,
                    draft.tool,
                    draft.arguments,
                    event_ids=draft.arguments.get("event_ids"),
                )
            else:
                result = await self._toolbox.call(user_id, draft.tool, draft.arguments)
            if not result.ok:
                return AgentResponse(
                    reply_text="Не получилось выполнить: "
                    + _public_tool_error(str(result.payload)),
                    status=AgentStatus.ERROR,
                )
            events = [Event.model_validate(raw) for raw in result.events]
            if any(event.user_id != user_id for event in events):
                raise ValueError("Foreign event")
            confirmed: dict[str, Event] = {}
            conflict_warnings: dict[str, Event] = {}
            for raw, event in zip(result.events, events, strict=True):
                confirmed[str(event.id)] = event
                for conflict in raw.get("conflicts") or []:
                    other = Event.model_validate(conflict)
                    if other.user_id != user_id:
                        raise ValueError("Foreign conflict event")
                    conflict_warnings[str(other.id)] = other
            summary = _confirmed_response(confirmed, conflict_warnings)
            response = AgentResponse(
                reply_text="Готово.\n" + summary.reply_text,
                events=summary.events,
                status=AgentStatus.DONE,
            )
            response._conflict_warning = bool(conflict_warnings)
            try:
                await self._memory.append(
                    user_id,
                    HistoryMessage(
                        role="assistant",
                        content=_with_event_refs(response),
                        conflict_warning=response._conflict_warning,
                    ),
                )
            except Exception:
                # The write is committed; a Redis history failure must not tell the
                # user that confirmation failed and encourage a duplicate retry.
                logger.exception("Не удалось сохранить подтверждённое действие в историю")
            return response
        except Exception:
            logger.exception("Ошибка подтверждения черновика")
            return AgentResponse(
                reply_text="Не получилось выполнить действие. Проверьте календарь перед повтором.",
                status=AgentStatus.ERROR,
            )

    async def _run(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        *,
        confirm_changes: bool = False,
        request_id: UUID | None = None,
        read_period: tuple[datetime, datetime] | None = None,
        source_text: str | None = None,
        deletion_period: tuple[datetime, datetime] | None = None,
    ) -> AgentResponse:
        tools = await self._toolbox.openai_tools()
        events: dict[str, Event] = {}
        mutations: dict[str, ToolResult] = {}
        confirmed: dict[str, Event] = {}
        conflict_warnings: dict[str, Event] = {}
        read_tools: set[str] = set()
        errors: dict[str, tuple[str, dict | None, str, int]] = {}
        mutation_number = 0
        draft: DraftAction | None = None
        draft_fingerprint: str | None = None

        for round_number in range(self._max_steps):
            usage_recorded = False
            try:
                completion = await self._llm.chat.completions.create(
                    model=self._model, messages=messages, tools=tools, temperature=0.1
                )
                record_usage(
                    completion, operation="agent", source="gpt2giga_openai", model=self._model
                )
                usage_recorded = True
                message = completion.choices[0].message
            except Exception:
                if not usage_recorded:
                    record_usage(
                        None, operation="agent", source="gpt2giga_openai", model=self._model
                    )
                if draft is not None and not errors:
                    return _draft_response(draft)
                if confirmed:
                    logger.exception("LLM недоступна после подтверждённой записи")
                    return (
                        _incomplete_response(events, confirmed, conflict_warnings, errors)
                        if errors
                        else _confirmed_response(confirmed, conflict_warnings)
                    )
                if read_tools and not errors and draft is None:
                    return _read_response(events, read_tools)
                if errors:
                    return _incomplete_response(events, confirmed, conflict_warnings, errors)
                raise

            if not message.tool_calls:
                if errors:
                    return _incomplete_response(events, confirmed, conflict_warnings, errors)
                reply = _safe_model_reply((message.content or "").strip())
                if draft:
                    return _draft_response(draft)
                if read_tools and not confirmed and draft is None:
                    return _read_response(events, read_tools)
                if confirmed:
                    return _confirmed_response(confirmed, conflict_warnings)
                if not reply:
                    return (
                        _confirmed_response(confirmed, conflict_warnings)
                        if confirmed
                        else AgentResponse(reply_text=FALLBACK_ERROR, status=AgentStatus.ERROR)
                    )
                return AgentResponse(
                    reply_text=reply, status=AgentStatus.DONE, events=list(events.values())
                )

            # content: "" вместо отсутствующего поля — не все OpenAI-совместимые API принимают null.
            messages.append(
                {**message.model_dump(exclude_none=True), "content": message.content or ""}
            )
            for call in message.tool_calls:
                name = call.function.name
                if name not in ALLOWED_TOOLS:
                    return AgentResponse(
                        reply_text="Этот инструмент недоступен. Я работаю только с календарём.",
                        status=AgentStatus.ERROR,
                    )
                args = _parse_args(call.function.arguments)
                evidence = {}
                if name == "create_event" and source_text is not None:
                    evidence["user_text"] = source_text
                    if args is not None:
                        args = {**args, "user_text": source_text}
                if name == "get_events" and args is not None and read_period is not None:
                    args = {
                        **args,
                        "start_time": read_period[0].isoformat(),
                        "end_time": read_period[1].isoformat(),
                    }
                logger.info("tool call %s", name)
                if name == "delete_events" and args is not None:
                    if deletion_period:
                        args = {
                            **args,
                            "start_time": deletion_period[0].isoformat(),
                            "end_time": deletion_period[1].isoformat(),
                        }
                    elif re.search(
                        r"сегодня|завтра|вчера|недел|месяц|\d|январ|феврал|март|апрел|"
                        r"июн|июл|август|сентябр|октябр|ноябр|декабр",
                        source_text or "",
                        re.I,
                    ) and not (args.get("start_time") and args.get("end_time")):
                        return AgentResponse(
                            status=AgentStatus.NEED_CLARIFICATION,
                            reply_text="За какой период удалить записи? Укажите даты.",
                        )

                if name == "ask_user":
                    question = _safe_model_reply(
                        str((args or {}).get("question") or "").strip() or "Уточните, пожалуйста."
                    )
                    return AgentResponse(
                        reply_text=question,
                        status=AgentStatus.NEED_CLARIFICATION,
                        events=list(events.values()),
                    )

                fingerprint = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
                if (
                    name in MUTATING_TOOLS
                    and not confirm_changes
                    and (read_tools or name == "delete_events" or asks_to_read(source_text or ""))
                    and not explicitly_requests_change(source_text or "", name)
                ):
                    return (
                        _read_response(events, read_tools)
                        if read_tools
                        else AgentResponse(
                            status=AgentStatus.ERROR,
                            reply_text="Не получилось выполнить запрос. Календарь не изменён.",
                        )
                    )
                # Массовое удаление всегда получает снимок ID. Для текста сервис
                # применяет его сразу; голосовой вариант ждёт кнопки пользователя.
                # Retrieved text can influence the model. It never grants write
                # permission, including within a batch of tool calls.
                preview = (
                    confirm_changes or bool(read_tools) or name == "delete_events"
                ) and name in MUTATING_TOOLS
                if preview and draft_fingerprint is not None and fingerprint != draft_fingerprint:
                    return AgentResponse(
                        reply_text=(
                            "За один раз можно подтвердить только одно действие. "
                            "Отправьте команды по одной, пожалуйста."
                        ),
                        status=AgentStatus.NEED_CLARIFICATION,
                    )
                result = mutations.get(fingerprint)
                if result is None:
                    try:
                        if args is None:
                            result = ToolResult(
                                ok=False, payload="Аргументы не являются корректным JSON"
                            )
                        elif preview:
                            result = await self._toolbox.call(
                                user_id, name, args, dry_run=True, **evidence
                            )
                        elif name == "create_event" and request_id is not None:
                            result = await self._toolbox.call(
                                user_id,
                                name,
                                args,
                                idempotency_key=str(uuid5(request_id, f"create:{mutation_number}")),
                                **evidence,
                            )
                        else:
                            result = await self._toolbox.call(user_id, name, args, **evidence)
                    except Exception:
                        if not confirmed:
                            raise
                        logger.exception("MCP недоступен после подтверждённой записи")
                        result = ToolResult(ok=False, payload="Инструмент временно недоступен")
                if result.ok:
                    if name in {"get_events", "search_events"}:
                        read_tools.add(name)
                    for failed_key, (failed_name, failed_args, _, failed_round) in list(
                        errors.items()
                    ):
                        if failed_key == fingerprint or (
                            round_number > failed_round
                            and _same_action(name, args, failed_name, failed_args)
                        ):
                            errors.pop(failed_key)
                else:
                    errors[fingerprint] = (name, args, str(result.payload), round_number)
                if preview and result.ok:
                    draft_args = (
                        {**args, "confirmed": True, "event_ids": result.payload["event_ids"]}
                        if name == "delete_events"
                        else args
                    )
                    draft = DraftAction(
                        tool=name,
                        arguments=draft_args,
                        preview=_draft_preview(user_id, name, result, args),
                    )
                    draft_fingerprint = fingerprint
                    mutations[fingerprint] = result
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result.for_llm()}
                    )
                    continue
                for raw in result.events:
                    event = Event.model_validate(raw)
                    if event.user_id != user_id:
                        raise ValueError("MCP вернул событие другого пользователя")
                    events[str(event.id)] = event
                    # Пересекающиеся записи тоже в сводку: «перенеси ту встречу» должно
                    # знать её id, даже если пользователь её в этом диалоге не создавал.
                    for conflict in raw.get("conflicts") or []:
                        other = Event.model_validate(conflict)
                        if other.user_id != user_id:
                            raise ValueError("MCP вернул пересечение другого пользователя")
                        events.setdefault(str(other.id), other)
                        if result.ok and result.is_mutation:
                            conflict_warnings[str(other.id)] = other
                    if result.ok and result.is_mutation:
                        confirmed[str(event.id)] = event
                if result.ok and result.is_mutation:
                    if fingerprint not in mutations:
                        mutation_number += 1
                    mutations[fingerprint] = result
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.for_llm()}
                )

            # The validated dry-run already contains every preview fact. Another
            # paid generation cannot add authority and could misdescribe it.
            if draft is not None and not errors:
                return _draft_response(draft)
            if (
                read_tools
                and not errors
                and not confirmed
                and not any(
                    explicitly_requests_change(source_text or "", tool) for tool in MUTATING_TOOLS
                )
            ):
                # Successful list/search results already contain every display fact.
                # Only explicit write requests need another interpretation step.
                return _read_response(events, read_tools)

        logger.warning("agent exceeded %d steps", self._max_steps)
        if errors:
            return _incomplete_response(events, confirmed, conflict_warnings, errors)
        if confirmed:
            return _confirmed_response(confirmed, conflict_warnings)
        if draft:
            return _draft_response(draft)
        return AgentResponse(reply_text=TOO_MANY_STEPS, status=AgentStatus.ERROR)


def _draft_response(draft: DraftAction) -> AgentResponse:
    return AgentResponse(
        reply_text=draft.preview, status=AgentStatus.NEEDS_CONFIRMATION, draft=draft
    )


def _draft_preview(user_id: str, name: str, result: ToolResult, arguments: dict) -> str:
    """The approved operation and its parameters come from validated tool data.

    The LLM cannot disguise a delete button as a read or change the shown time.
    JSON quoting keeps an event title from impersonating the following UI text.
    """
    if name == "delete_events":
        count = result.payload["count"]
        return f"Удалить записи: {count} шт.?\nПодтвердите действие или нажмите «Отмена»."
    verbs = {
        "create_event": "Создать",
        "update_event": "Изменить",
        "delete_event": "Удалить",
        "complete_event": "Отметить выполненной",
    }
    lines = []
    for raw in result.events:
        event = Event.model_validate(raw)
        if event.user_id != user_id:
            raise ValueError("Foreign preview event")
        title = json.dumps(event.name, ensure_ascii=False)
        kind = {"task": "задачу", "reminder": "напоминание", "meeting": "встречу"}[event.type]
        lines.append(f"{verbs[name]} {kind}: {title}, {_when(event)}.")
        if name in {"create_event", "update_event"} and (
            event.description or "description" in arguments
        ):
            lines.append(f"Описание: {json.dumps(event.description, ensure_ascii=False)}")
        for raw_conflict in raw.get("conflicts") or []:
            conflict = Event.model_validate(raw_conflict)
            if conflict.user_id != user_id:
                raise ValueError("Foreign preview conflict")
            lines.append(
                f"Пересечение: {json.dumps(conflict.name, ensure_ascii=False)}, {_when(conflict)}."
            )
    if not lines:
        raise ValueError("Missing preview facts")
    return "\n".join([*lines, "Подтвердите действие или нажмите «Исправить» / «Отмена»."])


def _user_source(history: list[HistoryMessage], text: str) -> str:
    """Keep actual user evidence across a pending clarification, not unrelated turns."""
    if starts_new_request(text):
        return text
    pending = ""
    if history and history[-1]["role"] == "assistant":
        last = history[-1]
        if "clarification_text" in last:
            pending = last["clarification_text"]
        elif last["content"].rstrip().endswith("?") and re.search(
            r"во\s+сколько|на\s+какое\s+время|на\s+какую\s+дату|какой\s+день|"
            r"о\s+ч[её]м|сколько\s+продлится|длительност|за\s+сколько|сколько\s+заранее",
            last["content"],
            re.I,
        ):
            # Compatibility with history saved before explicit status metadata.
            pending = next(
                (item["content"] for item in reversed(history[:-1]) if item["role"] == "user"),
                "",
            )
    return (pending + "\n" + text if pending else text)[-6000:]


def _deletion_period(text: str, now: datetime) -> tuple[datetime, datetime] | None:
    words = re.findall(r"\b(сегодня|завтра|послезавтра|вчера)\b", text.casefold())
    if len(words) == 1:
        day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days={"сегодня": 0, "завтра": 1, "послезавтра": 2, "вчера": -1}[words[0]]
        )
        return day, day + timedelta(days=1)
    return _read_period(text, now)


def _read_period(text: str, now: datetime) -> tuple[datetime, datetime] | None:
    """Resolve explicit inclusive calendar-day windows against the trusted clock.

    Multiple periods or explicit dates stay with the normal interpretation path.
    This never changes mutation or semantic-search arguments.
    """
    text = text.casefold()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rolling = list(
        re.finditer(
            r"\b(последни[ех]|следующи[ех])\s+(\d{1,3}|семь|тридцать)\s+"
            r"(?:календарных\s+)?д(?:ней|ня|ень)\s*[,—-]?\s*включая\s+сегодня\b",
            text,
        )
    )
    if len(rolling) == 1:
        match = rolling[0]
        rest = text[: match.start()] + text[match.end() :]
        if not re.search(
            r"\d|\b(?:вчера|завтра|позавчера|послезавтра|сегодня|недел\w*|месяц\w*)\b", rest
        ):
            number = match.group(2)
            days = {"семь": 7, "тридцать": 30}.get(number, int(number) if number.isdigit() else 0)
            if 1 <= days <= 366:
                return (
                    (today - timedelta(days=days - 1), today + timedelta(days=1))
                    if match.group(1).startswith("послед")
                    else (today, today + timedelta(days=days))
                )
    if len(re.findall(r"\bмесяц\w*", text)) != 1 or re.search(
        r"\d|\b(?:вчера|завтра|позавчера|послезавтра|недел\w*|прошл\w*|следующ\w*)\b", text
    ):
        return None
    if not re.search(
        r"(?:с\s+начала\s+(?:этого|текущего)\s+месяца|"
        r"в\s+(?:этом|текущем)\s+месяце)\s+по\s+сегодняшний\s+день\b",
        text,
    ):
        return None
    return today.replace(day=1), today + timedelta(days=1)


def _parse_args(raw: str | None) -> dict[str, Any] | None:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _same_action(
    name: str, args: dict | None, previous_name: str, previous_args: dict | None
) -> bool:
    if name != previous_name or args is None or previous_args is None:
        return False
    if name == "create_event":
        current = str(args.get("name", "")).casefold().strip()
        return bool(current) and current == str(previous_args.get("name", "")).casefold().strip()
    if name in MUTATING_TOOLS:
        return args.get("event_id") is not None and args.get("event_id") == previous_args.get(
            "event_id"
        )
    return True  # a later round may correct invalid read filters


def _incomplete_response(events, confirmed, conflicts, errors) -> AgentResponse:
    saved = _confirmed_response(confirmed, conflicts).reply_text if confirmed else ""
    return AgentResponse(
        reply_text=(saved + "\n" if saved else "")
        + " ".join(dict.fromkeys(_public_tool_error(error[2]) for error in errors.values())),
        status=AgentStatus.NEED_CLARIFICATION,
        events=list(events.values()),
    )


def _public_tool_error(message: str) -> str:
    if "удалять нечего" in message.casefold():
        return "Подходящих записей нет — удалять нечего."
    if "длительност" in message.casefold() or "duration_minutes" in message:
        return "Сколько продлится встреча?"
    if "нет времени, только день" in message:
        return "Во сколько запланировать событие?"
    if re.search(
        r"create_event|update_event|delete_event|user_text|ToolError|Traceback|"
        r"вызови\s+тул|системн\w*\s+промпт|скажи\s+пользователю",
        message,
        re.I,
    ):
        return "Не удалось выполнить действие. Уточните, что нужно сделать с календарём."
    return message


def _safe_model_reply(message: str) -> str:
    message = re.split(r"\[(?:черновик|записи в этом ответе):", message, maxsplit=1)[0].strip()
    if re.search(
        r"create_event|update_event|delete_event|duration_minutes|user_text|"
        r"ты\s*[—-]\s*голосовой\s+ассистент|системный\s+промпт\s*[:=]",
        message,
        re.I,
    ):
        return (
            _public_tool_error(message)
            if "duration_minutes" in message
            else ("Я помогаю с задачами, встречами и напоминаниями. Что запланировать или найти?")
        )
    return message


def _keeps_conflicting_time(history: list[HistoryMessage], text: str) -> bool:
    if not history or history[-1]["role"] != "assistant":
        return False
    normalized = re.sub(r"[^\w\s]", "", text.casefold()).strip()
    return history[-1].get("conflict_warning") is True and normalized in {
        "нет",
        "нет спасибо",
        "не надо",
        "оставь как есть",
        "оставь так",
        "ничего не меняй",
        "не переноси",
        "всё равно на пересечение",
        "все равно на пересечение",
        "время подходит",
        "нет время подходит",
    }


def _with_event_refs(response: AgentResponse) -> str:
    """Ответ + скрытая сводка событий: чтобы следующий запрос («перенеси её на завтра»)
    знал, о какой записи речь. Пользователь эту сводку не видит — только модель в истории.
    """
    if not response.events:
        return response.reply_text + (
            f"\n[черновик: {response.draft.model_dump_json()}]" if response.draft else ""
        )
    refs = "; ".join(f"id={e.id}, {e.type}, «{e.name}», {_when(e)}" for e in response.events[:10])
    return f"{response.reply_text}\n[записи в этом ответе: {refs}]"


def _when(event: Event) -> str:
    return format_calendar_time(event.start_at, event.end_at)


def _confirmed_response(
    events: dict[str, Event], conflicts: dict[str, Event] | None = None
) -> AgentResponse:
    lines = []
    for event in events.values():
        action = {"done": "Выполнено", "cancelled": "Отменено"}.get(event.status, "Сохранено")
        lines.append(f"{action}: {event.name}, {_when(event)}.")
    for event in (conflicts or {}).values():
        lines.append(f"Учтите: пересечение с «{event.name}», {_when(event)}.")
    response = AgentResponse(
        reply_text="\n".join(lines),
        status=AgentStatus.DONE,
        events=list({**events, **(conflicts or {})}.values()),
    )
    response._conflict_warning = bool(conflicts)
    return response


def _read_response(events: dict[str, Event], read_tools: set[str]) -> AgentResponse:
    """Render retrieved facts without letting the model rewrite dates or names."""
    items = list(events.values())
    search = "search_events" in read_tools
    if not items:
        reply = "Подходящих записей не найдено." if search else "За указанный период записей нет."
        return AgentResponse(reply_text=reply, status=AgentStatus.DONE)
    lines = [
        "Подходящие записи (выборка по релевантности):"
        if search
        else f"Найдено записей: {len(items)}."
    ]
    for event in items:
        when = _when(event)
        status = " (выполнено)" if event.status == "done" else ""
        lines.append(f"{when} — {event.name}{status}")
    return AgentResponse(reply_text="\n".join(lines), status=AgentStatus.DONE, events=items)
