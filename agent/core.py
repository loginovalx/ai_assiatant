"""Цикл агента: LLM предлагает вызвать тул → выполняем → отдаём результат → … → ответ.

Один запрос пользователя = до agent_max_steps раундов. Обычно хватает двух:
вызвать тул и сформулировать ответ.
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from agent.memory import ConversationMemory, HistoryMessage
from agent.prompt import build_system_prompt
from agent.tools import MUTATING_TOOLS, McpToolbox, ToolResult
from common.schemas import AgentResponse, AgentStatus, DraftAction, Event

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
    ) -> AgentResponse:
        now = now or datetime.now(self._tz)
        try:
            history = await self._memory.load(user_id)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_system_prompt(now)},
                *history,
                {"role": "user", "content": text},
            ]
            response = await self._run(
                user_id, messages, confirm_changes=confirm_changes, request_id=request_id
            )
        except Exception:  # граница сервиса: бот должен получить ответ, а не 500
            logger.exception("LLM или MCP недоступны")
            return AgentResponse(reply_text=FALLBACK_ERROR, status=AgentStatus.ERROR)

        # В историю пишем только завершённые обмены: после ошибки повтор должен
        # начинаться с чистого листа, а не с половины старого запроса.
        if response.status != AgentStatus.ERROR:
            try:
                await self._memory.append(
                    user_id,
                    HistoryMessage(role="user", content=text),
                    HistoryMessage(role="assistant", content=_with_event_refs(response)),
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
                    reply_text=f"Не получилось выполнить: {result.payload}",
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
            try:
                await self._memory.append(
                    user_id, HistoryMessage(role="assistant", content=_with_event_refs(response))
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
            try:
                completion = await self._llm.chat.completions.create(
                    model=self._model, messages=messages, tools=tools, temperature=0.1
                )
                message = completion.choices[0].message
            except Exception:
                if confirmed:
                    logger.exception("LLM недоступна после подтверждённой записи")
                    return (
                        _incomplete_response(events, confirmed, conflict_warnings, errors)
                        if errors
                        else _confirmed_response(confirmed, conflict_warnings)
                    )
                if read_tools and not errors and draft is None:
                    return _read_response(events, read_tools)
                raise

            if not message.tool_calls:
                if errors:
                    return _incomplete_response(events, confirmed, conflict_warnings, errors)
                reply = (message.content or "").strip()
                if draft and reply:
                    draft.preview = reply
                    return AgentResponse(
                        reply_text=reply, status=AgentStatus.NEEDS_CONFIRMATION, draft=draft
                    )
                if read_tools and not confirmed and draft is None:
                    return _read_response(events, read_tools)
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
                args = _parse_args(call.function.arguments)
                logger.info("tool call %s %s", name, args)

                if name == "ask_user":
                    question = (
                        str((args or {}).get("question") or "").strip() or "Уточните, пожалуйста."
                    )
                    return AgentResponse(
                        reply_text=question,
                        status=AgentStatus.NEED_CLARIFICATION,
                        events=list(events.values()),
                    )

                fingerprint = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
                # Массовое удаление подтверждаем всегда, даже в текстовом режиме. Иначе тул
                # отказывает служебной ошибкой, модель пересказывает её пользователю как есть,
                # а на «да» вызывает тул заново без снимка ID — и удаляет всё, что подходит
                # на тот момент, а не те записи, число которых человек видел в вопросе.
                preview = (confirm_changes or name == "delete_events") and name in MUTATING_TOOLS
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
                            result = await self._toolbox.call(user_id, name, args, dry_run=True)
                        elif name == "create_event" and request_id is not None:
                            result = await self._toolbox.call(
                                user_id,
                                name,
                                args,
                                idempotency_key=str(uuid5(request_id, f"create:{mutation_number}")),
                            )
                        else:
                            result = await self._toolbox.call(user_id, name, args)
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
                    draft = DraftAction(tool=name, arguments=draft_args, preview="")
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

        logger.warning("agent exceeded %d steps", self._max_steps)
        if errors:
            return _incomplete_response(events, confirmed, conflict_warnings, errors)
        if confirmed:
            return _confirmed_response(confirmed, conflict_warnings)
        return AgentResponse(reply_text=TOO_MANY_STEPS, status=AgentStatus.ERROR)


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
        + "Не удалось выполнить действие: "
        + "; ".join(error[2] for error in errors.values())
        + ". Уточните запрос.",
        status=AgentStatus.NEED_CLARIFICATION,
        events=list(events.values()),
    )


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
    if event.start_at is None:
        return "без даты"
    if event.end_at is None:
        return event.start_at.isoformat()
    return f"{event.start_at.isoformat()}–{event.end_at.strftime('%H:%M')}"


def _confirmed_response(
    events: dict[str, Event], conflicts: dict[str, Event] | None = None
) -> AgentResponse:
    lines = []
    for event in events.values():
        action = {"done": "Выполнено", "cancelled": "Отменено"}.get(event.status, "Сохранено")
        when = event.start_at.strftime("%d.%m.%Y в %H:%M") if event.start_at else "без даты"
        lines.append(f"{action}: {event.name}, {when}.")
    for event in (conflicts or {}).values():
        lines.append(f"Учтите: пересечение с «{event.name}», {_when(event)}.")
    return AgentResponse(
        reply_text="\n".join(lines),
        status=AgentStatus.DONE,
        events=list({**events, **(conflicts or {})}.values()),
    )


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
        when = "без даты"
        if event.start_at:
            when = f"{event.start_at:%d.%m.%Y}, {event.weekday}, {event.start_at:%H:%M}"
            if event.end_at:
                end = (
                    event.end_at.strftime("%H:%M")
                    if event.end_at.date() == event.start_at.date()
                    else event.end_at.strftime("%d.%m.%Y %H:%M")
                )
                when += f"–{end}"
        status = " (выполнено)" if event.status == "done" else ""
        lines.append(f"{when} — {event.name}{status}")
    return AgentResponse(reply_text="\n".join(lines), status=AgentStatus.DONE, events=items)
