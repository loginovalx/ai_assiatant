"""Цикл агента: LLM предлагает вызвать тул → выполняем → отдаём результат → … → ответ.

Один запрос пользователя = до agent_max_steps раундов. Обычно хватает двух:
вызвать тул и сформулировать ответ.
"""

import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from agent.memory import ConversationMemory, HistoryMessage
from agent.prompt import build_system_prompt
from agent.tools import McpToolbox, ToolResult
from common.schemas import AgentResponse, AgentStatus, Event

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

    async def handle(self, user_id: str, text: str, now: datetime | None = None) -> AgentResponse:
        now = now or datetime.now(self._tz)
        try:
            history = await self._memory.load(user_id)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_system_prompt(now)},
                *history,
                {"role": "user", "content": text},
            ]
            response = await self._run(user_id, messages)
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

    async def _run(self, user_id: str, messages: list[dict[str, Any]]) -> AgentResponse:
        tools = await self._toolbox.openai_tools()
        events: dict[str, Event] = {}
        mutations: dict[str, ToolResult] = {}
        confirmed: dict[str, Event] = {}
        conflict_warnings: dict[str, Event] = {}
        last_error: str | None = None

        for _ in range(self._max_steps):
            try:
                completion = await self._llm.chat.completions.create(
                    model=self._model, messages=messages, tools=tools, temperature=0.1
                )
            except Exception:
                if confirmed:
                    logger.exception("LLM недоступна после подтверждённой записи")
                    return _confirmed_response(confirmed, conflict_warnings)
                raise
            message = completion.choices[0].message

            if not message.tool_calls:
                if last_error:
                    return AgentResponse(
                        reply_text=f"Не удалось выполнить действие: {last_error}. Уточните запрос.",
                        status=AgentStatus.NEED_CLARIFICATION,
                        events=list(events.values()),
                    )
                reply = (message.content or "").strip()
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
                result = mutations.get(fingerprint) or (
                    await self._toolbox.call(user_id, name, args)
                    if args is not None
                    else ToolResult(ok=False, payload="Аргументы не являются корректным JSON")
                )
                last_error = None if result.ok else str(result.payload)
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
                    mutations[fingerprint] = result
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.for_llm()}
                )

        logger.warning("agent exceeded %d steps", self._max_steps)
        if confirmed:
            return _confirmed_response(confirmed, conflict_warnings)
        return AgentResponse(reply_text=TOO_MANY_STEPS, status=AgentStatus.ERROR)


def _parse_args(raw: str | None) -> dict[str, Any] | None:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _with_event_refs(response: AgentResponse) -> str:
    """Ответ + скрытая сводка событий: чтобы следующий запрос («перенеси её на завтра»)
    знал, о какой записи речь. Пользователь эту сводку не видит — только модель в истории.
    """
    if not response.events:
        return response.reply_text
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
