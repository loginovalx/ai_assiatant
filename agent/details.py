"""Fresh, owner-scoped answers about the last unambiguously selected event."""

import re
from datetime import datetime, timedelta
from uuid import UUID

from agent.memory import HistoryMessage
from agent.tools import McpToolbox
from common.display import format_calendar_datetime
from common.schemas import AgentResponse, AgentStatus, Event


def is_event_details_question(text: str) -> bool:
    value = text.casefold().strip().rstrip(" .!?")
    subject = r"(?:она|он|эта\s+встреча|этот\s+созвон|эта\s+запись)"
    return bool(
        re.fullmatch(
            rf"(?:во\s+сколько\s+{subject}(?:\s+теперь)?(?:\s+начн[её]тся|\s+начинается)?"
            rf"(?:\s+и\s+сколько\s+продлится)?|сколько\s+{subject}\s+(?:длится|продлится))",
            value,
        )
    )


async def read_event_details(
    toolbox: McpToolbox,
    user_id: str,
    history: list[HistoryMessage],
    now: datetime,
) -> AgentResponse:
    last = history[-1] if history and history[-1]["role"] == "assistant" else {}
    try:
        event_id = str(UUID(last.get("focused_event_id") or ""))
    except (ValueError, TypeError, AttributeError):
        return AgentResponse(
            reply_text="О какой записи вы спрашиваете? Укажите название.",
            status=AgentStatus.NEED_CLARIFICATION,
        )
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    result = await toolbox.call(
        user_id,
        "get_events",
        {
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(days=1)).isoformat(),
            "include_done": True,
        },
        event_ids=[event_id],
    )
    if not result.ok:
        return AgentResponse(
            reply_text="Не удалось проверить запись. Попробуйте ещё раз позже.",
            status=AgentStatus.ERROR,
        )
    events = [Event.model_validate(raw) for raw in result.events]
    if any(e.user_id != user_id for e in events):
        raise ValueError("Foreign event in contextual details")
    if len(events) != 1 or str(events[0].id) != event_id:
        return AgentResponse(
            reply_text="Не удалось найти эту запись. Уточните её название.",
            status=AgentStatus.NEED_CLARIFICATION,
        )
    event = events[0]
    when = format_calendar_datetime(event.start_at) if event.start_at else "дата и время не указаны"
    duration = (
        f"Длительность — {event.duration_minutes} мин."
        if event.duration_minutes is not None
        else "У напоминания нет длительности."
        if event.type == "reminder"
        else "Длительность не указана."
    )
    status = {"cancelled": " Запись отменена.", "done": " Запись выполнена."}.get(event.status, "")
    response = AgentResponse(
        reply_text=f"«{event.name}»: {when}. {duration}{status}",
        status=AgentStatus.DONE,
        events=[event],
    )
    response._focused_event_id = event_id
    return response
