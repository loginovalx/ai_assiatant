"""Fresh, owner-scoped answers about the last unambiguously selected event."""

import re
from datetime import datetime, timedelta
from uuid import UUID

from agent.memory import HistoryMessage
from agent.tools import McpToolbox
from common.display import format_calendar_time
from common.schemas import AgentResponse, AgentStatus, Event
from common.timeutils import calendar_text

_REFERENCE = re.compile(
    r"\b(?:он|она|оно|его|е[её]|него|не[её])\b|"
    r"\b(?:эт(?:от|а|о|у|ой|ого)|тот|та|ту|той|того)\s+"
    r"(?:(?:завтрашн|сегодняшн|предстоящ)\w*\s+)?"
    r"(?:созвон\w*|встреч\w*|звонок|звонка|событи\w*|напоминани\w*|задач\w*|запис\w*)\b",
    re.I,
)
_SINGULAR = re.compile(
    r"\b(?:созвон|встреч[ауе]|звонок|событие|напоминание|задач[ауе]|запись)\b",
    re.I,
)
_MUTATION = re.compile(
    r"\b(?:поставь|создай|добавь|напомни|удали|отмени|перенеси|измени|поменяй|"
    r"переименуй|отметь|заверши|исправь)(?:те)?\b",
    re.I,
)


def _quoted_names(text: str) -> list[str]:
    return [a or b or c for a, b, c in re.findall(r'«([^»]+)»|"([^"\n]+)"|`([^`]+)`', text)]


def is_single_event_time_question(text: str) -> bool:
    """Recognize read intent and a singular subject independently of word order."""
    value = calendar_text(text).casefold()
    noun_refs = [match[0] for match in _REFERENCE.finditer(value) if " " in match[0].strip()]
    if len(noun_refs) > 1 or len(_quoted_names(text)) > 1:
        return False
    if _MUTATION.search(value) or re.search(
        r"\b(?:они|эти|те|все|какие)\b|\bи\s+(?:покажи|найди|перечисли)\b|"
        r"\bпокажи\b.*\b(?:календарь|расписание)\b",
        value,
    ):
        return False
    time = re.search(
        r"\b(?:во\s+сколько|до\s+скольки|когда|длительност\w*)\b|"
        r"\bсколько\b.*\b(?:длит\w*|продл\w*|займ[её]т)\b",
        value,
    )
    return bool(
        time and (_REFERENCE.search(value) or _SINGULAR.search(value) or _quoted_names(text))
    )


def is_event_details_question(text: str) -> bool:
    return bool(
        is_single_event_time_question(text)
        and _REFERENCE.search(calendar_text(text))
        and not _quoted_names(text)
    )


def format_event_details(event: Event) -> AgentResponse:
    when = (
        format_calendar_time(event.start_at, event.end_at if event.duration_minutes else None)
        if event.start_at
        else "дата и время не указаны"
    )
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
    response._focused_event_id = str(event.id)
    return response


def single_event_read_response(events: list[Event], query: str) -> AgentResponse:
    """A broad tool result cannot turn a singular time question into a day list."""
    names = _quoted_names(query)
    if "\n" in query:
        names.append(query.rsplit("\n", 1)[1].strip().strip('«»" .!?'))
    exact = [e for e in events if any(e.name.casefold() == name.casefold() for name in names)]
    candidates = exact or events
    if len(candidates) == 1:
        return format_event_details(candidates[0])
    choices = "; ".join(f"«{e.name}»" for e in candidates[:3])
    return AgentResponse(
        reply_text=(
            f"О какой записи вы спрашиваете? Уточните название: {choices}."
            if choices
            else "Не нашёл такую запись. Уточните её название."
        ),
        status=AgentStatus.NEED_CLARIFICATION,
        events=candidates[:10],
    )


async def read_event_details(
    toolbox: McpToolbox,
    user_id: str,
    history: list[HistoryMessage],
    now: datetime,
    query: str = "",
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
    words = calendar_text(query).casefold()
    days = set(re.findall(r"\b(сегодня|завтра|послезавтра|вчера)\b", words))
    kinds = {
        kind
        for pattern, kind in (
            (r"\b(?:созвон\w*|встреч\w*)\b", "meeting"),
            (r"\bнапоминани\w*\b", "reminder"),
            (r"\bзадач\w*\b", "task"),
        )
        if re.search(pattern, words)
    }
    mismatch = bool(kinds and event.type not in kinds) or len(days) > 1
    if len(days) == 1:
        day = days.pop()
        wanted = (
            now + timedelta(days={"сегодня": 0, "завтра": 1, "послезавтра": 2, "вчера": -1}[day])
        ).date()
        mismatch = (
            mismatch
            or event.start_at is None
            or event.start_at.astimezone(now.tzinfo).date() != wanted
        )
    if mismatch:
        return AgentResponse(
            reply_text=(
                "Выбранная запись не соответствует указанному дню или типу. "
                "Уточните название нужной записи."
            ),
            status=AgentStatus.NEED_CLARIFICATION,
        )
    return format_event_details(event)
