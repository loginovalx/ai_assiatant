"""Файл .ics (RFC 5545): на телефоне открывается как «Добавить событие в календарь»."""

from datetime import UTC, datetime, timedelta

from common.schemas import Event, EventStatus, EventType

# Сколько показывать в календаре, если запись сама время не занимает (end_at пустой):
# у напоминания и дела без длительности есть только момент, а календарю нужен отрезок.
_DISPLAY_DURATION = {
    EventType.TASK: timedelta(minutes=30),
    EventType.REMINDER: timedelta(minutes=15),
}
# За сколько до начала календарь покажет уведомление.
_ALARM_BEFORE = {
    EventType.MEETING: timedelta(minutes=15),
    EventType.TASK: timedelta(0),
    EventType.REMINDER: timedelta(0),
}


def build_ics(event: Event, now: datetime | None = None) -> str:
    if event.start_at is None:
        raise ValueError("У события нет даты — его нельзя добавить в календарь")

    start = event.start_at.astimezone(UTC)
    end = (
        event.end_at.astimezone(UTC)
        if event.end_at is not None
        else start + _DISPLAY_DURATION[event.type]
    )
    alarm_minutes = int(_ALARM_BEFORE[event.type].total_seconds() // 60)
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    title = f"✓ Выполнено: {event.name}" if event.status == EventStatus.DONE else event.name

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//voice-assistant//RU",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{event.id}@voice-assistant",
        f"DTSTAMP:{_fmt(stamp)}",
        f"DTSTART:{_fmt(start)}",
        f"DTEND:{_fmt(end)}",
        f"SUMMARY:{_escape(title)}",
        "STATUS:CANCELLED" if event.status == EventStatus.CANCELLED else "STATUS:CONFIRMED",
    ]
    if event.description and event.description != event.name:
        lines.append(f"DESCRIPTION:{_escape(event.description)}")
    if event.status == EventStatus.ACTIVE:
        lines += [
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_escape(event.name)}",
            f"TRIGGER:-PT{alarm_minutes}M",
            "END:VALARM",
        ]
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "".join(_fold(line) + "\r\n" for line in lines)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _escape(text: str) -> str:
    # RFC 5545 text values encode line breaks as the two literal characters \\n.
    # Normalize all newline spellings first, including bare CR, so user text cannot
    # inject a new iCalendar property or component.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str, limit: int = 75) -> str:
    """Строки длиннее 75 байт переносятся (RFC 5545, 3.1). Кириллица — 2 байта на букву."""
    parts, current = [], ""
    for char in line:
        if len((current + char).encode()) > limit:
            parts.append(current)
            current = " " + char
        else:
            current += char
    parts.append(current)
    return "\r\n".join(parts)
