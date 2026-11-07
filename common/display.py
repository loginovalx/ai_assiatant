"""Human-readable calendar date and time formatting."""

from datetime import datetime
from zoneinfo import ZoneInfo

from common.config import get_settings
from common.timeutils import WEEKDAYS_RU

MONTHS_RU_GENITIVE = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def _local_datetime(value: datetime, tz: ZoneInfo | None) -> datetime:
    display_tz = tz or get_settings().tz
    if value.tzinfo is None:
        return value.replace(tzinfo=display_tz)
    return value.astimezone(display_tz)


def _date_with_weekday(value: datetime) -> str:
    return (
        f"{value.day} {MONTHS_RU_GENITIVE[value.month - 1]} {value.year}, "
        f"{WEEKDAYS_RU[value.weekday()]}"
    )


def format_calendar_datetime(value: datetime | None, tz: ZoneInfo | None = None) -> str:
    """Format one calendar moment in the configured timezone, including its weekday."""
    if value is None:
        return "без даты"
    local = _local_datetime(value, tz)
    return f"{_date_with_weekday(local)}, в {local:%H:%M}"


def format_calendar_time(
    start_at: datetime | None,
    end_at: datetime | None = None,
    tz: ZoneInfo | None = None,
) -> str:
    """Format an undated entry, a moment, or a time interval with its local date(s)."""
    if start_at is None:
        return "без даты"

    start = _local_datetime(start_at, tz)
    if end_at is None:
        return format_calendar_datetime(start, tz)

    end = _local_datetime(end_at, tz)
    if start.date() == end.date():
        return f"{_date_with_weekday(start)}, с {start:%H:%M} до {end:%H:%M}"

    return f"с {_date_with_weekday(start)}, {start:%H:%M} до {_date_with_weekday(end)}, {end:%H:%M}"
