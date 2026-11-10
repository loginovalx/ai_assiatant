from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from agent.core import _confirmed_response, _draft_preview, _read_response, _when
from agent.tools import ToolResult
from common.display import format_calendar_datetime, format_calendar_time
from common.schemas import Event

MOSCOW = ZoneInfo("Europe/Moscow")


def test_calendar_datetime_uses_natural_russian_date_and_weekday():
    value = datetime(2026, 11, 8, 12, 0, tzinfo=MOSCOW)

    assert format_calendar_datetime(value, MOSCOW) == "8 ноября 2026, воскресенье, в 12:00"


def test_calendar_interval_shows_one_date_and_readable_time_range():
    start = datetime(2026, 11, 8, 12, 0, tzinfo=MOSCOW)
    end = datetime(2026, 11, 8, 13, 0, tzinfo=MOSCOW)

    assert format_calendar_time(start, end, MOSCOW) == (
        "8 ноября 2026, воскресенье, с 12:00 до 13:00"
    )


def test_calendar_interval_crossing_midnight_shows_both_dates():
    start = datetime(2026, 11, 8, 23, 30, tzinfo=MOSCOW)
    end = start + timedelta(hours=1)

    assert format_calendar_time(start, end, MOSCOW) == (
        "с 8 ноября 2026, воскресенье, 23:30 до 9 ноября 2026, понедельник, 00:30"
    )


def test_calendar_datetime_converts_to_configured_default_timezone():
    utc_value = datetime(2026, 11, 8, 21, 0, tzinfo=ZoneInfo("UTC"))

    assert format_calendar_datetime(utc_value) == "9 ноября 2026, понедельник, в 00:00"


def test_undated_calendar_entry_is_labeled_clearly():
    assert format_calendar_time(None, None, MOSCOW) == "без даты"


def test_agent_calendar_renderers_use_the_same_readable_display():
    raw = {
        "id": uuid4(),
        "user_id": "u",
        "type": "meeting",
        "name": "Встреча",
        "description": "",
        "start_at": datetime(2026, 11, 8, 9, 0, tzinfo=ZoneInfo("UTC")),
        "duration_minutes": 60,
        "status": "active",
        "created_at": datetime(2026, 11, 1, tzinfo=MOSCOW),
        "updated_at": datetime(2026, 11, 1, tzinfo=MOSCOW),
    }
    event = Event.model_validate(raw)
    expected = "8 ноября 2026, воскресенье, с 12:00 до 13:00"

    assert _when(event) == expected
    preview = _draft_preview("u", "create_event", ToolResult(ok=True, payload={}, events=[raw]), {})
    assert expected in preview
    assert expected in _confirmed_response({str(event.id): event}).reply_text
    reminder = event.model_copy(update={"id": uuid4(), "type": "reminder", "name": "Позвонить"})
    schedule = _read_response({str(e.id): e for e in [event, reminder]}, {"get_events"})
    assert schedule.reply_text == (
        "Расписание\n\n8 ноября 2026, воскресенье\n"
        "• 12:00–13:00 — Встреча · Встреча\n"
        "• 12:00 — Позвонить · Напоминание"
    )
