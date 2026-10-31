from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from mcp_server.server import mcp

USER = "ambiguous-deadline-user"
RUSSIAN_MONTHS = (
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
)


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


def upcoming_weekday(now: datetime, weekday: int, hour: int = 18) -> datetime:
    days = (weekday - now.weekday()) % 7
    if days == 0:
        days = 7
    return (now + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)


async def test_task_deadline_at_end_of_week_requires_a_day(client, pool, tz):
    deadline = datetime.now(tz) + timedelta(days=4)
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "task",
            "name": "Купить билеты",
            "user_text": "Поставь задачу купить билеты до конца недели",
            "start_at": deadline.isoformat(),
        },
        raise_on_error=False,
    )

    assert result.is_error
    assert "какого дня" in result.content[0].text.casefold()
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


@pytest.mark.parametrize(
    ("event_type", "phrase"),
    [
        ("meeting", "Поставь встречу на этой неделе в 10 утра"),
        ("reminder", "Напомни на следующей неделе в 10 утра купить билеты"),
        ("reminder", "Напомни на следующей неделе в 10.00 купить билеты"),
    ],
)
async def test_week_range_with_hour_requires_an_exact_day(client, pool, tz, event_type, phrase):
    start = datetime.now(tz) + timedelta(days=2)
    args = {
        "user_id": USER,
        "event_type": event_type,
        "name": "Купить билеты",
        "user_text": phrase,
        "start_at": start.replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
    }
    if event_type == "meeting":
        args["duration_minutes"] = 60

    result = await client.call_tool("create_event", args, raise_on_error=False)

    assert result.is_error
    assert "какой день" in result.content[0].text.casefold()
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_task_week_deadline_accepts_clarified_weekday_dry_run(client, pool, tz):
    now = datetime.now(tz)
    deadline = upcoming_weekday(now, 5)
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "task",
            "name": "Купить билеты",
            "description": "Поставь задачу купить билеты до конца недели",
            "user_text": "Поставь задачу купить билеты до конца недели\nДо субботы",
            "start_at": deadline.isoformat(),
            "dry_run": True,
        },
        raise_on_error=False,
    )

    assert not result.is_error, result.content
    assert result.structured_content["start_at"] == deadline.isoformat()
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_specific_user_text_overrides_ambiguous_description(client, pool, tz):
    deadline = upcoming_weekday(datetime.now(tz), 5)
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "task",
            "name": "Купить билеты",
            "description": "Поставь задачу купить билеты до конца недели",
            "user_text": "До субботы",
            "start_at": deadline.isoformat(),
            "dry_run": True,
        },
        raise_on_error=False,
    )

    assert not result.is_error, result.content
    assert result.structured_content["start_at"] == deadline.isoformat()
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


@pytest.mark.parametrize("date_style", ["iso", "russian", "numeric"])
async def test_task_week_deadline_accepts_explicit_calendar_date_dry_run(
    client, pool, tz, date_style
):
    now = datetime.now(tz)
    deadline = upcoming_weekday(now, 5)
    if date_style == "iso":
        date_text = deadline.date().isoformat()
    elif date_style == "russian":
        date_text = f"{deadline.day} {RUSSIAN_MONTHS[deadline.month - 1]}"
    else:
        date_text = deadline.strftime("%d.%m")
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "task",
            "name": "Купить билеты",
            "user_text": f"Поставь задачу купить билеты до конца недели, срок {date_text}",
            "start_at": deadline.isoformat(),
            "dry_run": True,
        },
        raise_on_error=False,
    )

    assert not result.is_error, result.content
    assert result.structured_content["start_at"] == deadline.isoformat()
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_weekly_list_query_is_not_gated(client, tz):
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await client.call_tool(
        "get_events",
        {
            "user_id": USER,
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(days=7)).isoformat(),
        },
    )

    assert not result.is_error
    assert result.structured_content["result"] == []
