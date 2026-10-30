"""Длительность записей и предупреждение о пересечениях по времени.

Правило: встреча занимает [start_at, start_at + длительность), по умолчанию час.
Дело занимает время, только если у него названа длительность. Напоминание — момент.
Пересечение не запрещает запись: create/update возвращают её вместе со списком conflicts.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from fastmcp import Client
from pydantic import ValidationError

from api.ics import build_ics
from common.schemas import Event, EventCreate, EventStatus, EventType
from mcp_server.server import mcp

USER = "user-1"


def tomorrow(tz, hour: int, minute: int = 0) -> datetime:
    base = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(days=1, hours=hour, minutes=minute)


async def add(repo, type_, name, start=None, duration=None, user=USER):
    return await repo.create(
        EventCreate(user_id=user, type=type_, name=name, start_at=start, duration_minutes=duration)
    )


# --- схема ---------------------------------------------------------------------


def test_meeting_end_defaults_to_one_hour(tz):
    start = tomorrow(tz, 15)
    event = Event(
        id=uuid4(),
        user_id=USER,
        type=EventType.MEETING,
        name="Встреча",
        description="",
        start_at=start,
        status=EventStatus.ACTIVE,
        created_at=start,
        updated_at=start,
    )
    assert event.duration_minutes is None
    assert event.end_at == start + timedelta(hours=1)


def test_task_without_duration_has_no_end(tz):
    data = EventCreate(
        user_id=USER, type=EventType.TASK, name="Сдать отчёт", start_at=tomorrow(tz, 18)
    )
    assert data.duration_minutes is None


def test_reminder_cannot_have_duration(tz):
    with pytest.raises(ValidationError, match="У напоминания нет длительности"):
        EventCreate(
            user_id=USER,
            type=EventType.REMINDER,
            name="Позвонить",
            start_at=tomorrow(tz, 10),
            duration_minutes=30,
        )


def test_undated_task_cannot_have_duration():
    with pytest.raises(ValidationError, match="только у записи с датой"):
        EventCreate(user_id=USER, type=EventType.TASK, name="Порисовать", duration_minutes=60)


async def test_db_rejects_duration_on_reminder(pool, tz):
    """Вторая линия защиты — CHECK в базе, на случай кода в обход Pydantic."""
    import asyncpg

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO events (user_id, type, name, start_at, duration_minutes) "
            "VALUES ($1, 'reminder', 'x', $2, 30)",
            USER,
            tomorrow(tz, 10),
        )


# --- репозиторий -----------------------------------------------------------------


async def test_duration_is_stored(repo, tz):
    event = await add(repo, EventType.MEETING, "Планёрка", tomorrow(tz, 10), duration=30)
    assert event.duration_minutes == 30
    assert event.end_at == tomorrow(tz, 10, 30)
    assert (await repo.get(USER, event.id)).duration_minutes == 30


@pytest.mark.parametrize(
    ("start", "duration", "conflicts"),
    [
        ((15, 30), None, True),  # 15:30–16:30 внутри 15:00–16:00
        ((16, 0), None, False),  # встык после: 16:00 — уже свободно
        ((14, 30), 30, False),  # встык до: 14:30–15:00
        ((14, 30), 60, True),  # 14:30–15:30 задевает начало
        ((14, 0), 180, True),  # 14:00–17:00 накрывает целиком
    ],
)
async def test_meeting_overlap(repo, tz, start, duration, conflicts):
    existing = await add(repo, EventType.MEETING, "Созвон с Иваном", tomorrow(tz, 15))
    new = await add(repo, EventType.MEETING, "Новая встреча", tomorrow(tz, *start), duration)

    found = await repo.find_conflicts(new)
    assert [e.id for e in found] == ([existing.id] if conflicts else [])


async def test_task_with_duration_blocks_time(repo, tz):
    focus = await add(
        repo, EventType.TASK, "Поработать над отчётом", tomorrow(tz, 10), duration=120
    )
    meeting = await add(repo, EventType.MEETING, "Созвон", tomorrow(tz, 11))
    assert [e.id for e in await repo.find_conflicts(meeting)] == [focus.id]
    assert [e.id for e in await repo.find_conflicts(focus)] == [meeting.id]


async def test_deadline_task_and_reminder_do_not_block_time(repo, tz):
    await add(repo, EventType.TASK, "Сдать отчёт", tomorrow(tz, 11))  # срок, не занятость
    await add(repo, EventType.REMINDER, "Выпить таблетку", tomorrow(tz, 11))
    meeting = await add(repo, EventType.MEETING, "Созвон", tomorrow(tz, 11))
    assert await repo.find_conflicts(meeting) == []

    reminder = await add(repo, EventType.REMINDER, "Позвонить", tomorrow(tz, 11, 15))
    assert await repo.find_conflicts(reminder) == []  # напоминание само время не занимает


async def test_done_cancelled_and_other_users_are_ignored(repo, tz):
    done = await add(repo, EventType.MEETING, "Прошедшая", tomorrow(tz, 15))
    await repo.set_status(USER, done.id, EventStatus.DONE)
    cancelled = await add(repo, EventType.MEETING, "Отменённая", tomorrow(tz, 15))
    await repo.set_status(USER, cancelled.id, EventStatus.CANCELLED)
    await add(repo, EventType.MEETING, "Чужая", tomorrow(tz, 15), user="user-2")

    meeting = await add(repo, EventType.MEETING, "Моя", tomorrow(tz, 15))
    assert await repo.find_conflicts(meeting) == []


async def test_long_meeting_started_earlier_is_found(repo, tz):
    """Встреча 09:00–14:00 задевает 13:00, хотя началась задолго до неё."""
    long = await add(repo, EventType.MEETING, "Стратегическая сессия", tomorrow(tz, 9), 300)
    short = await add(repo, EventType.MEETING, "Обед с клиентом", tomorrow(tz, 13))
    assert [e.id for e in await repo.find_conflicts(short)] == [long.id]


# --- MCP-тулы ----------------------------------------------------------------------


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


async def test_create_warns_but_still_saves(client, tz):
    first = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Созвон с Иваном",
            "start_at": tomorrow(tz, 15).isoformat(),
            "duration_minutes": 60,
        },
    )
    assert first.structured_content["conflicts"] == []
    assert first.structured_content["end_at"] == tomorrow(tz, 16).isoformat()

    second = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Встреча с дизайнером",
            "start_at": tomorrow(tz, 15, 30).isoformat(),
            "duration_minutes": 30,
        },
    )
    data = second.structured_content
    assert data["id"] and data["status"] == "active"  # сохранено, а не отклонено
    assert data["end_at"] == tomorrow(tz, 16).isoformat()
    assert [c["name"] for c in data["conflicts"]] == ["Созвон с Иваном"]


async def test_update_into_busy_slot_warns_and_skips_itself(client, tz):
    await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Планёрка",
            "start_at": tomorrow(tz, 10).isoformat(),
            "duration_minutes": 60,
        },
    )
    other = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Созвон",
            "start_at": tomorrow(tz, 14).isoformat(),
            "duration_minutes": 60,
        },
    )
    event_id = other.structured_content["id"]

    moved = await client.call_tool(
        "update_event",
        {"user_id": USER, "event_id": event_id, "start_at": tomorrow(tz, 10, 30).isoformat()},
    )
    assert [c["name"] for c in moved.structured_content["conflicts"]] == ["Планёрка"]

    shortened = await client.call_tool(
        "update_event",
        {
            "user_id": USER,
            "event_id": event_id,
            "start_at": tomorrow(tz, 11).isoformat(),
            "duration_minutes": 45,
        },
    )
    data = shortened.structured_content
    assert data["conflicts"] == []  # 11:00 — Планёрка уже закончилась, себя не считаем
    assert data["duration_minutes"] == 45
    assert data["end_at"] == tomorrow(tz, 11, 45).isoformat()


async def test_update_duration_on_reminder_is_rejected(client, tz):
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "reminder",
            "name": "Позвонить",
            "start_at": tomorrow(tz, 10).isoformat(),
        },
    )
    result = await client.call_tool(
        "update_event",
        {"user_id": USER, "event_id": created.structured_content["id"], "duration_minutes": 30},
        raise_on_error=False,
    )
    assert result.is_error and "нет длительности" in result.content[0].text


async def test_duration_is_optional_in_tool_schema():
    async with Client(mcp) as c:
        tools = {t.name: t.input_schema for t in await c.list_tools()}
    for name in ("create_event", "update_event"):
        assert "duration_minutes" in tools[name]["properties"]
        assert "duration_minutes" not in tools[name].get("required", [])


# --- календарь -----------------------------------------------------------------------


def test_ics_uses_real_duration():
    start = datetime.fromisoformat("2026-10-30T10:00:00+03:00")
    event = Event(
        id=uuid4(),
        user_id=USER,
        type=EventType.MEETING,
        name="Длинная встреча",
        description="",
        start_at=start,
        duration_minutes=150,
        status=EventStatus.ACTIVE,
        created_at=start,
        updated_at=start,
    )
    text = build_ics(event, now=start)
    assert "DTSTART:20261030T070000Z\r\n" in text
    assert "DTEND:20261030T093000Z\r\n" in text
