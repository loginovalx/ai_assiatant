"""Date-scoped bulk cancellation targets only the previewed events in a period."""

from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from common.schemas import EventCreate, EventType
from mcp_server.server import mcp

USER = "dated-bulk-delete-user"


@pytest.fixture
async def client():
    async with Client(mcp) as connection:
        yield connection


async def _create(repo, user_id: str, name: str, event_type: EventType, start_at=None):
    return await repo.create(
        EventCreate(user_id=user_id, type=event_type, name=name, start_at=start_at)
    )


async def test_dated_reminder_delete_uses_half_open_period_and_exact_snapshot(
    client, repo, pool, tz
):
    start = datetime(2026, 11, 9, tzinfo=tz)
    end = start + timedelta(days=1)
    at_start = await _create(repo, USER, "Напоминание в начале", EventType.REMINDER, start)
    inside = await _create(
        repo, USER, "Напоминание сегодня", EventType.REMINDER, start + timedelta(hours=12)
    )
    before = await _create(
        repo, USER, "Напоминание вчера", EventType.REMINDER, start - timedelta(microseconds=1)
    )
    at_end = await _create(repo, USER, "Напоминание завтра", EventType.REMINDER, end)
    foreign = await _create(
        repo, "another-user", "Чужое напоминание", EventType.REMINDER, start + timedelta(hours=1)
    )

    period = {"start_time": start.isoformat(), "end_time": end.isoformat()}
    preview_result = await client.call_tool(
        "delete_events",
        {"user_id": USER, "event_types": ["reminder"], **period},
        raise_on_error=False,
    )
    assert preview_result.is_error
    assert "будет удалено записей — 2" in preview_result.content[0].text

    preview = await client.call_tool(
        "delete_events",
        {"user_id": USER, "event_types": ["reminder"], "dry_run": True, **period},
    )
    snapshot = preview.structured_content["result"]["event_ids"]
    assert set(snapshot) == {str(at_start.id), str(inside.id)}

    added_after_preview = await _create(
        repo, USER, "Новое напоминание сегодня", EventType.REMINDER, start + timedelta(hours=18)
    )
    confirmed = await client.call_tool(
        "delete_events",
        {
            "user_id": USER,
            "event_types": ["reminder"],
            "event_ids": snapshot,
            "confirmed": True,
            **period,
        },
    )

    assert {event["id"] for event in confirmed.structured_content["result"]} == set(snapshot)
    rows = await pool.fetch("SELECT id, status FROM events")
    statuses = {str(row["id"]): row["status"] for row in rows}
    assert statuses[str(at_start.id)] == statuses[str(inside.id)] == "cancelled"
    assert statuses[str(before.id)] == statuses[str(at_end.id)] == "active"
    assert statuses[str(foreign.id)] == statuses[str(added_after_preview.id)] == "active"


async def test_dated_all_types_excludes_undated_rows_from_preview_and_confirmation(
    client, repo, pool, tz
):
    start = datetime(2026, 11, 9, tzinfo=tz)
    end = start + timedelta(days=1)
    dated_task = await _create(
        repo, USER, "Задача сегодня", EventType.TASK, start + timedelta(hours=3)
    )
    dated_reminder = await _create(
        repo, USER, "Напоминание сегодня", EventType.REMINDER, start + timedelta(hours=4)
    )
    undated_task = await _create(repo, USER, "Задача без срока", EventType.TASK)
    tomorrow = await _create(repo, USER, "Напоминание завтра", EventType.REMINDER, end)
    period = {"start_time": start.isoformat(), "end_time": end.isoformat()}

    preview = await client.call_tool("delete_events", {"user_id": USER, "dry_run": True, **period})
    snapshot = preview.structured_content["result"]["event_ids"]
    assert set(snapshot) == {str(dated_task.id), str(dated_reminder.id)}

    confirmed = await client.call_tool(
        "delete_events",
        {"user_id": USER, "event_ids": snapshot, "confirmed": True, **period},
    )
    assert {event["id"] for event in confirmed.structured_content["result"]} == set(snapshot)
    assert (
        await pool.fetchval("SELECT status FROM events WHERE id = $1", undated_task.id) == "active"
    )
    assert await pool.fetchval("SELECT status FROM events WHERE id = $1", tomorrow.id) == "active"


async def test_dated_confirmation_requires_complete_bounds_and_preview_snapshot(client, repo, tz):
    start = datetime(2026, 11, 9, tzinfo=tz)
    await _create(repo, USER, "Напоминание сегодня", EventType.REMINDER, start + timedelta(hours=1))

    partial_period = await client.call_tool(
        "delete_events",
        {"user_id": USER, "start_time": start.isoformat()},
        raise_on_error=False,
    )
    assert partial_period.is_error
    assert "start_time и end_time" in partial_period.content[0].text

    without_snapshot = await client.call_tool(
        "delete_events",
        {
            "user_id": USER,
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(days=1)).isoformat(),
            "confirmed": True,
        },
        raise_on_error=False,
    )
    assert without_snapshot.is_error
    assert "event_ids" in without_snapshot.content[0].text
