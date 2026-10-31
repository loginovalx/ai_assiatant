"""Bulk-delete confirmations are bound to the exact previewed event snapshot."""

import pytest
from fastmcp import Client

from common.schemas import EventCreate, EventType
from mcp_server.server import mcp

USER = "bulk-delete-user"


@pytest.fixture
async def client():
    async with Client(mcp) as connection:
        yield connection


async def _create_task(repo, user_id: str, name: str):
    return await repo.create(EventCreate(user_id=user_id, type=EventType.TASK, name=name))


async def test_confirm_deletes_only_previewed_ids_and_preserves_new_and_foreign_events(pool, repo):
    first = await _create_task(repo, USER, "Первое дело")
    second = await _create_task(repo, USER, "Второе дело")
    foreign = await _create_task(repo, "another-user", "Чужое дело")

    async with Client(mcp) as client:
        preview_result = await client.call_tool(
            "delete_events",
            {"user_id": USER, "event_types": ["task"], "dry_run": True},
        )
        preview = preview_result.structured_content["result"]
        assert preview["count"] == 2
        assert set(preview["event_ids"]) == {str(first.id), str(second.id)}

        added_after_preview = await _create_task(repo, USER, "Новое дело")
        confirmed = await client.call_tool(
            "delete_events",
            {
                "user_id": USER,
                "event_types": ["task"],
                "event_ids": preview["event_ids"],
                "confirmed": True,
            },
        )

    deleted = confirmed.structured_content["result"]
    assert {event["id"] for event in deleted} == {str(first.id), str(second.id)}
    rows = await pool.fetch("SELECT id, status FROM events")
    statuses = {str(row["id"]): row["status"] for row in rows}
    assert statuses[str(first.id)] == statuses[str(second.id)] == "cancelled"
    assert statuses[str(added_after_preview.id)] == statuses[str(foreign.id)] == "active"


async def test_empty_preview_snapshot_deletes_nothing_even_when_matching_rows_exist(
    client, repo, pool
):
    event = await _create_task(repo, USER, "Добавлено после пустого снимка")

    result = await client.call_tool(
        "delete_events",
        {
            "user_id": USER,
            "event_types": ["task"],
            "event_ids": [],
            "confirmed": True,
        },
    )

    assert result.structured_content["result"] == []
    assert await pool.fetchval("SELECT status FROM events WHERE id = $1", event.id) == "active"


async def test_legacy_confirmed_bulk_delete_without_ids_still_deletes_matching_rows(client, repo):
    first = await _create_task(repo, USER, "Первое дело")
    second = await _create_task(repo, USER, "Второе дело")

    result = await client.call_tool(
        "delete_events",
        {"user_id": USER, "event_types": ["task"], "confirmed": True},
    )

    assert {event["id"] for event in result.structured_content["result"]} == {
        str(first.id),
        str(second.id),
    }
