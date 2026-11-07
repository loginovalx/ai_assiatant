from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from common.schemas import EventCreate, EventStatus, EventType
from mcp_server.server import mcp

USER = "search-bounds-user"


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


async def test_search_returns_requested_50_identical_matches(repo, tz):
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    for index in range(60):
        await repo.create(
            EventCreate(
                user_id=USER,
                type=EventType.TASK,
                name="Календарная контрольная группа",
                start_at=start + timedelta(minutes=index),
            )
        )

    hits = await repo.search(USER, "контрольная", start, start + timedelta(days=1), limit=50)
    assert len(hits) == 50
    assert all(hit.name == "Календарная контрольная группа" for hit in hits)


async def test_search_period_filters_owner_status_and_half_open_bounds(repo, tz):
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    inside = await repo.create(
        EventCreate(
            user_id=USER,
            type=EventType.TASK,
            name="Периодная контрольная группа",
            start_at=start + timedelta(hours=1),
        )
    )
    foreign = await repo.create(
        EventCreate(
            user_id="another-user",
            type=EventType.TASK,
            name="Периодная контрольная группа",
            start_at=start + timedelta(hours=2),
        )
    )
    cancelled = await repo.create(
        EventCreate(
            user_id=USER,
            type=EventType.TASK,
            name="Периодная контрольная группа",
            start_at=start + timedelta(hours=3),
        )
    )
    await repo.set_status(USER, cancelled.id, EventStatus.CANCELLED)
    at_end = await repo.create(
        EventCreate(
            user_id=USER,
            type=EventType.TASK,
            name="Периодная контрольная группа",
            start_at=start + timedelta(days=1),
        )
    )

    hits = await repo.search(USER, "контрольная", start, start + timedelta(days=1), limit=50)
    assert [hit.id for hit in hits] == [inside.id]
    assert foreign.id not in {hit.id for hit in hits}
    assert cancelled.id not in {hit.id for hit in hits}
    assert at_end.id not in {hit.id for hit in hits}


async def test_search_rejects_reversed_period(client, tz):
    now = datetime.now(tz)
    result = await client.call_tool(
        "search_events",
        {
            "user_id": USER,
            "query": "контрольная",
            "start_time": now.isoformat(),
            "end_time": (now - timedelta(days=1)).isoformat(),
        },
        raise_on_error=False,
    )
    assert result.is_error
    assert "раньше" in result.content[0].text


async def test_search_limit_is_capped_at_50(client, tz):
    result = await client.call_tool(
        "search_events",
        {"user_id": USER, "query": "контрольная", "limit": 51},
        raise_on_error=False,
    )
    assert result.is_error
