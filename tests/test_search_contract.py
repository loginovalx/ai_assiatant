import httpx
import pytest
from conftest import FakeEmbedder
from fastmcp import Client
from pydantic import ValidationError

from api.main import app
from common.repository import EventRepository
from common.schemas import EventCreate
from mcp_server.server import mcp


async def test_api_search_filters_type_before_limit(pool):
    async with app.router.lifespan_context(app):
        for index in range(15):
            await app.state.repo.create(
                EventCreate(user_id="u", type="task", name=f"Документы {index}")
            )
        reminder = await app.state.repo.create(
            EventCreate(user_id="u", type="task", name="Документы нотариуса")
        )
        # Use meeting as the minority type; a meeting requires a date.
        from datetime import datetime

        from common.config import get_settings

        meeting = await app.state.repo.create(
            EventCreate(
                user_id="u",
                type="meeting",
                name="Документы нотариуса",
                start_at=datetime.now(get_settings().tz),
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            response = await c.get(
                "/v1/events/search", params={"user_id": "u", "q": "документы", "type": "meeting"}
            )
            assert response.status_code == 200
            assert [event["id"] for event in response.json()] == [str(meeting.id)]
            assert str(reminder.id) not in {event["id"] for event in response.json()}
            assert (
                await c.get("/v1/events/search", params={"user_id": "u", "q": "   "})
            ).status_code == 422


class JudgedEmbedder(FakeEmbedder):
    unavailable = False

    async def relevant_indices(self, query, candidates):
        if self.unavailable:
            return None
        return {i for i, item in enumerate(candidates) if "презентац" in item["name"].lower()}


@pytest.mark.parametrize("unavailable", [False, True])
async def test_search_does_not_return_unverified_unrelated_semantic_hits(pool, tz, unavailable):
    embedder = JudgedEmbedder()
    embedder.unavailable = unavailable
    repo = EventRepository(pool, embedder, tz)
    wanted = await repo.create(
        EventCreate(user_id="u", type="task", name="Подготовить презентацию")
    )
    await repo.create(EventCreate(user_id="u", type="task", name="Купить молоко"))
    hits = await repo.search("u", "презентацию" if unavailable else "слайды", limit=10)
    assert [hit.id for hit in hits] == [wanted.id]


async def test_rename_rejects_blank_name_before_database_write(repo):
    event = await repo.create(EventCreate(user_id="u", type="task", name="Документы"))
    with pytest.raises(ValidationError):
        await repo.update("u", event.id, name="  ")
    assert (await repo.get("u", event.id)).name == "Документы"


async def test_blank_rename_cannot_be_previewed(pool, repo):
    event = await repo.create(EventCreate(user_id="u", type="task", name="Документы"))
    async with Client(mcp) as client:
        response = await client.call_tool(
            "update_event",
            {"user_id": "u", "event_id": str(event.id), "name": "   ", "dry_run": True},
            raise_on_error=False,
        )
    assert response.is_error
