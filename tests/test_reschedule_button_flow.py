from datetime import datetime, timedelta

import httpx
import pytest

from agent.main import app, get_agent
from agent.service import AgentService
from bot.agent_client import AgentClient
from bot.handler import BotHandler
from bot.state import StateStore
from common.schemas import EventCreate
from tests.scripted_llm import tool_call
from tests.test_agent_bot_flow import LocalMax


@pytest.mark.parametrize("correction", ["text", "preset"])
async def test_another_time_button_edits_owners_conflicting_draft_then_confirms(
    make_agent, pool, repo, redis, tz, monkeypatch, correction
):
    owner, other = "777", "888"
    initial_time = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    changed_time = (initial_time + timedelta(days=correction == "text")).replace(hour=12)
    busy = await repo.create(
        EventCreate(
            user_id=owner,
            type="meeting",
            name="Занято в 10",
            start_at=initial_time,
            duration_minutes=60,
        )
    )
    foreign = await repo.create(
        EventCreate(
            user_id=other,
            type="meeting",
            name="Чужая встреча",
            start_at=changed_time,
            duration_minutes=60,
        )
    )
    common = {"event_type": "meeting", "name": "Новый созвон", "duration_minutes": 30}
    agent, _ = make_agent(
        tool_call("create_event", {**common, "start_at": initial_time.isoformat()}),
        tool_call("create_event", {**common, "start_at": changed_time.isoformat()}),
    )
    service = AgentService(agent, redis)
    monkeypatch.setitem(app.dependency_overrides, get_agent, lambda: service)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "bot.agent_client.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.ASGITransport(app=app), **kwargs),
    )
    max_api = LocalMax()
    client = AgentClient("http://agent")
    edit_sources = []
    original_ask = client.ask

    async def traced_ask(*args, **kwargs):
        edit_sources.append(kwargs.get("edit_draft_id"))
        return await original_ask(*args, **kwargs)

    monkeypatch.setattr(client, "ask", traced_ask)
    bot = BotHandler(max_api, None, client, StateStore(redis), redis)

    async def message(text, mid, *, voice=False):
        await bot.handle_update(
            {
                "update_type": "message_created",
                "message": {
                    "sender": {"user_id": int(owner)},
                    "recipient": {"chat_type": "dialog"},
                    "body": {
                        "mid": mid,
                        **(
                            {"attachments": [{"type": "audio", "payload": {"transcription": text}}]}
                            if voice
                            else {"text": text}
                        ),
                    },
                },
            }
        )

    async def click(payload, callback_id):
        await bot.handle_update(
            {
                "update_type": "message_callback",
                "callback": {
                    "user": {"user_id": int(owner)},
                    "callback_id": callback_id,
                    "payload": payload,
                },
            }
        )

    await message("Встреча новый созвон завтра в 10 на полчаса", "create-meeting", voice=True)
    assert "Пересечение" in max_api.sent[-1]["text"]
    first = await bot.drafts.load(owner)
    button = next(
        b for row in max_api.sent[-1]["buttons"] for b in row if b["text"] == "Другое время"
    )
    await click(button["payload"], "choose-time")
    assert "день" in max_api.sent[-1]["text"] and "время" in max_api.sent[-1]["text"]
    assert await pool.fetchval("SELECT count(*) FROM events") == 2

    if correction == "text":
        await message("Послезавтра в 12:00", "edit-time")
    else:
        preset = next(b for row in max_api.sent[-1]["buttons"] for b in row if b["text"] == "12:00")
        await click(preset["payload"], "choose-noon")
    assert edit_sources == [None, str(first.id)]
    replacement = await bot.drafts.load(owner)
    assert replacement.id != first.id
    assert "12:00" in replacement.preview and "Новый созвон" in replacement.preview
    assert "Пересечение" not in replacement.preview and "Чужая" not in replacement.preview
    assert await pool.fetchval("SELECT count(*) FROM events") == 2

    await click(f"draft:time:{first.id}", "old-time")
    assert "устарел" in max_api.callbacks[-1]
    await click(f"draft:confirm:{replacement.id}", "confirm-new-time")
    await click(f"draft:confirm:{replacement.id}", "repeat-confirm")
    rows = await pool.fetch(
        "SELECT name, start_at, duration_minutes FROM events WHERE user_id=$1 AND id<>$2",
        owner,
        busy.id,
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "Новый созвон"
    assert rows[0]["start_at"] == changed_time and rows[0]["duration_minutes"] == 30
    assert (await repo.get(other, foreign.id)).start_at == changed_time
    assert await pool.fetchval("SELECT count(*) FROM events") == 3
