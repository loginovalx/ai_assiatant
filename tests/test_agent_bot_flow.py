from datetime import datetime, timedelta

import httpx

from agent.main import app, get_agent
from agent.service import AgentService
from bot.agent_client import AgentClient
from bot.handler import BotHandler
from bot.state import StateStore
from tests.scripted_llm import text, tool_call


class LocalMax:
    def __init__(self):
        self.sent = []
        self.callbacks = []

    async def send(self, chat_id, reply, **kwargs):
        self.sent.append({"text": reply, **kwargs})

    async def answer_callback(self, callback_id, reply=None):
        self.callbacks.append(reply)


async def test_bot_voice_to_real_agent_http_confirmation_and_database(
    make_agent, pool, redis, tz, monkeypatch
):
    tomorrow = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "reminder",
                "name": "Позвонить Ивану",
                "description": "Напомни завтра в 10:00 позвонить Ивану",
                "start_at": tomorrow.isoformat(),
            },
        ),
        text("Напомню завтра в 10:00 позвонить Ивану"),
    )
    service = AgentService(agent, redis)
    app.dependency_overrides[get_agent] = lambda: service
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "bot.agent_client.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.ASGITransport(app=app), **kwargs),
    )
    max_api = LocalMax()
    state = StateStore(redis)
    bot = BotHandler(max_api, None, AgentClient("http://agent"), state, redis)
    try:
        await bot.handle_update(
            {
                "update_type": "message_created",
                "message": {
                    "sender": {"user_id": 777},
                    "recipient": {"chat_type": "dialog"},
                    "body": {
                        "mid": "voice-1",
                        "attachments": [
                            {
                                "type": "audio",
                                "payload": {
                                    "transcription": "Напомни завтра в 10:00 позвонить Ивану",
                                },
                            }
                        ],
                    },
                },
            }
        )
        assert await pool.fetchval("SELECT count(*) FROM events") == 0
        assert max_api.sent[-1]["buttons"]
        callback = {
            "update_type": "message_callback",
            "callback": {
                "user": {"user_id": 777},
                "callback_id": "confirm-1",
                "payload": max_api.sent[-1]["buttons"][0][0]["payload"],
            },
        }
        await bot.handle_update(callback)
        assert max_api.sent[-1]["text"].startswith("Готово.")
        assert await state.current("777") == "idle"
        assert await pool.fetchval("SELECT count(*) FROM events") == 1
        row = await pool.fetchrow("SELECT user_id, name, start_at FROM events")
        assert row["user_id"] == "777" and row["name"] == "Позвонить Ивану"
        assert row["start_at"] == tomorrow
        await bot.handle_update(callback)
        assert await pool.fetchval("SELECT count(*) FROM events") == 1
    finally:
        app.dependency_overrides.pop(get_agent, None)
