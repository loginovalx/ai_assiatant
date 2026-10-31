"""Long-polling транспорт: апдейты из /updates доходят до обработчика, курсор сохраняется."""

import asyncio

import pytest

from bot.models import AgentMessageResponse
from bot.polling import PollingBot

UPDATE = {
    "update_type": "message_created",
    "message": {
        "sender": {"user_id": 555, "is_bot": False},
        "recipient": {"chat_type": "dialog", "chat_id": 555},
        "body": {"mid": "mid-poll-1", "text": "Что у меня завтра?"},
    },
}


class FakeMax:
    def __init__(self, responses):
        self.responses = list(responses)
        self.markers = []

    async def updates(self, marker, timeout=20):  # noqa: ASYNC109
        self.markers.append(marker)
        return self.responses.pop(0)

    async def send(self, chat_id, text, **kwargs):
        pass


class FakeAgent:
    def __init__(self):
        self.asked = []

    async def ask(self, user_id, text, confirm_changes=False, request_id=None):
        self.asked.append((user_id, text, confirm_changes))
        return AgentMessageResponse(reply_text="Отвечаю", status="done")


async def test_poll_dispatches_and_advances_marker(redis):
    max_api = FakeMax(
        [
            {"updates": [UPDATE], "marker": 101},
            {"updates": [], "marker": 102},
        ]
    )
    agent = FakeAgent()

    from bot import polling as polling_mod
    from bot.handler import BotHandler
    from bot.state import StateStore

    handler = BotHandler(max_api, None, agent, StateStore(redis), redis, None)
    bot = PollingBot(max_api, handler, redis, timeout=1)

    marker = await bot.poll_once(None)
    assert marker == 101
    assert await redis.get("bot:updates:marker") == "101"
    assert await bot.poll_once(marker) == 102

    # Апдейт обрабатывается задачей: ждём её, прежде чем проверять вызовы агента.
    while polling_mod._active:
        await asyncio.gather(*list(polling_mod._active), return_exceptions=True)

    assert agent.asked == [("555", "Что у меня завтра?", False)]


async def test_poll_without_marker_keeps_previous():
    max_api = FakeMax([{"updates": []}])
    bot = PollingBot(max_api, None, None, timeout=1)  # маркера нет — Redis не затрагивается
    assert await bot.poll_once(7) == 7


async def test_poll_skips_malformed_update(redis):
    max_api = FakeMax(
        [
            {
                "updates": [
                    {"update_type": 42},
                    "junk",
                    {"update_type": "message_created", "message": []},
                    {"update_type": "message_created", "message": {"body": []}},
                ],
                "marker": 5,
            }
        ]
    )
    bot = PollingBot(max_api, None, redis, timeout=1)
    assert await bot.poll_once(None) == 5


class MemoryRedis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, **kwargs):
        if kwargs.get("nx") and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)


class FailOnceHandler:
    def __init__(self):
        self.calls = 0

    async def handle_update(self, update):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary handler failure")


async def test_failed_handler_does_not_commit_poll_marker_or_drop_update():
    redis = MemoryRedis()
    max_api = FakeMax(
        [
            {"updates": [UPDATE], "marker": 101},
            {"updates": [UPDATE], "marker": 101},
        ]
    )
    handler = FailOnceHandler()
    bot = PollingBot(max_api, handler, redis, timeout=1)

    with pytest.raises(RuntimeError, match="update batch"):
        await bot.poll_once(None)
    assert await redis.get("bot:updates:marker") is None

    assert await bot.poll_once(None) == 101
    assert handler.calls == 2
    assert await redis.get("bot:updates:marker") == 101


async def test_poll_skips_malformed_nested_updates_without_blocking_cursor():
    redis = MemoryRedis()
    max_api = FakeMax(
        [
            {
                "updates": [
                    {"update_type": "message_created", "message": []},
                    {"update_type": "message_created", "message": {"body": []}},
                ],
                "marker": 102,
            }
        ]
    )
    bot = PollingBot(max_api, None, redis, timeout=1)

    assert await bot.poll_once(None) == 102
    assert await redis.get("bot:updates:marker") == 102
