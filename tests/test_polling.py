"""Long-polling транспорт: апдейты из /updates доходят до обработчика, курсор сохраняется."""

import asyncio

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
    max_api = FakeMax([{"updates": [{"update_type": 42}, "junk"], "marker": 5}])
    bot = PollingBot(max_api, None, redis, timeout=1)
    assert await bot.poll_once(None) == 5
