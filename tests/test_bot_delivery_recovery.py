"""Retry MAX message updates after a handler fails or is interrupted."""

import asyncio

import pytest

from bot.handler import BotHandler
from bot.models import AgentMessageResponse
from bot.state import StateStore


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

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def eval(self, script, numkeys, *args):
        key, expected, *rest = args
        if self.values.get(key) not in (expected, *rest):
            return 0
        if "redis.call('SET'" in script:
            self.values[key] = rest[0]
            return 1
        return await self.delete(key)


class FailAfterAgentMax:
    def __init__(self):
        self.sent = []
        self.fail_result = True

    async def send(self, chat_id, text, *, to_user=False, buttons=None):
        if text == "Запрос обработан" and self.fail_result:
            self.fail_result = False
            raise OSError("temporary MAX send failure")
        self.sent.append(text)

    async def answer_callback(self, callback_id, text=None):
        return None


class IdempotentAgent:
    def __init__(self):
        self.calls = []
        self.receipts = {}
        self.commits = 0

    async def ask(self, user_id, text, confirm_changes=False, request_id=None):
        self.calls.append(request_id)
        if request_id not in self.receipts:
            self.commits += 1
            self.receipts[request_id] = AgentMessageResponse(
                reply_text="Запрос обработан", status="done"
            )
        return self.receipts[request_id]


@pytest.mark.asyncio
async def test_failed_message_retries_with_same_idempotency_key_then_dedupes():
    redis = MemoryRedis()
    await redis.set("bot:greeted:777", "1")
    max_api = FailAfterAgentMax()
    agent = IdempotentAgent()
    bot = BotHandler(max_api, None, agent, StateStore(redis), redis)
    update = {
        "update_type": "message_created",
        "message": {
            "sender": {"user_id": 777},
            "recipient": {"chat_type": "dialog", "chat_id": 777},
            "body": {"mid": "delivery-1", "text": "Что у меня завтра?"},
        },
    }

    with pytest.raises(OSError, match="temporary MAX send failure"):
        await bot.handle_update(update)

    await bot.handle_update(update)
    await bot.handle_update(update)

    assert len(agent.calls) == 2  # failed delivery retried; completed duplicate is ignored
    assert agent.calls[0] == agent.calls[1]  # backend idempotency prevents duplicate writes
    assert agent.commits == 1
    assert redis.values["bot:done:delivery-1"] == "1"


@pytest.mark.asyncio
async def test_cancelled_message_releases_processing_claim_for_redelivery():
    redis = MemoryRedis()
    bot = BotHandler(None, None, None, StateStore(redis), redis)
    update = {
        "update_type": "message_created",
        "message": {
            "sender": {"user_id": 777},
            "recipient": {"chat_type": "dialog", "chat_id": 777},
            "body": {"mid": "delivery-cancelled", "text": "Что у меня завтра?"},
        },
    }

    started = asyncio.Event()

    async def blocked_handler(_update):
        started.set()
        await asyncio.Event().wait()

    bot._handle_update = blocked_handler
    task = asyncio.create_task(bot.handle_update(update))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await redis.get("bot:processing:delivery-cancelled") is None

    async def successful_handler(_update):
        return None

    bot._handle_update = successful_handler
    await bot.handle_update(update)
    assert redis.values["bot:done:delivery-cancelled"] == "1"
