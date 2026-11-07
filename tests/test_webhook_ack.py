"""Webhook acknowledgement is delayed until message processing completes."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import bot.webhook as webhook_module
from bot.handler import BotHandler, is_valid_update
from bot.state import StateStore
from bot.webhook import webhook


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

    async def eval(self, script, numkeys, *args):
        key, token = args
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


def make_request(handler, update=None):
    app = SimpleNamespace(state=SimpleNamespace(webhook_secret="test-secret", handler=handler))
    body = json.dumps(
        update
        if update is not None
        else {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 777},
                "recipient": {"chat_type": "dialog", "chat_id": 777},
                "body": {"mid": "webhook-1", "text": "hello"},
            },
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"x-max-bot-api-secret", b"test-secret")],
            "app": app,
        },
        receive,
    )


def test_missing_message_body_is_valid_for_message_endpoint_fallback():
    assert is_valid_update(
        {"update_type": "message_created", "message": {"sender": {}, "body": None}}
    )


@pytest.mark.asyncio
async def test_webhook_does_not_ack_before_handler_finishes():
    started = asyncio.Event()
    finish = asyncio.Event()

    class Handler:
        async def handle_update(self, update):
            started.set()
            await finish.wait()

    task = asyncio.create_task(webhook(make_request(Handler())))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not task.done()

    finish.set()
    assert await task == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_does_not_ack_duplicate_while_first_delivery_is_pending():
    started = asyncio.Event()
    finish = asyncio.Event()
    redis = MemoryRedis()
    handler = BotHandler(None, None, None, StateStore(redis), redis)

    async def blocked_handler(update):
        started.set()
        await finish.wait()

    handler._handle_update = blocked_handler
    first = asyncio.create_task(webhook(make_request(handler)))
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(HTTPException) as error:
        await webhook(make_request(handler))

    assert error.value.status_code == 503
    finish.set()
    assert await first == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_timeout_leaves_task_running_until_retry_can_ack(monkeypatch):
    monkeypatch.setattr(webhook_module, "_WEBHOOK_PROCESS_TIMEOUT_SECONDS", 0.01)
    started = asyncio.Event()
    finish = asyncio.Event()
    redis = MemoryRedis()
    handler = BotHandler(None, None, None, StateStore(redis), redis)

    async def blocked_handler(update):
        started.set()
        await finish.wait()

    handler._handle_update = blocked_handler
    with pytest.raises(HTTPException) as error:
        await webhook(make_request(handler))
    assert error.value.status_code == 503
    await asyncio.wait_for(started.wait(), timeout=1)
    assert len(webhook_module._active) == 1
    processing = next(iter(webhook_module._active))
    assert not processing.done()

    finish.set()
    await processing
    await asyncio.sleep(0)  # let the done callback remove the tracked task
    assert redis.values["bot:done:webhook-1"] == "1"
    monkeypatch.setattr(webhook_module, "_WEBHOOK_PROCESS_TIMEOUT_SECONDS", 1)
    assert await webhook(make_request(handler)) == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_skips_malformed_nested_message_without_retrying_poison_event():
    class Handler:
        async def handle_update(self, update):
            raise AssertionError("malformed update must not reach the handler")

    result = await webhook(
        make_request(Handler(), {"update_type": "message_created", "message": []})
    )
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_returns_retryable_status_for_handler_failure():
    class Handler:
        async def handle_update(self, update):
            raise OSError("temporary failure")

    with pytest.raises(HTTPException) as error:
        await webhook(make_request(Handler()))

    assert error.value.status_code == 503
