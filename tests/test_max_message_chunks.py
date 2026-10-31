import asyncio
import json

import httpx
import pytest

from bot.max_api import MaxAPI


def make_api(handler) -> MaxAPI:
    api = MaxAPI.__new__(MaxAPI)
    api.max_voice_bytes = 15_000_000
    api._send_locks = {}
    api._last_sent = {}
    api.client = httpx.AsyncClient(
        base_url="https://max.test", transport=httpx.MockTransport(handler)
    )
    api.media_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return api


@pytest.mark.parametrize(
    "text",
    ["Запись в календаре\n" * 400, "x" * 8_500],
    ids=["newline-separated", "one-unbroken-string"],
)
async def test_long_text_is_split_without_losing_any_characters(text, monkeypatch):
    requests = []

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    api = make_api(handler)
    try:
        await api.send(42, text)
    finally:
        await api.close()

    pieces = [json.loads(request.content)["text"] for request in requests]
    assert len(pieces) > 1
    assert all(len(piece) <= 4000 for piece in pieces)
    assert "".join(pieces) == text


async def test_buttons_are_attached_only_to_final_chunk(monkeypatch):
    requests = []
    buttons = [[{"text": "Open", "callback_data": "open"}]]

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    api = make_api(handler)
    try:
        await api.send(42, "row\n" * 1200, buttons=buttons)
    finally:
        await api.close()

    payloads = [json.loads(request.content) for request in requests]
    assert len(payloads) > 1
    assert all("attachments" not in payload for payload in payloads[:-1])
    assert payloads[-1]["attachments"] == [
        {"type": "inline_keyboard", "payload": {"buttons": buttons}}
    ]


async def test_recipient_target_and_429_retry_are_preserved(monkeypatch):
    requests = []
    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})

    api = make_api(handler)
    try:
        await api.send(42, "private", to_user=True)
        await api.send(42, "chat")
    finally:
        await api.close()

    assert [request.url.params.get("user_id") for request in requests[:2]] == ["42", "42"]
    assert [request.url.params.get("chat_id") for request in requests[:2]] == [None, None]
    assert requests[2].url.params.get("chat_id") == "42"
    assert requests[2].url.params.get("user_id") is None
    assert sleeps.count(1.0) == 1
