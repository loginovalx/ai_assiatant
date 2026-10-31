import json
from uuid import uuid4

import httpx
import pytest

from bot.agent_client import AgentClient, AgentUnavailable
from common.schemas import DraftAction


def transport_client(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "bot.agent_client.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


async def test_confirm_sends_draft_uuid_as_json_string(monkeypatch):
    draft = DraftAction(id=uuid4(), tool="create_event", arguments={}, preview="Создам задачу")
    received = []

    def handle(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "done", "reply_text": "Готово", "events": []})

    transport_client(monkeypatch, handle)
    result = await AgentClient("http://agent:8002").confirm("777", draft)
    assert result.status == "done"
    assert received[0]["draft"]["id"] == str(draft.id)


async def test_confirm_wraps_invalid_agent_response(monkeypatch):
    transport_client(monkeypatch, lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(AgentUnavailable):
        await AgentClient("http://agent:8002").confirm(
            "777", DraftAction(tool="create_event", arguments={}, preview="Test")
        )


async def test_transport_retry_preserves_request_id(monkeypatch):
    request_id = str(uuid4())
    received = []

    def handle(request):
        received.append(json.loads(request.content))
        if len(received) == 1:
            raise httpx.ReadTimeout("Receipt lost", request=request)
        return httpx.Response(200, json={"status": "done", "reply_text": "Готово", "events": []})

    transport_client(monkeypatch, handle)
    result = await AgentClient("http://agent:8002").ask(
        "777", "Создай задачу", request_id=request_id
    )
    assert result.status == "done"
    assert len(received) == 2 and received[0] == received[1]
    assert received[0]["request_id"] == request_id
