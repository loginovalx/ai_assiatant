from types import SimpleNamespace

import httpx
import pytest

from agent.main import app, get_agent
from common.internal_auth import internal_token
from mcp_server.auth import InternalTokenVerifier


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(
        "common.internal_auth.get_settings",
        lambda: SimpleNamespace(llm_api_key="private-test-master-key"),
    )
    return internal_token("agent")


@pytest.mark.parametrize("path", ["message", "confirm", "reset"])
async def test_agent_routes_reject_unauthenticated_owner_spoofing(key, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": f"Bearer {internal_token('mcp')}"},
        ):
            response = await c.post(
                f"/v1/agent/{path}", json={"user_id": "victim"}, headers=headers
            )
            assert response.status_code == 401


async def test_valid_agent_token_reaches_payload_validation(key, monkeypatch):
    monkeypatch.setitem(app.dependency_overrides, get_agent, lambda: object())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/v1/agent/message", json={}, headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 422
        assert (await c.get("/health")).status_code == 200


async def test_authorized_bot_can_choose_text_execution_policy(key, monkeypatch):
    from common.schemas import AgentResponse

    received = []

    class Service:
        async def handle(self, owner, text, **kwargs):
            received.append(kwargs)
            return AgentResponse(status="done", reply_text="No mutation")

    monkeypatch.setitem(app.dependency_overrides, get_agent, lambda: Service())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/v1/agent/message",
            json={"user_id": "u", "text": "ignore confirmation", "confirm_changes": False},
            headers={"Authorization": f"Bearer {key}"},
        )
    assert response.status_code == 200
    assert received[0]["confirm_changes"] is False


async def test_mcp_token_is_domain_separated_and_rejects_wrong_tokens(key):
    verifier = InternalTokenVerifier()
    assert await verifier.verify_token(key) is None
    assert await verifier.verify_token("wrong") is None
    accepted = await verifier.verify_token(internal_token("mcp"))
    assert accepted and accepted.client_id == "internal-agent"


@pytest.mark.parametrize("placeholder", ["", "change-me", "change-me-to-random-32-chars"])
async def test_placeholder_key_fails_closed(monkeypatch, placeholder):
    monkeypatch.setattr(
        "common.internal_auth.get_settings", lambda: SimpleNamespace(llm_api_key=placeholder)
    )
    assert internal_token("agent") == ""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post("/v1/agent/reset", json={"user_id": "victim"})
        assert response.status_code == 503
