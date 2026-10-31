import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import httpx
import pytest

from api.main import app, get_repo
from common.config import get_settings

TOKEN = "test-bot-token"


def signed_data(user_id=777, auth_date=None, **extra):
    fields = {
        "auth_date": str(int(time.time()) if auth_date is None else auth_date),
        "user": json.dumps({"id": user_id, "first_name": "Тест"}, ensure_ascii=False),
        **extra,
    }
    message = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    key = hmac.digest(b"WebAppData", TOKEN.encode(), "sha256")
    fields["hash"] = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


@pytest.fixture
async def auth_client(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_token", TOKEN)
    monkeypatch.setattr(settings, "api_allow_dev_user_id", False)
    app.dependency_overrides[get_repo] = lambda: None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_repo, None)


async def test_signed_identity_is_derived_on_server(auth_client):
    response = await auth_client.get("/v1/me", headers={"X-Max-Init-Data": signed_data()})
    assert response.status_code == 200
    assert response.json()["user_id"] == "777"


async def test_query_cannot_override_signed_identity(auth_client):
    response = await auth_client.get(
        "/v1/me?user_id=888", headers={"X-Max-Init-Data": signed_data()}
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "data",
    [
        "",
        "hash=bad",
        signed_data().replace("777", "888"),
        signed_data(auth_date=int(time.time()) - 7200),
        signed_data(auth_date=int(time.time()) + 300),
        signed_data() + "&user=duplicate",
        signed_data(user_id=True),
        signed_data(user_id=-1),
    ],
)
async def test_invalid_init_data_is_rejected(auth_client, data):
    response = await auth_client.get("/v1/me", headers={"X-Max-Init-Data": data})
    assert response.status_code == 401


async def test_dev_identity_requires_explicit_opt_in(auth_client, monkeypatch):
    assert (await auth_client.get("/v1/me?user_id=777")).status_code == 401
    monkeypatch.setattr(get_settings(), "api_allow_dev_user_id", True)
    assert (await auth_client.get("/v1/me?user_id=777")).json()["user_id"] == "777"
    response = await auth_client.get("/v1/me?user_id=777", headers={"X-Max-Init-Data": "hash=bad"})
    assert response.status_code == 401  # invalid signed data cannot use the dev fallback


@pytest.mark.parametrize("path", ["/v1/events", "/v1/events/search", "/v1/events/import"])
async def test_calendar_endpoints_require_identity(auth_client, path):
    if path.endswith("import"):
        response = await auth_client.post(path, files={"file": ("test.csv", b"id,date,time,text")})
    else:
        response = await auth_client.get(path)
    assert response.status_code == 401
