"""MAX initData verification: https://dev.max.ru/docs/webapps/validation."""

import hashlib
import hmac
import json
import time
from typing import Annotated
from urllib.parse import parse_qsl

from fastapi import Header, HTTPException, Query

from common.config import get_settings


def verify_init_data(raw: str, token: str, ttl: int, now: float | None = None) -> str:
    """Return the signed user's ID; reject ambiguity, forgery and expired launches."""
    try:
        pairs = parse_qsl(raw, keep_blank_values=True, strict_parsing=True, max_num_fields=32)
        fields = dict(pairs)
        if len(pairs) != len(fields):
            raise ValueError("Duplicate fields")
        signature = fields.pop("hash")
        if len(signature) != 64:
            raise ValueError("Invalid hash")
        message = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
        secret = hmac.digest(b"WebAppData", token.encode(), "sha256")
        expected = hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Invalid signature")
        age = (time.time() if now is None else now) - int(fields["auth_date"])
        if not -30 <= age <= ttl:
            raise ValueError("Expired data")
        user_id = json.loads(fields["user"])["id"]
        # MAX identifies users with a positive int64, not an arbitrary client string.
        if isinstance(user_id, bool) or not isinstance(user_id, int) or not 0 < user_id < 2**63:
            raise ValueError("Invalid user")
        return str(user_id)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(
            status_code=401, detail="Данные MAX недействительны. Откройте мини-приложение заново."
        ) from exc


def get_current_user_id(
    init_data: Annotated[str | None, Header(alias="X-Max-Init-Data", max_length=16384)] = None,
    user_id: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
) -> str:
    settings = get_settings()
    if init_data is not None:
        if not settings.max_token:
            raise HTTPException(status_code=503, detail="Авторизация MAX не настроена")
        verified = verify_init_data(
            init_data, settings.max_token, settings.max_init_data_ttl_seconds
        )
        if user_id is not None and user_id != verified:
            raise HTTPException(status_code=403, detail="Пользователь не совпадает с данными MAX")
        return verified
    if settings.api_allow_dev_user_id and user_id:
        return user_id
    raise HTTPException(status_code=401, detail="Откройте мини-приложение через MAX")
