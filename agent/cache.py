"""Stable per-user cache routing without sending a raw MAX identity upstream."""

import hashlib
import hmac

from common.config import get_settings


def prompt_cache_session_id(user_id: str) -> str:
    # gpt2giga forwards X-Session-ID to GigaChat. This is only a cache hint,
    # never an authorization credential or a substitute for conversation history.
    return hmac.new(
        get_settings().llm_api_key.encode(),
        f"voice-assistant:prompt-cache:v1:{user_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
