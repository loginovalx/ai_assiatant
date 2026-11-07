"""Service credentials are derived from the existing private LLM_API_KEY.

MAX authenticates the user to the bot; these tokens authenticate the trusted
bot/agent services. They are not end-user credentials or public API tokens.
"""

import hashlib
import hmac

from common.config import get_settings


def internal_token(service: str) -> str:
    if service not in {"agent", "mcp"}:
        raise ValueError("Unknown internal service")
    master = get_settings().llm_api_key
    if not master or master.startswith("change-me"):
        return ""
    return hmac.new(
        master.encode(), f"voice-assistant:{service}:v1".encode(), hashlib.sha256
    ).hexdigest()


def valid_internal_token(token: str, service: str) -> bool:
    expected = internal_token(service)
    return bool(expected) and hmac.compare_digest(token.encode(), expected.encode())
