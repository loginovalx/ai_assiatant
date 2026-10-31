"""Manage a MAX Webhook subscription from inside the bot Docker container.

Examples:
  python -m bot.subscribe add https://foo.trycloudflare.com/webhook
  python -m bot.subscribe list
  python -m bot.subscribe delete https://foo.trycloudflare.com/webhook
"""

import argparse
import asyncio
import json
import os
from urllib.parse import urlparse

import httpx

from bot.config import Settings


def verify_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.port not in (None, 443):
        raise ValueError("Webhook URL must be HTTPS on port 443, without a custom port")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Webhook URL must not include credentials or fragments")
    if parsed.path != "/webhook":
        raise ValueError("Webhook URL must end with /webhook")
    return value


async def run(command: str, url: str | None) -> None:
    settings = Settings.from_env()
    secret = os.environ.get("MAX_WEBHOOK_SECRET", "").strip()
    if not settings.max_token:
        raise SystemExit("MAX_TOKEN is missing")
    if command == "add":
        if not secret:
            raise SystemExit("Set MAX_WEBHOOK_SECRET in .env")
        assert url is not None
        payload = {
            "url": verify_url(url),
            "update_types": ["message_created", "bot_started", "message_callback"],
            "secret": secret,
        }
    elif command == "delete":
        assert url is not None
        verify_url(url)

    async with httpx.AsyncClient(
        base_url=settings.max_api_base,
        headers={"Authorization": settings.max_token},
        timeout=20,
    ) as client:
        if command == "add":
            response = await client.post("/subscriptions", json=payload)
        elif command == "list":
            response = await client.get("/subscriptions")
        else:
            response = await client.delete("/subscriptions", params={"url": url})
        response.raise_for_status()
        data = response.json()
        # Never print the bot token or the webhook secret.
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if isinstance(data, dict) and data.get("success") is False:
            raise SystemExit("MAX rejected the subscription operation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAX webhook subscription management")
    parser.add_argument("command", choices=["add", "list", "delete"])
    parser.add_argument("url", nargs="?")
    args = parser.parse_args()
    if args.command != "list" and not args.url:
        parser.error("The add/delete operation requires a webhook URL")
    asyncio.run(run(args.command, args.url))
