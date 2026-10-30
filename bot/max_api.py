"""Direct MAX HTTP API client; no dependency on unstable bot SDKs."""

import asyncio
import re
import time
from urllib.parse import urlparse

import httpx


class MaxAPIError(Exception):
    pass


# Files are fetched only from the trusted MAX CDN family, not arbitrary hosts.
_ALLOWED_MEDIA_DOMAINS = ("max.ru", "oneme.ru", "okcdn.ru", "mycdn.me")


def _safe_media_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == d or host.endswith(f".{d}") for d in _ALLOWED_MEDIA_DOMAINS
    )


class MaxAPI:
    def __init__(self, token: str, base_url: str, max_voice_bytes: int = 15_000_000):
        self.max_voice_bytes = max_voice_bytes
        self._send_locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._last_sent: dict[tuple[str, int], float] = {}
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": token},
            timeout=httpx.Timeout(40, connect=10),
        )
        # Separate client so the MAX API authorization token is not sent to the CDN.
        self.media_client = httpx.AsyncClient(timeout=45, follow_redirects=False)

    async def close(self) -> None:
        await self.client.aclose()
        await self.media_client.aclose()

    async def updates(self, marker: int | None, timeout: int = 20) -> dict:  # noqa: ASYNC109
        params: dict[str, str | int] = {
            "timeout": timeout,
            "limit": 100,
            "types": "message_created,bot_started,message_callback",
        }
        if marker is not None:
            params["marker"] = marker
        response = await self.client.get("/updates", params=params, timeout=timeout + 15)
        response.raise_for_status()
        return response.json()

    async def get_message(self, message_id: str) -> dict:
        """Fetch the complete message when an update omits media fields."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", message_id):
            raise MaxAPIError("Некорректный идентификатор сообщения")
        try:
            response = await self.client.get(f"/messages/{message_id}")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MaxAPIError("Не удалось получить подробности сообщения") from exc

    async def send(self, chat_id: int, text: str, *, to_user: bool = False, buttons=None) -> None:
        """Send in dialog/chat respecting the per-recipient rate limit.

        When MAX provides no chat_id for a dialog, use the sender's user_id.
        """
        target = ("user_id" if to_user else "chat_id", chat_id)
        lock = self._send_locks.setdefault(target, asyncio.Lock())
        async with lock:
            # MAX limits bursts to about two messages/sec per dialog.
            pause = 0.60 - (time.monotonic() - self._last_sent.get(target, -1e9))
            if pause > 0:
                await asyncio.sleep(pause)
            for attempt in range(3):
                response = await self.client.post(
                    "/messages",
                    params={target[0]: target[1]},
                    json={
                        "text": text[:4000],
                        **(
                            {
                                "attachments": [
                                    {"type": "inline_keyboard", "payload": {"buttons": buttons}}
                                ]
                            }
                            if buttons
                            else {}
                        ),
                    },
                )
                if response.status_code == 429 and attempt < 2:
                    await asyncio.sleep(1.0)
                    continue
                response.raise_for_status()
                self._last_sent[target] = time.monotonic()
                return

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        response = await self.client.post(
            "/answers",
            params={"callback_id": callback_id},
            json={"notification": text} if text else {},
        )
        response.raise_for_status()

    async def get_audio(self, url: str) -> bytes:
        return await self.get_file(url, self.max_voice_bytes, "голосовое сообщение")

    async def get_file(self, url: str, max_bytes: int, what: str = "файл") -> bytes:
        """Скачать вложение с CDN MAX с ограничением размера. Токен бота туда не уходит."""
        if not _safe_media_url(url):
            raise MaxAPIError(f"Недоверенный URL вложения ({what})")
        try:
            async with self.media_client.stream("GET", url) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and int(length) > max_bytes:
                    raise MaxAPIError(f"Слишком большой {what}")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise MaxAPIError(f"Слишком большой {what}")
                return bytes(data)
        except (httpx.HTTPError, ValueError) as exc:
            raise MaxAPIError(f"Не удалось скачать {what}") from exc
