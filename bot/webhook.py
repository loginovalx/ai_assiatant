"""MAX Webhook transport (native voice messages + text) for the existing bot.

Start: uvicorn bot.webhook:app --host 0.0.0.0 --port 8080
MAX subscription: POST https://platform-api2.max.ru/subscriptions
"""

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from os import environ
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from redis.asyncio import Redis

from bot.agent_client import AgentClient
from bot.asr import make_asr
from bot.config import Settings
from bot.handler import BotHandler, UpdateInProgress, is_valid_update
from bot.max_api import MaxAPI
from bot.reminders import ReminderSender
from bot.state import StateStore
from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository

logger = logging.getLogger(__name__)
_MAX_UPDATE_BYTES = 1_000_000
_WEBHOOK_PROCESS_TIMEOUT_SECONDS = 25
_active: set[asyncio.Task[None]] = set()


def valid_secret(actual: str | None, expected: str) -> bool:
    """Constant-time verification. Missing configured secret is never accepted."""
    return bool(actual and expected and hmac.compare_digest(actual, expected))


def is_message(update: Any) -> bool:
    return is_valid_update(update)


def _on_task_done(task: asyncio.Task[None]) -> None:
    _active.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except UpdateInProgress:
        logger.info("MAX webhook duplicate arrived while update is in progress")
    except Exception:
        logger.exception("MAX webhook update handler failed after request timeout")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    secret = environ.get("MAX_WEBHOOK_SECRET", "").strip()
    if not settings.max_token:
        raise RuntimeError("MAX_TOKEN is required")
    if not 5 <= len(secret) <= 256 or not all(
        char.isascii() and (char.isalnum() or char in "_-") for char in secret
    ):
        raise RuntimeError("MAX_WEBHOOK_SECRET must be 5..256 characters: letters, digits, _ or -")

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    max_api = MaxAPI(settings.max_token, settings.max_api_base, settings.max_voice_bytes)
    # Общие настройки (БД, часовой пояс) — те же, что у остальных сервисов.
    common = get_settings()
    pool = await create_pool(common.database_url)
    embedder = build_embedder(common)
    repo = EventRepository(pool, embedder, common.tz)
    scheduler: asyncio.Task[None] | None = None
    try:
        await redis.ping()
        state = StateStore(redis, ttl=settings.bot_state_ttl)
        if settings.reset_state_on_start:
            deleted = await state.reset_all()
            logger.info("Deleted %s stale bot states", deleted)
        asr = make_asr(
            settings.asr_provider,
            settings.whisper_model,
            settings.asr_http_url,
            settings.asr_http_token,
        )
        app.state.handler = BotHandler(
            max_api, asr, AgentClient(settings.agent_url), state, redis, repo
        )
        app.state.webhook_secret = secret
        sender = ReminderSender(
            repo,
            max_api,
            common.tz,
            interval_seconds=settings.reminder_interval_seconds,
            lookback_minutes=settings.reminder_lookback_minutes,
        )
        scheduler = asyncio.create_task(sender.run_forever())
        logger.info("MAX Webhook ready; ASR_PROVIDER=%s", settings.asr_provider)
        yield
    finally:
        if scheduler is not None:
            scheduler.cancel()
            await asyncio.gather(scheduler, return_exceptions=True)
        if _active:
            tasks = list(_active)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await max_api.close()
        await redis.aclose()
        await embedder.aclose()
        await pool.close()


app = FastAPI(title="MAX Bot Webhook", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "transport": "webhook"}


@app.post("/webhook")
async def webhook(request: Request) -> dict[str, bool]:
    actual = request.headers.get("X-Max-Bot-Api-Secret")
    if not valid_secret(actual, request.app.state.webhook_secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    body = await request.body()
    if len(body) > _MAX_UPDATE_BYTES:
        raise HTTPException(status_code=413, detail="Update is too large")
    try:
        update = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not is_message(update):
        logger.warning("MAX webhook received malformed update; skipping")
        return {"ok": True}

    if update.get("update_type") == "message_created":
        message = update.get("message")
        if not isinstance(message, dict) or not message:
            logger.warning("MAX webhook delivered message_created without message")
        else:
            body = message.get("body") or {}
            types = [a.get("type") for a in body.get("attachments") or [] if isinstance(a, dict)]
            logger.info("MAX webhook received message_created; attachments=%s", types)

    # MAX requires HTTP 200 within 30 seconds. Keep the work alive after our 25s
    # deadline so the next MAX retry can observe its completion or pending lease.
    task = asyncio.create_task(request.app.state.handler.handle_update(update))
    _active.add(task)
    task.add_done_callback(_on_task_done)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=_WEBHOOK_PROCESS_TIMEOUT_SECONDS)
    except UpdateInProgress as exc:
        raise HTTPException(status_code=503, detail="Update is already processing") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="Update is still processing") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Update processing failed") from exc
    return {"ok": True}
