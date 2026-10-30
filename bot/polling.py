"""MAX long-polling transport: тот же BotHandler, но без публичного HTTPS-адреса.

Вебхук требует домен с доверенным сертификатом; на машине без Docker и туннеля
бот проще поднимать циклом GET /updates: ``python -m bot.polling``.

Маркер курсора хранится в Redis: после рестарта бот не получит старые обновления
повторно, а то, что пришло, пока он лежал, — получит. Обновления обрабатываются
как задачки, чтобы длинный ASR/LLM-запрос не задерживал следующий длинный полл.
"""

import asyncio
import logging

import httpx
from redis.asyncio import Redis

from bot.agent_client import AgentClient
from bot.asr import make_asr
from bot.config import Settings
from bot.handler import BotHandler
from bot.max_api import MaxAPI, MaxAPIError
from bot.reminders import ReminderSender
from bot.state import StateStore
from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository

logger = logging.getLogger(__name__)
_MARKER_KEY = "bot:updates:marker"
_POLL_BACKOFF_SECONDS = 3.0
_active: set[asyncio.Task[None]] = set()


def _on_task_done(task: asyncio.Task[None]) -> None:
    _active.discard(task)
    if not task.cancelled():
        try:
            task.result()
        except Exception:
            logger.exception("MAX polling update handler failed")


class PollingBot:
    def __init__(
        self,
        max_api: MaxAPI,
        handler: BotHandler,
        redis: Redis,
        timeout: int = 20,
    ) -> None:
        self._max_api = max_api
        self._handler = handler
        self._redis = redis
        self._timeout = timeout

    async def poll_once(self, marker: int | None) -> int | None:
        """Один запрос /updates: раздать обновления, сохранить курсор. Новый marker → наружу."""
        data = await self._max_api.updates(marker, timeout=self._timeout)
        for update in data.get("updates") or []:
            if not isinstance(update, dict) or not isinstance(update.get("update_type"), str):
                logger.warning("MAX polling: пропуск неизвестного обновления")
                continue
            task = asyncio.create_task(self._handler.handle_update(update))
            _active.add(task)
            task.add_done_callback(_on_task_done)
        new_marker = data.get("marker")
        if new_marker is None:
            return marker
        await self._redis.set(_MARKER_KEY, int(new_marker))
        return int(new_marker)

    async def run_forever(self) -> None:
        marker_raw = await self._redis.get(_MARKER_KEY)
        marker = int(marker_raw) if marker_raw else None
        logger.info("MAX polling запущен (marker=%s)", marker)
        while True:
            try:
                marker = await self.poll_once(marker)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (401, 403):
                    raise RuntimeError("MAX отклонил MAX_TOKEN — проверьте токен бота") from exc
                logger.warning("MAX updates error: %s", exc)
                await asyncio.sleep(_POLL_BACKOFF_SECONDS)
            except (httpx.HTTPError, ValueError, MaxAPIError) as exc:
                # Сеть дрогнула — не падаем, пробуем снова; обработчики продолжаются сами.
                logger.warning("MAX updates недоступны: %s", exc)
                await asyncio.sleep(_POLL_BACKOFF_SECONDS)


async def stop_tasks() -> None:
    if _active:
        tasks = list(_active)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    settings = Settings.from_env()
    if not settings.max_token:
        raise RuntimeError("MAX_TOKEN обязателен: возьмите токен в MasterBot (bot/README.md)")
    common = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    max_api = MaxAPI(settings.max_token, settings.max_api_base, settings.max_voice_bytes)
    pool = await create_pool(common.database_url)
    embedder = build_embedder(common)
    repo = EventRepository(pool, embedder, common.tz)
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
    handler = BotHandler(max_api, asr, AgentClient(settings.agent_url), state, redis, repo)
    scheduler = asyncio.create_task(
        ReminderSender(
            repo,
            max_api,
            common.tz,
            interval_seconds=settings.reminder_interval_seconds,
            lookback_minutes=settings.reminder_lookback_minutes,
        ).run_forever()
    )
    bot = PollingBot(max_api, handler, redis, timeout=settings.polling_timeout)
    try:
        await redis.ping()
        await bot.run_forever()
    finally:
        scheduler.cancel()
        await asyncio.gather(scheduler, return_exceptions=True)
        await stop_tasks()
        await max_api.close()
        await redis.aclose()
        await embedder.aclose()
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
