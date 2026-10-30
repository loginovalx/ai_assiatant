"""Черновики голосовых команд в Redis: действие ждёт «Подтвердить» и в базу не попадает.

Почему Redis, а не таблица: черновик живёт минуты и не должен засорять календарь.
Переживает перезапуск бота, но сам протухает по TTL, если пользователь не ответил.
"""

import logging

from redis.asyncio import Redis

from bot.models import DraftAction

logger = logging.getLogger(__name__)


class DraftStore:
    def __init__(self, redis: Redis, ttl: int = 3600) -> None:
        self._redis = redis
        self._ttl = ttl

    @staticmethod
    def _key(user_id: str) -> str:
        return f"bot:draft:{user_id}"

    async def save(self, user_id: str, draft: DraftAction) -> None:
        await self._redis.set(self._key(user_id), draft.model_dump_json(), ex=self._ttl)

    async def load(self, user_id: str) -> DraftAction | None:
        raw = await self._redis.get(self._key(user_id))
        if not raw:
            return None
        try:
            return DraftAction.model_validate_json(raw)
        except ValueError:
            logger.warning("Испорченный черновик пользователя %s, удаляю", user_id)
            await self.drop(user_id)
            return None

    async def drop(self, user_id: str) -> None:
        await self._redis.delete(self._key(user_id))
