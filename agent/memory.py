"""Память диалога в Redis: ключ agent:ctx:{user_id}, последние N реплик, TTL 7 дней.

Храним только реплики пользователя и итоговые ответы ассистента, без промежуточных
вызовов тулов: так история короче, а обрезка списка никогда не разрывает пару
«вызов тула → результат», которую LLM требует видеть целиком.
"""

import json
from typing import Literal, NotRequired, TypedDict

from redis.asyncio import Redis


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    clarification_text: NotRequired[str]


class ConversationMemory:
    def __init__(self, redis: Redis, max_messages: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def key(user_id: str) -> str:
        return f"agent:ctx:{user_id}"

    async def load(self, user_id: str) -> list[HistoryMessage]:
        raw = await self._redis.lrange(self.key(user_id), -self._max, -1)
        return [json.loads(item) for item in raw]

    async def append(self, user_id: str, *messages: HistoryMessage) -> None:
        key = self.key(user_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(json.dumps(m, ensure_ascii=False) for m in messages))
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def reset(self, user_id: str) -> None:
        await self._redis.delete(self.key(user_id))
