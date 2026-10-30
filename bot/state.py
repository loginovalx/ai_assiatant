"""Per-user Redis state machine: idle -> recognizing -> thinking -> idle.

An empty key means idle. Value encodes stage and request ID so a stale job
cannot release a newer job's lock after the TTL elapsed.
"""

import uuid

from redis.asyncio import Redis

_TRANSITION = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0
"""

_RELEASE = """
local v = redis.call('GET', KEYS[1])
if v == ARGV[1] or v == ARGV[2] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class StateStore:
    def __init__(self, redis: Redis, ttl: int = 300):
        self.redis = redis
        self.ttl = ttl

    @staticmethod
    def _key(user_id: str) -> str:
        return f"bot:state:{user_id}"

    async def acquire(self, user_id: str, initial: str) -> str | None:
        if initial not in {"recognizing", "thinking"}:
            raise ValueError("Invalid initial state")
        request_id = uuid.uuid4().hex
        acquired = await self.redis.set(
            self._key(user_id), f"{initial}:{request_id}", ex=self.ttl, nx=True
        )
        return request_id if acquired else None

    async def transition(self, user_id: str, request_id: str) -> bool:
        return bool(
            await self.redis.eval(
                _TRANSITION,
                1,
                self._key(user_id),
                f"recognizing:{request_id}",
                f"thinking:{request_id}",
                self.ttl,
            )
        )

    async def release(self, user_id: str, request_id: str) -> None:
        await self.redis.eval(
            _RELEASE,
            1,
            self._key(user_id),
            f"recognizing:{request_id}",
            f"thinking:{request_id}",
        )

    async def current(self, user_id: str) -> str:
        value = await self.redis.get(self._key(user_id))
        if not value:
            return "idle"
        value = value.decode() if isinstance(value, bytes) else value
        return str(value).split(":", maxsplit=1)[0]

    async def reset_all(self) -> int:
        """At boot ONLY: single bot instance, as required by the team plan."""
        count = 0
        async for key in self.redis.scan_iter(match="bot:state:*", count=100):
            count += await self.redis.delete(key)
        return count
