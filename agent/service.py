"""Serialize a user's requests and replay completed results across HTTP retries."""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID, uuid5

from redis.asyncio import Redis
from redis.exceptions import RedisError

from agent.core import Agent
from common.schemas import AgentResponse, AgentStatus, DraftAction

logger = logging.getLogger(__name__)


class AgentBusy(Exception):
    pass


class RequestConflict(Exception):
    pass


class AgentService:
    def __init__(
        self, agent: Agent, redis: Redis, ttl_seconds: int = 7 * 86400, timeout_seconds: float = 65
    ):
        self._agent, self._redis, self._ttl = agent, redis, ttl_seconds
        self._timeout = timeout_seconds

    @asynccontextmanager
    async def _locked(self, user_id: str, wait: bool = False) -> AsyncIterator[None]:
        lock = self._redis.lock(
            f"agent:lock:{user_id}", timeout=max(90, self._timeout + 30), thread_local=False
        )
        if not await lock.acquire(blocking=wait, blocking_timeout=self._timeout if wait else None):
            raise AgentBusy
        try:
            yield
        finally:
            try:
                await lock.release()
            except RedisError:
                # The operation may already be committed; preserve its result.
                logger.exception("Не удалось освободить блокировку агента")

    async def _run(
        self,
        user_id: str,
        request_id: UUID | None,
        payload: dict,
        operation: Callable[[], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        try:
            async with asyncio.timeout(self._timeout):
                return await self._execute(user_id, request_id, payload, operation)
        except TimeoutError:
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Не успел завершить запрос. Проверьте календарь перед повтором.",
            )

    async def _execute(
        self,
        user_id: str,
        request_id: UUID | None,
        payload: dict,
        operation: Callable[[], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        try:
            key = f"agent:request:{user_id}:{request_id}" if request_id else None
            fingerprint = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            previous = json.loads(await self._redis.get(key) or "null") if key else None
            if previous and previous["fingerprint"] != fingerprint:
                raise RequestConflict
            if previous and previous.get("response"):
                return AgentResponse.model_validate(previous["response"])
            async with self._locked(user_id, wait=previous is not None):
                envelope = {"fingerprint": fingerprint, "response": None}
                if key:
                    previous = await self._redis.get(key)
                    if previous:
                        previous = json.loads(previous)
                        if previous["fingerprint"] != fingerprint:
                            raise RequestConflict
                        if previous.get("response"):
                            return AgentResponse.model_validate(previous["response"])
                    # Bind the ID before running, including when the operation later fails.
                    await self._redis.set(key, json.dumps(envelope), ex=self._ttl)
                result = await operation()
                if key and result.status != AgentStatus.ERROR:
                    envelope["response"] = result.model_dump(mode="json")
                    try:
                        await self._redis.set(
                            key, json.dumps(envelope, ensure_ascii=False), ex=self._ttl
                        )
                    except RedisError:
                        logger.exception("Не удалось сохранить результат запроса агента")
                return result
        except AgentBusy:
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Ещё обрабатываю предыдущий запрос. Повторите через несколько секунд.",
            )
        except RequestConflict:
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Этот request_id уже использован для другого запроса.",
            )
        except TimeoutError:
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Не успел завершить запрос. Проверьте календарь перед повтором.",
            )
        except (RedisError, ValueError, KeyError, TypeError):
            logger.exception("Не удалось проверить состояние запроса агента")
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Контекст временно недоступен. Попробуйте ещё раз позже.",
            )

    async def handle(
        self,
        user_id: str,
        text: str,
        *,
        confirm_changes: bool = False,
        request_id: UUID | None = None,
    ) -> AgentResponse:
        return await self._run(
            user_id,
            request_id,
            {"operation": "message", "text": text, "confirm_changes": confirm_changes},
            lambda: self._agent.handle(
                user_id, text, confirm_changes=confirm_changes, request_id=request_id
            ),
        )

    async def execute_draft(self, user_id: str, draft: DraftAction) -> AgentResponse:
        return await self._run(
            user_id,
            uuid5(draft.id, "confirm"),
            {"operation": "confirm", "draft": draft.model_dump(mode="json")},
            lambda: self._agent.execute_draft(user_id, draft),
        )

    async def reset(self, user_id: str) -> None:
        async with asyncio.timeout(self._timeout):
            async with self._locked(user_id):
                await self._agent.reset(user_id)
