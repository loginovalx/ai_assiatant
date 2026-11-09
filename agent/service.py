"""Serialize a user's requests and replay completed results across HTTP retries."""

import asyncio
import hashlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID, uuid5

from redis.asyncio import Redis
from redis.exceptions import RedisError

from agent.core import Agent
from agent.dialogue import starts_new_request
from common.schemas import AgentResponse, AgentStatus, DraftAction

logger = logging.getLogger(__name__)


class AgentBusy(Exception):
    pass


class RequestConflict(Exception):
    pass


class InvalidDraft(Exception):
    pass


def _replayed_response(envelope: dict) -> AgentResponse:
    response = AgentResponse.model_validate(envelope["response"])
    metadata = envelope.get("response_metadata") or {}
    response._focused_event_id = metadata.get("focused_event_id")
    response._conflict_warning = metadata.get("conflict_warning") is True
    return response


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
                return _replayed_response(previous)
            async with self._locked(user_id, wait=previous is not None):
                envelope = {"fingerprint": fingerprint, "response": None}
                if key:
                    previous = await self._redis.get(key)
                    if previous:
                        previous = json.loads(previous)
                        if previous["fingerprint"] != fingerprint:
                            raise RequestConflict
                        if previous.get("response"):
                            return _replayed_response(previous)
                if payload.get("operation") == "confirm":
                    issued = await self._redis.get(f"agent:pending:{user_id}")
                    supplied = json.dumps(payload["draft"], sort_keys=True, ensure_ascii=False)
                    if not issued or not secrets.compare_digest(issued.encode(), supplied.encode()):
                        raise InvalidDraft
                if key:
                    # Bind the ID before running, including when the operation later fails.
                    await self._redis.set(key, json.dumps(envelope), ex=self._ttl)
                result = await operation()
                if result.status == AgentStatus.NEEDS_CONFIRMATION and result.draft is not None:
                    await self._redis.set(
                        f"agent:pending:{user_id}",
                        json.dumps(
                            result.draft.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
                        ),
                        ex=300,
                    )
                    await self._redis.delete(f"agent:voice-review:{user_id}")
                receipt_saved = False
                if key and result.status != AgentStatus.ERROR:
                    envelope["response"] = result.model_dump(mode="json")
                    envelope["response_metadata"] = {
                        "focused_event_id": result._focused_event_id,
                        "conflict_warning": result._conflict_warning,
                    }
                    try:
                        await self._redis.set(
                            key, json.dumps(envelope, ensure_ascii=False), ex=self._ttl
                        )
                        receipt_saved = True
                    except RedisError:
                        logger.exception("Не удалось сохранить результат запроса агента")
                if (
                    payload.get("operation") == "confirm"
                    and result.status == AgentStatus.DONE
                    and receipt_saved
                ):
                    try:
                        await self._redis.delete(f"agent:pending:{user_id}")
                    except RedisError:
                        logger.exception("Не удалось очистить подтверждённый черновик")
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
        except InvalidDraft:
            return AgentResponse(
                status=AgentStatus.ERROR,
                reply_text="Подтверждение устарело или изменено. Отправьте команду заново.",
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
        edit_draft_id: UUID | None = None,
    ) -> AgentResponse:
        async def operation() -> AgentResponse:
            review_key = f"agent:voice-review:{user_id}"
            auto_key = f"agent:auto-draft:{user_id}:{request_id}" if request_id else None
            staged = await self._redis.get(auto_key) if auto_key else None
            if staged:
                # The previous text request may have committed before its receipt
                # was saved. Replay its exact snapshot instead of invoking the model.
                return await self._agent.execute_draft(
                    user_id, DraftAction.model_validate_json(staged)
                )
            review = bool(
                confirm_changes
                or edit_draft_id is not None
                or (not starts_new_request(text) and await self._redis.get(review_key))
            )
            kwargs = {}
            if edit_draft_id is not None:
                pending = await self._redis.get(f"agent:pending:{user_id}")
                issued = DraftAction.model_validate_json(pending) if pending else None
                if issued is None or issued.id != edit_draft_id:
                    raise InvalidDraft
                source = issued.arguments.get("user_text")
                if issued.tool in {"create_event", "update_event"} and isinstance(source, str):
                    kwargs["trusted_edit_source"] = source
            result = await self._agent.handle(
                user_id, text, confirm_changes=review, request_id=request_id, **kwargs
            )
            if review and result.status in {
                AgentStatus.NEED_CLARIFICATION,
                AgentStatus.NEEDS_CONFIRMATION,
            }:
                # Persist only an unfinished voice dialogue; standalone failed voice
                # requests must not make the next ordinary text require confirmation.
                await self._redis.set(review_key, "1", ex=self._ttl)
            if not review and result.status == AgentStatus.NEEDS_CONFIRMATION and result.draft:
                # Bulk deletion and writes after lookup still validate a snapshot.
                # Typed commands approve it immediately without an extra user step.
                if auto_key:
                    await self._redis.set(auto_key, result.draft.model_dump_json(), ex=self._ttl)
                result = await self._agent.execute_draft(user_id, result.draft)
            if result.status == AgentStatus.DONE:
                await self._redis.delete(review_key)
            return result

        return await self._run(
            user_id,
            request_id,
            {
                "operation": "message",
                "text": text,
                "confirm_changes": confirm_changes,
                "edit_draft_id": str(edit_draft_id) if edit_draft_id else None,
            },
            operation,
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
                await self._redis.delete(
                    f"agent:pending:{user_id}", f"agent:voice-review:{user_id}"
                )
