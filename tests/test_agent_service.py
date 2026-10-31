import asyncio
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from agent.service import AgentBusy, AgentService
from common.schemas import AgentResponse, DraftAction


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.fail_once = False
        self.entered = asyncio.Event()
        self.release = None

    async def handle(self, user_id, text, **kwargs):
        self.calls += 1
        self.entered.set()
        if self.release:
            await self.release.wait()
        if self.fail_once:
            self.fail_once = False
            return AgentResponse(status="error", reply_text="Попробуйте снова")
        return AgentResponse(status="done", reply_text=text)

    async def execute_draft(self, user_id, draft):
        self.calls += 1
        return AgentResponse(status="done", reply_text="Готово")

    async def reset(self, user_id):
        self.resets += 1


async def test_request_replay_across_instances_and_conflicting_payload(redis):
    agent = FakeAgent()
    service = AgentService(agent, redis)
    request_id = uuid4()
    first = await service.handle("u", "Test", request_id=request_id)
    second = await AgentService(agent, redis).handle("u", "Test", request_id=request_id)
    conflict = await service.handle("u", "Different", request_id=request_id)
    assert first == second and agent.calls == 1
    assert conflict.status == "error"
    assert (await service.handle("other", "Test", request_id=request_id)).status == "done"
    assert agent.calls == 2


async def test_failed_request_can_retry(redis):
    agent = FakeAgent()
    agent.fail_once = True
    service = AgentService(agent, redis)
    request_id = uuid4()
    assert (await service.handle("u", "Test", request_id=request_id)).status == "error"
    assert (await service.handle("u", "Test", request_id=request_id)).status == "done"
    assert agent.calls == 2


async def test_same_user_serialized_and_reset_cannot_overtake_request(redis):
    agent = FakeAgent()
    agent.release = asyncio.Event()
    first_service, second_service = AgentService(agent, redis), AgentService(agent, redis)
    pending = asyncio.create_task(first_service.handle("u", "First"))
    await agent.entered.wait()
    try:
        assert (await second_service.handle("u", "Second")).status == "error"
        with pytest.raises(AgentBusy):
            await second_service.reset("u")
        assert agent.calls == 1 and agent.resets == 0
    finally:
        agent.release.set()
        await pending
    await second_service.reset("u")
    assert agent.resets == 1


async def test_confirm_retry_uses_recorded_result(redis):
    agent = FakeAgent()
    service = AgentService(agent, redis)
    draft = DraftAction(tool="delete_events", arguments={"event_ids": []}, preview="Удалить")
    await service.execute_draft("u", draft)
    await AgentService(agent, redis).execute_draft("u", draft)
    assert agent.calls == 1


async def test_redis_failure_prevents_execution():
    class BrokenRedis:
        async def get(self, key):
            raise RedisConnectionError("offline")

        def lock(self, *args, **kwargs):
            raise RedisConnectionError("offline")

    agent = FakeAgent()
    result = await AgentService(agent, BrokenRedis()).handle("u", "Test", request_id=uuid4())
    assert result.status == "error" and agent.calls == 0


async def test_same_request_waits_and_replays_inflight_result(redis):
    agent = FakeAgent()
    agent.release = asyncio.Event()
    service = AgentService(agent, redis)
    request_id = uuid4()
    first = asyncio.create_task(service.handle("u", "Same", request_id=request_id))
    await agent.entered.wait()
    duplicate = asyncio.create_task(service.handle("u", "Same", request_id=request_id))
    try:
        await asyncio.sleep(0.02)
        assert not duplicate.done()
    finally:
        agent.release.set()
    assert await first == await duplicate
    assert agent.calls == 1


async def test_deadline_releases_user_for_retry(redis):
    agent = FakeAgent()
    agent.release = asyncio.Event()
    service = AgentService(agent, redis, timeout_seconds=0.01)
    result = await service.handle("u", "Slow")
    assert result.status == "error"
    agent.release.set()
    assert (await service.handle("u", "Next")).status == "done"
