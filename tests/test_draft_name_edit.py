from datetime import datetime, timedelta

from agent.service import AgentService
from tests.scripted_llm import tool_call


async def test_name_only_edit_preserves_trusted_time_without_saving_old_draft(
    make_agent, redis, pool, tz
):
    when = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    base = {"event_type": "reminder", "start_at": when.isoformat()}
    agent, _ = make_agent(
        tool_call("create_event", {**base, "name": "Позвонить Ване"}),
        tool_call("create_event", {**base, "name": "Позвонить Ивану"}),
    )
    service = AgentService(agent, redis)
    first = await service.handle("u", "Напомни завтра в 10 позвонить Ване", confirm_changes=True)
    assert first.draft
    second = await service.handle(
        "u", "Исправь имя: Ивану", confirm_changes=True, edit_draft_id=first.draft.id
    )
    assert second.status == "needs_confirmation"
    assert "Ивану" in second.reply_text and "10:00" in second.reply_text
    assert await pool.fetchval("SELECT count(*) FROM events") == 0
    assert (await service.execute_draft("u", first.draft)).status == "error"
    confirmed = await service.execute_draft("u", second.draft)
    assert confirmed.status == "done"
    assert confirmed.events[0].start_at == when
    assert confirmed.events[0].name == "Позвонить Ивану"
    assert await pool.fetchval("SELECT count(*) FROM events") == 1


async def test_lost_confirmation_cache_keeps_approval_for_idempotent_retry(
    make_agent, redis, pool, tz
):
    import json

    from redis.exceptions import ConnectionError as RedisConnectionError

    class FailingReceiptRedis:
        def __init__(self):
            self.failed = False

        def __getattr__(self, name):
            return getattr(redis, name)

        async def set(self, key, value, **kwargs):
            if (
                key.startswith("agent:request:")
                and json.loads(value).get("response", {})
                and json.loads(value)["response"]["status"] == "done"
                and not self.failed
            ):
                self.failed = True
                raise RedisConnectionError("Receipt write lost")
            return await redis.set(key, value, **kwargs)

    agent, _ = make_agent(tool_call("create_event", {"event_type": "task", "name": "One"}))
    service = AgentService(agent, FailingReceiptRedis())
    issued = (await service.handle("u", "Создай One", confirm_changes=True)).draft
    first = await service.execute_draft("u", issued)
    assert first.status == "done"
    retried = await service.execute_draft("u", issued)
    assert retried.status == "done"
    assert first.events[0].id == retried.events[0].id
    assert await pool.fetchval("SELECT count(*) FROM events") == 1
