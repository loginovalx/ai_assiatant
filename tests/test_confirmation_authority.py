from uuid import uuid4

import pytest

from agent.service import AgentService
from common.schemas import AgentResponse, DraftAction
from tests.test_agent_service import FakeAgent


class DraftAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.draft = DraftAction(
            tool="create_event", arguments={"name": "Approved"}, preview="Создать Approved"
        )

    async def handle(self, user_id, text, **kwargs):
        return AgentResponse(
            status="needs_confirmation", reply_text=self.draft.preview, draft=self.draft
        )


async def test_forged_unissued_confirmation_cannot_execute(redis):
    agent = DraftAgent()
    service = AgentService(agent, redis)
    result = await service.execute_draft("u", agent.draft)
    assert result.status == "error" and agent.calls == 0


@pytest.mark.parametrize("tamper", ["arguments", "tool", "owner", "id", "preview"])
async def test_issued_confirmation_is_bound_to_owner_and_exact_payload(redis, tamper):
    agent = DraftAgent()
    service = AgentService(agent, redis)
    issued = (await service.handle("u", "Создай Approved")).draft
    changes = {
        "arguments": {"name": "Injected"},
        "tool": "delete_events",
        "id": uuid4(),
        "preview": "Safe operation",
    }
    changed = issued.model_copy(update={tamper: changes[tamper]}) if tamper != "owner" else issued
    result = await service.execute_draft("victim" if tamper == "owner" else "u", changed)
    assert result.status == "error" and agent.calls == 0
    assert (await service.execute_draft("u", issued)).status == "done"
    assert agent.calls == 1


async def test_expired_and_superseded_drafts_are_rejected(redis):
    agent = DraftAgent()
    service = AgentService(agent, redis)
    first = (await service.handle("u", "One")).draft
    agent.draft = agent.draft.model_copy(update={"id": uuid4()})
    second = (await service.handle("u", "Two")).draft
    assert (await service.execute_draft("u", first)).status == "error"
    await redis.delete("agent:pending:u")
    assert (await service.execute_draft("u", second)).status == "error"
    assert agent.calls == 0


async def test_issued_draft_replay_recovers_receipt_without_second_commit(redis):
    agent = DraftAgent()
    service = AgentService(agent, redis)
    issued = (await service.handle("u", "Create")).draft
    first = await service.execute_draft("u", issued)
    second = await AgentService(agent, redis).execute_draft("u", issued)
    assert first == second and first.status == "done"
    assert agent.calls == 1
