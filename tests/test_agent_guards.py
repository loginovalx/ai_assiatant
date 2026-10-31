from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

from openai.types.chat import ChatCompletion

from agent.core import Agent
from agent.prompt import build_system_prompt
from agent.tools import ToolResult
from common.schemas import AgentStatus, DraftAction, Event


def completion(content=None, call=False, calls=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": name, "arguments": json_args},
            }
            for i, (name, json_args) in enumerate(calls, start=1)
        ]
    elif call:
        message["tool_calls"] = [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "create_event",
                    "arguments": '{"event_type":"task","name":"Test"}',
                },
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class Memory:
    def __init__(self, fail_load=False, fail_save=False):
        self.fail_load, self.fail_save = fail_load, fail_save

    async def load(self, user_id):
        if self.fail_load:
            raise ConnectionError("Redis unavailable")
        return []

    async def append(self, *args):
        if self.fail_save:
            raise ConnectionError("Redis unavailable")


class Toolbox:
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def openai_tools(self):
        return []

    async def call(self, *args, **kwargs):
        self.calls += 1
        value = self.result.pop(0) if isinstance(self.result, list) else self.result
        if isinstance(value, Exception):
            raise value
        return value


def setup_agent(script, result=None, memory=None):
    async def create(**kwargs):
        value = script.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    toolbox = Toolbox(result)
    return Agent(llm, "test", toolbox, memory or Memory(), ZoneInfo("Europe/Moscow")), toolbox


def mutation():
    now = datetime.now(ZoneInfo("Europe/Moscow"))
    event = Event(
        id=uuid4(),
        user_id="u",
        type="task",
        name="Test",
        description="",
        start_at=None,
        status="active",
        created_at=now,
        updated_at=now,
    )
    return ToolResult(
        ok=True,
        payload=event.model_dump(mode="json"),
        is_mutation=True,
        events=[event.model_dump(mode="json")],
    )


async def test_memory_load_failure_returns_error():
    agent, _ = setup_agent([], memory=Memory(fail_load=True))
    assert (await agent.handle("u", "Test")).status == AgentStatus.ERROR


async def test_failed_tool_cannot_be_confirmed_by_model():
    agent, _ = setup_agent(
        [completion(call=True), completion("Создал")], ToolResult(ok=False, payload="Нужно время")
    )
    result = await agent.handle("u", "Test")
    assert result.status == AgentStatus.NEED_CLARIFICATION
    assert result.reply_text != "Создал"


async def test_successful_read_does_not_erase_failed_mutation():
    agent, _ = setup_agent(
        [completion(call=True), completion(calls=[("get_events", "{}")]), completion("Создал")],
        [ToolResult(ok=False, payload="Нужно время"), ToolResult(ok=True, payload=[], events=[])],
    )
    result = await agent.handle("u", "Создай встречу")
    assert result.status == AgentStatus.NEED_CLARIFICATION
    assert result.reply_text != "Создал" and not result.events


async def test_late_tool_failure_preserves_committed_event():
    agent, _ = setup_agent(
        [completion(call=True), completion(calls=[("get_events", "{}")]), completion("Готово")],
        [mutation(), ConnectionError("MCP down")],
    )
    result = await agent.handle("u", "Создай и покажи")
    assert len(result.events) == 1 and result.events[0].name == "Test"
    assert result.status == AgentStatus.NEED_CLARIFICATION


async def test_distinct_create_success_does_not_hide_failed_create_in_batch():
    agent, _ = setup_agent(
        [
            completion(
                calls=[
                    ("create_event", '{"event_type":"task","name":"First"}'),
                    ("create_event", '{"event_type":"task","name":"Second"}'),
                ]
            ),
            completion("Создал оба"),
        ],
        [ToolResult(ok=False, payload="First failed"), mutation()],
    )
    result = await agent.handle("u", "Создай два дела")
    assert result.status == AgentStatus.NEED_CLARIFICATION
    assert "First failed" in result.reply_text and len(result.events) == 1


async def test_step_limit_does_not_hide_partial_failure():
    agent, _ = setup_agent(
        [completion(call=True)] + [completion(calls=[("get_events", "{}")])] * 5,
        [mutation()] + [ToolResult(ok=False, payload="Read failed")] * 5,
    )
    result = await agent.handle("u", "Создай и покажи")
    assert result.status == AgentStatus.NEED_CLARIFICATION
    assert len(result.events) == 1 and "Read failed" in result.reply_text


async def test_malformed_completion_after_commit_preserves_result():
    malformed = completion("bad").model_copy(update={"choices": []})
    agent, _ = setup_agent([completion(call=True), malformed], mutation())
    result = await agent.handle("u", "Создай")
    assert result.status == AgentStatus.DONE and len(result.events) == 1


async def test_confirmed_mutation_survives_model_failure():
    agent, _ = setup_agent([completion(call=True), ConnectionError("LLM down")], mutation())
    result = await agent.handle("u", "Test")
    assert result.status == AgentStatus.DONE
    assert len(result.events) == 1
    assert "Test" in result.reply_text


async def test_confirmed_mutation_survives_history_save_failure():
    agent, _ = setup_agent(
        [completion(call=True), completion("Создал")], mutation(), Memory(fail_save=True)
    )
    assert (await agent.handle("u", "Test")).status == AgentStatus.DONE


async def test_duplicate_mutation_is_not_executed_twice():
    agent, toolbox = setup_agent(
        [completion(call=True), completion(call=True), completion("Создал")], mutation()
    )
    result = await agent.handle("u", "Test")
    assert result.status == AgentStatus.DONE
    assert toolbox.calls == 1


async def test_duplicate_preview_is_deduplicated():
    agent, toolbox = setup_agent(
        [completion(call=True), completion(call=True), completion("Создал")], mutation()
    )
    result = await agent.handle("u", "Test", confirm_changes=True)
    assert result.status == AgentStatus.NEEDS_CONFIRMATION
    assert toolbox.calls == 1


async def test_multiple_distinct_previews_require_one_action_at_a_time():
    calls = [
        ("create_event", '{"event_type":"task","name":"Первое"}'),
        ("create_event", '{"event_type":"task","name":"Второе"}'),
    ]
    agent, toolbox = setup_agent([completion(calls=calls)], mutation())
    result = await agent.handle("u", "Создай два дела", confirm_changes=True)
    assert result.status == AgentStatus.NEED_CLARIFICATION
    assert "по одной" in result.reply_text
    assert result.draft is None
    assert toolbox.calls == 1


async def test_empty_model_reply_is_error():
    agent, _ = setup_agent([completion("")])
    assert (await agent.handle("u", "Test")).status == AgentStatus.ERROR


async def test_fallback_keeps_meeting_conflict_warning():
    saved = mutation()
    conflict = {**saved.events[0], "id": str(uuid4()), "name": "Other meeting"}
    saved.events[0]["conflicts"] = [conflict]
    agent, _ = setup_agent([completion(call=True), ConnectionError("LLM down")], saved)
    result = await agent.handle("u", "Test")
    assert result.status == AgentStatus.DONE
    assert "Other meeting" in result.reply_text
    assert len(result.events) == 2


async def test_confirm_draft_uses_actual_result_and_survives_history_failure():
    saved = mutation()
    conflict = {**saved.events[0], "id": str(uuid4()), "name": "Реальная встреча"}
    saved.events[0]["conflicts"] = [conflict]
    agent, _ = setup_agent([], saved, Memory(fail_save=True))
    draft = DraftAction(tool="create_event", arguments={"name": "Test"}, preview="Старый текст")
    result = await agent.execute_draft("u", draft)
    assert result.status == AgentStatus.DONE
    assert "Test" in result.reply_text
    assert "Реальная встреча" in result.reply_text
    assert "Старый текст" not in result.reply_text
    assert len(result.events) == 2


def test_prompt_gives_explicit_week_bounds():
    prompt = build_system_prompt(datetime(2026, 10, 30, 15, tzinfo=ZoneInfo("Europe/Moscow")))
    assert "2026-10-26T00:00:00+03:00" in prompt
    assert "2026-11-02T00:00:00+03:00" in prompt
    assert "2026-11-09T00:00:00+03:00" in prompt
