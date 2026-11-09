"""Model selection, continuation and fallback before any calendar mutation."""

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.core import Agent
from agent.memory import ConversationMemory
from agent.routing import select_model
from agent.tools import ASK_USER_TOOL, ToolResult, to_openai_tool
from common.embeddings import GigaChatEmbedder
from tests.scripted_llm import ScriptedLLM, text, tool_call
from tests.test_agent_guards import Toolbox, mutation

MAX = "GigaChat-2-Max"
PRO = "GigaChat-2-Pro"
CREATE = {"event_type": "task", "name": "Test"}
SIMPLE = "Напомни завтра в 18:00 купить молоко"


class Memory:
    def __init__(self):
        self.history = []

    async def load(self, user_id):
        return list(self.history)

    async def append(self, user_id, *messages):
        self.history.extend(messages)


def make_agent(script, results=None, *, memory=None, simple_model=PRO):
    llm, toolbox = ScriptedLLM(*script), Toolbox(results)
    agent = Agent(
        llm,
        MAX,
        toolbox,
        memory or Memory(),
        ZoneInfo("Europe/Moscow"),
        simple_model=simple_model,
    )
    return agent, llm, toolbox


@pytest.fixture(scope="session")
async def schemas():
    from mcp_server.server import mcp

    return [
        to_openai_tool(tool.name, tool.description or "", tool.parameters)
        for tool in await mcp.list_tools()
    ] + [ASK_USER_TOOL]


@pytest.mark.parametrize(
    "query",
    [
        "Что у меня завтра?",
        "Покажи расписание на сегодня",
        "Какие встречи у меня на следующей неделе?",
        "Покажи задачи на этой неделе",
        SIMPLE,
        "Напомни сегодня в 23:45 отправить отчёт",
        "Добавь задачу купить хлеб",
        "Создай задачу подготовить отчёт",
        "Поставь встречу с Иваном завтра в 15:00 на час",
        "Встреча с Иваном завтра в 15:00 на 30 минут",
    ],
)
def test_only_clear_allowlisted_requests_use_pro(query):
    assert select_model(query, [], primary_model=MAX, simple_model=PRO).model == PRO


@pytest.mark.parametrize(
    "query",
    [
        "Перенеси её на завтра",
        "Удали все задачи",
        "Поставь встречу с Иваном завтра",
        "Поставь встречу с Иваном завтра в 15:00",
        "Напомни завтра вечером купить молоко",
        "Напомни завтра в 3 купить молоко",
        "Напомни про вылет завтра в 20:00",
        "Напомни завтра в 18:00 про вылет в 20:00",
        "Напомни завтра в 18:00 сделать это",
        "Напомни завтра в 18:00 поздравить Иванова в пятницу",
        "Поставь встречу с Иваном завтра в пятницу в 15:00 на час",
        "Поставь встречу с Иваном завтра 2026-11-13 в 15:00 на час",
        "Напомни завтра в 18:00 купить молоко и создай встречу",
        "Напомни завтра в 18:00 купить молоко. Покажи календарь",
        "Напомни завтра в 18:00 купить молоко, если магазины открыты",
        "Создай задачу купить билеты до конца недели",
        "Создай задачу позвонить в пятницу",
        "Что у меня завтра и на следующей неделе?",
        "Найди встречу с Иваном",
        "Да",
        "Привет!",
    ],
)
def test_contextual_ambiguous_and_multi_action_requests_stay_on_max(query):
    assert select_model(query, [], primary_model=MAX, simple_model=PRO).model == MAX


async def test_simple_write_and_read_use_pro_without_a_classifier_call():
    agent, llm, toolbox = make_agent([tool_call("create_event", CREATE)], mutation())
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO]
    assert toolbox.calls == 1
    agent, llm, _ = make_agent(
        [tool_call("get_events", {})], ToolResult(ok=True, payload=[], events=[])
    )
    assert (await agent.handle("u", "Что у меня завтра?")).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO]


async def test_routing_can_be_disabled_for_all_requests():
    agent, llm, _ = make_agent([tool_call("create_event", CREATE)], mutation(), simple_model=None)
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert llm.requests[0]["model"] == MAX


@pytest.mark.parametrize("initial_model", [PRO, MAX])
async def test_clarification_keeps_its_model_across_agent_restart(initial_model):
    memory = Memory()
    agent, first, _ = make_agent(
        [tool_call("ask_user", {"question": "Во сколько?"})], memory=memory
    )
    initial = SIMPLE if initial_model == PRO else "Поставь встречу с Иваном завтра"
    assert (await agent.handle("u", initial)).status == "need_clarification"
    assert first.requests[0]["model"] == initial_model
    agent, second, _ = make_agent(
        [tool_call("ask_user", {"question": "Уточните длительность"})], memory=memory
    )
    await agent.handle("u", "В 15:00")
    assert second.requests[0]["model"] == initial_model
    assert all(set(message) <= {"role", "content"} for message in second.requests[0]["messages"])


async def test_a_new_independent_request_is_not_stuck_on_the_previous_model():
    memory = Memory()
    memory.history = [
        {
            "role": "assistant",
            "content": "Во сколько?",
            "pending_model": MAX,
            "clarification_text": "Поставь встречу с Иваном завтра",
        }
    ]
    agent, llm, _ = make_agent([tool_call("create_event", CREATE)], mutation(), memory=memory)
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert llm.requests[0]["model"] == PRO


@pytest.mark.parametrize(
    "bad",
    [
        tool_call("read_env", {}),
        tool_call("create_event", []),
        tool_call("create_event", {"event_type": "invalid", "name": "Test"}),
        tool_call("create_event", {"event_type": "task"}),
        tool_call("create_event", {**CREATE, "duration_minutes": -1}),
        tool_call("create_event", {**CREATE, "wrong_field": 1}),
        tool_call("delete_event", {"event_id": "00000000-0000-0000-0000-000000000000"}),
        text("Готово, создал"),
        text("Invalid completion").model_copy(update={"choices": []}),
        ConnectionError("Pro is unavailable"),
    ],
)
async def test_invalid_pro_results_upgrade_once_before_any_write(bad, schemas):
    agent, llm, toolbox = make_agent([bad, tool_call("create_event", CREATE)], mutation())

    async def openai_tools():
        return schemas

    toolbox.openai_tools = openai_tools
    response = await agent.handle("u", SIMPLE)
    assert response.status == "done" and toolbox.calls == 1
    assert [request["model"] for request in llm.requests] == [PRO, MAX]
    assert llm.requests[0]["extra_headers"] != llm.requests[1]["extra_headers"]


async def test_invalid_json_upgrades_before_mcp():
    bad = tool_call("create_event", CREATE)
    bad.choices[0].message.tool_calls[0].function.arguments = "{invalid"
    agent, llm, toolbox = make_agent([bad, tool_call("create_event", CREATE)], mutation())
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO, MAX]
    assert toolbox.calls == 1


async def test_simple_read_cannot_silently_become_a_limited_search():
    result = ToolResult(ok=True, payload=[], events=[])
    agent, llm, toolbox = make_agent(
        [tool_call("search_events", {"query": "завтра"}), tool_call("get_events", {})], result
    )
    assert (await agent.handle("u", "Что у меня завтра?")).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO, MAX]
    assert toolbox.calls == 1


async def test_whole_batch_is_checked_before_its_first_mutation():
    from tests.test_agent_guards import completion

    first = completion(calls=[("create_event", '{"name":"Test"}'), ("read_env", "{}")])
    agent, llm, toolbox = make_agent([first, tool_call("create_event", CREATE)], mutation())
    seen_generations = []
    original = toolbox.call

    async def call(*args, **kwargs):
        seen_generations.append(len(llm.requests))
        return await original(*args, **kwargs)

    toolbox.call = call
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert seen_generations == [2]


async def test_optional_nulls_do_not_trigger_a_paid_fallback(schemas):
    args = {**CREATE, "duration_minutes": None, "start_at": None}
    agent, llm, toolbox = make_agent([tool_call("create_event", args)], mutation())

    async def openai_tools():
        return schemas

    toolbox.openai_tools = openai_tools
    assert (await agent.handle("u", SIMPLE)).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO]


async def test_failed_max_does_not_retry_pro_or_execute_any_tool():
    agent, llm, toolbox = make_agent([ConnectionError("Pro"), ConnectionError("Max")])
    assert (await agent.handle("u", SIMPLE)).status == "error"
    assert [request["model"] for request in llm.requests] == [PRO, MAX]
    assert toolbox.calls == 0


async def test_ambiguous_write_transport_failure_never_restarts_with_max():
    agent, llm, toolbox = make_agent(
        [tool_call("create_event", CREATE)], ConnectionError("Receipt lost after commit")
    )
    assert (await agent.handle("u", SIMPLE)).status == "error"
    assert [request["model"] for request in llm.requests] == [PRO]
    assert toolbox.calls == 1


async def test_no_fallback_after_a_successful_write_even_in_a_pending_dialogue():
    memory = Memory()
    memory.history = [
        {
            "role": "assistant",
            "content": "Во сколько?",
            "pending_model": PRO,
            "pending_tool": "create_event",
            "clarification_text": "Напомни завтра купить молоко",
        }
    ]
    agent, llm, toolbox = make_agent(
        [tool_call("create_event", CREATE), ConnectionError("No receipt")],
        mutation(),
        memory=memory,
    )
    assert (await agent.handle("u", "В 18:00")).status == "done"
    assert [request["model"] for request in llm.requests] == [PRO, PRO]
    assert toolbox.calls == 1


async def test_upgrade_is_retained_for_the_next_clarification():
    memory = Memory()
    agent, llm, _ = make_agent(
        [ConnectionError("Pro"), tool_call("ask_user", {"question": "О чём напомнить?"})],
        memory=memory,
    )
    assert (await agent.handle("u", SIMPLE)).status == "need_clarification"
    agent, llm, _ = make_agent([text("Уточните запрос")], memory=memory)
    await agent.handle("u", "Купить молоко")
    assert llm.requests[0]["model"] == MAX


async def test_clarification_retains_the_original_intent_before_any_write():
    agent, llm, toolbox = make_agent(
        [
            tool_call("ask_user", {"question": "Уточните время"}),
            tool_call("delete_event", {"event_id": "00000000-0000-0000-0000-000000000000"}),
            tool_call("ask_user", {"question": "Уточните напоминание"}),
        ]
    )
    await agent.handle("u", SIMPLE)
    assert (await agent.handle("u", "В 19:00")).status == "need_clarification"
    assert [request["model"] for request in llm.requests] == [PRO, PRO, MAX]
    assert toolbox.calls == 0


async def test_pending_model_survives_real_redis_and_is_separate_per_user(redis):
    memory = ConversationMemory(redis, 10, 3600)
    agent, _, _ = make_agent([tool_call("ask_user", {"question": "Во сколько?"})], memory=memory)
    await agent.handle("u", SIMPLE)
    agent, llm, _ = make_agent(
        [tool_call("ask_user", {"question": "Уточните запрос"}), text("Уточните запрос")],
        memory=ConversationMemory(redis, 10, 3600),
    )
    await agent.handle("u", "В 18:00")
    await agent.handle("other", "В 18:00")
    assert [request["model"] for request in llm.requests] == [PRO, MAX]


def test_search_judge_model_is_independently_configurable(monkeypatch):
    from common.config import Settings

    monkeypatch.setattr("common.embeddings.GigaChat", lambda **kwargs: SimpleNamespace())
    assert GigaChatEmbedder(Settings(llm_model=MAX))._chat_model == MAX
    assert GigaChatEmbedder(Settings(llm_model=MAX, llm_search_model=PRO))._chat_model == PRO
