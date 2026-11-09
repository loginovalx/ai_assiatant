"""Avoid paid formatting passes while preserving unfinished agent work."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from openai import AsyncOpenAI

from agent.cache import prompt_cache_session_id
from agent.core import Agent
from agent.prompt import build_system_prompt
from agent.tools import ToolResult
from tests.scripted_llm import ScriptedLLM, text, tool_call
from tests.test_agent_guards import Memory, Toolbox, mutation


def make_agent(script, results):
    llm = ScriptedLLM(*script)
    toolbox = Toolbox(results)
    return Agent(llm, "test", toolbox, Memory(), ZoneInfo("Europe/Moscow")), llm, toolbox


@pytest.mark.parametrize(
    ("tool", "request_text", "status"),
    [
        ("create_event", "Создай задачу подготовить отчёт", "active"),
        ("create_event", "Напомни завтра в 12:00 позвонить Ивану", "active"),
        ("create_event", "Встреча с Иваном завтра в 12:00 на час", "active"),
        ("update_event", "Перенеси её на завтра в 12:00", "active"),
        ("complete_event", "Отметь задачу выполненной", "done"),
        ("delete_event", "Удали задачу", "cancelled"),
    ],
)
async def test_single_mutation_needs_one_generation(tool, request_text, status):
    result = mutation()
    result.events[0]["status"] = status
    agent, llm, toolbox = make_agent([tool_call(tool, {"name": "Test"})], result)
    response = await agent.handle("u", request_text)
    if tool == "delete_event":
        assert response.status == "needs_confirmation" and response.events == []
        response = await agent.execute_draft("u", response.draft)
    assert response.status == "done"
    assert response.events[0].status == status
    assert len(llm.requests) == 1
    assert toolbox.calls == (2 if tool == "delete_event" else 1)
    assert "Test" in response.reply_text


async def test_conflicts_and_history_survive_without_a_formatting_generation():
    result = mutation()
    conflict = mutation().events[0]
    conflict["name"] = "Другая встреча"
    result.events[0]["conflicts"] = [conflict]
    agent, llm, _ = make_agent([tool_call("create_event", {"name": "Test"})], result)
    saved = []

    async def append(user_id, *messages):
        saved.extend(messages)

    agent._memory.append = append
    response = await agent.handle("u", "Создай встречу завтра в 12:00 на час")
    assert response.status == "done" and response._conflict_warning
    assert "Другая встреча" in response.reply_text
    assert len(response.events) == 2
    assert saved[-1]["conflict_warning"] is True
    assert conflict["id"] in saved[-1]["content"]
    assert len(llm.requests) == 1


async def test_failed_mutation_can_be_repaired_before_returning():
    result = mutation()
    agent, llm, toolbox = make_agent(
        [
            tool_call("create_event", {"name": "Test"}),
            tool_call("create_event", {"name": "Test", "start_at": "2026-11-10T12:00:00+03:00"}),
        ],
        [ToolResult(ok=False, payload="Нужно время"), result],
    )
    response = await agent.handle("u", "Напомни завтра в 12:00 Test")
    assert response.status == "done"
    assert len(llm.requests) == toolbox.calls == 2
    assert "Нужно время" in llm.requests[1]["messages"][-1]["content"]
    assert llm.requests[0]["extra_headers"] == llm.requests[1]["extra_headers"]


@pytest.mark.parametrize(
    "request_text",
    [
        "Создай задачу Test и покажи календарь",
        "Создай задачу Test; покажи календарь",
        "Создай задачу Test. Покажи календарь",
        "Создай задачу Test, затем покажи календарь",
        "Создай задачу Test, show calendar",
    ],
)
async def test_create_then_read_keeps_the_followup(request_text):
    created = mutation()
    agent, llm, toolbox = make_agent(
        [
            tool_call("create_event", {"name": "Test"}),
            tool_call("get_events", {}),
            text("Готово"),
        ],
        [created, ToolResult(ok=True, payload=[], events=[])],
    )
    response = await agent.handle("u", request_text)
    assert response.status == "done"
    assert toolbox.calls == 2 and len(llm.requests) == 3


@pytest.mark.parametrize(
    "request_text",
    [
        "Создай задачу Test и создай задачу Second",
        "Создай задачу Test. Создай задачу Second",
        "Создай задачу Test. Напоминание завтра в 10:00",
        "Создай две задачи Test, Second",
    ],
)
async def test_multiple_mutations_are_not_cut_off(request_text):
    first, second = mutation(), mutation()
    second.events[0]["name"] = "Second"
    agent, llm, toolbox = make_agent(
        [
            tool_call("create_event", {"name": "Test"}),
            tool_call("create_event", {"name": "Second"}),
            text("Готово"),
        ],
        [first, second],
    )
    response = await agent.handle("u", request_text)
    assert response.status == "done"
    assert {event.name for event in response.events} == {"Test", "Second"}
    assert toolbox.calls == 2 and len(llm.requests) == 3


async def test_partial_failure_cannot_be_reported_as_success_by_the_fast_path():
    result = mutation()
    bad = ToolResult(ok=False, payload="Не удалось сохранить Second")
    agent, llm, _ = make_agent(
        [
            tool_call("create_event", {"name": "Test"}),
            tool_call("create_event", {"name": "Second"}),
            text("Готово"),
        ],
        [result, bad],
    )
    response = await agent.handle("u", "Создай задачу Test и создай задачу Second")
    assert response.status == "need_clarification"
    assert "Сохранено" in response.reply_text and "Second" in response.reply_text
    assert len(response.events) == 1 and len(llm.requests) == 3


async def test_voice_preview_and_confirmation_keep_their_existing_authority():
    result = mutation()
    agent, llm, toolbox = make_agent([tool_call("create_event", {"name": "Test"})], result)
    response = await agent.handle("u", "Создай задачу Test", confirm_changes=True)
    assert response.status == "needs_confirmation" and response.draft
    assert len(llm.requests) == toolbox.calls == 1
    confirmed = await agent.execute_draft("u", response.draft)
    assert confirmed.status == "done"
    assert toolbox.calls == 2 and len(llm.requests) == 1


def test_clock_changes_preserve_the_instruction_prefix_and_calendar_facts():
    now = datetime(2026, 11, 9, 15, 30, 1, tzinfo=ZoneInfo("Europe/Moscow"))
    first = build_system_prompt(now)
    second = build_system_prompt(now + timedelta(seconds=1))
    static_prefix = first.partition("## Сейчас")[0]
    assert len(static_prefix) > 9000
    assert second.startswith(static_prefix)
    assert first.count("## Сейчас") == 1
    assert now.isoformat() in first
    assert (now + timedelta(seconds=1)).isoformat() in second
    assert "2026-11-10: вторник" in first
    assert "Не назначай 09:00, 13:00 или 19:00 самостоятельно" in first


def test_cache_sessions_are_stable_private_and_bound_to_the_service_key(monkeypatch):
    monkeypatch.setattr("agent.cache.get_settings", lambda: SimpleNamespace(llm_api_key="first"))
    first = prompt_cache_session_id("MAX-user-12345")
    assert first == prompt_cache_session_id("MAX-user-12345")
    assert first != prompt_cache_session_id("MAX-user-67890")
    assert "12345" not in first and len(first) == 64
    monkeypatch.setattr("agent.cache.get_settings", lambda: SimpleNamespace(llm_api_key="second"))
    assert first != prompt_cache_session_id("MAX-user-12345")


async def test_real_sdk_sends_the_session_header_without_changing_request_context():
    seen = []

    def respond(request):
        seen.append((request.headers["X-Session-ID"], json.loads(request.content)))
        return httpx.Response(200, json=text("Привет").model_dump())

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as llm:
        agent = Agent(llm, "test", Toolbox(None), Memory(), ZoneInfo("Europe/Moscow"))
        for user_id in ("first", "first", "other"):
            assert (await agent.handle(user_id, "Привет")).status == "done"
    assert seen[0][0] == seen[1][0] != seen[2][0]
    assert all(
        payload["messages"][-1] == {"role": "user", "content": "Привет"} for _, payload in seen
    )
