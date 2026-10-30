"""Агент со скриптованной LLM, но с настоящими MCP-сервером, Postgres и Redis.

Проверяем не качество GigaChat (это делает датасет E), а нашу логику вокруг неё:
подстановку user_id, уточнения, ошибки тулов, историю, формат ответа.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
import pytest
from openai.types.chat import ChatCompletion
from redis.asyncio import Redis

from agent.core import Agent
from agent.memory import ConversationMemory
from agent.prompt import build_system_prompt
from agent.tools import McpToolbox, to_openai_tool
from common.schemas import AgentStatus
from mcp_server.server import mcp

USER = "user-1"
TEST_REDIS = os.environ.get("TEST_REDIS_URL")


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> ChatCompletion:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }
    )


def text(content: str) -> ChatCompletion:
    return _completion({"role": "assistant", "content": content})


def _completion(message: dict[str, Any]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "GigaChat-2-Max",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class ScriptedLLM:
    """Отдаёт заранее заданные ответы и запоминает, что ей прислали."""

    def __init__(self, *script: ChatCompletion | Exception) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # снимок
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
async def redis():
    if not TEST_REDIS:
        pytest.skip("TEST_REDIS_URL не задан")
    client = Redis.from_url(TEST_REDIS, decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
def make_agent(pool, redis, tz):
    def factory(*script):
        llm = ScriptedLLM(*script)
        agent = Agent(
            llm=llm,
            model="GigaChat-2-Max",
            toolbox=McpToolbox(mcp),
            memory=ConversationMemory(redis, max_messages=10, ttl_seconds=3600),
            tz=tz,
        )
        return agent, llm

    return factory


def tomorrow_at(tz, hour: int) -> datetime:
    now = datetime.now(tz)
    return (now + timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)


# --- схема тулов --------------------------------------------------------------


async def test_tools_hide_user_id_and_have_no_anyof():
    tools = await McpToolbox(mcp).openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert names == [
        "create_event",
        "get_events",
        "search_events",
        "update_event",
        "complete_event",
        "delete_event",
        "delete_events",
        "ask_user",
    ]

    dumped = json.dumps(tools, ensure_ascii=False)
    assert "user_id" not in dumped
    assert "anyOf" not in dumped and "additionalProperties" not in dumped

    create = tools[0]["function"]["parameters"]
    assert create["properties"]["event_type"]["enum"] == ["task", "meeting", "reminder"]
    assert set(create["required"]) == {"event_type", "name"}


def test_simplify_nullable_array():
    schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "types": {
                "anyOf": [
                    {"type": "array", "items": {"enum": ["a"], "type": "string"}},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Фильтр",
            },
        },
        "required": ["user_id"],
    }
    params = to_openai_tool("t", "d", schema)["function"]["parameters"]
    assert params["properties"] == {
        "types": {
            "type": "array",
            "items": {"enum": ["a"], "type": "string"},
            "description": "Фильтр",
        }
    }
    assert params["required"] == []


def test_prompt_has_time_and_calendar(tz):
    now = datetime(2026, 10, 29, 19, 45, 12, 999, tzinfo=tz)
    prompt = build_system_prompt(now)
    assert "2026-10-29T19:45:12+03:00, четверг" in prompt
    assert "2026-10-30: пятница, 30 октября (завтра)" in prompt
    assert "2026-11-02: понедельник, 2 ноября" in prompt


def test_prompt_forbids_inventing_hour_and_retelling_description(tz):
    """Три правила против багов, которые ловились на живом GigaChat: выдуманный час,
    молча потерянная длительность и пересказ description вместо полей записи."""
    prompt = build_system_prompt(datetime(2026, 10, 29, 19, 45, tzinfo=tz))
    assert "«завтра», «в пятницу», «15 ноября») — это не время" in prompt
    assert "Никогда не подставляй час сам" in prompt
    assert "создавай task со временем и длительностью" in prompt
    assert "Поле description — это исходная фраза пользователя" in prompt
    # Живой тест: «напомни о вылете в 20:00» бот поставил ровно на 20:00 — слишком поздно.
    assert "время самого\n   события, а не напоминания" in prompt
    assert "за сколько напомнить" in prompt
    # Живой тест: на «днём» бот ответил «напомню завтра днём» — без часа, и человек не узнал 13:00.
    assert "Ответ «напомню завтра днём» без часа — ошибка" in prompt


# --- сценарии -----------------------------------------------------------------


async def test_create_reminder(make_agent, tz, pool):
    start = tomorrow_at(tz, 10)
    agent, llm = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "reminder",
                "name": "Позвонить Ивану",
                "description": "Напомни мне завтра в 10 утра позвонить Ивану",
                "start_at": start.isoformat(),
            },
        ),
        text("Напомню завтра в 10:00 позвонить Ивану."),
    )
    response = await agent.handle(USER, "Напомни мне завтра в 10 утра позвонить Ивану")

    assert response.status == AgentStatus.DONE
    assert response.reply_text == "Напомню завтра в 10:00 позвонить Ивану."
    assert [e.name for e in response.events] == ["Позвонить Ивану"]
    assert response.events[0].start_at == start

    row = await pool.fetchrow("SELECT user_id, type FROM events")
    assert dict(row) == {"user_id": USER, "type": "reminder"}

    # Во втором раунде LLM видит результат тула.
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "Позвонить Ивану" in tool_msg["content"]


async def test_model_cannot_choose_user_id(make_agent, pool):
    agent, _ = make_agent(
        tool_call("create_event", {"event_type": "task", "name": "Чужая", "user_id": "victim"}),
        text("Готово."),
    )
    await agent.handle(USER, "Создай задачу")
    assert await pool.fetchval("SELECT user_id FROM events") == USER


async def test_clarification_then_create(make_agent, tz, redis):
    start = tomorrow_at(tz, 15)
    agent, llm = make_agent(
        tool_call("ask_user", {"question": "Во сколько встреча и сколько она продлится?"}),
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с дизайнером",
                "start_at": start.isoformat(),
                "duration_minutes": 90,
            },
        ),
        text("Поставил встречу с дизайнером на завтра, с 15:00 до 16:30."),
    )

    first = await agent.handle(USER, "Поставь встречу с дизайнером на завтра")
    assert first.status == AgentStatus.NEED_CLARIFICATION
    assert first.reply_text == "Во сколько встреча и сколько она продлится?"

    second = await agent.handle(USER, "В три часа дня, на полтора часа")
    assert second.status == AgentStatus.DONE
    assert second.events[0].name == "Встреча с дизайнером"
    assert second.events[0].end_at == start + timedelta(minutes=90)

    # Второй запрос к LLM содержит первый обмен: так модель склеивает уточнение с просьбой.
    sent = [m["content"] for m in llm.requests[1]["messages"] if m["role"] in ("user", "assistant")]
    assert sent[:3] == [
        "Поставь встречу с дизайнером на завтра",
        "Во сколько встреча и сколько она продлится?",
        "В три часа дня, на полтора часа",
    ]

    history = await ConversationMemory(redis, 10, 60).load(USER)
    assert len(history) == 4
    assert "id=" in history[-1]["content"]  # скрытая ссылка на запись для «перенеси её»


async def test_tool_error_goes_back_to_llm(make_agent):
    agent, llm = make_agent(
        tool_call("create_event", {"event_type": "meeting", "name": "Встреча с Иваном"}),
        text("Во сколько встреча с Иваном?"),
    )
    response = await agent.handle(USER, "Встреча с Иваном")

    error = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert "нужно указать дату и время" in error["error"]
    assert response.status == AgentStatus.DONE
    assert response.events == []


async def test_meeting_duration_unknown_falls_back_to_hour(make_agent, tz, pool):
    """Модель забыла спросить длительность → тул отказал с подсказкой → модель спросила →
    «не знаю» → встреча на час. Проверяем, что подсказка тула доходит до модели и что
    до ответа пользователя в базе ничего не появилось."""
    start = tomorrow_at(tz, 15)
    meeting = {"event_type": "meeting", "name": "Встреча с Олей", "start_at": start.isoformat()}
    agent, llm = make_agent(
        tool_call("create_event", meeting),
        tool_call("ask_user", {"question": "Сколько продлится встреча?"}, call_id="call_2"),
        tool_call("create_event", {**meeting, "duration_minutes": 60}),
        text("Поставил встречу с Олей завтра в 15:00 на час — если нужно, поменяю."),
    )

    first = await agent.handle(USER, "Встреча с Олей завтра в 15")
    error = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert "длительность встречи" in error["error"]
    assert first.status == AgentStatus.NEED_CLARIFICATION
    assert first.reply_text == "Сколько продлится встреча?"
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    second = await agent.handle(USER, "Не знаю")
    assert second.status == AgentStatus.DONE
    assert second.events[0].duration_minutes == 60
    assert second.events[0].end_at == start + timedelta(hours=1)


async def test_list_today(make_agent, tz):
    start = tomorrow_at(tz, 9)
    day = start.replace(hour=0)
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {"event_type": "reminder", "name": "Зарядка", "start_at": start.isoformat()},
        ),
        text("Ок"),
        tool_call(
            "get_events",
            {"start_time": day.isoformat(), "end_time": (day + timedelta(days=1)).isoformat()},
        ),
        text("Завтра: 09:00 Зарядка"),
    )
    await agent.handle(USER, "Напомни завтра в 9 про зарядку")
    response = await agent.handle(USER, "Что у меня завтра?")
    assert [e.name for e in response.events] == ["Зарядка"]


async def test_update_event_reschedule(make_agent, tz, pool):
    start = tomorrow_at(tz, 11)
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {"event_type": "task", "name": "Созвон с клиентом", "start_at": start.isoformat()},
        ),
        text("Поставил на 11:00 завтра."),
    )
    created = await agent.handle(USER, "Созвон с клиентом завтра в 11")
    event_id = str(created.events[0].id)

    new_start = tomorrow_at(tz, 16)
    agent2, _ = make_agent(
        tool_call("update_event", {"event_id": event_id, "start_at": new_start.isoformat()}),
        text("Перенёс на 16:00."),
    )
    response = await agent2.handle(USER, "Перенеси созвон на 16:00")

    assert response.status == AgentStatus.DONE
    assert response.events[0].start_at == new_start
    assert (
        await pool.fetchval("SELECT start_at FROM events WHERE id = $1", UUID(event_id))
        == new_start
    )


async def test_delete_all_fits_in_step_limit_with_confirmation(make_agent, pool):
    """Регрессия живого теста: «удали все задачи» упиралось в лимит шагов («Запутался»)."""
    for i in range(10):
        await pool.execute(
            "INSERT INTO events (user_id, type, name, description) VALUES ($1, 'task', $2, '')",
            USER,
            f"Дело {i}",
        )
    kinds = {"event_types": ["task", "reminder"]}
    agent, llm = make_agent(
        tool_call("delete_events", kinds),
        tool_call("ask_user", {"question": "Удалить все записи (10 шт.)?"}, call_id="call_2"),
        tool_call("delete_events", {**kinds, "confirmed": True}),
        text("Удалил 10 записей."),
    )

    first = await agent.handle(USER, "Удали все мои задачи и напоминания")
    assert first.status == AgentStatus.NEED_CLARIFICATION
    assert (
        "будет удалено записей — 10"
        in json.loads(llm.requests[1]["messages"][-1]["content"])["error"]
    )
    assert await pool.fetchval("SELECT count(*) FROM events WHERE status = 'active'") == 10

    second = await agent.handle(USER, "Да")
    assert second.status == AgentStatus.DONE
    assert await pool.fetchval("SELECT count(*) FROM events WHERE status = 'active'") == 0


async def test_reminder_with_duration_becomes_task(make_agent, tz, pool):
    """Названную длительность нельзя терять: тул отказывает, модель пересоздаёт как дело."""
    start = tomorrow_at(tz, 10)
    args = {"name": "Самолёт в Нижний Новгород", "start_at": start.isoformat()}
    agent, llm = make_agent(
        tool_call("create_event", {**args, "event_type": "reminder", "duration_minutes": 120}),
        tool_call("create_event", {**args, "event_type": "task", "duration_minutes": 120}),
        text("Поставил как дело завтра с 10:00 до 12:00 — у напоминаний длительности не бывает."),
    )
    response = await agent.handle(USER, "Напомни, завтра в 10 самолёт, на два часа")

    error = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert "создавай task" in error["error"]
    assert response.status == AgentStatus.DONE
    assert response.events[0].duration_minutes == 120
    assert response.events[0].end_at == start + timedelta(hours=2)
    assert await pool.fetchval("SELECT type FROM events") == "task"


async def test_conflict_reaches_llm_and_can_be_rescheduled(make_agent, tz, pool):
    """Пересечение: запись сохраняется, модель видит conflicts, а пересекающаяся встреча
    попадает в скрытую сводку — «перенеси ту встречу» работает следующим сообщением."""
    existing, _ = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Созвон с Иваном",
                "start_at": tomorrow_at(tz, 15).isoformat(),
                "duration_minutes": 60,
            },
        ),
        text("Поставил созвон с Иваном завтра в 15:00."),
    )
    first = await existing.handle(USER, "Созвон с Иваном завтра в 15")
    ivan_id = str(first.events[0].id)

    agent, llm = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с дизайнером",
                "start_at": tomorrow_at(tz, 15).replace(minute=30).isoformat(),
                "duration_minutes": 30,
            },
        ),
        text("Создал. Учтите: в это время уже есть созвон с Иваном с 15:00 до 16:00."),
    )
    response = await agent.handle(USER, "Встреча с дизайнером завтра в 15:30 на полчаса")

    tool_message = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert [c["name"] for c in tool_message["conflicts"]] == ["Созвон с Иваном"]
    assert "user_id" not in tool_message["conflicts"][0]  # шум вырезан и во вложенных записях
    assert {e.name for e in response.events} == {"Встреча с дизайнером", "Созвон с Иваном"}
    assert await pool.fetchval("SELECT count(*) FROM events WHERE status = 'active'") == 2

    history = await agent._memory.load(USER)
    assert ivan_id in history[-1]["content"]  # id пересекающейся встречи есть в сводке
    assert "–16:00" in history[-1]["content"]  # и время окончания


async def test_complete_and_delete_event(make_agent, tz, pool):
    start = tomorrow_at(tz, 9)
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {"event_type": "task", "name": "Выгрузить отчёт", "start_at": start.isoformat()},
        ),
        text("Создал."),
    )
    created = await agent.handle(USER, "Выгрузить отчёт завтра в 9")
    event_id = str(created.events[0].id)

    agent2, _ = make_agent(
        tool_call("complete_event", {"event_id": event_id}), text("Отметил выполненным.")
    )
    await agent2.handle(USER, "Отчёт готов")
    assert await pool.fetchval("SELECT status FROM events WHERE id = $1", UUID(event_id)) == "done"

    agent3, _ = make_agent(tool_call("delete_event", {"event_id": event_id}), text("Удалил."))
    await agent3.handle(USER, "Удали эту задачу")
    assert (
        await pool.fetchval("SELECT status FROM events WHERE id = $1", UUID(event_id))
        == "cancelled"
    )


async def test_llm_down_returns_error_and_keeps_history_clean(make_agent, redis):
    agent, _ = make_agent(httpx.ConnectError("gpt2giga недоступен"))
    response = await agent.handle(USER, "Что у меня сегодня?")
    assert response.status == AgentStatus.ERROR
    assert await redis.exists(ConversationMemory.key(USER)) == 0


async def test_broken_json_arguments(make_agent):
    agent, llm = make_agent(
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_events", "arguments": "{oops"},
                    }
                ],
            }
        ),
        text("Не понял период, уточните."),
    )
    await agent.handle(USER, "Что у меня?")
    assert "корректным JSON" in llm.requests[1]["messages"][-1]["content"]


async def test_step_limit(make_agent):
    loop = [tool_call("search_events", {"query": "x"}, f"c{i}") for i in range(6)]
    agent, _ = make_agent(*loop)
    response = await agent.handle(USER, "Найди")
    assert response.status == AgentStatus.ERROR


async def test_reset_and_history_trim(redis):
    memory = ConversationMemory(redis, max_messages=4, ttl_seconds=100)
    for i in range(3):
        await memory.append(
            USER, {"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}
        )
    assert [m["content"] for m in await memory.load(USER)] == ["q1", "a1", "q2", "a2"]
    assert 0 < await redis.ttl(ConversationMemory.key(USER)) <= 100

    await memory.reset(USER)
    assert await memory.load(USER) == []


async def test_http_endpoint(make_agent):
    from agent.main import app, get_agent

    agent, _ = make_agent(text("Привет! Я помогаю с задачами и встречами."))
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            ok = await client.post("/v1/agent/message", json={"user_id": USER, "text": "Привет"})
            bad = await client.post("/v1/agent/message", json={"user_id": USER, "text": ""})
            reset = await client.post("/v1/agent/reset", json={"user_id": USER})
    finally:
        app.dependency_overrides.clear()

    assert ok.status_code == 200
    assert ok.json() == {
        "reply_text": "Привет! Я помогаю с задачами и встречами.",
        "status": "done",
        "events": [],
    }
    assert bad.status_code == 422
    assert reset.status_code == 204
