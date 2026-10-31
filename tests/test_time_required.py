"""Час не выдумывается: если пользователь назвал только день, тул требует спросить время.

Регрессия живого теста: «Напомни позвонить Марине» → «Когда напомнить?» → «Завтра»
→ бот молча ставил 14:00. Правило было в промпте, но GigaChat его игнорировал,
поэтому проверка продублирована в коде.
"""

from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from common.timeutils import mentions_time
from mcp_server.server import mcp
from tests.scripted_llm import text, tool_call

USER = "user-1"


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


def tomorrow(tz, hour: int = 14):
    return (datetime.now(tz) + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни позвонить Марине",
        "Напомни позвонить Марине. Завтра",
        "Поставь встречу с Марией на пятницу",
        "Напомни про отчёт 15 января",
        "Созвон в понедельник",
        "завтра утром",
        "сегодня вечером",
        "в обед",
        "",
    ],
)
def test_day_without_hour_is_not_a_time(phrase):
    assert not mentions_time(phrase)


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни завтра в 10 позвонить Ивану",
        "в пятницу в 14:00",
        "через час",
        "через полчаса",
        "через полминуты",
        "через четверть часа",
        "в четверть седьмого",
        "в три часа дня",
        "в полдень",
        "через 20 минут",
        "15 января в 10",
    ],
)
def test_named_hour_is_recognised(phrase):
    assert mentions_time(phrase)


@pytest.mark.parametrize("event_type", ["reminder", "meeting"])
async def test_tool_refuses_to_invent_hour(client, pool, tz, event_type):
    args = {
        "user_id": USER,
        "event_type": event_type,
        "name": "Позвонить Марине",
        "description": "Напомни позвонить Марине. Завтра",
        "start_at": tomorrow(tz).isoformat(),
    }
    if event_type == "meeting":
        args["duration_minutes"] = 60  # у напоминания длительности не бывает

    result = await client.call_tool("create_event", args, raise_on_error=False)
    assert result.is_error
    assert "нет времени, только день" in result.content[0].text
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    # Пользователь ответил «в 10» — его слова попадают в description, запись создаётся.
    answered = {**args, "description": "Напомни позвонить Марине. Завтра в 10"}
    created = await client.call_tool("create_event", answered)
    assert created.structured_content["name"] == "Позвонить Марине"


async def test_task_deadline_without_hour_still_allowed(client, tz):
    """У дела срок без часа законен: «сдать отчёт до пятницы» — это 18:00 по промпту."""
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "task",
            "name": "Сдать отчёт",
            "description": "Сдать отчёт до пятницы",
            "start_at": (
                datetime.now(tz) + timedelta(days=(4 - datetime.now(tz).weekday()) % 7 or 7)
            )
            .replace(hour=18, minute=0, second=0, microsecond=0)
            .isoformat(),
        },
    )
    assert created.structured_content["name"] == "Сдать отчёт"


async def test_part_of_day_requires_exact_time(client, tz, pool):
    """«Завтра вечером» требует уточнения; произвольные 19:00 не сохраняются."""
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "reminder",
            "name": "Позвонить Марине",
            "description": "Напомни завтра вечером позвонить Марине",
            "start_at": tomorrow(tz, 19).isoformat(),
        },
        raise_on_error=False,
    )
    assert created.is_error
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_named_weekday_mismatch_is_not_saved(client, tz, pool):
    now = datetime.now(tz)
    monday = (now + timedelta(days=(0 - now.weekday()) % 7 or 7)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    args = {
        "user_id": USER,
        "event_type": "reminder",
        "name": "Оплатить интернет",
        "user_text": "Напомни в понедельник в девять утра оплатить интернет",
        "start_at": (monday + timedelta(days=1)).isoformat(),
    }
    rejected = await client.call_tool("create_event", args, raise_on_error=False)
    assert rejected.is_error
    assert await pool.fetchval("SELECT count(*) FROM events") == 0
    accepted = await client.call_tool("create_event", {**args, "start_at": monday.isoformat()})
    assert accepted.structured_content["start_at"] == monday.isoformat()


async def test_full_clarification_flow_through_agent(make_agent, tz, pool):
    """Сквозной сценарий из жалобы: «Напомни позвонить Марине» → «Завтра» → «В 10»."""
    start = tomorrow(tz, 10)
    base = {"event_type": "reminder", "name": "Позвонить Марине", "start_at": start.isoformat()}
    agent, _ = make_agent(
        tool_call("ask_user", {"question": "Когда вам напомнить?"}),
        # Пользователь ответил «Завтра» — модель пробует создать, тул не даёт выдумать час.
        tool_call("create_event", {**base, "description": "Напомни позвонить Марине. Завтра"}),
        tool_call("ask_user", {"question": "Во сколько напомнить?"}, call_id="call_2"),
        tool_call("create_event", {**base, "description": "Напомни позвонить Марине. Завтра в 10"}),
        text("Напомню завтра, 31 октября, в 10:00."),
    )

    assert (await agent.handle(USER, "Напомни позвонить Марине")).reply_text == (
        "Когда вам напомнить?"
    )
    second = await agent.handle(USER, "Завтра")
    assert second.reply_text == "Во сколько напомнить?"  # час не выдуман
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    third = await agent.handle(USER, "В 10")
    assert third.events[0].start_at == start


async def test_user_text_with_time_passes_despite_bare_description(client, tz):
    """Модель кладёт суть в description, а время — в start_at. Время считаем названным,
    если оно есть в дословной реплике пользователя (user_text), а не в description."""
    args = {
        "user_id": USER,
        "event_type": "reminder",
        "name": "Позвонить Ивану",
        "description": "позвоните Ивану",
        "start_at": tomorrow(tz, hour=10).isoformat(),
        "user_text": "Напомни завтра в 10 утра позвонить Ивану",
    }
    result = await client.call_tool("create_event", args, raise_on_error=False)
    assert not result.is_error, result.content[0].text


async def test_user_text_without_time_still_refuses_invented_hour(client, tz):
    """А вот если в реплике пользователя времени не было — выдуманный час отвергается."""
    args = {
        "user_id": USER,
        "event_type": "reminder",
        "name": "Позвонить Марине",
        "description": "Напомни позвонить Марине",
        "start_at": tomorrow(tz, hour=10).isoformat(),
        "user_text": "Напомни завтра позвонить Марине",
    }
    result = await client.call_tool("create_event", args, raise_on_error=False)
    assert result.is_error
    assert "нет времени, только день" in result.content[0].text
