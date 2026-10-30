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
        "завтра утром",
        "сегодня вечером",
        "через час",
        "в три часа дня",
        "в полдень",
        "в обед",
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
            "start_at": tomorrow(tz, 18).isoformat(),
        },
    )
    assert created.structured_content["name"] == "Сдать отчёт"


async def test_part_of_day_is_enough(client, tz):
    """«Завтра вечером» — час модель вправе развернуть сама (19:00) и сказать об этом."""
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "reminder",
            "name": "Позвонить Марине",
            "description": "Напомни завтра вечером позвонить Марине",
            "start_at": tomorrow(tz, 19).isoformat(),
        },
    )
    assert created.structured_content["start_at"].endswith("19:00:00+03:00")


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
