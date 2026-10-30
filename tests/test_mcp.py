"""MCP-тулы так, как их увидит агент: через MCP-клиент, а не прямым вызовом функций."""

from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from mcp_server.server import mcp

USER = "user-1"


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


def iso(dt: datetime) -> str:
    return dt.isoformat()


async def test_tools_are_listed(client):
    names = {t.name for t in await client.list_tools()}
    assert names == {
        "create_event",
        "get_events",
        "search_events",
        "update_event",
        "complete_event",
        "delete_event",
        "delete_events",
    }


async def test_bulk_delete_asks_confirmation_then_deletes_only_mine(client, pool, tz):
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    for i in range(8):  # больше лимита шагов агента: цикл по одной записи не пролез бы
        await client.call_tool(
            "create_event", {"user_id": USER, "event_type": "task", "name": f"Дело {i}"}
        )
    await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "reminder", "name": "Зарядка", "start_at": iso(start)},
    )
    await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Встреча",
            "start_at": iso(start),
            "duration_minutes": 30,
        },
    )
    await client.call_tool(
        "create_event", {"user_id": "someone-else", "event_type": "task", "name": "Чужое"}
    )
    args = {"user_id": USER, "event_types": ["task", "reminder"]}

    # Без подтверждения — ничего не удалено, а в тексте число для вопроса пользователю.
    asked = await client.call_tool("delete_events", args, raise_on_error=False)
    assert asked.is_error and "будет удалено записей — 9" in asked.content[0].text
    assert await pool.fetchval("SELECT count(*) FROM events WHERE status = 'cancelled'") == 0

    done = await client.call_tool("delete_events", {**args, "confirmed": True})
    assert len(done.structured_content["result"]) == 9
    left = await pool.fetch("SELECT name FROM events WHERE status <> 'cancelled' ORDER BY name")
    assert [r["name"] for r in left] == ["Встреча", "Чужое"]  # встреча и чужая задача целы


async def test_bulk_delete_with_nothing_to_delete(client):
    result = await client.call_tool(
        "delete_events", {"user_id": USER, "event_types": ["task"]}, raise_on_error=False
    )
    assert result.is_error and "удалять нечего" in result.content[0].text


async def test_create_and_get_today(client, tz):
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Встреча с командой",
            "description": "Добавь встречу с командой через два часа",
            "start_at": iso(start),
            "duration_minutes": 60,
        },
    )
    assert created.structured_content["name"] == "Встреча с командой"
    assert created.structured_content["weekday"]

    day_start = start.replace(hour=0)
    result = await client.call_tool(
        "get_events",
        {
            "user_id": USER,
            "start_time": iso(day_start),
            "end_time": iso(day_start + timedelta(days=2)),
        },
    )
    assert [e["name"] for e in result.structured_content["result"]] == ["Встреча с командой"]


async def test_meeting_without_time_is_rejected_with_reason(client):
    result = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "meeting", "name": "Встреча с Иваном"},
        raise_on_error=False,
    )
    assert result.is_error
    assert "нужно указать дату и время" in result.content[0].text


async def test_meeting_without_duration_asks_instead_of_guessing(client, pool, tz):
    """Время есть, длительности нет: не сохраняем с часом молча, а просим агента спросить."""
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    args = {
        "user_id": USER,
        "event_type": "meeting",
        "name": "Встреча с Олей",
        "start_at": iso(start),
    }

    result = await client.call_tool("create_event", args, raise_on_error=False)
    assert result.is_error
    assert "длительность встречи" in result.content[0].text
    assert "duration_minutes=60" in result.content[0].text  # подсказка на случай «не знаю»
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    # Пользователь ответил «не знаю» — агент передаёт час, встреча создаётся.
    created = await client.call_tool("create_event", {**args, "duration_minutes": 60})
    assert created.structured_content["end_at"] == iso(start + timedelta(hours=1))


async def test_reminder_with_duration_points_to_task(client, pool, tz):
    """«Напомни про самолёт завтра в 10, на два часа» — это не момент времени.
    Тул отказывает и подсказывает модели создать task, чтобы названные 2 часа не пропали."""
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    args = {
        "user_id": USER,
        "event_type": "reminder",
        "name": "Самолёт в Нижний Новгород",
        "start_at": iso(start),
        "duration_minutes": 120,
    }

    result = await client.call_tool("create_event", args, raise_on_error=False)
    assert result.is_error
    assert "создавай task" in result.content[0].text
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    created = await client.call_tool("create_event", {**args, "event_type": "task"})
    assert created.structured_content["duration_minutes"] == 120
    assert created.structured_content["end_at"] == iso(start + timedelta(hours=2))


async def test_reminder_without_duration_has_no_end(client, tz):
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "reminder", "name": "Зарядка", "start_at": iso(start)},
    )
    # По этим двум полям модель обязана отвечать «длительности нет», а не пересказывать фразу.
    assert created.structured_content["duration_minutes"] is None
    assert created.structured_content["end_at"] is None


async def test_duration_still_optional_for_tasks(client, tz):
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Отчёт", "start_at": iso(start)},
    )
    assert created.structured_content["duration_minutes"] is None
    assert created.structured_content["end_at"] is None


async def test_time_in_past_is_rejected(client, tz):
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "reminder",
            "name": "Позвонить",
            "start_at": iso(datetime.now(tz) - timedelta(days=1)),
        },
        raise_on_error=False,
    )
    assert result.is_error and "уже прошло" in result.content[0].text


async def test_task_without_date_and_search(client):
    await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Подписать договор"},
    )
    result = await client.call_tool("search_events", {"user_id": USER, "query": "договор"})
    hits = result.structured_content["result"]
    assert hits[0]["name"] == "Подписать договор"
    assert hits[0]["start_at"] is None


async def test_update_event_reschedules_without_losing_name(client, tz):
    start = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Созвон", "start_at": iso(start)},
    )
    new_start = start + timedelta(hours=3)
    updated = await client.call_tool(
        "update_event",
        {"user_id": USER, "event_id": created.structured_content["id"], "start_at": iso(new_start)},
    )
    assert updated.structured_content["name"] == "Созвон"
    assert updated.structured_content["start_at"] == iso(new_start)


async def test_update_event_without_changes_is_rejected(client, tz):
    start = datetime.now(tz) + timedelta(hours=2)
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Задача", "start_at": iso(start)},
    )
    result = await client.call_tool(
        "update_event",
        {"user_id": USER, "event_id": created.structured_content["id"]},
        raise_on_error=False,
    )
    assert result.is_error
    assert "менять" in result.content[0].text


async def test_complete_event_marks_done(client, tz):
    start = datetime.now(tz) + timedelta(hours=1)
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Отчёт", "start_at": iso(start)},
    )
    completed = await client.call_tool(
        "complete_event", {"user_id": USER, "event_id": created.structured_content["id"]}
    )
    assert completed.structured_content["status"] == "done"


async def test_delete_event_hides_it_from_search(client):
    created = await client.call_tool(
        "create_event",
        {"user_id": USER, "event_type": "task", "name": "Черновик договора"},
    )
    deleted = await client.call_tool(
        "delete_event", {"user_id": USER, "event_id": created.structured_content["id"]}
    )
    assert deleted.structured_content["status"] == "cancelled"

    result = await client.call_tool("search_events", {"user_id": USER, "query": "договор"})
    assert result.structured_content["result"] == []


async def test_update_unknown_event_is_rejected(client, tz):
    result = await client.call_tool(
        "update_event",
        {
            "user_id": USER,
            "event_id": "00000000-0000-0000-0000-000000000000",
            "name": "Новое имя",
        },
        raise_on_error=False,
    )
    assert result.is_error and "не найдено" in result.content[0].text


async def test_bad_period_is_rejected(client, tz):
    now = datetime.now(tz)
    result = await client.call_tool(
        "get_events",
        {"user_id": USER, "start_time": iso(now), "end_time": iso(now)},
        raise_on_error=False,
    )
    assert result.is_error and "раньше" in result.content[0].text
