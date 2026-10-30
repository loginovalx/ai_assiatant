"""Голосовая команда не меняет данные без подтверждения.

Распознавание ошибается в именах, датах и числах, поэтому голосовое изменение сначала
показывается пользователю как черновик и выполняется только после «Подтвердить».
Текстовые команды выполняются сразу — в них опечатку видно самому пользователю.
"""

from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from common.schemas import AgentStatus, DraftAction, EventType
from mcp_server.server import DRAFT_EVENT_ID, mcp
from tests.scripted_llm import text, tool_call

USER = "user-1"


def tomorrow_at(tz, hour: int) -> datetime:
    return (datetime.now(tz) + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )


def reminder_args(tz, hour: int = 11, name: str = "Записаться к врачу") -> dict:
    return {
        "event_type": "reminder",
        "name": name,
        "description": f"Напомни завтра в {hour} записаться к врачу",
        "start_at": tomorrow_at(tz, hour).isoformat(),
    }


# --- уровень MCP: dry_run ничего не пишет ------------------------------------


@pytest.fixture
async def client(pool):
    async with Client(mcp) as c:
        yield c


async def test_dry_run_validates_and_finds_conflicts_without_saving(client, pool, tz):
    await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "meeting",
            "name": "Созвон с Иваном",
            "description": "Созвон с Иваном завтра в 11:00",
            "start_at": tomorrow_at(tz, 11).isoformat(),
            "duration_minutes": 60,
        },
    )

    planned = await client.call_tool(
        "create_event", {"user_id": USER, "dry_run": True, **reminder_args(tz)}
    )
    data = planned.structured_content
    assert data["name"] == "Записаться к врачу"
    assert data["id"] == str(DRAFT_EVENT_ID)  # записи ещё нет
    assert await pool.fetchval("SELECT count(*) FROM events") == 1  # только встреча


async def test_dry_run_still_rejects_invalid_commands(client, pool, tz):
    """Проверки работают и вхолостую: пользователю не покажут то, что потом не сохранится."""
    result = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "dry_run": True,
            "event_type": "meeting",
            "name": "Встреча",
            "description": "Поставь встречу завтра",  # часа нет
            "start_at": tomorrow_at(tz, 11).isoformat(),
            "duration_minutes": 60,
        },
        raise_on_error=False,
    )
    assert result.is_error and "нет времени, только день" in result.content[0].text


async def test_dry_run_delete_keeps_the_event(client, pool, tz):
    created = await client.call_tool(
        "create_event",
        {
            "user_id": USER,
            "event_type": "reminder",
            "name": "Позвонить",
            "description": "Напомни завтра в 11 позвонить",
            "start_at": tomorrow_at(tz, 11).isoformat(),
        },
    )
    event_id = created.structured_content["id"]

    shown = await client.call_tool(
        "delete_event", {"user_id": USER, "event_id": event_id, "dry_run": True}
    )
    assert shown.structured_content["name"] == "Позвонить"
    assert (
        await pool.fetchval("SELECT status FROM events WHERE id = $1::uuid", event_id) == "active"
    )


# --- уровень агента: черновик, подтверждение, правка, отмена ------------------


async def test_voice_creates_draft_not_record(make_agent, tz, pool):
    agent, _ = make_agent(
        tool_call("create_event", reminder_args(tz)),
        text("Напомню вам записаться к врачу завтра в 11:00."),
    )
    response = await agent.handle(
        USER, "Напомни завтра в 11 записаться к врачу", confirm_changes=True
    )

    assert response.status == AgentStatus.NEEDS_CONFIRMATION
    assert response.reply_text == "Напомню вам записаться к врачу завтра в 11:00."
    assert response.draft is not None
    assert response.draft.tool == "create_event"
    assert response.draft.status == "pending_confirmation"
    assert response.events == []  # наружу черновик записью не выдаём
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_confirmation_writes_to_database(make_agent, tz, pool):
    agent, _ = make_agent(
        tool_call("create_event", reminder_args(tz)),
        text("Напомню вам записаться к врачу завтра в 11:00."),
    )
    draft = (
        await agent.handle(USER, "Напомни завтра в 11 записаться к врачу", confirm_changes=True)
    ).draft

    confirmed = await agent.execute_draft(USER, draft)

    assert confirmed.status == AgentStatus.DONE
    assert confirmed.reply_text.startswith("Готово.")
    assert confirmed.events[0].name == "Записаться к врачу"
    assert confirmed.events[0].type == EventType.REMINDER
    assert await pool.fetchval("SELECT count(*) FROM events WHERE status = 'active'") == 1


async def test_draft_can_be_corrected_before_saving(make_agent, tz, pool):
    """Сценарий из ТЗ: распознало 10:00, пользователь пишет «давай в 12:30»."""
    agent, _ = make_agent(
        tool_call("create_event", reminder_args(tz, hour=10)),
        text("Напомню вам записаться к врачу завтра в 10:00."),
        tool_call("create_event", reminder_args(tz, hour=12)),
        text("Напомню вам записаться к врачу завтра в 12:30."),
    )

    first = await agent.handle(USER, "Напомни завтра в 11 записаться к врачу", confirm_changes=True)
    assert "10:00" in first.reply_text

    # Правка идёт тем же путём, что и голос: снова черновик, в базе по-прежнему пусто.
    second = await agent.handle(USER, "Давай в 12:30", confirm_changes=True)
    assert second.status == AgentStatus.NEEDS_CONFIRMATION
    assert "12:30" in second.reply_text
    assert await pool.fetchval("SELECT count(*) FROM events") == 0

    await agent.execute_draft(USER, second.draft)
    saved = await pool.fetchrow("SELECT name, start_at FROM events")
    assert saved["name"] == "Записаться к врачу"
    assert saved["start_at"] == tomorrow_at(tz, 12)


async def test_questions_by_voice_need_no_confirmation(make_agent, tz, pool):
    """«Что у меня завтра» ничего не меняет — подтверждать нечего, отвечаем сразу."""
    day = tomorrow_at(tz, 0)
    agent, _ = make_agent(
        tool_call(
            "get_events",
            {"start_time": day.isoformat(), "end_time": (day + timedelta(days=1)).isoformat()},
        ),
        text("Завтра ничего не запланировано."),
    )
    response = await agent.handle(USER, "Что у меня завтра?", confirm_changes=True)

    assert response.status == AgentStatus.DONE
    assert response.draft is None


async def test_text_commands_still_execute_immediately(make_agent, tz, pool):
    agent, _ = make_agent(
        tool_call("create_event", reminder_args(tz)),
        text("Напомню вам записаться к врачу завтра в 11:00."),
    )
    response = await agent.handle(USER, "Напомни завтра в 11 записаться к врачу")

    assert response.status == AgentStatus.DONE
    assert await pool.fetchval("SELECT count(*) FROM events") == 1


async def test_stale_draft_fails_cleanly(make_agent, tz, pool):
    """Запись удалили, пока черновик висел: пользователь получает понятный ответ, не 500."""
    agent, _ = make_agent()
    draft = DraftAction(
        tool="delete_event",
        arguments={"event_id": "00000000-0000-0000-0000-00000000dead"},
        preview="Удалю встречу с Иваном.",
    )

    response = await agent.execute_draft(USER, draft)
    assert response.status == AgentStatus.ERROR
    assert "Не получилось выполнить" in response.reply_text
