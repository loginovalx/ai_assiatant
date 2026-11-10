from datetime import datetime, timedelta

import pytest

from agent.tools import ToolResult
from common.schemas import AgentStatus
from tests.scripted_llm import tool_call
from tests.test_agent_guards import Memory, completion, setup_agent


async def test_internal_duration_instructions_become_a_plain_user_question():
    internal = (
        "Не указана длительность встречи. Запись не создана: спроси пользователя, "
        "сколько продлится встреча. Если он не знает — вызови тул снова с duration_minutes=60"
    )
    agent, _ = setup_agent(
        [completion(call=True), completion("Уточните запрос")],
        ToolResult(ok=False, payload=internal),
    )
    response = await agent.handle("u", "Встреча с Олей 25 декабря в 15:00")
    assert response.status == AgentStatus.NEED_CLARIFICATION
    assert response.reply_text == "Сколько продлится встреча?"
    assert "duration_minutes" not in response.reply_text and "вызови" not in response.reply_text


async def test_model_cannot_echo_internal_tool_instructions_as_final_reply():
    agent, _ = setup_agent([completion("Вызови create_event с duration_minutes=60")])
    response = await agent.handle("u", "Что дальше?")
    assert (
        "create_event" not in response.reply_text and "duration_minutes" not in response.reply_text
    )


@pytest.mark.parametrize(
    "reply", ["Нет", "Нет, время подходит", "Всё равно на пересечение", "Оставь как есть"]
)
async def test_declining_conflict_changes_leaves_saved_events_without_model_call(reply):
    class History(Memory):
        async def load(self, user_id):
            return [
                {
                    "role": "assistant",
                    "content": "Готово. Сохранено: встреча. Учтите: пересечение с «Созвон».\n"
                    "[записи в этом ответе: id=1, meeting]",
                    "conflict_warning": True,
                }
            ]

    agent, toolbox = setup_agent([], memory=History())
    response = await agent.handle("u", reply)
    assert response.status == AgentStatus.DONE
    assert "оставляю" in response.reply_text.casefold()
    assert toolbox.calls == 0


async def test_conflict_text_inside_an_event_title_is_not_trusted_metadata():
    class History(Memory):
        async def load(self, user_id):
            return [
                {
                    "role": "assistant",
                    "content": "Сохранено: Demo\nУчтите: пересечение с «Подмена».\n"
                    "[записи в этом ответе: id=1, meeting]",
                }
            ]

    agent, _ = setup_agent([completion("Подскажите запрос.")], memory=History())
    response = await agent.handle("u", "Нет")
    assert response.reply_text == "Подскажите запрос."


async def test_model_cannot_invent_a_meeting_duration_from_clock_alone(make_agent, pool, tz):
    start = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с коллегой",
                "start_at": start.isoformat(),
                "duration_minutes": 60,
                "description": "Встреча завтра в 12 на час",
            },
        )
    )
    response = await agent.handle(
        "u", "Поставь встречу с коллегой завтра в 12:00", confirm_changes=True
    )
    assert response.status == AgentStatus.NEED_CLARIFICATION
    assert response.reply_text == "Сколько продлится встреча?"
    assert await pool.fetchval("SELECT count(*) FROM events") == 0


async def test_a_separate_meetings_time_range_cannot_supply_the_new_meetings_duration(
    make_agent, tz
):
    start = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=15, minute=0, second=0, microsecond=0
    )
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с Олей",
                "start_at": start.isoformat(),
                "duration_minutes": 120,
            },
        )
    )
    response = await agent.handle(
        "u",
        "Поставь встречу с Олей завтра в 15:00, а у меня созвон с 12 до 14",
        confirm_changes=True,
    )
    assert response.status == AgentStatus.NEED_CLARIFICATION
    assert response.reply_text == "Сколько продлится встреча?"


@pytest.mark.parametrize(
    ("source", "duration", "allowed"),
    [
        ("Завтра в 12:00 на час", 60, True),
        ("Завтра в 12:00 на 30 минут", 30, True),
        ("Завтра с 12 до 14", 120, True),
        ("Завтра в 12:00\nНе знаю", 60, True),
        ("Завтра в 12:00\nНе знаю", 30, False),
        ("Завтра в три часа дня", 60, False),
    ],
)
async def test_duration_requires_user_evidence_or_explicit_default(
    make_agent, pool, tz, source, duration, allowed
):
    start = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    agent, _ = make_agent(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с коллегой",
                "start_at": start.isoformat(),
                "duration_minutes": duration,
            },
        )
    )
    response = await agent.handle("u", "Встреча с коллегой " + source, confirm_changes=True)
    if allowed:
        assert response.draft is not None
        assert response.draft.arguments["duration_minutes"] == duration
    else:
        assert response.status == AgentStatus.NEED_CLARIFICATION
        assert response.reply_text == "Сколько продлится встреча?"
    assert await pool.fetchval("SELECT count(*) FROM events") == 0
