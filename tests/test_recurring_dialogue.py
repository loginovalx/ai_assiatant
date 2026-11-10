from datetime import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from bot.recurring import RecurringReminders
from common.reminder_series import parse_clock, parse_command
from tests.test_time_choice_buttons import FakeMax, MemoryRedis, all_buttons


@pytest.mark.parametrize(
    "command",
    [
        "Напоминай мне каждый вторник в 10 поливать цветы",
        "Напоминай мне каждый вторник в десять утра поливать цветы",
        "Напоминай мне каждый вторник в 10:00 поливать цветы",
        "Напоминай мне поливать цветы каждый вторник в 10:00",
        "Каждый вторник в десять напоминай мне поливать цветы",
        "Напоминай мне каждый вторник поливать цветы в 10:00",
    ],
)
def test_natural_repeat_is_understood_with_all_required_fields(command):
    rule = parse_command(command)
    assert rule.weekday == 1 and rule.clock == time(10)
    assert rule.name == "поливать цветы"


@pytest.mark.parametrize("clock", ["10", "в 10:00", "в десять утра", "10.00", "10 00"])
def test_clock_answers_are_accepted(clock):
    assert parse_clock(clock) == time(10)


async def test_yes_stays_with_repeat_preview_and_button_saves_once(tz):
    redis, api, repo = MemoryRedis(), FakeMax(), AsyncMock()
    repeat = RecurringReminders(repo, redis, api, tz)
    assert await repeat.message("777", "Напоминай каждый вторник в 10:00 поливать цветы", uuid4())
    draft = await repeat._load("777")
    assert await repeat.message("777", "Да", uuid4(), is_voice=True)
    assert await repeat._load("777") is not None
    assert "Каждый вторник в 10:00" in api.sent[-1]["text"]
    button = next(b for b in all_buttons(api.sent[-1]) if b["text"] == "Сохранить")
    await repeat.callback("777", "save1", button["payload"])
    await repeat.callback("777", "save2", button["payload"])
    assert repo.save_reminder_series.await_count == 1
    assert repo.save_reminder_series.await_args.args[1] == draft.series_id


async def test_typed_yes_saves_repeat_instead_of_asking_again(tz):
    api, repo = FakeMax(), AsyncMock()
    repeat = RecurringReminders(repo, MemoryRedis(), api, tz)
    await repeat.message("777", "Напоминай каждый вторник в 10:00 поливать цветы", uuid4())
    assert await repeat.message("777", "Да", uuid4())
    assert await repeat._load("777") is None
    repo.save_reminder_series.assert_awaited_once()
    assert "сохранено" in api.sent[-1]["text"]
    assert api.answered == []


async def test_missing_clock_answer_completes_original_repeating_request(tz):
    redis, api = MemoryRedis(), FakeMax()
    repeat = RecurringReminders(AsyncMock(), redis, api, tz)
    assert await repeat.message("777", "Напоминай поливать цветы каждый вторник", uuid4())
    assert await repeat.message("777", "В десять утра", uuid4())
    draft = await repeat._load("777")
    assert draft is not None
    assert draft.rule.name == "поливать цветы"
    assert draft.rule.weekday == 1 and draft.rule.clock == time(10)


async def test_invalid_clock_can_be_corrected_without_repeating_the_request(tz):
    repeat = RecurringReminders(AsyncMock(), MemoryRedis(), FakeMax(), tz)
    await repeat.message("777", "Напоминай каждый вторник в 29:00 поливать цветы", uuid4())
    assert await repeat.message("777", "В 10:00", uuid4())
    draft = await repeat._load("777")
    assert draft is not None and draft.rule.clock == time(10)
    assert draft.rule.name == "поливать цветы"


async def test_cancelling_time_clarification_drops_the_repeat_request(tz):
    repeat = RecurringReminders(AsyncMock(), MemoryRedis(), FakeMax(), tz)
    assert await repeat.message("777", "Напоминай поливать цветы каждый вторник", uuid4())
    assert await repeat.message("777", "Отмена", uuid4())
    assert not await repeat.message("777", "В 10:00", uuid4())


@pytest.mark.parametrize("voice", [False, True])
async def test_real_bot_repeat_flow_never_calls_agent_or_asks_for_date(repo, pool, redis, voice):
    from bot.handler import BotHandler
    from bot.state import StateStore

    api, agent = FakeMax(), AsyncMock()
    bot = BotHandler(api, None, agent, StateStore(redis), redis, repo)
    phrase = "Напоминай мне каждый вторник в десять утра поливать цветы"
    body = {"mid": uuid4().hex}
    if voice:
        body["attachments"] = [{"type": "audio", "payload": {"transcription": phrase}}]
    else:
        body["text"] = phrase
    await bot.handle_update(
        {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 777},
                "recipient": {"chat_type": "dialog"},
                "body": body,
            },
        }
    )
    assert "Каждый вторник в 10:00" in api.sent[-1]["text"]
    assert "поливать цветы" in api.sent[-1]["text"]
    payload = next(b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == "Сохранить")
    update = {
        "update_type": "message_callback",
        "callback": {
            "user": {"user_id": 777},
            "callback_id": uuid4().hex,
            "payload": payload,
        },
    }
    await bot.handle_update(update)
    await bot.handle_update(update)
    row = await pool.fetchrow("SELECT name,weekday,clock FROM reminder_series WHERE user_id='777'")
    assert row["weekday"] == 1 and row["clock"] == time(10)
    assert row["name"] == "поливать цветы"
    assert await pool.fetchval("SELECT count(*) FROM reminder_series WHERE user_id='777'") == 1
    agent.ask.assert_not_awaited()
    assert await bot.state.current("777") == "idle"
