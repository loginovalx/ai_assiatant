from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from bot.recurring import RecurringReminders
from bot.reminders import ReminderSender
from common.reminder_series import ReminderRule, parse_command
from tests.test_time_choice_buttons import FakeMax, all_buttons


@pytest.mark.parametrize(
    ("command", "weekday"),
    [
        ("Напоминай каждый понедельник в 09:00 проверить отчёт", 0),
        ("Напомни по средам в 9:00 проверить отчёт", 2),
        ("Напоминай каждый день в 09:00 проверить отчёт", None),
    ],
)
def test_parse_repeat(command, weekday):
    assert parse_command(command) == ReminderRule(
        name="проверить отчёт", weekday=weekday, clock=time(9)
    )


def test_parse_reminder_from_natural_create_command():
    assert parse_command("поставь напоминание на каждый вторник в 17:48 полить цветы") == (
        ReminderRule(name="полить цветы", weekday=1, clock=time(17, 48))
    )


@pytest.mark.parametrize(
    "command",
    [
        "Создай задачу каждый понедельник в 09:00 проверить отчёт",
        "Напоминай каждый понедельник в 29:00 проверить отчёт",
        "Напоминай каждый день в 09:00 проверить отчёт до конца декабря",
    ],
)
def test_refuses_unsupported_schedule(command):
    with pytest.raises(ValueError):
        parse_command(command)


async def test_preview_edit_and_confirm_use_only_entered_clock(redis, tz):
    repo = AsyncMock()
    api = FakeMax()
    repeats = RecurringReminders(repo, redis, api, tz)
    series = uuid4()

    async def press(label):
        payload = next(b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == label)
        await repeats.callback("777", uuid4().hex, payload)

    assert await repeats.message("777", "Напоминай каждый понедельник в 09:00 отчёт", series)
    assert await repeats.message("777", "Исправить", uuid4())
    await press("Изменить время")
    assert await repeats.message("777", "25:00", uuid4())
    assert "ЧЧ:ММ" in api.sent[-1]["text"]
    assert await repeats.message("777", "10:30", uuid4())
    assert "Каждый понедельник в 10:30" in api.sent[-1]["text"]
    repo.save_reminder_series.assert_not_called()
    await press("Да")
    assert [b["text"] for b in all_buttons(api.sent[-1])] == ["Изменить", "Отключить"]
    args = repo.save_reminder_series.await_args.args
    assert args[:3] == ("777", series, ReminderRule(name="отчёт", weekday=0, clock=time(10, 30)))


async def test_reminder_series_rolls_forward_edits_and_stops(repo, pool, tz):
    now = datetime(2030, 1, 7, 8, tzinfo=tz)
    rule = ReminderRule(name="Проверить отчёт", weekday=0, clock=time(9))
    series, op = uuid4(), uuid4()
    assert await repo.save_reminder_series("777", series, rule, now, operation_id=op)
    assert await repo.save_reminder_series("777", series, rule, now, operation_id=op)
    assert await pool.fetchval("SELECT count(*) FROM events") == 2
    api = FakeMax()
    assert await ReminderSender(repo, api, tz).tick(now + timedelta(hours=1, minutes=1)) == 1
    assert any("Повтор напоминания" == b["text"] for b in all_buttons(api.sent[-1]))

    later = now + timedelta(weeks=10)
    await repo.fill_reminders(later, timedelta(hours=1))
    assert await pool.fetchval("SELECT max(start_at) FROM events") == later + timedelta(
        days=7, hours=1
    )
    new = rule.model_copy(update={"clock": time(11)})
    assert await repo.save_reminder_series("777", series, new, later, op, operation_id=uuid4())
    assert await pool.fetchval(
        "SELECT min(start_at) FROM events WHERE start_at >= $1", later
    ) == later + timedelta(hours=3)
    assert not await repo.stop_reminder_series("other", series, later)
    assert await repo.stop_reminder_series("777", series, later)
    await repo.fill_reminders(later + timedelta(days=8), timedelta(hours=1))
    assert await pool.fetchval("SELECT count(*) FROM events WHERE start_at >= $1", later) == 0
