"""Автоматическая отправка напоминаний: планировщик берёт наступившее из базы и шлёт в MAX."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from bot.reminders import ReminderSender, format_notification
from common.calendar_import import parse_calendar_csv
from common.repository import EventRepository
from common.schemas import Event, EventCreate, EventStatus, EventType

USER = "100500"


class FakeMax:
    """Вместо настоящего MAX: записывает отправленное, умеет падать по требованию."""

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self.fail = fail

    async def send(
        self, chat_id: int, text: str, *, to_user: bool = False, buttons: list | None = None
    ) -> None:
        if self.fail:
            raise OSError("MAX недоступен")
        self.sent.append((chat_id, text))


@pytest.fixture
def sender(repo, tz):
    def build(max_api, lookback_minutes: int = 60) -> ReminderSender:
        return ReminderSender(repo, max_api, tz, lookback_minutes=lookback_minutes)

    return build


async def add(repo: EventRepository, start_at, type_=EventType.REMINDER, name="Позвонить Ивану"):
    return await repo.create(EventCreate(user_id=USER, type=type_, name=name, start_at=start_at))


async def test_sends_due_reminder_once(repo, sender, tz):
    now = datetime.now(tz).replace(microsecond=0)
    await add(repo, now - timedelta(minutes=1))
    max_api = FakeMax()

    assert await sender(max_api).tick(now) == 1
    assert max_api.sent == [(int(USER), "Напоминание: Позвонить Ивану")]

    # Второй тик (и перезапуск сервиса) повтором не оборачивается: отметка в базе.
    assert await sender(max_api).tick(now) == 0
    assert len(max_api.sent) == 1


async def test_future_reminder_waits(repo, sender, tz):
    now = datetime.now(tz).replace(microsecond=0)
    await add(repo, now + timedelta(minutes=5))
    max_api = FakeMax()

    assert await sender(max_api).tick(now) == 0
    assert await sender(max_api).tick(now + timedelta(minutes=6)) == 1


async def test_old_events_are_not_blasted_after_downtime(repo, sender, tz):
    """В импортированном календаре есть прошлогодние даты: вывалить их пользователю нельзя."""
    now = datetime.now(tz).replace(microsecond=0)
    old_csv = "id,date,time,text\n1,2024-01-07,14:45,Старое дело\n".encode()
    rows, _ = parse_calendar_csv(old_csv, tz)
    await repo.import_calendar(USER, rows)
    await add(repo, now - timedelta(hours=5), name="Напоминание из вчера")
    await add(repo, now - timedelta(minutes=10), name="Только что наступило")
    max_api = FakeMax()

    assert await sender(max_api).tick(now) == 1
    assert "Только что наступило" in max_api.sent[0][1]


async def test_done_and_cancelled_are_skipped(repo, sender, tz):
    now = datetime.now(tz).replace(microsecond=0)
    done = await add(repo, now - timedelta(minutes=1), name="Выполненное")
    await repo.set_status(USER, done.id, EventStatus.DONE)
    cancelled = await add(repo, now - timedelta(minutes=1), name="Удалённое")
    await repo.set_status(USER, cancelled.id, EventStatus.CANCELLED)
    max_api = FakeMax()

    assert await sender(max_api).tick(now) == 0


async def test_failed_send_is_retried_next_tick(repo, sender, tz):
    now = datetime.now(tz).replace(microsecond=0)
    await add(repo, now - timedelta(minutes=1))

    broken = FakeMax(fail=True)
    assert await sender(broken).tick(now) == 0  # MAX лежал — отметку сняли обратно

    working = FakeMax()
    assert await sender(working).tick(now) == 1
    assert len(working.sent) == 1


async def test_meeting_notification_mentions_time(repo, sender, tz):
    now = datetime.now(tz).replace(microsecond=0)
    start = now - timedelta(minutes=1)
    await add(repo, start, type_=EventType.MEETING, name="Созвон с Иваном")
    max_api = FakeMax()

    await sender(max_api).tick(now)
    assert max_api.sent[0][1] == f"Напоминание: встреча «Созвон с Иваном» в {start:%H:%M}"


def test_reminder_text_matches_spec(tz):
    """Текст из ТЗ дословно: «Напоминание: позвонить Ивану.»"""
    now = datetime.now(tz)
    event = Event(
        id=uuid4(),
        user_id=USER,
        type=EventType.REMINDER,
        name="позвонить Ивану",
        description="",
        start_at=now,
        status=EventStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    assert format_notification(event) == "Напоминание: позвонить Ивану"
