from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from common.schemas import EventCreate, EventStatus, EventType

USER = "user-1"


def at(tz, day_offset: int, hour: int, minute: int = 0) -> datetime:
    base = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(days=day_offset, hours=hour, minutes=minute)


async def add(repo, type_, name, start=None, user=USER, description=""):
    return await repo.create(
        EventCreate(user_id=user, type=type_, name=name, start_at=start, description=description)
    )


def test_meeting_needs_time():
    with pytest.raises(ValidationError, match="нужно указать дату и время"):
        EventCreate(user_id=USER, type=EventType.MEETING, name="Встреча с Иваном")


def test_naive_time_is_user_timezone(tz):
    data = EventCreate(
        user_id=USER, type=EventType.REMINDER, name="Позвонить", start_at=datetime(2026, 10, 30, 10)
    )
    assert data.start_at.isoformat() == "2026-10-30T10:00:00+03:00"


async def test_create_returns_local_time_and_weekday(repo, tz):
    event = await add(repo, EventType.REMINDER, "Позвонить Ивану", datetime(2026, 10, 30, 7, 0))
    assert event.start_at.utcoffset() == timedelta(hours=3)
    assert event.start_at.hour == 7
    assert event.weekday == "пятница"
    assert event.status == "active"


async def test_task_without_date(repo):
    event = await add(repo, EventType.TASK, "Купить билеты")
    assert event.start_at is None and event.weekday is None


async def test_list_today_half_open_interval(repo, tz):
    today, tomorrow = at(tz, 0, 0), at(tz, 1, 0)
    await add(repo, EventType.MEETING, "Встреча с командой", at(tz, 0, 15))
    await add(repo, EventType.REMINDER, "Позвонить Ивану", at(tz, 0, 10))
    await add(repo, EventType.MEETING, "Ровно в полночь", tomorrow)  # уже завтра
    await add(repo, EventType.TASK, "Без даты")

    events = await repo.list_by_period(USER, today, tomorrow)
    assert [e.name for e in events] == ["Позвонить Ивану", "Встреча с командой"]

    with_undated = await repo.list_by_period(USER, today, tomorrow, include_undated=True)
    assert with_undated[-1].name == "Без даты"


async def test_list_filters_type_user_and_cancelled(repo, tz, pool):
    week_start, week_end = at(tz, 0, 0), at(tz, 7, 0)
    await add(repo, EventType.MEETING, "Встреча с дизайнером", at(tz, 2, 11))
    await add(repo, EventType.TASK, "Отправить отчёт", at(tz, 3, 18))
    await add(repo, EventType.MEETING, "Чужая встреча", at(tz, 2, 11), user="user-2")
    cancelled = await add(repo, EventType.MEETING, "Отменённая", at(tz, 4, 11))
    await pool.execute("UPDATE events SET status = 'cancelled' WHERE id = $1", cancelled.id)

    meetings = await repo.list_by_period(USER, week_start, week_end, [EventType.MEETING])
    assert [e.name for e in meetings] == ["Встреча с дизайнером"]


async def test_search_by_word_form(repo, tz):
    await add(repo, EventType.TASK, "Подписать договор с подрядчиком")
    await add(repo, EventType.TASK, "Купить билеты")

    hits = await repo.search(USER, "договору")
    assert [h.name for h in hits] == ["Подписать договор с подрядчиком"]
    assert "text" in hits[0].matched_by


async def test_search_ignores_extra_words(repo):
    """«задачу про договор»: слова «задача» в записи нет, но запись должна найтись."""
    await add(repo, EventType.TASK, "Подписать договор")
    hits = await repo.search(USER, "задачу про договор")
    assert [h.name for h in hits] == ["Подписать договор"]


async def test_search_with_typo(repo):
    await add(repo, EventType.TASK, "Сделать презентацию для инвесторов")
    hits = await repo.search(USER, "презинтацию")
    assert hits and hits[0].name == "Сделать презентацию для инвесторов"
    assert "fuzzy" in hits[0].matched_by


async def test_search_past_meeting_with_person(repo, tz):
    await add(repo, EventType.MEETING, "Встреча с Иваном", at(tz, -3, 14))
    await add(repo, EventType.MEETING, "Встреча с дизайнером", at(tz, -2, 11))
    hits = await repo.search(USER, "встреча с Иваном")
    assert hits[0].name == "Встреча с Иваном"


async def test_search_is_per_user(repo):
    await add(repo, EventType.TASK, "Подписать договор", user="user-2")
    assert await repo.search(USER, "договор") == []


async def test_search_by_meaning(semantic_repo):
    """Общих слов нет: «слайды» ↔ «презентация» — находит только ветка эмбеддингов."""
    await add(semantic_repo, EventType.TASK, "Подготовить презентацию")
    await add(semantic_repo, EventType.TASK, "Купить молоко")

    hits = await semantic_repo.search(USER, "слайды")
    assert hits[0].name == "Подготовить презентацию"
    assert hits[0].matched_by == ["semantic"]


async def test_search_ranks_multiple_signals_higher(semantic_repo):
    await add(semantic_repo, EventType.TASK, "Подготовить презентацию")
    await add(semantic_repo, EventType.TASK, "Слайды для отчёта")

    hits = await semantic_repo.search(USER, "презентация")
    assert hits[0].name == "Подготовить презентацию"
    assert set(hits[0].matched_by) >= {"text", "semantic"}


async def test_postpone_clears_notified_and_checks_owner(repo, tz):
    """«На час позже»: время сдвинуто, напоминание вернётся в очередь; чужая кнопка не двигает."""
    now = datetime.now(tz)
    event = await add(repo, EventType.REMINDER, "Позвонить в сервис", now - timedelta(minutes=1))
    await repo.claim_due_events(now, timedelta(hours=1), 10)

    later = await repo.postpone(USER, event.id, now + timedelta(hours=1))
    assert later.start_at > now + timedelta(minutes=59)
    assert await repo.claim_due_events(now + timedelta(hours=1, seconds=1), timedelta(hours=1), 10)

    assert await repo.postpone("other-user", event.id, now + timedelta(hours=2)) is None
    cancelled = await add(repo, EventType.TASK, "Прочее")
    await repo.set_status(USER, cancelled.id, EventStatus.CANCELLED)
    assert await repo.postpone(USER, cancelled.id, now + timedelta(hours=1)) is None
