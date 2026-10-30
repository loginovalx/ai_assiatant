"""Импорт тестового календаря из CSV: разбор файла и попадание строк в общий календарь."""

from datetime import datetime, timedelta

import pytest

from common.calendar_import import (
    IMPORTED_DURATION_MINUTES,
    CalendarImportError,
    parse_calendar_csv,
)
from common.repository import EventRepository
from common.schemas import EventCreate, EventType

USER = "user-1"

HEADER = "id,date,time,text\n"
THREE_ROWS = HEADER + (
    "1,2024-01-07,14:45,Подготовить техническое задание\n"
    "2,2024-01-12,13:45,Отправить договор Ивану\n"
    "3,2024-01-14,14:30,Отправить итоги встречи Марии\n"
)


def test_parses_rows_with_user_timezone(tz):
    rows, errors = parse_calendar_csv(THREE_ROWS.encode(), tz)
    assert errors == []
    assert [r.external_id for r in rows] == ["1", "2", "3"]
    assert rows[0].name == "Подготовить техническое задание"
    assert rows[0].start_at == datetime(2024, 1, 7, 14, 45, tzinfo=tz)


def test_excel_bom_does_not_break_first_column(tz):
    """Excel пишет CSV с BOM, и наивный разбор видит колонку «﻿id» вместо «id»."""
    rows, errors = parse_calendar_csv(b"\xef\xbb\xbf" + THREE_ROWS.encode(), tz)
    assert errors == [] and len(rows) == 3


def test_missing_columns_rejected(tz):
    with pytest.raises(CalendarImportError, match="time, text"):
        parse_calendar_csv(b"id,date\n1,2024-01-07\n", tz)


def test_empty_file_rejected(tz):
    with pytest.raises(CalendarImportError, match="ни одной строки"):
        parse_calendar_csv(HEADER.encode(), tz)


def test_bad_rows_are_skipped_but_good_ones_load(tz):
    data = HEADER + (
        "1,2024-01-07,14:45,Хорошая строка\n"
        "2,07.01.2024,14:45,Дата не в том формате\n"
        "3,2024-01-07,25:99,Время не в том формате\n"
        "4,2024-01-07,14:45,\n"
        ",2024-01-07,14:45,Без id\n"
        "1,2024-01-08,10:00,Повтор id из первой строки\n"
    )
    rows, errors = parse_calendar_csv(data.encode(), tz)

    assert [r.external_id for r in rows] == ["1"]
    assert len(errors) == 5
    assert "строка 3" in errors[0] and "ГГГГ-ММ-ДД" in errors[0]
    assert "ЧЧ:ММ" in errors[1]
    assert "повторяется" in errors[4]


def test_long_text_truncated_to_column_limit(tz):
    rows, _ = parse_calendar_csv((HEADER + f"1,2024-01-07,14:45,{'я' * 300}\n").encode(), tz)
    assert len(rows[0].name) == 200


async def test_import_is_idempotent_and_visible_in_calendar(repo: EventRepository, tz):
    """Повторная загрузка того же файла не плодит дубли, а события видны как обычные."""
    rows, _ = parse_calendar_csv(THREE_ROWS.encode(), tz)

    assert await repo.import_calendar(USER, rows) == 3
    assert await repo.import_calendar(USER, rows) == 0  # тот же файл второй раз

    day = datetime(2024, 1, 7, tzinfo=tz)
    found = await repo.list_by_period(USER, day, day + timedelta(days=1))
    assert [e.name for e in found] == ["Подготовить техническое задание"]
    assert found[0].duration_minutes == IMPORTED_DURATION_MINUTES


async def test_imported_event_is_found_by_text_search(repo: EventRepository, tz):
    rows, _ = parse_calendar_csv(THREE_ROWS.encode(), tz)
    await repo.import_calendar(USER, rows)

    hits = await repo.search(USER, "договор")
    assert [h.name for h in hits] == ["Отправить договор Ивану"]


async def test_imported_event_occupies_time_and_conflicts(repo: EventRepository, tz):
    """Главное требование ТЗ: загруженный календарь учитывается при создании новых встреч."""
    rows, _ = parse_calendar_csv(THREE_ROWS.encode(), tz)
    await repo.import_calendar(USER, rows)

    meeting = await repo.create(
        EventCreate(
            user_id=USER,
            type=EventType.MEETING,
            name="Новая встреча",
            start_at=datetime(2024, 1, 7, 15, 0, tzinfo=tz),  # внутри слота 14:45–15:45
            duration_minutes=30,
        )
    )
    conflicts = await repo.find_conflicts(meeting)
    assert [c.name for c in conflicts] == ["Подготовить техническое задание"]


async def test_import_separates_users(repo: EventRepository, tz):
    rows, _ = parse_calendar_csv(THREE_ROWS.encode(), tz)
    await repo.import_calendar(USER, rows)
    # Тот же внешний id у другого пользователя — отдельная запись, а не конфликт.
    assert await repo.import_calendar("user-2", rows) == 3

    day = datetime(2024, 1, 7, tzinfo=tz)
    mine = await repo.list_by_period(USER, day, day + timedelta(days=1))
    assert len(mine) == 1
