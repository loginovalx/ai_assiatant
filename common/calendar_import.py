"""Разбор CSV с тестовым календарём: id, date, time, text.

Формат фиксированный (см. ТЗ), других таблиц не поддерживаем. Файл разбирается целиком
в память и проверяется построчно: в LLM он не попадает, события сохраняются в ту же
таблицу events, что и созданные голосом, и дальше живут как обычные записи календаря.
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

REQUIRED_COLUMNS = ("id", "date", "time", "text")
MAX_ROWS = 5000
MAX_NAME_LENGTH = 200  # столько же, сколько CHECK в схеме events
MAX_REPORTED_ERRORS = 20
_TIME_FORMATS = ("%H:%M", "%H:%M:%S")

# Импортированное событие занимает время: иначе бот не увидит пересечения
# при «поставь встречу завтра в 14:00». Длительность календарного слота по умолчанию.
IMPORTED_DURATION_MINUTES = 60


class CalendarImportError(ValueError):
    """Файл нельзя импортировать целиком: не тот формат или он пуст."""


@dataclass(frozen=True)
class CalendarRow:
    external_id: str
    start_at: datetime
    name: str


def parse_calendar_csv(data: bytes, tz: ZoneInfo) -> tuple[list[CalendarRow], list[str]]:
    """Вернуть корректные строки и список ошибок по некорректным.

    Одна битая строка не отменяет весь импорт: остальные загрузятся, а про проблемные
    пользователь получит понятный список. Нечитаемый файл — CalendarImportError.
    """
    try:
        # utf-8-sig: Excel сохраняет CSV с BOM, и без этого первая колонка станет "﻿id".
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CalendarImportError("Файл не в кодировке UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text))
    columns = {(name or "").strip().lower() for name in (reader.fieldnames or [])}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise CalendarImportError(f"В файле нет обязательных колонок: {', '.join(missing)}")

    rows: list[CalendarRow] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line, raw in enumerate(reader, start=2):  # строка 1 — заголовок
        if len(rows) >= MAX_ROWS:
            errors.append(f"Файл длиннее {MAX_ROWS} строк: остальные пропущены")
            break
        try:
            row = _parse_row({(k or "").strip().lower(): v for k, v in raw.items()})
        except ValueError as exc:
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(f"строка {line}: {exc}")
            continue
        if row.external_id in seen:
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(f"строка {line}: id {row.external_id} повторяется в файле")
            continue
        seen.add(row.external_id)
        rows.append(CalendarRow(row.external_id, row.start_at.replace(tzinfo=tz), row.name))

    if not rows and not errors:
        raise CalendarImportError("В файле нет ни одной строки с данными")
    return rows, errors


def _parse_row(raw: dict[str, str | None]) -> CalendarRow:
    external_id = (raw.get("id") or "").strip()
    date_text = (raw.get("date") or "").strip()
    time_text = (raw.get("time") or "").strip()
    name = " ".join((raw.get("text") or "").split())

    if not external_id:
        raise ValueError("пустой id")
    if not name:
        raise ValueError("пустой text")
    try:
        day = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"дата «{date_text}» не в формате ГГГГ-ММ-ДД") from exc

    for fmt in _TIME_FORMATS:
        try:
            moment = datetime.strptime(time_text, fmt).time()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"время «{time_text}» не в формате ЧЧ:ММ")

    # Наивный datetime: часовой пояс пользователя проставит parse_calendar_csv.
    return CalendarRow(external_id, datetime.combine(day, moment), name[:MAX_NAME_LENGTH])
