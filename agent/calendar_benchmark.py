"""Build deterministic list/search cases from imported calendar rows."""

from collections import defaultdict
from datetime import datetime, time, timedelta

from agent.evaluation import EvalCase
from common.calendar_import import CalendarRow

_EVENT_TYPES = ["task", "meeting", "reminder"]
_LEADING_VERBS = frozenset(
    "подготовиться подготовить напомнить проверить согласовать "
    "отправить обсудить провести собрать передать созвон встреча демо".split()
)
_VARIANT_TEMPLATES = (
    "Поищи запись про {topic}.",
    "Что у меня было записано насчёт {topic}?",
    "Найди календарное событие, связанное с {topic}.",
    "Вспомни запись о {topic}.",
)


def _shift_month(month_start: datetime, offset: int) -> datetime:
    index = month_start.year * 12 + month_start.month - 1 + offset
    year, month_index = divmod(index, 12)
    return datetime(year, month_index + 1, 1, tzinfo=month_start.tzinfo)


def _periods(now: datetime) -> list[tuple[str, str, datetime, datetime]]:
    day = now.date()
    today = datetime.combine(day, time.min, tzinfo=now.tzinfo)
    tomorrow = today + timedelta(days=1)
    month = datetime(day.year, day.month, 1, tzinfo=now.tzinfo)
    next_month = _shift_month(month, 1)
    week = today - timedelta(days=day.weekday())  # Weeks start Monday.
    yesterday = today - timedelta(days=1)
    previous_week, next_week = week - timedelta(days=7), week + timedelta(days=7)
    last_7, next_7 = today - timedelta(days=6), today + timedelta(days=7)
    previous_month, month_after_next = _shift_month(month, -1), _shift_month(month, 2)
    last_30, next_30 = today - timedelta(days=29), today + timedelta(days=30)
    current_year = datetime(day.year, 1, 1, tzinfo=now.tzinfo)
    previous_year_end = current_year - timedelta(days=1)
    previous_month_end = previous_month + timedelta(days=1)
    current_month_end = month + timedelta(days=3)
    next_month_end = next_month + timedelta(days=1)
    previous_year_start = previous_year_end - timedelta(days=2)
    current_year_end = current_year + timedelta(days=3)
    current_month_last3 = current_month_end - timedelta(days=1)
    current_year_last3 = current_year_end - timedelta(days=1)
    return [
        ("today", "Что у меня сегодня?", today, tomorrow),
        ("yesterday", "Что было вчера?", yesterday, today),
        ("tomorrow", "Что у меня завтра?", tomorrow, tomorrow + timedelta(days=1)),
        (
            "previous_week",
            "Прошлая календарная неделя (понедельник-воскресенье)?",
            previous_week,
            week,
        ),
        ("current_week", "Текущая календарная неделя (понедельник-воскресенье)?", week, next_week),
        (
            "next_week",
            "Следующая календарная неделя (понедельник-воскресенье)?",
            next_week,
            next_week + timedelta(days=7),
        ),
        ("last_7_days", "Последние 7 дней, включая сегодня?", last_7, tomorrow),
        ("next_7_days", "Следующие 7 дней, включая сегодня?", today, next_7),
        (
            "month_to_date",
            "Покажи записи с начала этого месяца по сегодняшний день включительно.",
            month,
            tomorrow,
        ),
        ("previous_month", "Прошлый календарный месяц?", previous_month, month),
        ("current_month", "Записи за текущий календарный месяц.", month, next_month),
        ("next_month", "Следующий календарный месяц?", next_month, month_after_next),
        ("rest_of_month", "С завтрашнего дня до конца месяца?", tomorrow, next_month),
        (
            "prev_month_day",
            f"Что было {previous_month:%Y-%m-%d}?",
            previous_month,
            previous_month_end,
        ),
        (
            "month_first_3",
            f"С {month:%Y-%m-%d} по {current_month_last3:%Y-%m-%d} включительно?",
            month,
            current_month_end,
        ),
        ("next_month_day", f"Что будет {next_month:%Y-%m-%d}?", next_month, next_month_end),
        ("last_30_days", "Последние 30 дней, включая сегодня?", last_30, tomorrow),
        ("next_30_days", "Следующие 30 дней, включая сегодня?", today, next_30),
        (
            "year_last_3",
            f"С {previous_year_start:%Y-%m-%d} по {previous_year_end:%Y-%m-%d} включительно?",
            previous_year_start,
            current_year,
        ),
        (
            "year_first_3",
            f"С {current_year:%Y-%m-%d} по {current_year_last3:%Y-%m-%d} включительно?",
            current_year,
            current_year_end,
        ),
    ]


def _aware_start(row: CalendarRow, tzinfo) -> datetime:
    if row.start_at.tzinfo is None:
        return row.start_at.replace(tzinfo=tzinfo)
    return row.start_at.astimezone(tzinfo)


def _search_topic(title: str) -> str:
    words = title.split()
    if len(words) > 1 and words[0].casefold() in _LEADING_VERBS:
        return " ".join(words[1:])
    return title


def _search_cases(rows: list[CalendarRow], now: datetime) -> list[EvalCase]:
    by_title: dict[str, list[CalendarRow]] = defaultdict(list)
    for row in rows:
        by_title[row.name].append(row)
    rare = [
        (title, sorted(items, key=lambda item: (_aware_start(item, now.tzinfo), item.external_id)))
        for title, items in by_title.items()
        if 1 <= len(items) <= 5
    ]
    rare.sort(key=lambda pair: (len(pair[1]), pair[0].casefold(), pair[1][0].external_id))
    selected = rare[:10]

    cases = []
    for index, (title, items) in enumerate(selected, start=1):
        ids = [row.external_id for row in items]
        cases.append(
            EvalCase(
                id=f"S{index:02d}",
                text=f"Найди запись с точным названием «{title}».",
                action="search_events",
                expected={
                    "required_result_ids": ids,
                    "forbidden_ids": [],
                    "benchmark_kind": "exact_title",
                },
                reference_now=now,
                history=[],
                user_id="calendar-benchmark",
                seed_ids=[],
            )
        )
    for index, (title, items) in enumerate(selected, start=1):
        ids = [row.external_id for row in items]
        template = _VARIANT_TEMPLATES[(index - 1) % len(_VARIANT_TEMPLATES)]
        topic = _search_topic(title)
        cases.append(
            EvalCase(
                id=f"S{index + 10:02d}",
                text=template.format(topic=topic),
                action="search_events",
                expected={
                    "required_result_ids": ids,
                    "forbidden_ids": [],
                    "benchmark_kind": "lexical_variant",
                },
                reference_now=now,
                history=[],
                user_id="calendar-benchmark",
                seed_ids=[],
            )
        )
    return cases


def build_calendar_cases(rows: list[CalendarRow], now: datetime) -> list[EvalCase]:
    """Build cases from row facts; weeks start Monday and list bounds are half-open."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    ordered_rows = sorted(rows, key=lambda item: (_aware_start(item, now.tzinfo), item.external_id))
    cases = []
    for index, (label, query, start, end) in enumerate(_periods(now), start=1):
        ids = [
            row.external_id for row in ordered_rows if start <= _aware_start(row, now.tzinfo) < end
        ]
        cases.append(
            EvalCase(
                id=f"L{index:02d}",
                text=query,
                action="get_events",
                expected={
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "event_types": list(_EVENT_TYPES),
                    "expected_ids": ids,
                    "benchmark_kind": label,
                },
                reference_now=now,
                history=[],
                user_id="calendar-benchmark",
                seed_ids=[],
            )
        )
    return cases + _search_cases(rows, now)
