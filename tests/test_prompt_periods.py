from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from agent.prompt import build_system_prompt

MOSCOW = ZoneInfo("Europe/Moscow")


def line(name: str, start: datetime, end: datetime) -> str:
    return f"- {name}: {start.isoformat()} — {end.isoformat()}"


def check(prompt: str, expected: list[tuple[str, datetime, datetime]]) -> None:
    for name, start, end in expected:
        assert line(name, start, end) in prompt


def midnight(value: datetime) -> datetime:
    return datetime.combine(value.date(), time.min, tzinfo=value.tzinfo)


def test_day_and_week_periods_include_reported_boundary_fixes():
    now = datetime(2026, 10, 30, 9, 0, tzinfo=MOSCOW)
    today = midnight(now)
    tomorrow = today + timedelta(days=1)
    monday = today - timedelta(days=today.weekday())
    previous_monday = monday - timedelta(days=7)
    next_monday = monday + timedelta(days=7)
    prompt = build_system_prompt(now)
    check(
        prompt,
        [
            ("вчера", today - timedelta(days=1), today),
            ("сегодня", today, tomorrow),
            ("завтра", tomorrow, tomorrow + timedelta(days=1)),
            ("прошлая календарная неделя", previous_monday, monday),
            ("эта неделя", monday, next_monday),
            ("следующая неделя", next_monday, next_monday + timedelta(days=7)),
            ("последние 7 календарных дней, включая сегодня", today - timedelta(days=6), tomorrow),
            ("следующие 7 календарных дней, включая сегодня", today, today + timedelta(days=7)),
        ],
    )


def test_month_and_rolling_period_bounds():
    now = datetime(2026, 10, 30, 9, 0, tzinfo=MOSCOW)
    today = midnight(now)
    tomorrow = today + timedelta(days=1)
    prompt = build_system_prompt(now)
    check(
        prompt,
        [
            (
                "прошлый месяц",
                datetime(2026, 9, 1, tzinfo=MOSCOW),
                datetime(2026, 10, 1, tzinfo=MOSCOW),
            ),
            (
                "этот месяц",
                datetime(2026, 10, 1, tzinfo=MOSCOW),
                datetime(2026, 11, 1, tzinfo=MOSCOW),
            ),
            (
                "следующий месяц",
                datetime(2026, 11, 1, tzinfo=MOSCOW),
                datetime(2026, 12, 1, tzinfo=MOSCOW),
            ),
            ("месяц по сегодня включительно", datetime(2026, 10, 1, tzinfo=MOSCOW), tomorrow),
            ("остаток месяца с завтра", tomorrow, datetime(2026, 11, 1, tzinfo=MOSCOW)),
            (
                "последние 30 календарных дней, включая сегодня",
                datetime(2026, 10, 1, tzinfo=MOSCOW),
                tomorrow,
            ),
            (
                "следующие 30 календарных дней, включая сегодня",
                today,
                datetime(2026, 11, 29, tzinfo=MOSCOW),
            ),
        ],
    )


def test_periods_handle_leap_day_and_february():
    prompt = build_system_prompt(datetime(2024, 3, 1, 9, 0, tzinfo=MOSCOW))
    check(
        prompt,
        [
            ("вчера", datetime(2024, 2, 29, tzinfo=MOSCOW), datetime(2024, 3, 1, tzinfo=MOSCOW)),
            (
                "прошлый месяц",
                datetime(2024, 2, 1, tzinfo=MOSCOW),
                datetime(2024, 3, 1, tzinfo=MOSCOW),
            ),
            (
                "следующий месяц",
                datetime(2024, 4, 1, tzinfo=MOSCOW),
                datetime(2024, 5, 1, tzinfo=MOSCOW),
            ),
        ],
    )


def test_month_periods_handle_year_rollover():
    prompt = build_system_prompt(datetime(2025, 1, 1, 9, 0, tzinfo=MOSCOW))
    check(
        prompt,
        [
            ("вчера", datetime(2024, 12, 31, tzinfo=MOSCOW), datetime(2025, 1, 1, tzinfo=MOSCOW)),
            (
                "прошлый месяц",
                datetime(2024, 12, 1, tzinfo=MOSCOW),
                datetime(2025, 1, 1, tzinfo=MOSCOW),
            ),
            (
                "следующий месяц",
                datetime(2025, 2, 1, tzinfo=MOSCOW),
                datetime(2025, 3, 1, tzinfo=MOSCOW),
            ),
        ],
    )
