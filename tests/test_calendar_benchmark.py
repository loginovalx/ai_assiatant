from datetime import datetime
from zoneinfo import ZoneInfo

from agent.calendar_benchmark import build_calendar_cases
from common.calendar_import import CalendarRow

MOSCOW = ZoneInfo("Europe/Moscow")


def rows(*items: tuple[str, str, str]) -> list[CalendarRow]:
    return [
        CalendarRow(external_id=event_id, start_at=datetime.fromisoformat(start), name=name)
        for event_id, start, name in items
    ]


def test_today_list_uses_half_open_bounds_and_actual_external_ids():
    events = rows(
        ("699", "2026-10-29T23:59:00+03:00", "Boundary before"),
        ("700", "2026-10-30T00:00:00+03:00", "Boundary at start"),
        ("701", "2026-10-30T23:59:00+03:00", "Last minute"),
        ("702", "2026-10-31T00:00:00+03:00", "Boundary at end"),
    )

    cases = build_calendar_cases(events, datetime(2026, 10, 30, 9, tzinfo=MOSCOW))
    today = next(case for case in cases if case.id == "L01")

    assert today.action == "get_events"
    assert today.expected["expected_ids"] == ["700", "701"]
    assert today.expected["start_time"] == "2026-10-30T00:00:00+03:00"
    assert today.expected["end_time"] == "2026-10-31T00:00:00+03:00"
    assert today.expected["event_types"] == ["task", "meeting", "reminder"]


def test_previous_month_includes_leap_day_and_year_boundary_is_correct():
    events = rows(
        ("feb28", "2024-02-28T23:00:00+03:00", "February 28"),
        ("feb29", "2024-02-29T23:00:00+03:00", "Leap day"),
        ("mar01", "2024-03-01T00:00:00+03:00", "March boundary"),
        ("dec31", "2023-12-31T23:00:00+03:00", "Year boundary"),
    )
    cases = build_calendar_cases(events, datetime(2024, 3, 1, 9, tzinfo=MOSCOW))
    previous_month = next(case for case in cases if case.id == "L10")

    assert previous_month.expected["start_time"] == "2024-02-01T00:00:00+03:00"
    assert previous_month.expected["end_time"] == "2024-03-01T00:00:00+03:00"
    assert previous_month.expected["expected_ids"] == ["feb28", "feb29"]

    year_cases = build_calendar_cases(events, datetime(2024, 1, 1, 9, tzinfo=MOSCOW))
    year_boundary = next(case for case in year_cases if case.id == "L19")
    assert year_boundary.expected["start_time"] == "2023-12-29T00:00:00+03:00"
    assert year_boundary.expected["end_time"] == "2024-01-01T00:00:00+03:00"
    assert year_boundary.expected["expected_ids"] == ["dec31"]


def test_builder_returns_20_list_and_20_search_cases_for_rare_titles():
    events = [
        CalendarRow(
            external_id=str(index),
            start_at=datetime(2024, 1, index, 12, tzinfo=MOSCOW),
            name=f"Unique benchmark item {index}",
        )
        for index in range(1, 11)
    ]
    cases = build_calendar_cases(events, datetime(2026, 10, 30, 9, tzinfo=MOSCOW))

    assert len(cases) == 40
    assert [case.id for case in cases] == [
        *(f"L{index:02d}" for index in range(1, 21)),
        *(f"S{index:02d}" for index in range(1, 21)),
    ]
    exact, variant = cases[20], cases[30]
    assert exact.action == variant.action == "search_events"
    assert exact.expected["benchmark_kind"] == "exact_title"
    assert exact.expected["required_result_ids"] == [events[0].external_id]
    assert exact.expected["forbidden_ids"] == []
    assert variant.expected["benchmark_kind"] == "lexical_variant"
    assert variant.expected["required_result_ids"] == [events[0].external_id]
    assert variant.expected["forbidden_ids"] == []


def test_week_periods_are_monday_based_and_half_open():
    cases = build_calendar_cases([], datetime(2026, 10, 30, 9, tzinfo=MOSCOW))
    previous, current, following = [next(c for c in cases if c.id == f"L{i:02}") for i in (4, 5, 6)]

    assert (previous.expected["start_time"], previous.expected["end_time"]) == (
        "2026-10-19T00:00:00+03:00",
        "2026-10-26T00:00:00+03:00",
    )
    assert (current.expected["start_time"], current.expected["end_time"]) == (
        "2026-10-26T00:00:00+03:00",
        "2026-11-02T00:00:00+03:00",
    )
    assert (following.expected["start_time"], following.expected["end_time"]) == (
        "2026-11-02T00:00:00+03:00",
        "2026-11-09T00:00:00+03:00",
    )


def test_all_list_windows_are_at_most_31_calendar_days():
    cases = build_calendar_cases([], datetime(2026, 10, 30, 9, tzinfo=MOSCOW))
    list_cases = [case for case in cases if case.action == "get_events"]

    assert len(list_cases) == 20
    assert all(
        (
            datetime.fromisoformat(case.expected["end_time"])
            - datetime.fromisoformat(case.expected["start_time"])
        ).days
        <= 31
        for case in list_cases
    )
