import csv
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agent.evaluation import EvalCase, judge_case, load_cases
from common.schemas import (
    AgentResponse,
    AgentStatus,
    Event,
    EventStatus,
    EventType,
)

DEFAULT_START = datetime.fromisoformat("2026-10-30T10:00:00+03:00")


def make_event(
    *,
    name: str = "Позвонить Ивану",
    event_type: EventType = EventType.REMINDER,
    start_at: datetime | None = DEFAULT_START,
) -> Event:
    now = datetime.fromisoformat("2026-10-29T09:00:00+03:00")
    return Event(
        id=uuid4(),
        user_id="case-user",
        type=event_type,
        name=name,
        description="",
        start_at=start_at,
        status=EventStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


def case(action: str, expected: dict) -> EvalCase:
    return EvalCase(
        id="T-test",
        text="test",
        action=action,
        expected=expected,
        reference_now=datetime.fromisoformat("2026-10-29T09:00:00+03:00"),
        history=[],
        user_id="case-user",
        seed_ids=[],
    )


def response(
    status: AgentStatus = AgentStatus.DONE,
    events: list[Event] | None = None,
    reply_text: str = "Готово",
    draft=None,
) -> AgentResponse:
    return AgentResponse(reply_text=reply_text, status=status, events=events or [], draft=draft)


def test_load_cases_parses_bom_json_and_fixed_reference_time(tmp_path: Path):
    path = tmp_path / "Dataset.csv"
    columns = [
        "id",
        "text",
        "expected_action",
        "expected_fields",
        "reference_now",
        "user_id",
        "dialogue_history",
        "seed_ids",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "id": "T001",
                "text": "Напомни завтра в десять позвонить Ивану",
                "expected_action": "create_event",
                "expected_fields": json.dumps({"event_type": "reminder"}),
                "reference_now": "Thu Oct 29 2026 09:00:00 GMT+0300 (Moscow Standard Time)",
                "user_id": "case-user",
                "dialogue_history": "[]",
                "seed_ids": '["E01"]',
            }
        )

    [loaded] = load_cases(path)
    assert loaded.id == "T001"
    assert loaded.expected == {"event_type": "reminder"}
    assert loaded.reference_now == datetime.fromisoformat("2026-10-29T09:00:00+03:00")
    assert loaded.seed_ids == ["E01"]


def test_create_requires_successful_tool_and_persisted_row():
    expected = {
        "event_type": "reminder",
        "name": "Позвонить Ивану",
        "start_at": "2026-10-30T10:00:00+03:00",
    }
    event = make_event(name="Позвонить Ивану!")
    tool = {
        "name": "create_event",
        "arguments": {"event_type": "reminder", "name": event.name},
        "ok": True,
    }

    assert judge_case(case("create_event", expected), response(), [event], {}, [tool]) == []

    no_persisted_row = judge_case(case("create_event", expected), response(), [], {}, [tool])
    assert any("persisted created event" in reason for reason in no_persisted_row)

    failed_tool = {**tool, "ok": False}
    prose_only = judge_case(case("create_event", expected), response(), [event], {}, [failed_tool])
    assert any("no successful create_event" in reason for reason in prose_only)


def test_create_rejects_wrong_persisted_event_even_when_tool_args_match():
    expected = {
        "event_type": "reminder",
        "name": "Позвонить Ивану",
        "start_at": "2026-10-30T10:00:00+03:00",
    }
    event = make_event(
        name="Почистить зубы", start_at=datetime.fromisoformat("2026-10-30T11:00:00+03:00")
    )
    tool = {
        "name": "create_event",
        "arguments": expected,
        "ok": True,
    }

    reasons = judge_case(case("create_event", expected), response(), [event], {}, [tool])
    assert any("event name mismatch" in reason for reason in reasons)
    assert any("event start_at mismatch" in reason for reason in reasons)


def test_get_events_checks_success_bounds_types_and_real_ids():
    event = make_event(name="Встреча с дизайнером", event_type=EventType.MEETING)
    expected = {
        "expected_ids": ["E01"],
        "forbidden_ids": [],
        "start_time": "2026-10-30T00:00:00+03:00",
        "end_time": "2026-10-31T00:00:00+03:00",
        "event_types": ["meeting"],
    }
    call = {
        "name": "get_events",
        "ok": True,
        "arguments": {
            "start_time": expected["start_time"],
            "end_time": expected["end_time"],
            "event_types": ["meeting"],
        },
    }
    actual = response(events=[event])
    assert (
        judge_case(case("get_events", expected), actual, [], {"E01": str(event.id)}, [call]) == []
    )

    wrong_id = make_event(name="Встреча с дизайнером", event_type=EventType.MEETING)
    reasons = judge_case(
        case("get_events", expected),
        response(events=[wrong_id]),
        [],
        {"E01": str(event.id)},
        [call],
    )
    assert any("do not exactly match" in reason for reason in reasons)

    wrong_bounds = {
        **call,
        "arguments": {**call["arguments"], "end_time": "2026-11-01T00:00:00+03:00"},
    }
    reasons = judge_case(
        case("get_events", expected), actual, [], {"E01": str(event.id)}, [wrong_bounds]
    )
    assert any("end_time" in reason for reason in reasons)


def test_get_events_requires_logical_mappings_and_rejects_failed_tool():
    expected = {
        "expected_ids": ["E01"],
        "forbidden_ids": ["E12"],
        "start_time": "2026-10-30T00:00:00+03:00",
        "end_time": "2026-10-31T00:00:00+03:00",
        "event_types": ["task", "meeting", "reminder"],
    }
    call = {
        "name": "get_events",
        "ok": False,
        "arguments": {
            "start_time": expected["start_time"],
            "end_time": expected["end_time"],
            "event_types": None,
        },
    }
    reasons = judge_case(case("get_events", expected), response(), [], {}, [call])
    assert any("no successful get_events" in reason for reason in reasons)

    call["ok"] = True
    reasons = judge_case(case("get_events", expected), response(), [], {}, [call])
    assert sum("missing logical ID mapping" in reason for reason in reasons) == 2


def test_get_events_judge_treats_empty_types_as_all_like_mcp():
    event = make_event()
    expected = {
        "expected_ids": ["E01"],
        "start_time": "2026-10-30T00:00:00+03:00",
        "end_time": "2026-10-31T00:00:00+03:00",
        "event_types": ["task", "meeting", "reminder"],
    }
    call = {
        "name": "get_events",
        "ok": True,
        "arguments": {
            "start_time": expected["start_time"],
            "end_time": expected["end_time"],
            "event_types": [],
        },
    }
    assert (
        judge_case(
            case("get_events", expected),
            response(events=[event]),
            [],
            {"E01": str(event.id)},
            [call],
        )
        == []
    )


def test_search_requires_expected_real_id_and_explicit_no_results_reply():
    event = make_event(name="Купить билеты")
    expected = {
        "required_result_ids": ["E05"],
        "forbidden_ids": ["E12"],
        "must_report_no_results": False,
    }
    call = {"name": "search_events", "arguments": {"query": "билеты"}, "ok": True}
    reasons = judge_case(
        case("search_events", expected),
        response(events=[event]),
        [],
        {"E05": str(event.id), "E12": "foreign"},
        [call],
    )
    assert reasons == []

    no_results = {**expected, "required_result_ids": [], "must_report_no_results": True}
    reasons = judge_case(
        case("search_events", no_results),
        response(reply_text="Не нашлось записей"),
        [],
        {"E12": "foreign"},
        [call],
    )
    assert reasons == []
    reasons = judge_case(
        case("search_events", no_results),
        response(reply_text="Готово"),
        [],
        {"E12": "foreign"},
        [call],
    )
    assert any("does not clearly report no results" in reason for reason in reasons)


def test_clarification_rejects_persisted_event_but_allows_a_rejected_tool():
    clarification = case("ask_clarification", {"must_not_create": True})
    created = make_event()
    reasons = judge_case(
        clarification,
        response(status=AgentStatus.NEED_CLARIFICATION),
        [created],
        {},
        [{"name": "create_event", "arguments": {}, "ok": False}],
    )
    assert any("persisted event" in reason for reason in reasons)
    assert (
        judge_case(
            clarification,
            response(status=AgentStatus.NEED_CLARIFICATION),
            [],
            {},
            [{"name": "create_event", "arguments": {}, "ok": False}],
        )
        == []
    )
