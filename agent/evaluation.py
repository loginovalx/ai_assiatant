"""Pure helpers for loading and judging the fixed Dataset.csv cases."""

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from common.schemas import AgentResponse, Event


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str
    action: str
    expected: dict[str, Any]
    reference_now: datetime
    history: list[dict[str, Any]]
    user_id: str
    seed_ids: list[str]


_COLUMNS = {
    "id",
    "text",
    "expected_action",
    "expected_fields",
    "reference_now",
    "user_id",
    "dialogue_history",
    "seed_ids",
}
_REFERENCE_TIME = re.compile(
    r"^(?P<timestamp>[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{4} "
    r"\d{2}:\d{2}:\d{2} GMT[+-]\d{4})(?: \([^)]*\))?$"
)
_MUTATIONS = {
    "create_event",
    "update_event",
    "complete_event",
    "delete_event",
    "delete_events",
}
_NO_RESULTS = re.compile(
    r"\b(?:не\s+найден\w*|не\s+нашл\w*|ничего\s+(?:не\s+)?(?:найден\w*|не\s+нашл\w*)|"
    r"нет\s+(?:подходящ\w*|запис\w*|событи\w*)|не\s+обнаружен\w*)\b",
    re.IGNORECASE,
)


def _load_json_cell(value: str, case_id: str, column: str, expected_type: type) -> Any:
    try:
        result = json.loads(value or ("[]" if expected_type is list else "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{case_id}: invalid JSON in {column}") from exc
    if not isinstance(result, expected_type):
        raise ValueError(f"{case_id}: {column} must contain {expected_type.__name__}")
    return result


def _parse_reference_time(value: str, case_id: str) -> datetime:
    match = _REFERENCE_TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{case_id}: unsupported reference_now format")
    try:
        return datetime.strptime(match.group("timestamp"), "%a %b %d %Y %H:%M:%S GMT%z")
    except ValueError as exc:
        raise ValueError(f"{case_id}: invalid reference_now timestamp") from exc


def load_cases(path: Path) -> list[EvalCase]:
    """Load Dataset.csv, including its UTF-8 BOM and JSON-encoded columns."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = _COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset.csv is missing columns: {', '.join(sorted(missing))}")
        cases = []
        seen_ids: set[str] = set()
        for row in reader:
            case_id = (row.get("id") or "").strip()
            if not case_id:
                raise ValueError("Dataset.csv contains a row without an id")
            if case_id in seen_ids:
                raise ValueError(f"Dataset.csv contains duplicate id {case_id}")
            seen_ids.add(case_id)
            action = (row.get("expected_action") or "").strip()
            if action not in {"create_event", "get_events", "search_events", "ask_clarification"}:
                raise ValueError(f"{case_id}: unsupported expected_action {action!r}")
            history = _load_json_cell(
                row.get("dialogue_history", ""), case_id, "dialogue_history", list
            )
            if any(not isinstance(item, dict) for item in history):
                raise ValueError(f"{case_id}: dialogue_history entries must be objects")
            seed_ids = _load_json_cell(row.get("seed_ids", ""), case_id, "seed_ids", list)
            if any(not isinstance(item, str) for item in seed_ids):
                raise ValueError(f"{case_id}: seed_ids entries must be strings")
            cases.append(
                EvalCase(
                    id=case_id,
                    text=row.get("text", ""),
                    action=action,
                    expected=_load_json_cell(
                        row.get("expected_fields", ""), case_id, "expected_fields", dict
                    ),
                    reference_now=_parse_reference_time(row.get("reference_now", ""), case_id),
                    history=history,
                    user_id=row.get("user_id", ""),
                    seed_ids=seed_ids,
                )
            )
    return cases


def _normal_name(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _event_id(event: Event) -> str:
    return str(event.id)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _logical_ids(
    labels: list[str], mapping: dict[str, str], case_id: str
) -> tuple[set[str], list[str]]:
    missing = [label for label in labels if label not in mapping]
    return {str(mapping[label]) for label in labels if label in mapping}, [
        f"{case_id}: missing logical ID mapping for {label}" for label in missing
    ]


def _successful_calls(calls: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [call for call in calls if call.get("name") == name and call.get("ok") is True]


def _has_mutation_attempt(calls: list[dict[str, Any]]) -> bool:
    return any(call.get("name") in _MUTATIONS and call.get("ok") is True for call in calls)


def judge_case(
    case: EvalCase,
    response: AgentResponse,
    created: list[Event],
    logical_ids: dict[str, str],
    calls: list[dict[str, Any]],
) -> list[str]:
    """Return deterministic acceptance failures for one case; never performs I/O."""
    reasons: list[str] = []
    expected = case.expected

    if case.action == "create_event":
        if response.status != "done":
            reasons.append(f"expected done status, got {response.status}")
        successful = _successful_calls(calls, "create_event")
        if not successful:
            reasons.append("no successful create_event tool call")
        elif len(successful) != 1:
            reasons.append(f"expected one successful create_event call, got {len(successful)}")
        if len(created) != 1:
            reasons.append(f"expected one persisted created event, got {len(created)}")
        if len(created) == 1:
            event = created[0]
            wanted_type = expected.get("event_type")
            if wanted_type is not None and str(event.type) != wanted_type:
                reasons.append(f"event type mismatch: expected {wanted_type}, got {event.type}")
            wanted_name = expected.get("name")
            if wanted_name is not None and _normal_name(event.name) != _normal_name(wanted_name):
                reasons.append(f"event name mismatch: expected {wanted_name!r}, got {event.name!r}")
            wanted_start = _parse_datetime(expected.get("start_at"))
            if wanted_start is None:
                if expected.get("start_at") is not None or event.start_at is not None:
                    reasons.append("event start_at mismatch: expected no date/time")
            elif event.start_at is None:
                reasons.append("event start_at is missing")
            elif event.start_at.tzinfo is None or abs(event.start_at - wanted_start) >= timedelta(
                seconds=60
            ):
                reasons.append(
                    f"event start_at mismatch: expected {wanted_start.isoformat()}, "
                    f"got {event.start_at.isoformat()}"
                )
            if event.start_at is not None and event.start_at.tzinfo is not None:
                local_start = event.start_at.astimezone(case.reference_now.tzinfo)
                if (
                    expected.get("date") is not None
                    and local_start.date().isoformat() != expected["date"]
                ):
                    reasons.append(f"event date mismatch: expected {expected['date']}")
                if (
                    expected.get("time") is not None
                    and local_start.strftime("%H:%M") != expected["time"]
                ):
                    reasons.append(f"event time mismatch: expected {expected['time']}")
        return reasons

    if case.action == "ask_clarification":
        if response.status != "need_clarification":
            reasons.append(f"expected need_clarification status, got {response.status}")
        if created:
            reasons.append(f"clarification created {len(created)} persisted event(s)")
        if response.draft is not None:
            reasons.append("clarification returned a mutation draft")
        if _has_mutation_attempt(calls):
            reasons.append("clarification attempted a mutating tool")
        return reasons

    if case.action not in {"get_events", "search_events"}:
        return [f"unsupported action {case.action!r}"]

    if response.status != "done":
        reasons.append(f"expected done status, got {response.status}")
    if created:
        reasons.append(f"read action created {len(created)} persisted event(s)")
    if _has_mutation_attempt(calls):
        reasons.append("read action attempted a mutating tool")
    tool_calls = _successful_calls(calls, case.action)
    if not tool_calls:
        reasons.append(f"no successful {case.action} tool call")
        return reasons
    arguments = tool_calls[-1].get("arguments") or {}
    actual_ids = {_event_id(event) for event in response.events}

    expected_labels = (
        expected.get("expected_ids", [])
        if case.action == "get_events"
        else expected.get("required_result_ids", [])
    )
    expected_ids, mapping_errors = _logical_ids(expected_labels, logical_ids, case.id)
    reasons.extend(mapping_errors)
    forbidden_ids, mapping_errors = _logical_ids(
        expected.get("forbidden_ids", []), logical_ids, case.id
    )
    reasons.extend(mapping_errors)
    if forbidden_ids & actual_ids:
        reasons.append("response contains forbidden event ID(s)")

    if case.action == "get_events":
        if actual_ids != expected_ids:
            reasons.append("get_events result IDs do not exactly match expected_ids")
        for field in ("start_time", "end_time"):
            wanted = _parse_datetime(expected.get(field))
            got = _parse_datetime(arguments.get(field))
            if wanted is None or got is None or got.tzinfo is None or got != wanted:
                reasons.append(f"get_events {field} does not match expected bound")
        expected_types = set(expected.get("event_types", []))
        supplied_types = arguments.get("event_types")
        actual_types = (
            {"task", "meeting", "reminder"} if supplied_types is None else set(supplied_types)
        )
        if actual_types != expected_types:
            reasons.append("get_events event_types do not match expected types")
    elif case.action == "search_events":
        if not expected_ids <= actual_ids:
            reasons.append("search_events omitted required result ID(s)")
        if expected.get("must_report_no_results"):
            if response.events:
                reasons.append("search_events returned events for a no-results case")
            if not _NO_RESULTS.search(response.reply_text):
                reasons.append("search_events reply does not clearly report no results")
    return reasons
