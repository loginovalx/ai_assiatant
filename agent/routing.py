"""Conservative local model selection; no paid classification request."""

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from agent.dialogue import starts_new_request
from agent.memory import HistoryMessage
from agent.policy import is_single_action_request
from agent.tools import ALLOWED_TOOLS, HIDDEN_ARGS, MUTATING_TOOLS
from common.timeutils import mentions_duration

_DAY = r"(?:сегодня|завтра|послезавтра)"
_CLOCK = r"(?:[01]?\d|2[0-3]):[0-5]\d"
_SIMPLE_READ = re.compile(
    r"(?:что\s+у\s+меня|покажи(?:\s+мне)?(?:\s+мои)?\s+"
    r"(?:задачи|встречи|напоминания|дела|расписание|календарь)|"
    r"какие(?:\s+у\s+меня)?\s+(?:задачи|встречи|напоминания|дела)(?:\s+у\s+меня)?)"
    r"\s+(?:на\s+)?(?:сегодня|завтра|этой\s+неделе|следующей\s+неделе)",
    re.I,
)
_CONTEXT_OR_COMPLEXITY = re.compile(
    r"\b(?:е[её]|его|их|эту|этот|это|ту|тот|предыдущ\w*|прошл\w*|"
    r"кажд\w*|ежеднев\w*|еженедел\w*|кроме|если|вместо|раньше|позже|"
    r"через|до|после|заранее|полчаса|утром|вечером|дн[её]м|ночью|вчера|позавчера|"
    r"понедельник\w*|вторник\w*|сред[ауые]\w*|четверг\w*|пятниц\w*|суббот\w*|"
    r"воскресень\w*|недел\w*|месяц\w*|январ\w*|феврал\w*|март\w*|апрел\w*|"
    r"ма[йя]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}\b",
    re.I,
)


@dataclass
class ModelRoute:
    model: str
    reason: str
    expected_tool: str | None = None


def select_model(
    text: str,
    history: list[HistoryMessage],
    *,
    primary_model: str,
    simple_model: str | None,
    editing_draft: bool = False,
) -> ModelRoute:
    if not simple_model or simple_model == primary_model:
        return ModelRoute(primary_model, "routing_disabled")
    last = history[-1] if history and history[-1]["role"] == "assistant" else {}
    if editing_draft or not starts_new_request(text):
        pending = last.get("pending_model")
        if pending or last.get("clarification_text") or editing_draft:
            # Old/unknown model metadata falls back to the primary model.
            model = pending if pending in {primary_model, simple_model} else primary_model
            expected = last.get("pending_tool")
            if model == simple_model and expected not in {"create_event", "get_events"}:
                model = primary_model
            return ModelRoute(
                model,
                "pending_dialogue",
                expected if expected in {"create_event", "get_events"} else None,
            )
    value = text.strip().rstrip(" .!?").casefold()
    if _SIMPLE_READ.fullmatch(value):
        return ModelRoute(simple_model, "simple_calendar_read", "get_events")
    if len(value) > 240 or _CONTEXT_OR_COMPLEXITY.search(value):
        return ModelRoute(primary_model, "complex_or_contextual")
    if not is_single_action_request(value, "create_event"):
        return ModelRoute(primary_model, "not_a_simple_creation")
    if re.fullmatch(rf"напомни(?:\s+мне)?\s+{_DAY}\s+в\s+{_CLOCK}\s+.+", value):
        # Exactly one day and clock excludes a second event's time.
        if len(re.findall(rf"\b{_DAY}\b", value)) == len(re.findall(_CLOCK, value)) == 1:
            return ModelRoute(simple_model, "simple_reminder", "create_event")
    if re.match(r"^(?:создай|добавь|запиши|поставь)(?:\s+мне)?\s+задачу\s+\S", value):
        if not re.search(r"\d|\b(?:сегодня|завтра|послезавтра|недел\w*|месяц\w*|в\s+\w+)\b", value):
            return ModelRoute(simple_model, "simple_undated_task", "create_event")
    if re.match(rf"^(?:(?:создай|добавь|поставь|запланируй)\s+)?(?:на\s+)?(?:{_DAY}\s+)?(?:встреч[ау]|созвон)\s+", value):
        if len(re.findall(rf"\b{_DAY}\b", value)) == len(
            re.findall(_CLOCK, value)
        ) == 1 and mentions_duration(value):
            return ModelRoute(simple_model, "simple_meeting", "create_event")
    return ModelRoute(primary_model, "ambiguous_creation")


def completion_problem(message: Any, tools: list[dict], expected_tool: str | None) -> str | None:
    """Preflight the entire proposed batch before any tool may run."""
    calls = message.tool_calls
    if not calls:
        if expected_tool or not (message.content or "").strip():
            return "missing_tool_or_reply"
        return None
    if expected_tool and len(calls) > 1:
        return "unexpected_batch"
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    for call in calls:
        name = call.function.name
        if name not in ALLOWED_TOOLS or (schemas and name not in schemas):
            return "unknown_tool"
        if expected_tool == "get_events" and name not in {"get_events", "ask_user"}:
            return "unexpected_tool"
        if name in MUTATING_TOOLS and expected_tool and name != expected_tool:
            return "unexpected_mutation"
        try:
            args = json.loads(call.function.arguments or "{}")
        except (ValueError, TypeError):
            return "invalid_json"
        if not isinstance(args, dict):
            return "invalid_arguments"
        if name in schemas:
            schema = {**schemas[name], "additionalProperties": False}
            required = set(schema.get("required", []))
            # The simplified schema loses optional nullable unions. Defer null
            # validation to MCP; service-owned arguments are never model-controlled.
            checked = {
                key: value
                for key, value in args.items()
                if key not in HIDDEN_ARGS and (value is not None or key in required)
            }
            if not Draft202012Validator(schema).is_valid(checked):
                return "invalid_tool_schema"
    return None
