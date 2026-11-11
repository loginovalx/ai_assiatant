"""Ground simple voice creations in fields that can be read directly from the transcript."""

import re
from datetime import datetime, timedelta

from agent.policy import is_single_action_request
from common.reminder_series import CLOCK_EXPRESSION, clock_needs_daypart, parse_clock
from common.schemas import MAX_DURATION_MINUTES

_DAYS = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
_DAY = rf"(?:на\s+)?(?P<day>{'|'.join(_DAYS)})"
_TIME = rf"(?:в|на)\s+(?P<clock>{CLOCK_EXPRESSION})"
_MEETING_VERB = r"(?:назначь|поставь|создай|добавь|запланируй)"
_MEETING_KIND = r"(?:встречу|созвон)"
_PERSON = r"с(?:о)?\s+[а-яё-]+(?:\s+[а-яё-]+){0,2}"
_MEETING_NAME = rf"(?P<name>{_MEETING_KIND}(?:\s+{_PERSON})?)"
_LENGTH = r"на\s+(?P<duration>.+?)"
_MEETINGS = tuple(
    re.compile(rf"{_MEETING_VERB}\s+{body}", re.I)
    for body in (
        rf"{_MEETING_NAME}\s+{_DAY}\s+{_TIME}\s+{_LENGTH}",
        rf"{_DAY}\s+{_MEETING_NAME}\s+{_TIME}\s+{_LENGTH}",
        rf"{_MEETING_NAME}\s+{_TIME}\s+{_LENGTH}\s+{_DAY}",
        rf"(?P<kind>{_MEETING_KIND})\s+{_DAY}\s+(?P<person>{_PERSON}),?\s+"
        rf"{_TIME}(?:\s+{_LENGTH})?",
    )
)
_REMINDERS = tuple(
    re.compile(rf"напомни(?:\s+мне)?\s+{body}", re.I)
    for body in (
        rf"{_DAY}\s+{_TIME}[,:]?\s+(?P<name>.+)",
        rf"{_DAY}\s+(?P<name>.+?)\s+{_TIME}",
        rf"(?P<name>.+?)\s+{_DAY}\s+{_TIME}",
    )
)
_DURATIONS = {"полчаса": 30, "час": 60, "полтора часа": 90, "два часа": 120, "четверть часа": 15}
_MINUTES = {
    "пять": 5,
    "десять": 10,
    "пятнадцать": 15,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "сорок пять": 45,
    "пятьдесят": 50,
    "шестьдесят": 60,
}


def grounded_voice_create(source: str, reference: datetime) -> dict | None:
    """Read explicit fields from a single spoken command."""
    value = " ".join(line.strip(" .!?") for line in source.splitlines())
    value = " ".join(value.split())
    if not is_single_action_request(value, "create_event"):
        return None
    meeting = next((match for pattern in _MEETINGS if (match := pattern.fullmatch(value))), None)
    match = meeting or next(
        (match for pattern in _REMINDERS if (match := pattern.fullmatch(value))), None
    )
    if not match:
        return None
    clock = parse_clock(match["clock"])
    if clock is None or clock_needs_daypart(match["clock"], clock):
        return None
    duration = None
    if meeting and meeting["duration"] is not None:
        length = meeting["duration"].casefold()
        minutes = re.fullmatch(r"(.+?)\s+минут(?:у|ы)?", length)
        if minutes:
            count = minutes[1]
            duration = int(count) if count.isdecimal() and len(count) <= 3 else _MINUTES.get(count)
        else:
            duration = _DURATIONS.get(length)
        if duration is None or not 1 <= duration <= MAX_DURATION_MINUTES:
            return None
    fields = match.groupdict()
    name = fields.get("name") or f"{fields['kind']} {fields['person']}"
    name = name.strip(' «»"')
    if not name or len(name) > 200:
        return None
    if meeting:
        name = re.sub(r"^встречу\b", "Встреча", name, flags=re.I)
    name = name[0].upper() + name[1:]
    day = reference.date() + timedelta(days=_DAYS[match["day"].casefold()])
    return {
        "event_type": "meeting" if meeting else "reminder",
        "name": name,
        "start_at": datetime.combine(day, clock, reference.tzinfo).isoformat(),
        "duration_minutes": duration,
    }
