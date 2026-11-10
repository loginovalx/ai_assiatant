"""Calendar rules and the supported Russian commands for repeating reminders."""

import re
from datetime import datetime, time, timedelta
from uuid import UUID

from pydantic import BaseModel, Field

DAYS = ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье")
_DAY_FORMS = (
    "понедельник(?:ам)?",
    "вторник(?:ам)?",
    "сред[уа]м?",
    "четверг(?:ам)?",
    "пятниц[уа]м?",
    "суббот[уа]м?",
    "воскресень[ея]м?",
)
CLOCK_HOUR_WORDS = (
    "ноль",
    "час",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
    "двадцать",
    "двадцать один",
    "двадцать два",
    "двадцать три",
)
_HOUR = rf"(?:\d{{1,2}}|{'|'.join(reversed(CLOCK_HOUR_WORDS))})"
_HOUR_UNIT = r"(?:\s+час(?:а|ов)?)?"
_PART = r"(?:утра|дня|вечера|ночи|утром|дн[её]м|вечером|ночью)"
CLOCK_EXPRESSION = (
    rf"{_HOUR}(?:[:. ]\d{{2}}|\s+ноль\s+ноль|{_HOUR_UNIT})(?:\s+{_PART})?"
    rf"\b(?![:.]\d|\d|\s+(?:\d{{2}}|час(?:а|ов)?|{_PART})\b)"
)
_CLOCK_IN_TEXT = re.compile(rf"\bв\s+{CLOCK_EXPRESSION}", re.I)
_CADENCE = rf"(?:каждый день|ежедневно|(?:кажд(?:ый|ую|ое)|по) (?:{'|'.join(_DAY_FORMS)}))"
_VERB = r"(?:напомни(?:те)?|напоминай(?:те)?|напоминать)"
_CREATE = rf"(?:{_VERB}\s+(?:мне\s+)?|"
_CREATE += r"(?:поставь|создай|добавь|запланируй)(?:\s+мне)?\s+напоминание\s+|(?:я\s+)?хочу\s+)"
_SCHEDULE = rf"(?P<cadence>{_CADENCE})\s*[,—]?\s+в\s+(?P<clock>{CLOCK_EXPRESSION})"
_COMMANDS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"{_CREATE}(?:на\s+)?{_SCHEDULE}\s*[,—:]?\s+(?P<name>.+)",
        rf"{_SCHEDULE}\s+{_VERB}\s+(?:мне\s+)?(?P<name>.+)",
        rf"{_CREATE}(?P<name>.+?)\s+(?:на\s+)?{_SCHEDULE}",
        rf"{_CREATE}(?:на\s+)?(?P<cadence>{_CADENCE})\s+(?P<name>.+?)\s+в\s+"
        rf"(?P<clock>{CLOCK_EXPRESSION})",
    )
)
_REPEATING = re.compile(
    rf"\b(?:{_CADENCE}|кажд\w*\s+(?:\d+|день|недел\w*|месяц\w*|год\w*|будн\w*)"
    r"|каждые\s+\w+\s+(?:дн\w*|недел\w*|месяц\w*|год\w*)"
    r"|еженедель\w*|ежемесяч\w*|ежегод\w*|по будням|по выходным)\b",
    re.I,
)


class ReminderRule(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    weekday: int | None = Field(default=None, ge=0, le=6)
    clock: time

    def first_after(self, now: datetime) -> datetime:
        first = datetime.combine(now.date(), self.clock, tzinfo=now.tzinfo)
        if self.weekday is not None:
            first += timedelta(days=(self.weekday - first.weekday()) % 7)
        if first <= now:
            first += timedelta(days=7 if self.weekday is not None else 1)
        return first

    @property
    def cadence_label(self) -> str:
        return (
            "Каждый день"
            if self.weekday is None
            else (
                "Каждый понедельник",
                "Каждый вторник",
                "Каждую среду",
                "Каждый четверг",
                "Каждую пятницу",
                "Каждую субботу",
                "Каждое воскресенье",
            )[self.weekday]
        )

    @property
    def schedule(self) -> str:
        return f"{self.cadence_label} в {self.clock:%H:%M}"


class SavedReminderRule(ReminderRule):
    id: UUID
    revision: UUID


def parse_clock(value: str) -> time | None:
    value = re.sub(r"^в\s+", "", " ".join(value.casefold().split()).strip(" .!?"))
    match = re.fullmatch(
        rf"(?P<hour>{_HOUR})(?:[:. ](?P<minute>\d{{2}})|\s+ноль\s+ноль|{_HOUR_UNIT})"
        rf"(?:\s+(?P<part>{_PART}))?",
        value,
    )
    if not match:
        return None
    raw_hour = match["hour"]
    hour = int(raw_hour) if raw_hour.isdecimal() else CLOCK_HOUR_WORDS.index(raw_hour)
    minute = int(match["minute"] or 0)
    if hour > 23 or minute > 59:
        return None
    if match["part"] in {"дня", "вечера", "днем", "днём", "вечером"} and hour < 12:
        hour += 12
    if match["part"] in {"ночи", "ночью"} and hour == 12:
        hour = 0
    return time(hour, minute)


def complete_clock(text: str, clock: time) -> str:
    """Fill or correct the clock in a pending recurring request."""
    value = f"в {clock:%H:%M}"
    updated, count = _CLOCK_IN_TEXT.subn(value, text, count=1)
    return updated if count else f"{text.rstrip(' .!?')} {value}"


def clock_needs_daypart(value: str, clock: time) -> bool:
    return 1 <= clock.hour <= 11 and bool(re.fullmatch(rf"{_HOUR}{_HOUR_UNIT}", value, re.I))


def parse_command(text: str) -> ReminderRule | None:
    parsed = parse_request(text)
    return parsed[0] if parsed else None


def parse_request(text: str) -> tuple[ReminderRule, bool] | None:
    """Return the recurring rule and whether its hour needs morning/evening clarification."""
    source = " ".join(text.split()).strip(" .!?")
    if not _REPEATING.search(re.sub(r'«[^»]*»|"[^"]*"', "", source)):
        return None
    if re.match(r"^(?:что|покажи|найди|какие|когда|во сколько|удали|отмени)\b", source, re.I):
        return None
    match = next((match for pattern in _COMMANDS if (match := pattern.fullmatch(source))), None)
    if not match:
        raise ValueError
    name = match["name"].strip(' «»"')
    if (
        not name
        or len(name) > 200
        or re.search(rf"\b(?:до\s+(?:\d|конца)|на\s+\d+\s+недел|{_CADENCE})", name, re.I)
    ):
        raise ValueError
    day = next(
        (
            i
            for i, form in enumerate(_DAY_FORMS)
            if re.search(rf"\b{form}\b", match["cadence"], re.I)
        ),
        None,
    )
    clock = parse_clock(match["clock"])
    if clock is None:
        raise ValueError
    return ReminderRule(name=name, weekday=day, clock=clock), clock_needs_daypart(
        match["clock"], clock
    )
