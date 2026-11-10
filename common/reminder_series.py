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
_CLOCK = r"(?:[01]?\d|2[0-3]):[0-5]\d"
_CADENCE = rf"(?:каждый день|ежедневно|(?:кажд(?:ый|ую|ое)|по) (?:{'|'.join(_DAY_FORMS)}))"
_VERB = r"(?:напомни(?:те)?|напоминай(?:те)?|напоминать)"
_CREATE = rf"(?:{_VERB}\s+(?:мне\s+)?|"
_CREATE += r"(?:поставь|создай|добавь|запланируй)(?:\s+мне)?\s+напоминание\s+)"
_SCHEDULE = rf"(?P<cadence>{_CADENCE})\s+в\s+(?P<clock>{_CLOCK})"
_COMMANDS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"{_CREATE}(?:на\s+)?{_SCHEDULE}\s*[,—:]?\s+(?P<name>.+)",
        rf"{_SCHEDULE}\s+{_VERB}\s+(?:мне\s+)?(?P<name>.+)",
        rf"{_CREATE}(?P<name>.+?)\s+(?:на\s+)?{_SCHEDULE}",
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
    def schedule(self) -> str:
        day = (
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
        return f"{day} в {self.clock:%H:%M}"


class SavedReminderRule(ReminderRule):
    id: UUID
    revision: UUID


def parse_clock(value: str) -> time | None:
    match = re.fullmatch(_CLOCK, value.strip())
    if not match:
        return None
    hour, minute = map(int, match[0].split(":"))
    return time(hour, minute)


def parse_command(text: str) -> ReminderRule | None:
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
    return ReminderRule(name=name, weekday=day, clock=time.fromisoformat(match["clock"].zfill(5)))
