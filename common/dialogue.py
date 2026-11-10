"""Shared clarification cues; they never authorize calendar mutations."""

import re

from common.timeutils import (
    allows_default_duration,
    ambiguous_hour,
    has_explicit_weekday_or_date,
    has_meeting_day,
    mentions_duration,
    mentions_start_time,
    mentions_week_range,
    requested_clock,
)

MEETING_DAY_QUESTION = "На какой день запланировать встречу?"
PAST_TIME_QUESTION = (
    "Это время уже прошло. Во сколько запланировать событие? Можно указать другую дату."
)

_APPOINTMENT_WORD = r"(?:встре?ч[ауе]|созвон|заняти[еяю]|урок|консультаци[яю]|тренировк[ау])"

_CLOCK_QUESTION = re.compile(
    r"\bво\s+сколько\b|\b(?:в|на)\s+какое\s+(?:точное\s+)?время\b|"
    r"\bкакое\s+(?:точное\s+)?время\b|"
    r"\b(?:назови|назовите|укажи|укажите|уточни|уточните|напиши|напишите)"
    r"\s+(?:пожалуйста\s+)?(?:точное\s+)?время\b|\bдат[ау]\s+и\s+(?:точное\s+)?время\b",
    re.I,
)
_DURATION_QUESTION = re.compile(
    r"\bдлительност\w*\b|\bсколько\b.*\b(?:продлит\w*|длит\w*|минут\w*|час\w*)\b",
    re.I,
)


def asks_for_clock_time(text: str) -> bool:
    return bool(_CLOCK_QUESTION.search(text))


def asks_for_duration(text: str) -> bool:
    return bool(_DURATION_QUESTION.search(text))


def asks_for_day(text: str) -> bool:
    return bool(re.search(r"\b(?:какой\s+день|какую\s+дату)\b", text, re.I))


def is_clarification_question(text: str) -> bool:
    if asks_for_clock_time(text):
        return True
    question = "?" in text or re.search(
        r"\b(?:сколько|укажите|напишите|уточните|назовите)\b", text, re.I
    )
    return bool(
        question
        and (
            asks_for_clock_time(text)
            or asks_for_duration(text)
            or asks_for_day(text)
            or re.search(
                r"\b(?:о\s+ч[её]м\s+напомнить|как\s+назвать)\b",
                text,
                re.I,
            )
        )
    )


def _appointment_text(text: str) -> str:
    value = re.sub(r'"[^"\n]*"|«[^»]*»|`[^`]*`', "", text)
    if re.search(r"\b(?:задач\w*|напомни\w*|подготов\w*)\b", value, re.I):
        return ""
    return value


def is_person_appointment(text: str) -> bool:
    return bool(
        re.search(rf"\b{_APPOINTMENT_WORD}\s+(?:с|со|у)\s+\w+", _appointment_text(text), re.I)
    )


def is_appointment_request(text: str) -> bool:
    return bool(re.search(rf"\b{_APPOINTMENT_WORD}\b", _appointment_text(text), re.I))


def clarification_text(question: str, source: str) -> str:
    """Ask both missing fields together, while retaining an already named field."""
    if "уже прошло" in question.casefold():
        return question
    if (
        (asks_for_day(question) or asks_for_clock_time(question) or asks_for_duration(question))
        and mentions_week_range(source)
        and not has_explicit_weekday_or_date(source)
    ):
        if re.search(r"следующ\w*\s+недел", source, re.I):
            return "На какой день на следующей неделе запланировать?"
        return "На какой день на этой неделе запланировать?"
    appointment = _appointment_text(source)
    if (
        (asks_for_clock_time(question) or asks_for_duration(question))
        and is_appointment_request(source)
        and not re.search(r"\b(?:перенес|отлож|измени|поменя|продли|сократи)\w*", appointment, re.I)
        and not has_meeting_day(appointment)
    ):
        return MEETING_DAY_QUESTION
    exact_clock = requested_clock(source)
    appointment_has_clock = exact_clock is not None and is_appointment_request(source)
    if asks_for_clock_time(question) and (hour := ambiguous_hour(source)) is not None:
        return f"Во сколько запланировать: в {hour:02}:00 или в {hour + 12:02}:00?"
    if not is_person_appointment(source) and not appointment_has_clock:
        return question
    if asks_for_day(question):
        return question
    if not (asks_for_clock_time(question) or asks_for_duration(question)):
        return question
    clock = mentions_start_time(source)
    length = mentions_duration(source) or allows_default_duration(source)
    if not clock and not length:
        return "Во сколько начнётся встреча и сколько она продлится?"
    if not clock:
        return "Во сколько начнётся встреча?"
    if not length:
        return "Сколько продлится встреча?"
    return question
