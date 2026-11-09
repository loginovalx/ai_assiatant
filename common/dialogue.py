"""Shared clarification cues; they never authorize calendar mutations."""

import re

from common.timeutils import allows_default_duration, mentions_duration, mentions_start_time

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
            or re.search(
                r"\b(?:какой\s+день|какую\s+дату|о\s+ч[её]м\s+напомнить|как\s+назвать)\b",
                text,
                re.I,
            )
        )
    )


def is_person_appointment(text: str) -> bool:
    value = re.sub(r'"[^"\n]*"|«[^»]*»|`[^`]*`', "", text)
    if re.search(r"\b(?:задач\w*|напомни\w*|подготов\w*)\b", value, re.I):
        return False
    return bool(
        re.search(
            r"\b(?:встреч[ауе]|созвон|заняти[еяю]|урок|консультаци[яю]|тренировк[ау])"
            r"\s+(?:с|со|у)\s+\w+",
            value,
            re.I,
        )
    )


def clarification_text(question: str, source: str) -> str:
    """Ask both missing fields together, while retaining an already named field."""
    if not is_person_appointment(source):
        return question
    if re.search(r"\b(?:какой\s+день|какую\s+дату)\b", question, re.I):
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
