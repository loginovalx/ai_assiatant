import re
from datetime import datetime
from zoneinfo import ZoneInfo

WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


def ensure_aware(dt: datetime, tz: ZoneInfo) -> datetime:
    """Время без часового пояса считаем временем пользователя (tz), а не UTC.

    LLM часто присылает «2026-10-30T10:00:00» без смещения. Если записать это как UTC,
    напоминание «в 10 утра» сработает в 13:00 по Москве.
    """
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def mentions_time(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}[:.]\d{2}\b|\bв\s+\d{1,2}\b|"
            r"\bчерез\s+(?:(?:\d+|два|две|три|четыре|пять|полтора|полчаса)\s+)?(?:час|минут)|"
            r"\b(?:утром|днём|днем|вечером|полдень|полночь|обед)\b|"
            r"\bв\s+(?:час|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать)\b",
            text,
            re.IGNORECASE,
        )
    )
