import re
from datetime import datetime, timedelta
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
    # Проверяем слова-времени, включая прописные формы: «без четверти три»,
    # «половине шестого», «шестнадцать сорок пять», «восемь утра» — иначе гейт
    # отвергает записи, где модель честно извлекла время словами (регрессия T013).
    return bool(
        re.search(
            r"\b\d{1,2}[:.]\d{2}\b|\bв\s+\d{1,2}\b|"
            r"\bчерез\s+(?:полчаса|полминуты|четверть\s+часа)\b|"
            r"\bчерез\s+(?:(?:\d+|два|две|три|четыре|пять|полтора|полчаса)\s+)?(?:час|минут)|"
            r"\b(?:полдень|полночь)\b|"
            r"\bв\s+(?:час|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|"
            r"одиннадцать|двенадцать)\b|"
            r"\b(?:без\s+)?четверт(?:ь|и)\b|\bполовин|\bпол-?\w+го\b|"
            r"\b(?:ноль|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|"
            r"шестнадцать|семнадцать|восемнадцать|девятнадцать|двадцать|сорок|пятьдесят)\b|"
            r"\b(?:часов|часа|минут)\b",
            text,
            re.IGNORECASE,
        )
    )


def named_weekday(text: str) -> int | None:
    """An explicit single weekday after в/на/до/к; ranges remain ambiguous."""
    forms = (
        r"понедельник\w*",
        r"вторник\w*",
        r"сред[ауые]\w*",
        r"четверг\w*",
        r"пятниц\w*",
        r"суббот\w*",
        r"воскресень\w*",
    )
    prefix = r"\b(?:в|на|до|к|ко)\s+(?:(?:эт|следующ|ближайш)\w*\s+)?"
    found = {day for day, form in enumerate(forms) if re.search(prefix + form + r"\b", text, re.I)}
    return next(iter(found)) if len(found) == 1 else None


def resolve_weekday_mismatch(text: str, start: datetime, now: datetime) -> datetime | None:
    """Resolve an unqualified weekday from the server calendar, never an explicit date."""
    weekday = named_weekday(text)
    if weekday is None or start.weekday() == weekday:
        return None
    explicit = (
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}\b|"
        r"\b(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*|"
        r"\b(?:сегодня|завтра|послезавтра|через|прошл\w*|кажд\w*|месяц\w*|после|перед)\b|"
        r"\bследующ\w*\s+(?!недел)"
    )
    if re.search(explicit, text, re.I):
        return None
    days = (weekday - now.weekday()) % 7
    next_week = bool(re.search(r"следующ\w*\s+недел", text, re.I))
    this_week = bool(re.search(r"эт\w*\s+недел", text, re.I))
    if next_week or this_week:
        days = weekday - now.weekday() + (7 if next_week else 0)
    resolved = (now + timedelta(days=days)).replace(
        hour=start.hour, minute=start.minute, second=start.second, microsecond=start.microsecond
    )
    if resolved < now:
        if this_week:
            return None
        resolved += timedelta(days=7)
    return resolved
