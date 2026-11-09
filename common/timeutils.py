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


_DURATION_AMOUNT = (
    r"(?:\d+(?:[,.]\d+)?|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|"
    r"девять|десять|одиннадцать|двенадцать|пятнадцать|двадцать|тридцать|сорок|"
    r"пятьдесят|шестьдесят|полтора|полторы)"
)
_DURATION_LENGTH = rf"(?:{_DURATION_AMOUNT}\s+(?:час\w*|мин\w*)|час(?:а)?|полчаса|четверть\s+часа)"


def mentions_duration(text: str, start_at: datetime | None = None) -> bool:
    """Recognize length without interpreting 'в три часа' as duration."""
    length = _DURATION_LENGTH
    if re.search(
        rf"\b(?:на|длится|продлится|длительность|продолжительность)\s+(?:около\s+)?{length}\b|"
        rf"(?:^|[\n,;])\s*{length}\s*[.!]?\s*(?:$|\n)",
        text,
        re.I,
    ):
        return True
    # A range for a different existing meeting is not evidence for this meeting's length.
    return any(
        start_at is None or (int(match[1]), int(match[2] or 0)) == (start_at.hour, start_at.minute)
        for match in re.finditer(
            r"\bс\s+(\d{1,2})(?::(\d{2}))?\s+до\s+\d{1,2}(?::\d{2})?\b", text, re.I
        )
    )


def allows_default_duration(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\n)\s*(?:не\s+знаю|не\s*важно|как\s+обычно|сам\s+реши)\s*[.!]?\s*(?:$|\n)",
            text,
            re.I,
        )
    )


def mentions_start_time(text: str) -> bool:
    # An explicit length such as "на 30 минут" cannot supply the missing clock.
    without_length = re.sub(
        rf"\b(?:на|длится|продлится|длительность|продолжительность)\s+(?:около\s+)?{_DURATION_LENGTH}\b",
        "",
        text,
        flags=re.I,
    )
    without_length = re.sub(
        rf"(?:^|[\n,;])\s*{_DURATION_AMOUNT}\s+мин\w*\s*[.!]?\s*(?=$|\n)",
        "",
        without_length,
        flags=re.I,
    )
    return mentions_time(without_length)


def mentions_time(text: str) -> bool:
    # Проверяем слова-времени, включая прописные формы: «без четверти три»,
    # «половине шестого», «шестнадцать сорок пять», «восемь утра» — иначе гейт
    # отвергает записи, где модель честно извлекла время словами (регрессия T013).
    return bool(
        re.search(
            r"\b\d{1,2}[:.]\d{2}\b|\bв\s+\d{1,2}\b|"
            r"\bс\s+\d{1,2}(?::\d{2})?\s+до\s+\d{1,2}(?::\d{2})?\b|"
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
