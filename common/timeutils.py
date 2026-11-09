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


def mentions_duration(text: str, start_at: datetime | None = None) -> bool:
    """Recognize length without interpreting 'в три часа' as duration."""
    amount = (
        r"(?:\d+(?:[,.]\d+)?|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|"
        r"девять|десять|одиннадцать|двенадцать|пятнадцать|двадцать|тридцать|сорок|"
        r"пятьдесят|шестьдесят|полтора|полторы)"
    )
    length = rf"(?:{amount}\s+(?:час\w*|мин\w*)|час(?:а)?|полчаса|четверть\s+часа)"
    if re.search(
        rf"\b(?:на|длится|продлится|длительность|продолжительность)\s+(?:около\s+)?{length}\b|"
        rf"(?:^|\n)\s*{length}\s*[.!]?\s*(?:$|\n)",
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


RECURRENCE_UNSUPPORTED = (
    "Повторяющиеся события пока не поддерживаются. Запись не создана и не изменена. "
    "Пришлите отдельный запрос с конкретной датой и временем."
)


def calendar_text(text: str) -> str:
    """Quoted titles are data, not scheduling instructions."""
    return re.sub(r'"[^"\n]*"|«[^»]*»|`[^`]*`', "", text)


def mentions_recurrence(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:ежеднев\w*|еженедел\w*|ежемесяч\w*|ежегод\w*)\b|"
            r"\bкажд\w*\s+(?:(?:\d+|втор\w*|трет\w*)\s+)?(?:день|дня|дней|"
            r"недел\w*|месяц\w*|год\w*|утро|вечер|понедельник\w*|вторник\w*|"
            r"сред[ау]\w*|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*)\b|"
            r"\bпо\s+(?:понедельникам|вторникам|средам|четвергам|пятницам|субботам|"
            r"воскресеньям|будням|выходным)\b",
            calendar_text(text),
            re.I,
        )
    )


def named_weekday(text: str) -> int | None:
    """Recognize a weekday with a preposition or as a standalone clarification."""
    text = calendar_text(text)
    forms = (
        r"понедельник\w*",
        r"вторник\w*",
        r"сред[ауые]\w*",
        r"четверг\w*",
        r"пятниц\w*",
        r"суббот\w*",
        r"воскресень\w*",
    )
    prefix = r"(?:\b(?:в|во|на|до|к|ко)\s+|^\s*)(?:(?:эт|следующ|ближайш)\w*\s+)?"
    found = {
        day for day, form in enumerate(forms) if re.search(prefix + form + r"\b", text, re.I | re.M)
    }
    return next(iter(found)) if len(found) == 1 else None


def resolve_calendar_start(text: str, start: datetime, now: datetime) -> datetime | None:
    """Resolve clear day/week instructions, preserving the proposed local clock time.

    Always compute the date, even when the model picked the right weekday in the
    wrong week. Explicit calendar dates and complex expressions stay with the
    existing validation/clarification flow.
    """
    text = calendar_text(text)
    # "в 10.00" is a clock time, not an explicit day/month.
    without_clock = re.sub(r"\bв\s+\d{1,2}[.:]\d{2}\b", "", text, flags=re.I)
    absolute = (
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}\b|"
        r"\b(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"
    )
    if re.search(absolute, without_clock, re.I):
        return None
    relative_days = set(re.findall(r"\b(сегодня|завтра|послезавтра|вчера)\b", text, re.I))
    relative_days = {day.casefold() for day in relative_days}
    if len(relative_days) > 1:
        raise ValueError("Названо несколько дней. Уточните конкретную дату события.")
    if relative_days:
        day = relative_days.pop()
        return (
            now + timedelta(days={"сегодня": 0, "завтра": 1, "послезавтра": 2, "вчера": -1}[day])
        ).replace(
            hour=start.hour, minute=start.minute, second=start.second, microsecond=start.microsecond
        )
    weekday = named_weekday(text)
    if weekday is None:
        return None
    explicit = (
        r"\b(?:через|прошл\w*|кажд\w*|месяц\w*|после|перед)\b|"
        r"\bследующ\w*\s+(?!недел)"
    )
    if re.search(explicit, text, re.I):
        return None
    days = (weekday - now.weekday()) % 7
    next_week = bool(re.search(r"следующ\w*\s+недел", text, re.I))
    this_week = bool(re.search(r"эт\w*\s+недел", text, re.I))
    if next_week and this_week:
        raise ValueError("Названо несколько недель. Уточните конкретную дату события.")
    if next_week or this_week:
        days = weekday - now.weekday() + (7 if next_week else 0)
    resolved = (now + timedelta(days=days)).replace(
        hour=start.hour, minute=start.minute, second=start.second, microsecond=start.microsecond
    )
    if resolved < now and not (this_week or next_week):
        resolved += timedelta(days=7)
    return resolved
