"""Boundaries between a new request and a reply to a clarification."""

import re


def starts_new_request(text: str) -> bool:
    text = text.casefold().strip()
    if re.match(r"^(?:лучше|давай|нет[, ]|на другое|в другое)", text):
        return False
    if re.match(r"^(?:поставь|перенеси|измени|поменяй)\s+(?:на|в|е[её]|его|это|другое)\b", text):
        return False
    if re.search(r"\b(?:встреч[ау]|созвон|задач[ау]|напоминание)\b", text) and re.search(
        r"\b(?:поставь|создай|добавь|запланируй|назначь|перенеси)\b", text
    ):
        return True
    return bool(
        re.match(
            r"^(?:(?:ок|хорошо|пожалуйста)[, ]+)?(?:напомни|поставь|создай|добавь|назначь|удали|"
            r"отмени|перенеси|покажи|найди|встреча|созвон|задача|что у меня)\b",
            text,
        )
    )


def direct_reply(text: str) -> tuple[str, bool] | None:
    value = text.casefold().strip(" .!?")
    if value in {"забей", "не надо", "отмена", "отмени", "забыли"}:
        return "Отменил текущую просьбу. Календарь не изменён.", True
    if re.search(r"(?:миллион|миллиард|тысяч)\w*\s+лет", value):
        return "Укажите реальную дату и время напоминания.", False
    numeric_date = re.search(r"\bна\s+(\d{1,2})\s+(\d{1,2})\b", value)
    if numeric_date and (
        not 1 <= int(numeric_date[1]) <= 31 or not 1 <= int(numeric_date[2]) <= 12
    ):
        return "Уточните дату: напишите день, название месяца и год, например «25 мая 2027».", False
    if re.search(r"(?:инструмент|тул|промпт|system prompt)", value) and re.search(
        r"(?:покажи|напиши|выдай|раскрой|какие|параметр)", value
    ):
        return (
            "Внутренние инструкции и параметры инструментов не раскрываю. Помогу с календарём.",
            False,
        )
    if re.search(r"(?:файл|csv)", value) and re.search(r"(?:кин|присл|отправ|загруз|можно)", value):
        return (
            "Можно прислать CSV-файл с колонками id, date, time, text — "
            "загружу события в ваш календарь. Другие форматы пока не поддерживаются.",
            False,
        )
    if re.search(r"\bв\s+\d{1,2}(?::\d{2})?\s+и\s+\d{1,2}(?::\d{2})?\b", value):
        return (
            "Это две отдельные встречи? Пришлите каждую отдельным сообщением "
            "с датой, временем и длительностью.",
            False,
        )
    return None
