"""A retrieved calendar entry cannot authorize a new mutation on a read request."""

import re

_CHANGE_WORDS = {
    "create_event": (
        r"поставь(?:те)?|создай(?:те)?|добавь(?:те)?|запиши(?:те)?|"
        r"запланируй(?:те)?|напомни(?:те)?|создать|добавить|create|add|schedule"
    ),
    "update_event": (
        r"перенеси(?:те)?|измени(?:те)?|поменяй(?:те)?|замени(?:те)?|отложи(?:те)?|"
        r"перенести|изменить|поменять|заменить|отложить|reschedule|move|update"
    ),
    "delete_event": r"удали(?:те)?|убери(?:те)?|отмени(?:те)?|очисти(?:те)?|"
    r"удалить|убрать|отменить|очистить|delete|remove|cancel",
    "complete_event": r"заверши(?:те)?|выполни(?:те)?|отметь(?:те)?|пометь(?:те)?|закрой(?:те)?|"
    r"завершить|выполнить|отметить|пометить|закрыть|complete",
}


def explicitly_requests_change(source: str, tool: str) -> bool:
    """Authorize only the requested kind of change, using actual user text."""
    source = re.sub(r'"[^"\n]*"|«[^»]*»|`[^`]*`', "", source)
    kind = "delete_event" if tool == "delete_events" else tool
    words = _CHANGE_WORDS.get(kind)
    if words is None:
        return False
    if kind == "create_event" and re.match(
        r"^\s*(?:встреча|созвон|задача|напоминание)\b", source, re.I
    ):
        return True
    return bool(
        re.search(
            r"(?:^|[\n.!?,;:]\s*|\b(?:и|затем|потом)\s+)"
            r"(?:(?:пожалуйста|можешь|можете|нужно|надо|хочу|прошу)\s*,?\s*)*"
            rf"(?:{words})\b",
            source,
            re.I,
        )
    )


def asks_to_read(source: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:(?:привет|пожалуйста)[,!]?\s+)?(?:покажи|найди|найти|показать|список|"
            r"что\s+у\s+меня|какие|какая|какой|когда|есть\s+ли|show|find|list|search)\b",
            source,
            re.I,
        )
    )
