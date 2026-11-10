"""Short lived per-user time button choices for clarification and draft edits."""

import re
from uuid import UUID, uuid4

from common.dialogue import asks_for_clock_time as asks_for_clock_time

TIME_CHOICE_TTL_SECONDS = 300
TIME_CHOICE_OPTIONS = (
    ("09:00", "0900"),
    ("12:00", "1200"),
    ("18:00", "1800"),
    ("Другое время", "manual"),
)


CHOICE_OPTIONS = {
    "time": TIME_CHOICE_OPTIONS,
    "day": (
        ("Пн", "mon"),
        ("Вт", "tue"),
        ("Ср", "wed"),
        ("Чт", "thu"),
        ("Пт", "fri"),
        ("Сб", "sat"),
        ("Вс", "sun"),
        ("Другая дата", "manual-day"),
    ),
    "duration": (
        ("30 минут", "30m"),
        ("60 минут", "60m"),
        ("120 минут", "120m"),
        ("Другая длительность", "manual-duration"),
    ),
}
CHOICE_REPLIES = {
    "0900": "В 09:00",
    "1200": "В 12:00",
    "1800": "В 18:00",
    "mon": "В понедельник",
    "tue": "Во вторник",
    "wed": "В среду",
    "thu": "В четверг",
    "fri": "В пятницу",
    "sat": "В субботу",
    "sun": "В воскресенье",
    "30m": "На 30 минут",
    "60m": "На 60 минут",
    "120m": "На 120 минут",
}

for _hour in range(1, 12):
    _choices = tuple((f"{h:02}:00", f"hour{h:02}") for h in (_hour, _hour + 12))
    CHOICE_OPTIONS[f"hour{_hour:02}"] = _choices
    CHOICE_REPLIES.update({value: f"В {label}" for label, value in _choices})


def clock_alternatives_kind(question: str) -> str | None:
    match = re.fullmatch(r"Во сколько запланировать: в (\d{2}):00 или в (\d{2}):00\?", question)
    if match:
        first, second = map(int, match.groups())
        if 1 <= first <= 11 and second == first + 12:
            return f"hour{first:02}"
    return None


def choice_kind(selection: str) -> str | None:
    return next(
        (
            kind
            for kind, options in CHOICE_OPTIONS.items()
            if any(value == selection for _, value in options)
        ),
        None,
    )


def time_choice_buttons(choice_id: UUID, kind: str = "time") -> list[list[dict[str, str]]]:
    options = [
        {
            "type": "callback",
            "text": label,
            "payload": f"time-choice:{choice_id}:{value}",
        }
        for label, value in CHOICE_OPTIONS[kind]
    ]
    if kind == "day":
        return [options[:4], options[4:7], options[7:]]
    if len(options) <= 3:
        return [options]
    return [options[:3], options[3:]]


def parse_time_choice_payload(payload: str) -> tuple[UUID, str] | None:
    parts = payload.split(":")
    if len(parts) != 3 or parts[0] != "time-choice":
        return None
    try:
        choice_id = UUID(parts[1])
    except ValueError:
        return None
    if choice_kind(parts[2]) is None:
        return None
    return choice_id, parts[2]


class TimeChoiceStore:
    """One active, one use chooser per user, bound to their current draft if any."""

    def __init__(self, redis) -> None:
        self._redis = redis

    @staticmethod
    def _key(user_id: str) -> str:
        return f"bot:time-choice:{user_id}"

    async def issue(
        self,
        user_id: str,
        draft_id: UUID | None = None,
        *,
        choice_id: UUID | None = None,
        kind: str = "time",
    ) -> UUID:
        choice_id = choice_id or uuid4()
        await self._redis.set(
            self._key(user_id),
            self._value(choice_id, draft_id, kind),
            ex=TIME_CHOICE_TTL_SECONDS,
        )
        return choice_id

    async def consume(
        self,
        user_id: str,
        choice_id: UUID,
        current_draft_id: UUID | None,
        *,
        kind: str = "time",
    ) -> bool:
        if not await self.matches(user_id, choice_id, current_draft_id, kind=kind):
            return False
        await self._redis.delete(self._key(user_id))
        return True

    async def matches(
        self,
        user_id: str,
        choice_id: UUID,
        current_draft_id: UUID | None,
        *,
        kind: str = "time",
    ) -> bool:
        raw = await self._redis.get(self._key(user_id))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw == self._value(choice_id, current_draft_id, kind)

    @staticmethod
    def _value(choice_id: UUID, draft_id: UUID | None, kind: str) -> str:
        suffix = "" if kind == "time" else f":{kind}"
        return f"{choice_id}:{draft_id or ''}{suffix}"

    async def clear(self, user_id: str) -> None:
        await self._redis.delete(self._key(user_id))


def is_explicit_keep_confirmation(text: str) -> bool:
    """Recognize only direct keep-the-current-draft phrases, not broad negations."""
    words = " ".join(re.findall(r"[\w]+", text.casefold().replace("ё", "е")))
    return words in {
        "нет время подходит",
        "все равно на пересечение",
        "оставь как есть",
        "оставь все как есть",
    }
