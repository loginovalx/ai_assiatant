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


def time_choice_buttons(choice_id: UUID) -> list[list[dict[str, str]]]:
    options = [
        {
            "type": "callback",
            "text": label,
            "payload": f"time-choice:{choice_id}:{value}",
        }
        for label, value in TIME_CHOICE_OPTIONS
    ]
    return [options[:3], options[3:]]


def parse_time_choice_payload(payload: str) -> tuple[UUID, str] | None:
    parts = payload.split(":")
    if len(parts) != 3 or parts[0] != "time-choice":
        return None
    try:
        choice_id = UUID(parts[1])
    except ValueError:
        return None
    if parts[2] not in {value for _, value in TIME_CHOICE_OPTIONS}:
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
        self, user_id: str, draft_id: UUID | None = None, *, choice_id: UUID | None = None
    ) -> UUID:
        choice_id = choice_id or uuid4()
        await self._redis.set(
            self._key(user_id),
            f"{choice_id}:{draft_id or ''}",
            ex=TIME_CHOICE_TTL_SECONDS,
        )
        return choice_id

    async def consume(
        self,
        user_id: str,
        choice_id: UUID,
        current_draft_id: UUID | None,
    ) -> bool:
        if not await self.matches(user_id, choice_id, current_draft_id):
            return False
        await self._redis.delete(self._key(user_id))
        return True

    async def matches(
        self,
        user_id: str,
        choice_id: UUID,
        current_draft_id: UUID | None,
    ) -> bool:
        raw = await self._redis.get(self._key(user_id))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str):
            return False
        try:
            stored_choice_id, stored_draft_id = raw.split(":", maxsplit=1)
        except ValueError:
            return False
        if stored_choice_id != str(choice_id):
            return False
        return stored_draft_id == str(current_draft_id or "")

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
