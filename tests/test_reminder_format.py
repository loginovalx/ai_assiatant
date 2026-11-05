from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from bot.reminders import ReminderSender, format_notification
from common.schemas import Event, EventStatus, EventType


def make_event(start_at: datetime) -> Event:
    return Event(
        id=uuid4(),
        user_id="100500",
        type=EventType.MEETING,
        name="Созвон с Сашей",
        description="",
        start_at=start_at,
        status=EventStatus.ACTIVE,
        created_at=start_at,
        updated_at=start_at,
    )


def test_notification_converts_database_utc_to_configured_timezone():
    event = make_event(datetime(2026, 11, 5, 9, 30, tzinfo=UTC))

    assert format_notification(event, ZoneInfo("Europe/Moscow")) == (
        "Напоминание: встреча «Созвон с Сашей» в 12:30"
    )


def test_notification_keeps_existing_one_argument_call_compatible():
    local_time = datetime(2026, 11, 5, 12, 30, tzinfo=ZoneInfo("Europe/Moscow"))
    event = make_event(local_time)

    assert format_notification(event) == "Напоминание: встреча «Созвон с Сашей» в 12:30"


class RecordingRepo:
    def __init__(self, event: Event) -> None:
        self.event = event

    async def claim_due_events(self, *args, **kwargs) -> list[Event]:
        return [self.event]

    async def mark_notified(self, *args) -> None:
        pass

    async def unmark_notified(self, *args) -> None:
        pass


class RecordingMax:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, chat_id: int, text: str, **kwargs) -> None:
        self.sent.append(text)


async def test_sender_uses_its_configured_timezone_for_delivery():
    tz = ZoneInfo("Europe/Moscow")
    event = make_event(datetime(2026, 11, 5, 9, 30, tzinfo=UTC))
    api = RecordingMax()
    sender = ReminderSender(RecordingRepo(event), api, tz)

    assert await sender.tick(event.start_at.astimezone(tz) + timedelta(minutes=1)) == 1
    assert api.sent == ["Напоминание: встреча «Созвон с Сашей» в 12:30"]
