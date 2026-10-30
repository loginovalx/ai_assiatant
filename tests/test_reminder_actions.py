from datetime import datetime, timedelta

from bot.handler import BotHandler
from bot.reminders import ReminderSender
from bot.state import StateStore
from common.schemas import EventCreate


class Max:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def answer_callback(self, *args):
        pass


async def test_notification_offers_done_and_snooze(repo, tz):
    now = datetime.now(tz)
    event = await repo.create(
        EventCreate(
            user_id="777", type="reminder", name="Test", start_at=now - timedelta(minutes=1)
        )
    )
    api = Max()
    await ReminderSender(repo, api, tz).tick(now)
    buttons = api.sent[0][2]["buttons"][0]
    assert [b["payload"] for b in buttons] == [f"event:done:{event.id}", f"event:snooze:{event.id}"]


async def test_done_and_snooze_use_callback_owner(repo, redis, tz):
    now = datetime.now(tz)
    event = await repo.create(
        EventCreate(
            user_id="777", type="reminder", name="Test", start_at=now - timedelta(minutes=1)
        )
    )
    await repo.claim_due_events(now, timedelta(hours=1), 10)
    api = Max()
    bot = BotHandler(api, None, None, StateStore(redis), redis, repo)

    def callback(user, action):
        return {
            "update_type": "message_callback",
            "callback": {
                "callback_id": "c1",
                "user": {"user_id": user},
                "payload": f"event:{action}:{event.id}",
            },
        }

    await bot.handle_update(callback(888, "done"))
    assert (await repo.get("777", event.id)).status == "active"

    await bot.handle_update(callback(777, "snooze"))
    postponed = await repo.get("777", event.id)
    assert postponed.start_at > now + timedelta(minutes=59)
    assert await repo.claim_due_events(
        postponed.start_at + timedelta(seconds=1), timedelta(hours=1), 10
    )

    await bot.handle_update(callback(777, "done"))
    assert (await repo.get("777", event.id)).status == "done"
