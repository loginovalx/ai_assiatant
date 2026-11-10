"""Real bot → agent HTTP → MCP flow, with only model and storage substituted.

The scripted model deliberately returns imperfect types and ordinary questions.
These tests verify confirmation/state contracts, not live GigaChat language quality.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastmcp import FastMCP

from agent.core import Agent
from agent.main import app, get_agent
from agent.memory import ConversationMemory
from agent.service import AgentService
from agent.tools import McpToolbox
from bot.agent_client import AgentClient
from bot.handler import BotHandler
from bot.state import StateStore
from common.dialogue import is_person_appointment
from common.repository import EventRepository
from common.schemas import Event, EventCreate, EventStatus
from common.timeutils import ambiguous_hour, mentions_start_time, requested_clock
from mcp_server import server as calendar
from tests.scripted_llm import ScriptedLLM, text, tool_call
from tests.test_time_choice_buttons import FakeMax, all_buttons, callback, text_message


class Repository:
    """Owner-scoped in-memory records; real MCP validation remains in the loop."""

    creation_id = staticmethod(EventRepository.creation_id)

    def __init__(self, tz):
        self.rows = {}
        self.tz = tz
        self.writes = 0

    async def get(self, owner, event_id):
        event = self.rows.get(event_id)
        return event if event is not None and event.user_id == owner else None

    async def create(self, data, idempotency_key=None):
        event_id = self.creation_id(data.user_id, idempotency_key) if idempotency_key else uuid4()
        if event_id not in self.rows:
            now = datetime.now(self.tz)
            self.rows[event_id] = Event(
                id=event_id, **data.model_dump(), status="active", created_at=now, updated_at=now
            )
            self.writes += 1
        return self.rows[event_id]

    async def find_conflicts(self, event):
        if event.end_at is None:
            return []
        return [
            other
            for other in self.rows.values()
            if other.user_id == event.user_id
            and other.id != event.id
            and other.status == "active"
            and other.end_at is not None
            and other.start_at < event.end_at
            and event.start_at < other.end_at
        ]

    async def preview_cancellations(self, owner, types, start, end):
        return [
            event
            for event in self.rows.values()
            if event.user_id == owner
            and event.status != "cancelled"
            and (types is None or event.type in types)
            and (start is None or (event.start_at is not None and start <= event.start_at < end))
        ]

    async def set_status(self, owner, event_id, status):
        event = await self.get(owner, event_id)
        if event is not None and event.status != status:
            event = event.model_copy(update={"status": status})
            self.rows[event_id] = event
            self.writes += 1
        return event

    async def update(self, owner, event_id, **changes):
        event = await self.get(owner, event_id)
        if event is not None:
            event = event.model_copy(
                update={key: value for key, value in changes.items() if value is not None}
            )
            self.rows[event_id] = event
            self.writes += 1
        return event

    async def cancel_selected(self, owner, event_ids, types, start, end):
        allowed = {event.id for event in await self.preview_cancellations(owner, types, start, end)}
        return [
            await self.set_status(owner, item, EventStatus.CANCELLED)
            for item in event_ids
            if item in allowed
        ]


@pytest.fixture
def session(redis, tz, monkeypatch):
    repo = Repository(tz)

    @asynccontextmanager
    async def lifespan(server):
        yield {"repo": repo}

    mcp = FastMCP("confirmation-regression", lifespan=lifespan)
    for function in (
        calendar.create_event,
        calendar.update_event,
        calendar.delete_event,
        calendar.delete_events,
    ):
        mcp.tool(function)
    original = httpx.AsyncClient

    def build(*script):
        model = ScriptedLLM(*script)
        agent = Agent(
            model, "GigaChat-2-Max", McpToolbox(mcp), ConversationMemory(redis, 10, 3600), tz
        )
        service = AgentService(agent, redis)
        monkeypatch.setitem(app.dependency_overrides, get_agent, lambda: service)
        monkeypatch.setattr(
            "bot.agent_client.httpx.AsyncClient",
            lambda **kwargs: original(transport=httpx.ASGITransport(app=app), **kwargs),
        )
        api = FakeMax()
        bot = BotHandler(api, None, AgentClient("http://agent"), StateStore(redis), redis)

        async def message(value, voice=False):
            update = text_message(value, uuid4().hex)
            if voice:
                body = update["message"]["body"]
                body.pop("text")
                body["attachments"] = [{"type": "audio", "payload": {"transcription": value}}]
            await bot.handle_update(update)

        async def click(label=None, payload=None):
            payload = payload or next(
                b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == label
            )
            await bot.handle_update(callback(payload, callback_id=uuid4().hex))

        return bot, api, message, click, model, service

    return build, repo


async def seed(repo, owner="777", name="Запись"):
    return await repo.create(EventCreate(user_id=owner, type="task", name=name))


async def test_bulk_delete_no_and_old_button_never_write(session):
    build, repo = session
    await seed(repo)
    bot, api, message, click, _, _ = build(tool_call("delete_events", {"confirmed": True}))
    await message("Удали все мои записи")
    assert "Вы уверены" in api.sent[-1]["text"]
    assert [b["text"] for b in all_buttons(api.sent[-1])] == ["Удалить", "Нет"]
    delete = all_buttons(api.sent[-1])[0]["payload"]
    assert repo.writes == 1
    await click("Нет")
    await click(payload=delete)
    assert repo.writes == 1
    assert await bot.drafts.load("777") is None
    assert "устарел" in api.answered[-1][1]


async def test_bulk_approval_deletes_only_approved_snapshot_once(session):
    build, repo = session
    first = await seed(repo, name="Оплатить интернет")
    second = await repo.create(
        EventCreate(
            user_id="777",
            type="reminder",
            name="Позвонить в сервис",
            start_at=datetime(2030, 11, 12, 9, tzinfo=repo.tz),
        )
    )
    foreign = await seed(repo, "other", "Чужая задача")
    _, api, message, click, model, _ = build(tool_call("delete_events", {"confirmed": True}))
    await message("Удали все мои записи")
    preview = api.sent[-1]["text"]
    assert "удалить 2 записи?" in preview
    assert "Задача «Оплатить интернет» — без даты" in preview
    assert "Напоминание «Позвонить в сервис» — 12 ноября 2030, вторник, в 09:00" in preview
    assert "Чужая задача" not in preview
    assert len(model.requests) == 1
    delete = all_buttons(api.sent[-1])[0]["payload"]
    assert repo.writes == 3
    later = await seed(repo)
    await click(payload=delete)
    assert api.sent[-1]["text"] == "Удалено записей: 2."
    assert repo.rows[first.id].status == repo.rows[second.id].status == "cancelled"
    assert repo.rows[foreign.id].status == repo.rows[later.id].status == "active"
    writes = repo.writes
    await click(payload=delete)
    assert repo.writes == writes


async def test_single_delete_also_requires_button(session):
    build, repo = session
    event = await seed(repo)
    _, api, message, click, _, _ = build(tool_call("delete_event", {"event_id": str(event.id)}))
    await message("Удали задачу Запись")
    assert repo.rows[event.id].status == "active"
    assert "Вы уверены" in api.sent[-1]["text"]
    await click("Удалить")
    assert repo.rows[event.id].status == "cancelled"


async def test_expired_agent_approval_clears_bot_draft_with_honest_retry_message(session, redis):
    build, repo = session
    event = await seed(repo)
    bot, api, message, click, _, _ = build(tool_call("delete_event", {"event_id": str(event.id)}))
    await message("Удали задачу Запись")
    await redis.delete("agent:pending:777")
    await click("Удалить")
    assert repo.rows[event.id].status == "active"
    assert await bot.drafts.load("777") is None
    assert "Отправьте запрос заново" in api.answered[-1][1]


async def test_lost_preview_receipt_replays_original_deletion_snapshot(session, redis, monkeypatch):
    build, repo = session
    first = await seed(repo)
    _, _, _, _, model, service = build(tool_call("delete_events", {"confirmed": True}))
    request_id = uuid4()
    key = f"agent:request:777:{request_id}"
    original_set = redis.set
    lost = False

    async def fail_receipt(name, value, **kwargs):
        nonlocal lost
        if name == key and '"response": {' in value and not lost:
            from redis.exceptions import ConnectionError

            lost = True
            raise ConnectionError("receipt lost after preview")
        return await original_set(name, value, **kwargs)

    monkeypatch.setattr(redis, "set", fail_receipt)
    response = await service.handle("777", "Удали все мои записи", request_id=request_id)
    assert lost and response.status == "needs_confirmation"
    later = await seed(repo)
    replay = await service.handle("777", "Удали все мои записи", request_id=request_id)
    assert replay.draft.id == response.draft.id and len(model.requests) == 1
    assert repo.rows[first.id].status == "active"
    assert (await service.execute_draft("777", replay.draft)).status == "done"
    assert repo.rows[first.id].status == "cancelled" and repo.rows[later.id].status == "active"


def appointment(tz, hour=12, minutes=30, person="Аней", **extra):
    start = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    return tool_call(
        "create_event",
        {
            "event_type": "task",
            "name": f"Занятие с {person}",
            "start_at": start.isoformat(),
            "duration_minutes": minutes,
            **extra,
        },
    )


async def test_voice_edit_replaces_draft_and_commits_changed_values(session, redis, tz):
    build, repo = session
    source = "Поставь на завтра занятие с Аней в 12:00 на 30 минут"
    bot, api, message, click, _, _ = build(
        appointment(tz, description=source), appointment(tz, 13, 45, "Ольгой")
    )
    await message(source, voice=True)
    preview = api.sent[-1]
    assert "Создать встречу «Занятие с Аней»?" in preview["text"]
    assert "Описание:" not in preview["text"]
    assert "30 мин." in preview["text"] and repo.writes == 0
    old_save = next(b["payload"] for b in all_buttons(preview) if b["text"] == "Сохранить")
    assert await redis.ttl("agent:pending:777") >= 3500
    assert await redis.ttl("bot:draft:777") >= 3500
    await click("Исправить")
    assert await bot.drafts.is_editing("777", await bot.drafts.load("777"))
    await message("Поставь встречу с Ольгой завтра в 13:00 на 45 минут")
    assert repo.writes == 0 and "Ольгой" in api.sent[-1]["text"]
    save = next(b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == "Сохранить")
    await click(payload=old_save)
    assert repo.writes == 0
    await click(payload=save)
    assert repo.writes == 1 and await bot.drafts.load("777") is None
    event = next(iter(repo.rows.values()))
    assert event.type == "meeting" and event.name == "Занятие с Ольгой"
    assert event.start_at.hour == 13 and event.duration_minutes == 45


async def test_voice_save_claim_is_retried_as_a_draft(session, tz):
    build, repo = session
    when = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    _, api, message, click, _, _ = build(
        text("Сохранено: забрать посылку, завтра в 19:00."),
        tool_call(
            "create_event",
            {"event_type": "reminder", "name": "Забрать посылку", "start_at": when.isoformat()},
        ),
        tool_call(
            "create_event",
            {
                "event_type": "reminder",
                "name": "Забрать посылку",
                "start_at": when.replace(minute=30).isoformat(),
            },
        ),
    )
    await message("Напомни завтра в 19:00 забрать посылку", voice=True)
    assert repo.writes == 0
    assert {b["text"] for b in all_buttons(api.sent[-1])} >= {"Сохранить", "Исправить", "Отмена"}
    assert "19:00" in api.sent[-1]["text"]
    await message("измени на 19:30")
    assert repo.writes == 0 and "19:30" in api.sent[-1]["text"]
    await click("Сохранить")
    assert repo.writes == 1
    assert next(iter(repo.rows.values())).start_at == when.replace(minute=30)


async def test_unverified_voice_save_never_claims_success(session):
    build, repo = session
    claim = text("Готово! Сохранено: забрать посылку.")
    _, api, message, _, _, _ = build(claim, claim)
    await message("Напомни завтра в 19:00 забрать посылку", voice=True)
    assert repo.writes == 0
    assert "Календарь не изменён" in api.sent[-1]["text"]


async def test_clock_only_edit_rejects_delete_and_preserves_date_and_status(session, tz):
    build, repo = session
    when = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    event = await repo.create(
        EventCreate(user_id="777", type="reminder", name="Забрать посылку", start_at=when)
    )
    _, api, message, _, model, _ = build(
        tool_call("delete_event", {"event_id": str(event.id)}),
        tool_call(
            "update_event",
            {
                "event_id": str(event.id),
                "start_at": (when + timedelta(days=1)).isoformat(),
                "name": "Другая запись",
            },
        ),
    )
    await message("измени на 19:30")
    saved = repo.rows[event.id]
    assert repo.writes == 2
    assert saved.start_at == when.replace(minute=30)
    assert saved.status == EventStatus.ACTIVE and saved.name == event.name
    assert "19:30" in api.sent[-1]["text"] and "Отменено" not in api.sent[-1]["text"]
    assert {tool["function"]["name"] for tool in model.requests[0]["tools"]} == {
        "update_event",
        "ask_user",
    }


async def test_clock_only_voice_confirmation_keeps_previewed_date(session, tz):
    build, repo = session
    when = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    event = await repo.create(
        EventCreate(user_id="777", type="reminder", name="Забрать посылку", start_at=when)
    )
    _, api, message, click, _, _ = build(
        tool_call("update_event", {"event_id": str(event.id), "start_at": when.isoformat()})
    )
    await message("Измени на 19:30", voice=True)
    assert repo.rows[event.id].start_at == when and "19:30" in api.sent[-1]["text"]
    await repo.update("777", event.id, start_at=when + timedelta(days=1))
    await click("Сохранить")
    assert repo.rows[event.id].start_at == when.replace(minute=30)


async def test_voice_cannot_invent_hour_duration_for_lesson(session, tz):
    build, repo = session
    _, api, message, click, _, _ = build(
        appointment(tz, minutes=60),
        text("Сколько продлится занятие с Аней?"),
        appointment(tz, minutes=30),
    )
    await message("Поставь на завтра занятие с Аней. В 12:00", voice=True)
    assert "Сколько продлится" in api.sent[-1]["text"]
    assert repo.writes == 0
    await message("30 минут")
    assert "30 мин." in api.sent[-1]["text"] and repo.writes == 0
    await click("Сохранить")
    assert repo.writes == 1


async def test_voice_edit_of_existing_meeting_saves_latest_values_after_review(session, tz):
    build, repo = session
    start = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    event = await repo.create(
        EventCreate(
            user_id="777",
            type="meeting",
            name="Занятие с Аней",
            start_at=start,
            duration_minutes=60,
        )
    )
    _, api, message, click, _, _ = build(
        tool_call("update_event", {"event_id": str(event.id), "name": "Занятие с Ольгой"}),
        tool_call(
            "update_event",
            {
                "event_id": str(event.id),
                "name": "Занятие с Ольгой",
                "start_at": start.replace(hour=13).isoformat(),
            },
        ),
    )
    await message("Переименуй занятие с Аней в Занятие с Ольгой", voice=True)
    assert repo.rows[event.id].name == "Занятие с Аней"
    await click("Исправить")
    await message("Перенеси на 13:00")
    assert repo.writes == 1 and "13:00" in api.sent[-1]["text"]
    await click("Сохранить")
    assert repo.writes == 2
    assert repo.rows[event.id].name == "Занятие с Ольгой"
    assert repo.rows[event.id].start_at.hour == 13
    assert repo.rows[event.id].duration_minutes == 60


async def test_voice_preview_preserves_real_description(session, tz):
    build, _ = session
    _, api, message, _, _, _ = build(appointment(tz, description="Взять учебник"))
    await message("Поставь занятие с Аней завтра в 12:00 на 30 минут. Взять учебник", voice=True)
    assert 'Описание: "Взять учебник"' in api.sent[-1]["text"]


@pytest.mark.parametrize(("answer", "minutes"), [("На 45 минут", 45), ("60", 60)])
async def test_plain_model_question_keeps_time_buttons_and_both_missing_fields(
    session, tz, answer, minutes
):
    build, repo = session
    _, api, message, click, model, _ = build(
        text("На какое время завтра поставить занятие с Ольгой?"),
        text("Сколько продлится занятие с Ольгой?"),
        appointment(tz, minutes=minutes, person="Ольгой"),
        text("Занятие сохранено."),
    )
    await message("Поставь на завтра занятие с Ольгой")
    assert api.sent[-1]["text"] == "Во сколько начнётся встреча и сколько она продлится?"
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "09:00",
        "12:00",
        "18:00",
        "Другое время",
    ]
    await click("12:00")
    assert api.sent[-1]["text"] == "Сколько продлится встреча?"
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "30 минут",
        "60 минут",
        "120 минут",
        "Другая длительность",
    ]
    assert repo.writes == 0
    await message(answer)
    assert model.requests[2]["messages"][-1]["content"] == f"На {minutes} минут"
    assert repo.writes == 1
    event = next(iter(repo.rows.values()))
    assert event.start_at.hour == 12 and event.duration_minutes == minutes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("занятие с Аней на 30 минут", False),
        ("30 минут", False),
        ("занятие с Аней в три часа на час", True),
        ("занятие с Аней через 30 минут на час", True),
        ("12:00, 45 минут", True),
    ],
)
def test_start_time_is_not_duration(value, expected):
    assert mentions_start_time(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        "Напомни завтра в 12:00 про занятие с Аней",
        "Создай задачу подготовиться к встрече с Аней",
        "Создай задачу «занятие с Аней»",
    ],
)
def test_explicit_tasks_and_reminders_keep_their_type(value):
    assert not is_person_appointment(value)


@pytest.fixture
def request_clock(monkeypatch, tz):
    clock = [datetime(2030, 11, 10, 1, 10, tzinfo=tz)]

    class Clock(datetime):
        @classmethod
        def now(cls, zone=None):
            return clock[0].astimezone(zone) if zone else clock[0].replace(tzinfo=None)

    monkeypatch.setattr("agent.core.datetime", Clock)
    monkeypatch.setattr("mcp_server.server.datetime", Clock)
    return clock


async def test_read_reschedule_shorten_rename_keeps_tomorrow(session, request_clock, tz):
    build, repo = session
    target = datetime(2030, 11, 11, 11, tzinfo=tz)
    # A known creation ID lets the scripted model imitate the exact regression.
    event = await repo.create(
        EventCreate(
            user_id="777",
            type="meeting",
            name="ПРОВЕРКА Созвон с Анной",
            start_at=target,
            duration_minutes=60,
        )
    )
    _, api, message, _, _, _ = build(
        text("Этот созвон начнётся завтра в 11:00 и продлится 1 час."),
        tool_call(
            "update_event",
            {
                "event_id": str(event.id),
                "start_at": target.replace(day=12, hour=12, minute=20).isoformat(),
                "user_text": "послезавтра",
                "date_reference": "2030-11-20T01:10:00+03:00",
            },
        ),
        tool_call("update_event", {"event_id": str(event.id), "duration_minutes": 20}),
        tool_call(
            "update_event", {"event_id": str(event.id), "name": "ПРОВЕРКА Обсуждение макета"}
        ),
    )
    await message("Во сколько начнётся этот созвон и сколько продлится?")
    await message("Перенеси его на завтра в 12:20.")
    assert "11 ноября 2030" in api.sent[-1]["text"]
    assert repo.rows[event.id].start_at == target.replace(hour=12, minute=20)
    await message("Сократи до 20 минут.")
    await message("Переименуй её в «ПРОВЕРКА Обсуждение макета».")
    saved = repo.rows[event.id]
    assert saved.start_at == target.replace(hour=12, minute=20)
    assert saved.duration_minutes == 20 and saved.end_at.hour == 12 and saved.end_at.minute == 40
    assert saved.name == "ПРОВЕРКА Обсуждение макета"


async def test_voice_reschedule_approval_after_midnight_keeps_preview_day(
    session, request_clock, tz
):
    build, repo = session
    request_clock[0] = datetime(2030, 11, 10, 23, 59, tzinfo=tz)
    target = datetime(2030, 11, 11, 11, tzinfo=tz)
    event = await repo.create(
        EventCreate(
            user_id="777",
            type="meeting",
            name="Созвон",
            start_at=target,
            duration_minutes=60,
        )
    )
    _, api, message, click, _, _ = build(
        tool_call(
            "update_event",
            {
                "event_id": str(event.id),
                "start_at": target.replace(day=12, hour=12, minute=20).isoformat(),
            },
        )
    )
    await message("Перенеси созвон на завтра в 12:20", voice=True)
    assert "11 ноября 2030" in api.sent[-1]["text"]
    assert repo.rows[event.id].start_at == target
    request_clock[0] = datetime(2030, 11, 11, 0, 1, tzinfo=tz)
    await click("Сохранить")
    assert repo.rows[event.id].start_at == target.replace(hour=12, minute=20)
    assert repo.rows[event.id].duration_minutes == 60


async def test_week_reminder_chooses_day_then_time(session, request_clock, tz):
    build, repo = session
    _, api, message, click, model, _ = build(
        text("Какой день на следующей неделе и во сколько напомнить?"),
        text("Во сколько напомнить во вторник?"),
        tool_call(
            "create_event",
            {
                "event_type": "reminder",
                "name": "Проверить документы",
                "start_at": datetime(2030, 11, 19, 9, tzinfo=tz).isoformat(),
            },
        ),
        text("Напоминание сохранено."),
    )
    await message("Напомни на следующей неделе проверить документы")
    assert "На какой день на следующей неделе" in api.sent[-1]["text"]
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "Пн",
        "Вт",
        "Ср",
        "Чт",
        "Пт",
        "Сб",
        "Вс",
        "Другая дата",
    ]
    await click("Вт")
    assert model.requests[-1]["messages"][-1]["content"] == "Во вторник"
    assert repo.writes == 0
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "09:00",
        "12:00",
        "18:00",
        "Другое время",
    ]
    await click("09:00")
    assert repo.writes == 1
    assert next(iter(repo.rows.values())).start_at == datetime(2030, 11, 12, 9, tzinfo=tz)


async def test_duration_button_and_conflict_wait_for_explicit_save(session, tz):
    build, repo = session
    day = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=12,
        minute=20,
        second=0,
        microsecond=0,
    )
    await repo.create(
        EventCreate(
            user_id="777",
            type="meeting",
            name="Обсуждение макета",
            start_at=day,
            duration_minutes=20,
        )
    )
    proposed = tool_call(
        "create_event",
        {
            "event_type": "meeting",
            "name": "Встреча с Антоном",
            "start_at": day.replace(minute=30).isoformat(),
            "duration_minutes": 60,
        },
    )
    bot, api, message, click, model, _ = build(
        text("Сколько продлится встреча?"),
        proposed,
        proposed,
    )
    await message("Поставь завтра встречу с Антоном в 12:30")
    await click("60 минут")
    assert model.requests[-1]["messages"][-1]["content"] == "На 60 минут"
    assert repo.writes == 1
    reply = api.sent[-1]["text"]
    assert "у вас уже запланировано\n• «Обсуждение макета» — 12:20–12:40" in reply
    assert reply.endswith("Всё равно сохранить «Встреча с Антоном» на 12:30–13:30?")
    assert "Длительность" not in reply
    assert len(model.requests) == 2  # no generation to describe the overlap
    assert (await bot.drafts.load("777")).confirm_overlap
    await click("Отмена")
    assert repo.writes == 1
    await message("Поставь завтра встречу с Антоном в 12:30 на час")
    assert repo.writes == 1
    await click("Сохранить")
    reply = api.sent[-1]["text"]
    assert reply.startswith("Готово.\nСохранено: Встреча с Антоном")
    assert "с 12:30 до 13:30" in reply
    assert "Учтите" not in reply and "Обсуждение макета" not in reply
    assert repo.writes == 2
    assert (
        next(
            event for event in repo.rows.values() if event.name.endswith("Антоном")
        ).duration_minutes
        == 60
    )


async def test_misspelled_meeting_asks_day_before_time_and_duration(session, request_clock, tz):
    build, repo = session
    _, api, message, click, _, _ = build(
        text("Во сколько и сколько продлится встреча с Антоном?"),
        text("Во сколько начнётся встреча?"),
        text("Сколько продлится встреча?"),
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с Антоном",
                "start_at": datetime(2030, 11, 11, 9, tzinfo=tz).isoformat(),
                "duration_minutes": 30,
            },
        ),
        text("Встреча сохранена."),
    )
    await message("пастафь встрчу с антоном")
    assert "На какой день" in api.sent[-1]["text"]
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "Пн",
        "Вт",
        "Ср",
        "Чт",
        "Пт",
        "Сб",
        "Вс",
        "Другая дата",
    ]
    assert repo.writes == 0
    await click("Пн")
    assert [b["text"] for b in all_buttons(api.sent[-1])] == [
        "09:00",
        "12:00",
        "18:00",
        "Другое время",
    ]
    await click("09:00")
    assert repo.writes == 0
    await click("30 минут")
    assert repo.writes == 1
    saved = next(iter(repo.rows.values()))
    assert saved.start_at == datetime(2030, 11, 11, 9, tzinfo=tz)
    assert saved.duration_minutes == 30


async def test_meeting_with_clock_and_duration_still_requires_user_day(session, request_clock, tz):
    build, repo = session
    _, api, message, _, _, _ = build(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с Антоном",
                "start_at": datetime(2030, 11, 10, 9, tzinfo=tz).isoformat(),
                "duration_minutes": 30,
            },
        ),
        text("Сохранено."),
    )
    await message("Поставь встречу с Антоном в 09:00 на 30 минут")
    assert repo.writes == 0
    assert "На какой день" in api.sent[-1]["text"]
    assert "Пн" in [b["text"] for b in all_buttons(api.sent[-1])]


async def test_voice_meeting_keeps_explicit_clock_when_agent_asks_again(session):
    build, _ = session
    _, api, message, _, _, _ = build(
        tool_call("ask_user", {"question": "Во сколько запланировать встречу?"}),
    )
    await message("Назначь встречу завтра в 5:00", voice=True)
    assert "Сколько продлится встреча?" == api.sent[-1]["text"]
    assert "30 минут" in [b["text"] for b in all_buttons(api.sent[-1])]


@pytest.mark.parametrize(
    ("request_text", "choice"),
    [
        ("Назначь встречу завтра в пять часов на полчаса", "button"),
        ("Назначь встречу завтра на пять на полчаса", "voice"),
    ],
)
async def test_voice_meeting_asks_only_about_morning_or_evening(session, tz, request_text, choice):
    build, repo = session
    proposed = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=5, minute=0, second=0, microsecond=0
    )
    _, api, message, click, model, _ = build(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча",
                "start_at": proposed.isoformat(),
                "duration_minutes": 30,
            },
        ),
    )
    await message(request_text, voice=True)
    assert [b["text"] for b in all_buttons(api.sent[-1])] == ["05:00", "17:00"]
    assert model.requests == []
    if choice == "button":
        await click("17:00")
    else:
        await message("Вечером", voice=True)
    assert "17:00" in api.sent[-1]["text"]
    assert repo.writes == 0
    await click("Сохранить")
    assert [event.start_at for event in repo.rows.values()] == [proposed.replace(hour=17)]


@pytest.mark.parametrize(
    "clock", ["в 5:00", "в 5.00", "на 5.00", "в пять утра", "в пять ноль ноль"]
)
async def test_voice_meeting_uses_spoken_clock_even_if_model_proposes_wrong_hour(
    session, tz, clock
):
    build, repo = session
    tomorrow = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=17, minute=0, second=0, microsecond=0
    )
    _, api, message, click, _, _ = build(
        tool_call("ask_user", {"question": "Во сколько запланировать встречу?"}),
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча",
                "start_at": tomorrow.isoformat(),
                "duration_minutes": 30,
            },
        ),
    )
    await message(f"Назначь встречу завтра {clock} на полчаса", voice=True)
    assert "05:00" in api.sent[-1]["text"]
    assert "17:00" not in api.sent[-1]["text"]
    await click("Сохранить")
    assert [event.start_at for event in repo.rows.values()] == [tomorrow.replace(hour=5)]


@pytest.mark.parametrize(
    "phrase", ["на 05.12", "в 05.12.2030", "с 05:00 до 06:00", "в 05:00 или 17:00"]
)
def test_date_or_multiple_clocks_do_not_override_the_model_time(phrase):
    assert requested_clock(f"Назначь встречу {phrase}") is None


@pytest.mark.parametrize(
    "phrase", ["в пять тридцать", "в пять часов и десять минут", "на пять декабря"]
)
def test_clock_clarification_does_not_replace_minutes_or_dates(phrase):
    assert ambiguous_hour(f"Назначь встречу {phrase}") is None


async def test_voice_meeting_does_not_loop_if_model_repeats_the_same_question(session):
    build, repo = session
    question = tool_call("ask_user", {"question": "Во сколько запланировать встречу?"})
    _, api, message, _, _, _ = build(question, question)
    await message("Назначь встречу завтра в 05:00 на полчаса", voice=True)
    assert "Распознал время 05:00" in api.sent[-1]["text"]
    assert not all_buttons(api.sent[-1])
    assert repo.writes == 0


async def test_past_meeting_returns_clear_time_question(session, request_clock, tz):
    build, repo = session
    request_clock[0] = datetime(2030, 11, 10, 14, 11, tzinfo=tz)
    _, api, message, _, _, _ = build(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с Антоном",
                "start_at": datetime(2030, 11, 10, 9, tzinfo=tz).isoformat(),
                "duration_minutes": 30,
            },
        ),
        text("Сохранено."),
    )
    await message("Поставь встречу с Антоном сегодня в 09:00 на 30 минут")
    assert repo.writes == 0
    assert api.sent[-1]["text"] == (
        "Это время уже прошло. Во сколько запланировать событие? Можно указать другую дату."
    )
    assert "Другое время" in [b["text"] for b in all_buttons(api.sent[-1])]


async def test_reschedule_rejects_cancelled_record(session, tz):
    build, repo = session
    when = (datetime.now(tz) + timedelta(days=1)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    event = await repo.create(
        EventCreate(user_id="777", type="reminder", name="Забрать посылку", start_at=when)
    )
    await repo.set_status("777", event.id, EventStatus.CANCELLED)
    _, api, message, _, _, _ = build(
        tool_call(
            "update_event",
            {
                "event_id": str(event.id),
                "start_at": when.replace(minute=30).isoformat(),
            },
        ),
        text("Изменено."),
    )
    await message("измени на 19:30")
    assert repo.writes == 2
    assert repo.rows[event.id].start_at == when
    assert api.sent[-1]["text"].startswith("Эта запись уже отменена.")


async def test_name_only_edit_keeps_meeting_day_time_and_duration(session, request_clock, tz):
    build, repo = session
    start = datetime(2030, 11, 21, 7, tzinfo=tz)
    _, api, message, click, model, _ = build(
        tool_call(
            "create_event",
            {
                "event_type": "meeting",
                "name": "Встреча с Сольгой",
                "start_at": start.isoformat(),
                "duration_minutes": 20,
            },
        )
    )
    await message("Поставь встречу с Сольгой 21 ноября 2030 в 07:00 на 20 минут", voice=True)
    old_save = next(b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == "Сохранить")
    await click("Исправить")
    await message("Встреча с Ольгой")
    assert len(model.requests) == 1 and repo.writes == 0
    assert "Встреча с Ольгой" in api.sent[-1]["text"]
    assert "21 ноября 2030" in api.sent[-1]["text"]
    assert "с 07:00 до 07:20" in api.sent[-1]["text"]
    new_save = next(b["payload"] for b in all_buttons(api.sent[-1]) if b["text"] == "Сохранить")
    await click(payload=old_save)
    assert repo.writes == 0
    await click(payload=new_save)
    assert repo.writes == 1
    event = next(iter(repo.rows.values()))
    assert (event.name, event.start_at, event.duration_minutes) == ("Встреча с Ольгой", start, 20)
