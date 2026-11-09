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
from common.timeutils import mentions_start_time
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
        return []

    async def preview_cancellation_ids(self, owner, types, start, end):
        return [
            event.id
            for event in self.rows.values()
            if event.user_id == owner
            and event.status == "active"
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
        allowed = await self.preview_cancellation_ids(owner, types, start, end)
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
    first, second, foreign = await seed(repo), await seed(repo), await seed(repo, "other")
    _, api, message, click, _, _ = build(tool_call("delete_events", {"confirmed": True}))
    await message("Удали все мои записи")
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


async def test_plain_model_question_keeps_time_buttons_and_both_missing_fields(session, tz):
    build, repo = session
    _, api, message, click, _, _ = build(
        text("На какое время завтра поставить занятие с Ольгой?"),
        text("Сколько продлится занятие с Ольгой?"),
        appointment(tz, minutes=45, person="Ольгой"),
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
    assert not all_buttons(api.sent[-1]) and repo.writes == 0
    await message("На 45 минут")
    assert repo.writes == 1
    event = next(iter(repo.rows.values()))
    assert event.start_at.hour == 12 and event.duration_minutes == 45


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
