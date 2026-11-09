"""Live QA regressions through the agent and MCP, without a paid model call."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

import mcp_server.server as calendar_server
from agent.core import Agent
from agent.dialogue import direct_reply
from agent.service import AgentService
from agent.tools import McpToolbox
from common.repository import EventRepository
from common.schemas import Event, EventCreate
from common.timeutils import mentions_recurrence, named_weekday, resolve_calendar_start
from tests.scripted_llm import ScriptedLLM, text, tool_call
from tests.test_model_routing import Memory

TZ = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 11, 9, 16, 48, tzinfo=TZ)


class CalendarMemory(Memory):
    async def reset(self, user_id):
        self.history.clear()


@pytest.fixture
def calendar(monkeypatch):
    clock = SimpleNamespace(now=NOW)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now.astimezone(tz) if tz else clock.now.replace(tzinfo=None)

    monkeypatch.setattr(calendar_server, "datetime", Clock)

    class Repository:
        creation_id = staticmethod(EventRepository.creation_id)

        def __init__(self):
            self.rows = {}
            self.writes = 0

        async def create(self, data, **kwargs):
            key = kwargs.get("idempotency_key")
            event = Event(
                **data.model_dump(),
                id=self.creation_id(data.user_id, key) if key else uuid4(),
                status="active",
                created_at=clock.now,
                updated_at=clock.now,
            )
            self.rows[event.id] = event
            self.writes += 1
            return event

        async def get(self, user_id, event_id):
            event = self.rows.get(event_id)
            return event if event and event.user_id == user_id else None

        async def update(self, user_id, event_id, **changes):
            event = await self.get(user_id, event_id)
            if event is None:
                return None
            event = event.model_copy(update={k: v for k, v in changes.items() if v is not None})
            self.rows[event_id] = event
            self.writes += 1
            return event

        async def find_conflicts(self, event):
            return []

        async def list_by_period(self, user_id, start, end, types, include_undated):
            return sorted(
                [
                    event
                    for event in self.rows.values()
                    if event.user_id == user_id
                    and event.status != "cancelled"
                    and (not types or event.type in types)
                    and (
                        (event.start_at is not None and start <= event.start_at < end)
                        or (include_undated and event.type == "task" and event.start_at is None)
                    )
                ],
                key=lambda event: event.start_at or start,
            )

    repo = Repository()

    @lifespan
    async def resources(server):
        yield {"repo": repo}

    server = FastMCP("calendar-test", lifespan=resources)
    server.tool(calendar_server.create_event)
    server.tool(calendar_server.update_event)
    server.tool(calendar_server.get_events)
    return repo, McpToolbox(server), clock


def make_agent(toolbox, *script):
    llm = ScriptedLLM(*script)
    return Agent(llm, "test", toolbox, CalendarMemory(), TZ), llm


@pytest.mark.parametrize("proposed", ["2026-11-18", "2026-11-24", "2026-11-10"])
@pytest.mark.parametrize("dry_run", [False, True])
async def test_clarified_next_week_tuesday_corrects_wrong_day_and_wrong_week(
    calendar, proposed, dry_run
):
    repo, toolbox, _ = calendar
    result = await toolbox.call(
        "u",
        "create_event",
        {
            "event_type": "reminder",
            "name": "ТЕСТ проверить документы",
            "start_at": f"{proposed}T09:00:00+03:00",
        },
        user_text="Напомни на следующей неделе ТЕСТ проверить документы.\nвторник\nВ 09:00",
        dry_run=dry_run,
    )
    assert result.ok, result.payload
    assert result.events[0]["start_at"] == "2026-11-17T09:00:00+03:00"
    assert result.events[0]["weekday"] == "вторник"
    assert repo.writes == (0 if dry_run else 1)


@pytest.mark.parametrize("dry_run", [False, True])
async def test_reschedule_tomorrow_uses_current_day_and_preserves_duration(calendar, dry_run):
    repo, toolbox, _ = calendar
    current = await repo.create(
        EventCreate(
            user_id="u",
            type="meeting",
            name="Дизайнер",
            start_at=datetime(2026, 11, 10, 12, tzinfo=TZ),
            duration_minutes=10,
        )
    )
    agent, _ = make_agent(
        toolbox,
        tool_call(
            "update_event",
            {
                "event_id": str(current.id),
                "start_at": "2026-11-11T17:00:00+03:00",
                "user_text": "Перенеси на послезавтра",
                "date_reference": "2099-01-01T00:00:00Z",
            },
        ),
        text("Готово"),
    )
    response = await agent.handle(
        "u",
        "Перенеси эту встречу на завтра в 17:00.",
        now=NOW,
        confirm_changes=dry_run,
    )
    assert response.status == ("needs_confirmation" if dry_run else "done")
    assert "10 ноября 2026" in response.reply_text and "17:10" in response.reply_text
    if dry_run:
        assert repo.rows[current.id].start_at == current.start_at
        response = await agent.execute_draft("u", response.draft)
        assert response.status == "done"
    assert repo.rows[current.id].start_at == datetime(2026, 11, 10, 17, tzinfo=TZ)
    assert repo.rows[current.id].duration_minutes == 10


async def test_actual_three_turn_clarification_retains_week_day_and_time(calendar):
    repo, toolbox, _ = calendar
    agent, _ = make_agent(
        toolbox,
        tool_call("ask_user", {"question": "Какой день на следующей неделе выбрать?"}),
        tool_call("ask_user", {"question": "Во сколько во вторник напомнить?"}),
        tool_call(
            "create_event",
            {
                "event_type": "reminder",
                "name": "Документы",
                "start_at": "2026-11-18T09:00:00+03:00",
            },
        ),
        text("Готово"),
    )
    first = await agent.handle("u", "Напомни на следующей неделе проверить документы", now=NOW)
    second = await agent.handle("u", "вторник", now=NOW)
    third = await agent.handle("u", "В 09:00", now=NOW)
    assert first.status == second.status == "need_clarification"
    assert third.status == "done"
    assert next(iter(repo.rows.values())).start_at == datetime(2026, 11, 17, 9, tzinfo=TZ)
    assert repo.writes == 1


@pytest.mark.parametrize("tool", ["create_event", "update_event"])
async def test_confirmation_across_midnight_keeps_preview_date(calendar, tool):
    repo, toolbox, clock = calendar
    clock.now = datetime(2026, 11, 9, 23, 59, tzinfo=TZ)
    args = {"start_at": "2026-11-11T17:00:00+03:00"}
    if tool == "create_event":
        args.update(event_type="reminder", name="Документы")
        command = "Напомни завтра в 17:00 проверить документы"
    else:
        current = await repo.create(EventCreate(user_id="u", type="task", name="Документы"))
        args["event_id"] = str(current.id)
        command = "Перенеси её на завтра в 17:00"
    agent, llm = make_agent(toolbox, tool_call(tool, args))
    response = await agent.handle("u", command, now=clock.now, confirm_changes=True)
    assert response.status == "needs_confirmation"
    assert "10 ноября 2026" in response.reply_text
    clock.now += timedelta(minutes=2)
    confirmed = await agent.execute_draft("u", response.draft)
    assert confirmed.status == "done"
    assert "10 ноября 2026" in confirmed.reply_text
    assert next(iter(repo.rows.values())).start_at == datetime(2026, 11, 10, 17, tzinfo=TZ)
    assert len(llm.requests) == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "Напоминай каждый понедельник в 09:00 проверить отчёт",
        "Поставь еженедельную встречу в 09:00 на час",
        "Напомни по вторникам в 09:00 проверить документы",
        "Каждый день в 09:00",
    ],
)
async def test_recurrence_refused_without_model_or_calendar_write(calendar, phrase):
    repo, toolbox, _ = calendar
    agent, llm = make_agent(toolbox)
    result = await agent.handle("u", phrase, now=NOW)
    assert "Повторяющиеся события пока не поддерживаются" in result.reply_text
    assert repo.writes == len(llm.requests) == 0


async def test_recurrence_answer_clears_unfinished_creation(calendar):
    repo, toolbox, _ = calendar
    agent, llm = make_agent(toolbox)
    agent._memory.history = [
        {
            "role": "assistant",
            "content": "Когда напомнить?",
            "clarification_text": "Напомни проверить отчёт",
        }
    ]
    result = await agent.handle("u", "Каждый понедельник в 09:00", now=NOW)
    assert "не поддерживаются" in result.reply_text
    assert agent._memory.history == []
    assert repo.writes == len(llm.requests) == 0


@pytest.mark.parametrize(
    "phrase",
    [
        "Найди еженедельную встречу",
        "Удали еженедельную встречу",
        "Заверши ежедневную задачу",
    ],
)
def test_recurrence_guard_does_not_block_read_delete_or_complete(phrase):
    assert direct_reply(phrase) is None


async def test_edited_reschedule_draft_keeps_trusted_day_and_confirmation_authority(
    calendar,
    redis,
    monkeypatch,
):
    repo, toolbox, _ = calendar
    monkeypatch.setattr("agent.core.datetime", calendar_server.datetime)
    current = await repo.create(EventCreate(user_id="u", type="task", name="Документы"))
    args = {"event_id": str(current.id), "start_at": "2026-11-11T17:00:00+03:00"}
    agent, llm = make_agent(
        toolbox,
        tool_call("update_event", args),
        tool_call("update_event", {**args, "start_at": "2026-11-11T18:00:00+03:00"}),
    )
    service = AgentService(agent, redis)
    first = await service.handle("u", "Перенеси её на завтра в 17:00", confirm_changes=True)
    assert first.draft
    second = await service.handle(
        "u",
        "Лучше в 18:00",
        confirm_changes=True,
        edit_draft_id=first.draft.id,
    )
    assert second.draft and "10 ноября 2026" in second.reply_text
    assert repo.rows[current.id].start_at is None
    assert (await service.execute_draft("u", first.draft)).status == "error"
    assert (await service.execute_draft("u", second.draft)).status == "done"
    assert repo.rows[current.id].start_at == datetime(2026, 11, 10, 18, tzinfo=TZ)
    assert len(llm.requests) == 2


@pytest.mark.parametrize("tool", ["create_event", "update_event"])
@pytest.mark.parametrize("dry_run", [False, True])
async def test_mcp_also_refuses_recurring_requests_before_any_write(calendar, tool, dry_run):
    repo, toolbox, _ = calendar
    arguments = {"start_at": "2026-11-10T09:00:00+03:00"}
    if tool == "create_event":
        arguments.update(event_type="reminder", name="Отчёт")
    else:
        arguments["event_id"] = str(uuid4())
    result = await toolbox.call(
        "u",
        tool,
        arguments,
        user_text="Напоминай каждый понедельник в 09:00 проверить отчёт",
        dry_run=dry_run,
    )
    assert not result.ok and "Повторяющиеся события пока не поддерживаются" in result.payload
    assert repo.writes == 0


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Вторник", 1),
        ("во вторник в 09:00", 1),
        ("На следующей неделе\nвторник\n09:00", 1),
        ("В среду", 2),
        ("Встреча с дизайнером", None),
        ("Создай задачу «Во вторник»", None),
        ("Во вторник и в среду", None),
    ],
)
def test_weekday_clarifications_and_quoted_titles(source, expected):
    assert named_weekday(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Завтра в 17:00", "2026-11-10"),
        ("послезавтра в 17:00", "2026-11-11"),
        ("сегодня в 17:00", "2026-11-09"),
        ("Вчера в 17:00", "2026-11-08"),
        ("На следующей неделе\nвторник\nВ 17:00", "2026-11-17"),
        ("На этой неделе во вторник в 17:00", "2026-11-10"),
        ("Встреча «Завтра в пятницу» завтра в 17:00", "2026-11-10"),
        ("Во вторник в 10.00", "2026-11-10"),
    ],
)
def test_date_resolution_preserves_local_clock(source, expected):
    proposed = datetime(2026, 11, 18, 17, 15, tzinfo=TZ)
    resolved = resolve_calendar_start(source, proposed, NOW)
    assert resolved.date().isoformat() == expected
    assert (resolved.hour, resolved.minute) == (17, 15)


@pytest.mark.parametrize("tool", ["create_event", "update_event"])
async def test_explicit_past_day_cannot_be_shifted_to_future_by_model(calendar, tool):
    repo, toolbox, _ = calendar
    args = {"start_at": "2026-11-10T10:00:00+03:00"}
    if tool == "create_event":
        args.update(event_type="reminder", name="Письмо")
    else:
        args["event_id"] = str(uuid4())
    result = await toolbox.call("u", tool, args, user_text="Напомни вчера в 10:00 отправить письмо")
    assert not result.ok and "уже прошло" in result.payload
    assert repo.writes == 0


@pytest.mark.parametrize(
    "source",
    [
        "Во вторник 2026-11-17 в 09:00",
        "17 ноября 2026 в 09:00",
        "17.11.2026 в 09:00",
    ],
)
def test_explicit_dates_are_not_silently_replaced(source):
    assert resolve_calendar_start(source, datetime(2026, 11, 17, 9, tzinfo=TZ), NOW) is None


def test_conflicting_relative_days_require_clarification():
    with pytest.raises(ValueError, match="несколько дней"):
        resolve_calendar_start("Завтра или послезавтра в 09:00", NOW, NOW)


async def test_quoted_recurrence_is_a_literal_task_title(calendar):
    repo, toolbox, _ = calendar
    agent, _ = make_agent(
        toolbox,
        tool_call(
            "create_event",
            {
                "event_type": "task",
                "name": "Каждый понедельник",
            },
        ),
    )
    response = await agent.handle("u", "Создай задачу «Каждый понедельник»", now=NOW)
    assert response.status == "done"
    assert next(iter(repo.rows.values())).name == "Каждый понедельник"


def test_recurrence_detector_does_not_treat_every_task_as_a_schedule():
    assert not mentions_recurrence("Проверь каждую задачу")
    assert not mentions_recurrence('Создай задачу "Ежедневный отчёт"')


async def test_internal_date_evidence_is_hidden_from_model_schemas(calendar):
    _, toolbox, _ = calendar
    for tool in await toolbox.openai_tools():
        assert (
            not {"user_text", "date_reference"}
            & tool["function"]["parameters"]["properties"].keys()
        )
