"""Time choice buttons stay scoped to the current user and pending draft."""

from typing import Any

import pytest

from bot.handler import BotHandler
from bot.models import AgentMessageResponse, DraftAction
from bot.state import StateStore

USER = 777


class MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.expirations: dict[str, int] = {}

    async def set(self, key, value, **kwargs):
        if kwargs.get("nx") and key in self.values:
            return None
        self.values[key] = value
        if "ex" in kwargs:
            self.expirations[key] = kwargs["ex"]
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.expirations.pop(key, None)
        return int(self.values.pop(key, None) is not None)

    async def eval(self, script, numkeys, *args):
        key, expected, *rest = args
        if self.values.get(key) not in (expected, *rest):
            return 0
        if "redis.call('SET'" in script:
            self.values[key] = rest[0]
            return 1
        return await self.delete(key)


class FakeMax:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.answered: list[tuple[str, str | None]] = []

    async def send(self, chat_id, text, *, to_user=False, buttons=None, text_format=None):
        self.sent.append(
            {"chat_id": chat_id, "text": text, "buttons": buttons, "format": text_format}
        )

    async def answer_callback(self, callback_id, text=None):
        self.answered.append((callback_id, text))


class FakeAgent:
    def __init__(self, *replies: AgentMessageResponse) -> None:
        self.replies = list(replies)
        self.asked: list[dict[str, Any]] = []
        self.confirmed: list[DraftAction] = []

    async def ask(self, user_id, text, confirm_changes=False, request_id=None, edit_draft_id=None):
        self.asked.append(
            {
                "user_id": user_id,
                "text": text,
                "confirm_changes": confirm_changes,
                "request_id": request_id,
                "edit_draft_id": edit_draft_id,
            }
        )
        return self.replies.pop(0)

    async def confirm(self, user_id, draft):
        self.confirmed.append(draft)
        return AgentMessageResponse(reply_text="Готово.", status="done")

    async def reset(self, user_id):
        pass


def draft() -> DraftAction:
    return DraftAction(
        tool="create_event",
        arguments={
            "event_type": "meeting",
            "name": "Созвон",
            "start_at": "2026-11-10T10:00:00+03:00",
            "duration_minutes": 30,
        },
        preview="Создать встречу 10 ноября в 10:00.",
    )


def callback(payload: str, *, user_id: int = USER, callback_id: str = "cb-1"):
    return {
        "update_type": "message_callback",
        "callback": {
            "callback_id": callback_id,
            "payload": payload,
            "user": {"user_id": user_id},
        },
    }


def text_message(text: str, mid: str = "message-1"):
    return {
        "update_type": "message_created",
        "message": {
            "sender": {"user_id": USER},
            "recipient": {"chat_type": "dialog", "chat_id": USER},
            "body": {"mid": mid, "text": text},
        },
    }


def build_bot(*replies: AgentMessageResponse):
    redis = MemoryRedis()
    max_api = FakeMax()
    agent = FakeAgent(*replies)
    bot = BotHandler(max_api, None, agent, StateStore(redis), redis)
    return bot, max_api, agent, redis


def all_buttons(message: dict[str, Any]) -> list[dict[str, str]]:
    return [button for row in (message.get("buttons") or []) for button in row]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Во сколько напомнить?",
        "Когда состоится встреча? Укажите дату и время.",
    ],
)
async def test_missing_clock_clarification_offers_unique_ttl_bound_presets(question: str):
    bot, max_api, _, redis = build_bot()
    response = AgentMessageResponse(reply_text=question, status="need_clarification")

    await bot._result(str(USER), USER, response)

    buttons = all_buttons(max_api.sent[-1])
    assert [button["text"] for button in buttons] == [
        "09:00",
        "12:00",
        "18:00",
        "Другое время",
    ]
    payloads = [button["payload"] for button in buttons]
    assert len(set(payloads)) == 4
    chooser_ids = {payload.split(":")[1] for payload in payloads}
    assert len(chooser_ids) == 1
    key = f"bot:time-choice:{USER}"
    assert redis.expirations[key] == 300


@pytest.mark.asyncio
async def test_duration_only_clarification_offers_duration_presets():
    bot, max_api, _, redis = build_bot()
    response = AgentMessageResponse(
        reply_text="Какова длительность встречи?", status="need_clarification"
    )

    await bot._result(str(USER), USER, response)

    assert [b["text"] for b in all_buttons(max_api.sent[-1])] == [
        "30 минут",
        "60 минут",
        "120 минут",
        "Другая длительность",
    ]
    assert f"bot:time-choice:{USER}" in redis.values


@pytest.mark.asyncio
async def test_draft_time_button_opens_presets_and_preset_runs_normal_edit_flow():
    current = draft()
    replacement = DraftAction(
        tool="create_event",
        arguments={
            **current.arguments,
            "start_at": "2026-11-10T09:00:00+03:00",
        },
        preview="Создать встречу 10 ноября в 09:00.",
    )
    bot, max_api, agent, redis = build_bot(
        AgentMessageResponse(
            reply_text=replacement.preview,
            status="needs_confirmation",
            draft=replacement,
        )
    )
    await bot.drafts.save(str(USER), current)

    await bot.handle_update(callback(f"draft:time:{current.id}"))

    buttons = all_buttons(max_api.sent[-1])
    assert [button["text"] for button in buttons] == [
        "09:00",
        "12:00",
        "18:00",
        "Другое время",
    ]
    assert redis.expirations[f"bot:time-choice:{USER}"] == 300
    selected = next(button for button in buttons if button["text"] == "09:00")

    await bot.handle_update(callback(selected["payload"], callback_id="select-09"))

    assert len(agent.asked) == 1
    assert agent.asked[0]["user_id"] == str(USER)
    assert agent.asked[0]["text"] == "В 09:00"
    assert agent.asked[0]["confirm_changes"] is True
    assert agent.asked[0]["edit_draft_id"] == str(current.id)
    assert agent.asked[0]["request_id"] is not None
    assert agent.confirmed == []
    assert await bot.drafts.load(str(USER)) == replacement
    assert f"bot:time-choice:{USER}" not in redis.values


@pytest.mark.asyncio
async def test_manual_choice_requests_text_or_voice_and_keeps_draft():
    current = draft()
    bot, max_api, agent, _ = build_bot()
    await bot.drafts.save(str(USER), current)
    await bot.handle_update(callback(f"draft:time:{current.id}"))
    manual = next(
        button for button in all_buttons(max_api.sent[-1]) if button["text"] == "Другое время"
    )

    await bot.handle_update(callback(manual["payload"], callback_id="select-manual"))

    assert "напиши или скажи голосом" in max_api.sent[-1]["text"].casefold()
    assert await bot.drafts.load(str(USER)) == current
    assert agent.asked == [] and agent.confirmed == []


async def test_failed_preset_result_delivery_retries_original_draft_and_request_id():
    current = draft()
    replacement = DraftAction(
        tool=current.tool,
        arguments={**current.arguments, "start_at": "2026-11-10T12:00:00+03:00"},
        preview="Новый вариант на 12:00.",
    )
    result = AgentMessageResponse(
        reply_text=replacement.preview, status="needs_confirmation", draft=replacement
    )
    bot, max_api, agent, _ = build_bot(result, result)
    await bot.drafts.save(str(USER), current)
    await bot.handle_update(callback(f"draft:time:{current.id}"))
    selection = next(b for b in all_buttons(max_api.sent[-1]) if b["text"] == "12:00")
    send = max_api.send
    failed = False

    async def fail_once(chat_id, text, **kwargs):
        nonlocal failed
        if text == replacement.preview and not failed:
            failed = True
            raise OSError("temporary delivery failure")
        await send(chat_id, text, **kwargs)

    max_api.send = fail_once
    update = callback(selection["payload"], callback_id="retry-preset")
    with pytest.raises(OSError, match="temporary delivery failure"):
        await bot.handle_update(update)
    assert await bot.drafts.load(str(USER)) == current
    await bot.handle_update(update)
    assert await bot.drafts.load(str(USER)) == replacement
    assert len(agent.asked) == 2
    assert agent.asked[0]["request_id"] == agent.asked[1]["request_id"]
    assert agent.asked[0]["edit_draft_id"] == agent.asked[1]["edit_draft_id"] == str(current.id)
    assert await bot.state.current(str(USER)) == "idle"


@pytest.mark.asyncio
async def test_old_or_foreign_choice_callbacks_are_rejected():
    bot, max_api, _, _ = build_bot()
    await bot._result(
        str(USER), USER, AgentMessageResponse(reply_text="Во сколько?", status="need_clarification")
    )
    old_choice = all_buttons(max_api.sent[-1])[0]["payload"]
    await bot._result(
        str(USER), USER, AgentMessageResponse(reply_text="Во сколько?", status="need_clarification")
    )
    current_choice = all_buttons(max_api.sent[-1])[0]["payload"]
    assert old_choice != current_choice

    await bot.handle_update(callback(old_choice, callback_id="old"))
    assert "устарел" in (max_api.answered[-1][1] or "").casefold()

    await bot.handle_update(callback(current_choice, user_id=888, callback_id="foreign"))
    assert "устарел" in (max_api.answered[-1][1] or "").casefold()
    assert len(max_api.sent) == 2


@pytest.mark.asyncio
async def test_new_user_message_clears_old_time_choice():
    bot, max_api, agent, _ = build_bot(AgentMessageResponse(reply_text="Записал.", status="done"))
    await bot._result(
        str(USER), USER, AgentMessageResponse(reply_text="Во сколько?", status="need_clarification")
    )
    stale_choice = all_buttons(max_api.sent[-1])[0]["payload"]

    await bot.handle_update(text_message("В 14:00", mid="manual-answer"))
    assert agent.asked[0]["text"] == "В 14:00"

    await bot.handle_update(callback(stale_choice, callback_id="stale-after-message"))
    assert "устарел" in (max_api.answered[-1][1] or "").casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    ["нет, время подходит", "всё равно на пересечение", "оставь как есть"],
)
async def test_explicit_keep_phrase_confirms_pending_draft_directly(phrase: str):
    current = draft()
    bot, _, agent, _ = build_bot()
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), current)

    await bot.handle_update(text_message(phrase, mid=f"keep-{phrase}"))

    assert agent.confirmed == [current]
    assert agent.asked == []
    assert await bot.drafts.load(str(USER)) is None


@pytest.mark.asyncio
async def test_bare_no_still_cancels_pending_draft():
    current = draft()
    bot, _, agent, _ = build_bot()
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), current)

    await bot.handle_update(text_message("нет", mid="bare-no"))

    assert agent.confirmed == [] and agent.asked == []
    assert await bot.drafts.load(str(USER)) is None


@pytest.mark.asyncio
async def test_keep_phrase_cancels_pending_delete_without_calling_agent():
    deletion = DraftAction(
        tool="delete_event", arguments={"event_id": "event-1"}, preview="Удалить встречу"
    )
    bot, max_api, agent, _ = build_bot()
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), deletion)

    await bot.handle_update(text_message("оставь как есть", mid="keep-delete"))

    assert agent.confirmed == [] and agent.asked == []
    assert await bot.drafts.load(str(USER)) is None
    assert "без изменений" in max_api.sent[-1]["text"]


@pytest.mark.asyncio
async def test_negative_reschedule_text_is_not_treated_as_keep():
    current = draft()
    bot, _, agent, _ = build_bot(
        AgentMessageResponse(reply_text="Новый вариант.", status="need_clarification")
    )
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), current)

    await bot.handle_update(text_message("нет, перенеси на 12:00", mid="reschedule"))

    assert agent.confirmed == []
    assert agent.asked[0]["text"] == "нет, перенеси на 12:00"
    assert agent.asked[0]["edit_draft_id"] == str(current.id)


@pytest.mark.parametrize(
    "start",
    [
        text_message("/start"),
        {"update_type": "bot_started", "user": {"user_id": USER}},
    ],
)
async def test_welcome_only_shows_calendar_format_after_import_button(start):
    bot, api, agent, _ = build_bot()
    await bot.handle_update(start)
    welcome = api.sent[-1]
    assert welcome["text"].startswith("Привет! Я голосовой ассистент команды Нейм.Тим")
    assert "**Весь календарь**" in welcome["text"]
    assert "мини-приложении" in welcome["text"]
    assert "CSV" not in welcome["text"]
    assert welcome["format"] == "markdown"
    buttons = {button["text"]: button["payload"] for button in all_buttons(welcome)}
    assert buttons == {"Загрузить календарь": "quick:import", "Что я умею": "quick:features"}
    await bot.handle_update(callback(buttons["Что я умею"], callback_id="features"))
    assert "**Задачи и встречи**" in api.sent[-1]["text"]
    assert "Повторяющиеся напоминания" in api.sent[-1]["text"]
    await bot.handle_update(callback(buttons["Загрузить календарь"]))
    assert "CSV UTF-8" in api.sent[-1]["text"]
    assert all(column in api.sent[-1]["text"] for column in ("`id`", "`date`", "`time`", "`text`"))
    assert not agent.asked
