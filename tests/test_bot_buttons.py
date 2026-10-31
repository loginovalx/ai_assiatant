"""Кнопки бота и их текстовые эквиваленты.

Кнопки — дополнение к чату, а не замена: «да» и «отмена» словами делают то же самое.
Настоящие MAX и агент заменены заглушками: проверяем маршрутизацию бота, а не их API.
"""

from typing import Any

import pytest
from redis.asyncio import Redis

from bot.handler import QUICK_ACTIONS, BotHandler, confirmation_buttons
from bot.models import AgentMessageResponse, DraftAction
from bot.state import StateStore

USER = 777


class FakeMax:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.answered: list[tuple[str, str | None]] = []

    async def send(self, chat_id, text, *, to_user=False, buttons=None) -> None:
        self.sent.append({"chat_id": chat_id, "text": text, "buttons": buttons})

    async def answer_callback(self, callback_id, text=None) -> None:
        self.answered.append((callback_id, text))

    async def get_message(self, mid):  # pragma: no cover — в этих тестах не нужен
        return {}


class FakeAgent:
    """Отдаёт заготовленные ответы и запоминает, с какими флагами её звали."""

    def __init__(self, *replies: AgentMessageResponse, confirm_reply=None) -> None:
        self.replies = list(replies)
        self.asked: list[dict[str, Any]] = []
        self.confirmed: list[DraftAction] = []
        self.confirm_reply = confirm_reply

    async def ask(
        self, user_id, text, confirm_changes=False, request_id=None
    ) -> AgentMessageResponse:
        self.asked.append({"text": text, "confirm_changes": confirm_changes})
        return self.replies.pop(0)

    async def confirm(self, user_id, draft) -> AgentMessageResponse:
        self.confirmed.append(draft)
        return self.confirm_reply or AgentMessageResponse(
            reply_text=f"Готово. {draft.preview}", status="done"
        )


DRAFT = DraftAction(
    tool="create_event",
    arguments={"event_type": "reminder", "name": "Записаться к врачу"},
    preview="Напомню вам записаться к врачу завтра в 11:00.",
)


def draft_reply() -> AgentMessageResponse:
    return AgentMessageResponse(reply_text=DRAFT.preview, status="needs_confirmation", draft=DRAFT)


@pytest.fixture
async def redis_client():
    import os

    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL не задан")
    client = Redis.from_url(url, decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
def handler(redis_client):
    def build(agent: FakeAgent) -> tuple[BotHandler, FakeMax]:
        max_api = FakeMax()
        bot = BotHandler(
            max_api, asr=None, agent=agent, state=StateStore(redis_client), redis=redis_client
        )
        return bot, max_api

    return build


def text_message(body_text: str, mid: str = "m1") -> dict[str, Any]:
    return {
        "update_type": "message_created",
        "message": {
            "sender": {"user_id": USER},
            "recipient": {"chat_type": "dialog"},
            "body": {"mid": mid, "text": body_text},
        },
    }


def callback(payload: str) -> dict[str, Any]:
    if payload.startswith("draft:") and len(payload.split(":")) == 2:
        payload += ":" + str(DRAFT.id)
    return {
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb-1",
            "payload": payload,
            "user": {"user_id": USER},
            "message": {"recipient": {"chat_id": USER}},
        },
    }


async def test_greeting_offers_quick_actions(handler):
    bot, max_api = handler(FakeAgent())
    await bot.handle_update(text_message("/start"))

    assert max_api.sent[0]["buttons"] == QUICK_ACTIONS
    assert "AI-ассистент" in max_api.sent[0]["text"]


async def test_confirm_button_executes_draft(handler, redis_client):
    agent = FakeAgent()
    bot, max_api = handler(agent)
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(callback("draft:confirm"))

    assert agent.confirmed == [DRAFT]
    assert max_api.sent[-1]["text"].startswith("Готово.")
    assert max_api.answered == [("cb-1", "✅ Подтверждено")]  # кнопки убраны
    assert await bot.drafts.load(str(USER)) is None


async def test_failed_button_confirmation_keeps_draft_and_reports_failure(handler):
    failure = AgentMessageResponse(reply_text="Не получилось выполнить.", status="error")
    agent = FakeAgent(confirm_reply=failure)
    bot, max_api = handler(agent)
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(callback("draft:confirm"))

    assert max_api.answered == [("cb-1", "Не получилось выполнить. Черновик сохранён")]
    assert max_api.sent[-1]["text"] == failure.reply_text
    assert await bot.drafts.load(str(USER)) == DRAFT


async def test_cancel_button_drops_draft_without_executing(handler):
    agent = FakeAgent()
    bot, max_api = handler(agent)
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(callback("draft:cancel"))

    assert agent.confirmed == []
    assert "Отменил" in max_api.sent[-1]["text"]
    assert await bot.drafts.load(str(USER)) is None


async def test_edit_button_asks_what_to_change_and_keeps_draft(handler):
    bot, max_api = handler(FakeAgent())
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(callback("draft:edit"))

    assert "Что поправить" in max_api.sent[-1]["text"]
    assert await bot.drafts.load(str(USER)) is not None  # ждём правку


async def test_old_confirmation_button_cannot_execute_replacement_draft(handler):
    agent = FakeAgent()
    bot, max_api = handler(agent)
    await bot._result(str(USER), USER, draft_reply())
    old_payload = max_api.sent[-1]["buttons"][0][0]["payload"]
    replacement = DraftAction(tool="delete_events", arguments={"event_ids": []}, preview="Удалить")
    await bot.drafts.save(str(USER), replacement)
    update = callback(old_payload)
    update["callback"]["payload"] = old_payload
    await bot.handle_update(update)
    assert agent.confirmed == []
    assert (await bot.drafts.load(str(USER))).id == replacement.id


async def test_yes_in_words_works_like_the_button(handler):
    agent = FakeAgent()
    bot, _ = handler(agent)
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(text_message("да"))

    assert agent.confirmed == [DRAFT]
    assert agent.asked == []  # в агента как новый запрос не ушло


async def test_failed_text_confirmation_keeps_draft_for_retry(handler):
    failure = AgentMessageResponse(reply_text="Временная ошибка.", status="error")
    agent = FakeAgent(confirm_reply=failure)
    bot, max_api = handler(agent)
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(text_message("да"))

    assert max_api.sent[-1]["text"] == failure.reply_text
    assert await bot.drafts.load(str(USER)) == DRAFT


async def test_free_text_while_draft_pending_goes_to_agent_as_edit(handler):
    agent = FakeAgent(draft_reply())
    bot, max_api = handler(agent)
    await bot.redis.set(f"bot:greeted:{USER}", "1")
    await bot.drafts.save(str(USER), DRAFT)

    await bot.handle_update(text_message("давай в 12:30"))

    assert agent.asked[0]["text"] == "давай в 12:30"
    assert agent.confirmed == []
    assert max_api.sent[-1]["buttons"] == confirmation_buttons(DRAFT.id)


async def test_quick_action_runs_like_a_typed_question(handler):
    agent = FakeAgent(AgentMessageResponse(reply_text="Завтра ничего нет.", status="done"))
    bot, max_api = handler(agent)

    await bot.handle_update(callback("quick:tomorrow"))

    assert agent.asked == [{"text": "Что у меня завтра?", "confirm_changes": False}]
    assert max_api.sent[-1]["text"] == "Завтра ничего нет."


async def test_text_command_is_not_deferred(handler):
    agent = FakeAgent(AgentMessageResponse(reply_text="Напомню завтра в 11:00.", status="done"))
    bot, max_api = handler(agent)
    await bot.redis.set(f"bot:greeted:{USER}", "1")

    await bot.handle_update(text_message("напомни завтра в 11 записаться к врачу"))

    assert agent.asked[0]["confirm_changes"] is False  # текст выполняется сразу
    assert max_api.sent[-1]["buttons"] is None


_CSV_BYTES = "id,date,time,text\n1,2026-11-05,10:00,Позвонить Ивану\n".encode()


class CsvMax(FakeMax):
    """Апдейт приходит без имени файла; подробности от /messages называют его CSV."""

    def __init__(self) -> None:
        super().__init__()
        self.downloaded: list[str] = []

    async def get_message(self, mid):
        return {
            "body": {
                "mid": mid,
                "attachments": [
                    {
                        "type": "file",
                        "payload": {"filename": "calendar.csv", "url": "https://mycdn.me/x"},
                    }
                ],
            }
        }

    async def get_file(self, url, max_bytes, what="файл"):
        self.downloaded.append(url)
        return _CSV_BYTES


class FakeRepo:
    def __init__(self) -> None:
        self.imported: list[tuple[str, int]] = []

    async def import_calendar(self, user_id, rows) -> int:
        self.imported.append((user_id, len(rows)))
        return len(rows)


async def test_csv_file_without_filename_in_update_is_imported(redis_client):
    """MAX не всегда передаёт имя файла в апдейте: CSV не должен превращаться в «голос»."""
    max_api = CsvMax()
    repo = FakeRepo()
    bot = BotHandler(
        max_api,
        asr=None,
        agent=FakeAgent(),
        state=StateStore(redis_client),
        redis=redis_client,
        repo=repo,
    )
    update = {
        "update_type": "message_created",
        "message": {
            "sender": {"user_id": USER},
            "recipient": {"chat_type": "dialog", "chat_id": USER},
            "body": {
                "mid": "m-csv",
                "text": "",
                "attachments": [{"type": "file", "payload": {}}],
            },
        },
    }
    await bot.handle_update(update)

    assert max_api.downloaded == ["https://mycdn.me/x"]
    assert repo.imported == [(str(USER), 1)]
    assert any("Загрузил событий: 1" in s["text"] for s in max_api.sent)
