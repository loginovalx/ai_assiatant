"""MAX message routing + user-friendly progress + Redis FSM."""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from redis.asyncio import Redis

from agent.dialogue import starts_new_request
from agent.policy import asks_to_read
from bot.agent_client import AgentClient, AgentUnavailable
from bot.asr import SpeechRecognitionError, SpeechRecognizer
from bot.drafts import DraftStore
from bot.max_api import MaxAPI, MaxAPIError
from bot.models import AgentMessageResponse
from bot.recurring import RecurringReminders
from bot.state import StateStore
from bot.time_choices import (
    CHOICE_REPLIES,
    TimeChoiceStore,
    asks_for_clock_time,
    choice_kind,
    is_explicit_keep_confirmation,
    parse_time_choice_payload,
    time_choice_buttons,
)
from common.calendar_import import CalendarImportError, parse_calendar_csv
from common.config import get_settings
from common.dialogue import asks_for_day, asks_for_duration
from common.repository import EventRepository
from common.schemas import DraftAction, EventStatus

logger = logging.getLogger(__name__)

_HELP = (
    "Привет! Я голосовой ассистент команды Нейм.Тим, помогу с задачами, встречами "
    "и напоминаниями.\n\n"
    "Напиши или скажи голосом: «Напоминай по понедельникам в 9:00 проверить отчёт». "
    "А чтобы узнать планы, просто спроси: «Что у меня завтра?»\n\n"
    "**Весь календарь** — в мини-приложении слева от поля ввода. "
    "Там можно посмотреть расписание и добавить запись.\n\n"
    "Есть расписание в Excel? Нажми кнопку ниже."
)
_FEATURES_HELP = (
    "**Что я умею**\n\n"
    "**Задачи и встречи**\n"
    "«Добавь задачу подготовить презентацию»\n"
    "«Поставь встречу с Анной завтра в 15:00 на полчаса»\n"
    "«Перенеси встречу с Анной на 16:00»\n\n"
    "**Напоминания**\n"
    "«Напомни сегодня в 18:00 позвонить маме»\n"
    "«Напоминай по понедельникам в 9:00 проверить отчёт»\n"
    "Чтобы изменить или остановить повтор, напиши «Повторяющиеся напоминания».\n\n"
    "**Календарь и поиск**\n"
    "«Что у меня завтра?»\n"
    "«Найди встречу с Анной»\n"
    "В мини-приложении можно посмотреть планы по дням, добавить запись "
    "и отметить задачу выполненной."
)
_IMPORT_HELP = (
    "**Как загрузить календарь из Excel**\n\n"
    "1. Назови колонки `id`, `date`, `time`, `text`.\n"
    "2. Сохрани таблицу как **CSV UTF-8** и прикрепи файл к сообщению.\n\n"
    "**Пример строки**\n"
    "`1,2026-11-13,14:30,Встреча с Марией`\n\n"
    "У каждой строки свой номер. Дата — ГГГГ-ММ-ДД, время — ЧЧ:ММ. "
    "Каждая запись займёт 1 час."
)
# Больше 1000 строк CSV для теста не нужно, а файл на мегабайты — повод отказать сразу.
_MAX_CSV_BYTES = 5_000_000
_BUSY = "Ещё думаю над прошлым запросом. Попробуй через несколько секунд."
_UPDATE_LEASE_SECONDS = 300
_RELEASE_UPDATE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class UpdateInProgress(RuntimeError):
    """A duplicate MAX delivery arrived while its original handler still owns a lease."""


def is_valid_update(update: Any) -> bool:
    """Check just the nested shapes the handler dereferences; missing bodies are valid."""
    if not isinstance(update, dict) or not isinstance(update.get("update_type"), str):
        return False
    kind = update["update_type"]
    if kind == "message_created":
        message = update.get("message")
        if not isinstance(message, dict) or not message:
            return False
        return all(
            value is None or isinstance(value, dict)
            for value in (message.get("sender"), message.get("recipient"), message.get("body"))
        )
    if kind == "message_callback":
        callback = update.get("callback")
        if not isinstance(callback, dict):
            return False
        user = callback.get("user")
        payload = callback.get("payload")
        return (user is None or isinstance(user, dict)) and (
            payload is None or isinstance(payload, str)
        )
    if kind == "bot_started":
        user = update.get("user")
        return user is None or isinstance(user, dict)
    return True


_AUDIO_EXT = (".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac")
CONFIRM_BUTTONS = [
    [
        {"type": "callback", "text": "Подтвердить", "payload": "draft:confirm"},
        {"type": "callback", "text": "Исправить", "payload": "draft:edit"},
        {"type": "callback", "text": "Отмена", "payload": "draft:cancel"},
    ]
]
QUICK_ACTIONS = [
    [{"type": "callback", "text": "Загрузить календарь", "payload": "quick:import"}],
    [{"type": "callback", "text": "Что я умею", "payload": "quick:features"}],
]


def confirmation_buttons(
    draft_id: UUID,
    *,
    can_change_time: bool = False,
    destructive: bool = False,
    saving: bool = False,
) -> list[list[dict[str, str]]]:
    if destructive:
        return [
            [
                {"type": "callback", "text": "Удалить", "payload": f"draft:confirm:{draft_id}"},
                {"type": "callback", "text": "Нет", "payload": f"draft:cancel:{draft_id}"},
            ]
        ]
    buttons = [
        [{**button, "payload": f"{button['payload']}:{draft_id}"} for button in row]
        for row in CONFIRM_BUTTONS
    ]
    if can_change_time:
        buttons.append(
            [{"type": "callback", "text": "Другое время", "payload": f"draft:time:{draft_id}"}]
        )
    if saving:
        buttons[0][0]["text"] = "Сохранить"
    return buttons


def can_change_draft_time(draft: DraftAction) -> bool:
    return draft.tool in {"create_event", "update_event"} and bool(draft.arguments.get("start_at"))


def _attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    body = message.get("body") or {}
    return [a for a in (body.get("attachments") or []) if isinstance(a, dict)]


def _attachment_filename(attachment: dict[str, Any]) -> str:
    payload = attachment.get("payload") or {}
    return str(
        payload.get("filename") or payload.get("name") or attachment.get("filename") or ""
    ).lower()


def extract_csv(message: dict[str, Any]) -> str | None:
    """URL вложенного CSV-файла, если он есть: им пользователь загружает календарь."""
    for attachment in _attachments(message):
        if attachment.get("type") not in {"file", None} or not _attachment_filename(
            attachment
        ).endswith(".csv"):
            continue
        payload = attachment.get("payload") or {}
        url = payload.get("url") or attachment.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def _is_audio_attachment(attachment: dict[str, Any]) -> bool:
    kind = attachment.get("type")
    if kind in {"audio", "voice"}:
        return True
    if kind != "file":
        return False
    if _attachment_filename(attachment).endswith(".csv"):
        return False  # это загрузка календаря, а не голосовое
    payload = attachment.get("payload") or {}
    filename = str(
        payload.get("filename") or payload.get("name") or attachment.get("filename") or ""
    ).lower()
    mime = str(payload.get("mime_type") or payload.get("content_type") or "").lower()
    if filename:
        return filename.endswith(_AUDIO_EXT)
    if mime:
        return mime.startswith("audio/")
    # MAX can omit filename/mime in update. Treat an untyped file as
    # an audio candidate in this voice-focused bot; ASR will reject non-audio.
    return True


def extract_audio(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (MAX audio download URL, optional transcription)."""
    for attachment in _attachments(message):
        if not _is_audio_attachment(attachment):
            continue
        payload = attachment.get("payload") or {}
        text = payload.get("transcription") or attachment.get("transcription")
        url = payload.get("url") or attachment.get("url")
        return (
            url if isinstance(url, str) and url else None,
            text.strip() if isinstance(text, str) and text.strip() else None,
        )
    return None, None


def _audio_present(message: dict[str, Any]) -> bool:
    return any(_is_audio_attachment(a) for a in _attachments(message))


def _untyped_file_present(message: dict[str, Any]) -> bool:
    """Вложение-файл без имени и mime: в апдейте оно выглядит и как CSV, и как аудио.

    Такие файлы нельзя различить по самому апдейту — подробности берём у /messages.
    """
    return any(
        a.get("type") == "file"
        and _is_audio_attachment(a)
        and not _attachment_filename(a)
        and not str(
            (a.get("payload") or {}).get("mime_type")
            or (a.get("payload") or {}).get("content_type")
            or ""
        )
        for a in _attachments(message)
    )


def _brief_attachment_diagnostics(message: dict[str, Any]) -> list[dict[str, Any]]:
    """No raw URL, token, transcription, or personal filename in logs."""
    return [
        {
            "type": a.get("type"),
            "url": bool((a.get("payload") or {}).get("url")),
            "filename": bool((a.get("payload") or {}).get("filename")),
            "transcription": bool((a.get("payload") or {}).get("transcription")),
        }
        for a in _attachments(message)
    ]


class BotHandler:
    def __init__(
        self,
        max_api: MaxAPI,
        asr: SpeechRecognizer,
        agent: AgentClient,
        state: StateStore,
        redis: Redis,
        repo: EventRepository | None = None,
    ):
        self.max_api = max_api
        self.asr = asr
        self.agent = agent
        self.state = state
        self.redis = redis
        self.repo = repo
        self.drafts = DraftStore(redis)
        self.time_choices = TimeChoiceStore(redis)
        self.recurring = (
            RecurringReminders(repo, redis, max_api, get_settings().tz)
            if repo is not None
            else None
        )

    async def _result(self, user_id, chat_id, result, to_user=True):
        buttons = None
        choice_id = None
        pending = None
        kind = None
        reply_text = result.reply_text
        if result.status == "needs_confirmation" and result.draft:
            buttons = confirmation_buttons(
                result.draft.id,
                can_change_time=can_change_draft_time(result.draft),
                destructive=result.draft.tool in {"delete_event", "delete_events"},
                saving=result.draft.tool in {"create_event", "update_event"},
            )
        elif result.status == "need_clarification":
            if asks_for_day(reply_text):
                kind = "day"
                week = " на следующей неделе" if "следующей неделе" in reply_text.casefold() else ""
                reply_text = f"На какой день{week} запланировать? Выбери день или напиши дату."
            elif asks_for_clock_time(reply_text):
                kind = "time"
            elif asks_for_duration(reply_text):
                kind = "duration"
            if kind:
                pending = await self.drafts.load(user_id)
                choice_id = uuid4()
                buttons = time_choice_buttons(choice_id, kind)
        await self.max_api.send(
            chat_id, reply_text or "Агент не вернул ответ", to_user=to_user, buttons=buttons
        )
        # Keep the previous draft/chooser usable if delivery fails. The user's state
        # lock prevents callbacks from racing the state publication after send.
        if result.status == "needs_confirmation" and result.draft:
            await self.drafts.save(user_id, result.draft)
        elif result.status == "error" and result.reply_text.startswith("Подтверждение устарело"):
            await self.drafts.drop(user_id)
        if choice_id is not None:
            await self.time_choices.issue(
                user_id, pending.id if pending is not None else None, choice_id=choice_id, kind=kind
            )
        else:
            await self.time_choices.clear(user_id)

    async def _send_guide(self, user_id: str, callback_id: str, payload: str, guide: str) -> None:
        delivery = f"bot:guide-delivery:{user_id}:{callback_id}"
        burst = f"bot:guide-burst:{user_id}:{payload}"
        claim = uuid4().hex
        if not await self.redis.set(delivery, claim, nx=True, ex=86400):
            return
        try:
            if await self.redis.set(burst, claim, nx=True, ex=5):
                await self.max_api.send(int(user_id), guide, to_user=True, text_format="markdown")
        except BaseException:
            await self.redis.eval(_RELEASE_UPDATE, 1, delivery, claim)
            await self.redis.eval(_RELEASE_UPDATE, 1, burst, claim)
            raise
        await self.max_api.answer_callback(callback_id)

    async def _callback(self, update):
        callback = update.get("callback") or {}
        user = (callback.get("user") or {}).get("user_id")
        payload = callback.get("payload") or ""
        callback_id = callback.get("callback_id")
        if user is None or not callback_id:
            return
        user_id = str(user)
        guide = {"quick:import": _IMPORT_HELP, "quick:features": _FEATURES_HELP}.get(payload)
        if guide:
            await self._send_guide(user_id, callback_id, payload, guide)
            return
        if payload.startswith("quick:"):
            question = {
                "quick:today": "Что у меня сегодня?",
                "quick:tomorrow": "Что у меня завтра?",
            }.get(payload)
            if question:
                await self.handle_update(
                    {
                        "update_type": "message_created",
                        "message": {
                            "sender": {"user_id": user},
                            "recipient": {"chat_type": "dialog"},
                            "body": {"mid": "callback:" + callback_id, "text": question},
                        },
                    }
                )
            await self.max_api.answer_callback(callback_id)
            return
        request_id = await self.state.acquire(user_id, "thinking")
        if request_id is None:
            await self.max_api.answer_callback(callback_id, _BUSY)
            return
        try:
            if payload.startswith("repeat:") and self.recurring:
                await self.recurring.callback(user_id, callback_id, payload)
                return
            if payload.startswith("event:"):
                await self._event_action(user_id, callback_id, payload)
                return
            elif payload.startswith("time-choice:"):
                parsed_choice = parse_time_choice_payload(payload)
                if parsed_choice is None:
                    await self.max_api.answer_callback(callback_id, "Выбор времени устарел")
                    return
                draft = await self.drafts.load(user_id)
                current_draft_id = draft.id if draft is not None else None
                choice_id, selection = parsed_choice
                kind = choice_kind(selection)
                if not await self.time_choices.matches(
                    user_id, choice_id, current_draft_id, kind=kind
                ):
                    await self.max_api.answer_callback(callback_id, "Выбор времени устарел")
                    return
                if selection not in CHOICE_REPLIES:
                    await self.max_api.send(
                        int(user),
                        {
                            "day": "Напиши или скажи голосом нужную дату.",
                            "duration": "Укажи длительность в минутах, например 45.",
                            "time": (
                                "Напиши или скажи голосом новое время. Остальные данные сохраню "
                                "и покажу новый вариант для подтверждения."
                            ),
                        }[kind],
                        to_user=True,
                    )
                    await self.time_choices.consume(user_id, choice_id, current_draft_id, kind=kind)
                    await self.max_api.answer_callback(callback_id)
                else:
                    answer = CHOICE_REPLIES[selection]
                    await self.max_api.answer_callback(callback_id, f"Выбрано: {answer}")
                    await self.handle_update(
                        {
                            "update_type": "message_created",
                            "message": {
                                "sender": {"user_id": user},
                                "recipient": {"chat_type": "dialog"},
                                "body": {
                                    "mid": "callback:" + callback_id,
                                    "text": answer,
                                },
                            },
                        },
                        _state_request_id=request_id,
                    )
                    await self.time_choices.consume(user_id, choice_id, current_draft_id, kind=kind)
            else:
                draft = await self.drafts.load(user_id)
                parts = payload.split(":")
                if len(parts) != 3 or parts[0] != "draft" or not draft or parts[2] != str(draft.id):
                    await self.max_api.answer_callback(callback_id, "Черновик устарел")
                    return
                action = parts[1]
                if action == "cancel":
                    await self.agent.reset(user_id)
                    await self.drafts.drop(user_id)
                    await self.time_choices.clear(user_id)
                    await self.max_api.send(int(user), "Отменил действие.", to_user=True)
                    await self.max_api.answer_callback(callback_id, "Отменено")
                elif action == "edit":
                    if not await self.drafts.is_editing(user_id, draft):
                        await self.max_api.send(
                            int(user),
                            "Что изменить: название, время или длительность?",
                            to_user=True,
                        )
                        await self.drafts.begin_edit(user_id, draft)
                    await self.max_api.answer_callback(callback_id)
                elif action == "time":
                    if not can_change_draft_time(draft):
                        await self.max_api.answer_callback(
                            callback_id, "В этом действии нет времени"
                        )
                        return
                    choice_id = await self.time_choices.issue(user_id, draft.id)
                    await self.max_api.send(
                        int(user),
                        "Какой день и время выбрать? Нажми на один из вариантов ниже или "
                        "выбери «Другое время», чтобы написать или сказать его голосом. "
                        "Остальные данные сохраню и покажу новый вариант для подтверждения.",
                        to_user=True,
                        buttons=time_choice_buttons(choice_id),
                    )
                    await self.drafts.begin_edit(user_id, draft)
                    await self.max_api.answer_callback(callback_id)
                elif action == "confirm":
                    result = await self.agent.confirm(user_id, draft)
                    succeeded = result.status == "done"
                    if succeeded:
                        await self.drafts.drop(user_id)
                    await self._result(user_id, int(user), result)
                    await self.max_api.answer_callback(
                        callback_id,
                        "✅ Подтверждено"
                        if succeeded
                        else "Черновик устарел. Отправьте запрос заново"
                        if result.reply_text.startswith("Подтверждение устарело")
                        else "Не получилось выполнить. Черновик сохранён",
                    )
                else:
                    await self.max_api.answer_callback(callback_id, "Черновик устарел")
        finally:
            await self.state.release(user_id, request_id)

    async def _event_action(self, user_id: str, callback_id: str, payload: str) -> None:
        """Кнопки под уведомлением: «Готово» или «На час позже».

        repo сам ограничивает действие владельцем записи: чужой user_id вернёт None,
        и база не изменится — но колбэк всё равно отвечаем, чтобы кнопка не крутилась.
        """
        if self.repo is None:
            await self.max_api.answer_callback(callback_id, "Действие недоступно")
            return
        _, action, raw_id = payload.split(":", 2)
        try:
            event_id = UUID(raw_id)
        except ValueError:
            await self.max_api.answer_callback(callback_id, "Не поняла, о какой записи речь")
            return
        if action == "done":
            event = await self.repo.set_status(user_id, event_id, EventStatus.DONE)
            await self.max_api.answer_callback(
                callback_id, "Готово 👍" if event else "Запись не найдена"
            )
        elif action == "snooze":
            new_start = datetime.now(get_settings().tz) + timedelta(hours=1)
            event = await self.repo.postpone(user_id, event_id, new_start)
            await self.max_api.answer_callback(
                callback_id, "Отложила на час." if event else "Запись не найдена"
            )
        else:
            await self.max_api.answer_callback(callback_id, "Неизвестное действие")

    async def _send_welcome(self, chat_id: int, *, to_user: bool = False) -> None:
        await self.max_api.send(
            chat_id, _HELP, to_user=to_user, buttons=QUICK_ACTIONS, text_format="markdown"
        )

    async def _import_csv(
        self, user_id: str, url: str, reply: Callable[[str], Awaitable[None]]
    ) -> None:
        """Загрузить присланный CSV в тот же календарь, где живут созданные голосом записи."""
        if self.repo is None:
            await reply("Загрузка календаря сейчас недоступна.")
            return
        await reply("Загружаю календарь из файла…")
        try:
            data = await self.max_api.get_file(url, _MAX_CSV_BYTES, "файл календаря")
            rows, errors = parse_calendar_csv(data, get_settings().tz)
            imported = await self.repo.import_calendar(user_id, rows)
        except (MaxAPIError, CalendarImportError) as exc:
            await reply(f"Не получилось загрузить календарь: {exc}")
            return

        skipped = len(rows) - imported
        parts = [f"Загрузил событий: {imported}."]
        if skipped:
            parts.append(f"Уже были в календаре: {skipped}.")
        if errors:
            parts.append("Пропущено строк с ошибками: " + str(len(errors)))
            parts.append("Например — " + "; ".join(errors[:3]))
        parts.append("Теперь спрашивай: «что у меня завтра?»")
        await reply(" ".join(parts))

    async def handle_update(
        self, update: dict[str, Any], *, _state_request_id: str | None = None
    ) -> None:
        """Handle an update and persist message dedupe only after successful work.

        Redis distinguishes a leased in-progress delivery from a completed one.
        A crash leaves only the expiring lease, so polling/webhook retries remain
        possible; stable request IDs make committed agent writes safe to repeat if
        the crash happened before completion was recorded.
        """
        message = update.get("message") or {}
        mid = (message.get("body") or {}).get("mid")
        if update.get("update_type") != "message_created" or mid is None:
            if _state_request_id is None:
                await self._handle_update(update)
            else:
                await self._handle_update(update, _state_request_id=_state_request_id)
            return

        mid = str(mid)
        done_key = f"bot:done:{mid}"
        pending_key = f"bot:processing:{mid}"
        if await self.redis.get(done_key):
            return

        claim = uuid4().hex
        if not await self.redis.set(pending_key, claim, nx=True, ex=_UPDATE_LEASE_SECONDS):
            raise UpdateInProgress("MAX update is already being processed")
        try:
            # Close the race where another worker completed after the first read.
            if await self.redis.get(done_key):
                return
            if _state_request_id is None:
                await self._handle_update(update)
            else:
                await self._handle_update(update, _state_request_id=_state_request_id)
            await self.redis.set(done_key, "1", ex=86400)
        finally:
            await self.redis.eval(_RELEASE_UPDATE, 1, pending_key, claim)

    async def _handle_update(
        self, update: dict[str, Any], *, _state_request_id: str | None = None
    ) -> None:
        kind = update.get("update_type")
        if kind == "message_callback":
            await self._callback(update)
            return
        if kind == "bot_started":
            user = update.get("user") or {}
            user_id = user.get("user_id")
            chat_id = update.get("chat_id")
            if user_id is not None:
                # Представились здесь — не повторяем приветствие перед первым ответом.
                await self.redis.set(f"bot:greeted:{user_id}", "1")
            if chat_id is not None:
                await self._send_welcome(int(chat_id))
            elif user_id is not None:
                await self._send_welcome(int(user_id), to_user=True)
            return
        if kind != "message_created":
            return

        message = update.get("message") or {}
        mid = (message.get("body") or {}).get("mid")
        sender = message.get("sender") or {}
        if sender.get("is_bot"):
            return
        user_id = sender.get("user_id")
        recipient = message.get("recipient") or {}
        chat_id = recipient.get("chat_id")
        # Private assistant: don't disclose task information in a group or channel.
        if recipient.get("chat_type") not in (None, "dialog"):
            logger.info(
                "MAX message skipped: not a dialog (chat_type=%s)", recipient.get("chat_type")
            )
            return
        if user_id is None:
            return
        to_user = chat_id is None
        chat_id = int(user_id if to_user else chat_id)
        user_id = str(user_id)
        if _state_request_id is None:
            await self.time_choices.clear(user_id)

        async def reply(message_text: str) -> None:
            await self.max_api.send(chat_id, message_text, to_user=to_user)

        text = ((message.get("body") or {}).get("text") or "").strip()
        if text in {"/start", "/help"}:
            await self.redis.set(f"bot:greeted:{user_id}", "1")
            await self._send_welcome(chat_id, to_user=to_user)
            return

        # Первое сообщение от незнакомого пользователя: сначала представимся.
        if await self.redis.set(f"bot:greeted:{user_id}", "1", nx=True):
            await self._send_welcome(chat_id, to_user=to_user)

        csv_url = extract_csv(message)
        if csv_url:
            await self._import_csv(user_id, csv_url, reply)
            return
        logger.info(
            "MAX incoming: text=%s attachments=%s",
            bool(text),
            _brief_attachment_diagnostics(message),
        )
        url, transcription = extract_audio(message)
        is_voice = _audio_present(message)
        fetched: dict[str, Any] | None = None

        # The MAX event may omit payload.url/filename (or all attachments).
        # The message endpoint can return richer details for this same mid.
        # Файл без имени может оказаться CSV-календарём, поэтому подробности нужны
        # до того, как решаем: это голос или файл.
        if (
            not text
            and mid
            and (not is_voice or not (url or transcription) or _untyped_file_present(message))
        ):
            try:
                full_message = await self.max_api.get_message(str(mid))
                fetched = full_message
                logger.info(
                    "MAX fetched message: attachments=%s",
                    _brief_attachment_diagnostics(full_message),
                )
                fetched_url, fetched_transcription = extract_audio(full_message)
                if _audio_present(full_message):
                    is_voice = True
                    url = fetched_url or url
                    transcription = fetched_transcription or transcription
            except MaxAPIError as exc:
                logger.warning("MAX message details unavailable: %s", exc)

        # CSV без имени в апдейте: узнаём о нём только из подробностей сообщения.
        if fetched is not None:
            csv_url = extract_csv(fetched)
            if csv_url:
                await self._import_csv(user_id, csv_url, reply)
                return

        if is_voice and not (url or transcription):
            logger.warning(
                "Audio without URL/transcription, details=%s",
                _brief_attachment_diagnostics(message),
            )
            await reply(
                "Аудиосообщение получено, но MAX не передал ссылку для распознавания. "
                "Попробуй отправить голосовое ещё раз."
            )
            return
        if not text and not is_voice:
            await reply("Пришли текст или голосовое сообщение. Для MP3 выбери отправку файла.")
            return

        owns_state_lock = _state_request_id is None
        request_id = _state_request_id or await self.state.acquire(
            user_id, "recognizing" if is_voice else "thinking"
        )
        if request_id is None:
            await reply(_BUSY)
            return

        try:
            if is_voice:
                await reply("🎤 Голос получил, распознаю…")
                if transcription:
                    text = transcription
                else:
                    audio = await self.max_api.get_audio(url)
                    text = await self.asr.transcribe(audio)
                if not text.strip():
                    raise SpeechRecognitionError("Не получилось распознать речь")
                await self.state.transition(user_id, request_id)
                # Распознанный текст пользователю не показываем: он бывает кривым,
                # а ответ агент всё равно даёт верный (фидбек QA от роли E).
                await reply("⏳ Записал в блокнот, думаю…")
            else:
                await reply("⏳ Думаю над запросом…")

            if self.recurring and await self.recurring.message(
                user_id,
                text,
                uuid5(NAMESPACE_URL, f"repeat:{user_id}:{mid or request_id}"),
                is_voice=is_voice,
            ):
                if await self.drafts.load(user_id):
                    await self.agent.reset(user_id)
                    await self.drafts.drop(user_id)
                return

            pending = await self.drafts.load(user_id)
            keep_phrase = pending is not None and is_explicit_keep_confirmation(text)
            if (
                pending
                and starts_new_request(text)
                and (asks_to_read(text) or not await self.drafts.is_editing(user_id, pending))
            ):
                await self.agent.reset(user_id)
                await self.drafts.drop(user_id)
                pending = None
            if keep_phrase and not can_change_draft_time(pending):
                await self.agent.reset(user_id)
                await self.drafts.drop(user_id)
                await reply("Оставил календарь без изменений.")
                return
            if pending and (
                text.casefold().strip(" .!") in {"да", "подтвердить", "подтверждаю"} or keep_phrase
            ):
                if is_voice:
                    await self._result(
                        user_id,
                        chat_id,
                        AgentMessageResponse(
                            reply_text=pending.preview,
                            status="needs_confirmation",
                            draft=pending,
                        ),
                        to_user,
                    )
                    return
                result = await self.agent.confirm(user_id, pending)
                if result.status == "done":
                    await self.drafts.drop(user_id)
            elif pending and text.casefold().strip(" .!") in {
                "нет",
                "отмена",
                "отмени",
                "забей",
                "не надо",
                "забыли",
            }:
                await self.agent.reset(user_id)
                await self.drafts.drop(user_id)
                await self.time_choices.clear(user_id)
                await reply("Отменил действие.")
                return
            else:
                result = await self.agent.ask(
                    user_id=user_id,
                    text=text,
                    # Voice and corrections of its pending draft always need review.
                    # AgentService retains this rule across clarification replies.
                    confirm_changes=is_voice or pending is not None,
                    request_id=str(uuid5(NAMESPACE_URL, f"max:{user_id}:{mid}")) if mid else None,
                    **({"edit_draft_id": str(pending.id)} if pending is not None else {}),
                )
            await self._result(user_id, chat_id, result, to_user)
        except (SpeechRecognitionError, MaxAPIError) as exc:
            logger.warning("Audio problem for user %s: %s", user_id, exc)
            await reply(f"Не получилось обработать голос: {exc}. Попробуй ещё раз.")
        except AgentUnavailable:
            logger.exception("Agent request failed for user %s", user_id)
            await reply("Агент временно недоступен. Попробуй ещё раз позже.")
        finally:
            if owns_state_lock:
                await self.state.release(user_id, request_id)
