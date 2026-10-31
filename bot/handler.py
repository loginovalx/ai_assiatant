"""MAX message routing + user-friendly progress + Redis FSM."""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from redis.asyncio import Redis

from bot.agent_client import AgentClient, AgentUnavailable
from bot.asr import SpeechRecognitionError, SpeechRecognizer
from bot.drafts import DraftStore
from bot.max_api import MaxAPI, MaxAPIError
from bot.state import StateStore
from common.calendar_import import CalendarImportError, parse_calendar_csv
from common.config import get_settings
from common.repository import EventRepository
from common.schemas import EventStatus

logger = logging.getLogger(__name__)

_HELP = (
    "Привет! Я твой персональный AI-ассистент.\n\n"
    "Помогаю вести дела и расписание: создаю встречи и напоминания, нахожу события "
    "в календаре, рассказываю о предстоящем и напоминаю о нём в нужное время.\n\n"
    "Напиши или скажи голосом, например:\n"
    "• «Что у меня завтра?»\n"
    "• «Напомни в 18:00 отправить отчёт»\n"
    "• «Поставь встречу с Марией на пятницу в 14:00»\n"
    "• «Когда у меня следующая встреча с Иваном?»\n\n"
    "Можно обычным языком — я разберусь. А ещё можно прислать CSV-файл "
    "с календарём (колонки id, date, time, text), и я загружу его себе."
)
# Больше 1000 строк CSV для теста не нужно, а файл на мегабайты — повод отказать сразу.
_MAX_CSV_BYTES = 5_000_000
_BUSY = "Ещё думаю над прошлым запросом. Попробуй через несколько секунд."
_AUDIO_EXT = (".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac")
CONFIRM_BUTTONS = [
    [
        {"type": "callback", "text": "Подтвердить", "payload": "draft:confirm"},
        {"type": "callback", "text": "Исправить", "payload": "draft:edit"},
        {"type": "callback", "text": "Отмена", "payload": "draft:cancel"},
    ]
]
QUICK_ACTIONS = [
    [
        {"type": "callback", "text": "Сегодня", "payload": "quick:today"},
        {"type": "callback", "text": "Завтра", "payload": "quick:tomorrow"},
    ]
]


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
        _is_audio_attachment(a)
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

    async def _result(self, user_id, chat_id, result, to_user=True):
        buttons = None
        if result.status == "needs_confirmation" and result.draft:
            await self.drafts.save(user_id, result.draft)
            buttons = CONFIRM_BUTTONS
        await self.max_api.send(
            chat_id, result.reply_text or "Агент не вернул ответ", to_user=to_user, buttons=buttons
        )

    async def _callback(self, update):
        callback = update.get("callback") or {}
        user = (callback.get("user") or {}).get("user_id")
        payload = callback.get("payload") or ""
        callback_id = callback.get("callback_id")
        if user is None or not callback_id:
            return
        user_id = str(user)
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
            if payload.startswith("event:"):
                await self._event_action(user_id, callback_id, payload)
                return
            draft = await self.drafts.load(user_id)
            if payload == "draft:cancel":
                await self.drafts.drop(user_id)
                await self.max_api.send(int(user), "Отменил действие.", to_user=True)
                await self.max_api.answer_callback(callback_id, "Отменено")
            elif payload == "draft:edit":
                await self.max_api.send(int(user), "Что поправить?", to_user=True)
                await self.max_api.answer_callback(callback_id)
            elif payload == "draft:confirm" and draft:
                result = await self.agent.confirm(user_id, draft)
                succeeded = result.status == "done"
                if succeeded:
                    await self.drafts.drop(user_id)
                await self._result(user_id, int(user), result)
                await self.max_api.answer_callback(
                    callback_id,
                    "✅ Подтверждено"
                    if succeeded
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

    async def _greet_if_new(self, user_id: str, reply: Callable[[str], Awaitable[None]]) -> None:
        """Приветствие ровно один раз на пользователя: флаг переживает перезапуск бота."""
        if await self.redis.set(f"bot:greeted:{user_id}", "1", nx=True):
            await reply(_HELP)

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

    async def handle_update(self, update: dict[str, Any]) -> None:
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
                await self.max_api.send(int(chat_id), _HELP)
            elif user_id is not None:
                await self.max_api.send(int(user_id), _HELP, to_user=True)
            return
        if kind != "message_created":
            return

        message = update.get("message") or {}
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

        async def reply(message_text: str) -> None:
            await self.max_api.send(chat_id, message_text, to_user=to_user)

        mid = (message.get("body") or {}).get("mid")
        if mid is not None:
            is_new = await self.redis.set(f"bot:seen:{mid}", "1", nx=True, ex=86400)
            if not is_new:
                return

        text = ((message.get("body") or {}).get("text") or "").strip()
        if text in {"/start", "/help"}:
            await self.redis.set(f"bot:greeted:{user_id}", "1")
            await self.max_api.send(chat_id, _HELP, to_user=to_user, buttons=QUICK_ACTIONS)
            return

        # Первое сообщение от незнакомого пользователя: сначала представимся.
        await self._greet_if_new(user_id, reply)

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

        request_id = await self.state.acquire(user_id, "recognizing" if is_voice else "thinking")
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

            pending = await self.drafts.load(user_id)
            if pending and text.casefold().strip(" .!") in {"да", "подтвердить", "подтверждаю"}:
                result = await self.agent.confirm(user_id, pending)
                if result.status == "done":
                    await self.drafts.drop(user_id)
            elif pending and text.casefold().strip(" .!") in {"нет", "отмена", "отмени"}:
                await self.drafts.drop(user_id)
                await reply("Отменил действие.")
                return
            else:
                result = await self.agent.ask(
                    user_id=user_id, text=text, confirm_changes=is_voice or pending is not None
                )
            await self._result(user_id, chat_id, result, to_user)
        except (SpeechRecognitionError, MaxAPIError) as exc:
            logger.warning("Audio problem for user %s: %s", user_id, exc)
            await reply(f"Не получилось обработать голос: {exc}. Попробуй ещё раз.")
        except AgentUnavailable:
            logger.exception("Agent request failed for user %s", user_id)
            await reply("Агент временно недоступен. Попробуй ещё раз позже.")
        finally:
            await self.state.release(user_id, request_id)
