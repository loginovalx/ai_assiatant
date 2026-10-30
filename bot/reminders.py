"""Фоновая отправка напоминаний в MAX, когда наступило их время.

Источник истины — таблица events, а не память процесса: после перезапуска бота
уже созданные напоминания никуда не денутся, а отправленные повторно не придут
(отметка notified_at ставится в том же запросе, которым событие забирается в работу).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from bot.max_api import MaxAPI, MaxAPIError
from common.repository import EventRepository
from common.schemas import Event, EventType

logger = logging.getLogger(__name__)


def event_buttons(event_id: UUID) -> list[list[dict[str, str]]]:
    """Кнопки под уведомлением: закрыть напоминание или отложить на час."""
    return [
        [
            {"type": "callback", "text": "Готово", "payload": f"event:done:{event_id}"},
            {"type": "callback", "text": "На час позже", "payload": f"event:snooze:{event_id}"},
        ]
    ]


def format_notification(event: Event) -> str:
    """Текст напоминания. Для встреч и дел добавляем время: оно помогает сориентироваться."""
    if event.type == EventType.REMINDER:
        return f"Напоминание: {event.name}"
    kind = "встреча" if event.type == EventType.MEETING else "дело"
    when = event.start_at.strftime("%H:%M") if event.start_at else ""
    return f"Напоминание: {kind} «{event.name}» в {when}"


class ReminderSender:
    """Раз в interval секунд проверяет базу и шлёт наступившие напоминания.

    lookback ограничивает «догоняющие» отправки: если бот лежал сутки, пользователь
    получит только то, что наступило за последний час, а не всю пропущенную историю.
    В импортированном календаре встречаются и прошлогодние даты — их не трогаем.
    """

    def __init__(
        self,
        repo: EventRepository,
        max_api: MaxAPI,
        tz: ZoneInfo,
        interval_seconds: int = 30,
        lookback_minutes: int = 60,
        batch: int = 50,
    ) -> None:
        self._repo = repo
        self._max_api = max_api
        self._tz = tz
        self._interval = interval_seconds
        self._lookback = timedelta(minutes=lookback_minutes)
        self._batch = batch

    async def run_forever(self) -> None:
        logger.info("Планировщик напоминаний запущен: раз в %d с", self._interval)
        while True:
            try:
                sent = await self.tick()
                if sent:
                    logger.info("Отправлено напоминаний: %d", sent)
            except asyncio.CancelledError:
                raise
            except Exception:  # один сбой не должен останавливать планировщик навсегда
                logger.exception("Сбой тика планировщика напоминаний")
            await asyncio.sleep(self._interval)

    async def tick(self, now: datetime | None = None) -> int:
        """Один проход: забрать наступившее и отправить. Возвращает число отправленных."""
        now = now or datetime.now(self._tz)
        due = await self._repo.claim_due_events(now, self._lookback, self._batch)
        sent = 0
        for event in due:
            try:
                # chat_id у личного диалога с ботом совпадает с user_id отправителя.
                await self._max_api.send(
                    int(event.user_id),
                    format_notification(event),
                    to_user=True,
                    buttons=event_buttons(event.id),
                )
                sent += 1
            except (MaxAPIError, ValueError, OSError) as exc:
                # Снимаем отметку: пользователь не получил сообщение, пробуем в следующий раз.
                logger.warning("Не удалось отправить напоминание %s: %s", event.id, exc)
                await self._repo.unmark_notified(event.id)
            except Exception:
                logger.exception("Неожиданная ошибка при отправке напоминания %s", event.id)
                await self._repo.unmark_notified(event.id)
        return sent
