"""MAX controls for recurring reminders. All choices use validated, structured values."""

import re
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from common.display import format_calendar_time
from common.reminder_series import DAYS, ReminderRule, parse_clock, parse_command
from common.schemas import DRAFT_TTL_SECONDS

_EXAMPLE = (
    "Для повтора укажи день и время. Например: "
    "«Поставь напоминание на каждый вторник в 17:48 полить цветы». "
    "Можно также выбрать «каждый день»."
)


class RepeatDraft(BaseModel):
    id: UUID
    series_id: UUID
    revision: UUID | None = None
    rule: ReminderRule
    waiting_for_time: bool = False
    first_at: datetime | None = None


def _button(label, action, ident):
    return {"type": "callback", "text": label, "payload": f"repeat:{action}:{ident}"}


class RecurringReminders:
    def __init__(self, repo, redis, max_api, tz):
        self.repo = repo
        self.redis = redis
        self.api = max_api
        self.tz = tz

    def _key(self, user):
        return f"bot:repeat:{user}"

    async def _load(self, user):
        raw = await self.redis.get(self._key(user))
        return RepeatDraft.model_validate_json(raw) if raw else None

    async def _store(self, user, draft):
        await self.redis.set(self._key(user), draft.model_dump_json(), ex=DRAFT_TTL_SECONDS)

    async def _send(self, user, text, buttons=None):
        await self.api.send(int(user), text, to_user=True, buttons=buttons)

    async def _preview(self, user, draft):
        rule = draft.rule
        first = rule.first_after(datetime.now(self.tz))
        await self._send(
            user,
            f"Напоминание «{rule.name}»\n{rule.schedule}.\n"
            f"Первое напоминание — {format_calendar_time(first)}.\n"
            "Повтор будет работать, пока ты его не остановишь.",
            [
                [
                    _button("Сохранить", "save", draft.id),
                    _button("Изменить", "edit", draft.id),
                    _button("Отмена", "cancel", draft.id),
                ]
            ],
        )
        await self._store(user, draft.model_copy(update={"first_at": first}))

    async def message(self, user, text, series_id):
        value = text.strip(" .!?").casefold()
        if value in {"повторяющиеся напоминания", "покажи повторяющиеся напоминания"}:
            await self.redis.delete(self._key(user))
            rules = await self.repo.list_reminder_series(user)
            if not rules:
                await self._send(user, "Активных повторов пока нет.")
            for rule in rules:
                await self._send(
                    user,
                    f"«{rule.name}»\n{rule.schedule}.",
                    [[_button("Настроить повтор", "manage", rule.id)]],
                )
            return True
        try:
            rule = parse_command(text)
        except ValueError:
            await self._send(user, _EXAMPLE)
            return True
        if rule:
            await self._preview(user, RepeatDraft(id=uuid4(), series_id=series_id, rule=rule))
            return True
        draft = await self._load(user)
        if draft and value in {"отмена", "отмени", "нет"}:
            await self.redis.delete(self._key(user))
            await self._send(user, "Отменил изменения.")
            return True
        if draft and draft.waiting_for_time and re.match(r"^\s*\d", text):
            clock = parse_clock(text)
            if clock is None:
                await self._send(user, "Введи время в формате ЧЧ:ММ, например 09:30.")
                return True
            rule = draft.rule.model_copy(update={"clock": clock})
            await self._preview(
                user,
                draft.model_copy(update={"id": uuid4(), "rule": rule, "waiting_for_time": False}),
            )
            return True
        if draft:
            await self.redis.delete(self._key(user))
        return False

    async def callback(self, user, callback_id, payload):
        try:
            _, action, raw_id = payload.split(":", 2)
            ident = UUID(raw_id)
        except ValueError:
            await self.api.answer_callback(callback_id, "Кнопка устарела")
            return
        now = datetime.now(self.tz)
        reply = None
        if action in {"manage", "stop", "stopyes", "change"}:
            rule = next(
                (r for r in await self.repo.list_reminder_series(user) if r.id == ident), None
            )
            if rule is None:
                reply = "Повтор уже остановлен или недоступен"
            elif action == "manage":
                await self._send(
                    user,
                    f"«{rule.name}»\n{rule.schedule}.",
                    [
                        [
                            _button("Изменить", "change", ident),
                            _button("Остановить повтор", "stop", ident),
                        ]
                    ],
                )
            elif action == "stop":
                await self._send(
                    user,
                    f"Остановить будущие напоминания «{rule.name}»?",
                    [
                        [
                            _button("Остановить повтор", "stopyes", ident),
                            _button("Назад", "manage", ident),
                        ]
                    ],
                )
            elif action == "stopyes":
                await self.repo.stop_reminder_series(user, ident, now)
                await self._send(user, f"Повтор «{rule.name}» остановлен.")
            else:
                await self._choose(
                    user,
                    RepeatDraft(id=uuid4(), series_id=ident, revision=rule.revision, rule=rule),
                )
        else:
            draft = await self._load(user)
            if draft is None or draft.id != ident:
                reply = "Черновик устарел"
            elif action == "edit":
                await self._choose(user, draft)
            elif action == "time":
                await self._store(user, draft.model_copy(update={"waiting_for_time": True}))
                await self._send(user, "Напиши новое время, например 10:30.")
            elif action == "cancel":
                await self.redis.delete(self._key(user))
                await self._send(user, "Отменил изменения.")
            elif action == "save" and draft.first_at is not None and draft.first_at <= now:
                await self._preview(user, draft.model_copy(update={"id": uuid4()}))
                reply = "Первая дата прошла. Проверь новое расписание."
            elif action == "save" and draft.first_at is not None:
                saved = await self.repo.save_reminder_series(
                    user, draft.series_id, draft.rule, now, draft.revision, operation_id=draft.id
                )
                if saved:
                    await self._send(
                        user,
                        f"Повтор «{draft.rule.name}» сохранён.\n{draft.rule.schedule}.",
                        [[_button("Настроить повтор", "manage", draft.series_id)]],
                    )
                else:
                    await self._send(user, "Повтор изменился. Открой его настройки заново.")
                await self.redis.delete(self._key(user))
            elif action in {f"day{i}" for i in range(8)}:
                day = int(action[-1])
                rule = draft.rule.model_copy(update={"weekday": day if day < 7 else None})
                await self._preview(
                    user,
                    draft.model_copy(
                        update={"id": uuid4(), "rule": rule, "waiting_for_time": False}
                    ),
                )
            else:
                reply = "Кнопка устарела"
        await self.api.answer_callback(callback_id, reply)

    async def _choose(self, user, draft):
        days = [_button(day, f"day{i}", draft.id) for i, day in enumerate(DAYS)]
        await self._send(
            user,
            "Выбери новый день или измени время.",
            [
                days[:3],
                days[3:6],
                [days[6], _button("Каждый день", "day7", draft.id)],
                [
                    _button("Изменить время", "time", draft.id),
                    _button("Отмена", "cancel", draft.id),
                ],
            ],
        )
        await self._store(user, draft)
