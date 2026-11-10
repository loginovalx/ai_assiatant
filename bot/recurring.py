"""MAX controls for recurring reminders. All choices use validated, structured values."""

import re
from datetime import datetime, time
from uuid import UUID, uuid4

from pydantic import BaseModel

from common.display import format_calendar_time
from common.reminder_series import DAYS, ReminderRule, complete_clock, parse_clock, parse_request
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
    choosing_daypart: bool = False
    first_at: datetime | None = None


def _button(label, action, ident):
    return {"type": "callback", "text": label, "payload": f"repeat:{action}:{ident}"}


def repeat_actions(series_id: UUID) -> list[list[dict[str, str]]]:
    return [[_button("Изменить", "change", series_id), _button("Отключить", "stop", series_id)]]


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
            "Сохранить повтор?",
            [
                [
                    _button("Да", "save", draft.id),
                    _button("Нет", "cancel", draft.id),
                ]
            ],
        )
        await self._store(user, draft.model_copy(update={"first_at": first}))

    async def _ask_daypart(self, user, draft):
        rule = draft.rule
        early = f"{rule.clock:%H:%M}"
        late = f"{rule.clock.hour + 12:02}:{rule.clock.minute:02}"
        await self._store(user, draft)
        await self._send(
            user,
            f"Напоминание «{rule.name}»\n{rule.cadence_label}. Ты имел в виду {early} или {late}? "
            "Можно ответить голосом.",
            [
                [
                    _button(early, "early", draft.id),
                    _button(late, "late", draft.id),
                    _button("Отмена", "cancel", draft.id),
                ]
            ],
        )

    async def _preview_at(self, user, draft, clock):
        rule = draft.rule.model_copy(update={"clock": clock})
        await self._preview(
            user,
            draft.model_copy(
                update={
                    "id": uuid4(),
                    "rule": rule,
                    "waiting_for_time": False,
                    "choosing_daypart": False,
                }
            ),
        )

    async def message(self, user, text, series_id, *, is_voice=False):
        value = text.strip(" .!?").casefold()
        source_key = self._key(user) + ":source"
        draft = await self._load(user)
        source = await self.redis.get(source_key)
        if (draft or source) and value in {"отмена", "отмени", "нет"}:
            await self.redis.delete(self._key(user))
            await self.redis.delete(source_key)
            await self._send(user, "Отменил изменения.")
            return True
        if draft and value in {"да", "подтверждаю", "сохранить", "подтвердить"}:
            if draft.choosing_daypart:
                await self._ask_daypart(user, draft)
            elif draft.waiting_for_time:
                await self._send(user, "Напиши новое время, например 10:30.")
            elif is_voice:
                await self._send(user, "Нажми «Да» под черновиком.")
            else:
                await self.callback(user, None, f"repeat:save:{draft.id}")
            return True
        if draft and not draft.choosing_daypart and value == "исправить":
            await self._choose(user, draft)
            return True
        if value in {"повторяющиеся напоминания", "покажи повторяющиеся напоминания"}:
            await self.redis.delete(self._key(user))
            await self.redis.delete(source_key)
            rules = await self.repo.list_reminder_series(user)
            if not rules:
                await self._send(user, "Активных повторов пока нет.")
            for rule in rules:
                await self._send(
                    user,
                    f"Напоминание «{rule.name}»\n{rule.schedule}.",
                    repeat_actions(rule.id),
                )
            return True
        clock_answer = parse_clock(text)
        if draft and draft.choosing_daypart:
            if value in {"утром", "ночи", "ночью", "днем", "днём", "вечером"}:
                offset = 12 if value in {"днем", "днём", "вечером"} else 0
                clock_answer = draft.rule.clock.replace(hour=draft.rule.clock.hour + offset)
            if clock_answer is not None:
                await self._preview_at(user, draft, clock_answer)
                return True
        if (
            source
            and not clock_answer
            and (
                value in {"да", "подтверждаю", "сохранить", "подтвердить"}
                or re.match(r"^\s*(?:в\s+)?\d", text)
            )
        ):
            await self._send(user, "Укажи время, например «в 10:00» или «в десять утра».")
            return True
        candidate = complete_clock(source, clock_answer) if source and clock_answer else text
        try:
            parsed = parse_request(candidate)
        except ValueError:
            await self.redis.delete(self._key(user))
            try:
                can_fix_clock = parse_request(complete_clock(candidate, time(10))) is not None
            except ValueError:
                can_fix_clock = False
            if can_fix_clock:
                await self.redis.set(source_key, candidate, ex=DRAFT_TTL_SECONDS)
                await self._send(
                    user, "Во сколько напоминать? Например, «в 10:00» или «в десять утра»."
                )
            else:
                await self.redis.delete(source_key)
                await self._send(user, _EXAMPLE)
            return True
        if parsed:
            rule, ambiguous = parsed
            new_draft = RepeatDraft(
                id=uuid4(), series_id=series_id, rule=rule, choosing_daypart=is_voice and ambiguous
            )
            if new_draft.choosing_daypart:
                await self._ask_daypart(user, new_draft)
            else:
                await self._preview(user, new_draft)
            await self.redis.delete(source_key)
            return True
        if (
            draft
            and draft.waiting_for_time
            and (clock_answer or re.match(r"^\s*(?:в\s+)?\d", text))
        ):
            if clock_answer is None:
                await self._send(user, "Введи время в формате ЧЧ:ММ, например 09:30.")
                return True
            await self._preview_at(user, draft, clock_answer)
            return True
        if draft:
            await self.redis.delete(self._key(user))
        await self.redis.delete(source_key)
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
                    f"Напоминание «{rule.name}»\n{rule.schedule}.",
                    repeat_actions(ident),
                )
            elif action == "stop":
                await self._send(
                    user,
                    f"Отключить повтор? Напоминания «{rule.name}» больше не будут приходить.",
                    [
                        [
                            _button("Отключить", "stopyes", ident),
                            _button("Назад", "manage", ident),
                        ]
                    ],
                )
            elif action == "stopyes":
                await self.repo.stop_reminder_series(user, ident, now)
                await self._send(user, f"Повтор напоминания «{rule.name}» отключён.")
            else:
                await self._choose(
                    user,
                    RepeatDraft(id=uuid4(), series_id=ident, revision=rule.revision, rule=rule),
                )
        else:
            draft = await self._load(user)
            if draft is None or draft.id != ident:
                reply = "Черновик устарел"
            elif draft.choosing_daypart and action in {"early", "late"}:
                clock = draft.rule.clock
                await self._preview_at(
                    user, draft, clock.replace(hour=clock.hour + (12 if action == "late" else 0))
                )
            elif draft.choosing_daypart and action != "cancel":
                reply = "Сначала выбери время."
            elif action == "edit":
                await self._choose(user, draft)
            elif action == "time":
                await self._store(user, draft.model_copy(update={"waiting_for_time": True}))
                await self._send(user, "Напиши новое время, например 10:30.")
            elif action == "cancel":
                await self.redis.delete(self._key(user))
                await self._send(user, "Отменил изменения.")
            elif action == "save" and draft.waiting_for_time:
                reply = "Сначала укажи новое время."
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
                        f"Напоминание «{draft.rule.name}» сохранено.\n{draft.rule.schedule}.\n"
                        "Можно изменить расписание или отключить повтор.",
                        repeat_actions(draft.series_id),
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
        if callback_id is not None:
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
