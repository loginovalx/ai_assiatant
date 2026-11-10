"""Весь SQL проекта в одном месте. MCP-сервер и HTTP API ходят в базу только через этот класс."""

import hashlib
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from zoneinfo import ZoneInfo

import asyncpg

from common.calendar_import import IMPORTED_DURATION_MINUTES, CalendarRow
from common.embeddings import EMBEDDING_DIM, Embedder
from common.reminder_series import ReminderRule, SavedReminderRule
from common.schemas import (
    DEFAULT_MEETING_MINUTES,
    MAX_DURATION_MINUTES,
    Event,
    EventCreate,
    EventStatus,
    EventType,
    SearchHit,
    busy_interval,
)

_COLUMNS = (
    "id, user_id, type, name, description, start_at, duration_minutes, status, "
    "created_at, updated_at"
)


def _validate_optional_period(start: datetime | None, end: datetime | None) -> None:
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be earlier than end")


# Пересечение полуинтервалов: [a_start, a_end) и [b_start, b_end) пересекаются,
# если a_start < b_end и b_start < a_end. Встреча 15:00–16:00 и встреча в 16:00 — не конфликт.
# Какие записи занимают время — те же правила, что busy_interval в schemas.py.
# Условие «start_at > начало − сутки» отсекает лишнее по индексу (user_id, start_at):
# запись не длится дольше суток, значит начавшаяся раньше нас точно не заденет.
_CONFLICTS_SQL = f"""
SELECT {_COLUMNS} FROM events
WHERE user_id = $1
  AND status = 'active'
  AND ($4::uuid IS NULL OR id <> $4)
  AND (type = 'meeting' OR (type = 'task' AND duration_minutes IS NOT NULL))
  AND start_at > $2::timestamptz - make_interval(mins => $5)
  AND start_at < $3::timestamptz
  AND start_at + make_interval(mins => COALESCE(duration_minutes, $6)) > $2::timestamptz
ORDER BY start_at
"""

# Поиск: три независимых способа найти запись, результаты склеиваются через
# Reciprocal Rank Fusion: score = Σ 1 / (60 + место в выдаче способа).
# Почему RRF: оценки ts_rank, trigram и косинуса несравнимы между собой,
# а места в выдаче сравнимы. Константа 60 — стандартное значение из статьи про RRF.
#   text     — полнотекстовый поиск по-русски: «договору» найдёт «договор»;
#   fuzzy    — триграммы: «презинтация» найдёт «презентация»;
#   semantic — эмбеддинги GigaChat: «слайды» найдёт «презентацию».
# Слова запроса объединяем через ИЛИ (а не И, как plainto_tsquery), иначе
# «задачу про договор» не найдёт запись «Подписать договор» из-за слова «задача».
_SEARCH_SQL = f"""
WITH base AS MATERIALIZED (
    SELECT * FROM events
    WHERE user_id = $1
      AND status <> 'cancelled'
      AND ($3::timestamptz IS NULL OR start_at >= $3)
      AND ($4::timestamptz IS NULL OR start_at < $4)
      AND ($7::text[] IS NULL OR type = ANY($7))
),
q AS (
    SELECT replace(plainto_tsquery('russian', $2)::text, ' & ', ' | ')::tsquery AS tq
),
by_text AS (
    SELECT b.id, row_number() OVER (ORDER BY ts_rank_cd(b.search_tsv, q.tq) DESC) AS rn
    FROM base b, q
    WHERE numnode(q.tq) > 0 AND b.search_tsv @@ q.tq
    ORDER BY rn LIMIT GREATEST($6, 20)
),
by_fuzzy AS (
    SELECT b.id, row_number() OVER (ORDER BY word_similarity($2, b.search_text) DESC) AS rn
    FROM base b
    WHERE word_similarity($2, b.search_text) >= 0.4
    ORDER BY rn LIMIT GREATEST($6, 20)
),
by_meaning AS (
    SELECT b.id, row_number() OVER (ORDER BY b.embedding <=> $5::vector) AS rn
    FROM base b
    WHERE $5::vector IS NOT NULL AND b.embedding IS NOT NULL
    ORDER BY rn LIMIT GREATEST($6, 10)
),
ranked AS (
    SELECT id, 'text' AS source, rn FROM by_text
    UNION ALL SELECT id, 'fuzzy', rn FROM by_fuzzy
    UNION ALL SELECT id, 'semantic', rn FROM by_meaning
),
scored AS (
    SELECT id,
           sum(1.0 / (60 + rn))::float8 AS score,
           array_agg(source ORDER BY source) AS matched_by
    FROM ranked
    GROUP BY id
)
SELECT {", ".join(f"e.{c}" for c in _COLUMNS.split(", "))}, s.score, s.matched_by
FROM scored s
JOIN events e ON e.id = s.id
ORDER BY s.score DESC, e.start_at DESC NULLS LAST
LIMIT $6
"""


class EventRepository:
    def __init__(self, pool: asyncpg.Pool, embedder: Embedder, tz: ZoneInfo) -> None:
        self._pool = pool
        self._embedder = embedder
        self._tz = tz

    @staticmethod
    def creation_id(user_id: str, idempotency_key: UUID) -> UUID:
        return uuid5(NAMESPACE_URL, f"voice-assistant:{user_id}:{idempotency_key}")

    async def create(self, data: EventCreate, idempotency_key: UUID | None = None) -> Event:
        embedding = await self._embed_one(_embedding_text(data.type, data.name, data.description))
        event_id = self.creation_id(data.user_id, idempotency_key) if idempotency_key else uuid4()
        return await self._insert_event(self._pool, data, embedding, event_id)

    async def _insert_event(self, db, data: EventCreate, embedding, event_id: UUID) -> Event:
        row = await db.fetchrow(
            f"""
            INSERT INTO events (id, user_id, type, name, description, start_at, duration_minutes,
                                embedding)
            VALUES ($8, $1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET id = events.id
            WHERE events.user_id = EXCLUDED.user_id
            RETURNING {_COLUMNS}
            """,
            data.user_id,
            data.type.value,
            data.name,
            data.description,
            data.start_at,
            data.duration_minutes,
            embedding,
            event_id,
        )
        return self._to_event(row)

    async def create_checked(
        self, data: EventCreate, request_id: UUID, confirmation: str | None
    ) -> tuple[Event | None, list[Event], str | None]:
        """Serialize Mini App submissions per user and recheck the approved conflicts."""
        event_id = self.creation_id(data.user_id, request_id)
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"miniapp-create:{data.user_id}",
            )
            existing = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM events WHERE id = $1 AND user_id = $2",
                event_id,
                data.user_id,
            )
            if existing:
                return self._to_event(existing), [], None
            now = datetime.now(self._tz)
            candidate = Event(
                id=event_id,
                **data.model_dump(),
                status=EventStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
            conflicts = await self._find_conflicts(conn, candidate)
            if conflicts:
                revisions = sorted(
                    f"{event.id}:{event.updated_at.isoformat()}" for event in conflicts
                )
                payload = "\n".join([str(request_id), data.model_dump_json(), *revisions])
                snapshot = hashlib.sha256(payload.encode()).hexdigest()
                if confirmation != snapshot:
                    return None, conflicts, snapshot
            embedding = await self._embed_one(
                _embedding_text(data.type, data.name, data.description)
            )
            return await self._insert_event(conn, data, embedding, event_id), [], None

    async def find_conflicts(self, event: Event) -> list[Event]:
        """Активные встречи и дела пользователя, пересекающиеся с event по времени.

        Пусто, если сама запись время не занимает (напоминание, дело без длительности).
        Саму запись в ответ не включаем — это важно после update.
        """
        return await self._find_conflicts(self._pool, event)

    async def _find_conflicts(self, db, event: Event) -> list[Event]:
        interval = busy_interval(event.type, event.start_at, event.duration_minutes)
        if interval is None:
            return []
        start, end = interval
        rows = await db.fetch(
            _CONFLICTS_SQL,
            event.user_id,
            start,
            end,
            event.id,
            MAX_DURATION_MINUTES,
            DEFAULT_MEETING_MINUTES,
        )
        return [self._to_event(r) for r in rows]

    async def get(self, user_id: str, event_id: UUID) -> Event | None:
        row = await self._pool.fetchrow(
            f"SELECT {_COLUMNS} FROM events WHERE id = $1 AND user_id = $2",
            event_id,
            user_id,
        )
        return None if row is None else self._to_event(row)

    async def update(
        self,
        user_id: str,
        event_id: UUID,
        name: str | None = None,
        description: str | None = None,
        start_at: datetime | None = None,
        event_type: EventType | None = None,
        duration_minutes: int | None = None,
    ) -> Event | None:
        """Частичное обновление: None в параметре — «не менять».

        Эмбеддинг пересчитывается, только если изменился смысл записи (name/description/type);
        перенос времени и длительности его не трогает.
        """
        current = await self.get(user_id, event_id)
        if current is None or current.status == EventStatus.CANCELLED:
            return None

        new_name = name if name is not None else current.name
        new_description = description if description is not None else current.description
        new_type = event_type if event_type is not None else current.type
        new_start = start_at if start_at is not None else current.start_at
        new_duration = (
            duration_minutes if duration_minutes is not None else current.duration_minutes
        )
        validated = EventCreate(
            user_id=user_id,
            type=new_type,
            name=new_name,
            description=new_description,
            start_at=new_start,
            duration_minutes=new_duration,
        )
        new_name, new_description = validated.name, validated.description

        recompute = name is not None or description is not None or event_type is not None
        embedding = (
            await self._embed_one(_embedding_text(new_type, new_name, new_description))
            if recompute
            else None
        )

        row = await self._pool.fetchrow(
            f"""
            UPDATE events
            SET name = $3, description = $4, type = $5, start_at = $6, duration_minutes = $7,
                embedding = COALESCE($8, embedding),
                notified_at = CASE WHEN start_at IS DISTINCT FROM $6 THEN NULL ELSE notified_at END,
                notification_claimed_at = CASE WHEN start_at IS DISTINCT FROM $6
                    THEN NULL ELSE notification_claimed_at END,
                notification_claim_token = CASE WHEN start_at IS DISTINCT FROM $6
                    THEN NULL ELSE notification_claim_token END
            WHERE id = $1 AND user_id = $2 AND status <> 'cancelled'
            RETURNING {_COLUMNS}
            """,
            event_id,
            user_id,
            new_name,
            new_description,
            new_type.value,
            new_start,
            new_duration,
            embedding,
        )
        return self._to_event(row) if row is not None else None

    async def set_status(self, user_id: str, event_id: UUID, status: EventStatus) -> Event | None:
        """Отметить выполненной (done) или удалить мягко (cancelled — пропадает из выдач)."""
        row = await self._pool.fetchrow(
            f"""
            UPDATE events SET status = $3
            WHERE id = $1 AND user_id = $2
            RETURNING {_COLUMNS}
            """,
            event_id,
            user_id,
            status.value,
        )
        return None if row is None else self._to_event(row)

    async def import_calendar(self, user_id: str, rows: Sequence[CalendarRow]) -> int:
        """Импорт строк CSV как событий календаря. Возвращает число реально добавленных.

        Повторная загрузка того же файла ничего не дублирует: (user_id, external_id)
        уникальны. Тип — task с длительностью: такое событие занимает время, поэтому
        бот увидит пересечение при «поставь встречу завтра в 14:00».
        Эмбеддинги не считаем: на тысяче строк это тысяча запросов к GigaChat, а поиск
        по тексту (полнотекстовый и по опечаткам) работает и без них.
        """
        if not rows:
            return 0
        inserted = await self._pool.fetch(
            """
            INSERT INTO events (user_id, type, name, description, start_at,
                                duration_minutes, external_id)
            SELECT $1, 'task', r.name, '', r.start_at, $2, r.external_id
            FROM unnest($3::text[], $4::timestamptz[], $5::text[])
                 AS r(name, start_at, external_id)
            ON CONFLICT (user_id, external_id) WHERE external_id IS NOT NULL DO NOTHING
            RETURNING id
            """,
            user_id,
            IMPORTED_DURATION_MINUTES,
            [r.name for r in rows],
            [r.start_at for r in rows],
            [r.external_id for r in rows],
        )
        return len(inserted)

    async def list_reminder_series(self, user_id: str) -> list[SavedReminderRule]:
        rows = await self._pool.fetch(
            "SELECT id, revision, name, weekday, clock FROM reminder_series "
            "WHERE user_id = $1 AND active ORDER BY name, id",
            user_id,
        )
        return [SavedReminderRule.model_validate(dict(row)) for row in rows]

    async def save_reminder_series(
        self,
        user_id: str,
        series_id: UUID,
        rule: ReminderRule,
        now: datetime,
        revision: UUID | None = None,
        *,
        operation_id: UUID,
    ) -> bool:
        first = rule.first_after(now.astimezone(self._tz))
        async with self._pool.acquire() as conn, conn.transaction():
            if revision is None:
                await conn.execute(
                    """
                    INSERT INTO reminder_series
                        (id, user_id, name, weekday, clock, timezone, starts_at,
                         generated_until, revision)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    series_id,
                    user_id,
                    rule.name,
                    rule.weekday,
                    rule.clock,
                    self._tz.key,
                    first,
                    first.date() - timedelta(days=1),
                    operation_id,
                )
                saved = await conn.fetchval(
                    "SELECT active FROM reminder_series WHERE id = $1 AND user_id = $2",
                    series_id,
                    user_id,
                )
            else:
                saved = await conn.fetchval(
                    """
                    UPDATE reminder_series SET weekday = $4, clock = $5, starts_at = $6,
                        generated_until = $7, revision = $8
                    WHERE id = $1 AND user_id = $2 AND revision = $3 AND active
                    RETURNING true
                    """,
                    series_id,
                    user_id,
                    revision,
                    rule.weekday,
                    rule.clock,
                    first,
                    first.date() - timedelta(days=1),
                    operation_id,
                )
                if saved:
                    await self._delete_future_reminders(conn, series_id, now)
                else:
                    saved = await conn.fetchval(
                        "SELECT active FROM reminder_series "
                        "WHERE id = $1 AND user_id = $2 AND revision = $3",
                        series_id,
                        user_id,
                        operation_id,
                    )
            if saved:
                await self._fill_reminders(conn, now, now, series_id)
            return bool(saved)

    async def stop_reminder_series(self, user_id: str, series_id: UUID, now: datetime) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            stopped = await conn.fetchval(
                "UPDATE reminder_series SET active = false "
                "WHERE id = $1 AND user_id = $2 RETURNING true",
                series_id,
                user_id,
            )
            if stopped:
                await self._delete_future_reminders(conn, series_id, now)
            return bool(stopped)

    @staticmethod
    async def _delete_future_reminders(conn, series_id: UUID, now: datetime) -> None:
        await conn.execute(
            "DELETE FROM events WHERE reminder_series_id = $1 AND start_at >= $2 "
            "AND notified_at IS NULL AND status = 'active'",
            series_id,
            now,
        )

    async def fill_reminders(self, now: datetime, lookback: timedelta) -> None:
        await self._fill_reminders(self._pool, now, now - lookback)

    @staticmethod
    async def _fill_reminders(conn, now: datetime, earliest: datetime, series_id=None) -> None:
        # Generate a rolling calendar window. The watermark and rows commit together.
        await conn.execute(
            """
            WITH selected AS (
                SELECT s.*, ($1::timestamptz AT TIME ZONE timezone)::date + 7 AS horizon
                FROM reminder_series s
                WHERE active AND ($3::uuid IS NULL OR id = $3)
                  AND generated_until < ($1 AT TIME ZONE timezone)::date + 7
                ORDER BY generated_until, id
                LIMIT 100 FOR UPDATE SKIP LOCKED
            ), advanced AS (
                UPDATE reminder_series s SET generated_until = selected.horizon
                FROM selected WHERE s.id = selected.id
            )
            INSERT INTO events (user_id, type, name, start_at, reminder_series_id, occurrence_at)
            SELECT s.user_id, 'reminder', s.name, slot.starts, s.id, slot.starts
            FROM selected s
            CROSS JOIN LATERAL generate_series(
                GREATEST(s.generated_until + 1, ($2::timestamptz AT TIME ZONE s.timezone)::date)
                    ::timestamp, s.horizon::timestamp, interval '1 day'
            ) AS day(value)
            CROSS JOIN LATERAL (
                SELECT (day.value::date + s.clock) AT TIME ZONE s.timezone AS starts
            ) AS slot
            WHERE (s.weekday IS NULL OR extract(isodow FROM day.value) = s.weekday + 1)
              AND slot.starts >= s.starts_at AND slot.starts >= $2
            ON CONFLICT (reminder_series_id, occurrence_at) DO NOTHING
            """,
            now,
            earliest,
            series_id,
        )

    async def claim_due_events(
        self,
        now: datetime,
        lookback: timedelta,
        limit: int = 50,
        *,
        claim_token: UUID | None = None,
    ) -> list[Event]:
        """Extend the schedule, then claim unsent events for five minutes."""
        await self.fill_reminders(now, lookback)
        rows = await self._pool.fetch(
            f"""
            UPDATE events SET notification_claimed_at = $1, notification_claim_token = $4
            WHERE id IN (
                SELECT id FROM events
                WHERE status = 'active' AND notified_at IS NULL
                  AND start_at IS NOT NULL AND start_at <= $1 AND start_at > $2
                  AND (reminder_series_id IS NULL OR EXISTS (
                      SELECT 1 FROM reminder_series s WHERE s.id = reminder_series_id AND s.active
                  ))
                  AND (notification_claimed_at IS NULL
                       OR notification_claimed_at <= $1 - interval '5 minutes')
                ORDER BY start_at
                LIMIT $3
                FOR UPDATE SKIP LOCKED
            )
            RETURNING {_COLUMNS}, reminder_series_id
            """,
            now,
            now - lookback,
            limit,
            claim_token or uuid4(),
        )
        return [self._to_event(r) for r in rows]

    async def postpone(self, user_id: str, event_id: UUID, start_at: datetime) -> Event | None:
        """Отложить событие по кнопке из уведомления: новое время и возврат в очередь отправки.

        Отметка notified_at сбрасывается тем же запросом, иначе напоминание не придёт повторно.
        user_id в условии: чужую запись по кнопке под уведомлением изменить нельзя.
        """
        row = await self._pool.fetchrow(
            f"""
            UPDATE events SET start_at = $3, notified_at = NULL,
                notification_claimed_at = NULL, notification_claim_token = NULL
            WHERE id = $1 AND user_id = $2 AND status = 'active'
            RETURNING {_COLUMNS}
            """,
            event_id,
            user_id,
            start_at,
        )
        return None if row is None else self._to_event(row)

    async def mark_notified(self, event_id: UUID, claim_token: UUID) -> None:
        """Acknowledge delivery only while this worker owns the claim."""
        await self._pool.execute(
            "UPDATE events SET notified_at = now(), notification_claimed_at = NULL, "
            "notification_claim_token = NULL WHERE id = $1 AND notification_claim_token = $2",
            event_id,
            claim_token,
        )

    async def unmark_notified(self, event_id: UUID, claim_token: UUID | None = None) -> None:
        """Вернуть событие в очередь: отправка в MAX не удалась, попробуем в следующий тик."""
        await self._pool.execute(
            "UPDATE events SET notification_claimed_at = NULL, notification_claim_token = NULL "
            "WHERE id = $1 AND ($2::uuid IS NULL OR notification_claim_token = $2)",
            event_id,
            claim_token,
        )

    async def cancel_all(
        self,
        user_id: str,
        types: Sequence[EventType] | None = None,
        dry_run: bool = False,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event] | int:
        """Мягко удалить все видимые записи пользователя (опционально — только данных типов).

        dry_run=True ничего не меняет и возвращает количество: им тул спрашивает подтверждение.
        Иначе возвращает удалённые записи. Если задан период, учитываются только датированные
        события из полуинтервала [start, end). Чужие записи не затрагиваются никогда.
        """
        _validate_optional_period(start, end)
        type_values = [t.value for t in types] if types else None
        if dry_run:
            return await self._pool.fetchval(
                """
                SELECT count(*) FROM events
                WHERE user_id = $1 AND status <> 'cancelled'
                  AND ($2::text[] IS NULL OR type = ANY($2))
                  AND ($3::timestamptz IS NULL OR (start_at >= $3 AND start_at < $4))
                """,
                user_id,
                type_values,
                start,
                end,
            )
        rows = await self._pool.fetch(
            f"""
            UPDATE events SET status = 'cancelled'
            WHERE user_id = $1 AND status <> 'cancelled'
              AND ($2::text[] IS NULL OR type = ANY($2))
              AND ($3::timestamptz IS NULL OR (start_at >= $3 AND start_at < $4))
            RETURNING {_COLUMNS}
            """,
            user_id,
            type_values,
            start,
            end,
        )
        return [self._to_event(r) for r in rows]

    async def preview_cancellations(
        self,
        user_id: str,
        types: Sequence[EventType] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        """Read the records and IDs shown in one cancellation snapshot."""
        _validate_optional_period(start, end)
        type_values = [t.value for t in types] if types else None
        rows = await self._pool.fetch(
            f"""
            SELECT {_COLUMNS} FROM events
            WHERE user_id = $1 AND status <> 'cancelled'
              AND ($2::text[] IS NULL OR type = ANY($2))
              AND ($3::timestamptz IS NULL OR (start_at >= $3 AND start_at < $4))
            ORDER BY start_at NULLS LAST, created_at, id
            """,
            user_id,
            type_values,
            start,
            end,
        )
        return [self._to_event(row) for row in rows]

    async def cancel_selected(
        self,
        user_id: str,
        event_ids: Sequence[UUID],
        types: Sequence[EventType] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        """Cancel only previewed IDs still owned by the user and within all given filters."""
        _validate_optional_period(start, end)
        if not event_ids:
            return []
        type_values = [t.value for t in types] if types else None
        rows = await self._pool.fetch(
            f"""
            UPDATE events SET status = 'cancelled'
            WHERE id = ANY($1::uuid[]) AND user_id = $2 AND status <> 'cancelled'
              AND ($3::text[] IS NULL OR type = ANY($3))
              AND ($4::timestamptz IS NULL OR (start_at >= $4 AND start_at < $5))
            RETURNING {_COLUMNS}
            """,
            list(event_ids),
            user_id,
            type_values,
            start,
            end,
        )
        return [self._to_event(r) for r in rows]

    async def list_by_period(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        types: Sequence[EventType] | None = None,
        include_undated: bool = False,
    ) -> list[Event]:
        """События в полуинтервале [start, end): «сегодня» = [00:00 сегодня, 00:00 завтра).

        include_undated добавляет активные задачи без даты в конец списка.
        Отменённые события не возвращаются.
        """
        rows = await self._pool.fetch(
            f"""
            SELECT {_COLUMNS} FROM events
            WHERE user_id = $1
              AND status <> 'cancelled'
              AND ($4::text[] IS NULL OR type = ANY($4))
              AND (
                    (start_at >= $2 AND start_at < $3)
                 OR ($5 AND start_at IS NULL AND status = 'active')
              )
            ORDER BY start_at NULLS LAST, created_at
            """,
            user_id,
            start,
            end,
            [t.value for t in types] if types else None,
            include_undated,
        )
        return [self._to_event(r) for r in rows]

    async def search(
        self,
        user_id: str,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 5,
        types: Sequence[EventType] | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        embedding = await self._embed_one(query)
        judge = getattr(self._embedder, "relevant_indices", None)
        rows = await self._pool.fetch(
            _SEARCH_SQL,
            user_id,
            query,
            start,
            end,
            embedding,
            max(limit, 50) if embedding is not None and judge is not None else limit,
            [kind.value for kind in types] if types else None,
        )
        if rows and embedding is not None and judge is not None:
            keys = list(dict.fromkeys((row["type"], row["name"]) for row in rows))
            try:
                relevant = await judge(query, [{"type": kind, "name": name} for kind, name in keys])
            except Exception:
                relevant = None
            if relevant is None:
                # Provider unavailable: keep only independently supported lexical hits.
                rows = [row for row in rows if {"text", "fuzzy"} & set(row["matched_by"])]
            else:
                allowed = {keys[index] for index in relevant if 0 <= index < len(keys)}
                rows = [row for row in rows if (row["type"], row["name"]) in allowed]
        return [
            SearchHit(
                **self._to_event(r).model_dump(exclude={"weekday", "end_at"}),
                score=round(r["score"], 5),
                matched_by=list(r["matched_by"]),
            )
            for r in rows[:limit]
        ]

    async def backfill_imported_embeddings(
        self, user_id: str, *, limit: int = 1000, batch_size: int = 16
    ) -> int:
        """Index missing embeddings for this user's CSV-imported events only.

        Each provider call is limited to at most 64 distinct texts. Updates are
        committed per batch and guarded by ``embedding IS NULL`` so retries or
        concurrent runs never replace a vector that already exists.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")

        rows = await self._pool.fetch(
            """
            SELECT id, type, name, description FROM events
            WHERE user_id = $1 AND external_id IS NOT NULL AND embedding IS NULL
            ORDER BY created_at, id
            LIMIT $2
            """,
            user_id,
            limit,
        )
        if not rows:
            return 0

        texts = [
            _embedding_text(EventType(row["type"]), row["name"], row["description"]) for row in rows
        ]
        vectors_by_text: dict[str, list[float]] = {}
        indexed = 0

        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            batch_texts = texts[offset : offset + batch_size]
            pending_texts = list(
                dict.fromkeys(text for text in batch_texts if text not in vectors_by_text)
            )
            if pending_texts:
                vectors = await self._embedder.embed(pending_texts)
                if vectors is None or len(vectors) != len(pending_texts):
                    raise RuntimeError("Embedding provider failed; current batch was not written")
                if any(len(vector) != EMBEDDING_DIM for vector in vectors):
                    raise RuntimeError(
                        "Embedding provider returned an invalid vector; batch not written"
                    )
                vectors_by_text.update(zip(pending_texts, vectors, strict=True))

            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for row, text in zip(batch_rows, batch_texts, strict=True):
                        updated = await conn.fetchval(
                            """
                            UPDATE events SET embedding = $3
                            WHERE id = $1 AND user_id = $2
                              AND external_id IS NOT NULL AND embedding IS NULL
                            RETURNING id
                            """,
                            row["id"],
                            user_id,
                            vectors_by_text[text],
                        )
                        if updated is not None:
                            indexed += 1

        return indexed

    async def _embed_one(self, text: str) -> list[float] | None:
        vectors = await self._embedder.embed([text])
        return vectors[0] if vectors else None

    def _to_event(self, row: asyncpg.Record | dict[str, Any]) -> Event:
        start_at = row["start_at"]
        return Event(
            id=row["id"],
            user_id=row["user_id"],
            type=EventType(row["type"]),
            name=row["name"],
            description=row["description"],
            # База отдаёт UTC; наружу — время пользователя, чтобы агенту не пересчитывать.
            start_at=start_at.astimezone(self._tz) if start_at else None,
            duration_minutes=row["duration_minutes"],
            status=EventStatus(row["status"]),
            reminder_series_id=row.get("reminder_series_id"),
            created_at=row["created_at"].astimezone(self._tz),
            updated_at=row["updated_at"].astimezone(self._tz),
        )


_TYPE_LABELS = {
    EventType.TASK: "Задача",
    EventType.MEETING: "Встреча",
    EventType.REMINDER: "Напоминание",
}


def _embedding_text(event_type: EventType, name: str, description: str) -> str:
    text = f"{_TYPE_LABELS[event_type]}: {name}"
    return f"{text}. {description}" if description and description != name else text
