"""Весь SQL проекта в одном месте. MCP-сервер и HTTP API ходят в базу только через этот класс."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg

from common.calendar_import import IMPORTED_DURATION_MINUTES, CalendarRow
from common.embeddings import Embedder
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
WITH base AS (
    SELECT * FROM events
    WHERE user_id = $1
      AND status <> 'cancelled'
      AND ($3::timestamptz IS NULL OR start_at >= $3)
      AND ($4::timestamptz IS NULL OR start_at < $4)
),
q AS (
    SELECT replace(plainto_tsquery('russian', $2)::text, ' & ', ' | ')::tsquery AS tq
),
by_text AS (
    SELECT b.id, row_number() OVER (ORDER BY ts_rank_cd(b.search_tsv, q.tq) DESC) AS rn
    FROM base b, q
    WHERE numnode(q.tq) > 0 AND b.search_tsv @@ q.tq
    ORDER BY rn LIMIT 20
),
by_fuzzy AS (
    SELECT b.id, row_number() OVER (ORDER BY word_similarity($2, b.search_text) DESC) AS rn
    FROM base b
    WHERE word_similarity($2, b.search_text) >= 0.4
    ORDER BY rn LIMIT 20
),
by_meaning AS (
    SELECT b.id, row_number() OVER (ORDER BY b.embedding <=> $5::vector) AS rn
    FROM base b
    WHERE $5::vector IS NOT NULL AND b.embedding IS NOT NULL
    ORDER BY rn LIMIT 10
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

    async def create(self, data: EventCreate) -> Event:
        embedding = await self._embed_one(_embedding_text(data.type, data.name, data.description))
        row = await self._pool.fetchrow(
            f"""
            INSERT INTO events (user_id, type, name, description, start_at, duration_minutes,
                                embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING {_COLUMNS}
            """,
            data.user_id,
            data.type.value,
            data.name,
            data.description,
            data.start_at,
            data.duration_minutes,
            embedding,
        )
        return self._to_event(row)

    async def find_conflicts(self, event: Event) -> list[Event]:
        """Активные встречи и дела пользователя, пересекающиеся с event по времени.

        Пусто, если сама запись время не занимает (напоминание, дело без длительности).
        Саму запись в ответ не включаем — это важно после update.
        """
        interval = busy_interval(event.type, event.start_at, event.duration_minutes)
        if interval is None:
            return []
        start, end = interval
        rows = await self._pool.fetch(
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
        if current is None:
            return None

        new_name = name if name is not None else current.name
        new_description = description if description is not None else current.description
        new_type = event_type if event_type is not None else current.type
        new_start = start_at if start_at is not None else current.start_at
        new_duration = (
            duration_minutes if duration_minutes is not None else current.duration_minutes
        )

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
                embedding = COALESCE($8, embedding)
            WHERE id = $1 AND user_id = $2
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
        return self._to_event(row)

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

    async def claim_due_events(
        self, now: datetime, lookback: timedelta, limit: int = 50
    ) -> list[Event]:
        """Забрать наступившие события, которые ещё не отправляли, и сразу пометить их.

        Пометка и выборка — один запрос: два процесса бота не отправят одно напоминание
        дважды. lookback не даёт после простоя вывалить пользователю всё за годы назад
        (в импортированном календаре даты бывают и прошлогодние).
        """
        rows = await self._pool.fetch(
            f"""
            UPDATE events SET notified_at = now()
            WHERE id IN (
                SELECT id FROM events
                WHERE status = 'active' AND notified_at IS NULL
                  AND start_at IS NOT NULL AND start_at <= $1 AND start_at > $2
                ORDER BY start_at
                LIMIT $3
                FOR UPDATE SKIP LOCKED
            )
            RETURNING {_COLUMNS}
            """,
            now,
            now - lookback,
            limit,
        )
        return [self._to_event(r) for r in rows]

    async def unmark_notified(self, event_id: UUID) -> None:
        """Вернуть событие в очередь: отправка в MAX не удалась, попробуем в следующий тик."""
        await self._pool.execute("UPDATE events SET notified_at = NULL WHERE id = $1", event_id)

    async def cancel_all(
        self, user_id: str, types: Sequence[EventType] | None = None, dry_run: bool = False
    ) -> list[Event] | int:
        """Мягко удалить все видимые записи пользователя (опционально — только данных типов).

        dry_run=True ничего не меняет и возвращает количество: им тул спрашивает подтверждение.
        Иначе возвращает удалённые записи. Чужие записи не затрагиваются никогда.
        """
        type_values = [t.value for t in types] if types else None
        if dry_run:
            return await self._pool.fetchval(
                """
                SELECT count(*) FROM events
                WHERE user_id = $1 AND status <> 'cancelled'
                  AND ($2::text[] IS NULL OR type = ANY($2))
                """,
                user_id,
                type_values,
            )
        rows = await self._pool.fetch(
            f"""
            UPDATE events SET status = 'cancelled'
            WHERE user_id = $1 AND status <> 'cancelled'
              AND ($2::text[] IS NULL OR type = ANY($2))
            RETURNING {_COLUMNS}
            """,
            user_id,
            type_values,
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
    ) -> list[SearchHit]:
        embedding = await self._embed_one(query)
        rows = await self._pool.fetch(_SEARCH_SQL, user_id, query, start, end, embedding, limit)
        return [
            SearchHit(
                **self._to_event(r).model_dump(exclude={"weekday", "end_at"}),
                score=round(r["score"], 5),
                matched_by=list(r["matched_by"]),
            )
            for r in rows
        ]

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
