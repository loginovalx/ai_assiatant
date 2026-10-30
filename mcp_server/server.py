"""MCP-сервер: агент (C) создаёт, получает и ищет события через эти тулы.

Запуск: python -m mcp_server.server  →  http://localhost:8001/mcp (streamable HTTP).

Описания тулов и параметров — это промпт для LLM: по ним GigaChat решает, какой тул
вызвать и как заполнить аргументы. Меняете их — проверяйте агента на датасете.
"""

import logging
from datetime import datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from fastmcp.server.lifespan import lifespan
from pydantic import Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository
from common.schemas import (
    DEFAULT_MEETING_MINUTES,
    MAX_DURATION_MINUTES,
    Event,
    EventCreate,
    EventStatus,
    EventType,
    EventWithConflicts,
    SearchHit,
    check_duration,
)
from common.timeutils import ensure_aware

logger = logging.getLogger(__name__)

# Небольшой запас на задержку между репликой пользователя и вызовом тула.
PAST_TOLERANCE = timedelta(minutes=5)

UserId = Annotated[str, Field(description="ID пользователя MAX. Передавай всегда, как есть.")]
EventTypeName = Literal["task", "meeting", "reminder"]
IsoDateTime = Annotated[
    datetime,
    Field(description="Дата и время в ISO 8601 со смещением, например 2026-10-30T10:00:00+03:00"),
]
Duration = Annotated[
    int | None,
    Field(
        description=(
            "Длительность в минутах: «на полчаса» → 30, «на два часа» → 120, «с 15 до 17» → 120. "
            "Для meeting обязательна: не назвали — сначала спроси пользователя; ответил "
            "«не знаю» — передай 60. Для task — только если назвали, иначе null. "
            "У reminder — всегда null: назвали длительность у «напомни» — это task."
        ),
        ge=1,
        le=MAX_DURATION_MINUTES,
    ),
]


@lifespan
async def resources(server: FastMCP):
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    embedder = build_embedder(settings)
    try:
        yield {"repo": EventRepository(pool, embedder, settings.tz)}
    finally:
        await embedder.aclose()
        await pool.close()


mcp = FastMCP(
    name="events",
    instructions=(
        "Хранилище задач, напоминаний и встреч пользователя. "
        "Все даты — ISO 8601 со смещением часового пояса. "
        "Периоды — полуинтервал [start_time, end_time)."
    ),
    lifespan=resources,
    # Внутренние ошибки (SQL и т. п.) не уходят в LLM, текст ToolError — уходит.
    mask_error_details=True,
)


def _repo(ctx: Context) -> EventRepository:
    return ctx.lifespan_context["repo"]


def _validation_message(error: ValidationError) -> str:
    return "; ".join(e["msg"].removeprefix("Value error, ") for e in error.errors())


@mcp.tool
async def create_event(
    user_id: UserId,
    event_type: Annotated[
        EventTypeName,
        Field(
            description=(
                "task — дело, которое нужно сделать (можно без времени); "
                "reminder — напомнить в конкретный момент; "
                "meeting — встреча или созвон с кем-то."
            )
        ),
    ],
    name: Annotated[
        str,
        Field(description="Короткая суть без даты, например «Позвонить Ивану»", max_length=200),
    ],
    ctx: Context,
    description: Annotated[
        str, Field(description="Исходная фраза пользователя целиком", max_length=2000)
    ] = "",
    start_at: Annotated[
        datetime | None,
        Field(
            description=(
                "Когда событие. Обязательно для meeting и reminder. "
                "Для task без срока — null. ISO 8601 со смещением."
            )
        ),
    ] = None,
    duration_minutes: Duration = None,
) -> EventWithConflicts:
    """Создать задачу, напоминание или встречу.

    Если пользователь не назвал время встречи или напоминания — не вызывай тул,
    а сначала спроси время. У встречи так же нужна длительность: не назвали — спроси,
    сколько она продлится (можно одним вопросом со временем).
    Если время пересекается с другими встречами или делами, запись всё равно сохраняется,
    а пересекающиеся записи приходят в conflicts — предупреди о них пользователя.
    """
    try:
        data = EventCreate(
            user_id=user_id,
            type=event_type,
            name=name,
            description=description,
            start_at=start_at,
            duration_minutes=duration_minutes,
        )
    except ValidationError as error:
        raise ToolError(_validation_message(error)) from error

    now = datetime.now(get_settings().tz).replace(microsecond=0)
    if data.start_at is not None and data.start_at < now - PAST_TOLERANCE:
        raise ToolError(
            f"Время {data.start_at.isoformat()} уже прошло (сейчас {now.isoformat()}). "
            "Проверь вычисление даты или уточни у пользователя."
        )

    # Длительность встречи решает, с чем она пересечётся, поэтому не подставляем час молча:
    # пусть агент спросит. «Не знаю» — тоже ответ: тогда агент сам передаст 60 и скажет об этом.
    # Проверка только здесь, а не в EventCreate: HTTP API и старые записи живут без длительности.
    if data.type == EventType.MEETING and data.duration_minutes is None:
        raise ToolError(
            "Не указана длительность встречи. Запись не создана: спроси пользователя, "
            "сколько продлится встреча. Если он не знает — вызови тул снова с "
            f"duration_minutes={DEFAULT_MEETING_MINUTES} и скажи, что поставил час."
        )

    repo = _repo(ctx)
    event = await repo.create(data)
    conflicts = await repo.find_conflicts(event)
    logger.info(
        "created event %s type=%s user=%s conflicts=%d",
        event.id,
        event.type,
        user_id,
        len(conflicts),
    )
    return EventWithConflicts(**_fields(event), conflicts=conflicts)


def _fields(event: Event) -> dict:
    """Поля события без вычисляемых (weekday, end_at): их модель посчитает сама."""
    return event.model_dump(exclude={"weekday", "end_at"})


@mcp.tool(annotations={"readOnlyHint": True})
async def get_events(
    user_id: UserId,
    start_time: IsoDateTime,
    end_time: Annotated[
        datetime,
        Field(description="Конец периода, не включительно. «Сегодня» → 00:00 завтрашнего дня"),
    ],
    ctx: Context,
    event_types: Annotated[
        list[EventTypeName] | None,
        Field(description='Фильтр по типам. null — все типы. «Какие встречи» → ["meeting"]'),
    ] = None,
    include_undated_tasks: Annotated[
        bool,
        Field(description="Добавить активные задачи без срока. true для «что мне нужно сделать»"),
    ] = False,
) -> list[Event]:
    """Список событий за период, отсортированный по времени.

    Для «что у меня сегодня / завтра / на неделе / в пятницу».
    """
    settings = get_settings()
    start = ensure_aware(start_time, settings.tz)
    end = ensure_aware(end_time, settings.tz)
    if start >= end:
        raise ToolError("start_time должен быть раньше end_time")

    types = [EventType(t) for t in event_types] if event_types else None
    return await _repo(ctx).list_by_period(user_id, start, end, types, include_undated_tasks)


@mcp.tool(annotations={"readOnlyHint": True})
async def search_events(
    user_id: UserId,
    query: Annotated[
        str,
        Field(
            description=(
                "О чём запись, своими словами: «презентация», «встреча с Иваном», «договор». "
                "Без служебных слов вроде «найди» и без дат."
            ),
            min_length=1,
            max_length=200,
        ),
    ],
    ctx: Context,
    start_time: Annotated[
        datetime | None, Field(description="Искать начиная с этого момента. Обычно null")
    ] = None,
    end_time: Annotated[
        datetime | None, Field(description="Искать до этого момента. Обычно null")
    ] = None,
    limit: Annotated[int, Field(ge=1, le=20)] = 5,
) -> list[SearchHit]:
    """Найти ранее созданные события по смыслу, даже если слова не совпадают точно.

    Для «найди задачу про презентацию», «когда у меня была встреча с Иваном».
    Результаты отсортированы по релевантности; если список пуст — ничего не найдено.
    """
    settings = get_settings()
    start = ensure_aware(start_time, settings.tz) if start_time else None
    end = ensure_aware(end_time, settings.tz) if end_time else None
    return await _repo(ctx).search(user_id, query.strip(), start, end, limit)


EventId = Annotated[
    str,
    Field(
        description=(
            "id события (UUID) — бери из скрытой сводки записей в истории диалога, не выдумывай"
        )
    ),
]
# GigaChat принимает только format: date/date-time/time в схеме тулов, поэтому event_id —
# строка, а не UUID (pydantic сгенерировал бы format: "uuid4" и GigaChat вернул бы 422).


def _parse_event_id(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as error:
        raise ToolError(f"event_id должен быть UUID, получено: {value!r}") from error


@mcp.tool
async def update_event(
    user_id: UserId,
    event_id: EventId,
    ctx: Context,
    name: Annotated[
        str | None, Field(description="Новое короткое название. null — не менять", max_length=200)
    ] = None,
    start_at: Annotated[
        datetime | None,
        Field(description="Новые дата и время, ISO 8601 со смещением. null — не менять"),
    ] = None,
    description: Annotated[
        str | None, Field(description="Новое описание. null — не менять", max_length=2000)
    ] = None,
    duration_minutes: Annotated[
        int | None,
        Field(
            description="Новая длительность в минутах («на два часа» → 120). null — не менять",
            ge=1,
            le=MAX_DURATION_MINUTES,
        ),
    ] = None,
) -> EventWithConflicts:
    """Изменить существующую задачу, напоминание или встречу: перенести время, переименовать,
    поменять длительность.

    Для «перенеси её на завтра», «подвинь встречу на 15:00», «переименуй задачу»,
    «встреча будет на полчаса».
    Если новое время пересекается с другими встречами или делами, изменение всё равно
    сохраняется, а пересекающиеся записи приходят в conflicts — предупреди о них пользователя.
    """
    if name is None and start_at is None and description is None and duration_minutes is None:
        raise ToolError("Не указано, что менять: нужно хотя бы одно новое значение")

    now = datetime.now(get_settings().tz).replace(microsecond=0)
    if start_at is not None and start_at < now - PAST_TOLERANCE:
        raise ToolError(
            f"Время {start_at.isoformat()} уже прошло (сейчас {now.isoformat()}). "
            "Проверь вычисление даты или уточни у пользователя."
        )

    repo = _repo(ctx)
    uuid = _parse_event_id(event_id)
    current = await repo.get(user_id, uuid)
    if current is None:
        raise ToolError("Событие не найдено")
    try:
        check_duration(
            current.type,
            start_at or current.start_at,
            duration_minutes if duration_minutes is not None else current.duration_minutes,
        )
    except ValueError as error:
        raise ToolError(str(error)) from error

    event = await repo.update(
        user_id,
        uuid,
        name=name,
        description=description,
        start_at=start_at,
        duration_minutes=duration_minutes,
    )
    if event is None:
        raise ToolError("Событие не найдено")
    conflicts = await repo.find_conflicts(event)
    logger.info("updated event %s user=%s conflicts=%d", event.id, user_id, len(conflicts))
    return EventWithConflicts(**_fields(event), conflicts=conflicts)


@mcp.tool
async def complete_event(user_id: UserId, event_id: EventId, ctx: Context) -> Event:
    """Отметить задачу, напоминание или встречу выполненной.

    Для «готово», «сделал», «встреча прошла».
    """
    event = await _repo(ctx).set_status(user_id, _parse_event_id(event_id), EventStatus.DONE)
    if event is None:
        raise ToolError("Событие не найдено")
    logger.info("completed event %s user=%s", event.id, user_id)
    return event


@mcp.tool
async def delete_event(user_id: UserId, event_id: EventId, ctx: Context) -> Event:
    """Удалить задачу, напоминание или встречу: пропадёт из списков и поиска.

    Для «удали», «отмени встречу», «эта задача больше не нужна».
    """
    event = await _repo(ctx).set_status(user_id, _parse_event_id(event_id), EventStatus.CANCELLED)
    if event is None:
        raise ToolError("Событие не найдено")
    logger.info("deleted event %s user=%s", event.id, user_id)
    return event


@mcp.tool
async def delete_events(
    user_id: UserId,
    ctx: Context,
    event_types: Annotated[
        list[EventTypeName] | None,
        Field(
            description=(
                'Какие типы удалить: «все задачи и напоминания» → ["task", "reminder"], '
                "«вообще всё» → null."
            )
        ),
    ] = None,
    confirmed: Annotated[
        bool,
        Field(description="true только после того, как пользователь ответил «да» на вопрос."),
    ] = False,
) -> list[Event]:
    """Удалить сразу ВСЕ записи пользователя заданных типов.

    Для «удали все мои задачи», «очисти всё». Для одной записи используй delete_event.
    Действие необратимо для пользователя, поэтому без confirmed=true тул ничего не удаляет:
    он ответит, сколько записей будет удалено, а ты спросишь подтверждение через ask_user.
    """
    types = [EventType(t) for t in event_types] if event_types else None
    repo = _repo(ctx)
    if not confirmed:
        count = await repo.cancel_all(user_id, types, dry_run=True)
        if count == 0:
            raise ToolError("Записей такого типа нет — удалять нечего. Так и скажи пользователю.")
        raise ToolError(
            f"Ничего не удалено: будет удалено записей — {count}. Спроси через ask_user: "
            f"«Удалить все записи ({count} шт.)? Это нельзя отменить.» Ответил «да» — "
            "вызови тул снова с confirmed=true; «нет» — ничего не делай."
        )
    deleted = await repo.cancel_all(user_id, types)
    logger.info("bulk deleted %d events user=%s types=%s", len(deleted), user_id, event_types)
    return deleted


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    mcp.run(
        transport="http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        path=settings.mcp_path,
        show_banner=False,
    )


if __name__ == "__main__":
    main()
