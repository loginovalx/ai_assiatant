"""HTTP API для Mini App (D).

Запуск: uvicorn api.main:app --port 8000. Документация: http://localhost:8000/docs
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.auth import get_current_user_id
from api.ics import build_ics
from common.calendar_import import CalendarImportError, parse_calendar_csv
from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository
from common.schemas import Event, EventStatus, EventType, ImportResult, SearchHit
from common.timeutils import ensure_aware

MAX_IMPORT_BYTES = 5_000_000
MINIAPP_INDEX = Path(__file__).resolve().parents[1] / "miniapp" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    embedder = build_embedder(settings)
    app.state.repo = EventRepository(pool, embedder, settings.tz)
    try:
        yield
    finally:
        await embedder.aclose()
        await pool.close()


app = FastAPI(title="Voice Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().api_cors_origins,
    allow_methods=["GET", "POST"],  # POST — загрузка календаря из Mini App
    allow_headers=["*"],
)


def get_repo(request: Request) -> EventRepository:
    return request.app.state.repo


Repo = Annotated[EventRepository, Depends(get_repo)]
UserId = Annotated[str, Depends(get_current_user_id)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/me")
async def me(user_id: UserId) -> dict[str, str]:
    return {"user_id": user_id, "timezone": get_settings().default_tz}


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
async def miniapp() -> FileResponse:
    return FileResponse(MINIAPP_INDEX)


@app.post("/v1/events/{event_id}/complete", response_model=Event)
async def complete_event(repo: Repo, event_id: UUID, user_id: UserId) -> Event:
    current = await repo.get(user_id, event_id)
    if current is None or current.status == EventStatus.CANCELLED:
        raise HTTPException(status_code=404, detail="Событие не найдено")
    result = await repo.set_status(user_id, event_id, EventStatus.DONE)
    if result is None:
        raise HTTPException(status_code=404, detail="Событие не найдено")
    return result


@app.get("/v1/events", response_model=list[Event])
async def list_events(
    repo: Repo,
    user_id: UserId,
    from_: Annotated[datetime, Query(alias="from", description="Начало периода (включительно)")],
    to: Annotated[datetime, Query(description="Конец периода (не включительно)")],
    type: Annotated[
        list[EventType] | None, Query(description="Можно несколько: ?type=task&type=meeting")
    ] = None,
    include_undated: Annotated[bool, Query(description="Добавить задачи без даты")] = False,
) -> list[Event]:
    tz = get_settings().tz
    start, end = ensure_aware(from_, tz), ensure_aware(to, tz)
    if start >= end:
        raise HTTPException(status_code=422, detail="'from' должен быть раньше 'to'")
    return await repo.list_by_period(user_id, start, end, type, include_undated)


@app.get("/v1/events/search", response_model=list[SearchHit])
async def search_events(
    repo: Repo,
    user_id: UserId,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
    type: Annotated[
        list[EventType] | None, Query(description="Фильтры типов до ограничения выдачи")
    ] = None,
) -> list[SearchHit]:
    if not q.strip():
        raise HTTPException(status_code=422, detail="Введите поисковый запрос")
    return await repo.search(user_id, q.strip(), limit=limit, types=type)


@app.post("/v1/events/import", response_model=ImportResult)
async def import_calendar(
    repo: Repo,
    user_id: UserId,
    file: Annotated[UploadFile, File(description="CSV с колонками id, date, time, text")],
) -> ImportResult:
    """Загрузить тестовый календарь из CSV в тот же справочник событий, что и голосовые записи.

    Повторная загрузка того же файла дублей не создаёт: строки сопоставляются по колонке id.
    """
    data = await file.read(MAX_IMPORT_BYTES + 1)
    if len(data) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="Файл больше 5 МБ")
    try:
        rows, errors = parse_calendar_csv(data, get_settings().tz)
    except CalendarImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    imported = await repo.import_calendar(user_id, rows)
    return ImportResult(
        imported=imported, duplicates=len(rows) - imported, invalid=len(errors), errors=errors
    )


@app.get(
    "/v1/events/{event_id}/ics",
    response_class=Response,
    responses={200: {"content": {"text/calendar": {}}}},
)
async def event_ics(repo: Repo, event_id: UUID, user_id: UserId) -> Response:
    event = await repo.get(user_id, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Событие не найдено")
    if event.start_at is None:
        raise HTTPException(
            status_code=409, detail="У задачи нет даты — её нельзя добавить в календарь"
        )
    return Response(
        content=build_ics(event),
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="event-{event.id}.ics"'},
    )
