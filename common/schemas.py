"""Контракты между сервисами. Меняем только по договорённости в общем чате.

Кто что использует:
- MCP-сервер и HTTP API: EventCreate, Event, SearchHit.
- Бот (B) и агент (C): AgentRequest, AgentResponse.
"""

from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from common.config import get_settings
from common.timeutils import WEEKDAYS_RU, ensure_aware


class EventType(StrEnum):
    TASK = "task"
    MEETING = "meeting"
    REMINDER = "reminder"


class EventStatus(StrEnum):
    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"


# Встреча и напоминание без времени не имеют смысла: агент должен уточнить время.
TIMED_TYPES = frozenset({EventType.MEETING, EventType.REMINDER})

# Встреча без названной длительности занимает час — так же её видит и календарь (.ics).
DEFAULT_MEETING_MINUTES = 60
MAX_DURATION_MINUTES = 24 * 60

DurationMinutes = Field(
    default=None,
    ge=1,
    le=MAX_DURATION_MINUTES,
    description="Длительность в минутах. null — по умолчанию: встреча час, дело время не занимает",
)


def check_duration(
    event_type: EventType, start_at: datetime | None, duration_minutes: int | None
) -> None:
    """Те же правила, что CHECK duration_valid в БД, но с текстом, понятным LLM."""
    if duration_minutes is None:
        return
    if event_type == EventType.REMINDER:
        raise ValueError(
            "У напоминания нет длительности: это момент времени. "
            "Пользователь назвал длительность — создавай task с этой длительностью"
        )
    if start_at is None:
        raise ValueError("Длительность можно указать только у записи с датой и временем")


def busy_interval(
    event_type: EventType, start_at: datetime | None, duration_minutes: int | None
) -> tuple[datetime, datetime] | None:
    """Какой отрезок времени запись занимает в расписании; None — не занимает.

    Встреча занимает всегда (по умолчанию час). Дело — только если у него есть время
    и названа длительность («поработать над отчётом с 10 до 12»). Напоминание — момент,
    а дело без длительности — это срок («сдать отчёт до 18:00»): с ними пересечений нет.
    """
    if start_at is None:
        return None
    if event_type == EventType.MEETING:
        minutes = duration_minutes or DEFAULT_MEETING_MINUTES
    elif event_type == EventType.TASK and duration_minutes:
        minutes = duration_minutes
    else:
        return None
    return start_at, start_at + timedelta(minutes=minutes)


class EventCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    type: EventType
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    start_at: datetime | None = None
    duration_minutes: int | None = DurationMinutes

    @field_validator("name", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("start_at")
    @classmethod
    def attach_timezone(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_aware(value, get_settings().tz)

    @model_validator(mode="after")
    def timed_types_need_start(self) -> "EventCreate":
        if self.type in TIMED_TYPES and self.start_at is None:
            raise ValueError(f"Для типа '{self.type}' нужно указать дату и время (start_at)")
        check_duration(self.type, self.start_at, self.duration_minutes)
        return self


class Event(BaseModel):
    id: UUID
    user_id: str
    type: EventType
    name: str
    description: str
    start_at: datetime | None
    duration_minutes: int | None = None
    status: EventStatus
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def weekday(self) -> str | None:
        """День недели словами: LLM нередко ошибается, вычисляя его сама."""
        return None if self.start_at is None else WEEKDAYS_RU[self.start_at.weekday()]

    @computed_field
    @property
    def end_at(self) -> datetime | None:
        """Конец занятого времени; None, если запись время не занимает (см. busy_interval)."""
        interval = busy_interval(self.type, self.start_at, self.duration_minutes)
        return None if interval is None else interval[1]


class EventWithConflicts(Event):
    """Результат создания или изменения: сама запись + с чем она пересекается по времени.

    Пересечение не запрещает запись, а только предупреждает: человек может осознанно
    поставить две встречи подряд или внахлёст.
    """

    conflicts: list[Event] = Field(
        default_factory=list,
        description="Активные встречи и дела, пересекающиеся по времени. Пусто — пересечений нет",
    )


class SearchHit(Event):
    score: float = Field(description="Чем больше, тем релевантнее (Reciprocal Rank Fusion)")
    matched_by: list[str] = Field(description="Какие способы поиска нашли запись")


class ImportResult(BaseModel):
    """Итог загрузки календаря из CSV."""

    imported: int = Field(description="Добавлено новых событий")
    duplicates: int = Field(description="Уже были в календаре: тот же id, повторная загрузка")
    invalid: int = Field(description="Строк пропущено из-за ошибок формата")
    errors: list[str] = Field(default_factory=list, description="Первые ошибки, по одной на строку")


class AgentStatus(StrEnum):
    DONE = "done"
    NEED_CLARIFICATION = "need_clarification"
    ERROR = "error"
    NEEDS_CONFIRMATION = "needs_confirmation"


class DraftAction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tool: str
    arguments: dict = Field(default_factory=dict)
    preview: str
    status: str = "pending_confirmation"


class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    confirm_changes: bool = False


class AgentResponse(BaseModel):
    reply_text: str
    status: AgentStatus
    events: list[Event] = []
    draft: DraftAction | None = None


class AgentConfirmRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    draft: DraftAction


class AgentResetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
