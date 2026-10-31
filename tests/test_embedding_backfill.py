from datetime import UTC, datetime, timedelta

import pytest

from common.calendar_import import CalendarRow
from common.embeddings import EMBEDDING_DIM
from common.repository import EventRepository
from scripts.index_calendar import index_calendar, parse_args


class RecordingEmbedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        self.calls.append(list(texts))
        if self.fail:
            return None
        return [self.vector(text) for text in texts]

    async def aclose(self) -> None:
        return None

    @staticmethod
    def vector(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        vector[sum(text.encode("utf-8")) % EMBEDDING_DIM] = 1.0
        return vector


async def import_rows(repo, user_id: str, names: list[str], start: datetime) -> None:
    rows = [CalendarRow(str(i), start + timedelta(minutes=i), name) for i, name in enumerate(names)]
    assert await repo.import_calendar(user_id, rows) == len(rows)


async def test_backfill_is_user_scoped_and_deduplicates_text(pool, tz):
    embedder = RecordingEmbedder()
    repo = EventRepository(pool, embedder, tz)
    start = datetime.now(UTC)
    await import_rows(
        repo, "user-1", ["Сходить к врачу", "Сходить к врачу", "Купить билеты"], start
    )
    await import_rows(repo, "user-2", ["Сходить к врачу"], start)

    indexed = await repo.backfill_imported_embeddings("user-1", limit=1000, batch_size=16)

    assert indexed == 3
    assert len(embedder.calls) == 1
    assert set(embedder.calls[0]) == {"Задача: Сходить к врачу", "Задача: Купить билеты"}
    rows = await pool.fetch(
        "SELECT user_id, embedding IS NOT NULL AS indexed FROM events WHERE external_id IS NOT NULL"
    )
    assert sum(row["indexed"] for row in rows if row["user_id"] == "user-1") == 3
    assert all(not row["indexed"] for row in rows if row["user_id"] == "user-2")


async def test_backfill_batches_and_does_not_replace_existing_vectors(pool, tz):
    embedder = RecordingEmbedder()
    repo = EventRepository(pool, embedder, tz)
    await import_rows(
        repo,
        "user-1",
        [f"Задача {i}" for i in range(20)],
        datetime.now(UTC),
    )
    original = [0.0] * EMBEDDING_DIM
    original[7] = 1.0
    await pool.execute(
        "UPDATE events SET embedding = $1 WHERE user_id = $2 AND external_id = $3",
        original,
        "user-1",
        "0",
    )

    indexed = await repo.backfill_imported_embeddings("user-1", limit=1000, batch_size=16)

    assert indexed == 19
    assert [len(call) for call in embedder.calls] == [16, 3]
    saved = await pool.fetchval(
        "SELECT embedding FROM events WHERE user_id = $1 AND external_id = $2",
        "user-1",
        "0",
    )
    assert saved.to_list() == original


async def test_backfill_embedder_failure_leaves_rows_unchanged(pool, tz):
    embedder = RecordingEmbedder(fail=True)
    repo = EventRepository(pool, embedder, tz)
    await import_rows(
        repo,
        "user-1",
        ["Сохранить исходник", "Подготовить отчёт"],
        datetime.now(UTC),
    )
    original = [0.0] * EMBEDDING_DIM
    original[11] = 1.0
    await pool.execute(
        "UPDATE events SET embedding = $1 WHERE user_id = $2 AND external_id = $3",
        original,
        "user-1",
        "0",
    )

    with pytest.raises(RuntimeError, match="current batch was not written"):
        await repo.backfill_imported_embeddings("user-1", limit=1000, batch_size=16)

    assert len(embedder.calls) == 1
    rows = await pool.fetch(
        "SELECT external_id, embedding FROM events WHERE user_id = $1 AND external_id IS NOT NULL",
        "user-1",
    )
    embeddings = {
        row["external_id"]: (None if row["embedding"] is None else row["embedding"].to_list())
        for row in rows
    }
    assert embeddings == {
        "0": original,
        "1": None,
    }


@pytest.mark.parametrize(
    ("limit", "batch_size"),
    [(0, 16), (1001, 16), (1, 0), (1, 65)],
)
async def test_backfill_rejects_unbounded_limits(repo, limit, batch_size):
    with pytest.raises(ValueError):
        await repo.backfill_imported_embeddings("user-1", limit=limit, batch_size=batch_size)


def test_cli_defaults_and_accepts_required_user_id(monkeypatch):
    monkeypatch.setattr("sys.argv", ["index_calendar", "--user-id", "123"])

    args = parse_args()

    assert args.user_id == "123"
    assert args.limit == 1000
    assert args.batch_size == 16


async def test_cli_refuses_missing_credentials_before_connecting(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.index_calendar.get_settings",
        lambda: SimpleNamespace(gigachat_credentials=" "),
    )

    async def unexpected_connect(_dsn):
        raise AssertionError("database should not be opened without credentials")

    monkeypatch.setattr("scripts.index_calendar.create_pool", unexpected_connect)
    with pytest.raises(RuntimeError, match="GIGACHAT_CREDENTIALS"):
        await index_calendar("123", 1000, 16)


@pytest.mark.parametrize("argument", ["--limit", "--batch-size"])
def test_cli_rejects_unbounded_values(monkeypatch, argument):
    monkeypatch.setattr("sys.argv", ["index_calendar", "--user-id", "123", argument, "1001"])

    with pytest.raises(SystemExit):
        parse_args()
