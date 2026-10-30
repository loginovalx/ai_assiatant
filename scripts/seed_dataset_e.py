"""Засеивает events датасетом E — для демонстрации семантического поиска по старым задачам.

Запуск из корня проекта:
    uv run python scripts/seed_dataset_e.py dataset_e.tsv.csv demo
"""

import asyncio
import csv
import sys
from datetime import datetime

from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder

BATCH_SIZE = 20

MEETING_WORDS = ("Встреча", "Созвон", "Демо", "планёрку", "Планёрку")


def guess_type(text: str) -> str:
    if text.startswith("Напомнить"):
        return "reminder"
    if any(w in text for w in MEETING_WORDS):
        return "meeting"
    return "task"


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        return list(csv.DictReader(f, delimiter=delimiter))


async def main(path: str, user_id: str) -> None:
    settings = get_settings()
    tz = settings.tz
    pool = await create_pool(settings.database_url)
    embedder = build_embedder(settings)

    rows = read_rows(path)
    print(f"Прочитано строк: {len(rows)}, колонки: {list(rows[0].keys()) if rows else '?'}")

    now = datetime.now(tz)
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        vectors = await embedder.embed([r["text"] for r in chunk]) or [None] * len(chunk)

        for row, vec in zip(chunk, vectors, strict=False):
            start_at = datetime.fromisoformat(f"{row['date']}T{row['time']}").replace(tzinfo=tz)
            await pool.execute(
                """
                INSERT INTO events (user_id, type, name, description, start_at, status, embedding)
                VALUES ($1, $2, $3, '', $4, $5, $6)
                """,
                user_id,
                guess_type(row["text"]),
                row["text"][:200],
                start_at,
                "done" if start_at < now else "active",
                vec,
            )
            inserted += 1
        print(f"{inserted}/{len(rows)}")

    await pool.close()
    await embedder.aclose()
    print(f"Готово: {inserted} событий у user_id={user_id!r}")


if __name__ == "__main__":
    asyncio.run(
        main(
            sys.argv[1] if len(sys.argv) > 1 else "dataset_e.tsv",
            sys.argv[2] if len(sys.argv) > 2 else "demo",
        )
    )
