"""Apply pending SQL migrations without deleting an existing database."""

import asyncio
from pathlib import Path

import asyncpg

from common.config import get_settings

MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"
LOCK_ID = 742193806


def read_migrations(directory: Path) -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.sql"))]


async def apply_migrations(
    connection: asyncpg.Connection, directory: Path = MIGRATIONS
) -> list[str]:
    applied = []
    sources = await asyncio.to_thread(read_migrations, directory)
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", LOCK_ID)
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        existing = {r["name"] for r in await connection.fetch("SELECT name FROM schema_migrations")}
        for name, sql in sources:
            if name in existing:
                continue
            await connection.execute(sql)
            await connection.execute("INSERT INTO schema_migrations(name) VALUES ($1)", name)
            applied.append(name)
    return applied


async def main() -> None:
    connection = await asyncpg.connect(get_settings().database_url)
    try:
        applied = await apply_migrations(connection)
        print("Applied: " + (", ".join(applied) if applied else "already up to date"))
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
