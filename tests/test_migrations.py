from pathlib import Path

import pytest
from asyncpg import PostgresSyntaxError

from scripts.migrate import apply_migrations


async def test_migrations_are_transactional_and_repeatable(pool, tmp_path: Path):
    # A temporary schema isolates the runner's bookkeeping from the project migrations.
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute("CREATE SCHEMA migration_test")
            await connection.execute("SET LOCAL search_path TO migration_test")
            (tmp_path / "001.sql").write_text("CREATE TABLE sample(id int)", encoding="utf-8")
            assert await apply_migrations(connection, tmp_path) == ["001.sql"]
            assert await apply_migrations(connection, tmp_path) == []
            (tmp_path / "002.sql").write_text("INSERT INTO sample VALUES (7)", encoding="utf-8")
            (tmp_path / "003.sql").write_text("INVALID SQL", encoding="utf-8")
            with pytest.raises(PostgresSyntaxError):
                await apply_migrations(connection, tmp_path)
            assert await connection.fetchval("SELECT count(*) FROM sample") == 0
            assert await connection.fetchval("SELECT count(*) FROM schema_migrations") == 1
            await connection.execute("DROP SCHEMA migration_test CASCADE")
