"""Тесты гоняются на настоящем Postgres с pgvector: SQL поиска на моках не проверить.

Нужна отдельная база, её содержимое удаляется:
    TEST_DATABASE_URL=postgresql://assistant:assistant@localhost:5432/assistant_test pytest
Без переменной тесты, которым нужна база, пропускаются.
"""

import hashlib
import math
import os
import re
from pathlib import Path
from typing import ClassVar

import pytest

TEST_DSN = os.environ.get("TEST_DATABASE_URL")
if TEST_DSN:
    # До первого get_settings(): MCP-сервер и API в тестах подключатся к тестовой базе.
    os.environ["DATABASE_URL"] = TEST_DSN
os.environ["GIGACHAT_CREDENTIALS"] = ""
os.environ["DEFAULT_TZ"] = "Europe/Moscow"
os.environ["API_ALLOW_DEV_USER_ID"] = "true"  # existing API tests use explicit development IDs

from common.config import get_settings  # noqa: E402
from common.db import create_pool  # noqa: E402
from common.embeddings import EMBEDDING_DIM, NullEmbedder  # noqa: E402
from common.repository import EventRepository  # noqa: E402

# Все миграции по порядку — так же, как их применяет Postgres-контейнер при первом старте.
MIGRATIONS = sorted((Path(__file__).parents[1] / "db" / "migrations").glob("*.sql"))


class FakeEmbedder:
    """Детерминированные «эмбеддинги» без сети: слова-синонимы попадают в одно измерение.

    Это не проверка качества GigaChat, а проверка того, что ветка поиска по смыслу
    в SQL работает: запись находится, хотя общих слов с запросом нет.
    """

    SYNONYMS: ClassVar[dict[str, str]] = {
        "слайд": "презентац",
        "созвон": "встреч",
        "контракт": "договор",
    }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def aclose(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        for word in re.findall(r"\w+", text.lower()):
            stem = word[:8] if len(word) <= 8 else word[: max(5, len(word) - 3)]
            stem = next((v for k, v in self.SYNONYMS.items() if stem.startswith(k)), stem)
            key = next((v for v in self.SYNONYMS.values() if stem.startswith(v)), stem)
            vec[int(hashlib.md5(key.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@pytest.fixture(scope="session")
async def pool():
    if not TEST_DSN:
        pytest.skip("TEST_DATABASE_URL не задан")
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    await conn.execute("DROP TABLE IF EXISTS events CASCADE")
    for migration in MIGRATIONS:
        await conn.execute(migration.read_text(encoding="utf-8"))
    await conn.close()

    pool = await create_pool(TEST_DSN)
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
async def _clean(request):
    if "pool" in request.fixturenames:
        pool = request.getfixturevalue("pool")
        await pool.execute("TRUNCATE events")
    yield


@pytest.fixture
def tz():
    return get_settings().tz


@pytest.fixture
def repo(pool, tz):
    return EventRepository(pool, NullEmbedder(), tz)


@pytest.fixture
def semantic_repo(pool, tz):
    return EventRepository(pool, FakeEmbedder(), tz)


@pytest.fixture
async def redis():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL не задан")
    from redis.asyncio import Redis

    client = Redis.from_url(url, decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
def make_agent(pool, redis, tz):
    from agent.core import Agent
    from agent.memory import ConversationMemory
    from agent.tools import McpToolbox
    from mcp_server.server import mcp
    from tests.scripted_llm import ScriptedLLM

    def factory(*script):
        return Agent(
            ScriptedLLM(*script),
            "GigaChat-2-Max",
            McpToolbox(mcp),
            ConversationMemory(redis, 10, 3600),
            tz,
        ), None

    return factory
