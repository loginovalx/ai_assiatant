"""Check real agent retrieval against IDs and dates derived from a calendar CSV.

Uses explicit EVAL_DATABASE_URL/EVAL_REDIS_URL and temporary owners. Requires the
configured LLM proxy and GigaChat embeddings; never polls or sends MAX messages.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import asyncpg
from openai import AsyncOpenAI
from redis.asyncio import Redis

from agent.calendar_benchmark import build_calendar_cases
from agent.core import Agent
from agent.evaluation import judge_case
from agent.memory import ConversationMemory
from common.calendar_import import parse_calendar_csv
from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository
from scripts.evaluate_agent import TracedToolbox, implementation_digest
from scripts.migrate import apply_migrations


def snapshot(rows):
    fields = ("id", "name", "description", "type", "start_at", "duration_minutes", "status")
    return {tuple(row[field] for field in fields) for row in rows}


async def run(args) -> dict:
    dsn, redis_url = os.getenv("EVAL_DATABASE_URL"), os.getenv("EVAL_REDIS_URL")
    if not dsn or not redis_url:
        raise ValueError("Set separate EVAL_DATABASE_URL and EVAL_REDIS_URL explicitly")
    os.environ["DATABASE_URL"], os.environ["REDIS_URL"] = dsn, redis_url
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.gigachat_credentials:
        raise ValueError("GIGACHAT_CREDENTIALS is required for the semantic retrieval check")
    now = datetime.fromisoformat(args.reference_time)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("--reference-time must include a UTC offset")
    now = now.astimezone(settings.tz)
    data = args.calendar.read_bytes()
    rows, errors = parse_calendar_csv(data, settings.tz)
    if errors:
        raise ValueError(f"Calendar has {len(errors)} import errors; fix them before benchmarking")
    cases = build_calendar_cases(rows, now)
    available_cases = len(cases)
    if args.only:
        requested = set(args.only)
        if requested - {case.id for case in cases}:
            raise ValueError("--only contains an unknown benchmark case ID")
        cases = [case for case in cases if case.id in requested]

    from mcp_server.server import mcp

    connection = await asyncpg.connect(dsn)
    try:
        await apply_migrations(connection)
    finally:
        await connection.close()
    pool = await create_pool(dsn)
    redis = Redis.from_url(redis_url, decode_responses=True)
    embedder = build_embedder(settings)
    llm = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
    owner = "calendar-check-" + uuid4().hex
    foreign_owner = owner + "-foreign"
    memory = ConversationMemory(redis, 10, 3600)
    repo = EventRepository(pool, embedder, settings.tz)
    report = {
        "benchmark": "source-calendar retrieval",
        "calendar_sha256": hashlib.sha256(data).hexdigest(),
        "calendar_rows": len(rows),
        "available_cases": available_cases,
        "selected_cases": len(cases),
        "full_40_case_coverage": available_cases == len(cases) == 40,
        "reference_now": now.isoformat(),
        "model": settings.llm_model,
        "implementation_sha256": implementation_digest(),
        "started_at": datetime.now(UTC).isoformat(),
        "oracle": "raw source dates and numeric IDs; exact lists and required search recall",
        "results": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        await redis.ping()
        report["imported"] = await repo.import_calendar(owner, rows)
        # Identical records belonging to another owner are an explicit isolation
        # probe, never part of the source-derived expected result set.
        await repo.import_calendar(foreign_owner, rows)
        mapping = {
            row["external_id"]: str(row["id"])
            for row in await pool.fetch(
                "SELECT id, external_id FROM events WHERE user_id=$1", owner
            )
        }
        foreign_ids = {
            str(row["id"])
            for row in await pool.fetch("SELECT id FROM events WHERE user_id=$1", foreign_owner)
        }
        print(f"Imported {len(mapping)} rows; indexing semantic vectors", flush=True)
        indexed = 0
        while True:
            count = await repo.backfill_imported_embeddings(owner)
            indexed += count
            if not count:
                break
        report["indexed"] = indexed
        before = await pool.fetch("SELECT * FROM events WHERE user_id=$1", owner)
        baseline = snapshot(before)
        baseline_ids = {row["id"] for row in before}
        foreign_before = snapshot(
            await pool.fetch("SELECT * FROM events WHERE user_id=$1", foreign_owner)
        )
        save()
        for index, original in enumerate(cases, 1):
            case = replace(original, user_id=owner)
            await memory.reset(owner)
            toolbox = TracedToolbox(mcp)
            agent = Agent(llm, settings.llm_model, toolbox, memory, settings.tz)
            result = {"id": case.id, "query": case.text, "expected": case.expected}
            try:
                with patch("mcp_server.server.datetime", wraps=datetime) as clock:
                    clock.now.return_value = now
                    response = await agent.handle(owner, case.text, now=now, request_id=uuid4())
                after = await pool.fetch("SELECT * FROM events WHERE user_id=$1", owner)
                created = [repo._to_event(row) for row in after if row["id"] not in baseline_ids]
                reasons = judge_case(case, response, created, mapping, toolbox.calls)
                if snapshot(after) != baseline:
                    reasons.append("Read action changed persistent calendar state")
                if foreign_ids & {str(event.id) for event in response.events}:
                    reasons.append("Response contains another owner's event")
                result.update(
                    status="failed" if reasons else "passed",
                    reasons=reasons,
                    response=response.model_dump(mode="json"),
                    calls=toolbox.calls,
                )
            except Exception as exc:
                result.update(status="failed", reasons=[type(exc).__name__], calls=toolbox.calls)
            report["results"].append(result)
            print(f"{index}/{len(cases)} {case.id}: {result['status']}", flush=True)
            save()
        report["foreign_calendar_unchanged"] = foreign_before == snapshot(
            await pool.fetch("SELECT * FROM events WHERE user_id=$1", foreign_owner)
        )
        report.update(
            passed=sum(row["status"] == "passed" for row in report["results"]),
            total=len(cases),
            finished_at=datetime.now(UTC).isoformat(),
        )
        save()
        print(f"RESULT: {report['passed']}/{report['total']}", flush=True)
        return report
    finally:
        await pool.execute(
            "DELETE FROM events WHERE user_id = ANY($1::text[])", [owner, foreign_owner]
        )
        await memory.reset(owner)
        await llm.close()
        await embedder.aclose()
        await redis.aclose()
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calendar", type=Path)
    parser.add_argument(
        "--reference-time", required=True, help="ISO timestamp including UTC offset"
    )
    parser.add_argument("--output", type=Path, required=True, help="Private JSON report path")
    parser.add_argument("--only", nargs="*", help="Run only selected L01-L20/S01-S20 cases")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    report = asyncio.run(run(args))
    if report["passed"] != report["total"] or not report["foreign_calendar_unchanged"]:
        raise SystemExit(1)
    if not args.only and not report["full_40_case_coverage"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
