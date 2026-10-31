"""Evaluate real GigaChat decisions against fixed-time CSV cases and persisted events.

Uses EVAL_DATABASE_URL/EVAL_REDIS_URL, unique users and per-case cleanup. Does not
poll MAX or send messages. Missing seed records make affected cases unverified.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import asyncpg
from openai import AsyncOpenAI
from redis.asyncio import Redis

from agent.core import Agent
from agent.evaluation import EvalCase, judge_case, load_cases
from agent.memory import ConversationMemory
from agent.tools import McpToolbox
from common.config import get_settings
from common.db import create_pool
from common.schemas import EventCreate, EventStatus
from scripts.migrate import apply_migrations


class TracedToolbox(McpToolbox):
    def __init__(self, target):
        super().__init__(target)
        self.calls = []

    async def call(self, user_id, name, arguments, **kwargs):
        entry = {"name": name, "arguments": arguments, "ok": False}
        self.calls.append(entry)
        result = await super().call(user_id, name, arguments, **kwargs)
        entry["ok"] = result.ok
        return result


def read_seeds(path: Path | None) -> list[dict]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or any(
        not isinstance(row, dict) or "id" not in row for row in data
    ):
        raise ValueError("Seeds must be a JSON array of event objects with logical id fields")
    ids = [row["id"] for row in data]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate seed IDs")
    return data


def implementation_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("agent", "common", "mcp_server"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


async def evaluate_case(case: EvalCase, seeds, pool, redis, llm, settings, mcp) -> dict:
    from common.embeddings import build_embedder
    from common.repository import EventRepository

    required = set(case.seed_ids)
    for field in ("expected_ids", "required_result_ids", "forbidden_ids"):
        required.update(case.expected.get(field, []))
    missing = required - {row["id"] for row in seeds}
    # Read queries need the original corpus, including distractors and foreign records.
    if missing or (case.action in {"get_events", "search_events"} and not seeds):
        return {
            "id": case.id,
            "status": "unverified",
            "reasons": ["Missing original seed events"],
            "missing_seed_ids": sorted(missing),
        }

    user_id = "eval-" + uuid4().hex
    owners = [user_id, user_id + "-foreign"]
    memory = ConversationMemory(redis, 10, 3600)
    embedder = build_embedder(settings)
    repo = EventRepository(pool, embedder, settings.tz)
    logical_ids = {}
    toolbox = TracedToolbox(mcp)
    agent = Agent(llm, settings.llm_model, toolbox, memory, settings.tz)
    try:
        for row in seeds:
            data = {k: v for k, v in row.items() if k not in {"id", "status"}}
            data["user_id"] = (
                user_id if row.get("user_id", case.user_id) == case.user_id else owners[1]
            )
            event = await repo.create(EventCreate.model_validate(data))
            if row.get("status"):
                await repo.set_status(event.user_id, event.id, EventStatus(row["status"]))
            logical_ids[row["id"]] = str(event.id)
        if case.history:
            await memory.append(user_id, *case.history)
        before = await pool.fetch("SELECT * FROM events WHERE user_id = $1", user_id)
        baseline_ids = {row["id"] for row in before}
        turns = []
        # Only this isolated evaluation process gets a fixed MCP clock. Runtime
        # endpoints retain the real server clock and accept no client time override.
        with patch("mcp_server.server.datetime", wraps=datetime) as fixed:
            fixed.now.return_value = case.reference_now
            result = await agent.handle(
                user_id, case.text, now=case.reference_now, request_id=uuid4()
            )
            turns.append(result.reply_text)
            if (
                case.action == "create_event"
                and case.expected.get("event_type") == "meeting"
                and result.status == "need_clarification"
                and re.search(r"длительн|продлится", result.reply_text, re.I)
            ):
                result = await agent.handle(
                    user_id, "Один час", now=case.reference_now, request_id=uuid4()
                )
                turns.append(result.reply_text)
        after = await pool.fetch("SELECT * FROM events WHERE user_id = $1", user_id)
        created = [repo._to_event(row) for row in after if row["id"] not in baseline_ids]
        reasons = judge_case(case, result, created, logical_ids, toolbox.calls)
        if case.action != "create_event":
            fields = ("id", "name", "description", "type", "start_at", "duration_minutes", "status")

            def snapshot(rows):
                return {tuple(row[k] for k in fields) for row in rows}

            if snapshot(before) != snapshot(after):
                reasons.append("Read or clarification changed persistent calendar state")
        return {
            "id": case.id,
            "status": "failed" if reasons else "passed",
            "reasons": reasons,
            "reply": result.reply_text,
            "turns": turns,
            "created": [event.model_dump(mode="json") for event in created],
            "calls": toolbox.calls,
        }
    finally:
        await pool.execute("DELETE FROM events WHERE user_id = ANY($1::text[])", owners)
        await memory.reset(user_id)
        await embedder.aclose()


async def run(args) -> dict:
    dsn, redis_url = os.getenv("EVAL_DATABASE_URL"), os.getenv("EVAL_REDIS_URL")
    if not dsn or not redis_url:
        raise ValueError("Set separate EVAL_DATABASE_URL and EVAL_REDIS_URL explicitly")
    os.environ["DATABASE_URL"], os.environ["REDIS_URL"] = dsn, redis_url
    if not args.seeds:
        os.environ["GIGACHAT_CREDENTIALS"] = ""  # creation/clarification need no embeddings
    get_settings.cache_clear()
    settings = get_settings()
    cases, seeds = load_cases(args.cases), read_seeds(args.seeds)
    if args.only:
        cases = [case for case in cases if case.id in set(args.only)]
    cases = cases[: args.limit]
    from mcp_server.server import mcp

    connection = await asyncpg.connect(dsn)
    try:
        await apply_migrations(connection)
    finally:
        await connection.close()
    pool = await create_pool(dsn)
    redis = Redis.from_url(redis_url, decode_responses=True)
    llm = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
    results = []
    report = {
        "model": settings.llm_model,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "implementation_sha256": implementation_digest(),
        "clock": "reference_now per case, including MCP time validation",
        "meeting_policy": "one duration clarification may be answered with One hour",
        "results": results,
    }
    try:
        await redis.ping()
        for index, case in enumerate(cases, 1):
            try:
                result = await evaluate_case(case, seeds, pool, redis, llm, settings, mcp)
            except Exception as exc:
                result = {"id": case.id, "status": "failed", "reasons": [type(exc).__name__]}
            results.append(result)
            print(f"{index}/{len(cases)} {case.id}: {result['status']}", flush=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        passed = sum(row["status"] == "passed" for row in results)
        verified = sum(row["status"] != "unverified" for row in results)
        report.update(
            passed=passed,
            verified=verified,
            total=len(cases),
            coverage=verified / len(cases) if cases else 0,
            verified_pass_rate=passed / verified if verified else 0,
            complete=bool(cases) and verified == len(cases),
            finished_at=datetime.now(UTC).isoformat(),
        )
        try:
            report["git_commit"] = (
                await asyncio.to_thread(
                    subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
                )
            ).strip()
        except subprocess.CalledProcessError:
            report["git_commit"] = "unknown"
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"RESULT: {passed}/{verified} verified; {len(cases) - verified} unverified", flush=True
        )
        return report
    finally:
        await llm.close()
        await redis.aclose()
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument(
        "--seeds", type=Path, help="Original JSON seed events; never inferred from expectations"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append")
    parser.add_argument("--limit", type=int, choices=range(1, 101), default=100, metavar="1..100")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    report = asyncio.run(run(args))
    raise SystemExit(0 if report["complete"] and report["verified_pass_rate"] >= 0.9 else 2)


if __name__ == "__main__":
    main()
