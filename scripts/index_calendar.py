"""Backfill semantic vectors for CSV-imported events of one user."""

import argparse
import asyncio

from common.config import get_settings
from common.db import create_pool
from common.embeddings import build_embedder
from common.repository import EventRepository


def bounded_int(value: str, *, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not minimum <= result <= maximum:
        raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="MAX user ID whose imports to index")
    parser.add_argument(
        "--limit",
        type=lambda value: bounded_int(value, minimum=1, maximum=1000),
        default=1000,
        help="maximum imported rows to process (1..1000; default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=lambda value: bounded_int(value, minimum=1, maximum=64),
        default=16,
        help="maximum distinct texts per embedding call (1..64; default: 16)",
    )
    return parser.parse_args()


async def index_calendar(user_id: str, limit: int, batch_size: int) -> int:
    settings = get_settings()
    if not settings.gigachat_credentials or not settings.gigachat_credentials.strip():
        raise RuntimeError("GIGACHAT_CREDENTIALS must be configured to index calendar imports")

    embedder = build_embedder(settings)
    try:
        pool = await create_pool(settings.database_url)
        try:
            repo = EventRepository(pool, embedder, settings.tz)
            return await repo.backfill_imported_embeddings(
                user_id, limit=limit, batch_size=batch_size
            )
        finally:
            await pool.close()
    finally:
        await embedder.aclose()


def main() -> None:
    args = parse_args()
    try:
        indexed = asyncio.run(index_calendar(args.user_id, args.limit, args.batch_size))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Indexed {indexed} imported calendar event(s).")


if __name__ == "__main__":
    main()
