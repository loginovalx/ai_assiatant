"""Calculate measured API cost from JSONL; no network calls or credentials required."""

import argparse
import json
from pathlib import Path

from common.usage import summarize_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--chat-rub-per-million", type=float, required=True)
    parser.add_argument("--embedding-rub-per-million", type=float, required=True)
    parser.add_argument(
        "--requests",
        type=int,
        required=True,
        help="Number of actual user requests represented by the log",
    )
    parser.add_argument("--monthly-requests", type=int, default=0)
    args = parser.parse_args()
    if args.requests <= 0 or args.monthly_requests < 0:
        parser.error("requests must be positive; monthly-requests must be nonnegative")
    records = []
    for line_number, line in enumerate(args.log.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("event") != "token_usage":
                raise ValueError("Expected token_usage object")
            records.append(record)
        except (ValueError, TypeError) as exc:
            parser.error(f"Invalid usage record at line {line_number}: {exc}")
    if not records:
        parser.error("No usage records; cannot claim a measured zero cost")
    result = summarize_usage(
        records, chat_rate=args.chat_rub_per_million, embedding_rate=args.embedding_rub_per_million
    )
    result["user_requests"] = args.requests
    result["measured_rub_per_request"] = result["measured_cost_rub"] / args.requests
    result["monthly_api_projection_rub"] = (
        result["measured_rub_per_request"] * args.monthly_requests if result["complete"] else None
    )
    result["billing_note"] = (
        "Estimate from observed API responses, not an account invoice. Excludes unobserved "
        "transport retries/timeouts, hosting, minimum payments and package expiry. "
        "Projection assumes the same scenario mix and cache behavior."
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
