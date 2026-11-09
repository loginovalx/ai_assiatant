"""Calculate measured API cost from JSONL; no network calls or credentials required."""

import argparse
import json
from pathlib import Path

from common.usage import summarize_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    tariffs = parser.add_mutually_exclusive_group(required=True)
    tariffs.add_argument("--chat-rub-per-million", type=float)
    tariffs.add_argument(
        "--model-rate",
        action="append",
        default=[],
        metavar="MODEL=RUB_PER_MILLION",
        help="Repeat for each chat model in a mixed-model log; supply your actual contract rates",
    )
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
    try:
        model_rates = {}
        for item in args.model_rate:
            model, separator, value = item.partition("=")
            if not separator or not model.strip() or model.strip() in model_rates:
                raise ValueError("Each --model-rate needs a unique MODEL=RATE")
            model_rates[model.strip()] = float(value)
        result = summarize_usage(
            records,
            chat_rate=args.chat_rub_per_million,
            chat_rates=model_rates if args.model_rate else None,
            embedding_rate=args.embedding_rub_per_million,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
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
