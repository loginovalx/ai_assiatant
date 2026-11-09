"""Provider-reported accounting without prompts, names, user identities or keys."""

import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from common.config import get_settings

logger = logging.getLogger(__name__)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def record_usage(
    response: Any,
    *,
    operation: str,
    source: str,
    model: str = "unknown",
    context_profile: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Native GigaChat total is billable; gpt2giga adds cached tokens back to it.

    A missing/invalid provider counter remains unknown. It must not silently
    become a zero-cost call. Failed requests can also be billed by the provider.
    """
    usage = _field(response, "usage")
    cached = _count(_field(usage, "precached_prompt_tokens"))
    if cached is None:
        cached = _count(_field(_field(usage, "prompt_tokens_details"), "cached_tokens")) or 0
    total = _count(_field(usage, "total_tokens"))
    prompt = _count(_field(usage, "prompt_tokens"))
    completion = _count(_field(usage, "completion_tokens"))
    if operation == "embeddings":
        items = _field(response, "data")
        counts = [_count(_field(_field(item, "usage"), "prompt_tokens")) for item in items or []]
        total = sum(counts) if counts and all(c is not None for c in counts) else None
        prompt, completion = total, 0
    billable = total
    if total is not None and source == "gpt2giga_openai":
        billable = total - cached if total >= cached else None
    record = {
        "event": "token_usage",
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "source": source,
        "model": _field(response, "model") or model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "reported_total_tokens": total,
        "billable_tokens": billable,
    }
    if context_profile is not None:
        record["context_profile"] = context_profile
    if request_id is not None:
        record["request_id"] = request_id
    line = json.dumps(record, ensure_ascii=False)
    logger.info("token_usage %s", line)
    path = os.getenv("USAGE_LOG_PATH") or get_settings().usage_log_path
    if path:
        try:
            # One append per small record; multiple services may share a mounted log.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            logger.warning("Could not append token usage log")
    return record


def summarize_usage(
    records: list[dict[str, Any]],
    *,
    chat_rate: float | None = None,
    embedding_rate: float,
    chat_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rates in RUB per million tokens, supplied explicitly for the active contract.

    A flat tariff remains supported for single-model logs. Mixed-model logs
    require an explicit rate for every model; unknown tariffs are never guessed.
    """
    if chat_rates is not None and chat_rate is not None:
        raise ValueError("Choose either chat_rate or per-model chat_rates")
    rates = [embedding_rate, *(chat_rates or {}).values()]
    if chat_rate is not None:
        rates.append(chat_rate)
    if any(not Decimal(str(rate)).is_finite() or rate < 0 for rate in rates):
        raise ValueError("Rates must be finite and nonnegative")
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "billable_tokens": 0, "unmeasured_calls": 0}
    )
    chat_models = set()
    for record in records:
        operation, model = record.get("operation", "unknown"), record.get("model", "unknown")
        if operation != "embeddings" and model != "unknown":
            chat_models.add(model.split(":", 1)[0])
        group = groups[(operation, model)]
        group["calls"] += 1
        tokens = _count(record.get("billable_tokens"))
        if tokens is None:
            group["unmeasured_calls"] += 1
        else:
            group["billable_tokens"] += tokens
    if len(chat_models) > 1 and chat_rates is None:
        raise ValueError("Split usage by chat model before applying a single tariff")
    rows, cost = [], Decimal(0)
    for (operation, model), group in sorted(groups.items()):
        if operation == "embeddings":
            rate = embedding_rate
        elif chat_rates is not None:
            base_model = model.split(":", 1)[0]
            if model in chat_rates:
                rate = chat_rates[model]
            elif base_model in chat_rates:
                rate = chat_rates[base_model]
            else:
                raise ValueError(f"Missing chat tariff for model {model}")
        elif chat_rate is not None:
            rate = chat_rate
        else:
            raise ValueError(f"Missing chat tariff for model {model}")
        value = Decimal(group["billable_tokens"]) * Decimal(str(rate)) / 1_000_000
        cost += value
        rows.append(
            {
                "operation": operation,
                "model": model,
                **group,
                "rub_per_million": rate,
                "measured_cost_rub": float(value),
            }
        )
    missing = sum(row["unmeasured_calls"] for row in rows)
    return {
        "calls": sum(row["calls"] for row in rows),
        "billable_tokens": sum(row["billable_tokens"] for row in rows),
        "unmeasured_calls": missing,
        "complete": missing == 0,
        "measured_cost_rub": float(cost),
        "by_operation": rows,
    }
