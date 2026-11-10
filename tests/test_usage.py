import json
from types import SimpleNamespace

import pytest

from common.usage import record_usage, summarize_usage


def test_proxy_cache_is_subtracted_once_and_no_payload_is_logged(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("USAGE_LOG_PATH", str(path))
    response = SimpleNamespace(
        model="GigaChat-2-Max:version",
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 10,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
        choices=[{"secret": "private request"}],
    )
    record = record_usage(response, operation="agent", source="gpt2giga_openai")
    assert record["billable_tokens"] == 30
    assert record["cached_tokens"] == 100
    assert json.loads(path.read_text()) == record
    assert "private request" not in path.read_text()


def test_native_total_is_already_billable(monkeypatch):
    monkeypatch.delenv("USAGE_LOG_PATH", raising=False)
    record = record_usage(
        {
            "model": "GigaChat-2-Max",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "precached_prompt_tokens": 100,
            },
        },
        operation="search_relevance",
        source="gigachat_native",
    )
    assert record["billable_tokens"] == 30


def test_embeddings_sum_provider_per_item_usage():
    record = record_usage(
        {
            "model": "Embeddings",
            "data": [{"usage": {"prompt_tokens": 7}}, {"usage": {"prompt_tokens": 11}}],
        },
        operation="embeddings",
        source="gigachat_native",
    )
    assert record["billable_tokens"] == 18


def test_missing_usage_is_unknown_not_free():
    record = record_usage(None, operation="agent", source="gpt2giga_openai")
    assert record["billable_tokens"] is None
    summary = summarize_usage([record], chat_rate=650, embedding_rate=14)
    assert summary["unmeasured_calls"] == 1
    assert summary["complete"] is False


def test_token_log_records_truncation_without_response_content():
    record = record_usage(
        {"choices": [{"finish_reason": "length", "message": {"content": "private"}}]},
        operation="agent",
        source="gpt2giga_openai",
    )
    assert record["finish_reason"] == "length"
    assert "private" not in json.dumps(record)


def test_cost_includes_relevance_and_embeddings_and_flags_malformed():
    records = [
        {"operation": "agent", "model": "GigaChat-2-Max", "billable_tokens": 2000},
        {"operation": "search_relevance", "model": "GigaChat-2-Max", "billable_tokens": 1000},
        {"operation": "embeddings", "model": "Embeddings", "billable_tokens": 1000},
        {"operation": "agent", "model": "GigaChat-2-Max", "billable_tokens": -10},
    ]
    summary = summarize_usage(records, chat_rate=650, embedding_rate=14)
    assert summary["measured_cost_rub"] == 1.964
    assert summary["unmeasured_calls"] == 1
    assert summary["billable_tokens"] == 4000


def test_usage_file_failure_does_not_break_successful_call(tmp_path, monkeypatch):
    monkeypatch.setenv("USAGE_LOG_PATH", str(tmp_path))
    record = record_usage(
        {"usage": {"total_tokens": 10}}, operation="agent", source="gpt2giga_openai"
    )
    assert record["billable_tokens"] == 10


def test_mixed_models_are_priced_with_their_own_rates():
    records = [
        {"operation": "agent", "model": "GigaChat-2-Max", "billable_tokens": 1000},
        {"operation": "agent", "model": "GigaChat-2-Pro:version", "billable_tokens": 1000},
        {"operation": "search_relevance", "model": "GigaChat-2-Pro", "billable_tokens": 500},
        {"operation": "embeddings", "model": "Embeddings", "billable_tokens": 1000},
    ]
    report = summarize_usage(
        records, chat_rates={"GigaChat-2-Max": 650, "GigaChat-2-Pro": 500}, embedding_rate=14
    )
    assert report["measured_cost_rub"] == 1.414
    assert report["complete"] and report["billable_tokens"] == 3500
    assert {row["rub_per_million"] for row in report["by_operation"]} == {650, 500, 14}


def test_missing_model_tariff_never_silently_uses_another_models_price():
    records = [{"operation": "agent", "model": "GigaChat-2-Pro", "billable_tokens": 1000}]
    with pytest.raises(ValueError, match="Missing chat tariff"):
        summarize_usage(records, chat_rates={"GigaChat-2-Max": 650}, embedding_rate=14)


def test_flat_rate_still_rejects_a_mixed_model_log():
    records = [
        {"operation": "agent", "model": name, "billable_tokens": 1000}
        for name in ("GigaChat-2-Max", "GigaChat-2-Pro")
    ]
    with pytest.raises(ValueError, match="single tariff"):
        summarize_usage(records, chat_rate=650, embedding_rate=14)


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_invalid_per_model_prices_are_rejected(rate):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_usage([], chat_rates={"GigaChat-2-Pro": rate}, embedding_rate=14)


def test_mixed_model_unknown_usage_stays_incomplete():
    records = [
        {"operation": "agent", "model": "GigaChat-2-Max", "billable_tokens": None},
        {"operation": "agent", "model": "GigaChat-2-Pro", "billable_tokens": 1000},
    ]
    report = summarize_usage(
        records, chat_rates={"GigaChat-2-Max": 650, "GigaChat-2-Pro": 500}, embedding_rate=14
    )
    assert not report["complete"] and report["unmeasured_calls"] == 1
    assert report["measured_cost_rub"] == 0.5
