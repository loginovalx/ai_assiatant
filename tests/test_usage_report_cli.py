import json
import subprocess
import sys


def test_cli_rejects_empty_log_instead_of_claiming_zero_cost(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.usage_report",
            str(log),
            "--chat-rub-per-million",
            "650",
            "--embedding-rub-per-million",
            "14",
            "--requests",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No usage records" in result.stderr


def test_cli_marks_incomplete_usage_and_omits_projection(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "token_usage",
                "operation": "agent",
                "model": "GigaChat-2-Max",
                "billable_tokens": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.usage_report",
            str(log),
            "--chat-rub-per-million",
            "650",
            "--embedding-rub-per-million",
            "14",
            "--requests",
            "1",
            "--monthly-requests",
            "3000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert not data["complete"]
    assert data["unmeasured_calls"] == 1
    assert data["monthly_api_projection_rub"] is None


def test_cli_accepts_separate_tariffs_and_calculates_a_mixed_model_projection(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "token_usage",
                    "operation": "agent",
                    "model": model,
                    "billable_tokens": 1000,
                }
            )
            for model in ("GigaChat-2-Pro", "GigaChat-2-Max")
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.usage_report",
            str(log),
            "--model-rate",
            "GigaChat-2-Pro=500",
            "--model-rate",
            "GigaChat-2-Max=650",
            "--embedding-rub-per-million",
            "14",
            "--requests",
            "2",
            "--monthly-requests",
            "1000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["complete"] and report["measured_cost_rub"] == 1.15
    assert report["monthly_api_projection_rub"] == 575


def test_cli_reports_missing_model_tariffs_without_a_traceback(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "token_usage",
                "operation": "agent",
                "model": "GigaChat-2-Pro",
                "billable_tokens": 1000,
            }
        )
        + "\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.usage_report",
            str(log),
            "--model-rate",
            "GigaChat-2-Max=650",
            "--embedding-rub-per-million",
            "14",
            "--requests",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing chat tariff" in result.stderr and "Traceback" not in result.stderr
