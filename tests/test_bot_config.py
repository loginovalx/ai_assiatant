import hmac

from bot.config import Settings


def test_local_bot_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAX_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_URL", raising=False)
    (tmp_path / ".env").write_text(
        "MAX_TOKEN=test-token\nAGENT_URL=http://127.0.0.1:8002\n", encoding="utf-8"
    )
    settings = Settings.from_env()
    assert hmac.compare_digest(settings.max_token, "test-token")
    assert settings.agent_url == "http://127.0.0.1:8002"
    assert "test-token" not in repr(settings)


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAX_TOKEN", "process-token")
    (tmp_path / ".env").write_text("MAX_TOKEN=file-token\n", encoding="utf-8")
    assert Settings.from_env().max_token == "process-token"
