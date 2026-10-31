"""Configuration from environment. MAX token is never baked into source code."""

from dataclasses import dataclass, field
from os import getenv

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    max_token: str = field(repr=False)
    max_api_base: str = "https://platform-api2.max.ru"
    redis_url: str = "redis://localhost:6379/0"
    agent_url: str = "http://localhost:8002"
    asr_provider: str = "whisper"  # whisper | http | disabled
    whisper_model: str = "base"
    asr_http_url: str = ""
    asr_http_token: str = field(default="", repr=False)
    bot_state_ttl: int = 300
    max_voice_bytes: int = 15_000_000
    polling_timeout: int = 20
    reset_state_on_start: bool = True
    # Как часто планировщик проверяет базу и насколько далеко в прошлое смотрит.
    # lookback не даёт после долгого простоя вывалить пользователю всё пропущенное.
    reminder_interval_seconds: int = 30
    reminder_lookback_minutes: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(".env", override=False)
        return cls(
            max_token=getenv("MAX_TOKEN", ""),
            max_api_base=getenv("MAX_API_BASE", "https://platform-api2.max.ru").rstrip("/"),
            redis_url=getenv("REDIS_URL", "redis://localhost:6379/0"),
            agent_url=getenv("AGENT_URL", "http://localhost:8002").rstrip("/"),
            asr_provider=getenv("ASR_PROVIDER", "whisper").lower(),
            whisper_model=getenv("WHISPER_MODEL", "base"),
            asr_http_url=getenv("ASR_HTTP_URL", ""),
            asr_http_token=getenv("ASR_HTTP_TOKEN", ""),
            bot_state_ttl=int(getenv("BOT_STATE_TTL", "300")),
            max_voice_bytes=int(getenv("MAX_VOICE_BYTES", "15000000")),
            polling_timeout=int(getenv("POLLING_TIMEOUT", "20")),
            reset_state_on_start=getenv("RESET_STATE_ON_START", "true").lower()
            in {"true", "1", "yes"},
            reminder_interval_seconds=int(getenv("REMINDER_INTERVAL_SECONDS", "30")),
            reminder_lookback_minutes=int(getenv("REMINDER_LOOKBACK_MINUTES", "60")),
        )
