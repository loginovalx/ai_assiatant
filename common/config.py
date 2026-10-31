from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки из переменных окружения или файла .env (см. .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://assistant:assistant@localhost:5432/assistant"
    redis_url: str = "redis://localhost:6379/0"

    # Часовой пояс, в котором трактуется время без явного смещения («завтра в 10»).
    default_tz: str = "Europe/Moscow"

    # GigaChat: ключ авторизации из личного кабинета developers.sber.ru.
    # Пусто — эмбеддинги выключены, поиск работает только по тексту.
    gigachat_credentials: str | None = Field(default=None, repr=False)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl: bool = False
    gigachat_embeddings_model: str = "Embeddings"

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001
    mcp_path: str = "/mcp"

    api_cors_origins: list[str] = ["*"]
    max_token: str = Field(default="", repr=False)
    api_allow_dev_user_id: bool = False
    max_init_data_ttl_seconds: int = Field(default=3600, ge=60, le=86400)

    # --- Агент ---
    # LLM через gpt2giga: OpenAI-совместимый прокси к GigaChat.
    llm_base_url: str = "http://localhost:8090/v1"
    llm_api_key: str = Field(default="change-me", repr=False)  # = GPT2GIGA_API_KEY
    llm_model: str = "GigaChat-2-Max"
    llm_timeout_seconds: float = 60
    mcp_url: str = "http://localhost:8001/mcp"
    agent_host: str = "0.0.0.0"
    agent_port: int = 8002
    agent_history_messages: int = 10  # сколько последних реплик помнить
    agent_context_ttl_days: int = 7
    agent_max_steps: int = 6  # максимум раундов «LLM → тулы» на один запрос

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.default_tz)


@lru_cache
def get_settings() -> Settings:
    return Settings()
