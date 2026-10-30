"""ASR adapters. Audio is forwarded as-is; no lossy re-encoding."""

import asyncio
import logging
from io import BytesIO
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class SpeechRecognitionError(Exception):
    """User-facing failure while recognizing speech."""


class SpeechRecognizer(Protocol):
    async def transcribe(self, audio_bytes: bytes) -> str: ...


class WhisperASR:
    """Local CPU inference, model lazily loaded on first voice message.

    Installation: uv sync --extra asr. Initial model download needs internet.
    PyAV inside faster-whisper can decode OGG/Opus and other common formats.
    """

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()

    async def transcribe(self, audio_bytes: bytes) -> str:
        try:
            async with self._lock:
                return await asyncio.to_thread(self._transcribe_sync, audio_bytes)
        except Exception as exc:
            logger.exception(
                "Whisper ASR failed, audio_size=%d bytes",
                len(audio_bytes),
            )
            raise SpeechRecognitionError("Не удалось распознать запись") from exc

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        segments, _ = self._model.transcribe(BytesIO(audio_bytes), language="ru", beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise SpeechRecognitionError("В сообщении не удалось распознать речь")
        return text


class HttpASR:
    """Generic multipart ASR adapter. Endpoint must return JSON {\"text\":\"...\"}.

    For provider-specific auth/request/response shapes implement a separate
    adapter, keeping the same .transcribe(bytes)->str interface.
    """

    def __init__(self, url: str, token: str = ""):
        self.url = url
        self.token = token

    async def transcribe(self, audio_bytes: bytes) -> str:
        if not self.url:
            raise SpeechRecognitionError("Не настроен ASR_HTTP_URL")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    self.url,
                    headers=headers,
                    files={"file": ("message.ogg", audio_bytes, "application/octet-stream")},
                )
                resp.raise_for_status()
                value = resp.json().get("text")
                if not isinstance(value, str) or not value.strip():
                    raise SpeechRecognitionError("ASR не вернул текст")
                return value.strip()
        except (httpx.HTTPError, ValueError) as exc:
            raise SpeechRecognitionError("Сервис распознавания временно недоступен") from exc


class DisabledASR:
    async def transcribe(self, audio_bytes: bytes) -> str:
        raise SpeechRecognitionError("ASR не настроен, отправьте сообщение текстом")


def make_asr(provider: str, whisper_model: str, http_url: str, token: str) -> SpeechRecognizer:
    if provider == "whisper":
        return WhisperASR(whisper_model)
    if provider == "http":
        return HttpASR(http_url, token)
    if provider == "disabled":
        return DisabledASR()
    raise ValueError(f"Unknown ASR_PROVIDER: {provider}")
