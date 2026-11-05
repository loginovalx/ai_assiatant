import pytest

from bot.asr import SpeechRecognitionError, WhisperASR


class Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.options: dict = {}

    def transcribe(self, audio, **options):
        self.options = options
        return iter(self.segments), object()


async def test_whisper_enables_voice_activity_detection_and_preserves_words():
    model = FakeWhisperModel([Segment("  Напомни позвонить Саше в 10:05.  ")])
    asr = WhisperASR()
    asr._model = model

    assert await asr.transcribe(b"audio") == "Напомни позвонить Саше в 10:05."
    assert model.options == {"language": "ru", "beam_size": 5, "vad_filter": True}


async def test_whisper_keeps_specific_no_speech_error():
    asr = WhisperASR()
    asr._model = FakeWhisperModel([])

    with pytest.raises(SpeechRecognitionError, match="В сообщении не удалось распознать речь"):
        await asr.transcribe(b"silence")
