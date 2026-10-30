"""Скриптованная LLM для тестов агента: отдаёт заранее заданные ответы.

Проверяем не качество GigaChat, а нашу логику вокруг неё, поэтому вместо
настоящей модели — список заготовленных ответов.
"""

import json
from typing import Any

from openai.types.chat import ChatCompletion


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> ChatCompletion:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }
    )


def text(content: str) -> ChatCompletion:
    return _completion({"role": "assistant", "content": content})


def _completion(message: dict[str, Any]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "GigaChat-2-Max",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class ScriptedLLM:
    """Отдаёт заранее заданные ответы и запоминает, что ей прислали."""

    def __init__(self, *script: ChatCompletion | Exception) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # снимок
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
