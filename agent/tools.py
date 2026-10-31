"""Мост между MCP-сервером и OpenAI-форматом тулов.

Список тулов и их описания берём с MCP-сервера, а не дублируем: единственный источник
правды — mcp_server/server.py. По дороге:
- убираем user_id из схемы: модель его не видит и не может подменить,
  агент подставляет его сам из запроса бота;
- упрощаем JSON Schema до подмножества, которое стабильно понимает GigaChat.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client, FastMCP

logger = logging.getLogger(__name__)

HIDDEN_ARGS = frozenset({"user_id", "dry_run", "idempotency_key"})

# Тул без MCP: им модель сообщает, что ей не хватает данных. Явный сигнал надёжнее,
# чем угадывать по знаку вопроса в ответе.
ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Задать пользователю уточняющий вопрос, если не хватает данных, "
            "например времени встречи. После вызова диалог ждёт ответа пользователя."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Короткий вопрос по-русски"}
            },
            "required": ["question"],
        },
    },
}

_KEEP_KEYS = frozenset(
    {
        "type",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "format",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
    }
)


@dataclass
class ToolResult:
    ok: bool
    payload: Any  # данные тула или текст ошибки
    is_mutation: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def for_llm(self) -> str:
        if not self.ok:
            return json.dumps({"error": self.payload}, ensure_ascii=False)
        return json.dumps(_compact(self.payload), ensure_ascii=False, default=str)


# Поля, которые модели не нужны: меньше токенов — быстрее и дешевле ответ.
_NOISE_FIELDS = frozenset({"user_id", "created_at", "updated_at"})


def _compact(value: Any) -> Any:
    """Рекурсивно: в create/update пересекающиеся записи вложены в поле conflicts."""
    if isinstance(value, list):
        return [_compact(v) for v in value]
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in _NOISE_FIELDS}
    return value


MUTATING_TOOLS = frozenset(
    {"create_event", "update_event", "delete_event", "complete_event", "delete_events"}
)


class McpToolbox:
    """target — URL MCP-сервера, или сам объект FastMCP (в тестах, без сети)."""

    def __init__(self, target: str | FastMCP) -> None:
        self._target = target
        self._schemas: list[dict[str, Any]] | None = None

    async def openai_tools(self) -> list[dict[str, Any]]:
        if self._schemas is None:
            async with Client(self._target) as client:
                tools = await client.list_tools()
            self._schemas = [
                to_openai_tool(t.name, t.description or "", t.input_schema) for t in tools
            ]
        return [*self._schemas, ASK_USER_TOOL]

    async def call(
        self,
        user_id: str,
        name: str,
        arguments: dict[str, Any],
        *,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        args = {k: v for k, v in arguments.items() if k not in HIDDEN_ARGS}
        args["user_id"] = user_id  # всегда наш, даже если модель прислала свой
        if dry_run and name in MUTATING_TOOLS:
            args["dry_run"] = True
        if idempotency_key is not None and name == "create_event":
            args["idempotency_key"] = idempotency_key
        async with Client(self._target) as client:
            result = await client.call_tool(name, args, raise_on_error=False)
        if result.is_error:
            # Текст ошибки тула (ToolError) написан для LLM: «нужно указать время» и т. п.
            message = " ".join(getattr(c, "text", "") for c in result.content).strip()
            logger.info("tool %s rejected: %s", name, message)
            return ToolResult(ok=False, payload=message or "Ошибка инструмента")

        data = result.structured_content
        if isinstance(data, dict) and set(data) == {"result"}:
            data = data["result"]  # FastMCP заворачивает списки в {"result": [...]}
        events = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        return ToolResult(ok=True, payload=data, is_mutation=name in MUTATING_TOOLS, events=events)


def to_openai_tool(name: str, description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    params = _simplify(input_schema)
    params["properties"] = {
        k: v for k, v in params.get("properties", {}).items() if k not in HIDDEN_ARGS
    }
    params["required"] = [r for r in params.get("required", []) if r not in HIDDEN_ARGS]
    return {
        "type": "function",
        "function": {"name": name, "description": description.strip(), "parameters": params},
    }


def _simplify(schema: dict[str, Any]) -> dict[str, Any]:
    """`anyOf: [X, null]` → X; служебные ключи (title, default, additionalProperties) — прочь."""
    any_of = schema.get("anyOf")
    if any_of:
        non_null = [s for s in any_of if s.get("type") != "null"]
        merged = {
            **(non_null[0] if non_null else {}),
            **{k: v for k, v in schema.items() if k != "anyOf"},
        }
        return _simplify(merged)

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _KEEP_KEYS:
            continue
        if key == "properties":
            out[key] = {prop: _simplify(sub) for prop, sub in value.items()}
        elif key == "items":
            out[key] = _simplify(value)
        else:
            out[key] = value
    return out
