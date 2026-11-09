"""Compare serialized context sizes offline. Counts characters, NOT GigaChat tokens."""

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.compact import plan_request
from agent.prompt import build_system_prompt
from agent.tools import ASK_USER_TOOL, to_openai_tool
from mcp_server.server import mcp

SAMPLES = (
    "Создай задачу «ТЕСТ Оплатить интернет».",
    "Создай задачу подготовить отчёт",
    "Напомни завтра в 18:00 ТЕСТ купить молоко.",
    "Поставь встречу «ТЕСТ Созвон с Иваном» завтра в 15:00 на час.",
    "Что у меня завтра?",
)


def payload_chars(messages, tools):
    return len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False))


async def compare():
    now = datetime(2026, 11, 9, 16, 48, tzinfo=ZoneInfo("Europe/Moscow"))
    tools = [
        to_openai_tool(t.name, t.description or "", t.parameters) for t in await mcp.list_tools()
    ]
    tools.append(ASK_USER_TOOL)
    rows = []
    for command in SAMPLES:
        baseline = payload_chars(
            [
                {"role": "system", "content": build_system_prompt(now)},
                {"role": "user", "content": command},
            ],
            tools,
        )
        plan = plan_request(command, now)
        assert plan is not None
        if not plan.direct:
            plan.prepare(command, tools)
            assert plan.active
        compact = 0 if plan.direct else payload_chars(plan.messages, plan.tools)
        rows.append(
            {
                "command": command,
                "baseline_payload_chars": baseline,
                "economy_payload_chars": compact,
                "economy_model_calls": 0 if plan.direct else 1,
                "payload_reduction_percent": round(100 * (1 - compact / baseline), 2),
            }
        )
    return {
        "note": (
            "Offline sizes, not measured tokens or live-model quality. No API/DB calls. "
            "Creation still uses embeddings when enabled."
        ),
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(compare()), ensure_ascii=False, indent=2))
