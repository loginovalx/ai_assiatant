"""Интерактивный чат с агентом для ручного тестирования.

Запуск: uv run python scripts/chat.py [user_id]
Команды: :yes — подтвердить, :reset — сбросить историю, :q — выйти.
"""

import sys

import httpx

from common.internal_auth import internal_token

BASE_URL = "http://localhost:8002"


def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
    print(f"Чат с агентом (user_id={user_id!r}). :reset — сброс истории, :q — выход.\n")
    client = httpx.Client(
        timeout=75, headers={"Authorization": f"Bearer {internal_token('agent')}"}
    )
    pending = None
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == ":q":
            break
        if text == ":reset":
            client.post(f"{BASE_URL}/v1/agent/reset", json={"user_id": user_id})
            pending = None
            print("(история сброшена)\n")
            continue

        if text == ":yes" and pending:
            resp = client.post(
                f"{BASE_URL}/v1/agent/confirm", json={"user_id": user_id, "draft": pending}
            )
        else:
            payload = {"user_id": user_id, "text": text}
            if pending:
                payload["edit_draft_id"] = pending["id"]
            resp = client.post(f"{BASE_URL}/v1/agent/message", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("draft"):
            pending = data["draft"]
        elif data["status"] == "done":
            pending = None
        print(f"Агент [{data['status']}]: {data['reply_text']}")
        for e in data.get("events", []):
            print(
                f"  · {e['type']} «{e['name']}» — {e.get('start_at') or 'без даты'} [{e['status']}]"
            )
        print()
    client.close()


if __name__ == "__main__":
    main()
