"""Интерактивный чат с агентом для ручного тестирования.

Запуск: uv run python scripts/chat.py [user_id]
Команды: :reset — сбросить историю, :q — выйти.
"""

import sys

import httpx

BASE_URL = "http://localhost:8002"


def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
    print(f"Чат с агентом (user_id={user_id!r}). :reset — сброс истории, :q — выход.\n")
    client = httpx.Client(timeout=60)
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
            print("(история сброшена)\n")
            continue

        resp = client.post(f"{BASE_URL}/v1/agent/message", json={"user_id": user_id, "text": text})
        data = resp.json()
        print(f"Агент [{data['status']}]: {data['reply_text']}")
        for e in data.get("events", []):
            print(
                f"  · {e['type']} «{e['name']}» — {e.get('start_at') or 'без даты'} [{e['status']}]"
            )
        print()


if __name__ == "__main__":
    main()
