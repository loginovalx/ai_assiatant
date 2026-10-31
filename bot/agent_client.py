"""HTTP boundary to Agent Backend owned by developer C."""

from uuid import uuid4

import httpx

from bot.models import AgentMessageRequest, AgentMessageResponse, DraftAction


class AgentUnavailable(Exception):
    pass


class AgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def _post(self, path: str, payload: dict) -> AgentMessageResponse:
        try:
            async with httpx.AsyncClient(timeout=75) as client:
                for attempt in range(2):
                    try:
                        response = await client.post(f"{self.base_url}{path}", json=payload)
                    except httpx.TransportError:
                        if attempt:
                            raise
                        continue  # same request/draft ID makes a lost receipt safe to retry
                    response.raise_for_status()
                    return AgentMessageResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise AgentUnavailable("Агент недоступен или вернул некорректный ответ") from exc
        raise AgentUnavailable("Агент недоступен")

    async def ask(
        self, user_id: str, text: str, confirm_changes: bool = False, request_id: str | None = None
    ) -> AgentMessageResponse:
        request = AgentMessageRequest(
            user_id=user_id,
            text=text,
            confirm_changes=confirm_changes,
            request_id=request_id or str(uuid4()),
        )
        return await self._post(
            "/v1/agent/message", request.model_dump(mode="json", exclude_none=True)
        )

    async def confirm(self, user_id: str, draft: DraftAction) -> AgentMessageResponse:
        return await self._post(
            "/v1/agent/confirm", {"user_id": user_id, "draft": draft.model_dump(mode="json")}
        )
