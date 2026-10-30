"""HTTP boundary to Agent Backend owned by developer C."""

import httpx

from bot.models import AgentMessageRequest, AgentMessageResponse, DraftAction


class AgentUnavailable(Exception):
    pass


class AgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def ask(
        self, user_id: str, text: str, confirm_changes: bool = False
    ) -> AgentMessageResponse:
        request = AgentMessageRequest(user_id=user_id, text=text, confirm_changes=confirm_changes)
        try:
            async with httpx.AsyncClient(timeout=75) as client:
                response = await client.post(
                    f"{self.base_url}/v1/agent/message",
                    json=request.model_dump(),
                )
                response.raise_for_status()
                return AgentMessageResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise AgentUnavailable("Агент недоступен или вернул некорректный ответ") from exc

    async def confirm(self, user_id: str, draft: DraftAction) -> AgentMessageResponse:
        try:
            async with httpx.AsyncClient(timeout=75) as client:
                response = await client.post(
                    f"{self.base_url}/v1/agent/confirm",
                    json={"user_id": user_id, "draft": draft.model_dump()},
                )
                response.raise_for_status()
                return AgentMessageResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise AgentUnavailable("Не удалось подтвердить действие") from exc
