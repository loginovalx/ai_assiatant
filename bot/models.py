"""Strict shared Agent API schema (copy into common/ only via its owner)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from common.schemas import DraftAction


class AgentMessageRequest(BaseModel):
    user_id: str
    text: str = Field(min_length=1)
    confirm_changes: bool = False


class AgentMessageResponse(BaseModel):
    reply_text: str
    status: Literal["done", "need_clarification", "needs_confirmation", "error"]
    events: list[dict[str, Any]] = Field(default_factory=list)
    draft: DraftAction | None = None
