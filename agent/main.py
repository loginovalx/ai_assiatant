"""HTTP-интерфейс агента для бота (B).

Запуск: uvicorn agent.main:app --port 8002. Документация: http://localhost:8002/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from redis.asyncio import Redis
from redis.exceptions import RedisError

from agent.core import Agent
from agent.memory import ConversationMemory
from agent.service import AgentBusy, AgentService
from agent.tools import McpToolbox
from common.config import get_settings
from common.schemas import AgentConfirmRequest, AgentRequest, AgentResetRequest, AgentResponse

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    llm = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
    agent = Agent(
        llm=llm,
        model=settings.llm_model,
        toolbox=McpToolbox(settings.mcp_url),
        memory=ConversationMemory(
            redis,
            max_messages=settings.agent_history_messages,
            ttl_seconds=settings.agent_context_ttl_days * 24 * 3600,
        ),
        tz=settings.tz,
        max_steps=settings.agent_max_steps,
    )
    app.state.agent = AgentService(
        agent,
        redis,
        ttl_seconds=settings.agent_context_ttl_days * 24 * 3600,
        timeout_seconds=settings.agent_request_timeout_seconds,
    )
    try:
        yield
    finally:
        await llm.close()
        await redis.aclose()


app = FastAPI(title="Voice Assistant Agent", version="0.1.0", lifespan=lifespan)


def get_agent(request: Request) -> AgentService:
    return request.app.state.agent


AgentDep = Annotated[AgentService, Depends(get_agent)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/message", response_model=AgentResponse, response_model_exclude_none=True)
async def message(body: AgentRequest, agent: AgentDep) -> AgentResponse:
    return await agent.handle(
        body.user_id, body.text, confirm_changes=body.confirm_changes, request_id=body.request_id
    )


@app.post("/v1/agent/confirm", response_model=AgentResponse)
async def confirm(body: AgentConfirmRequest, agent: AgentDep) -> AgentResponse:
    return await agent.execute_draft(body.user_id, body.draft)


@app.post("/v1/agent/reset", status_code=204)
async def reset(body: AgentResetRequest, agent: AgentDep) -> None:
    try:
        await agent.reset(body.user_id)
    except AgentBusy as exc:
        raise HTTPException(
            status_code=409, detail="Дождитесь завершения текущего запроса"
        ) from exc
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Контекст временно недоступен") from exc
