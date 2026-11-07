"""Authenticate the internal agent on MCP HTTP transport (not user identity)."""

from fastmcp.server.auth import AccessToken, TokenVerifier

from common.internal_auth import valid_internal_token


class InternalTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        if not valid_internal_token(token, "mcp"):
            return None
        return AccessToken(token=token, client_id="internal-agent", scopes=["events"])
