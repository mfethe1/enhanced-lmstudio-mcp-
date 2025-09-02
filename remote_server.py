import os
import json
import asyncio
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import server as mcp_server

# Config (read most values once; token is read dynamically for tests that monkeypatch env)
REMOTE_ENABLED = os.getenv("REMOTE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
REMOTE_BIND = os.getenv("REMOTE_BIND", "0.0.0.0:8787")
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "10"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))

def _get_remote_token() -> str:
    return os.getenv("MCP_REMOTE_TOKEN", "").strip()

app = FastAPI(title="Enhanced LM Studio MCP Remote Server", version="1.0")

# Basic CORS (can be tightened as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple token-bucket rate limiter per client key (token or IP)
class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.capacity = burst
        self.tokens = burst
        self.timestamp = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

_buckets = {}

def _client_key(request: Request, authorization: Optional[str]) -> str:
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    # Fallback to client IP
    return request.client.host if request.client else "unknown"

def _check_auth(authorization: Optional[str]):
    token_env = _get_remote_token()
    if token_env:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
        token = authorization.split(" ", 1)[1]
        if token != token_env:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")

async def _rate_limit(request: Request, authorization: Optional[str]):
    key = _client_key(request, authorization)
    bucket = _buckets.get(key)
    if bucket is None:
        bucket = TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)
        _buckets[key] = bucket
    if not bucket.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.post("/rpc")
async def rpc_endpoint(payload: dict, authorization: Optional[str] = Header(None), request: Request = None):
    if not REMOTE_ENABLED:
        raise HTTPException(status_code=403, detail="Remote access disabled")
    _check_auth(authorization)
    await _rate_limit(request, authorization)

    # Expect MCP-style JSON-RPC payload
    try:
        result = mcp_server.handle_tool_call(payload)
        # Ensure MCP text-only response already handled by server; just pass through
        return JSONResponse(result)
    except mcp_server.ValidationError as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32602, "message": f"Invalid params: {str(e)}"}
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}
        }, status_code=500)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not REMOTE_ENABLED:
        await ws.close(code=4403)
        return
    # Auth during upgrade via query param 'token' or header 'Authorization' (if framework forwards it)
    token = ws.query_params.get("token")
    try:
        token_env = _get_remote_token()
        if token_env:
            if token:
                if token != token_env:
                    await ws.close(code=4401)
                    return
            else:
                # If header is not forwarded, we enforce query token
                await ws.close(code=4401)
                return
        await ws.accept()
        # naive per-connection rate limiter
        bucket = TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)
        while True:
            data = await ws.receive_text()
            if not bucket.allow():
                await ws.send_text(json.dumps({"jsonrpc": "2.0", "error": {"code": 429, "message": "Rate limit exceeded"}}))
                continue
            try:
                payload = json.loads(data)
            except Exception:
                await ws.send_text(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}))
                continue
            try:
                result = mcp_server.handle_tool_call(payload)
            except mcp_server.ValidationError as e:
                result = {"jsonrpc": "2.0", "id": payload.get("id"), "error": {"code": -32602, "message": f"Invalid params: {str(e)}"}}
            except Exception as e:
                result = {"jsonrpc": "2.0", "id": payload.get("id"), "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}}
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return

# Entrypoint helper
async def serve():
    if not REMOTE_ENABLED:
        print("Remote endpoints disabled (REMOTE_ENABLED=false)")
        return
    host, port = REMOTE_BIND.split(":", 1)
    import uvicorn
    config = uvicorn.Config(app, host=host, port=int(port), log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

