"""Calls MCP tools with retries.

Retry policy: exponential backoff with full jitter (delay = U(0, min(cap, base * 2**attempt))),
never shorter than a server-supplied Retry-After, only for errors the server marks retryable.
Permanent errors (NOT_FOUND, validation) fail immediately.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import Client

RETRYABLE = {"RATE_LIMITED", "UNAVAILABLE", "TIMEOUT"}
_CODE = re.compile(r"\b(RATE_LIMITED|UNAVAILABLE|TIMEOUT|NOT_FOUND|INVALID)\b")
_RETRY_AFTER = re.compile(r"retry_after=([0-9.]+)")


class ToolCallError(RuntimeError):
    def __init__(self, server: str, tool: str, code: str, message: str, attempts: int) -> None:
        self.server, self.tool, self.code, self.attempts = server, tool, code, attempts
        super().__init__(f"{server}.{tool} failed after {attempts} attempt(s): {message}")


@dataclass
class RetryPolicy:
    max_attempts: int = 6
    base: float = 0.25
    cap: float = 8.0
    seed: int | None = None  # None: fresh jitter every run; int: reproducible per call

    def delay(self, attempt: int, retry_after: float | None, key: str = "") -> float:
        if self.seed is None:
            u = random.random()
        else:
            h = hashlib.blake2b(f"{self.seed}|{key}|{attempt}".encode(), digest_size=8).digest()
            u = int.from_bytes(h, "big") / 2**64
        d = u * min(self.cap, self.base * 2**attempt)
        return max(d, retry_after or 0.0)


@dataclass
class GatewayStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    slept: float = 0.0
    by_code: dict[str, int] = field(default_factory=dict)


class ToolGateway:
    """Holds one MCP client per server and routes `call(server, tool, args)` with retries."""

    def __init__(
        self,
        servers: dict[str, object],  # name -> MCPServer instance, URL, or StdioServerParameters
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._targets = servers
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._clients: dict[str, Client] = {}
        self._stack: AsyncExitStack | None = None
        self.stats = GatewayStats()

    async def __aenter__(self) -> ToolGateway:
        self._stack = AsyncExitStack()
        for name, target in self._targets.items():
            self._clients[name] = await self._stack.enter_async_context(Client(target))
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN002
        if self._stack:
            await self._stack.aclose()

    async def call(self, server: str, tool: str, args: dict) -> dict:
        client = self._clients[server]
        last_msg, last_code = "", "UNKNOWN"
        for attempt in range(self.policy.max_attempts):
            self.stats.calls += 1
            result = await client.call_tool(tool, args)
            text = result.content[0].text if result.content else ""
            if not result.is_error:
                return json.loads(text) if text else {}
            m = _CODE.search(text)
            last_code, last_msg = (m.group(1) if m else "UNKNOWN"), text
            self.stats.by_code[last_code] = self.stats.by_code.get(last_code, 0) + 1
            if last_code not in RETRYABLE or attempt == self.policy.max_attempts - 1:
                break
            ra = _RETRY_AFTER.search(text)
            key = f"{server}.{tool}:{json.dumps(args, sort_keys=True)}"
            d = self.policy.delay(attempt, float(ra.group(1)) if ra else None, key)
            self.stats.retries += 1
            self.stats.slept += d
            await self._sleep(d)
        self.stats.failures += 1
        raise ToolCallError(server, tool, last_code, last_msg, attempt + 1)
