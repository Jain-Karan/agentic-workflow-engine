"""Run a backlog through the graph, pausing at approval gates and resuming with a reviewer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langgraph.types import Command

Reviewer = Callable[[dict], Awaitable[dict]]


@dataclass
class TicketResult:
    ticket_id: str
    outcome: str  # done | failed | error
    decision: dict | None
    reviewed: bool = False
    approval: dict | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)


async def run_ticket(graph, ticket_id: str, reviewer: Reviewer) -> TicketResult:  # noqa: ANN001
    cfg = {"configurable": {"thread_id": ticket_id}}
    reviewed = False
    try:
        state = await graph.ainvoke({"ticket_id": ticket_id, "log": []}, cfg)
        while state.get("__interrupt__"):
            request = state["__interrupt__"][0].value
            answer = await reviewer(request)
            reviewed = True
            state = await graph.ainvoke(Command(resume=answer), cfg)
    except Exception as e:  # noqa: BLE001 - one bad ticket must not stop the backlog
        return TicketResult(ticket_id, "error", None, reviewed, error=f"{type(e).__name__}: {e}")
    return TicketResult(
        ticket_id,
        state.get("outcome", "error"),
        state.get("decision"),
        reviewed,
        state.get("approval"),
        log=state.get("log", []),
    )


async def run_backlog(
    graph,  # noqa: ANN001
    ticket_ids: list[str],
    reviewer: Reviewer,
    concurrency: int = 16,
) -> list[TicketResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(tid: str) -> TicketResult:
        async with sem:
            return await run_ticket(graph, tid, reviewer)

    return list(await asyncio.gather(*(one(t) for t in ticket_ids)))
