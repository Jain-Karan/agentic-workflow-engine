"""Backlog benchmark: accuracy, reliability under injected faults, and modeled handling time.

Handling-time model (minutes of human work in the *intake* queue; downstream adjuster and SIU
work is the same either way and is excluded). These are stated assumptions, not measurements:
change them in `TimeModel` and the benchmark recomputes.

  manual intake, per ticket   read + classify  +  policy lookup (claims with a policy number)
                              +  route/update ticket  +  settle (reserve, payment, close) if
                              the ticket is fast-tracked
  with the agent              a reviewer's time on every approval request, plus the full manual
                              time for tickets the agent hands back (manual triage, failures)
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from .backends import ERPSystem, FaultPlan, TicketingSystem
from .classify import Classifier, KeywordClassifier
from .domain import Action, Category, Ticket
from .gateway import RetryPolicy, ToolGateway
from .graph import build_graph, claim_id
from .mcp_servers import build_erp_server, build_ticketing_server
from .runner import TicketResult, run_backlog
from .synthetic import generate_backlog


@dataclass(frozen=True)
class TimeModel:
    triage: float = 3.0
    policy_lookup: float = 2.5
    route: float = 1.5
    settle: float = 4.0
    review: float = 2.0

    def manual(self, t: Ticket) -> float:
        m = self.triage + self.route
        if Category(t.labels["category"]).is_claim and t.policy_number:
            m += self.policy_lookup
        if t.labels["action"] == Action.AUTO_SETTLE.value:
            m += self.settle
        return m


class VirtualClock:
    """Backoff sleeps advance a counter instead of wall time, so 2,000 tickets with injected
    rate limits run in seconds while still exercising the real retry code path."""

    def __init__(self) -> None:
        self.t = 0.0

    async def sleep(self, d: float) -> None:
        self.t += d
        await asyncio.sleep(0)


def truth_reviewer(by_id: dict[str, Ticket]):  # noqa: ANN201
    """Simulated human: approves when the proposed action and payout match ground truth."""

    async def review(req: dict) -> dict:
        t = by_id[req["ticket_id"]]
        ok = (
            req["action"] == t.labels["action"]
            and abs(req["payout"] - float(t.labels["payout"])) < 0.01
        )
        return {
            "approved": ok,
            "reviewer": "sim-reviewer",
            "reason": "" if ok else "doesn't match the file",
        }

    return review


@dataclass
class BenchResult:
    tickets: int
    category_accuracy: float
    action_accuracy: float
    automated_without_human: float
    sent_for_approval: float
    handed_back_to_human: float
    failed: int
    payout_errors: int
    duplicate_payments: int
    tool_calls: int
    retries: int
    backoff_seconds: float
    manual_minutes: float
    agent_minutes: float
    handling_time_reduction: float

    def as_dict(self) -> dict:
        return asdict(self)


async def run_benchmark(
    n: int = 2000,
    seed: int = 7,
    faults: FaultPlan | None = None,
    classifier: Classifier | None = None,
    time_model: TimeModel = TimeModel(),  # noqa: B008
    concurrency: int = 16,
) -> tuple[BenchResult, list[TicketResult], ERPSystem]:
    tickets, policies = generate_backlog(n, seed)
    by_id = {t.ticket_id: t for t in tickets}
    faults = faults or FaultPlan()
    erp = ERPSystem(policies, faults)
    ticketing = TicketingSystem({t.ticket_id: t.public() for t in tickets}, faults)
    clock = VirtualClock()
    servers = {"erp": build_erp_server(erp), "ticketing": build_ticketing_server(ticketing)}
    async with ToolGateway(servers, RetryPolicy(seed=seed), sleep=clock.sleep) as gw:
        graph = build_graph(gw, classifier or KeywordClassifier())
        results = await run_backlog(graph, list(by_id), truth_reviewer(by_id), concurrency)
        stats = gw.stats

    cat_ok = act_ok = auto = review = handback = failed = pay_err = dup = 0
    manual_min = agent_min = 0.0
    for r in results:
        t = by_id[r.ticket_id]
        manual = time_model.manual(t)
        manual_min += manual
        d = r.decision or {}
        if d.get("category") == t.labels["category"]:
            cat_ok += 1
        if r.outcome != "done":
            failed += 1
            agent_min += manual
            continue
        final_action = d.get("action")
        if r.reviewed:
            review += 1
            agent_min += time_model.review
        # accuracy is judged on what the agent proposed, before any reviewer correction
        proposed = (r.approval and not r.approval.get("approved")) and "rejected" or final_action
        if proposed == t.labels["action"]:
            act_ok += 1
        if final_action == Action.MANUAL_TRIAGE.value:
            handback += 1
            agent_min += manual
        elif not r.reviewed:
            auto += 1
        intended = float(t.labels["payout"]) if t.labels["action"] == "auto_settle" else 0.0
        paid = erp.paid_for(claim_id(r.ticket_id))
        if abs(paid - intended) > 0.01 and final_action == Action.AUTO_SETTLE.value:
            pay_err += 1
        payments = [p for p in erp.payments if p.claim_id == claim_id(r.ticket_id)]
        if len(payments) > 1:
            dup += 1

    total = len(results) or 1
    return (
        BenchResult(
            tickets=len(results),
            category_accuracy=cat_ok / total,
            action_accuracy=act_ok / total,
            automated_without_human=auto / total,
            sent_for_approval=review / total,
            handed_back_to_human=handback / total,
            failed=failed,
            payout_errors=pay_err,
            duplicate_payments=dup,
            tool_calls=stats.calls,
            retries=stats.retries,
            backoff_seconds=round(clock.t, 1),
            manual_minutes=round(manual_min, 1),
            agent_minutes=round(agent_min, 1),
            handling_time_reduction=1 - agent_min / manual_min if manual_min else 0.0,
        ),
        results,
        erp,
    )
