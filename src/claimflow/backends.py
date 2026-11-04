"""Simulated ERP (policy admin + claims payments) and ticketing systems.

They stand in for the real systems an intake agent integrates with (Guidewire/SAP-style policy
and payments APIs, a ServiceNow/Zendesk-style ticket queue) and can inject the failures those
APIs actually produce: rate limits with Retry-After, brief outages, and timeouts that happen
*after* the write was committed, which is the case that turns naive retries into duplicate
payments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


class BackendError(Exception):
    """Raised by a backend; `code` is what the MCP layer reports to the agent."""

    def __init__(self, code: str, message: str = "", retry_after: float | None = None) -> None:
        self.code = code
        self.retry_after = retry_after
        super().__init__(message or code)

    def wire(self) -> str:
        s = f"{self.code}: {self.args[0]}"
        if self.retry_after is not None:
            s += f" retry_after={self.retry_after:g}"
        return s


def _unit(*parts: object) -> float:
    """Deterministic uniform [0, 1) from the given parts."""
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


@dataclass
class FaultPlan:
    """Per-call failure probabilities, applied to the tools named in `tools` (all if empty).

    Whether a call fails is a pure function of (seed, tool, call key, attempt number), not a
    draw from a shared random stream. With concurrent tickets the order in which calls reach
    the backend changes run to run, so a shared stream gave a different set of failures, and
    different retry counts, every time.
    """

    rate_limited: float = 0.0
    unavailable: float = 0.0
    timeout_after_commit: float = 0.0  # only meaningful for writes
    retry_after: float = 1.0
    tools: frozenset[str] = frozenset()
    seed: int = 0
    _attempts: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def applies(self, tool: str) -> bool:
        return not self.tools or tool in self.tools

    def before(self, tool: str, key: str = "") -> None:
        n = self._attempts.get((tool, key), 0)
        self._attempts[(tool, key)] = n + 1
        if not self.applies(tool):
            return
        r = _unit(self.seed, "before", tool, key, n)
        if r < self.rate_limited:
            raise BackendError("RATE_LIMITED", f"{tool} throttled", retry_after=self.retry_after)
        if r < self.rate_limited + self.unavailable:
            raise BackendError("UNAVAILABLE", f"{tool} upstream 503")

    def after_commit(self, tool: str, key: str = "") -> None:
        n = self._attempts.get((tool, key), 1) - 1
        if (
            self.applies(tool)
            and _unit(self.seed, "commit", tool, key, n) < self.timeout_after_commit
        ):
            raise BackendError("TIMEOUT", f"{tool} timed out waiting for upstream response")


NO_FAULTS = FaultPlan()


@dataclass
class Policy:
    policy_number: str
    holder: str
    status: str  # active | lapsed | cancelled
    coverages: dict[str, dict]  # coverage -> {"limit": float, "deductible": float}
    prior_claims_12m: int = 0
    fraud_watch: bool = False


@dataclass
class Payment:
    payment_id: str
    claim_id: str
    amount: float
    idempotency_key: str | None


class ERPSystem:
    def __init__(self, policies: dict[str, Policy], faults: FaultPlan = NO_FAULTS) -> None:
        self.policies = policies
        self.faults = faults
        self.reserves: dict[str, float] = {}
        self.payments: list[Payment] = []
        self.calls: dict[str, int] = {}

    def _count(self, tool: str) -> None:
        self.calls[tool] = self.calls.get(tool, 0) + 1

    def lookup_policy(self, policy_number: str) -> dict:
        self._count("lookup_policy")
        self.faults.before("lookup_policy", policy_number)
        p = self.policies.get(policy_number)
        if p is None:
            raise BackendError("NOT_FOUND", f"no policy {policy_number}")
        return {
            "policy_number": p.policy_number,
            "holder": p.holder,
            "status": p.status,
            "coverages": p.coverages,
            "prior_claims_12m": p.prior_claims_12m,
            "fraud_watch": p.fraud_watch,
        }

    def set_reserve(self, claim_id: str, amount: float) -> dict:
        self._count("set_reserve")
        self.faults.before("set_reserve", claim_id)
        self.reserves[claim_id] = amount  # setting a reserve is naturally idempotent
        self.faults.after_commit("set_reserve", claim_id)
        return {"claim_id": claim_id, "reserve": amount}

    def issue_payout(
        self, claim_id: str, amount: float, idempotency_key: str | None = None
    ) -> dict:
        self._count("issue_payout")
        self.faults.before("issue_payout", claim_id)
        if idempotency_key:
            for p in self.payments:
                if p.idempotency_key == idempotency_key:
                    return {"payment_id": p.payment_id, "amount": p.amount, "replayed": True}
        pay = Payment(f"PAY-{len(self.payments) + 1:06d}", claim_id, amount, idempotency_key)
        self.payments.append(pay)
        self.faults.after_commit("issue_payout", claim_id)
        return {"payment_id": pay.payment_id, "amount": amount, "replayed": False}

    def paid_for(self, claim_id: str) -> float:
        return sum(p.amount for p in self.payments if p.claim_id == claim_id)


class TicketingSystem:
    def __init__(self, tickets: dict[str, dict], faults: FaultPlan = NO_FAULTS) -> None:
        self.tickets = tickets  # ticket_id -> public ticket dict
        self.faults = faults
        self.state: dict[str, dict] = {tid: {"status": "new", "notes": []} for tid in tickets}
        self.calls: dict[str, int] = {}
        self._note_keys: dict[str, int] = {}  # idempotency key -> note index

    def _count(self, tool: str) -> None:
        self.calls[tool] = self.calls.get(tool, 0) + 1

    def get_ticket(self, ticket_id: str) -> dict:
        self._count("get_ticket")
        self.faults.before("get_ticket", ticket_id)
        if ticket_id not in self.tickets:
            raise BackendError("NOT_FOUND", f"no ticket {ticket_id}")
        return {**self.tickets[ticket_id], **self.state[ticket_id]}

    def update_ticket(self, ticket_id: str, **fields) -> dict:
        self._count("update_ticket")
        self.faults.before("update_ticket", ticket_id)
        self.state[ticket_id].update(fields)
        self.faults.after_commit("update_ticket", ticket_id)
        return {
            "ticket_id": ticket_id,
            **{k: v for k, v in self.state[ticket_id].items() if k != "notes"},
        }

    def add_note(self, ticket_id: str, note: str, idempotency_key: str | None = None) -> dict:
        self._count("add_note")
        self.faults.before("add_note", ticket_id)
        if idempotency_key and idempotency_key in self._note_keys:
            return {
                "ticket_id": ticket_id,
                "notes": len(self.state[ticket_id]["notes"]),
                "replayed": True,
            }
        self.state[ticket_id]["notes"].append(note)
        if idempotency_key:
            self._note_keys[idempotency_key] = len(self.state[ticket_id]["notes"]) - 1
        self.faults.after_commit("add_note", ticket_id)
        return {"ticket_id": ticket_id, "notes": len(self.state[ticket_id]["notes"])}
