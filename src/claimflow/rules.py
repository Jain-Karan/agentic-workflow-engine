"""Routing rules: from (category, confidence, reported amount, policy) to a Decision.

Kept separate from the LangGraph wiring so underwriting/claims staff can review the business
logic without reading graph code, and so it can be unit-tested exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import COVERAGE_FOR, Action, Category, Decision


@dataclass(frozen=True)
class RuleConfig:
    confidence_floor: float = 0.6  # below this the agent hands the ticket to a human
    auto_settle_max: float = 2500.0  # reported loss at or under this can be fast-tracked
    approval_over: float = 1000.0  # any payout above this waits for a human approval
    siu_prior_claims: int = 3
    fast_track: frozenset[Category] = frozenset(
        {Category.AUTO_GLASS, Category.PROPERTY_WATER, Category.THEFT}
    )


PRIORITY = {
    Category.LIABILITY: "P1",
    Category.PROPERTY_FIRE: "P1",
    Category.AUTO_COLLISION: "P2",
    Category.PROPERTY_WATER: "P2",
    Category.THEFT: "P3",
    Category.AUTO_GLASS: "P3",
}

ADJUSTER_QUEUE = {
    Category.AUTO_COLLISION: "adjuster-auto",
    Category.AUTO_GLASS: "adjuster-auto",
    Category.PROPERTY_WATER: "adjuster-property",
    Category.PROPERTY_FIRE: "adjuster-property",
    Category.THEFT: "adjuster-property",
    Category.LIABILITY: "adjuster-liability",
}


def decide(
    category: Category,
    confidence: float,
    reported_amount: float | None,
    policy: dict | None,
    cfg: RuleConfig = RuleConfig(),  # noqa: B008 - frozen dataclass, safe default
) -> Decision:
    if confidence < cfg.confidence_floor:
        return Decision(
            category,
            confidence,
            "P3",
            "intake-manual",
            Action.MANUAL_TRIAGE,
            reasons=[f"confidence {confidence:.2f} < {cfg.confidence_floor}"],
        )

    if not category.is_claim:
        return Decision(
            category,
            confidence,
            "P4",
            "service",
            Action.SERVICE_REPLY,
            reasons=["service request, not a claim"],
        )

    if policy is None:
        return Decision(
            category,
            confidence,
            "P3",
            "intake-manual",
            Action.MANUAL_TRIAGE,
            reasons=["no policy found for ticket"],
        )

    if policy["status"] != "active":
        return Decision(
            category,
            confidence,
            "P3",
            "coverage-review",
            Action.DENY_NO_COVERAGE,
            needs_approval=True,
            reasons=[f"policy {policy['status']}"],
        )

    coverage = policy["coverages"].get(COVERAGE_FOR[category])
    if coverage is None:
        return Decision(
            category,
            confidence,
            "P3",
            "coverage-review",
            Action.DENY_NO_COVERAGE,
            needs_approval=True,
            reasons=[f"no {COVERAGE_FOR[category]} coverage on policy"],
        )

    if policy["fraud_watch"] or policy["prior_claims_12m"] >= cfg.siu_prior_claims:
        why = "fraud watch" if policy["fraud_watch"] else f"{policy['prior_claims_12m']} claims"
        return Decision(category, confidence, "P2", "siu", Action.REFER_SIU, reasons=[why])

    priority = PRIORITY[category]
    if reported_amount is None:
        return Decision(
            category,
            confidence,
            priority,
            ADJUSTER_QUEUE[category],
            Action.ASSIGN_ADJUSTER,
            reasons=["no amount reported"],
        )

    payout = max(0.0, min(reported_amount - coverage["deductible"], coverage["limit"]))
    if category in cfg.fast_track and reported_amount <= cfg.auto_settle_max:
        return Decision(
            category,
            confidence,
            priority,
            "fast-track",
            Action.AUTO_SETTLE,
            payout=round(payout, 2),
            needs_approval=payout > cfg.approval_over,
            reasons=[f"fast-track {category.value} <= ${cfg.auto_settle_max:,.0f}"],
        )

    return Decision(
        category,
        confidence,
        priority,
        ADJUSTER_QUEUE[category],
        Action.ASSIGN_ADJUSTER,
        payout=round(payout, 2),
        reasons=[f"${reported_amount:,.0f} needs an adjuster"],
    )
