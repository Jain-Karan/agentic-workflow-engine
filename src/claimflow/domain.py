"""Domain model for first-notice-of-loss (FNOL) and service tickets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Category(StrEnum):
    AUTO_COLLISION = "auto_collision"
    AUTO_GLASS = "auto_glass"
    PROPERTY_WATER = "property_water"
    PROPERTY_FIRE = "property_fire"
    THEFT = "theft"
    LIABILITY = "liability"
    BILLING = "billing"
    POLICY_CHANGE = "policy_change"
    OTHER = "other"

    @property
    def is_claim(self) -> bool:
        return self not in (Category.BILLING, Category.POLICY_CHANGE, Category.OTHER)


# Which coverage on the policy a claim category draws on
COVERAGE_FOR = {
    Category.AUTO_COLLISION: "collision",
    Category.AUTO_GLASS: "comprehensive",
    Category.PROPERTY_WATER: "dwelling",
    Category.PROPERTY_FIRE: "dwelling",
    Category.THEFT: "personal_property",
    Category.LIABILITY: "liability",
}


class Action(StrEnum):
    AUTO_SETTLE = "auto_settle"  # pay now, close the ticket
    ASSIGN_ADJUSTER = "assign_adjuster"  # needs a field / desk adjuster
    REFER_SIU = "refer_siu"  # special investigations
    SERVICE_REPLY = "service_reply"  # billing / policy change, no claim
    MANUAL_TRIAGE = "manual_triage"  # agent not confident; a human triages from scratch
    DENY_NO_COVERAGE = "deny_no_coverage"


@dataclass
class Ticket:
    ticket_id: str
    channel: str  # email | web_form | phone_transcript
    subject: str
    body: str
    policy_number: str | None = None
    reported_amount: float | None = None
    # ground truth for the synthetic benchmark; never read by the agent
    labels: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("labels")
        return d


@dataclass
class Decision:
    category: Category
    confidence: float
    priority: str  # P1 (urgent) .. P4
    queue: str
    action: Action
    payout: float = 0.0
    needs_approval: bool = False
    reasons: list[str] = field(default_factory=list)
