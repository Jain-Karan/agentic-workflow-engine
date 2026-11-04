"""Seeded synthetic intake backlog modeled on a personal-lines insurer's FNOL queue.

Every ticket carries ground-truth labels (category, amount, expected action) produced by
applying the same routing rules to the *true* category and policy. The benchmark therefore
measures what the agent contributes (reading the ticket correctly, looking up the right
policy, executing reliably), not whether the rules themselves are good underwriting.
"""

from __future__ import annotations

import random

from .backends import Policy
from .domain import Category, Ticket
from .rules import RuleConfig, decide

# (subject templates, body templates). {amt} -> "$1,234" or omitted, {when}, {where}
TEMPLATES: dict[Category, tuple[list[str], list[str]]] = {
    Category.AUTO_COLLISION: (
        ["Accident on {where}", "Car accident claim", "Rear-ended this morning", "Fender bender"],
        [
            "I was rear-ended at a light on {where} {when}. Bumper and trunk are damaged.{amt}",
            "Another driver ran the stop sign and hit my car at the intersection {when}.{amt}",
            "Minor crash in a parking lot {when}, the other vehicle backed into my door.{amt}",
        ],
    ),
    Category.AUTO_GLASS: (
        ["Windshield crack", "Cracked windscreen", "Glass claim"],
        [
            "A rock hit my windshield on the highway {when} and the chip has spread into a crack.{amt}",
            "Windshield cracked overnight, probably from the cold.{amt}",
        ],
    ),
    Category.PROPERTY_WATER: (
        ["Water leak in kitchen", "Burst pipe", "Water damage claim", "Leak from upstairs"],
        [
            "A pipe burst under the sink {when} and water soaked the cabinets and floor.{amt}",
            "Our water heater failed and flooded part of the basement {when}.{amt}",
            "Water is coming through the ceiling from a leak in the bathroom above.{amt}",
        ],
    ),
    Category.PROPERTY_FIRE: (
        ["Kitchen fire", "Fire damage", "Smoke damage after fire"],
        [
            "We had a kitchen fire {when}. The fire department put it out but there is smoke "
            "damage throughout the first floor.{amt}",
            "Electrical fire in the garage {when}, flames reached the attic.{amt}",
        ],
    ),
    Category.THEFT: (
        ["Break-in at home", "Stolen items", "Burglary claim", "Bike stolen"],
        [
            "Someone broke in {when} and took a laptop and jewelry. Police report filed.{amt}",
            "My bike was stolen from the garage {when}; I have a police report number.{amt}",
            "Burglary {when}: back door forced, TV and camera missing.{amt}",
        ],
    ),
    Category.LIABILITY: (
        ["Guest injured", "Dog bite incident", "Letter from attorney"],
        [
            "A delivery driver slipped on our icy steps {when} and says he was injured.{amt}",
            "Our dog bit the neighbor's child {when}; they are asking us to cover medical bills.{amt}",
            "I received a letter from an attorney about a fall at my property.{amt}",
        ],
    ),
    Category.BILLING: (
        ["Billing question", "Charged twice", "Refund request", "Autopay issue"],
        [
            "I was charged twice for my premium this month, please refund one payment.{amt}",
            "My autopay failed and I want to make sure the policy did not lapse.",
            "Why did my bill go up at renewal?{amt}",
        ],
    ),
    Category.POLICY_CHANGE: (
        ["Add a driver", "Address change", "Update my policy"],
        [
            "I'd like to add a driver to my auto policy, my son just got his license.",
            "Please change my address, we moved to a new address on Elm Street.",
            "I want to update my policy and add coverage for a new boat.",
        ],
    ),
}

# Messages written the way customers actually write: vague, or using another category's words.
HARD_CASES: list[tuple[Category, str, str]] = [
    (
        Category.AUTO_COLLISION,
        "Windshield and hood damage",
        "Hit a deer {when}. Windshield is shattered and the hood is dented.{amt}",
    ),
    (
        Category.PROPERTY_WATER,
        "Damage claim",
        "Storm last night, rain got in through the roof and ruined the ceiling.{amt}",
    ),
    (
        Category.THEFT,
        "Car broken into",
        "Someone smashed my car window and stole my bag {when}.{amt}",
    ),
    (Category.OTHER, "Question", "Can someone call me back about my account? Thanks."),
    (
        Category.LIABILITY,
        "Incident at my house",
        "A guest fell down our stairs {when} and went to urgent care.{amt}",
    ),
]

WHEN = ["this morning", "yesterday", "last night", "on Saturday", "two days ago"]
WHERE = ["Main St", "the 101", "5th Avenue", "Oak Road", "the mall parking lot"]

AMOUNT_RANGE = {
    Category.AUTO_COLLISION: (900, 14000),
    Category.AUTO_GLASS: (250, 1600),
    Category.PROPERTY_WATER: (600, 18000),
    Category.PROPERTY_FIRE: (3000, 60000),
    Category.THEFT: (300, 9000),
    Category.LIABILITY: (1500, 40000),
    Category.BILLING: (80, 400),
}

CATEGORY_MIX = [
    (Category.AUTO_COLLISION, 0.20),
    (Category.AUTO_GLASS, 0.14),
    (Category.PROPERTY_WATER, 0.16),
    (Category.PROPERTY_FIRE, 0.04),
    (Category.THEFT, 0.10),
    (Category.LIABILITY, 0.06),
    (Category.BILLING, 0.17),
    (Category.POLICY_CHANGE, 0.09),
    (Category.OTHER, 0.04),
]


def _policy(rng: random.Random, n: int) -> Policy:
    auto = rng.random() < 0.6
    cov: dict[str, dict] = {}
    if auto:
        cov["collision"] = {"limit": 25000.0, "deductible": rng.choice([250.0, 500.0, 1000.0])}
        if rng.random() < 0.85:
            cov["comprehensive"] = {"limit": 25000.0, "deductible": rng.choice([0.0, 100.0, 250.0])}
    if not auto or rng.random() < 0.3:
        cov["dwelling"] = {"limit": 350000.0, "deductible": rng.choice([500.0, 1000.0, 2500.0])}
        cov["personal_property"] = {"limit": 50000.0, "deductible": rng.choice([250.0, 500.0])}
        cov["liability"] = {"limit": 300000.0, "deductible": 0.0}
    status = rng.choices(["active", "lapsed", "cancelled"], [0.93, 0.05, 0.02])[0]
    return Policy(
        policy_number=f"POL-{n:06d}",
        holder=f"Customer {n}",
        status=status,
        coverages=cov,
        prior_claims_12m=rng.choices([0, 1, 2, 3, 4], [0.62, 0.22, 0.09, 0.05, 0.02])[0],
        fraud_watch=rng.random() < 0.03,
    )


def _policy_dict(p: Policy) -> dict:
    return {
        "policy_number": p.policy_number,
        "holder": p.holder,
        "status": p.status,
        "coverages": p.coverages,
        "prior_claims_12m": p.prior_claims_12m,
        "fraud_watch": p.fraud_watch,
    }


def generate_backlog(
    n: int = 2000, seed: int = 7, hard_case_rate: float = 0.08
) -> tuple[list[Ticket], dict[str, Policy]]:
    rng = random.Random(seed)
    policies = {p.policy_number: p for p in (_policy(rng, i) for i in range(1, n // 2 + 1))}
    numbers = list(policies)
    cats, weights = zip(*CATEGORY_MIX, strict=True)
    tickets: list[Ticket] = []
    for i in range(1, n + 1):
        if rng.random() < hard_case_rate:
            cat, subject, body = rng.choice(HARD_CASES)
        else:
            cat = rng.choices(cats, weights)[0]
            subjects, bodies = TEMPLATES.get(cat, (["Question"], ["Please call me back."]))
            subject, body = rng.choice(subjects), rng.choice(bodies)
        amount = None
        lo_hi = AMOUNT_RANGE.get(cat)
        if lo_hi and rng.random() < 0.85:
            amount = float(round(rng.uniform(*lo_hi) / 5) * 5)
        amt_text = f" Estimate is about ${amount:,.0f}." if amount is not None else ""
        body = body.format(when=rng.choice(WHEN), where=rng.choice(WHERE), amt=amt_text)
        subject = subject.format(where=rng.choice(WHERE))

        r = rng.random()
        if r < 0.06:
            pn = None  # customer didn't include a policy number
        elif r < 0.08:
            pn = f"POL-{rng.randint(900000, 999999)}"  # typo / wrong number
        else:
            pn = rng.choice(numbers)
        if cat.is_claim and pn in policies:
            # claims come from customers whose policy plausibly covers the loss
            for _ in range(20):
                cand = policies[rng.choice(numbers)]
                if (cat in (Category.AUTO_COLLISION, Category.AUTO_GLASS)) == (
                    "collision" in cand.coverages
                ):
                    pn = cand.policy_number
                    break
        policy = policies.get(pn) if pn else None
        truth = decide(cat, 1.0, amount, _policy_dict(policy) if policy else None, RuleConfig())
        tickets.append(
            Ticket(
                ticket_id=f"T{i:05d}",
                channel=rng.choice(["email", "web_form", "phone_transcript"]),
                subject=subject,
                body=body,
                policy_number=pn,
                labels={
                    "category": cat.value,
                    "action": truth.action.value,
                    "amount": "" if amount is None else f"{amount:.2f}",
                    "payout": f"{truth.payout:.2f}",
                },
            )
        )
    return tickets, policies
