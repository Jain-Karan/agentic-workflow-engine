"""Ticket classification.

`KeywordClassifier` is an offline, deterministic baseline (weighted keyword evidence, softmax
confidence) so CI and the benchmark run without an API key. `LLMClassifier` uses any LangChain
chat model with structured output; `make_classifier("anthropic")` gives Claude.
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol

from pydantic import BaseModel, Field

from .domain import Category


class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)
    reported_amount: float | None = Field(
        default=None, description="Amount the customer says the loss or charge is, in USD"
    )


class Classifier(Protocol):
    async def classify(self, subject: str, body: str) -> Classification: ...


KEYWORDS: dict[Category, dict[str, float]] = {
    Category.AUTO_COLLISION: {
        "collision": 3,
        "rear-ended": 3,
        "rear ended": 3,
        "fender": 2,
        "crash": 2,
        "accident": 1.5,
        "bumper": 2,
        "hit": 1,
        "car": 1,
        "vehicle": 1,
        "intersection": 1.5,
    },
    Category.AUTO_GLASS: {"windshield": 3, "windscreen": 3, "chip": 1.5, "crack": 1, "glass": 2},
    Category.PROPERTY_WATER: {
        "water": 2,
        "leak": 2.5,
        "pipe": 2,
        "burst": 2,
        "flood": 2,
        "ceiling": 1,
        "basement": 1.5,
        "water heater": 2.5,
    },
    Category.PROPERTY_FIRE: {"fire": 3, "smoke": 2, "burn": 2, "kitchen fire": 3, "flames": 2},
    Category.THEFT: {
        "stolen": 3,
        "theft": 3,
        "break-in": 3,
        "broke in": 3,
        "burglary": 3,
        "missing": 1,
        "police report": 1.5,
    },
    Category.LIABILITY: {
        "slipped": 3,
        "injured": 2,
        "lawsuit": 3,
        "dog bit": 3,
        "neighbor": 1,
        "fell": 2,
        "attorney": 2,
    },
    Category.BILLING: {
        "bill": 2,
        "billing": 3,
        "charged twice": 3,
        "premium": 2,
        "refund": 2,
        "invoice": 2,
        "autopay": 2.5,
        "payment": 1,
    },
    Category.POLICY_CHANGE: {
        "add a driver": 3,
        "remove": 1.5,
        "change my address": 3,
        "new address": 2.5,
        "update my policy": 3,
        "add coverage": 3,
        "beneficiary": 2,
    },
}
_AMOUNT = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")


def extract_amount(text: str) -> float | None:
    vals = [float(m.replace(",", "")) for m in _AMOUNT.findall(text)]
    return max(vals) if vals else None


class KeywordClassifier:
    def __init__(self, temperature: float = 1.5) -> None:
        self.temperature = temperature

    def scores(self, text: str) -> dict[Category, float]:
        t = text.lower()
        return {cat: sum(w for kw, w in kws.items() if kw in t) for cat, kws in KEYWORDS.items()}

    async def classify(self, subject: str, body: str) -> Classification:
        text = f"{subject}\n{body}"
        s = self.scores(text)
        best = max(s, key=s.get)
        if s[best] == 0:
            return Classification(
                category=Category.OTHER, confidence=0.3, reported_amount=extract_amount(text)
            )
        z = [math.exp(v / self.temperature) for v in s.values()]
        conf = math.exp(s[best] / self.temperature) / sum(z)
        return Classification(
            category=best, confidence=round(conf, 3), reported_amount=extract_amount(text)
        )


SYSTEM = (
    "You triage insurance intake tickets. Pick exactly one category, give a calibrated "
    "confidence (0-1) and the dollar amount the customer reports, if any. Categories: "
    + ", ".join(c.value for c in Category)
)


class LLMClassifier:
    def __init__(self, chat_model) -> None:  # noqa: ANN001 - any LangChain BaseChatModel
        self.chain = chat_model.with_structured_output(Classification)

    async def classify(self, subject: str, body: str) -> Classification:
        return await self.chain.ainvoke(
            [("system", SYSTEM), ("human", f"Subject: {subject}\n\n{body}")]
        )


def make_classifier(kind: str = "keyword", model: str = "claude-sonnet-4-5") -> Classifier:
    if kind == "keyword":
        return KeywordClassifier()
    if kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("classifier=anthropic needs ANTHROPIC_API_KEY")
        return LLMClassifier(ChatAnthropic(model=model, temperature=0))
    raise ValueError(f"unknown classifier {kind!r}")
