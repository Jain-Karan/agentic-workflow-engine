import collections

import pytest

from claimflow.classify import KeywordClassifier, extract_amount
from claimflow.domain import Category
from claimflow.synthetic import generate_backlog

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("A rock hit my windshield and the chip spread", Category.AUTO_GLASS),
        ("pipe burst and the basement flooded", Category.PROPERTY_WATER),
        ("someone broke in and stole my laptop", Category.THEFT),
        ("I was charged twice for my premium", Category.BILLING),
        ("please change my address", Category.POLICY_CHANGE),
    ],
)
async def test_keyword_classifier(text, want):
    assert (await KeywordClassifier().classify("", text)).category == want


async def test_no_evidence_is_other_with_low_confidence():
    c = await KeywordClassifier().classify("Hi", "Can someone call me?")
    assert c.category == Category.OTHER and c.confidence < 0.6


def test_extract_amount_takes_largest():
    assert extract_amount("deductible $500, estimate $2,350.50") == 2350.5
    assert extract_amount("no numbers") is None


def test_backlog_is_seeded_and_labelled():
    a, pa = generate_backlog(300, seed=3)
    b, _ = generate_backlog(300, seed=3)
    assert [t.body for t in a] == [t.body for t in b]
    assert all({"category", "action", "payout"} <= t.labels.keys() for t in a)
    mix = collections.Counter(t.labels["category"] for t in a)
    assert mix["auto_collision"] > mix["property_fire"]
    assert len(pa) == 150
