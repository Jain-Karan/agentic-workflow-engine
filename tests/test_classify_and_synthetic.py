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


async def test_calibrated_confidence_on_held_out_backlog():
    """Regression: softmax confidence sent 12.5% of tickets to a human although the category
    was right. The table is fitted on seed 11; evaluate on seed 7. Confident-but-wrong must not
    rise: a wrong ticket processed automatically is worse than a right one handed back."""
    kc = KeywordClassifier()
    tickets, _ = generate_backlog(1000, seed=7)
    handed_back = confident_wrong = 0
    for t in tickets:
        c = await kc.classify(t.subject, t.body)
        ok = c.category.value == t.labels["category"]
        handed_back += ok and c.confidence < 0.6
        confident_wrong += (not ok) and c.confidence >= 0.6
    assert handed_back / len(tickets) < 0.08
    assert confident_wrong == 0


def test_unseen_evidence_state_falls_back_to_softmax():
    kc = KeywordClassifier(table={})
    s = kc.scores("windshield crack")
    assert kc.confidence(s) == round(kc.softmax_confidence(s), 3)


def test_fit_table_smooths_toward_half():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from calibrate_classifier import fit_table

    table = fit_table([("Glass", "windshield crack", "auto_glass")] * 3)
    assert list(table.values()) == [0.8]  # (3 + 1) / (3 + 2)
