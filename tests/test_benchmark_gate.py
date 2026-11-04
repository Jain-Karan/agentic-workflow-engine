"""CI gate: the agent must hold its quality and its safety properties on the synthetic backlog.
Floors are measured values minus a small margin; raise them when the agent improves."""

import pytest

from claimflow.backends import FaultPlan
from claimflow.bench import run_benchmark

pytestmark = pytest.mark.anyio

FAULTS = dict(rate_limited=0.05, unavailable=0.03, timeout_after_commit=0.02, seed=3)


@pytest.mark.parametrize("faulty", [False, True], ids=["clean", "faulty"])
async def test_backlog_quality_and_safety(faulty):
    res, _, _ = await run_benchmark(1000, seed=7, faults=FaultPlan(**FAULTS) if faulty else None)
    # safety: never pay twice, never double-write, never drop a ticket, even under faults
    assert res.duplicate_payments == 0
    assert res.duplicate_notes == 0
    assert res.payout_errors == 0
    assert res.failed == 0
    # quality floors
    assert res.category_accuracy >= 0.96
    assert res.action_accuracy >= 0.89
    assert res.handed_back_to_human <= 0.17
    assert res.handling_time_reduction >= 0.82
