import pytest

from claimflow.backends import ERPSystem, FaultPlan
from claimflow.gateway import RetryPolicy, ToolCallError, ToolGateway
from claimflow.mcp_servers import build_erp_server
from tests.conftest import make_policy

pytestmark = pytest.mark.anyio


async def test_successful_call_returns_json(servers, no_sleep):
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        p = await gw.call("erp", "lookup_policy", {"policy_number": "POL-000001"})
    assert p["status"] == "active" and "collision" in p["coverages"]


async def test_not_found_is_not_retried(servers, no_sleep):
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        with pytest.raises(ToolCallError) as e:
            await gw.call("erp", "lookup_policy", {"policy_number": "nope"})
    assert e.value.code == "NOT_FOUND" and e.value.attempts == 1
    assert no_sleep.slept == []


async def test_rate_limits_are_retried_with_retry_after(no_sleep):
    from claimflow.backends import _unit

    # faults are a pure function of the seed, so pick one whose first attempt is throttled
    seed = next(s for s in range(100) if _unit(s, "before", "lookup_policy", "POL-000001", 0) < 0.6)
    erp = ERPSystem(
        {"POL-000001": make_policy()}, FaultPlan(rate_limited=0.6, retry_after=2.0, seed=seed)
    )
    async with ToolGateway(
        {"erp": build_erp_server(erp)}, RetryPolicy(max_attempts=10, seed=1), sleep=no_sleep
    ) as gw:
        p = await gw.call("erp", "lookup_policy", {"policy_number": "POL-000001"})
    assert p["status"] == "active"
    assert no_sleep.slept and all(d >= 2.0 for d in no_sleep.slept)
    assert gw.stats.retries == len(no_sleep.slept)


async def test_gives_up_after_max_attempts(no_sleep):
    erp = ERPSystem({"POL-000001": make_policy()}, FaultPlan(unavailable=1.0))
    async with ToolGateway(
        {"erp": build_erp_server(erp)}, RetryPolicy(max_attempts=3), sleep=no_sleep
    ) as gw:
        with pytest.raises(ToolCallError) as e:
            await gw.call("erp", "lookup_policy", {"policy_number": "POL-000001"})
    assert e.value.code == "UNAVAILABLE" and e.value.attempts == 3
    assert len(no_sleep.slept) == 2


def test_backoff_grows_and_is_capped():
    pol = RetryPolicy(base=1.0, cap=4.0, seed=0)
    caps = [min(4.0, 2**a) for a in range(6)]
    for a, c in enumerate(caps):
        assert 0 <= pol.delay(a, None) <= c


async def test_payout_retry_after_commit_timeout_pays_once(no_sleep):
    erp = ERPSystem(
        {"POL-000001": make_policy()},
        FaultPlan(timeout_after_commit=1.0, tools=frozenset({"issue_payout"})),
    )
    async with ToolGateway(
        {"erp": build_erp_server(erp)}, RetryPolicy(max_attempts=3), sleep=no_sleep
    ) as gw:
        r = await gw.call(
            "erp",
            "issue_payout",
            {"claim_id": "C1", "amount": 100.0, "idempotency_key": "C1:payout"},
        )
    # first attempt committed then timed out; the retry is recognised as a replay
    assert r["replayed"] is True
    assert erp.calls["issue_payout"] == 2 and len(erp.payments) == 1


async def test_note_retry_after_commit_timeout_writes_once(no_sleep):
    """Regression: add_note had no idempotency key, so a timeout after the note was committed
    plus the gateway's retry wrote the note twice (46 of 2,000 tickets under injected faults)."""
    from claimflow.backends import TicketingSystem
    from claimflow.mcp_servers import build_ticketing_server
    from tests.conftest import TICKETS

    t = TicketingSystem(
        dict(TICKETS), FaultPlan(timeout_after_commit=1.0, tools=frozenset({"add_note"}))
    )
    async with ToolGateway(
        {"ticketing": build_ticketing_server(t)}, RetryPolicy(max_attempts=3), sleep=no_sleep
    ) as gw:
        r = await gw.call(
            "ticketing", "add_note", {"ticket_id": "T1", "note": "hi", "idempotency_key": "T1:note"}
        )
    assert r["replayed"] is True
    assert t.state["T1"]["notes"] == ["hi"]


async def test_fault_benchmark_is_reproducible_regardless_of_concurrency():
    """Regression: faults were drawn from one shared RNG in arrival order, so the same seed
    gave 260 / 256 / 222 retries on three runs of 600 tickets."""
    from claimflow.bench import run_benchmark

    f = dict(rate_limited=0.05, unavailable=0.03, timeout_after_commit=0.02, seed=3)
    runs = [await run_benchmark(200, faults=FaultPlan(**f), concurrency=c) for c in (1, 16, 16)]
    stats = {(r.retries, r.tool_calls, r.backoff_seconds) for r, _, _ in runs}
    assert len(stats) == 1
