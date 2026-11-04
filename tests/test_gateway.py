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
    erp = ERPSystem(
        {"POL-000001": make_policy()}, FaultPlan(rate_limited=0.6, retry_after=2.0, seed=1)
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
