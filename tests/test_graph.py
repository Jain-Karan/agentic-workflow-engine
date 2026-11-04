import pytest
from langgraph.types import Command

from claimflow.classify import KeywordClassifier
from claimflow.gateway import ToolGateway
from claimflow.graph import build_graph, claim_id
from claimflow.runner import run_backlog, run_ticket

pytestmark = pytest.mark.anyio


async def approve_all(req):
    return {"approved": True, "reviewer": "test"}


async def reject_all(req):
    return {"approved": False, "reviewer": "test", "reason": "check photos"}


async def test_small_glass_claim_auto_settles(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        r = await run_ticket(build_graph(gw, KeywordClassifier()), "T1", approve_all)
    assert r.outcome == "done" and not r.reviewed
    assert r.decision["action"] == "auto_settle"
    assert erp.paid_for(claim_id("T1")) == 540.0
    assert tickets.state["T1"]["status"] == "closed_paid"


async def test_payout_over_threshold_pauses_for_approval(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        graph = build_graph(gw, KeywordClassifier())
        cfg = {"configurable": {"thread_id": "T2"}}
        state = await graph.ainvoke({"ticket_id": "T2", "log": []}, cfg)
        req = state["__interrupt__"][0].value
        assert req["action"] == "auto_settle" and req["payout"] == 1300.0
        assert erp.paid_for(claim_id("T2")) == 0  # nothing paid while waiting
        state = await graph.ainvoke(Command(resume={"approved": True, "payout": 1200}), cfg)
    assert state["outcome"] == "done"
    assert erp.paid_for(claim_id("T2")) == 1200.0  # reviewer's edited amount


async def test_rejected_payout_becomes_adjuster_assignment(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        r = await run_ticket(build_graph(gw, KeywordClassifier()), "T2", reject_all)
    assert r.reviewed and r.decision["action"] == "assign_adjuster"
    assert erp.paid_for(claim_id("T2")) == 0
    assert tickets.state["T2"]["queue"] == "adjuster-desk"


async def test_billing_ticket_routes_to_service(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        r = await run_ticket(build_graph(gw, KeywordClassifier()), "T3", approve_all)
    assert r.decision["action"] == "service_reply"
    assert "lookup_policy" not in erp.calls  # no ERP call for non-claims


async def test_unknown_policy_is_handed_to_a_human(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        r = await run_ticket(build_graph(gw, KeywordClassifier()), "T4", approve_all)
    assert r.outcome == "done" and r.decision["action"] == "manual_triage"
    assert tickets.state["T4"]["queue"] == "intake-manual"


async def test_backlog_runs_concurrently(systems, servers, no_sleep):
    erp, tickets = systems
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        results = await run_backlog(
            build_graph(gw, KeywordClassifier()),
            ["T1", "T2", "T3", "T4"],
            approve_all,
            concurrency=4,
        )
    assert [r.outcome for r in results] == ["done"] * 4
    assert all(len(tickets.state[t]["notes"]) == 1 for t in tickets.state)


async def test_agent_sends_note_key_so_retries_dont_duplicate(no_sleep):
    """Regression at the graph level: the server-side fix alone passed its unit test while the
    agent still sent add_note without a key."""
    from claimflow.backends import ERPSystem, FaultPlan, TicketingSystem
    from claimflow.mcp_servers import build_erp_server, build_ticketing_server
    from tests.conftest import TICKETS, make_policy

    faults = FaultPlan(timeout_after_commit=1.0, tools=frozenset({"add_note"}))
    tickets = TicketingSystem(dict(TICKETS), faults)
    servers = {
        "erp": build_erp_server(ERPSystem({"POL-000001": make_policy()})),
        "ticketing": build_ticketing_server(tickets),
    }
    async with ToolGateway(servers, sleep=no_sleep) as gw:
        await run_ticket(build_graph(gw, KeywordClassifier()), "T1", approve_all)
    assert len(tickets.state["T1"]["notes"]) == 1
