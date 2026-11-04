"""The intake workflow as a LangGraph state machine.

    intake -> classify -> enrich -> decide --(needs approval)--> approve -> execute -> END
                                          \\-----------------------------/

`approve` calls `interrupt()`: the run pauses, its state is checkpointed under the ticket's
thread id, and a reviewer resumes it later with `Command(resume={...})`. Nothing waits in memory
for the human, and a resumed run continues from the gate instead of re-classifying.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .classify import Classifier
from .domain import Action, Category, Decision
from .gateway import ToolCallError, ToolGateway
from .rules import RuleConfig, decide


class FlowState(TypedDict, total=False):
    ticket_id: str
    ticket: dict
    category: str
    confidence: float
    reported_amount: float | None
    policy: dict | None
    decision: dict
    approval: dict
    outcome: str  # done | failed
    log: list[str]


def _decision_dict(d: Decision) -> dict:
    return {
        "category": d.category.value,
        "confidence": d.confidence,
        "priority": d.priority,
        "queue": d.queue,
        "action": d.action.value,
        "payout": d.payout,
        "needs_approval": d.needs_approval,
        "reasons": d.reasons,
    }


def claim_id(ticket_id: str) -> str:
    return f"CLM-{ticket_id}"


def build_graph(
    gw: ToolGateway,
    classifier: Classifier,
    rules: RuleConfig = RuleConfig(),  # noqa: B008
    checkpointer: Any = None,
):  # noqa: ANN201 - compiled graph type is internal to langgraph
    async def intake(state: FlowState) -> FlowState:
        t = await gw.call("ticketing", "get_ticket", {"ticket_id": state["ticket_id"]})
        return {"ticket": t, "log": [f"intake via {t['channel']}"]}

    async def classify(state: FlowState) -> FlowState:
        t = state["ticket"]
        c = await classifier.classify(t["subject"], t["body"])
        amount = c.reported_amount if c.reported_amount is not None else t.get("reported_amount")
        return {
            "category": c.category.value,
            "confidence": c.confidence,
            "reported_amount": amount,
            "log": state["log"] + [f"classified {c.category.value} ({c.confidence:.2f})"],
        }

    async def enrich(state: FlowState) -> FlowState:
        pn = state["ticket"].get("policy_number")
        if not Category(state["category"]).is_claim or not pn:
            return {"policy": None}
        try:
            policy = await gw.call("erp", "lookup_policy", {"policy_number": pn})
        except ToolCallError as e:
            if e.code != "NOT_FOUND":
                raise
            return {"policy": None, "log": state["log"] + [f"policy {pn} not found"]}
        return {"policy": policy, "log": state["log"] + [f"policy {pn} {policy['status']}"]}

    async def decide_node(state: FlowState) -> FlowState:
        d = decide(
            Category(state["category"]),
            state["confidence"],
            state.get("reported_amount"),
            state.get("policy"),
            rules,
        )
        return {
            "decision": _decision_dict(d),
            "log": state["log"] + [f"decided {d.action.value} -> {d.queue}"],
        }

    def approve(state: FlowState) -> FlowState:
        d = state["decision"]
        answer = interrupt(
            {
                "ticket_id": state["ticket_id"],
                "action": d["action"],
                "payout": d["payout"],
                "reasons": d["reasons"],
                "policy_number": state["ticket"].get("policy_number"),
            }
        )
        decision = dict(d)
        if answer.get("approved"):
            if answer.get("payout") is not None:
                decision["payout"] = float(answer["payout"])
            note = f"approved by {answer.get('reviewer', 'reviewer')}"
        else:
            # a rejected automatic action becomes a normal adjuster assignment
            decision.update(action=Action.ASSIGN_ADJUSTER.value, payout=0.0, queue="adjuster-desk")
            note = f"rejected by {answer.get('reviewer', 'reviewer')}: {answer.get('reason', '')}"
        return {"decision": decision, "approval": answer, "log": state["log"] + [note]}

    async def execute(state: FlowState) -> FlowState:
        tid, d = state["ticket_id"], state["decision"]
        action = Action(d["action"])
        cid = claim_id(tid)
        status = {
            Action.AUTO_SETTLE: "closed_paid",
            Action.ASSIGN_ADJUSTER: "assigned",
            Action.REFER_SIU: "referred_siu",
            Action.SERVICE_REPLY: "service",
            Action.MANUAL_TRIAGE: "needs_triage",
            Action.DENY_NO_COVERAGE: "pending_denial_letter",
        }[action]
        try:
            if action in (Action.AUTO_SETTLE, Action.ASSIGN_ADJUSTER) and d["payout"] > 0:
                await gw.call("erp", "set_reserve", {"claim_id": cid, "amount": d["payout"]})
            if action == Action.AUTO_SETTLE and d["payout"] > 0:
                await gw.call(
                    "erp",
                    "issue_payout",
                    {
                        "claim_id": cid,
                        "amount": d["payout"],
                        "idempotency_key": f"{cid}:payout",
                    },
                )
            await gw.call(
                "ticketing",
                "update_ticket",
                {
                    "ticket_id": tid,
                    "status": status,
                    "category": d["category"],
                    "priority": d["priority"],
                    "queue": d["queue"],
                },
            )
            await gw.call(
                "ticketing",
                "add_note",
                {
                    "ticket_id": tid,
                    "note": "; ".join(state["log"] + d["reasons"]),
                },
            )
        except ToolCallError as e:
            return {"outcome": "failed", "log": state["log"] + [str(e)]}
        return {"outcome": "done", "log": state["log"] + [f"executed {action.value}"]}

    def after_decide(state: FlowState) -> str:
        return "approve" if state["decision"]["needs_approval"] else "execute"

    g = StateGraph(FlowState)
    g.add_node("intake", intake)
    g.add_node("classify", classify)
    g.add_node("enrich", enrich)
    g.add_node("decide", decide_node)
    g.add_node("approve", approve)
    g.add_node("execute", execute)
    g.add_edge(START, "intake")
    g.add_edge("intake", "classify")
    g.add_edge("classify", "enrich")
    g.add_edge("enrich", "decide")
    g.add_conditional_edges("decide", after_decide, ["approve", "execute"])
    g.add_edge("approve", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
