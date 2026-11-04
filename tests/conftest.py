import logging

import pytest

from claimflow.backends import ERPSystem, FaultPlan, Policy, TicketingSystem
from claimflow.mcp_servers import build_erp_server, build_ticketing_server

logging.getLogger("mcp").setLevel(logging.CRITICAL)  # NOT_FOUND tests are expected to fail


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_policy(n="POL-000001", status="active", fraud=False, prior=0, **cov):
    coverages = cov or {
        "comprehensive": {"limit": 25000.0, "deductible": 100.0},
        "collision": {"limit": 25000.0, "deductible": 500.0},
        "dwelling": {"limit": 350000.0, "deductible": 1000.0},
        "personal_property": {"limit": 50000.0, "deductible": 250.0},
        "liability": {"limit": 300000.0, "deductible": 0.0},
    }
    return Policy(n, "Pat Doe", status, coverages, prior, fraud)


TICKETS = {
    "T1": {
        "ticket_id": "T1",
        "channel": "email",
        "subject": "Windshield crack",
        "body": "A rock hit my windshield and it cracked. Estimate is about $640.",
        "policy_number": "POL-000001",
        "reported_amount": None,
    },
    "T2": {
        "ticket_id": "T2",
        "channel": "web_form",
        "subject": "Burst pipe",
        "body": "A pipe burst under the sink and water soaked the floor. Estimate is about $2,300.",
        "policy_number": "POL-000001",
        "reported_amount": None,
    },
    "T3": {
        "ticket_id": "T3",
        "channel": "email",
        "subject": "Charged twice",
        "body": "I was charged twice for my premium this month, please refund.",
        "policy_number": "POL-000001",
        "reported_amount": None,
    },
    "T4": {
        "ticket_id": "T4",
        "channel": "email",
        "subject": "Kitchen fire",
        "body": "We had a kitchen fire, smoke damage everywhere. Estimate is about $30,000.",
        "policy_number": "POL-999999",
        "reported_amount": None,
    },
}


@pytest.fixture
def systems():
    erp = ERPSystem({"POL-000001": make_policy()})
    tickets = TicketingSystem(dict(TICKETS))
    return erp, tickets


@pytest.fixture
def servers(systems):
    erp, tickets = systems
    return {"erp": build_erp_server(erp), "ticketing": build_ticketing_server(tickets)}


@pytest.fixture
def no_sleep():
    slept = []

    async def sleep(d):
        slept.append(d)

    sleep.slept = slept
    return sleep


__all__ = ["FaultPlan", "make_policy"]
