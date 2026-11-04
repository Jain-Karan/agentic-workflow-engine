"""MCP tool servers over the ERP and ticketing backends.

In tests and the benchmark these run in-process (`mcp.Client(server)`); in a deployment each runs
as its own process over stdio or streamable HTTP:

    python -m claimflow.mcp_servers erp --data demo
"""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .backends import BackendError, ERPSystem, TicketingSystem


def _guard(fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    try:
        return fn(*args, **kwargs)
    except BackendError as e:
        # ToolError text reaches the client verbatim, so the agent can tell a retryable
        # RATE_LIMITED from a permanent NOT_FOUND
        raise ToolError(e.wire()) from e


def build_erp_server(erp: ERPSystem) -> MCPServer:
    s = MCPServer("erp", instructions="Policy administration and claims payments.")

    @s.tool()
    def lookup_policy(policy_number: str) -> dict:
        """Policy status, coverages (limit, deductible), prior claims and fraud watch flag."""
        return _guard(erp.lookup_policy, policy_number)

    @s.tool()
    def set_reserve(claim_id: str, amount: float) -> dict:
        """Set the loss reserve for a claim. Repeating the call with the same values is safe."""
        return _guard(erp.set_reserve, claim_id, amount)

    @s.tool()
    def issue_payout(claim_id: str, amount: float, idempotency_key: str | None = None) -> dict:
        """Issue a claim payment. Pass an idempotency_key so a retried call can't pay twice."""
        return _guard(erp.issue_payout, claim_id, amount, idempotency_key)

    return s


def build_ticketing_server(tickets: TicketingSystem) -> MCPServer:
    s = MCPServer("ticketing", instructions="Claims intake ticket queue.")

    @s.tool()
    def get_ticket(ticket_id: str) -> dict:
        """Ticket subject, body, channel, policy number and current status."""
        return _guard(tickets.get_ticket, ticket_id)

    @s.tool()
    def update_ticket(
        ticket_id: str,
        status: str | None = None,
        category: str | None = None,
        priority: str | None = None,
        queue: str | None = None,
    ) -> dict:
        """Set status / category / priority / queue on a ticket."""
        fields = {
            k: v
            for k, v in {
                "status": status,
                "category": category,
                "priority": priority,
                "queue": queue,
            }.items()
            if v is not None
        }
        return _guard(tickets.update_ticket, ticket_id, **fields)

    @s.tool()
    def add_note(ticket_id: str, note: str) -> dict:
        """Append an internal note to the ticket."""
        return _guard(tickets.add_note, ticket_id, note)

    return s


def main(argv: list[str] | None = None) -> int:
    from .synthetic import generate_backlog

    argv = argv if argv is not None else sys.argv[1:]
    which = argv[0] if argv else "erp"
    tickets, policies = generate_backlog(200, seed=1)
    if which == "erp":
        build_erp_server(ERPSystem(policies)).run()
    elif which == "ticketing":
        build_ticketing_server(TicketingSystem({t.ticket_id: t.public() for t in tickets})).run()
    else:
        print(f"unknown server {which!r}; use erp or ticketing", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
