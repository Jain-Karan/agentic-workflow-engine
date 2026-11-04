"""2,000-ticket backlog benchmark, clean and with injected API faults.

    python benchmarks/run_benchmark.py          # prints a Markdown table, writes results.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from claimflow.backends import FaultPlan
from claimflow.bench import run_benchmark

logging.getLogger("mcp").setLevel(logging.CRITICAL)

SCENARIOS = {
    "clean APIs": FaultPlan(),
    "faulty APIs (5% 429, 3% 503, 2% timeout-after-commit)": FaultPlan(
        rate_limited=0.05, unavailable=0.03, timeout_after_commit=0.02, seed=3
    ),
}

ROWS = [
    ("category accuracy", "category_accuracy", "{:.1%}"),
    ("action accuracy (before review)", "action_accuracy", "{:.1%}"),
    ("automated, no human touch", "automated_without_human", "{:.1%}"),
    ("sent for approval", "sent_for_approval", "{:.1%}"),
    ("handed back to a human", "handed_back_to_human", "{:.1%}"),
    ("failed tickets", "failed", "{}"),
    ("payout errors", "payout_errors", "{}"),
    ("duplicate payments", "duplicate_payments", "{}"),
    ("tool calls", "tool_calls", "{:,}"),
    ("retries", "retries", "{:,}"),
    ("backoff time (virtual s)", "backoff_seconds", "{:,}"),
    ("human minutes, manual intake", "manual_minutes", "{:,.0f}"),
    ("human minutes, with agent", "agent_minutes", "{:,.0f}"),
    ("intake handling time reduction", "handling_time_reduction", "{:.1%}"),
]


def main() -> int:
    out = {}
    for name, faults in SCENARIOS.items():
        res, _, _ = asyncio.run(run_benchmark(2000, seed=7, faults=faults))
        out[name] = res.as_dict()
    names = list(out)
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, key, fmt in ROWS:
        print(f"| {label} | " + " | ".join(fmt.format(out[n][key]) for n in names) + " |")
    (Path(__file__).parent / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
