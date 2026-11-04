"""`claimflow` command line.

    claimflow demo                   # run 20 synthetic tickets, print each decision
    claimflow bench -n 2000 --faults # backlog benchmark
    claimflow demo --classifier anthropic   # classify with Claude (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .backends import FaultPlan
from .bench import run_benchmark
from .classify import make_classifier


def _faults(on: bool) -> FaultPlan:
    if not on:
        return FaultPlan()
    return FaultPlan(rate_limited=0.05, unavailable=0.03, timeout_after_commit=0.02, seed=3)


def cmd_bench(args: argparse.Namespace) -> int:
    res, _, _ = asyncio.run(run_benchmark(args.n, args.seed, _faults(args.faults),
                                          make_classifier(args.classifier)))
    print(json.dumps(res.as_dict(), indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    _, results, _ = asyncio.run(run_benchmark(args.n, args.seed, _faults(args.faults),
                                              make_classifier(args.classifier)))
    for r in results:
        d = r.decision or {}
        flag = " [reviewed]" if r.reviewed else ""
        print(f"{r.ticket_id}  {d.get('category', '?'):15} {d.get('action', '?'):17} "
              f"{d.get('queue', '?'):18} ${d.get('payout', 0):>9,.2f}{flag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser(prog="claimflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, n in (("bench", cmd_bench, 2000), ("demo", cmd_demo, 20)):
        p = sub.add_parser(name)
        p.add_argument("-n", type=int, default=n)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--faults", action="store_true", help="inject API failures")
        p.add_argument("--classifier", default="keyword", choices=["keyword", "anthropic"])
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
