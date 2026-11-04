"""Fit the keyword classifier's confidence table on a *separate* labelled backlog.

Keyword evidence takes a small set of discrete states (top score, margin over the runner-up),
so calibration is histogram binning: confidence = smoothed empirical accuracy of that state.
States never seen during fitting get 0.5, below the agent's 0.6 floor, so they go to a human.

The benchmark uses seed 7; this fits on seed 11. On real tickets, refit from your own
labelled history: `fit_table(labelled_rows)` only needs (subject, body, true category).

    python benchmarks/calibrate_classifier.py   # writes src/claimflow/calibration.json
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from claimflow.classify import KeywordClassifier, evidence_state
from claimflow.synthetic import generate_backlog

OUT = Path(__file__).resolve().parents[1] / "src" / "claimflow" / "calibration.json"


def fit_table(rows: list[tuple[str, str, str]], alpha: float = 1.0) -> dict[str, float]:
    """rows: (subject, body, true category). Laplace-smoothed accuracy per evidence state."""
    kc = KeywordClassifier(table={})
    cells: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for subject, body, truth in rows:
        scores = kc.scores(f"{subject}\n{body}")
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            continue
        cell = cells[evidence_state(scores)]
        cell[0] += int(best.value == truth)
        cell[1] += 1
    return {k: round((ok + alpha) / (n + 2 * alpha), 4) for k, (ok, n) in sorted(cells.items())}


def main() -> int:
    tickets, _ = generate_backlog(4000, seed=11)
    table = fit_table([(t.subject, t.body, t.labels["category"]) for t in tickets])
    OUT.write_text(
        json.dumps({"fitted_on": "synthetic backlog, seed 11, n=4000", "states": table}, indent=2)
        + "\n"
    )
    print(f"wrote {len(table)} evidence states to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
