#!/usr/bin/env python3
"""Render the binned-recall charts for the current full-corpus runs.

Two charts per K, matching tt.md's layout:
  - Talkie base + web on ONE chart (cutoff 1930)
  - Typewriter on its OWN chart (cutoff 1913)

This is a thin wrapper around visualize_aggregate.py: it just calls that script's
CLI four times (K=10 and K=100 for each chart) so the binning/CI logic stays in
one place. Run with the local venv that has matplotlib:

    local/bin/python make_binned_charts.py
"""

import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")
FIGURES = Path("figures")
VIZ = "visualize_aggregate.py"

# (label, cutoff, output-stem, extra CLI args) per chart family.
CHARTS = [
    {
        "name": "Talkie (base + web)",
        "cutoff": 1930,
        "stem": "recall_binned_cloze",
        "args": [
            "--base", str(RESULTS / "cloze_talkie-base_details.csv"),
            "--web", str(RESULTS / "cloze_talkie-web_details.csv"),
            "--labels", "Base=Talkie-Base,Web=Talkie-Web",
        ],
    },
    {
        "name": "Typewriter",
        "cutoff": 1913,
        "stem": "recall_binned_typewriter",
        "args": [
            "--base", str(RESULTS / "cloze_typewriter_details.csv"),
            "--labels", "Base=Typewriter",
        ],
    },
]

KS = [10, 100]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    failures = []
    for chart in CHARTS:
        for k in KS:
            out = FIGURES / f"{chart['stem']}_{k}.png"
            title = f"Recall@{k} by Time Period — {chart['name']} (cutoff {chart['cutoff']})"
            cmd = [
                sys.executable, VIZ,
                "--cutoff", str(chart["cutoff"]),
                "--k", str(k),
                "--metric", "recall",
                "--title", title,
                "--output", str(out),
                *chart["args"],
            ]
            print(f"\n=== {chart['name']}  K={k}  ->  {out} ===")
            res = subprocess.run(cmd)
            if res.returncode != 0:
                failures.append((chart["name"], k))

    print("\n" + "=" * 60)
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("All charts rendered to figures/:")
    for chart in CHARTS:
        for k in KS:
            print(f"  - {FIGURES / f'{chart['stem']}_{k}.png'}")


if __name__ == "__main__":
    main()
