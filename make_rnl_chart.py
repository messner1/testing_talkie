#!/usr/bin/env python3
"""Recall-normalized leakage (RNL) bar chart for the current model set.

Reproduces the figures/rnl_comparison.png idea (lower RNL == better adherence to
the cutoff) but for the three models we actually ship now: Talkie-Base,
Talkie-Web and Typewriter. RNL = future-sample recall / in-cutoff recall, so it
is already normalized for each model's raw strength and is comparable across
models *despite* different cutoffs.

Typewriter is deliberately set apart visually (hatched bars, its own colour, its
cutoff in the legend) because it is a different model trained to a different
boundary (1913 vs 1930) and tested over a different past/future split of the same
50 350-sample cloze set. The per-model cutoff and future-sample count are printed
in an annotation box so the normalization is not mistaken for "same experiment".

All numbers are read straight from results/cloze_<model>_summary.json (produced
by evals/metrics.compute_cloze_summary); nothing is recomputed here.

Run: local/bin/python make_rnl_chart.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("figures")

# display label -> (summary json, bar colour, hatch, is_typewriter)
MODELS = {
    "Talkie-Base": (RESULTS / "cloze_talkie-base_summary.json", "#1f77b4", None),
    "Talkie-Web": (RESULTS / "cloze_talkie-web_summary.json", "#2ca02c", None),
    "Typewriter": (RESULTS / "cloze_typewriter_summary.json", "#9467bd", "//"),
}


def load(summary_path):
    s = json.loads(summary_path.read_text())
    ks = s["ks"]
    rnl = {k: s["metrics_by_k"][str(k)]["recall_normalized_leakage"] for k in ks}
    return {
        "ks": ks,
        "rnl": rnl,
        "cutoff": s["cutoff_year"],
        "future": s["future_samples"],
        "past": s["past_samples"],
        "total": s["total_samples"],
    }


def main():
    data = {name: load(p) for name, (p, _c, _h) in MODELS.items()}

    # ks should agree across models (same eval); use the first model's list.
    ks = data["Talkie-Base"]["ks"]
    names = list(MODELS.keys())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(ks))
    bar_w = 0.25

    for i, name in enumerate(names):
        _path, color, hatch = MODELS[name]
        d = data[name]
        offset = (i - 1) * bar_w  # centre the 3-bar group
        heights = [d["rnl"][k] for k in ks]
        bars = ax.bar(
            x + offset, heights, bar_w,
            color=color, alpha=0.9, edgecolor="black", linewidth=0.6,
            hatch=hatch,
            label=f"{name}  (≤{d['cutoff']})",
        )
        for bx, h in zip(x + offset, heights):
            ax.text(bx, h + 0.015, f"{h:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    # RNL == 1 is the break-even line: future-sense recall equals in-cutoff recall.
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(ks) - 0.5, 1.02, "RNL = 1  (future recall = in-cutoff recall)",
            ha="right", va="bottom", fontsize=8, color="red")

    ax.set_xlabel("k (top-k predictions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall-normalized leakage  (lower = better adherence)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Recall-normalized leakage by model and k", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylim(0, 1.5)  # headroom so the annotation box clears the tall Web bars
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotation: make the differing cutoff / past-future split explicit so the
    # normalization is read correctly.
    lines = ["model            cutoff   past / future samples"]
    for name in names:
        d = data[name]
        lines.append(f"{name:<15} {d['cutoff']}    {d['past']:>6} / {d['future']:<5}")
    note = "\n".join(lines)
    ax.text(0.985, 0.97, note, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "rnl_comparison_3models.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    # echo the table we plotted
    print("\nRNL by k:")
    print("  k     " + "  ".join(f"{n:>12}" for n in names))
    for k in ks:
        print(f"  {k:<5} " + "  ".join(f"{data[n]['rnl'][k]:>12.4f}" for n in names))


if __name__ == "__main__":
    main()
