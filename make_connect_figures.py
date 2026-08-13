#!/usr/bin/env python3
"""Figures for the joined selection/outcome analysis (methodologies/results.md).

Three panels, matching make_rnl_chart.py's conventions: Typewriter is hatched and given its
own colour throughout, because it is a different model trained to a different boundary (1913
vs 1930) with a different parameter count and tokenizer -- see materials.md §1. Talkie-Web is
NOT drawn as a baseline to subtract: it is ahistorical, it composes, and its presence in these
figures is evidence the mechanism is general rather than an artifact of restriction.

Numbers are recomputed from connect.py's loader rather than transcribed, so a figure cannot
drift away from the table it illustrates.

Run: local/bin/python make_connect_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import connect as C                                      # noqa: E402

FIGURES = Path("figures")
STYLE = {"talkie-base": ("Talkie-Base", "#1f77b4", None),
         "typewriter": ("Typewriter", "#9467bd", "//"),
         "talkie-web": ("Talkie-Web", "#2ca02c", None)}
ORDER = ("talkie-base", "typewriter", "talkie-web")


def fig_coinage(rows, path):
    """Certified coinage on a miss, by scaffold verdict, both arms."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, arm, title in zip(axes, ("post", "in"),
                              ("post-cutoff targets", "in-cutoff targets")):
        x = np.arange(len(ORDER))
        for off, scaff, alpha, lab in ((-0.2, False, 0.45, "not scaffolded"),
                                       (0.2, True, 1.0, "scaffolded")):
            vals, errs = [], []
            for m in ORDER:
                rs = [r for r in rows if r["model"] == m and r["arm"] == arm
                      and not r["hit10"] and r["scaffolded"] == scaff]
                v = [int(r["certified"]) for r in rs]
                p = sum(v) / len(v)
                vals.append(p)
                errs.append(1.96 * (p * (1 - p) / len(v)) ** 0.5)
            for i, m in enumerate(ORDER):
                ax.bar(x[i] + off, vals[i], 0.38, yerr=errs[i], capsize=3,
                       color=STYLE[m][1], alpha=alpha, hatch=STYLE[m][2],
                       edgecolor="white",
                       label=lab if i == 0 and arm == "post" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m][0] for m in ORDER])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("certified coinage rate on a miss")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Scaffolding raises coinage in every model — the mechanism is general",
                 fontsize=11)
    fig.text(0.5, 0.005, "a miss cannot be leakage of the target; pale = unscaffolded, "
                         "solid = scaffolded; bars are 95% normal-approximation intervals",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")


def fig_accounting(rows, path):
    """The composition-consistent share of post-cutoff hits, raw and echo-stripped."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ORDER))
    for off, key, alpha, lab in ((-0.19, "raw", 1.0, "target inside a licensed group"),
                                 (0.19, "clean", 0.45, "… excluding prompt echoes")):
        for i, m in enumerate(ORDER):
            hits = [r for r in rows if r["model"] == m and r["arm"] == "post" and r["hit10"]]
            comp = [r for r in hits if r["target_in_group"]]
            if key == "clean":
                comp = [r for r in comp if r["echo_members"] == 0]
            p = len(comp) / len(hits)
            ax.bar(x[i] + off, p, 0.36, color=STYLE[m][1], alpha=alpha,
                   hatch=STYLE[m][2], edgecolor="white",
                   label=lab if i == 0 else None)
            ax.text(x[i] + off, p + 0.006, f"{p:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[m][0] for m in ORDER])
    ax.set_ylabel("share of post-cutoff hits@10")
    ax.set_title("Upper bound on the composition component of RNL", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.text(0.5, 0.005, "per-model bound, not a comparison: Talkie-Web's RNL is 1.21, "
                         "so it has no deficit to account for",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")


def fig_dose(rows, path):
    """Donor presence and domain, on post-cutoff misses."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    miss = [r for r in rows if r["arm"] == "post" and not r["hit10"]]

    levels = ["none", "<=5", "6-15", ">15"]
    labels = ["no donor", "≤5 words", "6–15", ">15"]
    for m in ORDER:
        ys = []
        for lvl in levels:
            rs = [r for r in miss if r["model"] == m and r["locality"] == lvl]
            ys.append(sum(r["certified"] for r in rs) / len(rs) if rs else np.nan)
        axes[0].plot(range(len(levels)), ys, marker="o", color=STYLE[m][1],
                     linestyle="--" if STYLE[m][2] else "-", label=STYLE[m][0])
    axes[0].set_xticks(range(len(levels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("distance from nearest donor")
    axes[0].set_ylabel("certified coinage rate")
    axes[0].set_title("Donor presence matters; distance does not", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(2)
    for i, m in enumerate(ORDER):
        ys = []
        for dom in ("technical", "everyday"):
            rs = [r for r in miss if r["model"] == m and r["domain"] == dom]
            ys.append(sum(r["certified"] for r in rs) / len(rs) if rs else 0)
        axes[1].bar(x + (i - 1) * 0.26, ys, 0.24, color=STYLE[m][1],
                    hatch=STYLE[m][2], edgecolor="white", label=STYLE[m][0])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["technical", "everyday"])
    axes[1].set_title("Concentrated in the technical lexicon", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(axis="y", alpha=0.3, linewidth=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")


def main():
    FIGURES.mkdir(exist_ok=True)
    rows = C.load_rows()
    fig_coinage(rows, FIGURES / "connect_coinage_on_miss.png")
    fig_accounting(rows, FIGURES / "connect_accounting_bound.png")
    fig_dose(rows, FIGURES / "connect_dose_response.png")


if __name__ == "__main__":
    main()
