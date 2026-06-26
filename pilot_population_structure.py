#!/usr/bin/env python3
"""Pilot: word-level composition in post-cutoff recall (descriptive population structure).

Reads the existing cloze detail CSVs for the three models and computes recall RATES across
pre-defined strata (no per-item classification, no new model inference). The headline is a
model x anchoring **double dissociation**: the restricted models (talkie-base, typewriter)
recall a post-cutoff sense far more when the *lemma* is pre-cutoff (a morphological-semantic
ANCHOR exists) and recall rises with lemma entrenchment -- the signature of COMPOSITION; the
leakage-rich control (talkie-web) erases both signatures -- the signature of LEAKAGE.

Definitions (hard-coded):
  recall            = target word in the model's top-100 (0 < rank <= 100)
  cutoff            = 1930 for talkie-base & talkie-web, 1913 for typewriter
                      (same boundary on base & web so strata are item-comparable; for the
                       ahistorical web model the 1930 label is counterfactual)
  in-cutoff         = year <= cutoff;     post-cutoff = year > cutoff
  anchored sense    = year > cutoff and entry_start_year <= cutoff  (form in-training, sense new)
  unanchored word   = year > cutoff and entry_start_year >  cutoff  (form itself postdates training)
  lemma age         = year - entry_start_year  (entrenchment of the host word)

Outputs:
  * prints the per-model strata table, the entrenchment gradient, and the in->post direction
    diagnostic (cliff for restricted models, rise for the ahistorical web model);
  * writes analysis/pilot_population_structure.csv (tidy: model, stratum, N, recalled, rate);
  * with --figure, a grouped-bar figure to figures/pilot_dissociation.png (gitignored).

Run: local/bin/python pilot_population_structure.py [--figure]
CPU-only, deterministic, reads nothing but results/cloze_*_details.csv.
"""

import argparse
import csv
from pathlib import Path

RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIGURES = Path("figures")

# model -> (detail csv, cutoff, kind)
MODELS = [
    ("talkie-base", "cloze_talkie-base_details.csv", 1930, "restricted"),
    ("talkie-web", "cloze_talkie-web_details.csv", 1930, "leakage-rich"),
    ("typewriter", "cloze_typewriter_details.csv", 1913, "restricted"),
]

LEMMA_AGE_BINS = [(0, 50, "0-50y"), (50, 150, "50-150y"),
                  (150, 400, "150-400y"), (400, 10**9, "400y+")]


def _int(v):
    try:
        return int(float(v)) if v not in ("", "None", None) else None
    except (ValueError, TypeError):
        return None


def load(path):
    """Return list of (year, entry_start_year, rank) from a cloze detail CSV."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((_int(r.get("year")), _int(r.get("entry_start_year")),
                         _int(r.get("rank"))))
    return rows


def recalled(rank):
    return rank is not None and 0 < rank <= 100


def rate(rows, pred):
    sub = [r for r in rows if pred(r)]
    rec = [r for r in sub if recalled(r[2])]
    n, k = len(sub), len(rec)
    return n, k, (100.0 * k / n if n else 0.0)


def strata(rows, cutoff):
    """The four headline strata + the post-cutoff total, as (label, N, recalled, rate)."""
    out = []
    out.append(("in-cutoff recall",
                *rate(rows, lambda r: r[0] is not None and r[0] <= cutoff)))
    out.append(("post-cutoff (all)",
                *rate(rows, lambda r: r[0] is not None and r[0] > cutoff)))
    out.append(("post-cutoff anchored sense",
                *rate(rows, lambda r: r[0] is not None and r[0] > cutoff
                      and r[1] is not None and r[1] <= cutoff)))
    out.append(("post-cutoff unanchored word",
                *rate(rows, lambda r: r[0] is not None and r[0] > cutoff
                      and r[1] is not None and r[1] > cutoff)))
    return out


def gradient(rows, cutoff):
    """Anchored-sense recall by lemma-age bin."""
    out = []
    for lo, hi, name in LEMMA_AGE_BINS:
        n, k, p = rate(rows, lambda r, lo=lo, hi=hi: r[0] is not None and r[0] > cutoff
                       and r[1] is not None and r[1] <= cutoff
                       and lo <= (r[0] - r[1]) < hi)
        out.append((name, n, k, p))
    return out


def anchoring_coverage(rows, cutoff):
    """Fraction of post-cutoff items that carry a usable entry_start_year."""
    post = [r for r in rows if r[0] is not None and r[0] > cutoff]
    known = [r for r in post if r[1] is not None]
    return len(known), len(post), (100.0 * len(known) / len(post) if post else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="also write figures/pilot_dissociation.png")
    args = ap.parse_args()

    ANALYSIS.mkdir(exist_ok=True)
    tidy = []          # (model, stratum, N, recalled, rate)
    fig_data = {}      # model -> (anchored_rate, unanchored_rate, kind)

    for name, fname, cutoff, kind in MODELS:
        path = RESULTS / fname
        if not path.exists():
            print(f"!! missing {path}; skipping {name}")
            continue
        rows = load(path)
        st = strata(rows, cutoff)
        gr = gradient(rows, cutoff)
        kcov, ncov, pcov = anchoring_coverage(rows, cutoff)

        in_rate = st[0][3]
        post_rate = st[1][3]
        anch_rate = st[2][3]
        unanch_rate = st[3][3]
        direction = post_rate - in_rate
        diag = "RISE (ahistorical/no real cutoff)" if direction > 0 else "CLIFF (restriction biting)"

        print(f"################  {name}  ({kind}, cutoff {cutoff})  ################")
        for label, n, k, p in st:
            print(f"  {label:32s} N={n:6d}  recalled={k:6d}  rate={p:5.1f}%")
        print(f"  anchored - unanchored gap: {anch_rate - unanch_rate:+5.1f} pp")
        print(f"  in->post direction:        {direction:+5.1f} pp  [{diag}]")
        print(f"  anchoring coverage (post-cutoff items with entry year): "
              f"{kcov}/{ncov} = {pcov:.1f}%")
        print("  entrenchment gradient (anchored-sense recall by lemma age):")
        for nm, n, k, p in gr:
            print(f"      {nm:9s} N={n:5d}  rate={p:5.1f}%")
        print()

        for label, n, k, p in st:
            tidy.append((name, label, n, k, round(p, 1)))
        for nm, n, k, p in gr:
            tidy.append((name, f"anchored lemma-age {nm}", n, k, round(p, 1)))
        fig_data[name] = (anch_rate, unanch_rate, kind)

    out_csv = ANALYSIS / "pilot_population_structure.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "stratum", "N", "recalled", "rate_pct"])
        w.writerows(tidy)
    print(f"wrote {out_csv}  ({len(tidy)} rows)")

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            names = [m for m, *_ in MODELS if m in fig_data]
            anch = [fig_data[m][0] for m in names]
            unanch = [fig_data[m][1] for m in names]
            x = np.arange(len(names))
            w = 0.38
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - w / 2, anch, w, label="anchored novel sense", color="#1f77b4")
            ax.bar(x + w / 2, unanch, w, label="unanchored novel word", color="#d62728")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{m}\n({fig_data[m][2]})" for m in names])
            ax.set_ylabel("post-cutoff recall rate (%)")
            ax.set_title("Word-level composition vs leakage:\nanchoring dependence by model")
            ax.legend()
            for xi, a, u in zip(x, anch, unanch):
                ax.text(xi - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
                ax.text(xi + w / 2, u + 1, f"{u:.1f}", ha="center", fontsize=8)
            FIGURES.mkdir(exist_ok=True)
            out_png = FIGURES / "pilot_dissociation.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            print(f"wrote {out_png}")
        except ImportError as e:
            print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
