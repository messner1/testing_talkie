#!/usr/bin/env python3
"""Analyze the synthetic-cousin composition runs (leakage vs inference).

Joins results/composition_<model>_detailed.csv (category == synthetic_seeded)
back to the synthetic_<cutoff>.jsonl that produced it, to recover tier / strategy
/ seed_word (the composition output CSV only keeps `category`).

The question: did the data-restricted model rank any never-existed CORE cousin
of a recalled post-cutoff word in top-100?
  - YES, core cousins recalled  -> inference / composition.
  - NO, only the real seeds were recalled, cousins absent -> leakage.
  - floor tier ~never recalled    -> sanity check on the whole apparatus.

Run: local/bin/python analyze_synthetic_composition.py
"""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
ANALYSIS = Path("analysis")
# model -> (detailed csv, synthetic jsonl, cutoff)
RUNS = {
    "talkie-base": (RESULTS / "composition_talkie-base_detailed.csv",
                    Path("synthetic_1930.jsonl"), 1930),
    "talkie-web": (RESULTS / "composition_talkie-web_detailed.csv",
                   Path("synthetic_1930.jsonl"), 1930),
    "typewriter": (RESULTS / "composition_typewriter_detailed.csv",
                   Path("synthetic_1913.jsonl"), 1913),
}
LEVELS = ["high", "medium", "low"]


def load_meta(jsonl):
    """word(lower) -> {tier, strategy, seed_word, family}."""
    meta = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta[r["word"].strip().lower()] = {
            "tier": r.get("tier"), "strategy": r.get("strategy"),
            "seed_word": r.get("seed_word"), "family": r.get("family"),
        }
    return meta


def recalled(rank):
    return (rank > 0) & (rank <= 100)


def main():
    out = ["# Synthetic-cousin composition: leakage vs inference", ""]
    for model, (csv, jsonl, cutoff) in RUNS.items():
        if not csv.exists() or not jsonl.exists():
            out.append(f"## {model}: MISSING ({csv} / {jsonl})\n")
            continue
        df = pd.read_csv(csv)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        meta = load_meta(jsonl)
        df["wl"] = df["target_word"].str.strip().str.lower()
        for k in ("tier", "strategy", "seed_word", "family"):
            df[k] = df["wl"].map(lambda w: meta.get(w, {}).get(k))
        df["hit"] = recalled(df["rank"])

        n_words = df["wl"].nunique()
        out += [f"## {model}  (cutoff {cutoff})",
                f"- {n_words} synthetic cousins × {len(LEVELS)} context levels = {len(df)} rows",
                f"- ANY-level recall@100: **{df.groupby('wl')['hit'].any().sum()} / {n_words} words**",
                ""]

        # by tier × level
        out.append("### recall@100 by tier × context level (hits / rows)")
        out.append("| tier | high | medium | low |")
        out.append("| --- | --- | --- | --- |")
        for tier in ["core", "floor"]:
            sub = df[df["tier"] == tier]
            cells = []
            for lvl in LEVELS:
                s = sub[sub["context_level"] == lvl]
                cells.append(f"{int(s['hit'].sum())}/{len(s)}")
            out.append(f"| {tier} | {cells[0]} | {cells[1]} | {cells[2]} |")
        out.append("")

        # by strategy
        out.append("### recall@100 by strategy (any-level hits / words)")
        out.append("| strategy | tier | words | words recalled (any level) |")
        out.append("| --- | --- | --- | --- |")
        for strat, g in df.groupby("strategy"):
            tier = g["tier"].iloc[0]
            wr = g.groupby("wl")["hit"].any().sum()
            out.append(f"| {strat} | {tier} | {g['wl'].nunique()} | {int(wr)} |")
        out.append("")

        # exactly which cousins were recalled, with rank + the model's top-10
        hits = df[df["hit"]].sort_values("rank")
        out.append("### every recalled cousin")
        if len(hits) == 0:
            out.append("_none recalled at any level — pure leakage signature_\n")
        else:
            out.append("| word | seed | tier | strategy | level | rank | model top-10 |")
            out.append("| --- | --- | --- | --- | --- | --- | --- |")
            for _, r in hits.iterrows():
                top = str(r.get("top_10_words", "")).replace("|", " · ")
                out.append(f"| **{r['target_word']}** | {r['seed_word']} | {r['tier']} | "
                           f"{r['strategy']} | {r['context_level']} | {int(r['rank'])} | {top} |")
            out.append("")

        df.to_csv(ANALYSIS / f"synthcomp_{model}_joined.csv", index=False)

    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "synthetic_composition_summary.md").write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n-> analysis/synthetic_composition_summary.md")


if __name__ == "__main__":
    main()
