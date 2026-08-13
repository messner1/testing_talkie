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

CUEING (important).  The super sets mix two populations.  The original hand-built
words each carry a context written for that specific coinage; the mechanically
expanded words (analysis/synth_candidates_*_mech.json) inherit a *family template*
context written for a different word -- e.g. `backon`, `handon`, `headon` are all
scored against `scintillon`'s deuteron/neutron sentence, which points at none of
them.  An uncued item cannot be read as "the model declined to build this": it was
never asked.  We therefore flag every row CUED (its prompt is used by exactly one
target word at that context level) vs TEMPLATE, and report the headline rates on
the cued subset.  Cueing is confounded with tier and strategy -- core is 26/75
cued, floor 16/175 -- so the pooled tier/strategy tables are cue-availability
tables and are kept only for reference.

Run: local/bin/python analyze_synthetic_composition.py
"""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("super")   # expanded super-set run (Jul 10); old small-set CSVs are in results/
ANALYSIS = Path("analysis")
# model -> (detailed csv, synthetic jsonl, cutoff)
RUNS = {
    "talkie-base": (RESULTS / "composition_talkie-base_detailed.csv",
                    Path("synthetic_1930_super.jsonl"), 1930),
    "talkie-web": (RESULTS / "composition_talkie-web_detailed.csv",
                   Path("synthetic_1930_super.jsonl"), 1930),
    "typewriter": (RESULTS / "composition_typewriter_detailed.csv",
                   Path("synthetic_1913_super.jsonl"), 1913),
}
LEVELS = ["high", "medium", "low"]

# NOVELTY RE-SCREEN (2026-08-03).  The generator's novelty oracle -- a 348k-word Kaggle
# list plus a collision check against the cloze targets -- has a ~5% false-novelty rate.
# All 356 distinct synthetic targets were re-screened against English Wiktionary (batched
# API, 50 titles per call) and these 18 turned out to have an English entry, i.e. they are
# NOT never-existed coinages.  Detail: analysis/synthetic_novelty_rescreen.csv.
#
# Why this matters for the project: the whole point of the synthetic set is that recall of
# a never-written string cannot be leakage.  A target that is a real pre-cutoff English
# word breaks that guarantee outright (`faithist` 1882, `companioning` 1600s -- both freely
# available in Talkie's own corpus).  A target that is real but POST-cutoff (`hydrospace`
# 1963, `citrullinase`, `rankist`, `scintillon`) is weaker but not fatal: the model still
# cannot have memorised it from pre-1930 text, so it remains an anachronism, just not a
# manufactured one.  We drop all 18 from the headline and report them separately.
ATTESTED = {
    "caloron", "citrullinase", "companioning", "faithist", "familying", "ferrome",
    "firespace", "horson", "hydrospace", "interferase", "interferome", "medio",
    "melatoninergic", "mistone", "panni", "rankist", "scintillon", "tyraminergic",
}


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


def mark_cued(df):
    """Flag rows whose prompt belongs to them alone.

    A prompt shared by several target words at the same context level was written
    for one of them (the family template); the others are scored against a sentence
    that does not point at them.  Only a prompt used by exactly one target word is
    a genuine cue for that word.
    """
    users = df.groupby(["prefix", "context_level"])["wl"].transform("nunique")
    df["cued"] = users == 1
    return df


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
        df["attested"] = df["wl"].isin(ATTESTED)      # failed the novelty re-screen
        mark_cued(df)

        n_words = df["wl"].nunique()
        n_att = df[df["attested"]]["wl"].nunique()
        df = df[~df["attested"]].copy()               # headline excludes them
        cued_words = df[df["cued"]]["wl"].nunique()
        out += [f"## {model}  (cutoff {cutoff})",
                f"- {n_words} synthetic cousins × {len(LEVELS)} context levels = {len(df)} rows",
                f"- **{n_att} dropped** by the novelty re-screen (real English words; see "
                f"`analysis/synthetic_novelty_rescreen.csv`), leaving {n_words - n_att}",
                f"- **{cued_words} cued** (own context) / "
                f"{n_words - n_att - cued_words} template-cued (context written for another word)",
                f"- ANY-level recall@100, all surviving words: "
                f"{df.groupby('wl')['hit'].any().sum()} / {n_words - n_att}",
                ""]

        # ---- headline: cued items only -------------------------------------
        cu = df[df["cued"]]
        cw = cu.groupby("wl")["hit"].any()
        out.append("### HEADLINE — cued items only (the item's own context)")
        out.append(f"- ANY-level recall@100: **{int(cw.sum())} / {len(cw)} "
                   f"= {100*cw.mean():.1f}%**")
        out.append("")
        out.append("| stratum | words | recalled | rate |")
        out.append("| --- | --- | --- | --- |")
        for tier in ["core", "floor"]:
            g = cu[cu["tier"] == tier].groupby("wl")["hit"].any()
            if len(g):
                out.append(f"| tier={tier} | {len(g)} | {int(g.sum())} | {100*g.mean():.1f}% |")
        for lvl in LEVELS:
            s = cu[cu["context_level"] == lvl]
            if len(s):
                out.append(f"| context={lvl} | {len(s)} rows | {int(s['hit'].sum())} "
                           f"| {100*s['hit'].mean():.1f}% |")
        out.append("")
        out.append("For contrast, the template-cued items (no sentence pointing at them): "
                   f"**{int(df[~df['cued']].groupby('wl')['hit'].any().sum())} / "
                   f"{n_words - n_att - cued_words}** recalled. Their near-zero rate reflects the "
                   "missing cue, not a refusal to compose.")
        out.append("")

        # ---- reference tables (POOLED — cue-confounded, see module docstring)
        out.append("### reference: pooled tables (cue-confounded — read with care)")
        out.append("")
        out.append("#### recall@100 by tier × context level (hits / rows)")
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

        # by strategy, split by cue -- the split shows the confound directly
        out.append("#### recall@100 by strategy × cue (any-level hits / words)")
        out.append("| strategy | tier | cued words | cued recalled | template words | template recalled |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for strat, g in df.groupby("strategy"):
            tier = g["tier"].iloc[0]
            c = g[g["cued"]].groupby("wl")["hit"].any()
            t = g[~g["cued"]].groupby("wl")["hit"].any()
            out.append(f"| {strat} | {tier} | {len(c)} | {int(c.sum())} | {len(t)} | {int(t.sum())} |")
        out.append("")

        # exactly which cousins were recalled, with rank + the model's top-10
        hits = df[df["hit"]].sort_values("rank")
        out.append("### every recalled cousin")
        if len(hits) == 0:
            out.append("_none recalled at any level — pure leakage signature_\n")
        else:
            out.append("| word | seed | tier | strategy | cue | level | rank | model top-10 |")
            out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for _, r in hits.iterrows():
                top = str(r.get("top_10_words", "")).replace("|", " · ")
                out.append(f"| **{r['target_word']}** | {r['seed_word']} | {r['tier']} | "
                           f"{r['strategy']} | {'cued' if r['cued'] else 'template'} | "
                           f"{r['context_level']} | {int(r['rank'])} | {top} |")
            out.append("")

        df.to_csv(ANALYSIS / f"synthcomp_{model}_joined.csv", index=False)

    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "synthetic_composition_summary.md").write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n-> analysis/synthetic_composition_summary.md")


if __name__ == "__main__":
    main()
