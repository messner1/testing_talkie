#!/usr/bin/env python3
"""Assemble + validate the synthetic compositionality sets.

Input  : analysis/synth_candidates_<cutoff>.json  (authored by a generation
         subagent — one JSON list of candidate coinages with contexts).
Output : synthetic_<cutoff>.jsonl  in tron_test_cases.jsonl format, ready for
         `run_eval.py --task composition --tron-cases synthetic_<cutoff>.jsonl`.

The script does the things a language model can't do reliably on its own:
  * structural validation — exactly one [MASK] per context level, alphabetic word;
  * collision check — the coined word must NOT be a real word that appears as a
    `target_word` anywhere in the cloze details (i.e. it must be genuinely novel
    to the evaluation vocabulary) and must not equal its own seed word.

A candidate JSON object looks like:
  {
    "word": "homoserase",
    "seed_word": "synthase",
    "strategy": "affix_swap",
    "family": "enzyme_ase",
    "contexts": {"high": "... [MASK] ...", "medium": "...", "low": "..."}
  }

Run: local/bin/python make_synthetic_sets.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
ANALYSIS = Path("analysis")
CUTOFFS = [1930, 1913]
LEVELS = ("high", "medium", "low")

# Strategy -> experimental tier. "core" = morphologically-faithful cousins of the
# real post-cutoff coinages (the leakage-vs-inference test). "floor" = transparent
# English-stem coinages that should essentially never be recalled (sanity floor,
# matching the existing tron_test_cases synthetic style).
TIER = {
    "slot_reuse": "core", "affix_swap": "core", "analogical_blend": "core",
    "cross_family": "floor", "transparent_stem": "floor",
}

DETAILS = [
    RESULTS / "cloze_talkie-base_details.csv",
    RESULTS / "cloze_talkie-web_details.csv",
    RESULTS / "cloze_typewriter_details.csv",
]


def real_word_vocab():
    """Every word that really appears as a cloze target (lowercased)."""
    vocab = set()
    for p in DETAILS:
        if p.exists():
            col = pd.read_csv(p, usecols=["target_word"])["target_word"]
            vocab |= {str(w).strip().lower() for w in col.dropna()}
    return vocab


def seed_words(cutoff):
    p = ANALYSIS / f"postcutoff_seedwords_{cutoff}.csv"
    if not p.exists():
        return set()
    return {str(w).strip().lower() for w in pd.read_csv(p)["word"].dropna()}


def validate_candidate(c, cutoff, real_vocab, seen):
    """Return list of problems (empty == clean)."""
    problems = []
    word = str(c.get("word", "")).strip()
    wl = word.lower()
    if not word:
        return ["missing word"]
    if not word.isalpha():
        problems.append(f"word {word!r} not alphabetic")
    if wl in real_vocab:
        problems.append(f"word {word!r} is a REAL cloze-vocab word (not novel)")
    if wl == str(c.get("seed_word", "")).strip().lower():
        problems.append(f"word {word!r} equals its seed word")
    if wl in seen:
        problems.append(f"duplicate word {word!r} within set")
    ctxs = c.get("contexts", {})
    for lvl in LEVELS:
        t = ctxs.get(lvl)
        if not t or not isinstance(t, str):
            problems.append(f"missing {lvl} context")
            continue
        n = t.count("[MASK]")
        if n != 1:
            problems.append(f"{lvl} context has {n} [MASK] (need exactly 1)")
        # the coined word must not be given away verbatim in its own context
        if wl in t.lower():
            problems.append(f"{lvl} context leaks the answer {word!r}")
    return problems


def main():
    real_vocab = real_word_vocab()
    print(f"real cloze-target vocab: {len(real_vocab)} words")
    grand_total = 0

    for cutoff in CUTOFFS:
        cand_path = ANALYSIS / f"synth_candidates_{cutoff}.json"
        if not cand_path.exists():
            print(f"!! {cand_path} not found — skipping cutoff {cutoff} "
                  f"(generation subagent must write it first)")
            continue

        candidates = json.loads(cand_path.read_text())
        seeds = seed_words(cutoff)

        # Final manual drops (real words / homophones caught by the review pass).
        drops_path = ANALYSIS / f"synth_final_drops_{cutoff}.json"
        manual_drops = set()
        if drops_path.exists():
            manual_drops = {str(w).strip().lower() for w in json.loads(drops_path.read_text())}

        seen = set()
        kept, rejected = [], []

        for c in candidates:
            wl = str(c.get("word", "")).strip().lower()
            if wl in manual_drops:
                rejected.append((c.get("word"), ["manual drop (real word / homophone)"]))
                continue
            problems = validate_candidate(c, cutoff, real_vocab, seen)
            if problems:
                rejected.append((c.get("word"), problems))
                continue
            seen.add(wl)
            kept.append({
                "word": c["word"].strip(),
                "category": "synthetic_seeded",
                "year": None,
                "seed_word": c.get("seed_word"),
                "seed_cutoff": cutoff,
                "strategy": c.get("strategy"),
                "tier": TIER.get(c.get("strategy"), "core"),
                "family": c.get("family"),
                "contexts": {lvl: c["contexts"][lvl] for lvl in LEVELS},
            })

        out = Path(f"synthetic_{cutoff}.jsonl")
        with out.open("w") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")

        grand_total += len(kept)
        print(f"\ncutoff {cutoff}: kept {len(kept)} / {len(candidates)} candidates "
              f"-> {out}")
        if rejected:
            print(f"  rejected {len(rejected)}:")
            for w, probs in rejected:
                print(f"    - {w}: {'; '.join(probs)}")
        # quick strategy + tier spread
        strat, tiers = {}, {}
        for r in kept:
            strat[r["strategy"]] = strat.get(r["strategy"], 0) + 1
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        print(f"  strategy spread: {strat}")
        print(f"  tier spread: {tiers}")

    print(f"\nTOTAL kept across sets: {grand_total}")
    if grand_total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
