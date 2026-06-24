#!/usr/bin/env python3
"""Mechanistically expand the synthetic compositionality sets.

Instead of an agent inventing each coinage, this mints new never-existed words by a
*simple, deterministic rule*: take a family's productive affix and **swap in a novel,
high-frequency stem** (or a sibling affix), then drop the result into that family's
existing [MASK] contexts. Every new word is a single substitution away from the
established sets, and is kept only if it is genuinely not English.

"High-frequency" is grounded in the Kaggle morphemic-segmentation dataset
(`thedevastator/morphemic-segmentation-of-english-words`), read via kagglehub:
  * HIFREQ_STEMS   = the most *productive* roots in lookup.csv (root productivity =
                     how many derived words are built on the root). Token frequency is
                     not cleanly available in that dataset; root productivity is the apt
                     measure for coinage anyway.
  * affix_stems    = real stems attested with each family affix (for same-domain swaps).
  * REAL           = the 348k attested words = a novelty oracle: reject any coinage that
                     turns out to be a real English word.

Affix pool is restricted to the families already present in the sets (user choice).

Output (NON-destructive; originals untouched):
  * synthetic_<cutoff>_super.jsonl              = originals + mechanistic additions
  * analysis/synth_candidates_<cutoff>_mech.json = the mechanistic candidates only

Run: local/bin/python gen_synthetic_expansion.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# reuse the existing validator / tier map (module is import-safe)
from make_synthetic_sets import real_word_vocab, validate_candidate, TIER, LEVELS

ANALYSIS = Path("analysis")
CUTOFFS = [1930, 1913]

# family -> (affix, kind). Only the morphologically tractable families; the pure
# clippings (photometric_unit, luminous_intensity_unit, clipping) have no affix to
# swap and are reported as skipped. The scalar-prefix families are handled separately.
FAMILY_AFFIX = {
    "particle_on": ("on", "suffix"),
    "quasiparticle_on": ("on", "suffix"),
    "light_energy_unit": ("on", "suffix"),
    "particle_tron": ("tron", "suffix"),
    "enzyme_ase": ("ase", "suffix"),
    "receptor_ergic": ("ergic", "suffix"),
    "domain_space": ("space", "splinter"),
    "method_ing": ("ing", "suffix"),
    "belief_ist": ("ist", "suffix"),
    "secretion_crine": ("crine", "splinter"),
    "hormone_one": ("one", "suffix"),
    "body_ome": ("ome", "suffix"),
}
# within-domain sibling affixes -> a swap here yields a same-family (core) cousin
SIBLINGS = {"on": ["tron"], "tron": ["on"], "one": ["ome"], "ome": ["one"]}

# families that are scalar/combining-form clippings -> prefix-substitution coinages
PREFIX_FAMILIES = {"econ_clipping", "prefix_clipping", "economic_scale"}
SCALAR_PREFIXES = ["macro", "multi", "micro", "mega", "ultra", "nano", "hyper"]

# affixes whose dataset segmentation we can harvest stems from
HARVEST_AFFIXES = ["on", "tron", "ase", "ergic", "ist", "ing", "one", "ome"]

PROFANITY = {"fuck", "cock", "shit", "piss", "cunt", "dick", "tit", "ass"}
STEM_STOP = {"equ"}  # truncated fragments harvested as "roots"

# per (family x strategy) caps for a balanced spread
CAP_TRANSPARENT = 12
CAP_AFFIX_SWAP = 6
CAP_CROSS = 6


def load_morphology():
    """Pull lookup.csv from the Kaggle dataset and derive the banks/oracle."""
    import kagglehub
    root = Path(kagglehub.dataset_download(
        "thedevastator/morphemic-segmentation-of-english-words"))
    lk = pd.read_csv(root / "lookup.csv").dropna()
    seg = lk["x"].astype(str)
    real = {str(w).strip().lower() for w in lk["y"]}

    # high-frequency stems = most productive clean roots (first token of segmentation)
    roots = Counter()
    for s in seg:
        toks = s.split()
        if toks and not toks[0].startswith("##"):
            r = toks[0].lower()
            if r.isalpha() and len(r) >= 3 and r not in PROFANITY:
                roots[r] += 1
    hifreq = [r for r, _ in roots.most_common(40)
              if r not in STEM_STOP][:24]

    # stems attested with each harvestable affix
    affix_stems = {}
    typefreq = {}
    for a in HARVEST_AFFIXES:
        m = seg[seg.str.contains(rf"##{a}\b", regex=True)]
        typefreq[a] = int(len(m))
        stems = []
        for s in m:
            t = s.split()
            if t and not t[0].startswith("##") and "##" not in t[0] and t[0].isalpha() \
                    and len(t[0]) >= 3 and t[0].lower() not in PROFANITY:
                stems.append(t[0].lower())
        # dedup preserving order
        seen = set()
        affix_stems[a] = [x for x in stems if not (x in seen or seen.add(x))]
    return real, hifreq, affix_stems, typefreq


def near_real(w, real):
    """True if w is one doubled/undoubled consonant away from a real word.

    Catches misspellings that slip past exact-match novelty, e.g. 'runing' ->
    'running', 'doging' -> 'dogging', 'maning' -> 'manning'.
    """
    for i in range(len(w)):
        if w[:i] + w[i] + w[i:] in real:  # double the char at i
            return True
    for i in range(len(w) - 1):
        if w[i] == w[i + 1] and w[:i] + w[i + 1:] in real:  # undouble a pair
            return True
    return False


def coin(stem, affix, prefix=False):
    """Deterministic stem+affix join with light orthographic cleanup."""
    s, a = stem.lower(), affix.lower()
    if prefix:
        w = a + s
    else:
        if a and a[0] in "aeiou" and s.endswith("e"):
            s = s[:-1]
        w = s + a
    w = re.sub(r"(.)\1\1+", r"\1\1", w)  # collapse triple+ letters
    return w


def family_templates(cands):
    """family -> (contexts dict, seed_word) from the first candidate of that family."""
    tmpl = {}
    for c in cands:
        f = c.get("family")
        if f and f not in tmpl:
            tmpl[f] = (c["contexts"], c.get("seed_word"))
    return tmpl


def make_candidate(word, seed, strategy, family, contexts):
    return {
        "word": word, "seed_word": seed, "strategy": strategy,
        "family": family, "contexts": contexts,
    }


def generate(cutoff, real, hifreq, affix_stems, templates):
    """Yield mechanistic candidates for one cutoff."""
    out = []
    minted = set()  # avoid intra-generation dupes by word

    def add(word, seed, strategy, family, contexts):
        wl = word.lower()
        if wl in real or wl in minted or not word.isalpha() or near_real(wl, real):
            return False
        minted.add(wl)
        out.append(make_candidate(word, seed, strategy, family, contexts))
        return True

    # technical stems by source family (for cross-domain grafts)
    fam_stems = {}
    for fam, (affix, _kind) in FAMILY_AFFIX.items():
        fam_stems[fam] = affix_stems.get(affix, [])

    for fam, (affix, _kind) in FAMILY_AFFIX.items():
        if fam not in templates:
            continue
        contexts, seed = templates[fam]

        # 1) transparent_stem (floor): high-frequency native English stem + affix
        n = 0
        for stem in hifreq:
            if n >= CAP_TRANSPARENT:
                break
            if add(coin(stem, affix), seed, "transparent_stem", fam, contexts):
                n += 1

        # 2) affix_swap (core): same-domain novelty.
        #    - if the affix has a sibling, graft the sibling onto this affix's real
        #      stems (e.g. fermi+on attested -> fermi+tron novel);
        #    - else recombine real same-domain stems under the same affix (oracle drops
        #      the attested ones, keeping only novel combinations).
        n = 0
        sibs = SIBLINGS.get(affix, [])
        own = affix_stems.get(affix, [])
        if sibs:
            for sib in sibs:
                for stem in own:
                    if n >= CAP_AFFIX_SWAP:
                        break
                    if add(coin(stem, sib), seed, "affix_swap", fam, contexts):
                        n += 1
        else:
            for stem in own:
                if n >= CAP_AFFIX_SWAP:
                    break
                if add(coin(stem, affix), seed, "affix_swap", fam, contexts):
                    n += 1

        # 3) cross_family (floor): a technical stem from ANOTHER family + this affix
        n = 0
        for other_fam, stems in fam_stems.items():
            if other_fam == fam or n >= CAP_CROSS:
                continue
            for stem in stems[:3]:
                if n >= CAP_CROSS:
                    break
                if add(coin(stem, affix), seed, "cross_family", fam, contexts):
                    n += 1

    # scalar-prefix families: prefix + high-frequency stem
    # (sorted for deterministic emission order — set iteration varies per process)
    for fam in sorted(PREFIX_FAMILIES):
        if fam not in templates:
            continue
        contexts, seed = templates[fam]
        n = 0
        for pref in SCALAR_PREFIXES:
            for stem in hifreq:
                if n >= CAP_TRANSPARENT:
                    break
                if add(coin(stem, pref, prefix=True), seed, "cross_family", fam, contexts):
                    n += 1
            if n >= CAP_TRANSPARENT:
                break

    return out


def main():
    real, hifreq, affix_stems, typefreq = load_morphology()
    print(f"morphology: REAL oracle {len(real)} words | "
          f"{len(hifreq)} hi-freq stems | affix type-freq {typefreq}")
    print(f"hi-freq stems: {hifreq}\n")

    cloze_vocab = real_word_vocab()
    grand = 0

    for cutoff in CUTOFFS:
        cand_path = ANALYSIS / f"synth_candidates_{cutoff}.json"
        orig_jsonl = Path(f"synthetic_{cutoff}.jsonl")
        if not cand_path.exists() or not orig_jsonl.exists():
            print(f"!! skipping {cutoff}: missing {cand_path} / {orig_jsonl}")
            continue

        cands = json.loads(cand_path.read_text())
        templates = family_templates(cands)

        mech = generate(cutoff, real, hifreq, affix_stems, templates)

        # validate (structure + cloze-vocab novelty + seed/dup), tag tier
        seen = set()
        kept, rejected = [], []
        for c in mech:
            problems = validate_candidate(c, cutoff, cloze_vocab, seen)
            if problems:
                rejected.append((c["word"], problems))
                continue
            seen.add(c["word"].lower())
            kept.append({
                "word": c["word"], "category": "synthetic_seeded", "year": None,
                "seed_word": c.get("seed_word"), "seed_cutoff": cutoff,
                "strategy": c["strategy"], "tier": TIER.get(c["strategy"], "core"),
                "family": c["family"],
                "contexts": {lvl: c["contexts"][lvl] for lvl in LEVELS},
            })

        # originals (verbatim) + new ones not already present -> superset
        orig_rows = [json.loads(l) for l in orig_jsonl.read_text().splitlines() if l.strip()]
        orig_words = {r["word"].strip().lower() for r in orig_rows}
        additions = [r for r in kept if r["word"].lower() not in orig_words]
        superset = orig_rows + additions

        # superset MUST contain every original word
        super_words = {r["word"].strip().lower() for r in superset}
        assert orig_words <= super_words, "superset lost an original word!"

        out_super = Path(f"synthetic_{cutoff}_super.jsonl")
        with out_super.open("w") as f:
            for r in superset:
                f.write(json.dumps(r) + "\n")
        (ANALYSIS / f"synth_candidates_{cutoff}_mech.json").write_text(
            json.dumps(mech, indent=2))

        grand += len(additions)
        strat = Counter(r["strategy"] for r in additions)
        tier = Counter(r["tier"] for r in additions)
        fam = Counter(r["family"] for r in additions)
        print(f"=== cutoff {cutoff} ===")
        print(f"  originals {len(orig_rows)} + new {len(additions)} -> superset "
              f"{len(superset)}  ({out_super})")
        print(f"  new strategy: {dict(strat)}")
        print(f"  new tier:     {dict(tier)}")
        print(f"  new family:   {dict(fam)}")
        if rejected:
            print(f"  rejected {len(rejected)} (sample): "
                  f"{[w for w,_ in rejected[:8]]}")
        print()

    print(f"TOTAL new coinages across both sets: {grand}")


if __name__ == "__main__":
    main()
