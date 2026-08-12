#!/usr/bin/env python3
"""Build the *scaffolded* subset of the cloze corpus and its hand-labelling sheet.

MOTIVATION.  Composition needs materials; retrieval does not.  A model can only *build*
a word if the context hands it something to build from -- a stem, or another word
carrying the same affix.  Retrieval needs neither: a memorised string is reachable from
a purely associative cue.  That asymmetry is a premise of this project, not a finding
(it is why the synthetic probe set was built with high / medium / low context levels in
the first place), so we use it as a *selection criterion*: restrict attention to the
cloze items whose prompt actually supplies form material, and ask what happens there.

WHAT COUNTS AS SCAFFOLDING.  Only the text *before* the slot is fed to the model
(left-context autoregressive; see evals/cloze.py), so scaffolding means form material in
the recovered prefix.  We grade it by what it supplies, deliberately using orthographic
overlap rather than a morphological analyser:

  STEM+AFFIX   prefix contains a word sharing the target's stem AND one sharing its affix
  STEM         a word sharing the target's stem only        (hydroxyl -> hydroxide)
  AFFIX        a word sharing the target's affix only       (neutron, deuteron -> positron)
  NONE         no form material                             (the ~48k contrast set)

Whether a given pair is a "real" morphological relation is deliberately NOT presupposed.
The project's claim is about what the model's output *looks like*, not about licensed
word-formation, so the grade is a crude surface rule and the hand-labelling pass (below)
is what measures whether the rule captures anything real.

WHY THIS SUBSET IS THE INTERESTING ONE.  Post-cutoff recall@100 rises monotonically with
scaffold richness in both restricted models while their in-cutoff recall stays flat --
scaffolding does little for words the model has, and triples recall for words it should
not.  That ladder is printed by this script and is the motivation for the whole pass.

OUTPUTS
  analysis/scaffold_subset.csv        one row per (model, cloze item): grade, route,
                                      donors, moderators, rank -- the analysis table
  analysis/neighborhood_A_cohesion.csv  outcome sheet A -- candidates only, shuffled
  analysis/neighborhood_B_aptness.csv   outcome sheet B (--sheet-b, after A)
  analysis/neighborhood_KEY.csv       id -> model / grade / route / stratum / rank
  analysis/scaffold_prompt_labels.csv selection sheet -- prompt only, no model output

Three sheets, three visibilities, chosen so that no instrument can see the variable it
is meant to be evidence about. The prompt sheet validates SELECTION and hides all model
output. Sheet A measures the OUTCOME and hides the prompt, so it cannot see whether the
item was scaffolded. Sheet B needs the citation to judge aptness and therefore cannot be
blind, so it is asked only where Sheet A is negative.

Run: local/bin/python scaffold_subset.py [--quota N]
CPU-only, deterministic (fixed sample seed).
"""

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evals.cloze import extract_prefix          # noqa: E402  (reuse, do not reimplement)
import technical_composition as TC              # noqa: E402
import neighborhood_analysis as NA              # noqa: E402

RESULTS = Path("results")
ANALYSIS = Path("analysis")
csv.field_size_limit(10 ** 7)

MODELS = [("talkie-base", 1930), ("talkie-web", 1930), ("typewriter", 1913)]

# Grades, ordered by the effect actually observed rather than by any a-priori notion of
# "richness": stem material outperforms affix material (0.472 vs 0.279 for Talkie-Base
# post-cutoff), so these names are descriptive and must never be relabelled high/mid/low.
GRADES = ["STEM+AFFIX", "STEM", "AFFIX", "NONE"]
SCAFFOLDED = GRADES[:3]

STEM_CHARS = 5      # shared leading characters -> stem donor
AFFIX_CHARS = 4     # shared trailing characters -> affix donor
MIN_DONOR_LEN = 4

SEED = 20260806
PREFIX_TAIL = 320   # characters of prefix shown to the labeller (the slot end matters)


def item_id(prefix_char, *parts):
    """Content-derived, stable id.

    Sequential ids (P0001, P0002, ...) are assigned by position in a sampled list, so any
    change to the item pool silently re-points every existing hand label at a different
    citation. That happened once -- fixing the stem-detection rule grew the scaffolded
    pool and invalidated 25 of 26 labels. Hashing the content instead means an item keeps
    its id no matter how the pool is resampled, and labels stay attached to what was
    actually judged.
    """
    h = hashlib.sha1("␟".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix_char}{h[:8]}"


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
WORD_RE = re.compile(r"[A-Za-z]+")

MIN_SHARED_STEM = 4     # shortest substring that counts as a shared stem
STEM_COVERAGE = 0.6     # ...and it must be this much of the shorter word
ORTHO_MAX = 0.34        # normalized edit distance below which two forms are variants
MORPH_INFORMATIVE = 60  # a shared morpheme this common narrows nothing (-ation is 172)

_MORPH_TYPES = None


def morph_types():
    """How many corpus targets carry each morpheme -- the informativeness table."""
    global _MORPH_TYPES
    if _MORPH_TYPES is None:
        c = Counter()
        path = RESULTS / f"cloze_{MODELS[0][0]}_details.csv"
        seen = set()
        if path.exists():
            with open(path) as f:
                for r in csv.DictReader(f):
                    w = NA.norm(r.get("target_word"))
                    if w and w not in seen:
                        seen.add(w)
                        for m in segments(w):
                            c[m] += 1
        _MORPH_TYPES = c
    return _MORPH_TYPES


def longest_common_substring(a, b):
    """Length of the longest shared contiguous run. Small strings, so O(n*m) is fine."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


_SEGRES = None
LCS_PREFIXES = {"un", "in", "im", "non", "anti", "de", "dis", "pre", "post", "pro", "re",
                "sub", "super", "hyper", "hypo", "micro", "macro", "over", "under",
                "meta", "inter", "intra", "trans", "counter", "semi", "mono", "poly",
                "tele", "auto", "bio", "geo", "hydro", "electro", "photo"}


def _seg_resources():
    global _SEGRES
    if _SEGRES is None:
        _SEGRES = NA.load_resources()
    return _SEGRES


def segments(w):
    """Morphemes of a word: the cached segmentation table, else a prefix/suffix peel.

    Returns an empty set when the word is unanalyzable, which callers must treat as
    "unknown", never as "no shared morpheme".
    """
    _stops, lookup, suffixes, _attest = _seg_resources()
    if w in lookup and lookup[w]:
        return {m for m in lookup[w] if len(m) >= 3}
    out = set()
    for p in sorted(LCS_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p) + 2:
            out |= {p, w[len(p):]}
            break
    for s in suffixes:
        if len(s) >= 3 and len(w) > len(s) + 2 and w.endswith(s):
            out |= {s, w[:-len(s)]}
            break
    return out


def contained_as_morph(inner, outer):
    """Is ``inner`` a constituent of ``outer``, rather than an accidental substring?

    Bare containment is not enough: `flag` sits inside `camouflage` and `other` inside
    `mother`, neither of which is a morphological relation. Two conditions rule those out
    while keeping the real cases -- the inner word must sit at an **edge** of the outer
    (so `camouflage` fails, `flag` being an infix), and the leftover must be a plausible
    affix rather than a stray letter (so `mother` fails on a residue of `m`, while
    `unlock`, `nonsense` and `hypertext` pass on `un`, `non`, `hyper`).
    """
    if inner == outer or len(inner) < MIN_SHARED_STEM or not outer.startswith(inner) \
            and not outer.endswith(inner):
        return False
    residue = outer[len(inner):] if outer.startswith(inner) else outer[:-len(inner)]
    residue = residue.strip("-")
    if not residue:
        return False
    if residue in LCS_PREFIXES or len(residue) >= 3:
        return True
    _stops, _lookup, suffixes, _attest = _seg_resources()
    return residue in suffixes


def shares_stem(donor, target):
    """Do these share a stem, wherever it sits in the word?

    The original rule matched the first five characters, which silently misses every pair
    differing by a PREFIX -- `lock`/`unlock`, `presynaptic`/`postsynaptic`,
    `sense`/`nonsense`, `text`/`hypertext`. Those are stem relations, and treating them as
    affix-only matters: stem donors and affix-only donors turn out to be different
    recruitment routes with different behaviour.

    Deliberately orthographic, like everything else here, so it admits accidents such as
    `other`/`mother`. That is the agreed trade -- surface nearness is the criterion and
    the hand labels adjudicate.
    """
    d, t = donor.lower(), target.lower()
    if d == t or len(d) < MIN_SHARED_STEM:
        return False
    if len(t) > STEM_CHARS and d[:STEM_CHARS] == t[:STEM_CHARS]:
        return True                                    # shared prefix (original rule)
    if contained_as_morph(d, t) or contained_as_morph(t, d):
        return True                                    # containment: lock -> unlock
    # Orthographic variation must survive here, because it is the scratchpad's main move:
    # `ferrinase`/`ferratase`, `acetylergic`/`acetoergic`, `telemer`/`telomer`. A longest
    # common substring cannot see those -- it is contiguous, so one substitution in the
    # middle of a word halves it, and `telemer`/`telomer` scores 3. Worse, it is arbitrary
    # about position: `telemer`/`melemer` passes only because the substitution sits at the
    # edge. So this branch uses edit distance instead.
    #
    # Edit distance alone is not enough either -- `really`/`rally` is as near (0.17) as
    # `telemer`/`melemer` (0.14). The morpheme check is what separates them: `really`
    # analyses as real+ly, so the shared `ally` is not a constituent, while neither
    # `telemer` nor `melemer` analyses at all and their similarity is therefore variation
    # on one unanalyzable stem. The two conditions have to compose.
    ne = NA.norm_edit(d, t)
    sd, st = segments(d), segments(t)
    if not sd and not st:
        return ne <= ORTHO_MAX          # variation on a stem neither table can analyse
    shared = {m for m in (sd & st) if len(m) >= 3}
    if not shared:
        return False                    # one side analyses and excludes the shared run
    # A shared morpheme counts when it is contentful. `annotation`/`representation` share
    # `ation`, which 172 corpus targets also share and which narrows nothing; edit-near
    # pairs are admitted regardless, since those are variants rather than coincidences.
    return ne <= ORTHO_MAX or any(morph_types().get(m, 0) <= MORPH_INFORMATIVE
                                  for m in shared)


def tokenize_words(text):
    """Lowercased alphabetic tokens. Apostrophes and hyphens split, which is what we
    want: the model can only emit whole alphabetic words (evals/beam_search.py filters
    on ``.isalpha()``), so donors should be counted on the same terms."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


def find_donors(prefix, target):
    """Return (stem_donors, affix_donors) -- words in the prefix sharing form with target.

    Purely orthographic by design.  A donor is any prefix word that shares the target's
    first STEM_CHARS characters (stem donor) or its last AFFIX_CHARS (affix donor).

    **This rule alone is too coarse and must not be used on its own.**  Sharing an ending
    with a nearby word is not scaffolding: high-frequency endings occur constantly in
    natural prose, so `station` lands in `transcription`'s prefix by coincidence while
    `adrenergic` lands in `cholinergic`'s because Dale's sentence says "I suggest the
    words 'adrenergic' and ___".  Measured on Talkie-Base's post-cutoff items, the
    very-common-ending stratum recalls at 0.220 against an unscaffolded baseline of
    0.215 -- no lift whatever.  ``recruitment()`` below supplies the moderators that
    separate the two, and they are reported as strata rather than used as filters: the
    dead class is a negative control, and dropping it would invite exactly the
    cherry-picking objection it answers.
    """
    t = NA.norm(target)
    stem, affix = [], []
    if not t:
        return stem, affix
    for w in set(tokenize_words(prefix)):
        if w == t or len(w) < MIN_DONOR_LEN:
            continue
        if shares_stem(w, t):
            stem.append(w)
        if len(t) > AFFIX_CHARS and w[-AFFIX_CHARS:] == t[-AFFIX_CHARS:]:
            affix.append(w)
    return sorted(stem), sorted(affix)


def recruitment(prefix, target, donors, ending_types):
    """Is the donor actually recruited, or merely co-present?

    Two automatic components. The third -- the *construction* that recruits the donor
    into a parallel with the slot ("I suggest the words A and ___", "similarly", "as
    with", "so-called", enumeration, contrast) -- is deliberately NOT guessed at here.
    Its taxonomy is being derived from a blind prompt-level hand pass
    (`analysis/scaffold_prompt_labels.csv`) so the categories come from the citations
    rather than from a regex written in advance; the detector will be built against those
    labels and its precision reported.

    Returns (distance_to_slot_in_words, ending_type_count, rarity_bucket).
    """
    toks = tokenize_words(prefix)
    dset = set(donors)
    pos = [i for i, w in enumerate(toks) if w in dset]
    dist = (len(toks) - max(pos)) if pos else -1

    t = NA.norm(target)
    n_types = ending_types.get(t[-AFFIX_CHARS:], 0) if len(t) > AFFIX_CHARS else 0
    bucket = ("rare" if n_types <= 20 else
              "common" if n_types <= 100 else "very-common")
    return dist, n_types, bucket


def locality_bucket(dist):
    if dist < 0:
        return "none"
    return "<=5" if dist <= 5 else "6-15" if dist <= 15 else ">15"


def grade_of(stem, affix):
    if stem and affix:
        return "STEM+AFFIX"
    if stem:
        return "STEM"
    if affix:
        return "AFFIX"
    return "NONE"


# --------------------------------------------------------------------------- #
# Form-variants in the neighborhood (the compositional signature)
# --------------------------------------------------------------------------- #
def form_variants(target, neighbors, prefix_words, morph, stops, attest):
    """Top-10 neighbours that are form-variants of the target, EXCLUDING prompt echoes.

    Relatedness is the dictionary-free STEM-SIBLING / AFFIX-SIBLING union from
    neighborhood_analysis.classify_neighbor, so this measure is unaffected by the
    OCR-noise problem that invalidated the "made-up" flag on real cloze data.

    **Echoes are recorded, not discarded.** A form-variant that also appears in the
    prefix is ambiguous, and neither treatment of it is neutral:

      - Counting it inflates the scaffolded strata, because a richer scaffold supplies
        more words available to be copied, and language models copy.
      - Excluding it deflates them, because the form donor is a prefix word **by
        construction** of the grade -- so exclusion strips the single most likely
        form-variant out of exactly the stratum under test.

    Nor does presence-in-prompt settle the linguistic question. On `rotor` (donor
    `motor`) Talkie-Base ranks `rotor` 2nd and `motor` 3rd: it is holding both members of
    a one-character minimal pair at the top of its distribution, which is paradigm
    behaviour, not copying -- a copier would surface the donor and miss the target.

    So both counts are carried downstream and the analysis reports the result under each.
    The residual false positive that neither count catches -- a semantically apt
    competitor sharing a generic Latinate affix, `necrosis` -> `hypnosis` -- is not
    resolvable by any surface rule and is what the blind hand-labelling pass adjudicates.
    """
    attempts, echoes = [], []
    t = NA.norm(target)
    for nb in neighbors:
        n = NA.norm(nb)
        if not n or n == t:
            continue
        lab, _nov = NA.classify_neighbor(target, nb, morph, stops, attest)
        if lab not in ("STEM-SIBLING", "AFFIX-SIBLING"):
            continue
        (echoes if n in prefix_words else attempts).append(n)
    return attempts, echoes


# --------------------------------------------------------------------------- #
# Building the table
# --------------------------------------------------------------------------- #
def build(models=MODELS):
    """One row per (model, cloze item).  Grade/domain are model-independent; is_future
    and rank are not (Typewriter's cutoff is 1913)."""
    brown = TC.load_brown()
    stops, lookup, suffixes, attest = NA.load_resources()
    morph = NA.Morph(lookup, suffixes)

    # How informative is a shared ending? Type frequency over the corpus's own targets:
    # -tion covers 440 distinct target words, -ergic a handful.
    ending_types = Counter()
    first = RESULTS / f"cloze_{models[0][0]}_details.csv"
    with open(first) as f:
        for r in csv.DictReader(f):
            w = NA.norm(r.get("target_word"))
            if len(w) > AFFIX_CHARS:
                ending_types[w[-AFFIX_CHARS:]] += 1

    graded_cache = {}       # (text, target) -> grading tuple -- shared across models
    domain_cache = {}
    rows = []

    for model, cutoff in models:
        path = RESULTS / f"cloze_{model}_details.csv"
        if not path.exists():
            print(f"  ! missing {path}, skipping {model}")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                text = r.get("text") or ""
                target = NA.norm(r.get("target_word"))
                if not target:
                    continue

                key = (text, target)
                if key not in graded_cache:
                    prefix = extract_prefix(text, target)
                    stem, affix = find_donors(prefix, target)
                    dist, n_types, rarity = recruitment(prefix, target, stem + affix,
                                                        ending_types)
                    graded_cache[key] = (grade_of(stem, affix), stem, affix, prefix,
                                         frozenset(tokenize_words(prefix)),
                                         dist, n_types, rarity)
                (grade, stem, affix, prefix, prefix_words,
                 dist, n_types, rarity) = graded_cache[key]

                if target not in domain_cache:
                    domain_cache[target] = ("technical" if TC.is_technical(target, brown)
                                            else "everyday")

                rank = NA._int(r.get("rank"))
                neighbors = NA.parse_neighbors(r)
                attempts, echoes = form_variants(target, neighbors, prefix_words,
                                                 morph, stops, attest)
                rows.append({
                    # stable join key: identical to the hand sheets' ids, so labels can be
                    # joined to outcomes exactly rather than by target_word (targets repeat
                    # across cloze items, so a name-based join silently mismatches)
                    "item_id": item_id("P", target, prefix),
                    "nbr_id": item_id("S", model, target, prefix),
                    "n_attempts": len(attempts),
                    "attempts": "|".join(attempts),
                    "n_echoes": len(echoes),
                    "echoes": "|".join(echoes),
                    # two readings, both carried: strict excludes prefix words, any counts them
                    "comp_strict": int(len(attempts) >= 1),
                    "comp_any": int(len(attempts) + len(echoes) >= 1),
                    "model": model,
                    "cutoff": cutoff,
                    "target_word": target,
                    "year": NA._int(r.get("year")),
                    "entry_start_year": NA._int(r.get("entry_start_year")),
                    "is_future": int(NA._int(r.get("is_future")) or 0),
                    "grade": grade,
                    "domain": domain_cache[target],
                    "stem_donors": "|".join(stem),
                    "affix_donors": "|".join(affix),
                    # recruitment moderators -- strata, never filters
                    "donor_distance": dist,
                    "locality": locality_bucket(dist),
                    "ending_types": n_types,
                    # Ending rarity is only meaningful when the shared material IS the
                    # ending. With a stem donor the ending is incidental -- `quantizing`
                    # -> `quantization` is a stem relation that merely happens to end in
                    # -tion -- so scoring it by ending frequency mixes two populations.
                    "ending_rarity": (rarity if grade != "NONE" and not stem else "n/a"),
                    "route": ("paradigmatic" if stem else
                              "syntagmatic" if affix else "none"),
                    "rank": rank if rank is not None else -1,
                    "hit": int(rank is not None and 0 < rank <= 100),
                    "prefix": prefix,
                    "top_10_words": r.get("top_10_words") or "",
                })
    return rows


# --------------------------------------------------------------------------- #
# The ladder (this script's verification target)
# --------------------------------------------------------------------------- #
def print_ladder(rows):
    """Recall@100 by scaffold grade x stratum x model.

    Reproduces the motivating table.  If Talkie-Base post-cutoff is not
    0.215 / 0.279 / 0.472 / 0.750 for NONE / AFFIX / STEM / STEM+AFFIX, the grading
    rule has drifted and every downstream number is suspect.
    """
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[(r["model"], r["grade"], r["is_future"])][0] += 1
        agg[(r["model"], r["grade"], r["is_future"])][1] += r["hit"]

    models = sorted({r["model"] for r in rows})
    for stratum, isf in (("post-cutoff", 1), ("in-cutoff", 0)):
        print(f"\n  recall@100, {stratum}")
        print(f"    {'grade':<12} " + "  ".join(f"{m:>22}" for m in models))
        for g in reversed(GRADES):      # NONE first, richest last
            cells = []
            for m in models:
                n, h = agg[(m, g, isf)]
                cells.append(f"{h}/{n} = {h/n:.3f}" if n else "-")
            print(f"    {g:<12} " + "  ".join(f"{c:>22}" for c in cells))

    print("\n  scaffolded items by grade x domain (model-independent):")
    seen, dom = set(), Counter()
    for r in rows:
        k = (r["target_word"], r["prefix"])
        if k in seen:
            continue
        seen.add(k)
        dom[(r["grade"], r["domain"])] += 1
    for g in GRADES:
        print(f"    {g:<12} technical={dom[(g,'technical')]:<6} everyday={dom[(g,'everyday')]}")


def print_moderators(rows):
    """Is it scaffolding, or just a common ending sitting nearby?

    The very-common-ending stratum is the built-in negative control: if merely sharing an
    ending with a prefix word produced the effect, it would show up here.
    """
    base = MODELS[0][0]
    sub = [r for r in rows if r["model"] == base and r["is_future"] and r["grade"] != "NONE"]
    none = [r for r in rows if r["model"] == base and r["is_future"] and r["grade"] == "NONE"]
    def line(label, lv, s):
        if s:
            print(f"    {label:<18} {lv:<14} {len(s):>5} "
                  f"{sum(r['hit'] for r in s)/len(s):>11.3f} "
                  f"{sum(r['comp_strict'] for r in s)/len(s):>10.3f}")

    print(f"\n  recruitment — {base}, post-cutoff scaffolded (N={len(sub)})")
    print(f"    {'moderator':<18} {'level':<14} {'N':>5} {'recall@100':>11} {'form-var':>10}")
    for lv in ("paradigmatic", "syntagmatic"):
        line("route", lv, [r for r in sub if r["route"] == lv])
    # ending rarity is scored ONLY on the syntagmatic route, where the shared material is
    # in fact the ending; on the paradigmatic route it is incidental to the stem relation
    syn = [r for r in sub if r["route"] == "syntagmatic"]
    for lv in ("rare", "common", "very-common"):
        line("  ending rarity", lv, [r for r in syn if r["ending_rarity"] == lv])
    for lv in ("<=5", "6-15", ">15"):
        line("locality", lv, [r for r in sub if r["locality"] == lv])
    line("(unscaffolded)", "baseline", none)


# --------------------------------------------------------------------------- #
# The blind labelling sheets
# --------------------------------------------------------------------------- #
def sample_sheet(rows, quota):
    """Stratified sample of *items*, emitting every model's neighborhood for each.

    Sampling items rather than model-rows makes every comparison within-item: the same
    prefix and the same target, three top-10 lists.  That is what lets the analysis ask
    whether `cholinergic` looks compositional under Talkie-Base and retrieved under
    Talkie-Web without a between-items confound.

    Strata are grade x is_future x domain -- all model-independent, since the grading
    rule reads only text and target.  ``is_future`` is model-dependent (Typewriter's
    cutoff is 1913), so it is taken from the 1930 models and Typewriter rows inherit the
    item; the KEY file carries each model's own is_future for analysis.

    NONE is included as a contrast so the labeller has unscaffolded neighborhoods to
    calibrate against and cannot infer the stratum from what a neighborhood looks like.
    """
    by_item = defaultdict(dict)
    strata = {}
    for r in rows:
        if not r["top_10_words"]:
            continue
        item = (r["target_word"], r["prefix"])
        by_item[item][r["model"]] = r
        if r["model"] == "talkie-base":
            strata[item] = (r["grade"], r["is_future"], r["domain"])

    cells = defaultdict(list)
    for item, key in strata.items():
        cells[key].append(item)

    rng = random.Random(SEED)
    picked, shortfall = [], []
    for key in sorted(cells):
        pool = sorted(cells[key])
        take = min(quota, len(pool))
        if len(pool) < quota:
            shortfall.append((key, len(pool)))
        for item in rng.sample(pool, take):
            picked.extend(by_item[item][m] for m, _ in MODELS if m in by_item[item])

    rng.shuffle(picked)     # blind: a target's three model rows must not be adjacent
    return picked, shortfall


def write_sheet(picked):
    """Two neighborhood sheets with deliberately different visibility.

    The previous single sheet showed the annotator the prefix alongside the predictions,
    which let them see the donor and infer whether the item was scaffolded before judging
    its neighborhood -- importing the selection variable into the outcome measure. It also
    asked the wrong question: whether the predictions resemble *the target*, when the
    construct of interest is whether they resemble *each other*. A model enumerating a
    paradigm and missing the target is the informative case, and target-resemblance scores
    it as a null.

    SHEET A -- internal cohesion, on TWO independent axes. Shows the ten predicted words
    and nothing else: no target, no citation, no model, and in randomized order so
    predicted rank leaks nothing. Blind to scaffolding by construction, which is what lets
    it serve as an outcome measure for a scaffold-selected population.

    The two axes are FORM and MEANING, asked separately and COUNTED: how many of the ten
    belong to the largest group sharing shape, and how many to the largest group sharing
    sense. Separately, because a single "variants on a theme?" binary cannot separate a
    scratchpad from a semantic field -- both are variants on a theme -- and that separation
    is the entire outcome measure. Counted, because mixtures are the norm: most
    neighborhoods carry a formal group and a semantic group at once, of different sizes,
    and a binary scores four variants identically to ten. Counting moves every threshold
    into the analysis, where it can be reported as a curve instead of asserted.

    Three rules make the count well-behaved. A group needs two members, so a count is 0 or
    2..10 and never 1 -- which makes "no formal cohesion" mean exactly form_n == 0. Count
    the largest single group, not the union; the two counts may overlap and sum past ten.
    And shared material the slot's grammar forces on every candidate (all 3sg, all plural)
    is not form cohesion -- every candidate would carry it however the model got there, so
    it says nothing about composition versus recall.

    A fourth rule governs the meaning axis: a candidate whose sense is only recoverable by
    parsing it does not join the meaning group. `acetylcholide` reads as chemistry, but the
    only route to that reading is its morphology, so counting it restates the form judgment
    rather than adding independent evidence. Without this the flagship scratchpad scores
    form 10 / meaning 10 and the two axes stop separating. The test is deliberately NOT
    attestation -- that asks a blind annotator for a judgment this project has twice found
    unreliable -- only whether the sense can be stated without reading the parts.

    SHEET B -- aptness for the specific slot. Shows the citation and target. Sheet A
    already says whether the candidates cohere semantically at all; B asks the narrower
    question of whether that coherence fits THIS gap, which cannot be judged blind. Needed
    only where Sheet A found no formal cohesion. Generate with --sheet-b after A.
    """
    ANALYSIS.mkdir(exist_ok=True)
    sheet = ANALYSIS / "neighborhood_A_cohesion.csv"
    key = ANALYSIS / "neighborhood_KEY.csv"
    rng = random.Random(SEED + 2)

    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "candidates", "form_n", "meaning_n", "theme", "notes"])
        for r in picked:
            words = [x for x in (r["top_10_words"] or "").split("|") if x]
            rng.shuffle(words)          # predicted rank must not leak
            w.writerow([item_id("S", r["model"], r["target_word"], r["prefix"]),
                        " · ".join(words), "", "", "", ""])

    with open(key, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "grade", "route", "is_future",
                                          "domain", "target_word", "year", "rank", "hit",
                                          "stem_donors", "affix_donors", "locality",
                                          "ending_rarity"])
        w.writeheader()
        for r in picked:
            w.writerow({"id": item_id("S", r["model"], r["target_word"], r["prefix"]),
                        **{k: r[k] for k in
                        ["model", "grade", "route", "is_future", "domain", "target_word",
                         "year", "rank", "hit", "stem_donors", "affix_donors",
                         "locality", "ending_rarity"]}})
    return sheet, key


def write_sheet_b(picked):
    """Sheet B, restricted to the items Sheet A judged to have NO formal cohesion.

    Asked only where form cohesion is absent, because that is where retrieval is the live
    hypothesis: a neighborhood the model did not assemble out of shared material is
    evidence of recall only if something apt for this slot is there instead. Where the
    candidates do cohere formally the neighborhood is already the compositional signature
    and aptness does not decide anything.

    Note this routes on the FORM COUNT, not on the old single `cohesive` field. Routing on
    general cohesion sent semantically-cohesive neighborhoods -- the clean retrieval
    signature -- down the "already compositional" branch, so they never reached this sheet
    at all. `form_n == 0` is exact rather than a threshold, because a group needs two
    members: the annotator cannot record 1, so zero means no formal group existed.
    """
    # Sheet A is annotated through label_neighborhoods.py, which appends to a JSONL log
    # (last record per id wins). Read that rather than the CSV's blank column.
    import json
    log = ANALYSIS / "hand_labeling" / "neighborhood_A.jsonl"
    verdicts = {}
    if log.exists():
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                verdicts[rec["id"]] = rec.get("form")
    # Fall back to a hand-filled CSV column, for anyone editing the sheet directly.
    a_path = ANALYSIS / "neighborhood_A_cohesion.csv"
    if a_path.exists():
        with open(a_path) as f:
            for r in csv.DictReader(f):
                v = (r.get("form_n") or "").strip()
                if v.isdigit() and r["id"] not in verdicts:
                    verdicts[r["id"]] = int(v)
    # form_n == 0 only: there, aptness decides between retrieval and no route at all.
    # `is not None` guards the unsure marker, which is not the same as "no group".
    negative = {k for k, v in verdicts.items() if v is not None and int(v) == 0}
    if not negative:
        return None, 0

    out = ANALYSIS / "neighborhood_B_aptness.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "target_word", "prefix_tail", "candidates",
                    "semantic_apt", "notes"])
        for r in picked:
            rid = item_id("S", r["model"], r["target_word"], r["prefix"])
            if rid not in negative:
                continue
            tail = r["prefix"][-PREFIX_TAIL:]
            if len(r["prefix"]) > PREFIX_TAIL:
                tail = "..." + tail
            w.writerow([rid, r["target_word"], tail, r["top_10_words"], "", ""])
    return out, len(negative)


def write_prompt_sheet(rows, n_controls=45):
    """Blind PROMPT-level sheet: does the prompt actually set up the target's form?

    This validates **scaffold detection**, which the neighborhood sheet does not — that
    one labels what the model emitted, a different object. The distinction matters
    because the orthographic rule cannot tell a coincidental shared ending from a
    construction that recruits the donor, and the coincidental class turns out to have
    no effect at all.

    Blind in a stronger sense than the neighborhood sheet: **no model output appears at
    all** — no predictions, no rank, no model name. The labeller sees a citation and a
    target and judges the prompt on its own terms. If the model's answer were visible the
    judgment could not validate anything, since the answer is what we are trying to
    predict.

    ``construction`` is free text on purpose. The taxonomy of recruiting frames is meant
    to come out of the citations rather than out of a regex written in advance; the
    detector gets built against these labels afterwards and its precision reported.
    """
    items, seen = [], set()
    for r in rows:
        if r["model"] != MODELS[0][0]:      # prompts are model-independent
            continue
        k = (r["target_word"], r["prefix"])
        if k in seen:
            continue
        seen.add(k)
        if r["is_future"] and r["grade"] != "NONE":
            items.append((r, "scaffolded-post"))
        elif r["is_future"]:
            items.append((r, "control-post-unscaffolded"))

    focal = [x for x in items if x[1] == "scaffolded-post"]
    pool = [x for x in items if x[1] != "scaffolded-post"]
    rng = random.Random(SEED + 1)
    picked = focal + rng.sample(pool, min(n_controls, len(pool)))
    rng.shuffle(picked)     # controls must not be identifiable by position

    ANALYSIS.mkdir(exist_ok=True)
    sheet, key = ANALYSIS / "scaffold_prompt_labels.csv", ANALYSIS / "scaffold_prompt_labels_KEY.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "target_word", "prefix_tail",
                    "sets_up_form", "construction", "donor_near_slot", "notes"])
        for r, _kind in picked:
            tail = r["prefix"][-PREFIX_TAIL:]
            if len(r["prefix"]) > PREFIX_TAIL:
                tail = "..." + tail
            w.writerow([item_id("P", r["target_word"], r["prefix"]),
                        r["target_word"], tail, "", "", "", ""])

    with open(key, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "kind", "target_word", "year", "grade",
                                          "domain", "stem_donors", "affix_donors",
                                          "donor_distance", "locality", "ending_types",
                                          "ending_rarity"])
        w.writeheader()
        for r, kind in picked:
            w.writerow({"id": item_id("P", r["target_word"], r["prefix"]), "kind": kind,
                        **{c: r[c] for c in ["target_word", "year", "grade", "domain",
                                             "stem_donors", "affix_donors",
                                             "donor_distance", "locality",
                                             "ending_types", "ending_rarity"]}})
    return sheet, key, len(focal), len(picked) - len(focal)


def write_subset(rows):
    ANALYSIS.mkdir(exist_ok=True)
    out = ANALYSIS / "scaffold_subset.csv"
    cols = ["item_id", "nbr_id",
            "model", "cutoff", "target_word", "year", "entry_start_year", "is_future",
            "grade", "domain", "stem_donors", "affix_donors",
            "donor_distance", "locality", "ending_types", "ending_rarity", "route",
            "rank", "hit",
            "comp_strict", "comp_any", "n_attempts", "attempts", "n_echoes", "echoes",
            "top_10_words"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", type=int, default=9,
                    help="max ITEMS sampled per grade x stratum x domain cell "
                         "(each contributes one row per model)")
    ap.add_argument("--dump", type=int, default=0,
                    help="print N scaffolded post-cutoff neighborhoods and exit")
    ap.add_argument("--sheet-b", action="store_true",
                    help="generate neighborhood Sheet B for items Sheet A found formally uncohesive")
    args = ap.parse_args()

    print("Building scaffolded subset (this reads three 50k-row CSVs)...")
    rows = build()
    print(f"  {len(rows)} model-item rows over {len({r['target_word'] for r in rows})} words")

    if args.dump:
        shown = 0
        for r in rows:
            if r["grade"] == "NONE" or not r["is_future"]:
                continue
            print(f"\n  [{r['model']}] {r['target_word']} ({r['year']}) grade={r['grade']} "
                  f"domain={r['domain']} rank={r['rank']} "
                  f"-> strict={'Y' if r['comp_strict'] else 'n'} any={'Y' if r['comp_any'] else 'n'}")
            print(f"    donors   : stem={r['stem_donors'] or '-'} affix={r['affix_donors'] or '-'}")
            print(f"    attempts : {r['attempts'] or '(none)'}   <- form-variants not in prefix")
            print(f"    echoes   : {r['echoes'] or '(none)'}   <- form-variants also in prefix")
            print(f"    prefix   : ...{r['prefix'][-160:]}")
            print(f"    top10    : {r['top_10_words']}")
            shown += 1
            if shown >= args.dump:
                break
        return

    print_ladder(rows)

    out = write_subset(rows)
    print(f"\nwrote {out}")

    print_moderators(rows)

    picked, shortfall = sample_sheet(rows, args.quota)
    if args.sheet_b:
        out, n = write_sheet_b(picked)
        if out is None:
            print("\nSheet A is unlabelled (or every item had a formal group) — nothing to do.")
        else:
            print(f"\nwrote {out} ({n} items with form_n = 0)")
        return
    sheet, key = write_sheet(picked)
    print(f"\nwrote {sheet} ({len(picked)} neighborhoods, candidates only, order shuffled)")
    print(f"wrote {key}")

    psheet, pkey, n_focal, n_ctrl = write_prompt_sheet(rows)
    print(f"wrote {psheet} ({n_focal} scaffolded + {n_ctrl} control prompts, "
          f"no model output shown)")
    print(f"wrote {pkey}")
    if shortfall:
        print(f"  {len(shortfall)} item-strata below quota (sampled exhaustively):")
        for (g, isf, dom), n in shortfall:
            print(f"    {g:<11} {'post' if isf else 'in':<5} {dom:<10} n={n}")


if __name__ == "__main__":
    main()
