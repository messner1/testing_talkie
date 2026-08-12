#!/usr/bin/env python3
"""The automatic neighborhood measure: is this top-10 a paradigm being worked?

WHAT IS BEING MEASURED.  Given a model's ten predictions for a cloze slot -- shown with no
target, no prompt and no model identity -- how large is the largest group that constitutes
genuine paradigm exploration?  The construct is the model assembling candidates out of
shared material, as against reciting a semantic field, inflecting one word, or satisfying
the slot's grammar.

WHY NOT JUST CONNECTED COMPONENTS OVER SHARED SUBSTRINGS.  That is single-linkage
clustering and it chains.  Measured failure: `formative/demonstrative/interrogative` (share
`-ative`) and `proclitic/enclitic` (share `-clitic`) merged into one group of six because
`emphatic` contains BOTH `ati` and `tic` and acts as a single articulation point.  Removing
it splits the group cleanly.  Schone & Jurafsky (2000) named this failure in morphology
induction (`ally` -> `all`); the field's usual remedy is to require semantic similarity
between the two forms (Schone & Jurafsky 2000; Baroni et al. 2002; Soricut & Och 2015).
THAT REMEDY IS UNAVAILABLE HERE: our most informative candidates are coinages -- `hepton`,
`acetylcholide`, `endocrast` -- which have no distributional vector because they do not
occur.  So the fix has to be topological.  Bernhard (2010, MorphoNet) is the prior art for
using graph structure rather than connected components to recover morphological families.

THE FOUR EDGE TYPES.  Every pair of candidates is classified from its ALIGNMENT
(difflib.SequenceMatcher.get_opcodes), not from a distance score.  A distance collapses
`proclitic/enclitic` and `jolly/awfully` to similar numbers; the alignment does not:

    proclitic  / enclitic     [pro->en] =clitic     shared block is a ROOT (6 chars)
    jolly      / awfully      [jo->awfu] =lly       shared block is a common AFFIX (3)
    radiation  / injections   [-rad] =i [a->njec] =tion [+s]   four ops, shared affix
    vaccine    / vaccination  =vaccin [e->ation]    shorter word wholly contained

  INFLECTION   one is a grammatical inflection of the other (`dose`/`doses`).  Dropped:
               inflection is semantically empty, so it is one lexeme, not exploration.
  SINGLE_STEP  the shorter word is wholly contained in the longer (`vaccine`/`vaccination`,
               `other`/`otherwise`).  Dropped, following Goldsmith (2001): Linguistica
               discards signatures containing only one suffix, because a stem seen with a
               single affix is no evidence of structure.  A pattern applied once is not a
               pattern.  Albright & Hayes (2003) make the same point formally -- a single
               pair yields a rule of scope 1; two bases are the first generalisation.
  SHARED       a shared block with material differing on BOTH sides -- two siblings off one
               base (`replacement`/`displacement`) or one affix on two roots
               (`synthase`/`kinase`).  Kept.  These are Goldsmith's two signature axes.
  NONE         no shared block of MIN_RUN or more.

AFFIX SURPRISAL, NOT AFFIX PRODUCTIVITY.  Two `-tion` words in a ten-item list is what
English prose does; two `-ase` words is not.  This is deliberately framed as SURPRISAL and
not as productivity: Gaeta & Ricca (2006) show fixed-corpus productivity measures
systematically overestimate low-frequency affixes, so "`-ase` is more productive than
`-tion`" is a claim the literature would contest.  What we mean is that the co-occurrence is
unlikely by chance.  Type frequencies are counted off the attestation wordlist -- no
hand-built affix list, no curated resource.

Run:  local/bin/python neighborhood_measure.py --demo
      local/bin/python neighborhood_measure.py --null      # chance calibration
"""

import argparse
import collections
import json
import math
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import neighborhood_analysis as NA                       # noqa: E402

MIN_RUN = 3        # shortest shared block considered. The shortest string that can be an
                   # English morpheme; the same value governs the selection-side extractor.
MIN_WORD = 3
SEED = 20260811

# RETIRED. A pair-level surprisal floor of 9.77 governed the pairwise-edge route, which the
# ablation deleted -- nothing consults this now. Kept only as a record of the calibration:
# chance PAIR surprisal over 300 pseudo-neighborhoods was median 6.63, 95th percentile 9.77.
MIN_SURPRISAL = 9.77   # unused

_AFFIX = None
_ATTEST = None


def resources():
    global _ATTEST
    if _ATTEST is None:
        _ATTEST = NA.load_resources()[3]
    return _ATTEST


def affix_freq():
    """Type frequency of every word-final and word-initial n-gram, over the attestation
    wordlist. Counted, not curated: this is what separates `-tion` from `-ase` without a
    hand-built list of "grammatical" affixes, and it is the quantity Gaeta & Ricca's
    objection permits us to use."""
    global _AFFIX
    if _AFFIX is None:
        att = resources()
        c = collections.Counter()
        n = 0
        for w in att:
            if not w.isalpha() or len(w) < 4:
                continue
            n += 1
            for k in range(2, 8):
                if len(w) > k + 1:
                    c["$" + w[-k:]] += 1
                    c["^" + w[:k]] += 1
                # Interior frequency, counted separately. Scoring a word-medial match
                # against the word-final table gives it spurious rarity: `ima` is rare
                # word-finally and common inside words, so primary/animals scored 12.2 --
                # above hepton/proton's 8.9 -- on a pure coincidence.
                for i in range(len(w) - k + 1):
                    c["~" + w[i:i + k]] += 1
        _AFFIX = (c, n)
    return _AFFIX


def surprisal(block, where):
    """-log2 P(a random attested word carries this block at this edge). Higher = rarer =
    stronger evidence. `mid` blocks get the max of the two edge readings, since an interior
    match is not an affix and we have no interior frequency table."""
    c, n = affix_freq()
    if where == "end":
        f = c.get("$" + block, 0)
    elif where == "start":
        f = c.get("^" + block, 0)
    else:
        f = c.get("~" + block, 0)
    return -math.log2(max(f, 1) / n)


POS_CACHE = Path("cache/pos_sample.json")
CAND_POS_CACHE = Path("cache/candidate_pos.json")   # every candidate word, tagged once
# Calibrated the same way as MIN_SURPRISAL, against the same null: a block licenses a group
# only if k-of-n carriers is rarer than 95% of what length-matched real English words
# produce by accident. Chance set-surprisal over 300 pseudo-neighborhoods: median 6.68,
# 90th 13.24, 95th 14.92, 99th 25.77.
SET_MIN_SURPRISAL = 14.92

_POS = None
_POSRATE = {}


def pos_table():
    """A POS-tagged sample of the lexicon, cached to disk.

    Tagging all 188k attestation words with spaCy does not finish -- it hung a shell for
    minutes before being killed. A 40k sample tags in about a minute, is written once to
    cache/pos_sample.json, and gives base rates stable to three decimals, which is far more
    precision than a k-of-10 binomial needs.
    """
    global _POS
    if _POS is None:
        if not POS_CACHE.exists():
            sys.exit(f"missing {POS_CACHE} — build it with --build-pos-cache")
        _POS = json.loads(POS_CACHE.read_text())
    return _POS


def dominant_pos(words, frac=0.6):
    """The grammatical category the slot appears to demand, or None.

    THIS IS NOT OPTIONAL AND IT IS NOT A REFINEMENT. Measured: `-on` carried by 8 of 10
    candidates and `-ly` carried by 8 of 10 have set-level surprisal 34.4 and 34.9 -- they
    are statistically indistinguishable. One is a particle series the model built; the other
    is ten adverbs because the slot demanded an adverb. Nothing in the candidate list
    separates them, because the distinguishing fact is a property of the PROMPT, which this
    instrument is blind to by design.

    Part of speech is the one escape, because it is a property of the words themselves:
    66.6% of adverbs end in `-ly` against 0.03% of nouns, so conditioning the base rate on
    the dominant category collapses `actively` from 34.9 to 1.8 while `positron` stays at
    27.9.

    Tagged from the cached sample rather than by running spaCy at query time.
    """
    # Look the candidates up in a precomputed table if one exists. Tagging live costs
    # ~70ms per neighborhood, which is 3 hours over the 151k-neighborhood corpus for only
    # 75k distinct words; tagging the vocabulary once takes 17s. Falls back to live tagging
    # for words the table does not carry.
    global _CANDPOS
    try:
        _CANDPOS
    except NameError:
        _CANDPOS = json.loads(CAND_POS_CACHE.read_text()) if CAND_POS_CACHE.exists() else {}
    if _CANDPOS:
        c = collections.Counter(_CANDPOS[w.lower()] for w in words
                                if _CANDPOS.get(w.lower()) in
                                ("NOUN", "VERB", "ADJ", "ADV", "PROPN"))
        if sum(c.values()) >= frac * len(words):
            return c.most_common(1)[0][0]
        if len([w for w in words if w.lower() in _CANDPOS]) == len(words):
            return None                      # fully covered, no dominant category
    global _NLP
    try:
        _NLP
    except NameError:
        _NLP = None
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        except Exception:                                   # noqa: BLE001
            return None
    c = collections.Counter()
    for d in _NLP.pipe([w.lower() for w in words]):
        if d and d[0].pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
            c[d[0].pos_] += 1
    if not c:
        return None
    pos, n = c.most_common(1)[0]
    return pos if n >= frac * len(words) else None


def _rates(pos):
    """P(a word of this category carries block B at each edge), from the tagged sample."""
    if pos in _POSRATE:
        return _POSRATE[pos]
    tab = pos_table()
    words = [w for w, p in tab.items() if pos is None or p == pos]
    c = collections.Counter()
    for w in words:
        for k in range(2, 9):
            if len(w) > k:
                c["$" + w[-k:]] += 1
                c["^" + w[:k]] += 1
    _POSRATE[pos] = (c, max(len(words), 1))
    return _POSRATE[pos]


def _binom_ge(k, n, p):
    if p <= 0:
        return 1e-300
    if p >= 1:
        return 1.0
    return max(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(k, n + 1)), 1e-300)


def carried_blocks(words, kmin=2, kmax=8):
    """Every edge-anchored block carried by two or more candidates -> the carriers."""
    ws = [w.lower() for w in words]
    out = collections.defaultdict(set)
    for w in ws:
        for k in range(kmin, kmax + 1):
            if len(w) > k:
                out[("$", w[-k:])].add(w)
                out[("^", w[:k])].add(w)
    return {kb: v for kb, v in out.items() if len(v) >= 2}


def set_surprisal(words, side, block, pos=None):
    """-log2 P(k or more of n words of this category carry this block by chance).

    SET level, not pair level. A paradigm can be organised by an affix too short and too
    common to be surprising in any single PAIR while being extremely surprising across the
    SET: `-on` is 2 characters at 3.2% type frequency, so every pair sharing it scores 5.0
    (below the pair floor of 9.77) and the particle series was invisible -- the measure saw
    3 of the 8 words. As a set, 8 of 10 carrying it scores 27.9.
    """
    carriers = carried_blocks(words).get((side, block), set())
    k, n = len(carriers), len(words)
    c, tot = _rates(pos)
    p = c.get(side + block, 0) / tot
    return -math.log2(_binom_ge(k, n, p)), carriers


def _inflection(a, b):
    """Is one a grammatical inflection of the other?

    Uses lemminflect's GENERATIVE direction (getAllInflections) rather than getLemma.
    getLemma over-strips on unattested forms -- `endocrinous` -> `endocrinou`, `telemer` ->
    `telem` -- which would silently delete real paradigm members. Generating from the
    shorter form and checking membership cannot invent a relation that isn't there.
    """
    try:
        from lemminflect import getAllInflections
    except ImportError:
        return False
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    if lo == hi:
        return False
    for pos in ("NOUN", "VERB", "ADJ", "ADV"):
        for forms in getAllInflections(lo, upos=pos).values():
            if hi in forms:
                return True
    # Fallback for coinages lemminflect has no paradigm for: a bare inflectional suffix.
    for suf in ("s", "es", "ed", "ing"):
        if hi == lo + suf or (lo.endswith("e") and hi == lo[:-1] + suf):
            return True
    return False


def classify(a, b, min_run=MIN_RUN, use_inflection=True, use_single_step=True):
    """Type a pair from its ALIGNMENT. Returns (kind, block, where, surprisal)."""
    if len(a) < MIN_WORD or len(b) < MIN_WORD:
        return ("NONE", "", "", 0.0)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    blocks = [(n, a[i:i + n]) for i, j, n in sm.get_matching_blocks() if n]
    if not blocks:
        return ("NONE", "", "", 0.0)
    n, block = max(blocks)
    if n < min_run:
        return ("NONE", "", "", 0.0)
    where = ("start" if a.startswith(block) and b.startswith(block) else
             "end" if a.endswith(block) and b.endswith(block) else "mid")
    s = surprisal(block, where)
    if use_inflection and _inflection(a, b):
        return ("INFLECTION", block, where, s)
    # SINGLE_STEP: one word is the other plus material, i.e. the pattern was applied once.
    # Exact containment is too strict -- `vaccine`/`vaccination` share `vaccin`, and the
    # stray `e` on the shorter form made the containment test fail. What matters is that
    # essentially NOTHING differs on one side. Requiring both differing sides to carry at
    # least two characters captures "two siblings off one base" (`[re->dis]`) and excludes
    # "one derived from the other" (`[e->ation]`, `[+e]`, `[+wise]`).
    da = len(a) - len(block)
    db = len(b) - len(block)
    if use_single_step and min(da, db) < 2:
        return ("SINGLE_STEP", block, where, s)
    return ("SHARED", block, where, s)


def form_group(words, set_min_surprisal=SET_MIN_SURPRISAL, use_pos=True,
               use_inflection=True, use_single_step=True):
    """The largest group of candidates licensed by a shared affix.

    ONE PARAMETER, and it is measured rather than chosen: a block licenses a group only if
    k-of-n carriers is rarer than 95% of what length-matched real English words produce by
    accident (SET_MIN_SURPRISAL).

    WHAT WAS REMOVED, AND WHY -- from an ablation over 31 hand-labelled neighborhoods:

      * A PAIRWISE-EDGE ROUTE (connected components over surprising shared substrings, the
        "one root, many affixes" axis). Deleting it changed NOTHING: exact agreement 0.581
        either way, identical mean error, marginally better correlation without it. Every
        group it found, the set-level route found too. Its pair-level surprisal floor went
        with it, since nothing else consulted that threshold.
      * AN ARTICULATION-POINT SPLIT, which existed to stop one word bridging two paradigms
        (`emphatic` joining `-ative` to `-clitic` through 3-character coincidences). Also
        zero change on every statistic and every case: once weak links are excluded by
        surprisal, there are no spurious bridges left to cut. Two mechanisms had been
        solving one problem.

    WHAT IS LOAD-BEARING, same ablation:

      * The surprisal floor. Removing it drops exact agreement 0.581 -> 0.387, the largest
        single effect measured.
      * POS conditioning, which is NOT a refinement. `-on` carried by 8 of 10 and `-ly`
        carried by 8 of 10 score set-surprisal 34.4 and 34.9 -- statistically identical.
        One is a particle series the model built, the other is ten adverbs because the slot
        demanded an adverb, and nothing in the candidate list separates them: the
        distinguishing fact is a property of the PROMPT, which this instrument is blind to
        by design. Part of speech is the only escape, because it is a property of the words
        themselves. Conditioning collapses `actively` to 1.8 while `positron` stays at 27.9.
        Without it the measure returns 8 for a list of ten adverbs.
      * The inflection and single-step gates. These LOWER aggregate fit (mean error 0.77
        rises from 0.65-0.68 without them, correlation falls from 0.91-0.93 to 0.87) while
        preserving the annotator's explicit rulings that `dose`/`doses` and
        `vaccine`/`vaccination` are not paradigms. Kept deliberately: the aggregate gain
        from dropping them comes from fitting labels the annotator has said are unreliable,
        against rules the annotator stated outright.
    """
    ws = [w.lower() for w in words]
    pos = dominant_pos(ws) if use_pos else None
    best = []
    for (side, blk), carriers in carried_blocks(ws).items():
        if len(carriers) <= len(best):
            continue
        # Distinct LEXEMES, not surface forms: `dose`/`doses` carry `dose` between them but
        # are one lexeme; `vaccine`/`vaccination` are a single derivational step. Neither is
        # a pattern applied twice (Goldsmith 2001 discards single-suffix signatures for the
        # same reason; Albright & Hayes 2003 make it formal -- one pair is a rule of scope 1).
        lex = []
        for w in sorted(carriers):
            if any((use_inflection and _inflection(w, u))
                   or (use_single_step and classify(w, u)[0] == "SINGLE_STEP")
                   for u in lex):
                continue
            lex.append(w)
        if len(lex) < 2:
            continue
        ss, _ = set_surprisal(ws, side, blk, pos)
        if ss >= set_min_surprisal:
            best = sorted(carriers)
    return best


# --------------------------------------------------------------------------- #
# B -- pair co-membership scoring (Morpho Challenge protocol)
# --------------------------------------------------------------------------- #
def comembership(group_a, group_b, words):
    """Precision/recall/F1 over WORD PAIRS placed in the same group.

    The Morpho Challenge evaluation measure (Kurimo, Creutz, Virpioja et al.) scores
    co-membership of sampled word pairs rather than segmentation boundaries, because two
    analyses can agree on how many morphemes a word has while disagreeing about all of
    them. The same holds here: `kind` scored 2 under both the judge (`aim`/`aims`) and the
    automatic rule (`sight`/`thoughts`) with ZERO members in common. Count agreement
    reports that as perfect; this reports it as zero.
    """
    A, B = {w.lower() for w in group_a}, {w.lower() for w in group_b}
    ws = [w.lower() for w in words]
    pa = {(x, y) for i, x in enumerate(ws) for y in ws[i + 1:] if x in A and y in A}
    pb = {(x, y) for i, x in enumerate(ws) for y in ws[i + 1:] if x in B and y in B}
    if not pa and not pb:
        return (1.0, 1.0, 1.0)          # both say "no group" -- perfect agreement
    if not pa or not pb:
        return (0.0, 0.0, 0.0)
    tp = len(pa & pb)
    p = tp / len(pb)                     # b is the prediction, a the reference
    r = tp / len(pa)
    return (p, r, 2 * p * r / (p + r) if p + r else 0.0)


# --------------------------------------------------------------------------- #
# C -- matched-random null
# --------------------------------------------------------------------------- #
_BY_LEN = None


def _lexicon_by_length():
    global _BY_LEN
    if _BY_LEN is None:
        d = collections.defaultdict(list)
        for w in resources():
            if w.isalpha() and 3 <= len(w) <= 20:
                d[len(w)].append(w)
        _BY_LEN = {k: sorted(v) for k, v in d.items()}
    return _BY_LEN


def matched_null(words, trials=200, seed=SEED, **kw):
    """Chance group size for a pseudo-neighborhood of REAL words matched on length.

    Siew & Vitevitch (2018) calibrate orthographic-network structure against pseudo-lexicons
    matched for length and letter distribution. Resampling LETTERS is wrong here and I
    measured it failing: random letter strings contain n-grams that occur nowhere in
    English, so every chance edge scores maximal surprisal (16.82) and a null-calibrated
    threshold deletes the real signal -- `hepton`/`proton` and `synthase`/`kinase` both fell
    below it.

    Sampling real English words of the same lengths preserves orthographic structure and
    asks the question we actually mean: would ten arbitrary English words of these lengths
    show as much shared structure as the ten the model produced?
    """
    rng = random.Random(seed)
    by_len = _lexicon_by_length()
    lens = [len(w) for w in words if by_len.get(len(w))]
    if len(lens) < 2:
        return []
    out = []
    for _ in range(trials):
        fake = [rng.choice(by_len[n]) for n in lens]
        out.append(len(form_group(fake, **kw)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--null", action="store_true")
    args = ap.parse_args()

    CASES = {
        "positron (coinage series)":
            "lithion metron the hepton lithon argon helium proton electron protron".split(),
        "cholinergic (scratchpad)":
            ("cholinegic acetonic choline acetylcholine acetylcholic cholic acetylenic "
             "aceto acetylcholinic acetylcholide").split(),
        "thematic (two paradigms + hinge)":
            ("as interrogative formative the pronominal enclitic proclitic a "
             "demonstrative emphatic").split(),
        "actively (slot-forced -ly)":
            ("in actually through empirically actively immediately experientially "
             "experimentally directly practically").split(),
        "vaccine (single steps only)":
            "dose shot doses vaccine shots jab covid vaccination pfizer moderna".split(),
        "proof (orthographic variants)":
            ("testimonies proofes matter proof testimony proofs proofe evidence arguments "
             "evidences").split(),
        "apoptosis (inflection only)":
            ("attacks doses therapeutic radiation injections treatments and attack x "
             "treatment").split(),
        "daze (coincidence)":
            "quite well taken jolly dead awfully smashing damned fairly extant".split(),
    }
    if args.demo:
        for name, ws in CASES.items():
            g = form_group(ws)
            print(f"\n  {name}")
            print(f"    group ({len(g)}): {' · '.join(g) if g else '(none)'}")
            kinds = collections.Counter()
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    kinds[classify(ws[i].lower(), ws[j].lower())[0]] += 1
            print(f"    edge types: {dict(kinds)}")
    if args.null:
        print(f"  {'case':<38}{'real':>6}{'chance mean':>13}{'p':>8}")
        for name, ws in CASES.items():
            real = len(form_group(ws))
            null = matched_null(ws, trials=200)
            p = sum(1 for x in null if x >= real) / len(null)
            print(f"  {name:<38}{real:>6}{sum(null)/len(null):>13.2f}{p:>8.3f}")


if __name__ == "__main__":
    main()
