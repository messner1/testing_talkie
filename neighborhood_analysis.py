#!/usr/bin/env python3
"""Interpretable top-10 neighborhood analysis: the compositional near-miss.

Every cloze slot (hit or miss) carries the model's top-10 predicted WORDS
(`top_10_words`, pipe-delimited, whole words -- the backend returns distinct
word completions, not token fragments). This script reads those neighborhoods
and, using only transparent lookup tables + a string metric, labels each of a
target's 10 neighbors and asks whether the *composition machinery fired* even
when the exact target missed -- the compositional near-miss.

Tools (all interpretable, all on disk, CPU-only):
  1. stopword list          -> FILLER neighbors                (nltk 179 U sklearn 318)
  2. morpheme segmentation  -> STEM- / AFFIX-sibling detection (Kaggle lookup + suffixes)
  3. normalized edit dist   -> stem-sibling fallback/backstop  (stdlib Levenshtein)
  4. attestation wordlist   -> coined vs attested siblings     (dict U vocab U wordnet U seeds)

Neighbor taxonomy (per neighbor n vs target T), most-precise-first:
  FILLER          n is a stopword
  STEM-SIBLING    n,T share stem morpheme (lookup -> Porter -> edit<=THETA); n!=T
  AFFIX-SIBLING   n,T share the same productive affix, different stems (paradigm-mate)
  DISTANT-CONTENT non-filler, no shared morpheme (the car/automobile shape, by exclusion)
Every stem/affix sibling also gets a NOVELTY flag: coined (absent from the attestation
set) vs attested.  A *coined* sibling can only be built -> the composition tell.

Outcome class per item:
  hit                    0 < rank <= 100  (synthetic: rank != -1)
  compositional-near-miss  miss with >=1 COINED sibling
  retrieval-near-miss      miss with siblings, all attested
  blank-miss               miss, no sibling

TWO SCOPE RESTRICTIONS, both established by audit (see NEIGHBORHOOD_simple.md):

  1. SYNTHETIC: only CUED items count.  In the super sets most words inherit a
     family-template context written for a *different* word, so their top-10 is
     not a response to them at all; worse, items sharing a prompt share a neighbor
     list verbatim (750 rows -> 156 distinct prompts).  Views over synthetic data
     are restricted to items whose prompt is used by exactly one target word.

  2. REAL: the COINED/made-up flag is NOT valid on the OED cloze data.  Auditing
     it shows it fires on scanning noise and early-modern orthography -- `caſuall`,
     `ſtew`, `lengtli`, `blusb`, `femaie`, `commones`, `wordes`, `martiall` -- which
     Talkie-Base emits far more than Talkie-Web (4945 vs 874 in-cutoff) purely
     because it was trained on scanned historical text.  That is a corpus fact, not
     composition.  Real-data views therefore report relatedness only.  The flag IS
     valid on the synthetic sets, where the same audit returns genuine coinages
     (`ornithinase`, `ferrinase`, `juniorist`) and no scan noise.

Run:
  local/bin/python neighborhood_analysis.py            # analyses + write csv
  local/bin/python neighborhood_analysis.py --dump 20  # audit 20 neighborhoods
  local/bin/python neighborhood_analysis.py --selftest # sanity-check the classifier
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

RESULTS = Path("results")
ANALYSIS = Path("analysis")
MORPH_DIR = Path(os.path.expanduser(
    "~/.cache/kagglehub/datasets/thedevastator/morphemic-segmentation-of-english-words/versions/2"))

# model -> (synthetic joined csv, real cloze csv, cutoff, kind)
MODELS = [
    ("talkie-base", "synthcomp_talkie-base_joined.csv", "cloze_talkie-base_details.csv", 1930, "restricted"),
    ("talkie-web", "synthcomp_talkie-web_joined.csv", "cloze_talkie-web_details.csv", 1930, "leakage-rich"),
    ("typewriter", "synthcomp_typewriter_joined.csv", "cloze_typewriter_details.csv", 1913, "restricted"),
]

# Synthetic targets that FAILED the 2026-08-03 novelty re-screen: they have an English
# Wiktionary entry, so they are real words, not coinages.  Folded into the attestation set
# so they are never counted as "made-up" neighbours.  See
# analysis/synthetic_novelty_rescreen.{csv,md}.
RESCREEN_ATTESTED = {
    "caloron", "citrullinase", "companioning", "faithist", "familying", "ferrome",
    "firespace", "horson", "hydrospace", "interferase", "interferome", "medio",
    "melatoninergic", "mistone", "panni", "rankist", "scintillon", "tyraminergic",
}

THETA = 0.34        # normalized edit-distance threshold for stem-sibling backstop
MIN_AFFIX = 3       # min shared-suffix length for the generic affix-sibling path

csv.field_size_limit(10 ** 7)


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
_RESOURCES = None


def load_resources():
    """Memoised. Reads four files plus WordNet and costs ~1.2s, so a caller that invokes
    it per item turns a seconds-long job into an hours-long one -- which happened. The
    returned structures are shared and must be treated as read-only.
    """
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES
    from nltk.corpus import stopwords, wordnet as wn
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK

    stops = set(w.lower() for w in stopwords.words("english")) | set(SK)

    # morpheme segmentation: word -> [morphemes], stripping WordPiece '##'
    lookup = {}
    with open(MORPH_DIR / "lookup.csv") as f:
        for r in csv.DictReader(f):
            pieces = [p.replace("##", "").strip() for p in r["x"].split()]
            lookup[r["y"].strip().lower()] = [p for p in pieces if p]

    # productive suffixes: ASCII-latin, length >= MIN_AFFIX
    suffixes = set()
    with open(MORPH_DIR / "suffixes.csv") as f:
        for r in csv.DictReader(f):
            s = r["x"].strip().lower()
            if s.isascii() and s.isalpha() and len(s) >= MIN_AFFIX:
                suffixes.add(s)

    # attestation set: a sibling absent from this is "coined"
    attest = set()
    dictf = Path("/usr/share/dict/words")
    if dictf.exists():
        attest |= set(l.strip().lower() for l in open(dictf) if l.strip().isalpha())
    with open(MORPH_DIR / "vocabulary.csv") as f:
        attest |= set(r["x"].strip().lower() for r in csv.DictReader(f) if r["x"].strip().isalpha())
    attest |= set(w.lower() for w in wn.all_lemma_names())
    attest |= RESCREEN_ATTESTED

    _RESOURCES = (stops, lookup, sorted(suffixes, key=len, reverse=True), attest)
    return _RESOURCES


# --------------------------------------------------------------------------- #
# String / morphology helpers
# --------------------------------------------------------------------------- #
def norm(w):
    return (w or "").strip().lower()


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_edit(a, b):
    m = max(len(a), len(b))
    return levenshtein(a, b) / m if m else 0.0


class Morph:
    """Morpheme access with a lookup table + a suffix-list fallback."""

    def __init__(self, lookup, suffixes):
        self.lookup = lookup
        self.suffixes = suffixes            # sorted longest-first
        self._porter = None

    def porter(self, w):
        if self._porter is None:
            from nltk.stem import PorterStemmer
            self._porter = PorterStemmer()
        return self._porter.stem(w)

    @lru_cache(maxsize=100_000)
    def suffix(self, w):
        """Final productive morpheme: lookup last piece, else longest matching suffix."""
        if w in self.lookup and len(self.lookup[w]) >= 2:
            last = self.lookup[w][-1]
            if len(last) >= MIN_AFFIX:
                return last
        for s in self.suffixes:
            if len(w) > len(s) + 1 and w.endswith(s):
                return s
        return ""

    @lru_cache(maxsize=100_000)
    def stem(self, w):
        """Root morpheme: lookup first piece, else Porter stem."""
        if w in self.lookup and self.lookup[w]:
            return self.lookup[w][0]
        return self.porter(w)


FAMILY_AFFIX = {  # synthetic family -> the affix/compound-element it is built on
    "particle_on": "on", "enzyme_ase": "ase", "belief_ist": "ist",
    "method_ing": "ing", "receptor_ergic": "ergic", "domain_space": "space",
}


def family_affix(family):
    if not family:
        return ""
    if family in FAMILY_AFFIX:
        return FAMILY_AFFIX[family]
    return family.rsplit("_", 1)[-1]   # e.g. "*_ist" -> "ist"


# --------------------------------------------------------------------------- #
# Neighbor classification
# --------------------------------------------------------------------------- #
def classify_neighbor(target, neighbor, morph, stops, attest, key_affix=""):
    """Return (label, novelty) where label in
    {FILLER, STEM-SIBLING, AFFIX-SIBLING, DISTANT-CONTENT} and novelty in
    {'', 'attested', 'coined'} (empty for non-siblings)."""
    t, n = norm(target), norm(neighbor)
    if not n or n == t:
        return ("SELF" if n == t else "EMPTY"), ""
    if n in stops:
        return "FILLER", ""

    st, sn = morph.stem(t), morph.stem(n)
    shares_stem = (st and st == sn) or norm_edit(t, n) <= THETA

    # affix: prefer the known synthetic family affix (trusted at any length); else the
    # target's own suffix via the generic path (guarded by MIN_AFFIX to avoid -s/-er noise)
    aff = key_affix or morph.suffix(t)
    min_len = 2 if key_affix else MIN_AFFIX
    shares_affix = bool(aff) and len(aff) >= min_len and n.endswith(aff) and sn != st

    if shares_stem:
        label = "STEM-SIBLING"
    elif shares_affix:
        label = "AFFIX-SIBLING"
    else:
        return "DISTANT-CONTENT", ""

    novelty = "attested" if n in attest else "coined"
    return label, novelty


def profile(target, neighbors, morph, stops, attest, key_affix=""):
    """Per-item neighborhood profile from the 10 (whole-word) neighbors."""
    labels = []
    counts = defaultdict(int)
    n_coined = n_attested = 0
    for nb in neighbors:
        lab, nov = classify_neighbor(target, nb, morph, stops, attest, key_affix)
        labels.append((nb, lab, nov))
        if lab in ("SELF", "EMPTY"):
            continue
        counts[lab] += 1
        if nov == "coined":
            n_coined += 1
        elif nov == "attested":
            n_attested += 1
    scored = max(1, sum(1 for nb in neighbors if norm(nb) and norm(nb) != norm(target)))
    n_stem = counts["STEM-SIBLING"]
    n_affix = counts["AFFIX-SIBLING"]
    n_sib = n_stem + n_affix
    return {
        "labels": labels,
        "filler_frac": counts["FILLER"] / scored,
        "n_stem_sib": n_stem,
        "n_affix_sib": n_affix,
        "n_sib": n_sib,
        "n_coined_sib": n_coined,
        "n_attested_sib": n_attested,
        "n_distant": counts["DISTANT-CONTENT"],
        "_scored": scored,
    }


def outcome_class(rank, prof):
    """Outcome from rank + neighborhood profile:
      hit             target in top-100
      comp-near-miss  miss, >=1 COINED sibling      (built a non-existent paradigm member)
      retr-near-miss  miss, siblings all attested   (retrieved the real paradigm)
      distant-miss    miss, no sibling, distant-dominated (a semantic, non-morphological route)
      filler-miss     miss, no sibling, filler-dominated  (no route -- fell back on syntax)
    """
    if rank is not None and 0 < rank <= 100:
        return "hit"
    if prof["n_coined_sib"] >= 1:
        return "comp-near-miss"
    if prof["n_sib"] >= 1:
        return "retr-near-miss"
    n_filler = round(prof["filler_frac"] * max(1, prof["_scored"]))
    return "distant-miss" if prof["n_distant"] >= n_filler else "filler-miss"


# --------------------------------------------------------------------------- #
# Loading rows
# --------------------------------------------------------------------------- #
def _int(v):
    try:
        return int(float(v)) if v not in ("", "None", None) else None
    except (ValueError, TypeError):
        return None


def parse_neighbors(row):
    return [w for w in (row.get("top_10_words") or "").split("|") if w != ""]


def load_synthetic(fname):
    """Load a joined synthetic CSV and flag each row CUED vs template-cued.

    A prompt used by more than one target word at the same context level is a
    family template written for one of them; the rest are scored against a
    sentence that does not point at them (and share its top-10 verbatim).
    """
    path = ANALYSIS / fname
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "target": r.get("target_word", ""),
                "rank": _int(r.get("rank")),          # -1 for miss
                "neighbors": parse_neighbors(r),
                "tier": r.get("tier"), "strategy": r.get("strategy"),
                "family": r.get("family"), "seed_word": r.get("seed_word"),
                "context_level": r.get("context_level"),
                "prompt": (r.get("prefix", ""), r.get("context_level", "")),
            })
    users = defaultdict(set)
    for r in rows:
        users[r["prompt"]].add(r["target"])
    for r in rows:
        r["cued"] = len(users[r["prompt"]]) == 1
    return rows


def load_real(fname, cutoff):
    path = RESULTS / fname
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            year = _int(r.get("year"))
            esy = _int(r.get("entry_start_year"))
            rows.append({
                "target": r.get("target_word", ""),
                "rank": _int(r.get("rank")),
                "neighbors": parse_neighbors(r),
                "year": year, "entry_start_year": esy,
                "post": year is not None and year > cutoff,
                "anchored": (year is not None and year > cutoff
                             and esy is not None and esy <= cutoff),
                "unanchored": (year is not None and year > cutoff
                               and esy is not None and esy > cutoff),
                "in_cutoff": year is not None and year <= cutoff,
            })
    return rows


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def rate(rows, pred):
    sub = [r for r in rows if pred(r)]
    k = sum(1 for r in sub if r)
    return len(sub), k


def pct(k, n):
    return f"{100.0*k/n:5.1f}%" if n else "   -- "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0, metavar="N",
                    help="print N sampled synthetic neighborhoods with per-neighbor labels")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    print("loading resources ...", file=sys.stderr)
    stops, lookup, suffixes, attest = load_resources()
    morph = Morph(lookup, suffixes)

    if args.selftest:
        run_selftest(morph, stops, attest)
        return

    # augment attestation with the seed words (known-real anchors)
    ANALYSIS.mkdir(exist_ok=True)
    all_syn, all_real = {}, {}
    for name, sfile, rfile, cutoff, kind in MODELS:
        syn = load_synthetic(sfile)
        real = load_real(rfile, cutoff)
        for r in syn:
            if r["seed_word"]:
                attest.add(norm(r["seed_word"]))
        all_syn[name] = syn
        all_real[name] = real

    # compute profiles
    for name, *_ in [(m[0],) for m in MODELS]:
        for r in all_syn.get(name, []):
            aff = family_affix(r["family"])
            r["prof"] = profile(r["target"], r["neighbors"], morph, stops, attest, aff)
            r["outcome"] = outcome_class(r["rank"], r["prof"])
        for r in all_real.get(name, []):
            r["prof"] = profile(r["target"], r["neighbors"], morph, stops, attest)
            r["outcome"] = outcome_class(r["rank"], r["prof"])

    if args.dump:
        dump_neighborhoods(all_syn, args.dump)
        return

    tidy = write_tidy(all_syn, all_real)
    view1_synth_nearmiss(all_syn)
    view2_outcome_signatures(all_syn, all_real)
    view3_base_vs_web(all_real)
    view4_reference(all_real)
    print(f"\nwrote {ANALYSIS/'neighborhood_simple.csv'}  ({tidy} rows)")


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def view1_synth_nearmiss(all_syn):
    print("\n" + "=" * 78)
    print("VIEW 1 — synthetic: relatedness density among MISSES, CUED ITEMS ONLY")
    print("  (does the model float a cluster of RELATED forms though it missed the exact target?")
    print("   'made-up' = related & absent from the dictionary = unleakable = composition-specific")
    print("   -- valid here, unlike on the real data; see module docstring.")
    print("   Template-cued rows are excluded: their prompt names another word, and rows")
    print("   sharing a prompt share a neighbor list verbatim.)")
    print("=" * 78)
    for name, syn in all_syn.items():
        cued = [r for r in syn if r["cued"]]
        misses = [r for r in cued if r["outcome"] != "hit"]
        n_words = len({r["target"] for r in cued})
        print(f"\n#### {name}   (cued words={n_words}, cued rows={len(cued)}, "
              f"cued misses N={len(misses)}; excluded {len(syn)-len(cued)} template rows)")
        print(f"  {'stratum':22s} {'N':>5} {'>=1 related':>12} {'>=1 made-up':>12} "
              f"{'mean rel/nbhd':>13} {'mean filler':>12}")
        def line(label, sub):
            n = len(sub)
            if not n:
                return
            kr = sum(1 for r in sub if r["prof"]["n_sib"] >= 1)
            km = sum(1 for r in sub if r["prof"]["n_coined_sib"] >= 1)
            mr = sum(r["prof"]["n_sib"] for r in sub) / n
            mf = sum(r["prof"]["filler_frac"] for r in sub) / n
            print(f"  {label:22s} {n:5d} {pct(kr,n):>12} {pct(km,n):>12} {mr:13.1f} {mf:12.2f}")
        line("ALL cued misses", misses)
        # context level is the dominant variable: a weak cue names nothing
        for lvl in ("high", "medium", "low"):
            line(f"context={lvl}", [r for r in misses if r["context_level"] == lvl])
        for tier in ("core", "floor"):
            line(f"tier={tier}", [r for r in misses if r["tier"] == tier])


def view2_outcome_signatures(all_syn, all_real):
    print("\n" + "=" * 78)
    print("VIEW 2 — outcome-class signatures (mean neighborhood profile)")
    print("=" * 78)
    print("  (coinedSib is meaningful for SYNTHETIC only — on real data it tracks scan")
    print("   noise and archaic spelling, not composition; see module docstring.)")
    for setname, data in (("SYNTHETIC (cued only)", all_syn), ("REAL post-cutoff", all_real)):
        print(f"\n#### {setname}")
        print(f"  {'model':12s} {'outcome':16s} {'N':>6} {'filler':>7} {'stemSib':>8} "
              f"{'affixSib':>8} {'coinedSib':>9} {'distant':>8}")
        for name, rows in data.items():
            pool = rows
            if setname.startswith("REAL"):
                pool = [r for r in rows if r["post"]]
            else:
                pool = [r for r in rows if r["cued"]]
            for oc in ("hit", "comp-near-miss", "retr-near-miss", "distant-miss", "filler-miss"):
                sub = [r for r in pool if r["outcome"] == oc]
                n = len(sub)
                if not n:
                    continue
                def m(key):
                    return sum(r["prof"][key] for r in sub) / n
                print(f"  {name:12s} {oc:16s} {n:6d} {m('filler_frac'):7.2f} "
                      f"{m('n_stem_sib'):8.2f} {m('n_affix_sib'):8.2f} "
                      f"{m('n_coined_sib'):9.2f} {m('n_distant'):8.2f}")


def view3_base_vs_web(all_real):
    """Paired Base-vs-Web on identical real items, restricted to items BOTH miss.

    Restricting to shared misses removes the recall-rate confound: pooling hits and
    misses lets Web's much higher post-cutoff recall inflate its neighborhood
    statistics, since a model that lands the target also tends to surface its
    near-variants.  The made-up/coined column is omitted here -- on real data it
    measures scanning noise and archaic orthography (see module docstring).
    """
    print("\n" + "=" * 78)
    print("VIEW 3 — Base vs Web, identical real items, BOTH-MISS only")
    print("  (both-miss removes the recall-rate confound; relatedness only, no made-up)")
    print("=" * 78)
    base = {(r["target"], r.get("year")): r for r in all_real.get("talkie-base", [])}
    web = {(r["target"], r.get("year")): r for r in all_real.get("talkie-web", [])}

    def missed(r):
        return not (r["rank"] is not None and 0 < r["rank"] <= 100)

    print(f"  {'stratum':20s} {'N':>7} {'related B':>11} {'related W':>11} "
          f"{'distant B':>11} {'distant W':>11} {'filler B':>10} {'filler W':>10}")
    for label, pred in (("in-cutoff", lambda r: r["in_cutoff"]),
                        ("post-cutoff (all)", lambda r: r["post"]),
                        ("post anchored", lambda r: r["anchored"]),
                        ("post unanchored", lambda r: r["unanchored"])):
        common = [k for k in base if k in web and pred(base[k])
                  and missed(base[k]) and missed(web[k])]
        if not common:
            continue
        def m(d, key):
            return sum(d[k]["prof"][key] for k in common) / len(common)
        print(f"  {label:20s} {len(common):7d} {m(base,'n_sib'):11.3f} {m(web,'n_sib'):11.3f} "
              f"{m(base,'n_distant'):11.2f} {m(web,'n_distant'):11.2f} "
              f"{m(base,'filler_frac'):10.2f} {m(web,'filler_frac'):10.2f}")


def view4_reference(all_real):
    print("\n" + "=" * 78)
    print("VIEW 4 — plain reference: in-cutoff RECALLED words (ordinary retrieval)")
    print("  (coinedSib shown but NOT interpretable on real data — scan/archaic noise)")
    print("=" * 78)
    print(f"  {'model':12s} {'N':>6} {'filler':>7} {'stemSib':>8} {'affixSib':>8} "
          f"{'coinedSib*':>10} {'distant':>8}")
    for name, rows in all_real.items():
        sub = [r for r in rows if r["in_cutoff"] and r["rank"] and 0 < r["rank"] <= 100]
        n = len(sub)
        if not n:
            continue
        def m(key):
            return sum(r["prof"][key] for r in sub) / n
        print(f"  {name:12s} {n:6d} {m('filler_frac'):7.2f} {m('n_stem_sib'):8.2f} "
              f"{m('n_affix_sib'):8.2f} {m('n_coined_sib'):10.2f} {m('n_distant'):8.2f}")


# --------------------------------------------------------------------------- #
# CSV + dump + selftest
# --------------------------------------------------------------------------- #
def write_tidy(all_syn, all_real):
    out = ANALYSIS / "neighborhood_simple.csv"
    cols = ["model", "set", "target", "outcome", "rank", "filler_frac",
            "n_stem_sib", "n_affix_sib", "n_coined_sib", "n_distant",
            "tier", "strategy", "family", "context_level", "cued",
            "post", "anchored", "unanchored", "in_cutoff"]
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, rows in all_syn.items():
            for r in rows:
                p = r["prof"]
                w.writerow([name, "synthetic", r["target"], r["outcome"], r["rank"],
                            f"{p['filler_frac']:.3f}", p["n_stem_sib"], p["n_affix_sib"],
                            p["n_coined_sib"], p["n_distant"], r["tier"], r["strategy"],
                            r["family"], r["context_level"], r["cued"], "", "", "", ""])
                n += 1
        for name, rows in all_real.items():
            for r in rows:
                p = r["prof"]
                w.writerow([name, "real", r["target"], r["outcome"], r["rank"],
                            f"{p['filler_frac']:.3f}", p["n_stem_sib"], p["n_affix_sib"],
                            p["n_coined_sib"], p["n_distant"], "", "", "", "", "",
                            r["post"], r["anchored"], r["unanchored"], r["in_cutoff"]])
                n += 1
    return n


def dump_neighborhoods(all_syn, k):
    syn = all_syn.get("talkie-base", [])
    # deterministic spread: every len/k-th item, prefer cued high-context
    hi = [r for r in syn if r["context_level"] == "high" and r["cued"]] or syn
    step = max(1, len(hi) // k)
    picks = hi[::step][:k]
    for r in picks:
        p = r["prof"]
        print(f"\n[{r['outcome']}] target={r['target']}  seed={r['seed_word']}  "
              f"family={r['family']}  strategy={r['strategy']}  rank={r['rank']}")
        for nb, lab, nov in p["labels"]:
            tag = f"{lab}" + (f"/{nov}" if nov else "")
            print(f"    {nb:22s} -> {tag}")


def run_selftest(morph, stops, attest):
    cases = [
        ("proton", "electron", "AFFIX-SIBLING", "attested"),
        ("photon", "ideon", "AFFIX-SIBLING", "coined"),
        ("aerospace", "terraspace", None, None),   # share 'space'
        ("citrullinase", "citrullase", "STEM-SIBLING", None),
        ("car", "automobile", "DISTANT-CONTENT", ""),
        ("car", "the", "FILLER", ""),
    ]
    print("SELFTEST")
    ok = True
    fam_aff = {"aerospace": "space", "proton": "on", "photon": "on"}
    for t, n, exp_lab, exp_nov in cases:
        aff = fam_aff.get(t, "")
        lab, nov = classify_neighbor(t, n, morph, stops, attest, aff)
        flag = ""
        if exp_lab and lab != exp_lab:
            flag = f"  <-- expected {exp_lab}"; ok = False
        if exp_nov not in (None, "") and nov != exp_nov:
            flag += f"  <-- expected novelty {exp_nov}"; ok = False
        print(f"  {t:14s} ~ {n:12s} -> {lab}/{nov or '-'}{flag}")
    print("OK" if ok else "FAILURES ABOVE")


if __name__ == "__main__":
    main()
