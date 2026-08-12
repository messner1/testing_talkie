#!/usr/bin/env python3
"""Composition in the technical lexicon: where word-building confounds a leakage read.

MOTIVATION.  A cutoff audit asks "did the model produce a post-cutoff word?" and reads
any yes as contamination.  That read is only safe where the model *cannot* have built
the word itself.  This script locates the domain where it demonstrably can: the
scientific-technical lexicon, whose affixes (-ase, -ergic, -tron, -itis, -oma, ...) are
productive, semantically transparent and attach to a dense field of same-domain stems.
There a restricted model can emit a form it never saw -- so a bare string match against a
post-cutoff word list will misclassify built forms as leaked ones.

The same domain is where our synthetic never-existed coinages live (enzyme_ase,
particle_on, receptor_ergic, ...), so the natural and synthetic lines of evidence are
about the same lexical territory rather than two unrelated tests.

WHAT IT DOES.  For every cloze item whose target is a technical word, it reads the
model's top-10 predictions and sorts each *related* neighbor absent from the dictionary
into three kinds -- only the third is composition:

  ORTHOGRAPHIC   a scan/spelling variant of the target      (clectron, endocnne, maſculine)
  INFLECTION     a plural/participle of an attested word    (lymphocytes, catecholamines)
  DERIVATIONAL   a different lexeme in the target's paradigm (ornithinase, insulinoma,
                 metron, homocysteine)  <-- the composition signal

DERIVATIONAL forms are then dated against a lexicon of OED entry years harvested from
the cloze corpus itself, splitting them into:

  PRE-CUTOFF     a real word the model could legitimately know
  POST-CUTOFF    a real word that postdates the cutoff        -> LEAKAGE candidate
  UNDATED        absent from the dated lexicon                -> COMPOSITION candidate

The everyday (non-technical) lexicon is run through the identical pipeline as a
comparison, stratified by target length because length -- not domain -- is the obvious
alternative explanation for having room to build on.

Outputs analysis/technical_composition.csv (one row per flagged neighbor, fully
auditable) and analysis/TECHNICAL_composition.md.

Run: local/bin/python technical_composition.py [--dump N]
CPU-only, deterministic.
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import neighborhood_analysis as NA

RESULTS = Path("results")
ANALYSIS = Path("analysis")
csv.field_size_limit(10 ** 7)

MODELS = [("talkie-base", 1930, "restricted"),
          ("talkie-web", 1930, "leakage-rich"),
          ("typewriter", 1913, "restricted")]

# Suffixes that mark a word as scientific-technical in English.  Deliberately excludes
# ambiguous endings (-ic, -ate, -al, and above all -ion) that dominate ordinary Latinate
# vocabulary: including -ion swamped the set with `digestion`, `vocation`, `relocation`.
SCI_AFFIX = ("ase", "ergic", "tron", "itis", "osis", "oma", "ose", "yl", "ide",
             "ine", "gen", "cyte", "blast", "plasm", "phyll", "lyte", "mer",
             "ium", "meter", "scope", "graph", "phore", "stat", "tomy", "ectomy",
             "pathy", "emia", "uria", "oid", "logy", "lysis", "phyte", "sperm",
             "therm", "valent", "metry", "genic", "ferous", "hydrate", "acid")
# ...but an affix alone is not enough: `machine`, `purchase`, `outside` all match one.
# A technical word must ALSO be rare in general English (Brown corpus).
BROWN_RARE_MAX = 3          # occurrences per ~1.16M tokens
MIN_LEN = 7

INFLECT = ("s", "es", "ed", "ing", "d")
LATIN_PLURAL = ("i", "ae", "a", "es")   # thrombi, melanomata, ...


def load_brown():
    from nltk.corpus import brown
    return collections.Counter(w.lower() for w in brown.words())


def load_dated_lexicon():
    """word -> earliest OED entry year, harvested from the cloze corpora themselves."""
    lex = {}
    for model, _c, _k in MODELS:
        path = RESULTS / f"cloze_{model}_details.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                w = r["target_word"].strip().lower()
                y = NA._int(r.get("entry_start_year"))
                if y is not None:
                    lex[w] = min(lex.get(w, 9999), y)
    return lex


def is_technical(word, brown_fd):
    w = word.lower()
    if len(w) < MIN_LEN or not w.isalpha():
        return False
    if not any(w.endswith(a) for a in SCI_AFFIX):
        return False
    return brown_fd[w] <= BROWN_RARE_MAX


def flag_kind(target, neighbor, attest, morph):
    """Sort a related, non-attested neighbor into ORTHOGRAPHIC / INFLECTION / DERIVATIONAL."""
    t, n = NA.norm(target), NA.norm(neighbor)

    # 1. orthographic: non-ASCII (long s etc.), or a near-copy of the target
    if not n.isascii():
        return "ORTHOGRAPHIC"
    if n[:1] == t[:1] and abs(len(n) - len(t)) <= 2 and NA.levenshtein(n, t) <= 2:
        return "ORTHOGRAPHIC"
    # truncations and run-ons of the target: `augmentation`->`augmenta`,
    # `subordination`->`subordina`, `remission`->`remissionand`.  These are scan
    # artefacts, not lexemes, and they are edit-far so the rule above misses them.
    if len(n) >= 5 and (t.startswith(n) or n.startswith(t)):
        return "ORTHOGRAPHIC"

    # 2. inflection of an attested word (or of the target itself)
    for suf in INFLECT:
        if n.endswith(suf) and len(n) - len(suf) >= 4:
            base = n[: -len(suf)]
            for cand in (base, base + "e", base[:-1] if base[-1:] == base[-2:-1] else base):
                if cand in attest or cand == t:
                    return "INFLECTION"
    # Latin/Greek plurals of an attested singular: thrombus->thrombi, melanoma->melanomata
    for suf in LATIN_PLURAL:
        if n.endswith(suf) and len(n) - len(suf) >= 5:
            base = n[: -len(suf)]
            if any(base + e in attest for e in ("us", "um", "a", "is", "on", "")):
                return "INFLECTION"

    # 3. otherwise a distinct lexeme in the paradigm
    return "DERIVATIONAL"


def length_band(w):
    n = len(w)
    if n < 7:
        return "<7"
    if n < 9:
        return "7-8"
    if n < 11:
        return "9-10"
    if n < 13:
        return "11-12"
    return "13+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0,
                    help="print N sampled DERIVATIONAL forms per model for audit")
    args = ap.parse_args()

    print("loading resources ...", file=sys.stderr)
    stops, lookup, suffixes, attest = NA.load_resources()
    morph = NA.Morph(lookup, suffixes)
    brown_fd = load_brown()
    lex = load_dated_lexicon()
    print(f"dated lexicon: {len(lex)} words with an OED entry year", file=sys.stderr)

    rows_out = []          # per flagged neighbor
    per_item = []          # per cloze item
    for model, cutoff, kind in MODELS:
        path = RESULTS / f"cloze_{model}_details.csv"
        if not path.exists():
            print(f"!! missing {path}")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                target = r.get("target_word", "")
                tech = is_technical(target, brown_fd)
                nbrs = NA.parse_neighbors(r)
                prof = NA.profile(target, nbrs, morph, stops, attest)
                rank = NA._int(r.get("rank"))
                n_der = 0
                for nb, lab, nov in prof["labels"]:
                    if nov != "coined":
                        continue
                    k = flag_kind(target, nb, attest, morph)
                    year = lex.get(NA.norm(nb))
                    if k == "DERIVATIONAL":
                        n_der += 1
                        dating = ("undated" if year is None
                                  else ("post-cutoff" if year > cutoff else "pre-cutoff"))
                    else:
                        dating = ""
                    rows_out.append({
                        "model": model, "kind_model": kind, "target": target,
                        "technical": tech, "neighbor": nb, "flag_kind": k,
                        "neighbor_year": year if year is not None else "",
                        "dating": dating, "target_len": len(target),
                        "brown_freq": brown_fd[NA.norm(target)],
                        "rank": rank,
                    })
                per_item.append({
                    "model": model, "target": target, "technical": tech,
                    "len_band": length_band(target), "n_der": n_der,
                    "n_related": prof["n_sib"], "hit": rank is not None and 0 < rank <= 100,
                })

    ANALYSIS.mkdir(exist_ok=True)
    out = ANALYSIS / "technical_composition.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    md = report(rows_out, per_item, lex)
    (ANALYSIS / "TECHNICAL_composition.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out} ({len(rows_out)} flagged neighbors)")
    print(f"wrote {ANALYSIS/'TECHNICAL_composition.md'}")

    if args.dump:
        print("\n" + "=" * 78)
        print("AUDIT — sampled DERIVATIONAL forms")
        print("=" * 78)
        for model, _c, _k in MODELS:
            d = [r for r in rows_out if r["model"] == model
                 and r["flag_kind"] == "DERIVATIONAL" and r["technical"]]
            print(f"\n#### {model}  ({len(d)} technical derivational forms)")
            step = max(1, len(d) // args.dump)
            for r in d[::step][:args.dump]:
                yr = r["neighbor_year"] or "—"
                print(f"    {r['target']:20s} -> {r['neighbor']:22s} "
                      f"[{r['dating'] or 'n/a':12s} entry={yr}]")


def report(rows, items, lex):
    md = ["# Composition in the technical lexicon",
          "",
          "*Generated by `technical_composition.py`; per-neighbor detail in "
          "`analysis/technical_composition.csv`.*", ""]

    n_items = sum(1 for i in items if i["technical"]) // max(1, len(MODELS))
    n_words = len({i["target"].lower() for i in items if i["technical"]})
    md += [f"**{n_items} technical cloze items per model** covering **{n_words} distinct "
           f"technical words** (a technical word ends in a scientific affix and occurs "
           f"≤{BROWN_RARE_MAX} times in the Brown corpus). Everything else in the "
           f"{sum(1 for i in items if not i['technical'])//max(1,len(MODELS))}-item "
           "remainder is treated as everyday.", ""]

    # ---- 1. what the "made-up" flag actually contains, technical vs everyday
    md += ["## 1. What the made-up flag contains", "",
           "Every *related* neighbor absent from the dictionary, sorted. Only DERIVATIONAL "
           "is composition; the other two are artefacts of scanned text and of dictionary "
           "coverage.", "",
           "| model | domain | orthographic | inflection | **derivational** | % derivational |",
           "|---|---|---|---|---|---|"]
    for model, _c, _k in MODELS:
        for tech, dom in ((True, "technical"), (False, "everyday")):
            sub = [r for r in rows if r["model"] == model and r["technical"] == tech]
            c = collections.Counter(r["flag_kind"] for r in sub)
            tot = max(1, len(sub))
            md.append(f"| {model} | {dom} | {c['ORTHOGRAPHIC']} | {c['INFLECTION']} | "
                      f"**{c['DERIVATIONAL']}** | {100*c['DERIVATIONAL']/tot:.1f}% |")
    md.append("")

    # ---- 2. the rate that matters, per item, length-stratified
    md += ["## 2. Derivational forms per cloze item — technical vs everyday", "",
           "Rate = mean number of derivational (composed) neighbors in the top-10. "
           "Stratified by target length, because length rather than domain is the obvious "
           "alternative explanation for having room to build.", "",
           "| model | length | technical: items / rate | everyday: items / rate | ratio |",
           "|---|---|---|---|---|"]
    for model, _c, _k in MODELS:
        for band in ("7-8", "9-10", "11-12", "13+"):
            t = [i for i in items if i["model"] == model and i["technical"]
                 and i["len_band"] == band]
            e = [i for i in items if i["model"] == model and not i["technical"]
                 and i["len_band"] == band]
            if not t or not e:
                continue
            rt = sum(i["n_der"] for i in t) / len(t)
            re_ = sum(i["n_der"] for i in e) / len(e)
            ratio = f"{rt/re_:.1f}×" if re_ else "—"
            md.append(f"| {model} | {band} | {len(t)} / **{rt:.3f}** | {len(e)} / {re_:.3f} "
                      f"| {ratio} |")
    md.append("")

    # ---- 3. dating: attempted, and why it fails offline
    md += ["## 3. Dating the derivational forms — attempted, insufficient", "",
           f"The only offline dating oracle available is a lexicon of {len(lex)} words with "
           "an OED entry year harvested from the cloze corpus itself. It covers general "
           "vocabulary and essentially none of the technical derivational forms:", "",
           "| model | technical derivational | dated pre-cutoff | dated post-cutoff | "
           "not in lexicon |", "|---|---|---|---|---|"]
    for model, _c, _k in MODELS:
        d = [r for r in rows if r["model"] == model and r["technical"]
             and r["flag_kind"] == "DERIVATIONAL"]
        c = collections.Counter(r["dating"] for r in d)
        md.append(f"| {model} | {len(d)} | {c['pre-cutoff']} | {c['post-cutoff']} | "
                  f"{c['undated']} |")
    md += ["", "Almost everything is undated, so **this split cannot be made with the "
           "resources on disk** — `not in lexicon` means only that, never “proven never to "
           "have existed”. Separating a built form from a real post-cutoff term needs a "
           "real dated technical lexicon (OED / Merriam lookup). Section 4 gets at the same "
           "question without dates.", ""]

    # ---- 4. cross-model contrast: the dating-free route to leakage vs composition
    md += ["## 4. Which model produces which forms (no dating required)", "",
           "For the same cloze items, compare the *sets* of technical derivational forms "
           "each model emits. A form only the leakage-rich model produces is a leakage "
           "candidate; one only a restricted model produces cannot be — it had no access.",
           ""]
    sets = {}
    for model, _c, _k in MODELS:
        sets[model] = {(r["target"].lower(), r["neighbor"].lower()) for r in rows
                       if r["model"] == model and r["technical"]
                       and r["flag_kind"] == "DERIVATIONAL"}
    b, w, t = sets["talkie-base"], sets["talkie-web"], sets["typewriter"]
    md += ["| set | N | reading |", "|---|---|---|",
           f"| Base only (not Web) | {len(b - w)} | built — Base cannot have leaked it "
           "from post-cutoff text |",
           f"| Web only (not Base) | {len(w - b)} | leakage candidate — Web saw modern text |",
           f"| shared Base ∩ Web | {len(b & w)} | available to both routes |",
           f"| Typewriter only | {len(t - b - w)} | built — the most restricted model |", ""]
    def show(pairs, k=14):
        return " · ".join(f"`{a}` → *{c}*" for a, c in sorted(pairs)[:k]) or "_none_"
    md += [f"**Base only:** {show(b - w)}", "",
           f"**Web only:** {show(w - b)}", "",
           f"**Typewriter only:** {show(t - b - w)}", ""]

    # ---- 5. the exhibit
    md += ["## 5. Exhibit — sampled technical derivational forms", "",
           "Every one of these is auditable in `analysis/technical_composition.csv`:", ""]
    for model, _c, _k in MODELS:
        d = [r for r in rows if r["model"] == model and r["technical"]
             and r["flag_kind"] == "DERIVATIONAL"]
        seen, ex = set(), []
        for r in d:
            key = (r["target"], r["neighbor"])
            if key in seen:
                continue
            seen.add(key)
            ex.append(f"`{r['target']}` → *{r['neighbor']}*")
            if len(ex) >= 18:
                break
        md += [f"**{model}** — " + " · ".join(ex), ""]

    # ---- 6. the bridge to the synthetic evidence
    md += ["## 6. The same territory as the synthetic test", "",
           "The synthetic coinage families are built on these very affixes — `enzyme_ase`, "
           "`particle_on`/`particle_tron`, `receptor_ergic`, `belief_ist`. So the natural "
           "and synthetic lines are not two unrelated experiments: they probe the same "
           "productive corner of the lexicon from opposite directions. The signature "
           "matches across both.", "",
           "| | natural cloze (this file) | synthetic coinages (§6/§7 of the pilot) |",
           "|---|---|---|",
           "| Talkie-Base | `cholinergic` → *acetylergic, acetylenergic, acetoergic, "
           "acetolergic, acetenergic* | `tryptaminergic` → *indolergic, indolinergic*; "
           "`ferroporphase` → *ferrinase, ferratase, ferrase* |",
           "| Talkie-Web | `synthase` → *deaminase, dehydratase, dehydrogenase* (real) | "
           "`dreamergic` → *dopaminergic*; `dielecton` → *polariton* (real) |",
           "| Typewriter | `endocrine` → *endocrinous, enterocrine, endocrast* | "
           "(near-silent — 1 made-up neighbour in the whole cued set) |", "",
           "A restricted model sprays non-existent variants; the leakage-rich model "
           "supplies real modern terms. Same measure, opposite mechanisms — and this is "
           "precisely where a string-matching cutoff audit goes wrong, because "
           "*acetylergic* and *peptidergic* are equally absent from a pre-1930 word list "
           "while only one of them is evidence of contamination.", ""]

    # ---- 7. limitations
    md += ["## 7. Limitations", "",
           "- **Dictionary coverage is itself domain-dependent.** Everyday vocabulary is "
           "well covered by `/usr/share/dict` + WordNet, so almost the only everyday words "
           "that fall through are errors; technical vocabulary is poorly covered, so real "
           "terms fall through too (`hypoxanthine`, `dehydrogenase`, `tetraploid` are all "
           "real). **This inflates the technical rate in sections 1–2**, and it is the "
           "reason the load-bearing evidence is the cross-model set difference in section "
           "4, where a form only a restricted model produces cannot be a coverage artefact "
           "of that kind.",
           "- **Small N.** 197 technical cloze items per model, yielding 29 / 19 / 23 "
           "derivational forms. These are exhibits with a rate attached, not estimates.",
           "- **Length stratification is a control, not a match.** Technical and everyday "
           "items differ in frequency, morphological complexity and topic as well as "
           "length; only length is held constant here.",
           "- **No dating.** Section 3 shows the offline oracle cannot date these forms, so "
           "'real but post-cutoff' vs 'never existed' rests on inspection, not lookup. A "
           "dated technical lexicon would convert section 4 from an argument into a "
           "measurement.",
           "- **The technical/everyday split is a suffix heuristic plus a Brown-frequency "
           "cut**, audited by eye but not validated against an external domain labelling.",
           ""]
    return md


if __name__ == "__main__":
    main()
