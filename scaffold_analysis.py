#!/usr/bin/env python3
"""Characterize the scaffolded cloze subset: does compositional leakage look different?

THE DISCRIMINATOR.  Over top-K, a composed post-cutoff hit and a leaked one are the same
event -- both are "the model produced a word that postdates its cutoff", which is all the
leakage metric can see.  The claim of this pass is that the *neighborhood* around the hit
distinguishes them:

  COMPOSITIONAL   the top-10 carries form-variants of the target -- visible attempts,
                  the model searching a paradigm  (cholinergic -> acetylergic, acetoergic)
  RECALL          no form-variants; the other predictions are semantically apt but
                  formally unrelated -- competitors, not attempts  (cholinergic ->
                  muscarinic, nicotinic, ACh)

Morphological legitimacy is deliberately not required: orthographic nearness is enough.
The question is whether the output *looks like* building, not whether a linguist would
license it.

TWO LABEL SOURCES, ON PURPOSE.
  AUTO  presence of >=1 form-variant in the top-10, from the surface tools in
        neighborhood_analysis.py (morpheme segmentation + edit distance + shared affix).
        Uses no dictionary, so it is unaffected by the OCR-noise problem that invalidated
        the "made-up" flag on real data, and it runs over the whole corpus.
  HAND  the blind stratified sheet from scaffold_subset.py.  Supplies the judgment the
        tools cannot make -- whether the NON-variant neighbours are semantically apt --
        which is what separates RECALL from NEITHER.  Absence of form-variants is only
        evidence of retrieval if something apt is there instead.

AUTO runs now over everything; HAND refines it and is scored against it (confusion
matrix).  If the sheet is unlabelled the script still produces the full AUTO analysis and
says so.

Outputs analysis/SCAFFOLD_characterization.md.
Run: local/bin/python scaffold_analysis.py [--dump N]
CPU-only, deterministic.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import assoc as AS                          # noqa: E402  (lambda, G^2)
import neighborhood_analysis as NA          # noqa: E402
from scaffold_subset import GRADES, MODELS  # noqa: E402

ANALYSIS = Path("analysis")
RESULTS = Path("results")
csv.field_size_limit(10 ** 7)

SUBSET = ANALYSIS / "scaffold_subset.csv"
# Sheet A's key, not the deleted `scaffold_labels_KEY.csv`. The old pair pointed at a sheet
# that no longer exists, and its ids (S-hashes over model+target+prefix) are the same scheme
# but were emitted by a different generator -- joining Sheet A labels against it would have
# silently matched nothing and reported an empty audit as a clean one.
KEY = ANALYSIS / "neighborhood_KEY.csv"
OUT = ANALYSIS / "SCAFFOLD_characterization.md"

SIBLING_LABELS = ("STEM-SIBLING", "AFFIX-SIBLING")


# --------------------------------------------------------------------------- #
# Automatic labelling
# --------------------------------------------------------------------------- #
def load_subset():
    """Read the subset table. Form-variants were computed by scaffold_subset.py, where
    the prefix was in hand; this script only reads them."""
    if not SUBSET.exists():
        sys.exit(f"missing {SUBSET} -- run scaffold_subset.py first")
    rows = []
    with open(SUBSET) as f:
        for r in csv.DictReader(f):
            for k in ("is_future", "hit", "rank", "comp_strict", "comp_any",
                      "n_attempts", "n_echoes", "donor_distance", "ending_types"):
                r[k] = int(r[k])
            r["neighbors"] = NA.parse_neighbors(r)
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _cell(rows, pred):
    """'strict / any (N=…)' — both readings of a form-variant that is also a prompt word.

    strict counts only variants absent from the prefix; any also counts the prefix ones.
    Neither is neutral (see form_variants() in scaffold_subset.py), so both are shown and
    the reader can see whether a conclusion depends on the choice.
    """
    sub = [r for r in rows if pred(r)]
    if not sub:
        return "-"
    n = len(sub)
    s = sum(r["comp_strict"] for r in sub) / n
    a = sum(r["comp_any"] for r in sub) / n
    return f"{s:.3f} / {a:.3f} (N={n})"


def _hdr(models, extra=()):
    cols = list(extra) + list(models)
    return ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]


def table_ladder(rows, models, out):
    """View 1 -- form-variant presence by scaffold grade x stratum x model."""
    out.append("## 1. Does the neighborhood look compositional? By scaffold grade\n")
    out.append("Share of neighborhoods carrying at least one form-variant of the target. "
               "Each cell is **strict / any** — strict counts only form-variants absent "
               "from the prompt, `any` also counts ones the prompt supplied. Neither "
               "reading is neutral, so both are shown.\n")
    for stratum, isf in (("post-cutoff", 1), ("in-cutoff", 0)):
        out.append(f"\n**{stratum}**\n")
        out.extend(_hdr(models, ["scaffold"]))
        for g in reversed(GRADES):
            cells = [_cell(rows, lambda r, g=g, m=m, i=isf:
                           r["grade"] == g and r["model"] == m and r["is_future"] == i)
                     for m in models]
            out.append(f"| {g} | " + " | ".join(cells) + " |")


def table_recruitment(rows, models, out):
    """View 1b -- is it scaffolding, or a common ending happening to sit nearby?

    The orthographic grade cannot tell `adrenergic` -> `cholinergic` (Dale's sentence
    coordinates them: "I suggest the words 'adrenergic' and ___") from `station` ->
    `transcription` (both merely end in -tion, in unrelated prose). Two automatic
    moderators separate them; the third, the recruiting *construction*, is being derived
    from the blind prompt sheet rather than guessed at.

    These are reported as strata, never as filters. The very-common-ending class is a
    built-in negative control: if bare co-occurrence produced the effect, it would appear
    there. Filtering it out would produce cleaner numbers and forfeit the control.
    """
    out.append("\n\n## 1b. Two recruitment routes, and where co-occurrence does nothing\n")
    out.append("Recruitment comes in two forms, and conflating them hides the result:\n")
    out.append("- **paradigmatic** — the prompt supplies a *stem* and the target is a "
               "derivation off it (`quantizing` → `quantization`, `lock` → `unlock`). No "
               "syntactic frame is needed; the lexicon does the work.\n"
               "- **syntagmatic** — the prompt supplies only an *affix-mate*, so it needs "
               "a construction to recruit it (`the neutron and ___`).\n")
    out.append("Cells are recall@100 · form-variants (strict).\n")
    out.extend(_hdr(models, ["moderator", "level"]))

    def cells_for(pred):
        out_cells = []
        for m in models:
            s = [r for r in rows if r["model"] == m and r["is_future"] == 1 and pred(r)]
            out_cells.append(f"{sum(r['hit'] for r in s)/len(s):.3f} · "
                             f"{sum(r['comp_strict'] for r in s)/len(s):.3f} (N={len(s)})"
                             if s else "-")
        return out_cells

    for lv in ("paradigmatic", "syntagmatic"):
        out.append(f"| route | `{lv}` | " + " | ".join(
            cells_for(lambda r, lv=lv: r["route"] == lv)) + " |")
    # Ending rarity is scored ONLY on the syntagmatic route. On the paradigmatic route the
    # shared material is the stem and the ending is incidental -- `quantization` ends in
    # -tion but its donor is `quantizing`, so grading it by ending frequency mixes two
    # populations and understates the coincidence class.
    for lv in ("rare", "common", "very-common"):
        out.append(f"| ↳ ending rarity (syntagmatic only) | `{lv}` | " + " | ".join(
            cells_for(lambda r, lv=lv: r["route"] == "syntagmatic"
                      and r["ending_rarity"] == lv)) + " |")
    for lv in ("<=5", "6-15", ">15"):
        out.append(f"| donor locality | `{lv}` | " + " | ".join(
            cells_for(lambda r, lv=lv: r["grade"] != "NONE"
                      and r["locality"] == lv)) + " |")
    out.append("| *(unscaffolded)* | *baseline* | " + " | ".join(
        cells_for(lambda r: r["grade"] == "NONE")) + " |")

    out.append("\n**It is the stem that matters, not the affix.** For Talkie-Base the "
               "paradigmatic route recalls at roughly twice the unscaffolded baseline, "
               "while the syntagmatic route as a whole sits *at* it. An affix-mate in the "
               "prompt buys nothing on average; a stem buys a great deal.")
    out.append("\n**The negative control fires inside the syntagmatic route.** Split by "
               "how informative the shared ending is, affix-only items with a "
               "very-common ending — overwhelmingly `-tion`, shared by 440 distinct "
               "corpus targets — fall *below* the unscaffolded baseline, while rare "
               "endings run well above it. Sharing a frequent ending with a nearby word "
               "is not scaffolding. This class is reported rather than filtered: it is "
               "the cell that shows bare co-occurrence does nothing.")
    out.append("\n**A caution about the form-variant measure.** The inert stratum still "
               "produces form-variants at several times the baseline while gaining no "
               "recall. The measure is partly reading **base-rate affix productivity** — "
               "the model emits `-tion` words because `-tion` words are everywhere — "
               "rather than scaffolded composition. Form-variant rate is not by itself "
               "evidence of building; it has to be read against the ending's frequency.")
    out.append("\n*An earlier version of this table applied ending rarity to every "
               "scaffolded item and reported the inert stratum at 0.220. That figure was "
               "a mixture: stem-donor items that merely happen to end in a common suffix "
               "were pooled with genuine affix coincidences. Splitting by route separates "
               "them and the coincidence class is lower than first reported.*")
    out.append("\n*The third component — the construction that recruits the donor into a "
               "parallel with the slot ('I suggest the words A and ___', 'similarly', "
               "'as with', 'so-called', enumeration, contrast) — is deliberately not "
               "detected by a regex written in advance. Its taxonomy is being derived "
               "from the blind prompt sheet `analysis/scaffold_prompt_labels.csv`; the "
               "detector will be built against those labels and its precision reported.*")


def table_headline(rows, models, out):
    """View 2 -- the RNL numerator itself: post-cutoff HITS only."""
    out.append("\n\n## 2. Headline: the leakage events themselves\n")
    out.append("Restricted to **post-cutoff hits** — the numerator of the leakage metric. "
               "Each of these is an event a cutoff audit counts as contamination. The "
               "question is what share of them carry visible attempts.\n")
    out.extend(_hdr(models, ["scaffold"]))
    for g in reversed(GRADES):
        cells = [_cell(rows, lambda r, g=g, m=m:
                       r["grade"] == g and r["model"] == m
                       and r["is_future"] == 1 and r["hit"] == 1)
                 for m in models]
        out.append(f"| {g} | " + " | ".join(cells) + " |")

    out.append("\nIn-cutoff hits at the same grades, as the within-model control. These "
               "are legitimate recall, so a compositional signature here is a baseline "
               "rate rather than a leakage finding — what matters is the *gap* between "
               "the two tables:\n")
    out.extend(_hdr(models, ["scaffold"]))
    for g in reversed(GRADES):
        cells = [_cell(rows, lambda r, g=g, m=m:
                       r["grade"] == g and r["model"] == m
                       and r["is_future"] == 0 and r["hit"] == 1)
                 for m in models]
        out.append(f"| {g} | " + " | ".join(cells) + " |")


def table_breadth(rows, models, out):
    """View 3 -- is this confined to technical vocabulary?"""
    out.append("\n\n## 3. Breadth: is this a technical-vocabulary phenomenon?\n")
    out.append("Earlier work localized composition in the technical lexicon. Most "
               "scaffolded items are *everyday* vocabulary, so this is a real test.\n")
    out.extend(_hdr(models, ["stratum", "domain"]))
    for isf, lab in ((1, "post-cutoff"), (0, "in-cutoff")):
        for dom in ("technical", "everyday"):
            cells = [_cell(rows, lambda r, d=dom, m=m, i=isf:
                           r["grade"] != "NONE" and r["domain"] == d
                           and r["model"] == m and r["is_future"] == i)
                     for m in models]
            out.append(f"| {lab} | {dom} | " + " | ".join(cells) + " |")
    out.append("\n*Scaffolded technical items are scarce (38 STEM+AFFIX/STEM/AFFIX items "
               "in the corpus), so the technical row is an exhibit with a rate attached, "
               "not an estimate.*")


def table_accounting(rows, models, out):
    """View 4 -- granular accounting against the gross metric, with the ceiling stated."""
    out.append("\n\n## 4. Accounting against the leakage metric\n")
    out.append("RNL is the gross characterization; this pass accounts for one component "
               "of what produces it. The share is small and is stated here rather than "
               "left for a reader to compute.\n")
    out.append("| model | post-cutoff hits | scaffolded | + compositional (strict) | "
               "share (strict) | + compositional (any) | share (any) |")
    out.append("|---|---|---|---|---|---|---|")
    for m in models:
        hits = [r for r in rows if r["model"] == m and r["is_future"] == 1 and r["hit"]]
        scaf = [r for r in hits if r["grade"] != "NONE"]
        cs = sum(r["comp_strict"] for r in scaf)
        ca = sum(r["comp_any"] for r in scaf)
        n = len(hits) or 1
        out.append(f"| {m} | {len(hits)} | {len(scaf)} | {cs} | {100*cs/n:.1f}% | "
                   f"{ca} | {100*ca/n:.1f}% |")

    out.append("\n**The ceiling, stated up front.** If *every* scaffolded post-cutoff hit "
               "were reattributed from leakage to composition, Talkie-Base's RNL@10 moves "
               "0.1984 -> 0.1873 (-5.6%); at k=100, -2.5%. This pathway cannot move the "
               "aggregate metric much. That is a sizing result about one identified "
               "route, not a correction to the metric.")
    out.append("\n**A prevalence figure that must not be reported.** Loosening the donor "
               "rule (3-char suffix / 4-char stem / edit distance <=3) makes 45% of "
               "Talkie-Base's post-cutoff hits 'donor-supported' -- but 37.6% of *all* "
               "post-cutoff items qualify, a lift of 1.20x. That number is the base rate. "
               "Only the graded rule used here carries signal (stem-donor lift 2.27x).")


# --------------------------------------------------------------------------- #
# Hand labels
# --------------------------------------------------------------------------- #
def load_judge_recall():
    """Every cloze item joined to its judge verdict and to each model's recall.

    The join key is `item_id("P", target, prefix)` -- the same content hash the
    hand-labelling sheets use, so hand annotation, judge verdicts and recall outcomes all
    address the same row without re-identification. A positional join is NOT available and
    must not be invented: batch results return in completion order, not pool order (a
    positional match against the corpus scored 0.2%).

    FIREWALL. The judge saw only the citation and the target -- never a prediction, a rank,
    a model identity or a stratum. That is exactly what licenses joining its verdict to
    recall here: selection is prompt-side, the outcome is model-side, and neither instrument
    saw the other. Reversing that -- tuning the judge against recall -- would make every
    table below circular.
    """
    from evals.cloze import extract_prefix
    from scaffold_subset import item_id

    vpath = ANALYSIS / "judge" / "verdicts.jsonl"
    if not vpath.exists():
        return None, None
    V = {}
    for line in vpath.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            V[r["item_id"]] = r["judge"]

    out, models = {}, []
    for model, _ in MODELS:
        path = RESULTS / f"cloze_{model}_details.csv"
        if not path.exists():
            continue
        models.append(model)
        recs = []
        with open(path) as f:
            for r in csv.DictReader(f):
                t = (r.get("target_word") or "").strip().lower()
                if not t:
                    continue
                pre = extract_prefix(r.get("text") or "", t)
                if pre is None:
                    continue
                j = V.get(item_id("P", t, pre))
                if j is None:              # one of the 28 items with no usable verdict
                    continue
                recs.append({"judge": j, "future": r["is_future"] == "1",
                             "r10": r["correct@10"] == "1",
                             "r100": r["correct@100"] == "1"})
        out[model] = recs
    return out, models


def _rate(recs, pred, k):
    s = [x for x in recs if pred(x)]
    return (sum(x[k] for x in s) / len(s), len(s)) if s else (None, 0)


def _judge_cells(recs, isf, k):
    """(a, b, c, d) for scaffolded x recall within one stratum, in assoc's convention.

    `unsure` verdicts are excluded, as they are everywhere else in this analysis: the
    contrast is between a judged scaffold and a judged absence of one, and an abstention
    is neither. Returns None if either verdict group is empty.
    """
    def counts(verdict):
        s = [x for x in recs
             if x["future"] is isf and x["judge"]["verdict"] == verdict]
        hits = sum(x[k] for x in s)
        return hits, len(s) - hits
    (a, c), (b, d) = counts("scaffolded"), counts("not_scaffolded")
    return None if (a + c) == 0 or (b + d) == 0 else (a, b, c, d)


def table_judge_ladder(out):
    """View 0 -- the corpus-scale result: recall by judged scaffolding, with two controls.

    Reported as a difference-in-differences rather than as a single lift. A restricted
    model recalling more post-cutoff words when the prompt supplies recruitable material is
    evidence of composition only if the same prompts do not help it equally on words it
    already has. The in-cutoff stratum is that within-model control; Talkie-Web, which has
    seen the whole period and needs to compose nothing, is the across-model control.
    """
    data, models = load_judge_recall()
    if not data:
        out.append("## 0. Judge-graded recall\n\n_No `analysis/judge/verdicts.jsonl` — run "
                   "`scaffold_judge.py --collect all` first._\n")
        return

    out.append("## 0. Recall by judged scaffolding, whole corpus\n")
    out.append("Every cloze item judged against the published rubric, with no pre-filter. "
               "The judge saw the citation and the target and nothing else.\n")

    for k, klab in (("r100", "@100"), ("r10", "@10")):
        for isf, slab in ((True, "post-cutoff"), (False, "in-cutoff (control)")):
            out.append("")
            out.append("**recall%s, %s**" % (klab, slab))
            out.append("")
            out.extend(_hdr(models, ["verdict"]))
            for g in ("not_scaffolded", "unsure", "scaffolded"):
                cells = []
                for m in models:
                    v, n = _rate(data[m], lambda x, g=g, i=isf:
                                 x["future"] is i and x["judge"]["verdict"] == g, k)
                    cells.append("%.3f (n=%d)" % (v, n) if v is not None else "—")
                out.append("| %s | %s |" % (g, " | ".join(cells)))

    out.append("")
    out.append("**Two-way association** — λ, the log odds ratio of scaffolded against "
               "not-scaffolded on recall, within one stratum, tested by Dunning's G². "
               "This says scaffolding is associated with recall; it does not distinguish "
               "composition from scaffolded prompts simply being easier prompts.")
    out.append("")
    out.extend(_hdr(models, ["recall", "stratum"]))
    for k, klab in (("r100", "@100"), ("r10", "@10")):
        for isf, slab in ((True, "post-cutoff"), (False, "in-cutoff")):
            cells = []
            for m in models:
                t = _judge_cells(data[m], isf, k)
                if t is None:
                    cells.append("—")
                    continue
                l, orr, (lo, hi), g2 = AS.association(*t)
                cells.append("%.3f / OR %.2f [%.2f, %.2f] G²=%.1f%s"
                             % (l, orr, lo, hi, g2, AS.stars(g2)))
            out.append("| %s | %s | %s |" % (klab, slab, " | ".join(cells)))

    out.append("")
    out.append("**Three-way interaction** — λ₃ = λ(post-cutoff) − λ(in-cutoff): whether "
               "the association is *stronger* on words the model cannot have seen. This "
               "is the estimand. Its null is homogeneous association — one odds ratio in "
               "both strata — not an odds ratio of 1, and it is tested by the "
               "likelihood-ratio G² against the saturated 2×2×2 on 1 df. A model that "
               "composes should show a larger scaffolding benefit post-cutoff; a model "
               "that has seen everything should show the same benefit either side, i.e. "
               "λ₃ about 0.")
    out.append("")
    out.extend(_hdr(models, ["recall"]))
    for k, klab in (("r100", "@100"), ("r10", "@10")):
        cells = []
        for m in models:
            post, incut = _judge_cells(data[m], True, k), _judge_cells(data[m], False, k)
            if post is None or incut is None:
                cells.append("—")
                continue
            l3, orr, (lo, hi), g2 = AS.interaction(post, incut)
            cells.append("%.3f / OR %.2f [%.2f, %.2f] G²=%.2f%s"
                         % (l3, orr, lo, hi, g2, AS.stars(g2)))
        out.append("| %s | %s |" % (klab, " | ".join(cells)))

    out.append("")
    out.append("**Material alone, without recruitment** — post-cutoff recall@100 by the "
               "`material` field only. Shown because it is the closest analogue of the "
               "earlier orthographic grade, and because it does *not* reproduce that "
               "grade's ordering: judged over the whole corpus, bare stem material barely "
               "moves recall above `none`, and affix material sits above it. Material is "
               "not what carries the effect; recruitment is. The earlier ordering was "
               "measured on a pre-filtered pool and does not survive the full corpus.")
    out.append("")
    out.extend(_hdr(models, ["material"]))
    for g in ("none", "affix", "stem", "both"):
        cells = []
        for m in models:
            v, n = _rate(data[m], lambda x, g=g: x["future"]
                         and x["judge"]["material"] == g, "r100")
            cells.append("%.3f (n=%d)" % (v, n) if v is not None else "—")
        out.append("| %s | %s |" % (g, " | ".join(cells)))
    out.append("")


def load_hand():
    """Sheet A labels, joined to the KEY. Empty if unlabelled.

    Reads the JSONL annotation log rather than the sheet's blank columns, because
    `label_neighborhoods.py` writes there (last record per id wins). The previous version
    of this function read `analysis/scaffold_labels.csv`, which was deleted when Sheet A
    replaced it -- so section 5 silently degraded to a placeholder telling the reader to
    fill columns that no longer existed.

    Both axes are COUNTS (0, or 2..10 -- a group needs two members), not booleans. No
    threshold is applied here: `table_confusion` sweeps k, because the point of counting
    was to keep the cut movable.
    """
    log = ANALYSIS / "hand_labeling" / "neighborhood_A.jsonl"
    if not log.exists() or not KEY.exists():
        return {}
    key = {r["id"]: r for r in csv.DictReader(open(KEY))}
    recs = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # torn final line from a hard kill
        recs[r["id"]] = r

    hand = {}
    for rid, r in recs.items():
        f, m = r.get("form"), r.get("meaning")
        if f is None or m is None:        # unsure on either axis, or half-answered
            continue
        hand[rid] = {**key.get(rid, {}),
                     "hand_form_n": int(f), "hand_meaning_n": int(m),
                     "theme": r.get("theme", ""), "notes": r.get("notes") or ""}
    return hand


def table_confusion(rows, hand, out):
    """View 5 -- audit the automatic form rule against the blind Sheet A labels.

    TWO MEASURES, NOT ONE MEASURE TWICE. The automatic side counts form-variants **of the
    target** (`n_attempts`, from `form_variants(target, neighbors, ...)`). Sheet A counts
    the largest group sharing form **among the candidates**, with the target hidden. These
    are related but not the same construct, and the difference is deliberate: a
    neighborhood that enumerates a paradigm and misses the target is the informative case,
    and a target-relative measure scores it as a null. Read the table below as agreement
    between two views of formal structure, never as the hand labels validating the rule.

    The cut is swept rather than chosen. Both sides are counts, so fixing one k here would
    reintroduce exactly the threshold that counting was meant to keep movable.
    """
    out.append("\n\n## 5. Audit: the automatic form rule vs the Sheet A labels\n")
    if not hand:
        out.append("Sheet A (`analysis/neighborhood_A_cohesion.csv`, logged to "
                   "`analysis/hand_labeling/neighborhood_A.jsonl`) is **not yet "
                   "labelled**, so the automatic rule is unaudited and every number above "
                   "should be read as provisional. Annotate with "
                   "`local/bin/python label_neighborhoods.py` — two counts per "
                   "neighborhood, FORM then MEANING — then re-run.\n")
        out.append("This audit is the point of the hand pass: the grading rule is a crude "
                   "orthographic heuristic and the paper should report what it captures "
                   "rather than presuppose it captures a morphological relation.")
        return

    index = {(r["model"], r["target_word"], r["rank"]): r for r in rows}
    pairs = []
    for hid, h in sorted(hand.items()):
        k = (h.get("model"), h.get("target_word"), int(h.get("rank", -1)))
        r = index.get(k)
        if r is None:
            continue
        pairs.append((hid, h, r, int(r["n_attempts"]), h["hand_form_n"]))

    out.append(f"Sheet A labels: **{len(hand)}** neighborhoods, **{len(pairs)}** joined "
               "to the automatic table.\n")
    out.append("The automatic measure is target-relative (`n_attempts`); the hand measure "
               "is candidate-relative (`form_n`). Agreement is reported across thresholds "
               "rather than at one cut, because both sides are counts.\n")

    out.append("| k | auto≥k & hand≥k | auto≥k only | hand≥k only | neither | agreement |")
    out.append("|---|---|---|---|---|---|")
    for k in (1, 2, 3, 5):
        tp = sum(1 for *_, a, hh in pairs if a >= k and hh >= k)
        fp = sum(1 for *_, a, hh in pairs if a >= k and hh < k)
        fn = sum(1 for *_, a, hh in pairs if a < k and hh >= k)
        tn = sum(1 for *_, a, hh in pairs if a < k and hh < k)
        n = tp + fp + fn + tn
        out.append(f"| {k} | {tp} | {fp} | {fn} | {tn} | "
                   + (f"{(tp + tn) / n:.3f} |" if n else "— |"))

    out.append("\n### The hand measure alone: signature by threshold\n")
    out.append("What no surface tool sees — whether something semantically coherent was "
               "there instead. Absence of formal structure is evidence of retrieval only "
               "when competitors are present, and only the hand labels carry that.\n")
    models = sorted({h.get("model") for h in hand.values() if h.get("model")})
    out.append("| k | signature | " + " | ".join(models) + " |")
    out.append("|---" * (len(models) + 2) + "|")
    for k in (1, 3, 5):
        for cls, pred in (("compositional", lambda f, m, k=k: f >= k and m < k),
                          ("retrieval", lambda f, m, k=k: f < k and m >= k),
                          ("both", lambda f, m, k=k: f >= k and m >= k),
                          ("neither", lambda f, m, k=k: f < k and m < k)):
            cells = [str(sum(1 for h in hand.values()
                             if h.get("model") == mo
                             and pred(h["hand_form_n"], h["hand_meaning_n"])))
                     for mo in models]
            out.append(f"| {k} | {cls} | " + " | ".join(cells) + " |")

    misses = [(hid, h, r, a, hh) for hid, h, r, a, hh in pairs
              if (a >= 3) != (hh >= 3)]
    if misses:
        out.append(f"\n### Disagreements at k=3 ({len(misses)})\n")
        for hid, h, r, a, hh in misses[:25]:
            fv = "|".join(filter(None, [r["attempts"], r["echoes"]]))
            out.append(f"- `{hid}` **{h.get('target_word')}** ({h.get('model')}) — "
                       f"auto {a} ({fv or 'none'}) vs hand {hh}"
                       + (f" — *{h['notes']}*" if h.get("notes") else ""))
        if len(misses) > 25:
            out.append(f"- ... and {len(misses) - 25} more")


def load_outcome_judge(model="claude-sonnet-5"):
    """Neighborhood judge verdicts, keyed by `nbr_id`. Empty if the run has not happened.

    Counts are stored pre-derived (`form_n`, `meaning_n`) by `neighborhood_judge.py
    --collect`, which coerces a returned singleton to 0 -- a group of one is not a group,
    and `form_n == 0` has to keep meaning exactly what the hand instrument means by it.
    """
    path = ANALYSIS / "judge_nbr" / f"verdicts_{model}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["nbr_id"]] = r
    return out


def table_outcome_judge(rows, out, model="claude-sonnet-5"):
    """View 6 -- the outcome measure at scale, crossed against the scaffold verdict.

    This is the join the whole design exists to permit. Selection is judged from the prompt
    with the predictions hidden; the outcome is judged from the predictions with the prompt
    hidden; neither instrument can see the other's evidence. Both are reported across
    thresholds because both axes are counts and no cut is privileged.

    NOT YET REPORTABLE without the cross-model check. Both judges run on the same model and
    can share lexical priors even though they share no evidence. See
    `neighborhood_judge_rubric.json` -> withheld.correlated_instruments, and
    `neighborhood_judge.py --cross-check`.
    """
    nbr = load_outcome_judge(model)
    out.append("\n\n## 6. Neighborhood shape by scaffold verdict (judged)\n")
    if not nbr:
        out.append(f"_No `analysis/judge_nbr/verdicts_{model}.jsonl` — run "
                   f"`neighborhood_judge.py --submit --scope postcutoff` then `--collect "
                   f"all`._\n")
        return

    verdicts = {}
    vpath = ANALYSIS / "judge" / "verdicts.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                verdicts[j["item_id"]] = j["judge"]["verdict"]

    # `meaning_n` is gone: the outcome judge was reduced to a single form axis, so a record
    # carries `form_n` only. The meaning and form-dominant rows this view used to print
    # described a two-axis instrument that no longer exists and are not reconstructible.
    joined = []
    for r in nbr.values():
        v = verdicts.get(r["item_id"])
        if v:
            joined.append((r["model"], v, r["form_n"]))
    if not joined:
        out.append("_Neighborhood verdicts present but none joined to a scaffold "
                   "verdict; check that both runs cover the same scope._\n")
        return

    models = sorted({m for m, *_ in joined})
    out.append(f"{len(nbr)} neighborhoods judged on `{model}`, {len(joined)} joined to a "
               "scaffold verdict. Post-cutoff scope.\n")
    out.append("Share of neighborhoods whose form group reaches *k*, by whether the "
               "prompt was judged scaffolded, with λ and G² for the association. This is "
               "the post-cutoff two-way association only; the three-way interaction that "
               "is the estimand needs the in-cutoff arm too, and is reported by "
               "`neighborhood_judge.py --did`.\n")

    # k = 2 not 1: `form_n` is judge-assigned membership under code-enforced connectivity,
    # so it is never 1 -- the realised distribution is {0, 2, 3, ...}. A k=1 row would be
    # a duplicate of k=2 that looked like a separate threshold.
    for k in (2, 3, 5):
        out.append(f"\n**k = {k}**\n")
        out.extend(_hdr(models, ["measure"]))
        rates, assoc_cells = [], []
        for mo in models:
            counts = {}
            for verdict in ("scaffolded", "not_scaffolded"):
                s = [f for md, v, f in joined if md == mo and v == verdict]
                counts[verdict] = (sum(f >= k for f in s), len(s) - sum(f >= k for f in s))
            (a, c), (b, d) = counts["scaffolded"], counts["not_scaffolded"]
            rates.append(
                f"{a / (a + c):.3f} (n={a + c}) vs {b / (b + d):.3f} (n={b + d})"
                if (a + c) and (b + d) else "—")
            if (a + c) and (b + d):
                l, orr, (lo, hi), g2 = AS.association(a, b, c, d)
                assoc_cells.append("%.3f / OR %.2f [%.2f, %.2f] G²=%.1f%s"
                                   % (l, orr, lo, hi, g2, AS.stars(g2)))
            else:
                assoc_cells.append("—")
        out.append("| scaffolded vs not | " + " | ".join(rates) + " |")
        out.append("| λ (two-way) | " + " | ".join(assoc_cells) + " |")

    out.append("\n**Not reportable until the cross-model check has run.** Selection and "
               "outcome are judged by the same model here. They share no evidence — the "
               "outcome judge never sees the prompt — but they can share lexical priors, "
               "which would manufacture an association that is not in the data. Run "
               "`neighborhood_judge.py --cross-check --model claude-opus-5` and report the "
               "agreement alongside this table.\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0,
                    help="print N scaffolded post-cutoff neighborhoods with AUTO labels")
    args = ap.parse_args()

    print("loading subset ...", file=sys.stderr)
    rows = load_subset()
    models = [m for m, _ in MODELS if any(r["model"] == m for r in rows)]

    if args.dump:
        shown = 0
        for r in rows:
            if r["grade"] == "NONE" or not r["is_future"]:
                continue
            print(f"\n[{r['model']}] {r['target_word']} ({r['year']}) grade={r['grade']} "
                  f"rank={r['rank']}  strict={r['comp_strict']} any={r['comp_any']}")
            print(f"  attempts : {r['attempts'] or '(none)'}")
            print(f"  echoes   : {r['echoes'] or '(none)'}")
            print(f"  top10    : {'|'.join(r['neighbors'])}")
            shown += 1
            if shown >= args.dump:
                break
        return

    hand = load_hand()
    out = ["# Scaffolded cloze characterization: composition vs recall in the neighborhood",
           "",
           "*Generated by `scaffold_analysis.py` over `analysis/scaffold_subset.csv` "
           "(built by `scaffold_subset.py`). AUTO labels use the dictionary-free form "
           "relatedness union from `neighborhood_analysis.py`; HAND labels come from the "
           "blind sheet `analysis/scaffold_labels.csv`.*",
           "",
           "**Scaffolding is the selection criterion, not a finding.** Composition needs "
           "materials in context; retrieval does not. So we restrict to cloze items whose "
           "prefix supplies form material and ask what happens there. Grades are named "
           "descriptively — stem material outperforms affix material, so there is no "
           "high/mid/low ordering.",
           ""]

    table_judge_ladder(out)
    table_ladder(rows, models, out)
    table_recruitment(rows, models, out)
    table_headline(rows, models, out)
    table_breadth(rows, models, out)
    table_accounting(rows, models, out)
    table_confusion(rows, hand, out)
    table_outcome_judge(rows, out)

    ANALYSIS.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}")
    print("\n".join(out[:4]))
    for line in out:
        if line.startswith("## ") or line.startswith("| "):
            print(line)


if __name__ == "__main__":
    main()
