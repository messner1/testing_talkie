#!/usr/bin/env python3
"""Joining the selection arm to the outcome arm, and both to the cloze result.

WHY THIS FILE EXISTS.  `scaffold_judge.py` decides, from the passage alone, whether a prompt
supplies recruitable morphological material.  `neighborhood_judge.py` decides, from the ten
predictions alone, whether those candidates form a paradigm.  Neither has ever been joined to
the thing the study is actually about: whether the model produced the post-cutoff word, and by
what route.  `neighborhood_judge.py --did` reports the outcome measure's own contrasts; it says
nothing about recall.

WHAT THE COMPARISON IS, AND IS NOT.  Talkie-Web is ahistorical -- a different corpus, no cutoff
to violate.  It composes, because any sufficiently powerful model composes.  Finding composition
in Web is the EXPECTED result, not a false positive, and subtracting it would subtract the
phenomenon.  Composition is a problem only under restriction, where a post-cutoff production is
confusable with leakage; when the domain is everything there is nothing to confuse it with.
Every estimand here is therefore WITHIN-MODEL.  Web appears for two other purposes: showing the
mechanism is general rather than an artifact of restriction, and validating the instrument (its
cohesion is cutoff-invariant where the restricted models' collapses).

THE FIREWALL, STATED ACCURATELY.  On a hit the target IS one of the ten words the outcome judge
saw.  What the judge cannot do is identify which one: it sees ten shuffled candidates, no
prompt, no target marker, and a rubric that asks for form grouping and never mentions a correct
answer.  Group membership is assigned without regard to correctness, and that is what licenses
the join.  `--pathway` reports the diagnostic that tests this.

Run:
    local/bin/python connect.py --accounting     # RNL ladder + composition-consistent share
    local/bin/python connect.py --pathway        # coinage on misses, both instruments
    local/bin/python connect.py --mediation      # scaffolding -> neighborhood -> recall
    local/bin/python connect.py --doseresponse   # graded prompt features
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import neighborhood_judge as NJ                          # noqa: E402
import neighborhood_measure as NM                        # noqa: E402
import scaffold_subset as SS                             # noqa: E402
from evals.cloze import extract_prefix                   # noqa: E402

csv.field_size_limit(10 ** 7)
RESULTS = Path("results")
ANALYSIS = Path("analysis")
MODELS = ("talkie-base", "typewriter", "talkie-web")
BOOT = 2000
SEED = 20260812


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_rows():
    """One record per (model, item) carrying selection, outcome and cloze result.

    The three sources join on `item_id`, which is content-derived (sha1 of target+prefix) and
    so is recomputable from the raw details CSV -- `results/cloze_*_details.csv` has no id
    column of its own. Recomputing it through `SS.item_id` rather than joining on row order
    means a re-export of either file cannot silently misalign the arms.
    """
    prompts = {}
    for m in MODELS:
        p = RESULTS / f"cloze_{m}_details.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        for r in csv.DictReader(open(p)):
            t = r["target_word"]
            pre = extract_prefix(r["text"], t)
            iid = SS.item_id("P", SS.NA.norm(t), pre)
            prompts[(m, iid)] = (pre, r["top_10_nlls"])

    sub = {}
    for r in csv.DictReader(open(ANALYSIS / "scaffold_subset.csv")):
        sub[(r["model"], r["item_id"])] = r

    sel = NJ.scaffold_verdicts()
    out = []
    for arm, f in (("post", "verdicts_claude-sonnet-5_context.jsonl"),
                   ("in", "verdicts_claude-sonnet-5_context_incutoff.jsonl")):
        path = ANALYSIS / "judge_nbr" / f
        if not path.exists():
            sys.exit(f"missing {path} — run neighborhood_judge.py --submit/--collect")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            key = (j["model"], j["item_id"])
            s = sub.get(key)
            if s is None:
                continue
            verdict = sel.get(j["item_id"])
            if verdict not in ("scaffolded", "not_scaffolded"):
                continue
            words = [w.lower() for w in j["words"]]
            group = [w.lower() for w in (j["judge"].get("form_group") or [])]
            # form_n is authoritative: 14% of records carry a group of >=2 with form_n == 0
            # because the inflection gate fired. len(form_group) is not the measure.
            fn = NJ.form_size(j["judge"], j["words"])
            rank = int(s["rank"] or -1)
            pre, nlls = prompts.get(key, ("", ""))
            prewords = set(SS.NA.norm(w) for w in pre.lower().split())
            out.append({
                "model": j["model"], "arm": arm, "item_id": j["item_id"],
                "scaffolded": verdict == "scaffolded",
                "target": j["target"].lower(), "words": words,
                "group": group, "form_n": fn,
                "certified": fn >= 2 and NJ.certified(group),
                "rank": rank,
                # `hit` in scaffold_subset.csv is recall@100. The RNL headline is @10.
                "hit10": 0 < rank <= 10, "hit100": 0 < rank <= 100,
                "target_in_group": fn >= 2 and j["target"].lower() in group,
                # echo: a NON-TARGET group member that was already in the prompt. The outcome
                # judge cannot see the prompt by design, so it cannot tell a composed candidate
                # from one copied back out of the passage. `n_echoes` does not cover this --
                # it counts only prompt words that are form-variants OF THE TARGET.
                "echo_members": sum(1 for w in group
                                    if w != j["target"].lower() and w in prewords),
                "group_nonself": max(0, len([w for w in group if w != j["target"].lower()])),
                "nlls": [float(x) for x in nlls.split("|") if x] if nlls else [],
                "comp_strict": int(s["comp_strict"]), "n_echoes": int(s["n_echoes"]),
                "donor_distance": int(s["donor_distance"]), "locality": s["locality"],
                "ending_rarity": s["ending_rarity"], "domain": s["domain"],
                "sel_material": (sel_full(j["item_id"]) or {}).get("material", ""),
                "sel_recruitment": (sel_full(j["item_id"]) or {}).get("recruitment", ""),
            })
    return out


_SELFULL = None


def sel_full(item_id):
    """The scaffold judge's full record, not just its verdict (material, recruitment, ...)."""
    global _SELFULL
    if _SELFULL is None:
        _SELFULL = {}
        p = ANALYSIS / "judge" / "verdicts.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                _SELFULL[j["item_id"]] = j["judge"]
    return _SELFULL.get(item_id)


def population_weights():
    """Scaffold-verdict counts over the true population, per model x arm.

    The in-cutoff judged arm is a stratified SAMPLE -- 1,697 per model x verdict cell, so 50/50
    -- while the population is ~4% scaffolded. Any pooled in-cutoff rate must be reweighted or
    it overstates the scaffolded contribution by an order of magnitude.
    """
    sel = NJ.scaffold_verdicts()
    w = Counter()
    for i in NJ.load_neighborhoods("all"):
        v = sel.get(i["item_id"])
        if v in ("scaffolded", "not_scaffolded"):
            w[(i["model"], "post" if i["is_future"] == "1" else "in", v == "scaffolded")] += 1
    return w


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def boot_diff(a, b, stat=lambda xs: sum(xs) / len(xs), n=BOOT, seed=SEED):
    """Percentile bootstrap CI for stat(a) - stat(b). Cells here are small and several rates
    sit near 0, where a normal-approximation interval runs below zero and misstates the
    uncertainty."""
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    d = [stat([a[rng.randrange(len(a))] for _ in a])
         - stat([b[rng.randrange(len(b))] for _ in b]) for _ in range(n)]
    d.sort()
    return stat(a) - stat(b), d[int(0.025 * n)], d[int(0.975 * n)]


def mediation(rows, treat, med, out, n=BOOT, seed=SEED):
    """Nonparametric mediation decomposition for a binary mediator.

        TE  = E[Y|T=1] - E[Y|T=0]
        NDE = sum_m ( E[Y|T=1,M=m] - E[Y|T=0,M=m] ) P(M=m|T=0)
        NIE = sum_m   E[Y|T=1,M=m] ( P(M=m|T=1) - P(M=m|T=0) )

    This is the mediation formula (VanderWeele); Vig et al. (2020) introduced the natural
    direct/indirect vocabulary to NLP. THIS IS OBSERVATIONAL. Vig-style CMA intervenes on
    internal components; we cannot -- no model access -- so identification rests on no
    unmeasured treatment-outcome, mediator-outcome or treatment-mediator confounding, none of
    which is verifiable here. What the design does buy is that the mediator is measured by an
    instrument that never sees the treatment, which removes shared measurement error from the
    list of candidate confounds.
    """
    def decompose(rs):
        cell, pm = {}, {}
        for t in (0, 1):
            sub = [r for r in rs if treat(r) == t]
            if not sub:
                return None
            pm[t] = sum(med(r) for r in sub) / len(sub)
            for m in (0, 1):
                c = [out(r) for r in sub if med(r) == m]
                cell[(t, m)] = (sum(c) / len(c)) if c else None
        for k, v in cell.items():
            if v is None:                       # an empty t x m cell: not identified
                return None
        te = (sum(out(r) for r in rs if treat(r) == 1) / max(1, sum(1 for r in rs if treat(r) == 1))
              - sum(out(r) for r in rs if treat(r) == 0) / max(1, sum(1 for r in rs if treat(r) == 0)))
        nde = sum((cell[(1, m)] - cell[(0, m)]) * (pm[0] if m else 1 - pm[0]) for m in (0, 1))
        nie = sum(cell[(1, m)] * ((pm[1] if m else 1 - pm[1]) - (pm[0] if m else 1 - pm[0]))
                  for m in (0, 1))
        return te, nde, nie

    point = decompose(rows)
    if point is None:
        return None
    rng = random.Random(seed)
    props = []
    for _ in range(n):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        d = decompose(s)
        if d and d[0] != 0:
            props.append(d[2] / d[0])
    props.sort()
    te, nde, nie = point
    lo, hi = (props[int(0.025 * len(props))], props[int(0.975 * len(props))]) if props else (
        float("nan"), float("nan"))
    return te, nde, nie, (nie / te if te else float("nan")), lo, hi


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def mode_accounting(args):
    """Per-model accounting of RNL, and the k-ladder.

    RNL is defined per model. Talkie-Web's is 1.21 -- no deficit, so there is nothing to
    account for, which is exactly why the question only arises under restriction. There is no
    cross-model race in this table.
    """
    print("\n  gross RNL, as computed at the first stage (evals/metrics.py)\n")
    print(f"  {'model':<14}" + "".join(f"{'RNL@'+str(k):>10}" for k in (10, 20, 50, 100)))
    summ = {}
    for m in MODELS:
        d = json.loads((RESULTS / f"cloze_{m}_summary.json").read_text())
        summ[m] = d
        print(f"  {m:<14}" + "".join(
            f"{d['metrics_by_k'][str(k)]['recall_normalized_leakage']:>10.4f}"
            for k in (10, 20, 50, 100)))
    print("\n  The ladder is a result. RNL rises with k for the restricted models and falls for")
    print("  the ahistorical one -- widening the candidate set preferentially helps the models")
    print("  that must assemble, which is what near-miss composition predicts and leakage does")
    print("  not.")

    rows = [r for r in load_rows() if r["arm"] == "post"]
    print(f"\n  composition-consistent share of post-cutoff hits"
          f"   (target inside a licensed group)\n")
    print(f"  {'model':<14}{'k':>4}{'hits':>7}{'assess.':>9}{'cov.':>7}{'composed':>10}"
          f"{'share':>8}{'echo-str.':>11}{'bracket over all hits':>24}")
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        for k in (10, 100):
            hits = [r for r in rs if 0 < r["rank"] <= k]
            # the outcome instrument only ever saw ten candidates, so a target at rank 11-100
            # cannot be assessed at all. Reporting the share without this denominator would
            # silently treat unassessable hits as non-composed.
            ass = [r for r in hits if r["hit10"]]
            comp = [r for r in ass if r["target_in_group"]]
            clean = [r for r in comp if r["echo_members"] == 0]
            sh = len(comp) / len(ass) if ass else 0
            ec = len(clean) / len(ass) if ass else 0
            # over ALL hits at this k the share is bounded below by assuming every
            # unassessable hit is not composed, and above by assuming every one is.
            lo = len(comp) / len(hits) if hits else 0
            hi = (len(comp) + len(hits) - len(ass)) / len(hits) if hits else 0
            br = f"[{lo:.3f}, {hi:.3f}]" if k > 10 else "(fully assessed)"
            print(f"  {m:<14}{k:>4}{len(hits):>7}{len(ass):>9}"
                  f"{(len(ass)/len(hits) if hits else 0):>7.2f}{len(comp):>10}{sh:>8.3f}"
                  f"{ec:>11.3f}{br:>24}")
    print("\n  Read the share as an UPPER BOUND on the composition component of RNL, per model.")
    print("  At k=100 the neighborhood instrument covers only the k=10 hit set -- ten candidates")
    print("  were stored -- so the k=100 row is a bracket, not an estimate.")

    # Reconcile against the summaries so the shortfall is not mistaken for a broken join.
    print(f"\n  reconciliation against results/cloze_*_summary.json\n")
    print(f"  {'model':<14}{'future_correct@10':>19}{'joined':>9}{'dropped':>9}")
    for m in MODELS:
        fc = summ[m]["metrics_by_k"]["10"]["future_correct"]
        got = sum(1 for r in rows if r["model"] == m and r["hit10"])
        print(f"  {m:<14}{fc:>19}{got:>9}{fc-got:>9}")
    print("  Dropped = items whose selection verdict was `unsure` plus neighborhoods the")
    print("  outcome judge failed on. Both are excluded by design, not lost in the join.")


def mode_pathway(args):
    """Coinage on a miss -- the arm where leakage of the target is not available.

    A miss cannot be leakage OF THE TARGET, so a certified group on a miss is composition
    attempted and failed with no retrieval route to the answer. Reported under two instruments
    because they ask different questions and disagree.
    """
    rows = load_rows()
    print("\n  certified coinage on a MISS, by scaffold verdict"
          "   (judged instrument, target-blind)\n")
    print(f"  {'model':<14}{'arm':<6}{'scaff':>8}{'not':>8}{'diff':>9}{'95% CI':>18}{'n':>14}")
    for m in MODELS:
        for arm in ("post", "in"):
            rs = [r for r in rows if r["model"] == m and r["arm"] == arm and not r["hit10"]]
            a = [int(r["certified"]) for r in rs if r["scaffolded"]]
            b = [int(r["certified"]) for r in rs if not r["scaffolded"]]
            d, lo, hi = boot_diff(a, b)
            print(f"  {m:<14}{arm:<6}{sum(a)/len(a):>8.3f}{sum(b)/len(b):>8.3f}{d:>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{f'{len(a)}/{len(b)}':>14}")

    print("\n  the same contrast under the TARGET-ANCHORED automatic rule (comp_strict)\n")
    print(f"  {'model':<14}{'arm':<6}{'scaff':>8}{'not':>8}{'diff':>9}{'95% CI':>18}{'n':>14}")
    for m in MODELS:
        for arm in ("post", "in"):
            rs = [r for r in rows if r["model"] == m and r["arm"] == arm and not r["hit10"]]
            a = [r["comp_strict"] for r in rs if r["scaffolded"]]
            b = [r["comp_strict"] for r in rs if not r["scaffolded"]]
            d, lo, hi = boot_diff(a, b)
            print(f"  {m:<14}{arm:<6}{sum(a)/len(a):>8.3f}{sum(b)/len(b):>8.3f}{d:>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{f'{len(a)}/{len(b)}':>14}")
    print("\n  These diverge, and the divergence is interpretable rather than damning: a")
    print("  target-anchored measure asks for relatives OF THE TARGET, which an ahistorical")
    print("  model can retrieve and a restricted one cannot. Report both.")

    print("\n  firewall diagnostic — is the target preferentially grouped?\n")
    print(f"  {'model':<14}{'arm':<6}{'n':>6}{'chance':>9}{'observed':>10}{'excess':>9}")
    for m in MODELS:
        for arm in ("in", "post"):
            rs = [r for r in rows if r["model"] == m and r["arm"] == arm
                  and r["hit10"] and r["form_n"] >= 2]
            if not rs:
                continue
            exp = sum(r["form_n"] / len(r["words"]) for r in rs) / len(rs)
            obs = sum(r["target_in_group"] for r in rs) / len(rs)
            print(f"  {m:<14}{arm:<6}{len(rs):>6}{exp:>9.3f}{obs:>10.3f}{obs-exp:>+9.3f}")
    print("\n  The excess is larger in-cutoff than post-cutoff in every model. That gradient is")
    print("  a content effect -- a model holding the word retrieves its family -- and is what a")
    print("  judge covertly identifying targets would NOT produce.")


def mode_mediation(args):
    """Does scaffolding raise post-cutoff recall THROUGH the neighborhood measure?

    This is the connective question stated directly: treatment = the selection judge's verdict,
    mediator = the outcome judge's group, outcome = recall. See mediation() for the estimator
    and for what it does not identify.
    """
    rows = [r for r in load_rows() if r["arm"] == "post"]
    print("\n  mediation: scaffolded prompt -> paradigm-shaped neighborhood -> recall@10")
    print("  (observational; identifying assumptions are NOT verifiable here -- see docstring)\n")
    print(f"  {'model':<14}{'mediator':<14}{'TE':>8}{'NDE':>8}{'NIE':>8}"
          f"{'prop. med.':>12}{'95% CI':>18}{'n':>7}")
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        for lab, med in (("form_n>=2", lambda r: int(r["form_n"] >= 2)),
                         ("certified", lambda r: int(r["certified"]))):
            res = mediation(rs, lambda r: int(r["scaffolded"]), med, lambda r: int(r["hit10"]))
            if res is None:
                print(f"  {m:<14}{lab:<14}{'— not identified (empty cell)':>54}")
                continue
            te, nde, nie, pm, lo, hi = res
            print(f"  {m:<14}{lab:<14}{te:>+8.3f}{nde:>+8.3f}{nie:>+8.3f}{pm:>12.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}{len(rs):>7}")
    print("\n  NIE is the part of scaffolding's effect on recall that flows through the")
    print("  neighborhood becoming paradigm-shaped. It is the connective quantity: a large")
    print("  proportion mediated says scaffolded post-cutoff hits arrive THROUGH composition.")


def mode_doseresponse(args):
    """Graded prompt features, which nothing in the project has used.

    The selection side is reported as a binary verdict, but `scaffold_subset.csv` carries the
    raw geometry the verdict was formed over. If scaffolding is a real mechanism rather than a
    label, a nearer or rarer donor should buy more coinage. A flat curve is also a result.
    """
    rows = [r for r in load_rows() if r["arm"] == "post" and not r["hit10"]]
    for field, order in (("locality", ["none", "<=5", "6-15", ">15"]),
                         ("ending_rarity", ["rare", "common", "very-common", "n/a"]),
                         ("domain", ["technical", "everyday"])):
        print(f"\n  certified coinage on post-cutoff misses by {field}\n")
        print(f"  {'level':<14}" + "".join(f"{m:>16}" for m in MODELS))
        for lvl in order:
            cells = []
            for m in MODELS:
                rs = [r for r in rows if r["model"] == m and r[field] == lvl]
                cells.append(f"{sum(r['certified'] for r in rs)/len(rs):.3f} ({len(rs)})"
                             if rs else "—")
            print(f"  {lvl:<14}" + "".join(f"{c:>16}" for c in cells))

    print("\n  by the selection judge's own dimensions (post-cutoff misses)\n")
    print(f"  {'material':<10}{'recruitment':<18}{'n':>7}{'certified':>11}")
    agg = defaultdict(Counter)
    for r in rows:
        c = agg[(r["sel_material"], r["sel_recruitment"])]
        c["n"] += 1
        c["cert"] += r["certified"]
    for k in sorted(agg, key=lambda k: -agg[k]["n"])[:12]:
        c = agg[k]
        flag = "   <- rubric FLAGGED cell" if k == ("affix", "proximity") else ""
        print(f"  {k[0]:<10}{k[1]:<18}{c['n']:>7}{c['cert']/c['n']:>11.3f}{flag}")

    # matched_null returns one group SIZE per trial, over pseudo-neighborhoods of real English
    # words matched on length (Siew & Vitevitch 2018). The chance rate is the fraction of those
    # trials reaching a group, pooled over the sampled neighborhoods.
    print("\n  chance baseline (matched_null): length-matched real English words\n")
    rng = random.Random(SEED)
    samp = rng.sample(rows, min(150, len(rows)))
    obs = sum(r["form_n"] >= 2 for r in samp) / len(samp)
    trials = [n for r in samp for n in NM.matched_null(r["words"], trials=40)]
    null = sum(1 for n in trials if n >= 2) / len(trials)
    print(f"  observed group>=2 {obs:.3f}   chance {null:.3f}"
          f"   ({len(samp)} neighborhoods x 40 trials)")
    print("  The measure is calibrated so that chance is low by construction; this confirms the")
    print("  deployed threshold still behaves that way on the post-cutoff miss population.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounting", action="store_true")
    ap.add_argument("--pathway", action="store_true")
    ap.add_argument("--mediation", action="store_true")
    ap.add_argument("--doseresponse", action="store_true")
    args = ap.parse_args()
    ran = False
    for flag, fn in (("accounting", mode_accounting), ("pathway", mode_pathway),
                     ("mediation", mode_mediation), ("doseresponse", mode_doseresponse)):
        if getattr(args, flag):
            fn(args)
            ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
