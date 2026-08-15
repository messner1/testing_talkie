#!/usr/bin/env python3
"""Score the scaffold-perturbation experiment.

The question is whether removing the donor removes the paradigm-shaped group from the
model's predictions. Every item appears both before and after the edit, so each is
compared with itself: an item that showed a group before and does not after has
COLLAPSED; one that showed none before and does after has APPEARED. Items unchanged in
either direction carry no information about the edit, so the test is whether collapses
exceed appearances by more than chance -- an exact binomial on the discordant pairs.

BEFORE comes from the committed cloze decode, judged once by the neighbourhood
instrument. AFTER comes from the perturbation decode. Survival applies the same
connectivity enforcement the baseline passed through to whichever judged members remain
in the edited predictions, so a group that loses its linking member collapses even when
two members are still present.

Reads perturb_{a,b,c}/composition_{model}_detailed.csv, perturb_metadata.jsonl and the
judged baselines in analysis/judge_nbr/. Prints; writes nothing.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import neighborhood_judge as NJ

ANALYSIS = Path("analysis")
JUDGE_NBR = ANALYSIS / "judge_nbr"
MODELS = ("talkie-base", "talkie-web", "typewriter")

# Whether an item is post-cutoff is a fact about the model, not about the item. The frozen
# sample labels each item against the LOOSEST cutoff (1913), because that is what defines
# the pool; carrying that label into scoring puts items dated 1914-1930 in the post-cutoff
# column for the 1930 models, where they are in-cutoff. 47 items for Talkie-Base and 50
# for Talkie-Web sit in that window. Talkie-Web is ahistorical and its 1930 entry is a
# label of convenience, applied so the same items are classified the same way as for
# Talkie-Base -- see materials.md.
CUTOFF = {"talkie-base": 1930, "talkie-web": 1930, "typewriter": 1913}


def stratum_for(model, year):
    try:
        return "post" if int(year) > CUTOFF[model] else "incutoff"
    except (TypeError, ValueError):
        return "incutoff"          # undated items are treated as in-cutoff throughout


ARMS = ("donor_substituted", "donor_deleted", "placebo")
BATCHES = ("a", "b", "c")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_metadata():
    """(batch_file, target.lower()) -> item metadata.

    Keyed on the pair because the perturbed prefix hashes to a different item_id than the
    one the baseline was judged under, and because a target is unique only within a batch.
    """
    out = {}
    with open("perturb_metadata.jsonl") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)
                out[(m["batch"], m["target"].lower())] = m
    return out


def load_baselines():
    """(model, item_id) -> the judged verdict for the unedited neighbourhood."""
    out = {}
    for name in ("verdicts_claude-sonnet-5_context.jsonl",
                 "verdicts_claude-sonnet-5_context_incutoff.jsonl"):
        path = JUDGE_NBR / name
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[(r["model"], r["item_id"])] = r
    return out


def survival(baseline, new_words):
    """Enforced size of whatever the judged group retains in `new_words`.

    The same enforcement the baseline passed through -- largest connected component,
    inflection-only groups discounted -- applied to the surviving members. Members absent
    from the new predictions are dropped before enforcement, since form_size resolves
    membership against the word list it is given.
    """
    group = [w for w in (baseline["judge"].get("form_group") or [])
             if w.lower() in {x.lower() for x in new_words}]
    if len(group) < 2:
        return 0
    return NJ.form_size({"form_group": group}, new_words)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def binom_two_sided(b, c):
    """Exact two-sided binomial on the discordant pairs, p = 0.5.

    McNemar's test in its exact form. With b + c often under 40 the chi-square
    approximation is not safe, and the exact test costs nothing at this size.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    """Wilson interval -- behaves at the boundaries, where a normal interval does not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def collect():
    """One record per (model, arm, item): had a group before, has one after."""
    meta = load_metadata()
    base = load_baselines()
    csv.field_size_limit(10 ** 7)
    rows, unmatched = [], Counter()
    for batch in BATCHES:
        d = Path(f"perturb_{batch}")
        if not d.exists():
            continue
        for model in MODELS:
            path = d / f"composition_{model}_detailed.csv"
            if not path.exists():
                continue
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    m = meta.get((f"perturb_batch_{batch}.jsonl",
                                  r["target_word"].lower()))
                    if m is None:
                        unmatched["no_metadata"] += 1
                        continue
                    b = base.get((model, m["item_id"]))
                    if b is None:
                        unmatched["no_judged_baseline"] += 1
                        continue
                    new_words = r["top_10_words"].split("|")
                    rows.append({
                        "model": model, "arm": r["context_level"],
                        "item_id": m["item_id"],
                        "stratum": stratum_for(model, m["year"]),
                        "pool_stratum": m["stratum"],
                        "material": m["material"], "recruitment": m["recruitment"],
                        "before": int(b.get("form_n", 0)) >= 2,
                        "after": survival(b, new_words) >= 2,
                        "before_n": int(b.get("form_n", 0)),
                        "after_n": survival(b, new_words),
                    })
    return rows, unmatched


def table(rows):
    """(a, b, c, d): unchanged-present, collapsed, appeared, unchanged-absent."""
    t = Counter((r["before"], r["after"]) for r in rows)
    return t[(True, True)], t[(True, False)], t[(False, True)], t[(False, False)]


def report(rows, stratum, title):
    sel = [r for r in rows if r["stratum"] == stratum]
    print(f"\n{title}")
    print(f"  {'model':14}{'arm':20}{'a':>5}{'b':>5}{'c':>5}{'d':>5}"
          f"{'before':>9}{'after':>8}{'p':>10}")
    for model in MODELS:
        for arm in ARMS:
            sub = [r for r in sel if r["model"] == model and r["arm"] == arm]
            if not sub:
                continue
            a, b, c, d = table(sub)
            n = len(sub)
            pre, post = (a + b) / n, (a + c) / n
            p = binom_two_sided(b, c)
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"  {model:14}{arm:20}{a:>5}{b:>5}{c:>5}{d:>5}"
                  f"{pre:>9.3f}{post:>8.3f}{p:>10.4f} {star}")
    print("    a unchanged-present   b COLLAPSED   c appeared   d unchanged-absent")
    print("    p: exact two-sided binomial on the b/c discordant pairs")


def by_material(rows, stratum):
    print(f"\nCollapse rate by scaffold material -- {stratum}, substitution arm")
    print(f"  {'model':14}{'material':10}{'n':>5}{'collapsed':>11}{'rate':>8}{'95% CI':>16}")
    for model in MODELS:
        sub = [r for r in rows if r["stratum"] == stratum and r["model"] == model
               and r["arm"] == "donor_substituted" and r["before"]]
        for mat in ("stem", "affix", "both"):
            s = [r for r in sub if r["material"] == mat]
            if not s:
                continue
            k = sum(1 for r in s if not r["after"])
            lo, hi = wilson(k, len(s))
            print(f"  {model:14}{mat:10}{len(s):>5}{k:>11}{k/len(s):>8.3f}"
                  f"   [{lo:.2f}, {hi:.2f}]")
    print("    conditioned on a group being present before the edit;"
          " descriptive, not a moderation test")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", action="store_true", help="add the material breakdown")
    args = ap.parse_args()

    rows, unmatched = collect()
    if not rows:
        raise SystemExit("no decoded rows joined -- is perturb_{a,b,c}/ present?")
    print(f"joined {len(rows)} (model, arm, item) records"
          + (f"; unmatched {dict(unmatched)}" if unmatched else ""))
    print(f"  arms: {dict(Counter(r['arm'] for r in rows))}")

    report(rows, "post", "POST-CUTOFF -- the population the claim is about")
    report(rows, "incutoff", "IN-CUTOFF -- the matched comparison stratum")
    if args.material:
        by_material(rows, "post")


if __name__ == "__main__":
    main()
