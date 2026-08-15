#!/usr/bin/env python3
"""Score the scaffold-perturbation experiment.

The question is whether removing the donor removes the paradigm-shaped group from the
model's predictions. Every item appears both before and after the edit, so each is
compared with itself: an item that showed a group before and does not after has
COLLAPSED; one that showed none before and does after has APPEARED. Items unchanged in
either direction carry no information about the edit, so the effect is the CONDITIONAL
odds ratio over the discordant pairs -- b/c -- tested by the likelihood-ratio G^2 for
marginal homogeneity. That is the same statistic the rest of the project reports; a
within-item design simply uses the conditional estimator, which conditions each item on
its own baseline so the item drops out.

The quantity the experiment exists to estimate is the DONOR-SPECIFIC effect: how much more
a directional arm collapses than its placebo. Prompting is fragile, so an edit anywhere
disturbs predictions; only the excess over an edit that leaves the donor intact is about
the donor. That contrast is also paired on the item -- see `vs_placebo`.

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

import assoc as AS
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
    """Within-arm effect: did the edit remove the group?

    Every item is its own control, so the estimate is the CONDITIONAL odds ratio -- b/c
    over the discordant pairs -- and the test is the likelihood-ratio G^2 for marginal
    homogeneity. Same statistic as everywhere else in the project; the only difference is
    that a within-item design uses the conditional estimator, which conditions each item on
    its own baseline so the item drops out.
    """
    sel = [r for r in rows if r["stratum"] == stratum]
    print(f"\n{title}")
    print(f"  {'model':14}{'arm':20}{'a':>5}{'b':>5}{'c':>5}{'d':>5}"
          f"{'before':>8}{'after':>7}{'lambda':>9}{'OR':>8}{'95% CI':>18}{'G^2':>9}")
    thin = []
    for model in MODELS:
        for arm in ARMS:
            sub = [r for r in sel if r["model"] == model and r["arm"] == arm]
            if not sub:
                continue
            a, b, c, d = table(sub)
            n = len(sub)
            pre, post = (a + b) / n, (a + c) / n
            l, orr, (lo, hi), g2 = AS.paired(b, c)
            if b + c < 25:
                thin.append((model, arm, b, c, binom_two_sided(b, c)))
            print(f"  {model:14}{arm:20}{a:>5}{b:>5}{c:>5}{d:>5}"
                  f"{pre:>8.3f}{post:>7.3f}{l:>9.3f}{orr:>8.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18}{g2:>9.2f} {AS.stars(g2)}")
    print("    a unchanged-present   b COLLAPSED   c appeared   d unchanged-absent")
    print("    lambda = log(b/c), the conditional odds ratio; G^2 tests marginal homogeneity")
    print("    Where c = 0 no group ever appeared, so the odds ratio is Haldane-corrected"
          " and\n    bounded below only: read it as 'no reversals', not as a magnitude."
          " G^2 is the\n    quantity to compare across rows.")
    for model, arm, b, c, p in thin:
        print(f"    thin: {model} {arm} has b+c={b + c}; exact binomial p={p:.4f} "
              f"(cross-check, not reported)")


def vs_placebo(rows, stratum, title):
    """The donor-specific effect: does the directional arm collapse MORE than the placebo?

    This is the quantity the experiment exists to estimate, and it was previously asserted
    in prose as a difference of collapse rates with no test at all. That treatment was
    wrong twice over: it was on a different scale from every other result here, and it
    compared two PAIRED proportions as if they were independent -- the same item is decoded
    under both arms.

    Pairing is on the item. Restricted to items that showed a group before the edit under
    BOTH arms, so the two conditions are compared on the same stimuli rather than on two
    different subsets. b = collapsed under the directional arm but not the placebo,
    c = the reverse; concordant items carry no information about which arm is stronger.
    """
    print(f"\n{title}")
    print(f"  {'model':14}{'arm':20}{'n':>5}{'b':>5}{'c':>5}"
          f"{'lambda':>9}{'OR':>8}{'95% CI':>18}{'G^2':>9}")
    for model in MODELS:
        idx = defaultdict(dict)
        for r in rows:
            if r["stratum"] == stratum and r["model"] == model:
                idx[r["item_id"]][r["arm"]] = r
        for arm in ("donor_substituted", "donor_deleted"):
            b = c = n = 0
            for item in idx.values():
                a_row, p_row = item.get(arm), item.get("placebo")
                if not a_row or not p_row:
                    continue
                if not (a_row["before"] and p_row["before"]):
                    continue          # nothing to lose in one arm or the other
                n += 1
                lost_arm = not a_row["after"]
                lost_pla = not p_row["after"]
                b += lost_arm and not lost_pla
                c += lost_pla and not lost_arm
            if not n:
                continue
            l, orr, (lo, hi), g2 = AS.paired(b, c)
            print(f"  {model:14}{arm:20}{n:>5}{b:>5}{c:>5}"
                  f"{l:>9.3f}{orr:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
                  f"{g2:>9.2f} {AS.stars(g2)}")
    print("    paired on the item; both arms had a group before the edit")
    print("    b = collapsed under the edit but not the placebo, c = the reverse")


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
    vs_placebo(rows, "post", "DONOR-SPECIFIC EFFECT -- post-cutoff, each arm vs its placebo")
    report(rows, "incutoff", "IN-CUTOFF -- the matched comparison stratum")
    vs_placebo(rows, "incutoff", "DONOR-SPECIFIC EFFECT -- in-cutoff")
    if args.material:
        by_material(rows, "post")


if __name__ == "__main__":
    main()
