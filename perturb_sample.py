#!/usr/bin/env python3
"""Select and freeze the item set for the scaffold-perturbation experiment.

The experiment removes the donor from a scaffolded prompt and re-decodes, so the
population is the prompts the scaffold judge called `scaffolded`. Two strata:

  POST-CUTOFF   every scaffolded item dated after 1913, the loosest of the three
                cutoffs. 182 of them. This is the whole population, not a sample.

  IN-CUTOFF     a matched comparison stratum, drawn from scaffolded items that
                already carry a judged baseline neighborhood.

The post-cutoff count is 182 and not 426. The scaffold judge reads the prompt and
nothing else, so it ran once per item rather than once per (item, model); the
121/122/182 per-model figures are three views of one nested pool. Talkie-Base and
Talkie-Web share a 1930 cutoff and see 122 of these; Typewriter's 1913 cutoff makes
its 182 a superset. The 60 items in between are post-cutoff for Typewriter and
in-cutoff for the other two -- the same stimulus on either side of a cutoff, which
is the confusability the project is about. They are reported within model, never as
a cross-model contrast.

MATCHING. The in-cutoff stratum is matched on the joint `material` x `recruitment`
cross-tabulation, not on the two marginals -- matching marginals independently does
not reproduce the joint, and these two are not independent (proximity dominates stem
material). Cells are filled to the post-cutoff count, drawn without replacement, from
items already present in the outcome-judged in-cutoff sample so that every drawn item
carries a judged baseline.

Domain is deliberately not a matching variable. Whether composition concentrates in
the technical lexicon is one of the questions the experiment asks, and matching on it
would remove the variation being studied.

Writes analysis/perturb_sample_ids.txt (TSV: stratum, item_id, target, year,
material, recruitment) and prints the realised cell table including any shortfall.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import scaffold_subset as SS
from evals.cloze import extract_prefix

ANALYSIS = Path("analysis")
RESULTS = Path("results")
SCAFFOLD_VERDICTS = ANALYSIS / "judge" / "verdicts.jsonl"
INCUTOFF_DRAW = ANALYSIS / "incutoff_sample_ids.txt"
OUT = ANALYSIS / "perturb_sample_ids.txt"

# The loosest cutoff in the model set. An item after this date is post-cutoff for at
# least one model, which is what makes it usable as a post-cutoff observation at all.
LOOSEST_CUTOFF = 1913
CUTOFF = {"talkie-base": 1930, "talkie-web": 1930, "typewriter": 1913}

SAMPLE_SEED = 20260814


def load_scaffold_verdicts():
    """Every scaffold verdict, keyed by item_id, with the judge block intact.

    neighborhood_judge.scaffold_verdicts() keeps only the verdict string. The
    perturbation needs `donor` (which word to remove), `evidence` (the phrase that
    recruits it, for the deletion arm) and `material`/`recruitment` (the matching
    variables), so this loader keeps the whole block.
    """
    out = {}
    with SCAFFOLD_VERDICTS.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def year_of(rec):
    try:
        return int(rec["year"])
    except (TypeError, ValueError):
        return None


def scaffolded(verdicts):
    return {i: r for i, r in verdicts.items() if r["judge"].get("verdict") == "scaffolded"}


def judged_incutoff_pool(scaf):
    """Scaffolded in-cutoff items that already carry a judged baseline in all 3 models.

    Restricting to the intersection rather than the union costs pool size (1240 rather
    than 2023) and buys the thing the experiment needs: the same item has a judged
    unperturbed neighborhood for every model, so a collapse is measured against a real
    baseline in each, not against three different populations.
    """
    if not INCUTOFF_DRAW.exists():
        sys.exit(f"missing {INCUTOFF_DRAW} -- the frozen in-cutoff draw is required")
    drawn = defaultdict(set)
    with INCUTOFF_DRAW.open() as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                drawn[parts[2]].add(parts[1])
    per_model = {
        m: {i for i in ids
            if i in scaf and (year_of(scaf[i]) or 0) <= CUTOFF[m]}
        for m, ids in drawn.items()
    }
    return set.intersection(*per_model.values()) if per_model else set()


def cell_of(rec):
    j = rec["judge"]
    return (j.get("material") or "", j.get("recruitment") or "")


def draw_matched(post_cells, pool, scaf, seed=SAMPLE_SEED):
    """Fill each material x recruitment cell to its post-cutoff count.

    Returns (drawn_ids, shortfall_by_cell). A cell with insufficient supply takes what
    exists and reports the deficit; it is never backfilled from an adjacent cell, which
    would silently swap in a different recruitment type to preserve a round total.
    """
    by_cell = defaultdict(list)
    for i in pool:
        by_cell[cell_of(scaf[i])].append(i)
    for ids in by_cell.values():
        ids.sort()                       # deterministic order before seeding
    rng = random.Random(seed)
    drawn, shortfall = [], {}
    for cell, need in sorted(post_cells.items(), key=lambda kv: (-kv[1], kv[0])):
        avail = by_cell.get(cell, [])
        take = min(need, len(avail))
        if take < need:
            shortfall[cell] = need - take
        drawn.extend(rng.sample(avail, take))
    return drawn, shortfall


def prefix_index():
    """item_id -> (target_word, prefix, year, text) built from the widest cloze run.

    Typewriter's details CSV is used because its 1913 cutoff makes its post-cutoff set
    the superset; the prompts themselves are model-independent. item_id is recomputed
    the same way scaffold_subset builds it -- hashing (normalised target, prefix) --
    rather than joined on row order, which would break the moment either file is
    regenerated in a different order.
    """
    src = RESULTS / "cloze_typewriter_details.csv"
    if not src.exists():
        sys.exit(f"missing {src}")
    csv.field_size_limit(10 ** 7)
    idx = {}
    with src.open() as fh:
        for row in csv.DictReader(fh):
            target = row["target_word"]
            prefix = extract_prefix(row["text"], target)
            iid = SS.item_id("P", SS.NA.norm(target), prefix)
            idx[iid] = (target, prefix, row["year"], row["text"])
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--write", action="store_true",
                    help="write analysis/perturb_sample_ids.txt (default: report only)")
    args = ap.parse_args()

    verdicts = load_scaffold_verdicts()
    scaf = scaffolded(verdicts)
    idx = prefix_index()

    post = {i: r for i, r in scaf.items() if (year_of(r) or 0) > LOOSEST_CUTOFF}
    missing = [i for i in post if i not in idx]
    if missing:
        sys.exit(f"{len(missing)} post-cutoff items do not rejoin the cloze run: {missing[:5]}")

    post_cells = Counter(cell_of(r) for r in post.values())
    pool = judged_incutoff_pool(scaf)
    drawn, shortfall = draw_matched(post_cells, pool, scaf, seed=args.seed)

    print(f"post-cutoff scaffolded (> {LOOSEST_CUTOFF}): {len(post)}")
    n1930 = sum(1 for r in post.values() if (year_of(r) or 0) > 1930)
    print(f"  of which also post-1930 (Base/Web):    {n1930}")
    print(f"  the {len(post) - n1930}-item 1913-1930 window is post-cutoff for Typewriter only")
    print(f"judged in-cutoff pool (all 3 models):    {len(pool)}")
    print()
    print(f"{'material x recruitment':38} {'need':>5} {'avail':>6} {'drew':>5}")
    drew_cells = Counter(cell_of(scaf[i]) for i in drawn)
    by_cell_avail = Counter(cell_of(scaf[i]) for i in pool)
    for cell, need in sorted(post_cells.items(), key=lambda kv: (-kv[1], kv[0])):
        flag = "  SHORT by %d" % shortfall[cell] if cell in shortfall else ""
        print(f"{str(cell):38} {need:>5} {by_cell_avail.get(cell,0):>6} "
              f"{drew_cells.get(cell,0):>5}{flag}")
    print()
    print(f"in-cutoff stratum drawn: {len(drawn)} (target {len(post)}, "
          f"shortfall {sum(shortfall.values())})")

    if not args.write:
        print("\n(report only -- pass --write to freeze the draw)")
        return

    rows = []
    for i in sorted(post):
        m, rc = cell_of(scaf[i])
        rows.append(("post", i, idx[i][0], scaf[i]["year"], m, rc))
    for i in sorted(drawn):
        m, rc = cell_of(scaf[i])
        target = idx[i][0] if i in idx else scaf[i]["target"]
        rows.append(("incutoff", i, target, scaf[i]["year"], m, rc))
    with OUT.open("w") as fh:
        fh.write("# frozen by perturb_sample.py --write --seed %d\n" % args.seed)
        fh.write("stratum\titem_id\ttarget\tyear\tmaterial\trecruitment\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
