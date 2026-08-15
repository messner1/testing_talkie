#!/usr/bin/env python3
"""Collate every downstream judgement into sheets that can be checked by hand.

The judgements this project rests on live in five stores keyed by hashed ids: scaffold
verdicts in `analysis/judge/verdicts.jsonl`, neighbourhood verdicts in
`analysis/judge_nbr/`, decodes in `results/cloze_{model}_details.csv`, the perturbation in
`perturb_metadata.jsonl` and `perturb_{a,b,c}/`, and register labels from the dataset.
Checking a single reported result therefore means joining five files by hand. Nothing in
the repository writes a joined table -- `perturb_score.py` performs the join but prints.

This writes three sheets to `analysis/verification/`:

    items.csv          one row per (item, model): both judgements in full context
    perturbation.csv   one row per (item, model, arm), including the unedited baseline
    register.csv       one row per editorially labelled item

Every sheet carries the evidence a verdict was made from, not just the verdict. For the
scaffold judge that means `judge_window` -- the last 320 characters of the prefix, which
is all the judge was shown (`scaffold_judge.py`). A verdict can only be checked against
what its judge actually saw, so the full passage and the window are both present.

Blank `verify_*` and `notes` columns are provided for the reader to fill in.

Run: local/bin/python make_verification.py [--stratum post|incutoff|all]
                                           [--sample N] [--seed S] [--truncate]
CPU-only, deterministic. No API calls, no model loading.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import neighborhood_judge as NJ
import perturb_score as PS
import scaffold_subset as SS
from evals.cloze import extract_prefix

RESULTS = Path("results")
ANALYSIS = Path("analysis")
OUTDIR = ANALYSIS / "verification"
MODELS = ("talkie-base", "talkie-web", "typewriter")
WINDOW = 320          # what the scaffold judge sees; see scaffold_judge.py
ARMS = ("original", "donor_substituted", "donor_deleted", "placebo")


def flat(s, limit=None):
    """Collapse whitespace so a cell stays one spreadsheet row."""
    t = re.sub(r"\s+", " ", str(s or "")).strip()
    return t[-limit:] if limit and len(t) > limit else t


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_scaffold():
    path = ANALYSIS / "judge" / "verdicts.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            out[j["item_id"]] = j
    return out


def load_decodes():
    """(model, item_id) -> the committed cloze row, with item_id rebuilt on the join key."""
    csv.field_size_limit(10 ** 7)
    out = {}
    for model in MODELS:
        path = RESULTS / f"cloze_{model}_details.csv"
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                target = (r.get("target_word") or "").strip().lower()
                iid = SS.item_id("P", target, extract_prefix(r["text"], target))
                out[(model, iid)] = r
    return out


def load_register():
    """item_id -> (field, register class). Empty if the dataset is unavailable."""
    try:
        import register_labels as RL
        return {r[0]: (r[4], r[5]) for r in RL.load_labelled()}
    except Exception as exc:                     # dataset not cached, etc.
        print(f"  (register labels unavailable: {exc})", file=sys.stderr)
        return {}


_BROWN = None


def technical_marker(target):
    """The morphological marker used elsewhere, for comparison against the label.

    `load_brown()` reads and counts the whole Brown corpus, so it is cached here. Calling
    it per row turned a seconds-long build into a stalled one -- the same trap
    `neighborhood_judge.resources()` documents for `NA.load_resources`.
    """
    global _BROWN
    if _BROWN is None:
        try:
            import technical_composition as TC
            _BROWN = (TC.is_technical, TC.load_brown())
        except Exception:
            _BROWN = (None, None)
    fn, fd = _BROWN
    return fn(target, fd) if fn else ""


# --------------------------------------------------------------------------- #
# Sheets
# --------------------------------------------------------------------------- #
def sheet_items(keep, scaffold, nbr, decodes, reg, brown, truncate):
    rows = []
    for (model, iid), b in sorted(nbr.items()):
        sj = scaffold.get(iid)
        if sj is None:
            continue
        dec = decodes.get((model, iid), {})
        year = sj.get("year", "")
        stratum = PS.stratum_for(model, year)
        if stratum not in keep:
            continue
        passage = flat(dec.get("text", ""))
        window = flat(extract_prefix(dec.get("text", ""), b.get("target", "")), WINDOW)
        j = sj["judge"]
        donor = j.get("donor", "")
        # The judge sometimes names SEVERAL donors in one field, comma-separated
        # ("analysis, analyse"); 20.8% of scaffolded verdicts corpus-wide do. Checking the
        # field as a single token reports those as missing when every one of them is
        # present, so each is located separately.
        donors = [d.strip() for d in donor.split(",") if d.strip()]
        found = [d for d in donors
                 if re.search(rf"\b{re.escape(d)}\b", window, re.I)]
        sites = sum(len(re.findall(rf"\b{re.escape(d)}\b", window, re.I))
                    for d in donors)
        group = b["judge"].get("form_group") or []
        field, register = reg.get(iid, ("", ""))
        rows.append({
            "item_id": iid, "model": model, "target": b.get("target", ""),
            "year": year, "stratum_model": stratum,
            "stratum_pool": "post" if str(sj.get("is_future")) == "1" else "incutoff",
            "passage": "" if truncate else passage,
            "judge_window": window,
            "sc_verdict": j.get("verdict", ""), "sc_material": j.get("material", ""),
            "sc_donor": donor, "sc_shared": j.get("shared", ""),
            "sc_recruitment": j.get("recruitment", ""),
            "sc_evidence": flat(j.get("evidence", "")),
            "sc_usable": j.get("usable", ""),
            "sc_rationale": flat(j.get("rationale", "")),
            "n_donors_named": len(donors),
            "donors_in_window": f"{len(found)}/{len(donors)}" if donors else "",
            "n_donor_sites": sites,
            "rank": dec.get("rank", ""),
            "hit@10": dec.get("correct@10", ""), "hit@100": dec.get("correct@100", ""),
            "top_10_words": "|".join(b.get("words", [])),
            "nbr_id": b.get("nbr_id", ""),
            "form_n": b.get("form_n", ""),
            "form_group": "|".join(group),
            "form_basis": b["judge"].get("form_basis", ""),
            "form_kind": b["judge"].get("form_kind", ""),
            "form_inflection_discounted": b["judge"].get("form_inflection_discounted", ""),
            "nbr_rationale": flat(b["judge"].get("rationale", "")),
            "scratchpad": int(b.get("form_n", 0)) >= 2,
            "register_field": field or "", "register_class": register or "",
            "technical_marker": technical_marker(b.get("target", "")) if brown else "",
            "verify_scaffold": "", "verify_scratchpad": "", "notes": "",
        })
    return rows


def sheet_perturbation(keep, nbr, decodes, truncate, scaffold):
    """One row per (item, model, arm), `original` first so an item reads down."""
    meta = PS.load_metadata()
    by_item = {m["item_id"]: m for m in meta.values()}
    csv.field_size_limit(10 ** 7)

    decoded = defaultdict(dict)          # (model, item_id) -> arm -> row
    for batch in PS.BATCHES:
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
                    if m:
                        decoded[(model, m["item_id"])][r["context_level"]] = r

    rows = []
    for (model, iid), arms in sorted(decoded.items()):
        m = by_item.get(iid)
        b = nbr.get((model, iid))
        if m is None or b is None:
            continue
        stratum = PS.stratum_for(model, m["year"])
        if stratum not in keep:
            continue
        # The judge sometimes names several donors; `perturb_build` substituted only the
        # first, so the others survive into the edited passage. 77 of 323 built items are
        # affected. Those items are a WEAKENED substitution -- form material the judge
        # identified is still present -- so the column is carried here rather than left
        # for a reader to rediscover.
        named = [d.strip() for d in
                 ((scaffold.get(iid) or {}).get("judge", {}).get("donor", "")).split(",")
                 if d.strip()]
        residual = [d for d in named
                    if d.lower() != m["donor"].lower()
                    and d.lower() in m["prefix"].lower()]
        group = b["judge"].get("form_group") or []
        before_n = int(b.get("form_n", 0))
        dec = decodes.get((model, iid), {})
        edits = {
            "original": "—",
            "donor_substituted": f"{m['donor']} → {m['substitute']} "
                                 f"({m['n_donor_sites']} site"
                                 f"{'s' if m['n_donor_sites'] != 1 else ''})",
            "donor_deleted": f"deleted: {m['delete_method']}",
            "placebo": f"{m['placebo_word']} → {m['placebo_substitute']}",
        }
        for arm in ARMS:
            if arm == "original":
                words = b.get("words", [])
                passage = flat(m["prefix"])
                rank, nll = dec.get("rank", ""), ""
            else:
                r = arms.get(arm)
                if r is None:
                    continue
                words = r["top_10_words"].split("|")
                passage = flat(r.get("prefix", ""))
                rank, nll = r.get("rank", ""), r.get("target_nll", "")
            after_n = before_n if arm == "original" else PS.survival(b, words)
            surviving = [w for w in group
                         if w.lower() in {x.lower() for x in words}]
            rows.append({
                "item_id": iid, "model": model, "target": m["target"],
                "year": m["year"], "stratum_model": stratum, "arm": arm,
                "edit_summary": edits[arm],
                "passage": "" if truncate else passage,
                "passage_tail": flat(passage, WINDOW),
                "top_10_words": "|".join(words),
                "judged_group": "|".join(group),
                "surviving_members": "|".join(surviving),
                "form_n_before": before_n, "form_n_after": after_n,
                "scratchpad_before": before_n >= 2,
                "scratchpad_after": after_n >= 2,
                "collapsed": before_n >= 2 and after_n < 2,
                "rank": rank, "target_nll": nll,
                "material": m["material"], "recruitment": m["recruitment"],
                "delete_method": m["delete_method"],
                "donors_named": "|".join(named),
                "donors_left_in_place": "|".join(residual),
                "verify_edit": "", "verify_collapse": "", "notes": "",
            })
    return rows


def sheet_register(scaffold):
    try:
        import register_labels as RL
    except Exception as exc:
        print(f"  (skipping register.csv: {exc})", file=sys.stderr)
        return []
    rows = []
    for iid, target, year, raw, field, reg, cls in RL.load_labelled():
        j = (scaffold.get(iid) or {}).get("judge", {})
        rows.append({
            "item_id": iid, "target": target, "year": year,
            "raw_label": raw, "field": field or "", "register_class": reg or "excluded",
            "science_class": cls or "excluded",
            "latinate_ending": bool(RL.LATINATE_ENDING.search(target or "")),
            "sc_verdict": j.get("verdict", ""), "sc_material": j.get("material", ""),
            "sc_donor": j.get("donor", ""),
            "verify_label": "", "notes": "",
        })
    return rows


def write(path, rows, sample, seed):
    if not rows:
        print(f"  {path.name}: nothing to write")
        return
    if sample and sample < len(rows):
        # Sample whole ITEMS, not rows, so an item's arms stay together and a sampled
        # sheet remains readable down the column.
        keys = sorted({r["item_id"] for r in rows})
        rnd = random.Random(seed)
        chosen = set(rnd.sample(keys, min(sample, len(keys))))
        rows = [r for r in rows if r["item_id"] in chosen]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stratum", default="post",
                    choices=("post", "incutoff", "all"),
                    help="which stratum to emit (default: post, the claim's population)")
    ap.add_argument("--sample", type=int, default=0,
                    help="emit a random subsample of this many ITEMS")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--truncate", action="store_true",
                    help="omit full passage text, keeping only the judge's window")
    args = ap.parse_args()

    keep = {"post", "incutoff"} if args.stratum == "all" else {args.stratum}
    print("loading ...")
    scaffold = load_scaffold()
    nbr = PS.load_baselines()
    decodes = load_decodes()
    reg = load_register()
    brown = True
    print(f"  {len(scaffold)} scaffold verdicts, {len(nbr)} judged neighbourhoods, "
          f"{len(decodes)} decoded rows")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    write(OUTDIR / "items.csv",
          sheet_items(keep, scaffold, nbr, decodes, reg, brown, args.truncate),
          args.sample, args.seed)
    write(OUTDIR / "perturbation.csv",
          sheet_perturbation(keep, nbr, decodes, args.truncate, scaffold),
          args.sample, args.seed)
    write(OUTDIR / "register.csv", sheet_register(scaffold), 0, args.seed)


if __name__ == "__main__":
    main()
