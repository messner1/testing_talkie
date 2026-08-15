#!/usr/bin/env python3
"""LLM-as-judge for the neighborhood OUTCOME measure: group ten predictions by shared form.

WHAT THIS IS FOR.  The scaffolding analysis selects items by a property of the PROMPT
(`scaffold_judge.py`) and measures a property of the NEIGHBORHOOD.  The neighborhood
measure has until now been an automatic orthographic rule (`comp_strict`/`n_attempts` in
`scaffold_subset.py`), audited against 360 hand labels.  Hand annotation cannot reach the
151,050 neighborhoods the headline tables run over, so the hand pass shrinks to a
CALIBRATION set and this judge scales -- exactly the arrangement already used on the
selection side.

THE FIREWALL RUNS BOTH WAYS.  The selection judge sees the prompt and never the
predictions.  This judge sees the predictions and never the prompt.  If it could see the
prompt it could infer whether the item was scaffolded, and the outcome measure would be
contaminated by the selection variable.  Predictions are shown in randomized order so
rank leaks nothing either.

ONE AXIS, AFTER A PILOT.  An earlier version also asked for a semantic grouping, to tell
retrieval from unstructured output.  A 100-neighborhood pilot judged twice killed it: it
did not discriminate (a semantic group of 3+ was present at 0.959 whether or not the
prompt supplied recruitable material; mean size 6.43 vs 6.14) and was the less stable axis
(test-retest exact 0.719 against 0.906).  The analysis is now one-class -- it identifies
neighborhoods whose candidates form a paradigm and does not characterise the rest.

WHAT THE JUDGE IS NOT TOLD.  It is never told that a large form group will be read as
evidence the model assembled its candidates rather than recalling them, nor that any of
this concerns scaffolding or leakage.  The rubric describes a grouping task.  Told the
framing, it would produce the pattern rather than measure it.  See `withheld` in the
rubric JSON.

CONNECTIVITY IS ENFORCED, NOT TRUSTED.  The pilot found 8.6% of returned groups spanning
more than one component of the supplied overlap graph -- 0% at size 2-3 but 55.6% at 6+,
concentrated exactly where the result lives.  `form_size()` keeps only the largest
connected subset of whatever comes back.

CODE FINDS, THE JUDGE JUDGES -- twice over.
  * Orthographic overlaps among the ten candidates are computed exhaustively and handed
    over as a table.  Sonnet 5 cannot do reliable exhaustive character comparison; on the
    selection side it denied that `northern` shares letters with `southern`.
  * Attestation is looked up, never shown.  CERTIFICATION -- does the group contain a form
    that is unattested AND types as a distinct lexeme rather than an orthographic variant
    -- is computed downstream from the returned membership.  Showing the flag risked the
    judge dropping coinages out of the group as errors, removing the members that certify
    it.

CORRELATED INSTRUMENTS -- the live methodological risk.  Selection and outcome would both
run on the same model.  They cannot share evidence, but they can share lexical priors.
`--cross-check` re-runs on a second model; at the post-cutoff scope that costs ~$13, so
there is no reason to report the association without it.

TWO ARMS, BOUGHT SEPARATELY.  A scaffolded/not lift inside the post-cutoff arm is not the
result: scaffolded prompts are selected for supplying morphological material, which raises
form cohesion for ANY model, leakage or not.  talkie-web -- the unrestricted baseline --
shows 4.25x on certification post-cutoff for exactly that reason.  The estimand is the
difference-in-differences: that lift divided by the same lift among in-cutoff targets,
where composition is not needed because the word is in the training data.  `--scope
incutoff` buys the second arm; see sample_incutoff() for the draw and `--allocation` for
why its cells are equal-n rather than a mirror of the post-cutoff ones.

Run:
    local/bin/python neighborhood_judge.py --show-prompt
    local/bin/python neighborhood_judge.py --cost
    local/bin/python neighborhood_judge.py --calibrate          # against Sheet A hand labels
    local/bin/python neighborhood_judge.py --submit --scope postcutoff
    local/bin/python neighborhood_judge.py --collect all
    local/bin/python neighborhood_judge.py --allocation         # why the in-cutoff cells are equal-n
    local/bin/python neighborhood_judge.py --submit --scope incutoff --context
    local/bin/python neighborhood_judge.py --cross-check --model claude-opus-5
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import assoc as AS                                       # noqa: E402  (lambda, G^2)
import neighborhood_analysis as NA                       # noqa: E402  (attestation set)
import scaffold_judge as SJ                              # noqa: E402  (client, batching)
from scaffold_judge import lcs_run, _where               # noqa: E402  (do not reimplement)

ANALYSIS = Path("analysis")
JUDGE = ANALYSIS / "judge_nbr"
RUBRIC = Path("neighborhood_judge_rubric.json")
SUBSET = ANALYSIS / "scaffold_subset.csv"
SCAFFOLD_VERDICTS = ANALYSIS / "judge" / "verdicts.jsonl"
INCUT_IDS = ANALYSIS / "incutoff_sample_ids.txt"
SHEET_A = ANALYSIS / "neighborhood_A_cohesion.csv"
HAND_LOG = ANALYSIS / "hand_labeling" / "neighborhood_A.jsonl"
csv.field_size_limit(10 ** 7)

DEFAULT_MODEL = "claude-sonnet-5"
SEED = 20260811
SAMPLE_SEED = 20260812   # in-cutoff draw; separate from the presentation shuffle above
PER_CELL = 1697          # 6 cells (3 models x scaffolded/not) = 10,182, matching the
                         # post-cutoff pass. EQUAL, not proportional: only 426 of the
                         # 10,181 post-cutoff items are scaffolded, so mirroring those
                         # cell sizes would put 76-90% of the difference-in-differences
                         # variance in the newly bought arm. Equal-n sits within a few
                         # percent of the variance optimum at every threshold and assumes
                         # nothing about the rates. See --allocation.
MIN_RUN = 3          # shortest shared run listed in the overlap table
MIN_WORD = 3         # a word shorter than this cannot carry a usable run.
                     # Was 4, which hid the bare stem of a paradigm: `sex` never appeared
                     # in any overlap row despite linking sexist/sexualist/bisexist, and
                     # the same for `ma`, `rat`. Short stems are exactly what a technical
                     # paradigm is built on, so excluding them removed the signal.
CALIB_N = 80         # hand labels needed to calibrate; see --calibration-set

_RESOURCES = None


def resources():
    """(stops, morph lookup, suffixes, attestation set), loaded once.

    `NA.load_resources()` has NO internal cache and costs ~1.2s per call -- it re-reads the
    morpheme table, vocabulary.csv, /usr/share/dict/words and WordNet every time. Calling
    it per item turned a seconds-long validation into a seven-hour one. Everything here
    goes through this accessor; never call NA.load_resources() directly.
    """
    global _RESOURCES
    if _RESOURCES is None:
        _RESOURCES = NA.load_resources()
    return _RESOURCES


def attest_set():
    return resources()[3]


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def load_neighborhoods(scope="postcutoff", match_pos=True):
    """One row per (model, cloze item) with its ten predictions.

    Order is randomized per item under a fixed seed: predicted rank must not leak, and the
    hand sheet shuffles too, so judge and annotator see the same presentation.

    `incutoff` is not the whole in-cutoff population (140,869 neighborhoods, ~$210) but the
    stratified draw built by sample_incutoff().
    """
    if scope == "incutoff":
        return sample_incutoff(match_pos=match_pos)
    if not SUBSET.exists():
        sys.exit(f"missing {SUBSET} — run scaffold_subset.py first")
    out = []
    for r in csv.DictReader(open(SUBSET)):
        words = [w for w in (r["top_10_words"] or "").split("|") if w]
        if len(words) < 2:
            continue
        if scope == "postcutoff" and r["is_future"] != "1":
            continue
        rng = random.Random(f"{SEED}:{r['nbr_id']}")
        rng.shuffle(words)
        y = (r.get("year") or "").strip()
        yi = int(y) if y.isdigit() else None
        out.append({"nbr_id": r["nbr_id"], "item_id": r["item_id"],
                    "model": r["model"], "is_future": r["is_future"],
                    "target": r["target_word"], "words": words,
                    # A+B context: the slot's grammatical category and the citation's date
                    # band. Both describe the SLOT, never the passage. Attached here so
                    # every mode -- calibrate, submit, pilot -- sees the same frame.
                    "slot_pos": _target_pos().get(r["target_word"].lower()) or "unknown",
                    "year": y or None,
                    "register": (None if not yi else
                                 "early modern" if yi < 1700 else
                                 "18th-19th century" if yi < 1900 else "modern")})
    return out


def stem(args):
    """Filename stem for a run: model, context frame, and scope.

    Scope MUST be in the name. The post-cutoff pass and the in-cutoff arm are the two halves
    of one difference-in-differences and are bought separately; without the tag, submitting
    the second would overwrite the first's manifest and then its verdicts, and the batch ids
    are the only way back for ~30 days. `postcutoff` is left untagged so the files already on
    disk stay valid.
    """
    sc = getattr(args, "scope", "postcutoff")
    return (f"{args.model}"
            + ("_context" if getattr(args, "context", False) else "")
            + ("" if sc == "postcutoff" else f"_{sc}"))


_VERDICTS = None


def scaffold_verdicts():
    """item_id -> selection verdict, from the corpus-scale scaffold pass. Loaded once.

    The in-cutoff strata ARE the selection verdicts, so this is a hard dependency: without
    it there is no scaffolded/not contrast to match.
    """
    global _VERDICTS
    if _VERDICTS is None:
        if not SCAFFOLD_VERDICTS.exists():
            sys.exit(f"missing {SCAFFOLD_VERDICTS} — the in-cutoff sample is stratified on "
                     "the selection verdict; run scaffold_judge.py --collect first")
        _VERDICTS = {}
        for line in SCAFFOLD_VERDICTS.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                _VERDICTS[j["item_id"]] = j["judge"].get("verdict")
    return _VERDICTS


def _largest_remainder(shares, total):
    """Integer quotas summing exactly to `total`, apportioned by `shares`."""
    raw = {k: total * s for k, s in shares.items()}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[:total - sum(q.values())]:
        q[k] += 1
    return q


def sample_incutoff(per_cell=PER_CELL, match_pos=True, verbose=True):
    """The in-cutoff arm of the difference-in-differences, drawn to match the post-cutoff pass.

    Stratified on model x selection verdict, EQUAL n per cell (see PER_CELL), and within a
    cell matched to the post-cutoff arm's slot-POS distribution for the same verdict. POS is
    matched because we have direct evidence it drives this measure -- an adverb slot forces
    `-ly` on every candidate -- and the two arms differ (NOUN 0.617 in-cutoff vs 0.681
    post-cutoff). It is covariate balance, not a model of the outcome; `--no-pos-match`
    turns it off so the sensitivity is checkable.

    `unsure` selection verdicts are excluded, as they are on the post-cutoff side.

    The drawn ids are persisted to INCUT_IDS and reloaded thereafter, so submit, collect and
    analysis see one fixed sample even if the verdict file is ever regenerated.
    """
    # nbr_id is NOT unique: 8 ids collide across the 151,050 rows, and scaffold_subset.csv
    # carries 3 exactly duplicated rows (item P9f3850af, `shoot`, once per model). The sample
    # is therefore keyed on (nbr_id, item_id, model) and the pool de-duplicated on it, so a
    # duplicated row cannot be drawn -- or paid for -- twice.
    def key(i):
        return (i["nbr_id"], i["item_id"], i["model"])

    seen, every = set(), []
    for i in load_neighborhoods("all"):
        if key(i) not in seen:
            seen.add(key(i))
            every.append(i)

    if INCUT_IDS.exists():
        want = [tuple(l.split("\t")) for l in INCUT_IDS.read_text().splitlines() if l.strip()]
        idx = {key(i): i for i in every}
        missing = [k for k in want if k not in idx]
        if missing:
            sys.exit(f"{INCUT_IDS} lists {len(missing)} rows that no longer resolve "
                     f"(e.g. {missing[0]}) — the subset changed under the sample; "
                     "delete the file to redraw")
        if verbose:
            print(f"  in-cutoff sample: {len(want)} neighborhoods (reloaded from {INCUT_IDS})")
        return [idx[k] for k in want]

    v = scaffold_verdicts()
    arms = ("scaffolded", "not_scaffolded")
    pool, post = defaultdict(list), defaultdict(Counter)
    for i in every:
        vd = v.get(i["item_id"])
        if vd not in arms:
            continue
        if i["is_future"] == "1":
            post[vd][i["slot_pos"]] += 1
        else:
            pool[(i["model"], vd)].append(i)

    rng = random.Random(SAMPLE_SEED)
    out, short = [], []
    for cell in sorted(pool):
        cand = sorted(pool[cell], key=key)
        if not match_pos:
            rng.shuffle(cand)
            take = cand[:per_cell]
        else:
            tot = sum(post[cell[1]].values())
            quota = _largest_remainder({p: c / tot for p, c in post[cell[1]].items()}, per_cell)
            by = defaultdict(list)
            for i in cand:
                by[i["slot_pos"]].append(i)
            take, spare = [], []
            for p, bucket in by.items():
                rng.shuffle(bucket)
                n = quota.get(p, 0)
                take += bucket[:n]
                spare += bucket[n:]
                if n > len(bucket):
                    short.append((cell, p, n, len(bucket)))
            rng.shuffle(spare)                      # backfill any unfillable POS quota
            take += spare[:per_cell - len(take)]
        if len(take) < per_cell:
            short.append((cell, "*", per_cell, len(take)))
        out += take
    if verbose:
        print(f"  in-cutoff sample: {len(out)} neighborhoods across {len(pool)} cells "
              f"({per_cell}/cell, pos-matched={match_pos})")
        for cell, p, want, got in short:
            print(f"    ! {cell[0]}/{cell[1]} {p}: wanted {want}, pool had {got} — backfilled")
    INCUT_IDS.write_text("".join("\t".join(key(i)) + "\n" for i in out))
    if verbose:
        print(f"    ids written to {INCUT_IDS}")
    return out


_TPOS = None


def _target_pos():
    global _TPOS
    if _TPOS is None:
        p = Path("cache/target_pos.json")
        _TPOS = json.loads(p.read_text()) if p.exists() else {}
    return _TPOS


def overlaps_among(words, min_run=MIN_RUN, top=24):
    """Every pair of candidates sharing a run of `min_run`+ characters.

    Pairwise among the ten, not against a target -- the target is hidden and the construct
    is the candidates' relation to each other. Ten words is 45 pairs, so this is exhaustive
    rather than sampled.
    """
    out = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = words[i], words[j]
            if len(a) < MIN_WORD or len(b) < MIN_WORD:
                continue
            n, sub = lcs_run(a, b)
            if n < min_run:
                continue
            out.append({"a": a, "b": b, "shared": sub, "n": n,
                        "in_a": _where(sub, a), "in_b": _where(sub, b)})
    out.sort(key=lambda d: (-d["n"], d["a"], d["b"]))
    return out[:top]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def load_rubric():
    if not RUBRIC.exists():
        sys.exit(f"missing {RUBRIC}")
    return json.loads(RUBRIC.read_text())


def system_prompt(rub, context=False):
    """Render the rubric JSON into the judge's system prompt.

    Everything the judge is told comes from the JSON, so the published file and the
    deployed instrument cannot drift apart. `withheld` and `notes_for_maintainers` are
    never rendered -- the first would tell the judge what its output is used for.
    """
    L = [f"# {rub['purpose']}", ""]
    L += ["## Task", ""] + list(rub["task"]) + [""]
    v = rub["visibility"]
    L += ["## What you see, and what you must not", "",
          f"You see: {v['you_see']}", "",
          f"You do not see, and must not ask for: {v['you_do_not_see']}", ""]

    d1 = rub["dimension_1_form"]
    L += ["## The form group", "", d1["principle"], ""]
    for r in d1["rules"]:
        L.append(f"- {r}")
    L += ["", "Record what the group shares and of what kind:", ""]
    for k, desc in d1["kinds"].items():
        L.append(f"- `{k}` — {desc}")
    L += ["", d1["note_on_kinds"], "", d1["not_a_criterion"], ""]

    L += ["", "## What is explicitly not a criterion", ""]
    for n in rub["non_criteria"]:
        L.append(f"- {n}")

    L += ["", "## Output", ""]
    for f, desc in rub["output"].items():
        L.append(f"- `{f}` — {desc}")
    if rub.get("output_note"):
        L += ["", rub["output_note"]]
    L += ["", "## Worked examples", ""]
    for ex in rub["worked_examples"]:
        L.append(f"**{ex['candidates']}**")
        L.append(f"  - form_group: {ex.get('form_group', [])}"
                 + (f", basis `{ex['form_basis']}`" if ex.get("form_basis") else "")
                 + (f", kind `{ex['form_kind']}`" if ex.get("form_kind") else ""))
        if ex.get("form_inflection_discounted"):
            L.append("  - form_inflection_discounted: true")
        L.append(f"  - {ex['why']}")
        L.append("")
    L.append(rub["examples_are_excluded"])
    if context and rub.get("context_frame"):
        cf = rub["context_frame"]
        L += ["", "## What you are told about the gap", "", cf["preamble"], ""]
        for x in cf["rules"]:
            L.append(f"- {x}")
    return "\n".join(L)


def output_schema(rub):
    kinds = list(rub["dimension_1_form"]["kinds"]) + ["none"]
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "form_group": {"type": "array", "items": {"type": "string"}},
                "form_basis": {"type": "string"},
                "form_kind": {"type": "string", "enum": kinds},
                "form_inflection_discounted": {"type": "boolean"},
                "rationale": {"type": "string", "maxLength": 400},
            },
            "required": ["form_group", "form_basis", "form_kind",
                         "form_inflection_discounted", "rationale"],
            "additionalProperties": False,
        },
    }


# The rubric's output fields. A verdict missing any is discarded rather than half-counted.
FIELDS = ("form_group", "form_basis", "form_kind", "form_inflection_discounted",
          "rationale")


def user_message(item, context=False):
    """The ten words and the overlap table. NO attestation flags.

    v1.0 supplied an attestation flag per word so the judge could keep coinages out of the
    meaning group, which no longer exists. Certification is computed downstream from the
    returned form_group instead, and showing the flag became a liability: a judge that sees
    UNATTESTED may drop coinages OUT of the group as errors, removing precisely the members
    that certify it. The rubric's `DO NOT EXCLUDE A WORD FOR LOOKING WRONG` rule guards the
    same thing from the other side.
    """
    L = ["CANDIDATES (order is randomized and carries nothing):", ""]
    for w in item["words"]:
        L.append(f"  {w}")
    ov = overlaps_among(item["words"])
    L += ["", "ORTHOGRAPHIC OVERLAPS (computed for you — do not look for others):", ""]
    if not ov:
        L.append("  (no pair shares a run of three or more characters)")
    for d in ov:
        L.append(f"  {d['a']} / {d['b']}: share '{d['shared']}' "
                 f"({d['n']} chars, {d['in_a']} of {d['a']}, {d['in_b']} of {d['b']})")
    if context:
        L = ([f"THE GAP REQUIRES: {item.get('slot_pos') or 'unknown part of speech'}",
              f"THE TEXT DATES FROM: {item.get('year') or 'unknown'}"
              + (f"  ({item['register']} English)" if item.get("register") else ""),
              ""] + L)
    L += ["", "Report the largest single form group."]
    return "\n".join(L)


def request_params(rub, item, model, effort="off", context=False):
    p = {
        "model": model,
        "max_tokens": 3000,
        "system": [{"type": "text", "text": system_prompt(rub, context),
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "output_config": {"format": output_schema(rub)},
        "messages": [{"role": "user", "content": user_message(item, context)}],
    }
    if model in ("claude-sonnet-5", "claude-opus-5"):
        if effort == "off":
            p["thinking"] = {"type": "disabled"}
        else:
            p["thinking"] = {"type": "adaptive"}
            p["output_config"]["effort"] = effort
    return p


def judge_one(cl, rub, item, model, effort="off", context=False):
    """Returns (verdict|None, usage). NEVER raises on a bad response.

    One item that truncates mid-string must not destroy a run: an unhandled
    JSONDecodeError once killed a pilot at item 80 of 200. The caller records a None as a
    failure and carries on.
    """
    msg = cl.messages.create(**request_params(rub, item, model, effort, context))
    text = next((c.text for c in msg.content if c.type == "text"), None)
    if not text:
        _log_failure(item, model, "no text block", "", msg)
        return None, msg.usage
    try:
        return json.loads(text), msg.usage
    except json.JSONDecodeError as e:
        # Capture the raw text. Diagnosing a failure needs the response that caused it,
        # and discarding it left two pilot failures unexplainable. Logging only -- the
        # request is unchanged, so runs before and after this remain comparable.
        _log_failure(item, model, f"unparseable JSON: {e}", text, msg)
        return None, msg.usage


def _log_failure(item, model, reason, text, msg):
    JUDGE.mkdir(parents=True, exist_ok=True)
    with open(JUDGE / f"failures_{model}.jsonl", "a") as f:
        f.write(json.dumps({
            "nbr_id": item["nbr_id"], "target": item["target"], "words": item["words"],
            "reason": reason, "stop_reason": getattr(msg, "stop_reason", None),
            "output_tokens": getattr(getattr(msg, "usage", None), "output_tokens", None),
            "raw": (text or "")[-600:]}) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# Counts derived from member lists
# --------------------------------------------------------------------------- #
def form_size(v, words):
    """Size of the largest CONNECTED subset of whatever the judge returned.

    The rule is enforced here rather than trusted, because a pilot found 8.6% of returned
    groups spanning more than one component of the supplied overlap graph -- 0% at size
    2-3, but 55.6% at size 6+, i.e. concentrated exactly in the groups that drive the
    result. Judge proposes membership; code keeps the largest component. Enforcement costs
    nothing at size 2-3 and moves the size-5 lift from 12.0x to 10.0x.

    A group of one is not a group, so a surviving singleton returns 0 -- which keeps
    `form_size == 0` an exact statement about the absence of a paradigm.
    """
    g = [w.lower() for w in (v.get("form_group") or [])]
    if len(g) < 2:
        return 0
    ws = [w.lower() for w in words]
    parent = {w: w for w in ws}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d in overlaps_among(words, top=10 ** 6):
        a, b = find(d["a"].lower()), find(d["b"].lower())
        if a != b:
            parent[a] = b
    comp = defaultdict(list)
    for w in g:
        if w in parent:
            comp[find(w)].append(w)
    best = max(comp.values(), key=len, default=[])
    if len(best) < 2:
        return 0

    # A group whose ONLY relation is inflection is one lexeme, not a paradigm. The rubric
    # states this, and the judge does not reliably apply it -- on the held-out set it
    # returned `explanation`/`explanations` while itself setting form_inflection_discounted,
    # the contradiction showing through. Enforced here for the same reason connectivity is:
    # the judge proposes membership, code enforces the definitional constraints. This is not
    # a threshold; it is the rule the annotator stated when ruling shot/shots and
    # filter/filters both 0.
    import neighborhood_measure as _NM
    distinct = []
    for w in sorted(best, key=len):
        if any(_NM._inflection(w, u) for u in distinct):
            continue
        distinct.append(w)
    if len(distinct) < 2:
        return 0
    return len(best)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def mode_show_prompt(args):
    sp = system_prompt(load_rubric())
    print(sp)
    print(f"\n--- {len(sp)} chars ≈ {len(sp)//4} tokens ---", file=sys.stderr)
    items = load_neighborhoods(args.scope, args.match_pos)
    if items:
        print("\n=== example user message ===\n", file=sys.stderr)
        print(user_message(items[0]), file=sys.stderr)


def _calibration_ids():
    path = ANALYSIS / "neighborhood_A_calibration.txt"
    if not path.exists():
        return None
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def mode_calibration_set(args):
    """Choose which Sheet A items to hand-label, balanced on the scaffold verdict.

    Sheet A is 360 items and the judge only needs a calibration anchor, so this names a
    subset. Balanced across (scaffold verdict x stratum x model) so that agreement is not
    measured predominantly in one cell -- the affix over-acceptance on the selection side
    was only visible because the calibration set spanned material types.
    """
    rows = {r["nbr_id"]: r for r in csv.DictReader(open(SUBSET))}
    sheet = [r["id"] for r in csv.DictReader(open(SHEET_A))]
    verdicts = {}
    vpath = ANALYSIS / "judge" / "verdicts.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                verdicts[j["item_id"]] = j["judge"]["verdict"]

    cells = defaultdict(list)
    for sid in sheet:
        r = rows.get(sid)
        if not r:
            continue
        cells[(verdicts.get(r["item_id"], "unknown"), r["is_future"], r["model"])].append(sid)

    rng = random.Random(SEED)
    per = max(1, args.n // max(1, len(cells)))
    shuffled = {}
    for k in sorted(cells):
        pool = sorted(cells[k])
        rng.shuffle(pool)
        shuffled[k] = pool
    picked = [i for k in sorted(cells) for i in shuffled[k][:per]]
    # Small cells cannot supply their quota, so the balanced pass undershoots. Top up
    # round-robin from whatever is left rather than returning a short set -- the balance
    # is a floor on coverage, not a cap on n.
    depth = per
    while len(picked) < args.n:
        added = 0
        for k in sorted(cells):
            if len(picked) >= args.n:
                break
            if depth < len(shuffled[k]):
                picked.append(shuffled[k][depth])
                added += 1
        if not added:
            break
        depth += 1
    rng.shuffle(picked)
    picked = picked[:args.n]

    out = ANALYSIS / "neighborhood_A_calibration.txt"
    out.write_text("\n".join(picked) + "\n")
    print(f"  wrote {out} — {len(picked)} of {len(sheet)} Sheet A items")
    print(f"  cells: {len(cells)} (verdict × stratum × model), ~{per} each")
    print(f"\n  label them with:  local/bin/python label_neighborhoods.py --calibration")


def _hand_labels():
    if not HAND_LOG.exists():
        return {}
    out = {}
    for line in HAND_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Form only. The meaning axis was dropped in rubric v3.0; existing hand labels
        # keep their meaning counts on disk but nothing reads them.
        if r.get("form") is not None:
            out[r["id"]] = int(r["form"])
    return out


def mode_calibrate(args):
    hand = _hand_labels()
    if not hand:
        sys.exit("no hand labels yet — run label_neighborhoods.py --calibration first")
    items = {n["nbr_id"]: n for n in load_neighborhoods("all")}
    if args.heldout:
        hp = ANALYSIS / "neighborhood_A_heldout.txt"
        if not hp.exists():
            sys.exit(f"missing {hp}")
        keep = {l.strip() for l in hp.read_text().splitlines() if l.strip()}
        hand = {k: v for k, v in hand.items() if k in keep}
        print(f"  HELD-OUT scoring: restricted to {len(hand)} items never used for "
              f"rubric development")
    todo = [items[i] for i in hand if i in items]
    if args.context:
        # A+B: attach the slot's grammatical category and the citation's date band.
        # Both describe the SLOT, not the passage -- see rubric.notes.context_frame_ab.
        import csv as _csv
        tpos = json.loads(Path("cache/target_pos.json").read_text()) \
            if Path("cache/target_pos.json").exists() else {}
        meta = {r["nbr_id"]: r for r in _csv.DictReader(open(SUBSET))}
        for it in todo:
            m = meta.get(it["nbr_id"], {})
            y = m.get("year", "")
            it["slot_pos"] = tpos.get(it["target"].lower())
            it["year"] = y if y.isdigit() else None
            it["register"] = (None if not it["year"] else
                              "early modern" if int(y) < 1700 else
                              "18th-19th century" if int(y) < 1900 else "modern")
    if args.n:
        todo = todo[:args.n]
    print(f"  {len(todo)} items with both a hand label and a neighborhood\n")

    rub = load_rubric()
    cl = SJ.client()
    out_path = JUDGE / (f"calibration_{args.model}"
                        + ("_context" if args.context else "")
                        + ("_heldout" if args.heldout else "") + ".jsonl")
    JUDGE.mkdir(parents=True, exist_ok=True)
    rows, usage = [], Counter()
    with open(out_path, "w") as fh:
        for n, it in enumerate(todo, 1):
            v, u = judge_one(cl, rub, it, args.model, args.effort, args.context)
            if v is None or any(f not in v for f in FIELDS):
                print(f"  ! {it['nbr_id']} unusable — continuing")
                continue
            for f in ("input_tokens", "output_tokens"):
                usage[f] += getattr(u, f, 0) or 0
            usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            fh.write(json.dumps({"nbr_id": it["nbr_id"], "target": it["target"],
                                 "words": it["words"], "judge": v}) + "\n")
            rows.append((it, v))
            if n % 10 == 0:
                print(f"    {n}/{len(todo)}")

    print(f"\n  tokens/item: {usage['input_tokens']//max(len(rows),1)} fresh in, "
          f"{usage['cache_read']//max(len(rows),1)} cache-read, "
          f"{usage['output_tokens']//max(len(rows),1)} out")
    if not usage["cache_read"]:
        print("  ! cache never read — a full run would cost several times the estimate")

    _report_calibration(rows, hand)
    print(f"\n  wrote {out_path}")


def _report_calibration(rows, hand):
    """Both automatic measures against the annotator, on identical items.

    Reporting only the judge here would repeat the mistake this comparison exists to
    catch: the cheap measure agrees with the judge far better (0.81) than either agrees
    with the annotator (~0.5), because the two share a permissiveness about chaining that
    the annotator does not. Agreement between two automatic measures is not validation.
    """
    trip = [(hand[it["nbr_id"]], form_size(v, it["words"]),
             len(cheap_form_group(it["words"])), it, v)
            for it, v in rows if it["nbr_id"] in hand]
    if not trip:
        return
    print(f"\n  === {len(trip)} items with a hand label ===\n")
    print(f"  {'measure':<22}{'exact':>8}{'within-1':>10}{'mean|d|':>9}{'corr':>8}{'mean':>8}")
    hm = sum(h for h, _, _, _, _ in trip) / len(trip)
    for lab, idx in (("judge", 1), ("cheap rule", 2)):
        d = [abs(t[0] - t[idx]) for t in trip]
        g = [t[0] for t in trip]; q = [t[idx] for t in trip]
        mq = sum(q) / len(q)
        cv = sum((a - hm) * (b - mq) for a, b in zip(g, q)) / len(g)
        sg = (sum((a - hm) ** 2 for a in g) / len(g)) ** .5
        sq = (sum((b - mq) ** 2 for b in q) / len(q)) ** .5
        print(f"  {lab:<22}{sum(x == 0 for x in d)/len(d):>8.3f}"
              f"{sum(x <= 1 for x in d)/len(d):>10.3f}{sum(d)/len(d):>9.2f}"
              f"{(cv/(sg*sq) if sg and sq else float('nan')):>8.3f}{mq:>8.2f}")
    print(f"  {'(annotator)':<22}{'':>8}{'':>10}{'':>9}{'':>8}{hm:>8.2f}")

    print(f"\n  threshold agreement")
    print(f"    {'k':>3}{'judge':>10}{'cheap':>10}{'hand >= k':>12}")
    for k in (2, 3, 5):
        hs = [t[0] >= k for t in trip]
        print(f"    {k:>3}{sum((t[1] >= k) == h for t, h in zip(trip, hs))/len(trip):>10.3f}"
              f"{sum((t[2] >= k) == h for t, h in zip(trip, hs))/len(trip):>10.3f}"
              f"{sum(hs):>12}")

    print("\n  largest disagreements (hand | judge | cheap)")
    for h, j_, c, it, v in sorted(trip, key=lambda t: -max(abs(t[0]-t[1]), abs(t[0]-t[2])))[:6]:
        print(f"\n    {it['target']:<14} hand={h}  judge={j_}  cheap={c}")
        print(f"      {' · '.join(it['words'])}")
        print(f"      judge: {v.get('form_group')}")
        print(f"        basis={v.get('form_basis')!r} infl={v.get('form_inflection_discounted')}")
        print(f"      cheap: {cheap_form_group(it['words'])}")

    try:
        import neighborhood_measure as NM
        auto = [(h, len(NM.form_group(it["words"]))) for h, _, _, it, _ in trip]
        d2 = [abs(a - b) for a, b in auto]
        g2 = [a for a, _ in auto]; q2 = [b for _, b in auto]
        mg2, mq2 = sum(g2) / len(g2), sum(q2) / len(q2)
        cv2 = sum((a - mg2) * (b - mq2) for a, b in auto) / len(auto)
        sg2 = (sum((a - mg2) ** 2 for a in g2) / len(g2)) ** .5
        sq2 = (sum((b - mq2) ** 2 for b in q2) / len(q2)) ** .5
        print(f"  {'automatic measure':<22}{sum(x == 0 for x in d2)/len(d2):>8.3f}"
              f"{sum(x <= 1 for x in d2)/len(d2):>10.3f}{sum(d2)/len(d2):>9.2f}"
              f"{(cv2/(sg2*sq2) if sg2 and sq2 else float('nan')):>8.3f}{mq2:>8.2f}")
    except Exception as e:                                      # noqa: BLE001
        print(f"  (automatic measure unavailable: {e})")

    print("\n  READ: agreement with the annotator is the validation. Agreement between "
          "the\n  judge and the cheap rule is not — they share a bias the annotator does "
          "not.")


def cheap_form_group(words, min_run=MIN_RUN):
    """The automatic form group: link candidates sharing a contiguous run, take the
    largest connected component.

    ONE parameter, and it is not fitted. min_run=3 is the shortest string that can be an
    English morpheme, which is why the same value governs the selection-side overlap
    extractor. Its behaviour is a cliff rather than a tuned optimum -- agreement with the
    judge is 0.102 at 2 (every pair of English words shares two characters), 0.813 at 3,
    0.694 at 4, 0.558 at 5.

    WHAT WAS TRIED AND DROPPED, all measured against 10,167 judged neighborhoods:
      * Slot-inflection stripping (a hand-written suffix list plus a "K of ten share it"
        threshold): four parameters, and it scored WORSE than this against the judge --
        0.805 exact against 0.813, identical mean error.
      * Porter stemming: fixes the case it targets (on the 880 neighborhoods where the
        judge discounted slot inflection it gets mean 1.83 against the judge's 2.27, versus
        3.35 here) but is worse overall, 0.716 against 0.794 where the judge found a group,
        because it truncates shared material and merges distinct lexemes.
      * Linking on a shared morpheme from the project's segmentation table: threshold-free
        and principled, but the table covers 37.6% of candidates and fails exactly on the
        forms that carry the signal -- `lithon`, `metron`, `cholinegic` segment to nothing,
        `acetylcholic` to `acetylc`+`holic`. Scored 0.464.

    CAUTION. Those three were rejected on agreement with the JUDGE. Against the human
    annotator the picture differs: this measure scores a group of 7 on a neighborhood of
    ten `-ly` adverbs that the annotator marked 0, which the discarded inflection rule
    would have caught. See `_report_calibration` -- agreement between this measure and the
    judge (0.81) far exceeds either one's agreement with the annotator (~0.5), because both
    share a permissiveness about chaining through short runs that the annotator does not.

    SUBSTITUTION TEST against the judge, over 10,167 neighborhoods:
      * CERTIFICATION: substitutable. Agreement 0.9944; lifts 5.77x/7.26x/3.19x against the
        judge's 6.00x/7.46x/3.75x, ordering preserved.
      * form >= 2 and >= 3: substitutable, deltas -0.04 to -0.49, ordering preserved.
      * form >= 5: NOT SUBSTITUTABLE. Base 7.50x -> 4.06x while Web 5.82x -> 4.41x, so the
        ordering of the two models REVERSES. Any claim at this threshold needs the judge.
    """
    ws = [w.lower() for w in words]
    parent = {i: i for i in range(len(ws))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i], ws[j]
            if len(a) < min_run or len(b) < min_run:
                continue
            if lcs_run(a, b)[0] >= min_run:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    comp = defaultdict(list)
    for i in range(len(ws)):
        comp[find(i)].append(ws[i])
    best = max(comp.values(), key=len, default=[])
    return best if len(best) >= 2 else []


UNDERWOOD = Path("cache/underwood")
UNDERWOOD_URL = ("https://raw.githubusercontent.com/tedunderwood/DataMunging/master/"
                 "rulesets/{}.txt")
_DOCUMENTED = None


def documented():
    """Forms with published provenance as an OCR error or a historical spelling.

    Underwood's DataMunging rulesets (2013), built for English after 1700 against
    HathiTrust: `CorrectionRules.txt` maps ~98k OCR'd and variant forms to their correction
    with a corpus frequency (`fudden`->sudden, `roſes`->roses, `harium`->barium), and
    `VariantSpellings.txt` maps ~4k historical spellings to modern (`easie`->easy,
    `finde`->find). Long-s is covered directly.

    This is the POSITIVE test the certification cascade previously lacked. `flag_kind`
    decided by elimination -- orthographic, else inflection, else "the model built it" --
    so every failure of two negative tests defaulted to the reading that inflates the claim.
    A published lexicon says instead: this form is a known corruption of that word.
    """
    global _DOCUMENTED
    if _DOCUMENTED is None:
        if not (UNDERWOOD / "CorrectionRules.txt").exists():
            sys.exit(f"missing {UNDERWOOD}/ — run --fetch-rulesets")
        d = {}
        for line in (UNDERWOOD / "CorrectionRules.txt").read_text(errors="replace").splitlines():
            p = line.split("\t")
            if len(p) >= 2 and p[0]:
                d[p[0].lower()] = "ocr"
        toks = [t for t in re.split(r"[\t\n]", (UNDERWOOD / "VariantSpellings.txt")
                                    .read_text(errors="replace")) if t]
        for i in range(0, len(toks) - 1, 2):
            d.setdefault(toks[i].lower(), "spelling")
        _DOCUMENTED = d
    return _DOCUMENTED


def mode_fetch_rulesets(args):
    import urllib.request
    UNDERWOOD.mkdir(parents=True, exist_ok=True)
    for name in ("CorrectionRules", "VariantSpellings"):
        dst = UNDERWOOD / f"{name}.txt"
        urllib.request.urlretrieve(UNDERWOOD_URL.format(name), dst)
        print(f"  {dst}  {sum(1 for _ in dst.open(errors='replace')):>7} lines")


def certified(form_group, target=None):
    """Does the group contain a form the model could not have retrieved?

    A member is certifying when it is (a) absent from the attestation set and (b) not
    DOCUMENTED as an OCR error or a historical spelling. Nothing else is asked of it.

    This replaces a per-word type judgement (`technical_composition.flag_kind`: is this a
    distinct lexeme or an orthographic variant?) that was measured and dropped. It removed
    23-40% of certifications, model-correlated with the most heavily scanned model losing
    most, while moving no difference-in-differences point estimate by more than 8% -- it
    bought variance and nothing else. It was also the wrong shape for the construct: the
    unit of this measure is the GROUP, licensed at set-level surprisal over two or more
    distinct lexemes, and a per-word variant/lexeme call imports a word-level assumption
    into a set-level construct. What morphological decomposability is needed is already
    enforced by group membership, before certification is consulted.

    `target` is accepted and ignored; the rule is target-independent. Kept so callers that
    pass it keep working.
    """
    att = attest_set()
    doc = documented()
    return any(w.lower() not in att and w.lower() not in doc for w in form_group or [])


def pilot_sample(n, scope="postcutoff"):
    """Stratified pilot sample: scaffold verdict x stratum x model.

    Balanced rather than random so that the measure is not graded predominantly in one
    cell -- an unstable measure can look stable if the sample is homogeneous.
    """
    items = {i["nbr_id"]: i for i in load_neighborhoods("all")}
    verdicts = {}
    vpath = ANALYSIS / "judge" / "verdicts.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                verdicts[j["item_id"]] = j["judge"]["verdict"]

    cells = defaultdict(list)
    for nid, it in items.items():
        if scope == "postcutoff" and it["is_future"] != "1":
            continue
        v = verdicts.get(it["item_id"])
        if v not in ("scaffolded", "not_scaffolded"):
            continue
        cells[(v, it["is_future"], it["model"])].append(nid)

    rng = random.Random(SEED)
    shuffled = {k: sorted(v) for k, v in cells.items()}
    for v in shuffled.values():
        rng.shuffle(v)
    per = max(1, n // max(1, len(shuffled)))
    picked = [i for k in sorted(shuffled) for i in shuffled[k][:per]]
    depth = per
    while len(picked) < n:
        added = 0
        for k in sorted(shuffled):
            if len(picked) >= n:
                break
            if depth < len(shuffled[k]):
                picked.append(shuffled[k][depth]); added += 1
        if not added:
            break
        depth += 1
    return [items[i] for i in picked[:n]]


def mode_pilot(args):
    """Judge a sample twice and grade the measure on its own stability. No hand labels."""
    rub = load_rubric()
    items = pilot_sample(args.n, args.scope)
    cl = SJ.client()
    JUDGE.mkdir(parents=True, exist_ok=True)
    out_path = JUDGE / f"pilot_{args.model}.jsonl"

    # Resume rather than restart: a crash at item 80 of 200 should cost one item, not 80.
    done, recs = set(), defaultdict(list)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["nbr_id"], r["rep"]))
                recs[r["nbr_id"]].append(r)
        if done:
            print(f"  resuming: {len(done)} call(s) already on disk")
    todo = sum(1 for rep in range(args.replicates) for it in items
               if (it["nbr_id"], rep) not in done)
    print(f"  {len(items)} neighborhoods x {args.replicates} replicates "
          f"= {todo} call(s) to make\n")

    usage = Counter()
    with open(out_path, "a") as fh:
        for rep in range(args.replicates):
            for n, it in enumerate(items, 1):
                if (it["nbr_id"], rep) in done:
                    continue
                v, u = judge_one(cl, rub, it, args.model, args.effort)
                if v is None or any(f not in v for f in FIELDS):
                    usage["bad"] += 1
                    print(f"  ! {it['nbr_id']} rep{rep} unusable — continuing")
                    continue
                usage["in"] += getattr(u, "input_tokens", 0) or 0
                usage["out"] += getattr(u, "output_tokens", 0) or 0
                usage["cache"] += getattr(u, "cache_read_input_tokens", 0) or 0
                rec = {"nbr_id": it["nbr_id"], "rep": rep, "model": it["model"],
                       "target": it["target"], "words": it["words"], "judge": v}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                recs[it["nbr_id"]].append(rec)
                if n % 25 == 0:
                    print(f"    rep {rep + 1}: {n}/{len(items)}")
    if usage["bad"]:
        print(f"\n  {usage['bad']} call(s) returned nothing usable and were skipped")
    if not usage["cache"]:
        print("  ! cache never read")
    print(f"\n  wrote {out_path}")
    grade_pilot(recs)


def grade_pilot(recs):
    """Stability of the form axis. Needs no hand labels -- and proves nothing about
    correctness: a judge can repeat itself perfectly and still be wrong. Agreement with
    the annotator is in `--calibrate`."""
    full = {k: v for k, v in recs.items() if len(v) >= 2}
    print(f"\n  === grading {len(recs)} neighborhoods "
          f"({len(full)} with 2+ replicates) ===")
    first = [v[0] for v in recs.values()]

    print("\n  1. test-retest (same item, independent calls)")
    d = [abs(form_size(v[0]["judge"], v[0]["words"])
             - form_size(v[1]["judge"], v[1]["words"])) for v in full.values()]
    if d:
        print(f"     exact {sum(x == 0 for x in d)/len(d):.3f}   "
              f"within-1 {sum(x <= 1 for x in d)/len(d):.3f}   "
              f"mean |d| {sum(d)/len(d):.2f}")

    print("\n  2. membership stability (Jaccard over returned member lists)")
    js = []
    for v in full.values():
        a = {w.lower() for w in (v[0]["judge"].get("form_group") or [])}
        b = {w.lower() for w in (v[1]["judge"].get("form_group") or [])}
        if a or b:
            js.append(len(a & b) / len(a | b))
    if js:
        print(f"     mean Jaccard {sum(js)/len(js):.3f}   "
              f"identical {sum(x == 1.0 for x in js)/len(js):.3f}   (n={len(js)})")

    print("\n  3. spread")
    vals = [form_size(r["judge"], r["words"]) for r in first]
    mean = sum(vals) / len(vals)
    sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
    mode = Counter(vals).most_common(1)[0]
    print(f"     mean {mean:.2f}  sd {sd:.2f}  modal {mode[0]} at {mode[1]/len(vals):.1%}  "
          f"zero {sum(v == 0 for v in vals)/len(vals):.1%}")

    print("\n  4. how much did enforcing connectivity change the answer?")
    raw = [0 if len(r["judge"].get("form_group") or []) < 2
           else len(r["judge"]["form_group"]) for r in first]
    diff = [a - b for a, b in zip(raw, vals) if a != b]
    print(f"     {len(diff)}/{len(vals)} groups trimmed ({len(diff)/len(vals):.1%}); "
          f"mean reduction {sum(diff)/len(diff):.1f} words" if diff else
          "     no group needed trimming")

    cert = [certified(r["judge"].get("form_group"), r["target"]) for r in first]
    print(f"\n  5. certified groups (contain an assembled form): "
          f"{sum(cert)/len(cert):.3f}  (n={len(cert)})")


def mode_grade_pilot(args):
    path = JUDGE / f"pilot_{args.model}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run --pilot first")
    recs = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["nbr_id"]].append(r)
    grade_pilot(recs)


def mode_validate_cheap(args):
    """Does the automatic measure stand in for the judge?

    Reports agreement on the derived quantities the analysis uses -- certification and
    form size at each threshold -- and then the test that decides it: whether substituting
    the cheap measure changes a lift the paper would print.
    """
    path = JUDGE / f"verdicts_{stem(args)}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run --submit then --collect all first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"  {len(rows)} judged neighborhoods\n")

    pairs = []
    for r in rows:
        jg = r["judge"].get("form_group") or []
        cg = cheap_form_group(r["words"]) if r.get("words") else []
        pairs.append((r, r["form_n"], 0 if len(cg) < 2 else len(cg),
                      certified(jg, r["target"]), certified(cg, r["target"])))

    d = [abs(a - b) for _, a, b, _, _ in pairs]
    print(f"  form size: exact {sum(x == 0 for x in d)/len(d):.3f}   "
          f"within-1 {sum(x <= 1 for x in d)/len(d):.3f}   mean |d| {sum(d)/len(d):.2f}")
    cm = Counter((j, c) for _, _, _, j, c in pairs)
    n = sum(cm.values())
    print(f"\n  certification: agreement {(cm[(True,True)] + cm[(False,False)])/n:.4f}  "
          f"(judge-only {cm[(True,False)]}, cheap-only {cm[(False,True)]})")

    print(f"\n  {'threshold':<14}{'judge':>10}{'cheap':>10}{'agreement':>12}")
    for k in (2, 3, 5):
        jm = [a >= k for _, a, _, _, _ in pairs]
        cm2 = [b >= k for _, _, b, _, _ in pairs]
        print(f"  form >= {k:<7}{sum(jm)/len(jm):>10.3f}{sum(cm2)/len(cm2):>10.3f}"
              f"{sum(x == y for x, y in zip(jm, cm2))/len(jm):>12.3f}")

    verdicts = {}
    vpath = ANALYSIS / "judge" / "verdicts.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                verdicts[j["item_id"]] = j["judge"]["verdict"]
    by = defaultdict(lambda: defaultdict(Counter))
    for r, jn, cn, jc, cc in pairs:
        v = verdicts.get(r["item_id"])
        if v not in ("scaffolded", "not_scaffolded"):
            continue
        cell = by[r["model"]][v == "scaffolded"]
        cell["n"] += 1
        cell["jc"] += jc
        cell["cc"] += cc
        for k in (2, 3, 5):
            cell[f"j{k}"] += jn >= k
            cell[f"c{k}"] += cn >= k
    print("\n  lifts (scaffolded / not), judge vs cheap — the substitution test\n")
    print(f"  {'model':<14}{'measure':<14}{'judge':>10}{'cheap':>10}")
    for m in sorted(by):
        a, b = by[m][True], by[m][False]
        if not a["n"] or not b["n"]:
            continue
        for lab, jk, ck in [("certified", "jc", "cc")] + \
                           [(f"form>={k}", f"j{k}", f"c{k}") for k in (2, 3, 5)]:
            jl = (a[jk]/a["n"]) / (b[jk]/b["n"]) if b[jk] else float("nan")
            cl = (a[ck]/a["n"]) / (b[ck]/b["n"]) if b[ck] else float("nan")
            print(f"  {m:<14}{lab:<14}{jl:>9.2f}x{cl:>9.2f}x")


def mode_did(args):
    """Association between scaffolding and paradigm structure: two-way and three-way.

    Reports both. The TWO-WAY is the association within one stratum -- lambda, the log odds
    ratio of a single 2x2. The THREE-WAY is whether that association differs across the
    cutoff, lambda3 = lam_post - lam_in, and it is the estimand.

    It has to be the three-way. An association inside the post-cutoff arm alone is not
    evidence of composition: scaffolded prompts are SELECTED for supplying morphological
    material, so they raise form cohesion for any model whatever its cutoff -- talkie-web,
    which has seen the post-cutoff words, shows a large two-way association. Only the second
    difference asks whether the donor matters MORE where the target cannot have been
    memorised. So the null for the three-way is homogeneous association -- one odds ratio in
    both strata -- not an odds ratio of 1.

    Every statistic comes from scipy.stats or statsmodels via `assoc`; nothing here is
    hand-rolled. The mode name is kept for the CLI flag `--did`, though the presentation is
    no longer a difference-in-differences: that was a ratio of ratios on the relative-risk
    scale, and this project now reports one statistic, the log odds ratio, throughout.
    """
    arms = {}
    for arm, sc in (("post", "postcutoff"), ("in", "incutoff")):
        p = JUDGE / f"verdicts_{args.model}_context{'' if sc == 'postcutoff' else '_' + sc}.jsonl"
        if not p.exists():
            sys.exit(f"missing {p} — both arms are needed; --submit --scope {sc} --context")
        arms[arm] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    v = scaffold_verdicts()
    # the verdict records do not carry `register`; join it back from the item table
    band = {(i["nbr_id"], i["item_id"], i["model"]): i["register"]
            for i in load_neighborhoods("all")}
    cell = defaultdict(Counter)
    reg = defaultdict(Counter)
    for arm, rows in arms.items():
        for r in rows:
            vd = v.get(r["item_id"])
            if vd not in ("scaffolded", "not_scaffolded"):
                continue
            fn = form_size(r["judge"], r["words"])
            c = cell[(r["model"], arm, vd)]
            c["n"] += 1
            c["cert"] += certified(r["judge"].get("form_group") or [], r["target"])
            for k in (2, 3, 5):
                c[f"f{k}"] += fn >= k
            g = reg[(arm, band.get((r["nbr_id"], r["item_id"], r["model"])) or "unknown")]
            g["n"] += 1
            g["f2"] += fn >= 2

    models = sorted({m for m, _, _ in cell})
    measures = [("certified", "cert")] + [(f"form>={k}", f"f{k}") for k in (2, 3, 5)]

    print(f"\n  cells\n")
    print(f"  {'model':<14}{'arm':<6}{'verdict':<16}{'n':>6}" +
          "".join(f"{lab:>11}" for lab, _ in measures))
    for m in models:
        for arm in ("post", "in"):
            for vd in ("scaffolded", "not_scaffolded"):
                c = cell[(m, arm, vd)]
                print(f"  {m:<14}{arm:<6}{vd:<16}{c['n']:>6}" +
                      "".join(f"{c[f]:>6} {c[f]/c['n']:>4.2f}" for _, f in measures))

    def table(m, arm, f):
        """(a, b, c, d) for scaffolded x outcome, in assoc's cell convention."""
        s, o = cell[(m, arm, "scaffolded")], cell[(m, arm, "not_scaffolded")]
        return s[f], o[f], s["n"] - s[f], o["n"] - o[f]

    print(f"\n  association within each arm — lambda = log odds ratio, tested by G^2\n")
    print(f"  {'model':<14}{'measure':<12}{'arm':<6}{'lambda':>8}{'OR':>8}"
          f"{'95% CI':>16}{'G^2':>9}{'min exp':>9}")
    for m in models:
        for lab, f in measures:
            for arm in ("post", "in"):
                t = table(m, arm, f)
                l, orr, (lo, hi), g2 = AS.association(*t)
                print(f"  {m:<14}{lab:<12}{arm:<6}{l:>8.3f}{orr:>8.2f}"
                      f"{f'[{lo:.2f}, {hi:.2f}]':>16}{g2:>9.1f}"
                      f"{AS.min_expected(*t):>9.1f} {AS.stars(g2)}")

    # The estimand. Not whether scaffolding is associated with paradigm structure -- it is,
    # in every arm above, and a prompt SELECTED for supplying morphological material would
    # raise form cohesion for any model whatever its cutoff. The question is whether that
    # association is STRONGER on targets the model cannot have seen. So the null here is
    # that the two arms share one odds ratio (homogeneous association), not that the odds
    # ratio is 1, and the test is the likelihood ratio against the saturated 2x2x2.
    print(f"\n  interaction — does the association differ across the cutoff?\n")
    print(f"  {'model':<14}{'measure':<12}{'lambda3':>9}{'OR':>8}{'95% CI':>16}{'G^2':>9}")
    for m in models:
        for lab, f in measures:
            l3, orr, (lo, hi), g2 = AS.interaction(table(m, "post", f),
                                                   table(m, "in", f))
            print(f"  {m:<14}{lab:<12}{l3:>9.3f}{orr:>8.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>16}{g2:>9.2f} {AS.stars(g2)}")

    print("\n  lambda is the log odds ratio -- the interaction coefficient of the saturated"
          "\n  log-linear model. Intervals are profile likelihood, so they are the same"
          "\n  apparatus as G^2 rather than a second one. * G^2 > 3.84, ** > 6.63,"
          "\n  *** > 10.83 on 1 df.  `min exp` is the smallest expected cell count: below"
          "\n  about 5 is where chi-squared would fail and G^2 is doing real work."
          "\n  `certified` = a group member that is unattested AND not documented as an OCR"
          "\n  error or historical spelling (Underwood rulesets); `form>=k` is the same"
          "\n  measure with certification removed entirely -- the most inclusive reading.")

    # The triple difference -- each restricted model's interaction divided by talkie-web's --
    # was reported here and is RETIRED, not merely unreported. Talkie-web is ahistorical: it
    # has no cutoff to violate, so it composes across a date that means nothing to it.
    # Dividing by its rate subtracts the phenomenon rather than a confound. Its SE also
    # summed eight cell variances as if independent, across models that share ~2,883 items.
    # See neighborhood_measure.md and NOTES_internal.md. Talkie-web's own interaction is
    # reported in the table above, which is the use it retains.
    # The best-powered contrast in the design, and it uses no scaffolding at all. Restrict to
    # UNSCAFFOLDED neighborhoods and ask whether form cohesion survives the cutoff. A model
    # that has the post-cutoff words in training has no reason to lose cohesion crossing it;
    # a model that does not must fall back on whatever it can assemble. Cells are in the
    # thousands, so unlike the triple difference this is not power-bound.
    print(f"\n  cohesion across the cutoff, UNSCAFFOLDED neighborhoods only\n")
    print(f"  {'model':<14}{'measure':<10}{'in-cutoff':>11}{'post-cutoff':>13}"
          f"{'lambda':>9}{'OR':>7}{'95% CI':>16}{'G^2':>9}")
    for m in models:
        for k in (2, 3, 5):
            a, b = cell[(m, "in", "not_scaffolded")], cell[(m, "post", "not_scaffolded")]
            f = f"f{k}"
            # Cue is the post-cutoff stratum, outcome is form cohesion. A single 2x2, so
            # this is an association and not an interaction -- no second difference here.
            t = (b[f], a[f], b["n"] - b[f], a["n"] - a[f])
            l, orr, (lo, hi), g2 = AS.association(*t)
            print(f"  {m:<14}{'form>=' + str(k):<10}{a[f]/a['n']:>11.3f}"
                  f"{b[f]/b['n']:>13.3f}{l:>9.3f}{orr:>7.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>16}{g2:>9.1f} {AS.stars(g2)}")
    print("\n  An odds ratio below 1 means cohesion FALLS crossing the cutoff.")

    print(f"\n  measurement check — does the judge group differently by date band?\n")
    print(f"  {'arm':<6}{'register':<22}{'n':>7}{'form>=2':>10}")
    for (arm, r) in sorted(reg):
        g = reg[(arm, r)]
        print(f"  {arm:<6}{r:<22}{g['n']:>7}{g['f2']/g['n']:>10.3f}")


def mode_allocation(args):
    """Why the in-cutoff cells are equal-n: where the DiD variance would sit either way.

    The post-cutoff arm is already bought in full -- all 426 scaffolded post-cutoff items
    exist and all were judged -- so its precision is fixed and cannot be improved at any
    price. The only question is how to spend the matching budget so the NEW arm does not
    become the bottleneck. Mirroring the post-cutoff cell sizes does exactly that.
    """
    import math

    path = JUDGE / f"verdicts_{args.model}_context.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — the post-cutoff pass is the reference arm")
    v = scaffold_verdicts()
    cell = defaultdict(Counter)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vd = v.get(r["item_id"])
        if vd not in ("scaffolded", "not_scaffolded"):
            continue
        c = cell[(r["model"], vd)]
        c["n"] += 1
        fn = form_size(r["judge"], r["words"])
        for k in (2, 3, 5):
            c[f"f{k}"] += fn >= k

    def var(n, k):                       # delta-method variance of log(k/n)
        p = k / n
        return (1 - p) / (n * p) if 0 < p < 1 else float("inf")

    models = sorted({m for m, _ in cell})
    print(f"\n  post-cutoff arm as judged ({sum(c['n'] for c in cell.values())} items)\n")
    print(f"  {'model':<14}{'arm':<16}{'n':>6}{'f>=2':>7}{'f>=3':>7}{'f>=5':>7}")
    for m in models:
        for a in ("scaffolded", "not_scaffolded"):
            c = cell[(m, a)]
            print(f"  {m:<14}{a:<16}{c['n']:>6}"
                  + "".join(f"{c[f'f{k}']:>7}" for k in (2, 3, 5)))

    print(f"\n  share of difference-in-differences variance landing in the in-cutoff arm\n")
    print(f"  {'allocation':<26}{'form>=2':>10}{'form>=3':>10}{'form>=5':>10}")
    plans = {"proportional mirror": {(m, a): cell[(m, a)]["n"] for m in models
                                     for a in ("scaffolded", "not_scaffolded")},
             f"equal-n ({PER_CELL}/cell)": {(m, a): PER_CELL for m in models
                                            for a in ("scaffolded", "not_scaffolded")}}
    for lab, alloc in plans.items():
        row = []
        for k in (2, 3, 5):
            post = sum(var(cell[c]["n"], cell[c][f"f{k}"]) for c in cell)
            # in-cutoff rates are unknown; assume each cell's post-cutoff not_scaffolded rate
            inc = sum(var(alloc[c], alloc[c] * cell[(c[0], "not_scaffolded")][f"f{k}"]
                          / cell[(c[0], "not_scaffolded")]["n"]) for c in alloc)
            row.append(inc / (post + inc))
        print(f"  {lab:<26}" + "".join(f"{x:>9.1%} " for x in row))
    print(f"\n  totals: proportional {sum(plans['proportional mirror'].values())}, "
          f"equal-n {sum(plans[f'equal-n ({PER_CELL}/cell)'].values())}")


def mode_cost(args):
    rub = load_rubric()
    items = load_neighborhoods(args.scope, args.match_pos)
    sp = system_prompt(rub, args.context)
    sys_tok = len(sp) // 4
    per = len(user_message(items[0], args.context)) // 4 if items else 120
    print(f"\n  rubric ≈ {sys_tok} tokens; {args.scope} scope = {len(items)} neighborhoods\n")
    print(f"  {'model':<22}{'caches?':>9}{'batch':>10}{'realtime':>11}")
    for m, (pin, pout) in sorted(SJ.PRICES.items()):
        ok = sys_tok >= SJ.CACHE_MIN[m]
        sysc = sys_tok * pin * (0.1 if ok else 1.0)
        tot = len(items) * (sysc + per * pin + 200 * pout) / 1e6
        print(f"  {m:<22}{'yes' if ok else 'NO':>9}{tot*0.5:>9.0f}${tot:>10.0f}$")
    if sys_tok < SJ.CACHE_MIN.get(args.model, 0):
        print(f"\n  ! {args.model} needs a {SJ.CACHE_MIN[args.model]}-token prefix to "
              f"cache; the rubric is {sys_tok}.")


def mode_submit(args):
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    rub = load_rubric()
    items = load_neighborhoods(args.scope, args.match_pos)
    cl = SJ.client()
    print(f"  pre-warming the rubric cache ...")
    warm = request_params(rub, items[0], args.model, args.effort, args.context)
    warm["max_tokens"] = 1
    warm.pop("output_config", None)
    try:
        cl.messages.create(**warm)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (pre-warm skipped: {e})")

    per_req = len(json.dumps(request_params(rub, items[0], args.model, args.effort,
                                            args.context))) + 256
    shard = max(1, min(20000, int(150 * 1024 * 1024 / per_req)))
    JUDGE.mkdir(parents=True, exist_ok=True)
    man_path = JUDGE / f"batches_{stem(args)}.json"
    ids = []

    def save():
        man_path.write_text(json.dumps(
            {"model": args.model, "scope": args.scope, "n_items": len(items),
             "shard_size": shard, "batch_ids": ids}, indent=2))

    for start in range(0, len(items), shard):
        chunk = items[start:start + shard]
        reqs = [Request(custom_id=f"nbr-{start + i}",
                        params=MessageCreateParamsNonStreaming(
                            **request_params(rub, it, args.model, args.effort,
                                             args.context)))
                for i, it in enumerate(chunk)]
        try:
            b = cl.messages.batches.create(requests=reqs)
        except Exception as e:                               # noqa: BLE001
            save()
            sys.exit(f"\n  ! shard {len(ids)+1} failed: {e}\n"
                     f"  {len(ids)} shard(s) recorded in {man_path}; resume from item {start}.")
        ids.append(b.id)
        save()
        print(f"  shard {len(ids)}: {len(reqs)} requests -> {b.id}")
    print(f"\n  {len(items)} requests across {len(ids)} batch(es) on {args.model}")
    print(f"  collect with: local/bin/python neighborhood_judge.py --collect all "
          f"--model {args.model}")


def mode_collect(args):
    cl = SJ.client()
    man_path = JUDGE / f"batches_{stem(args)}.json"
    if args.collect == "all":
        if not man_path.exists():
            sys.exit(f"no manifest at {man_path}")
        man = json.loads(man_path.read_text())
        batch_ids, scope = man["batch_ids"], man.get("scope", "postcutoff")
    else:
        batch_ids, scope = [args.collect], args.scope

    items = load_neighborhoods(scope)
    pending = []
    for bid in batch_ids:
        b = cl.messages.batches.retrieve(bid)
        print(f"  {bid}: {b.processing_status}  {b.request_counts}")
        if b.processing_status != "ended":
            pending.append(bid)
    if pending:
        print(f"\n  {len(pending)} batch(es) still running — nothing written")
        return

    out = JUDGE / f"verdicts_{stem(args)}.jsonl"
    n, failed = 0, []
    with open(out, "w") as fh:
        for bid in batch_ids:
            for res in cl.messages.batches.results(bid):
                if res.result.type != "succeeded":
                    failed.append((res.custom_id, res.result.type))
                    continue
                idx = int(res.custom_id.split("-")[1])
                text = next((x.text for x in res.result.message.content
                             if x.type == "text"), None)
                if text is None:
                    failed.append((res.custom_id, "no text block"))
                    continue
                try:
                    v = json.loads(text)
                except json.JSONDecodeError:
                    failed.append((res.custom_id, "unparseable JSON"))
                    continue
                if any(f not in v for f in FIELDS):
                    failed.append((res.custom_id, "missing fields"))
                    continue
                it = items[idx]
                f_n = form_size(v, it["words"])
                # `words` is stored, not just the derived count: --validate-cheap has to
                # recompute the automatic group from the SAME presentation the judge saw
                # (order is seeded, but storing it removes any doubt), and re-deriving it
                # later from scaffold_subset.csv would silently drift if that file changed.
                fh.write(json.dumps({"nbr_id": it["nbr_id"], "item_id": it["item_id"],
                                     "model": it["model"], "is_future": it["is_future"],
                                     "target": it["target"], "words": it["words"],
                                     "form_n": f_n, "judge": v}) + "\n")
                n += 1
    if failed:
        print(f"  {len(failed)} unusable ({100*len(failed)/max(n+len(failed),1):.3f}%)")
        for r, c in Counter(r for _, r in failed).most_common():
            print(f"      {c:>5}  {r}")
    print(f"  wrote {n} verdicts to {out}")


def mode_cross_check(args):
    """Compare two models' verdicts on the same neighborhoods.

    The rubric's `withheld.correlated_instruments` block requires this before the
    association is reported: selection and outcome otherwise run on one model and can
    share lexical priors even though they share no evidence.
    """
    a = JUDGE / f"verdicts_{DEFAULT_MODEL}.jsonl"
    b = JUDGE / f"verdicts_{args.model}.jsonl"
    for p in (a, b):
        if not p.exists():
            sys.exit(f"missing {p} — run --submit/--collect for that model first")
    A = {json.loads(l)["nbr_id"]: json.loads(l) for l in a.read_text().splitlines() if l.strip()}
    B = {json.loads(l)["nbr_id"]: json.loads(l) for l in b.read_text().splitlines() if l.strip()}
    both = sorted(set(A) & set(B))
    print(f"  {len(both)} neighborhoods judged by both {DEFAULT_MODEL} and {args.model}\n")
    for axis in ("form_n",):
        d = [abs(A[i][axis] - B[i][axis]) for i in both]
        print(f"    {axis:<10} exact {sum(x==0 for x in d)/len(d):.3f}   "
              f"within-1 {sum(x<=1 for x in d)/len(d):.3f}   "
              f"mean |diff| {sum(d)/len(d):.2f}")
    print(f"\n    {'k':>3}{'agreement on form >= k':>26}")
    for k in (2, 3, 5):
        ag = sum((A[i]["form_n"] >= k) == (B[i]["form_n"] >= k) for i in both)
        print(f"    {k:>3}{ag/len(both):>26.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--calibration-set", action="store_true",
                    help="choose which Sheet A items to hand-label")
    ap.add_argument("--calibrate", action="store_true",
                    help="judge the hand-labelled items and report agreement")
    ap.add_argument("--pilot", action="store_true",
                    help="judge a stratified sample twice and grade both dimensions")
    ap.add_argument("--grade-pilot", action="store_true",
                    help="re-grade an existing pilot without new calls")
    ap.add_argument("--replicates", type=int, default=2,
                    help="calls per item; 2 gives test-retest")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", metavar="BATCH_ID")
    ap.add_argument("--validate-cheap", action="store_true",
                    help="does the automatic measure stand in for the judge?")
    ap.add_argument("--cross-check", action="store_true",
                    help="compare this --model's verdicts against the default model's")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SJ.PRICES))
    ap.add_argument("--scope", default="postcutoff",
                    choices=["postcutoff", "incutoff", "all"])
    ap.add_argument("--no-pos-match", dest="match_pos", action="store_false",
                    help="draw the in-cutoff sample without matching the post-cutoff "
                         "slot-POS distribution (sensitivity check)")
    ap.add_argument("--fetch-rulesets", action="store_true",
                    help="download Underwood's DataMunging rulesets into cache/underwood/")
    ap.add_argument("--did", action="store_true",
                    help="the difference-in-differences across both arms (the estimand)")
    ap.add_argument("--allocation", action="store_true",
                    help="show why the in-cutoff cells are equal-n rather than proportional")
    ap.add_argument("--n", type=int, default=CALIB_N)
    ap.add_argument("--heldout", action="store_true",
                    help="score only the held-out items (rubric frozen)")
    ap.add_argument("--context", action="store_true",
                    help="A+B: supply the slot's grammatical category and the "
                         "citation date band (never the passage)")
    ap.add_argument("--effort", default="off",
                    choices=["off", "low", "medium", "high"])
    args = ap.parse_args()

    if args.fetch_rulesets:
        mode_fetch_rulesets(args)
    elif args.did:
        mode_did(args)
    elif args.allocation:
        mode_allocation(args)
    elif args.show_prompt:
        mode_show_prompt(args)
    elif args.calibration_set:
        mode_calibration_set(args)
    elif args.calibrate:
        mode_calibrate(args)
    elif args.pilot:
        mode_pilot(args)
    elif args.grade_pilot:
        mode_grade_pilot(args)
    elif args.cost:
        mode_cost(args)
    elif args.submit:
        mode_submit(args)
    elif args.collect:
        mode_collect(args)
    elif args.validate_cheap:
        mode_validate_cheap(args)
    elif args.cross_check:
        mode_cross_check(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
