#!/usr/bin/env python3
"""LLM-as-judge for scaffold detection: calibrate, then judge the corpus.

WHY A JUDGE.  Deciding whether a prompt supplies material the target could be built
from is a multidimensional judgment, and every attempt to capture it in a rule needed
hand-set constants that could not be derived -- minimum donor length, shared-character
counts, an edit-distance cutoff, a morpheme-frequency cut.  Each was individually
defensible and none was justified.  Worse, the rule silently missed cases the argument
depends on: `acetylcholine` -> `cholinergic`, the stem donor in the paper's own flagship
citation, was invisible because the shared portion sits word-medially.

The judge replaces those constants with one assumption -- that a capable model can apply
a written definition -- and the definition is published as `scaffold_judge_rubric.json`
rather than buried in code.  The rubric IS the methodology.

THE PRE-FILTER IS OPTIONAL, AND `--all` TURNS IT OFF.  A cheap orthographic filter can
select candidates (LCS >= 4 anywhere, or a shared final-3 that some prefix word carries),
which takes 4,586 of 50,350 items at 2.18x a permutation null -- reported by `--pool`,
not assumed.  But judging the whole corpus costs about the same as judging a wide filtered
pool, so `--all` is the honest default: it removes the last component whose thresholds
were chosen rather than derived, and it answers "what does the filter miss?" by
measurement instead of estimate.  Keep the filter only when cost genuinely binds.

FIREWALL.  The judge sees the citation and the target and nothing else.  No model
predictions, no rank, no model identity, no experimental condition.  Selection is
prompt-side; the outcome measured against it is neighborhood-side; neither may see the
other, or the analysis is circular.

Run:
    local/bin/python scaffold_judge.py --pool                  # build candidates, report null
    local/bin/python scaffold_judge.py --calibrate             # judge the hand-labelled items
    local/bin/python scaffold_judge.py --cost                  # price the full run
    local/bin/python scaffold_judge.py --submit --all          # Batch API, whole corpus
    local/bin/python scaffold_judge.py --collect all           # retrieve, reassemble shards

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) for anything that calls the API.
`--pool` and `--cost` are offline.
"""

import argparse
import csv
import json
import os
import re
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evals.cloze import extract_prefix          # noqa: E402  (reuse, do not reimplement)
import neighborhood_analysis as NA              # noqa: E402  (levenshtein)
from scaffold_subset import item_id             # noqa: E402  (same id scheme)

RESULTS = Path("results")
ANALYSIS = Path("analysis")
JUDGE = ANALYSIS / "judge"
RUBRIC = Path("scaffold_judge_rubric.json")
csv.field_size_limit(10 ** 7)

# Sonnet 5 rather than Haiku 4.5: Haiku needs a 4096-token prefix to cache and the rubric
# renders to ~2400, so on Haiku the rubric is re-billed in full on every item and the
# corpus run costs MORE than on Sonnet ($85 vs $60) for a less capable judge. Calibrate
# before committing either way -- competence on this rubric is a hypothesis, not a given.
DEFAULT_MODEL = "claude-sonnet-5"
SEED = 20260807

# Pre-filter constants. Both derived from the lift curve against a permutation null
# (see the module docstring); neither is a free parameter to tune by eye.
MIN_LCS = 4          # shortest shared contiguous run that counts as stem material
SUFFIX_LEN = 3       # length of the word-final match that catches affix families
MIN_DONOR_LEN = 4    # a donor shorter than this cannot carry either signal
NEAR_IDENTICAL = 0.34  # normalized edit distance under which a pair is listed regardless
                       # of run length -- substitution variants are the scratchpad's main
                       # move (ferrinase/ferratase, acetylergic/acetoergic)

# Citation tail shown to the judge. 320 rather than a longer window for two reasons.
# First, calibration ran on the hand-labelling sheet's `prefix_tail`, which is capped at
# 320 -- judging the corpus on 700-char contexts would mean the reported agreement was
# measured on a different task from the one being run. Second, overlaps are extracted from
# exactly the text shown, so window length silently changes which donors exist at all; a
# wider window is a different instrument, not a more generous one.
PREFIX_CHARS = 320
SHARD_MAX = 20000    # requests per batch; the 256 MB body cap binds before the 100k cap
WORD_RE = re.compile(r"[A-Za-z]+")

# The rubric's output fields. A verdict missing any of them is discarded rather than
# half-counted -- structured outputs make this rare (28 of 50,350) but not impossible.
FIELDS = ("material", "donor", "shared", "usable", "recruitment", "evidence",
          "verdict", "rationale")


# --------------------------------------------------------------------------- #
# Candidate pool
# --------------------------------------------------------------------------- #
def lcs_len(a, b):
    """Length of the longest shared contiguous run. Position-agnostic on purpose:
    `cholin-` is word-initial in `cholinergic` and word-medial in `acetylcholine`, and an
    edge-anchored measure cannot see that."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            if ca == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def lcs_run(a, b):
    """(length, the shared run itself) — the substring, not just its size, because the
    judge is shown the actual letters rather than asked to find them."""
    prev = [0] * (len(b) + 1)
    best, end_a = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            if ca == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end_a = cur[j], i
        prev = cur
    return best, a[end_a - best:end_a]


def _where(sub, word):
    if word.startswith(sub):
        return "start"
    if word.endswith(sub):
        return "end"
    return "middle"


def overlaps(target, words, min_run=3, top=12):
    """Every passage word sharing a run of letters with the target, computed exhaustively.

    This exists because the judge cannot do it. Asked to find overlaps itself, Sonnet 5
    reported "no word in the passage shares any letters with 'southern'" for a passage
    containing `northern` (shared `thern`), and the same for `chromosomes`/`ribosome`
    (shared `osome`) -- five of seven calibration errors were denied overlaps of 4-5
    characters. Exhaustive character comparison across ~25 candidates is not what a
    language model is good at; deciding whether a shared run is usable and whether the
    passage recruits it, is. So code finds, the judge judges.

    Position is reported because it carries the stem/affix distinction the judge would
    otherwise have to infer: `thern` ends both words, `cholin` starts one and sits mid
    in the other. Edit distance is reported alongside the run because the two answer
    different questions: a 3-character run at 1 edit is a near-identical pair
    (`telemer`/`telomer`), the same run at 8 edits is a coincidence.
    """
    out = []
    for w in set(words):
        if w == target or len(w) < MIN_DONOR_LEN:
            continue
        n, sub = lcs_run(w, target)
        d = NA.levenshtein(w, target)
        nd = d / max(len(w), len(target))
        # Two admission routes, because one substitution mid-word halves a contiguous run:
        # `telemer` and `telomer` differ by ONE character yet share a run of only 3.
        # Run length alone would show that pair as a weak overlap, and could rank it below
        # a longer but coincidental match elsewhere in the passage.
        if n < min_run and nd > NEAR_IDENTICAL:
            continue
        out.append((n, nd, d, w, sub))
    # Rank by the stronger of the two signals, so near-identical pairs are never crowded
    # out of the table by longer, more distant coincidences.
    out.sort(key=lambda t: (-max(t[0] / 8.0, 1.0 - t[1]), t[3]))
    return [{"word": w, "shared": sub, "n": n, "edits": d,
             "in_donor": _where(sub, w), "in_target": _where(sub, target)}
            for n, nd, d, w, sub in out[:top]]


def item_stats(target, words):
    """(longest shared run with any prefix word, whether a prefix word shares the final-3)."""
    best = 0
    suffix_hit = False
    tail = target[-SUFFIX_LEN:] if len(target) > SUFFIX_LEN else None
    for w in words:
        if w == target or len(w) < MIN_DONOR_LEN:
            continue
        n = lcs_len(w, target)
        if n > best:
            best = n
        if tail and w.endswith(tail):
            suffix_hit = True
    return best, suffix_hit


def selected(target, words):
    best, suffix_hit = item_stats(target, words)
    return best >= MIN_LCS or suffix_hit


def load_corpus(model="talkie-base"):
    """One row per cloze item: target, the prefix the model actually saw, its words."""
    path = RESULTS / f"cloze_{model}_details.csv"
    if not path.exists():
        sys.exit(f"missing {path}")
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = (r.get("target_word") or "").strip().lower()
            if not t:
                continue
            prefix = extract_prefix(r.get("text") or "", t)
            if prefix is None:
                continue
            out.append({
                "target": t,
                "prefix": prefix,
                "words": list({w.lower() for w in WORD_RE.findall(prefix)}),
                "year": r.get("year"),
                "is_future": r.get("is_future"),
            })
    return out


def build_pool(items, report_null=False):
    pool = [it for it in items if selected(it["target"], it["words"])]
    null_n = None
    if report_null:
        # Permutation null: the same target scored against another item's prefix. This is
        # the false-positive rate of the pre-filter, and it is reported rather than assumed.
        rng = random.Random(SEED)
        perm = list(range(len(items)))
        rng.shuffle(perm)
        null_n = sum(1 for i, it in enumerate(items)
                     if selected(it["target"], items[perm[i]]["words"]))
    return pool, null_n


# --------------------------------------------------------------------------- #
# Prompt assembly — the rubric JSON is the single source of truth
# --------------------------------------------------------------------------- #
def load_rubric():
    if not RUBRIC.exists():
        sys.exit(f"missing {RUBRIC}")
    return json.loads(RUBRIC.read_text())


def system_prompt(rub):
    """Render the rubric JSON into the judge's system prompt.

    Everything the judge is told comes from the JSON, so the published file and the
    deployed instrument cannot drift apart. `withheld` and `notes_for_maintainers` are
    never rendered -- the first would contaminate the finding the judge helps test.
    """
    L = [f"# {rub['purpose']}", ""]
    fr = rub["framing"]
    L += [fr["why_this_matters"], "",
          f"**What you are given.** {fr['what_you_are_given']}", "",
          f"**What you must not use.** {fr['what_you_must_not_use']}", ""]

    for dim in sorted(rub["dimensions"], key=lambda d: d["order"]):
        L.append(f"## Dimension {dim['order']} — {dim['question']}")
        L.append("")
        if dim.get("principle"):
            L += [dim["principle"], ""]
        if dim.get("scope_warning"):
            L += [dim["scope_warning"], ""]
        for val, desc in dim.get("definitions", {}).items():
            L.append(f"- **{val}** — {desc}")
        if dim.get("test"):
            L += ["", f"**The test.** {dim['test']}"]
        for wd in dim.get("worked_distinctions", []):
            L.append(f"- *{wd['pair']}* → **{wd['verdict']}**. {wd['reason']}")
        for g in dim.get("guidance", []):
            L.append(f"- {g}")
        L.append("")

    v = rub["verdict"]
    L += ["## The verdict", ""]
    for val, rule in v["rules"].items():
        L.append(f"- **{val}** — {rule}")
    L.append("")

    L += ["## What is explicitly NOT a criterion", ""]
    for nc in rub["explicit_non_criteria"]:
        L.append(f"- **{nc['criterion']}.** {nc['why_excluded']}")
    L.append("")

    L += ["## Worked examples", ""]
    for ex in rub["worked_examples"]:
        L += [f"**{ex['label']}**", "",
              f"> …{ex['passage_tail']} ___", "",
              f"Target: `{ex['target']}`",
              f"Overlaps shown: {ex.get('overlaps_shown', 'none')}",
              f"Answer: `{json.dumps(ex['expected'], ensure_ascii=False)}`",
              f"Why: {ex['commentary']}", ""]

    L += ["## Output", "", rub["output_contract"]["instruction"], ""]
    for f_, d in rub["output_contract"]["fields"].items():
        L.append(f"- `{f_}` — {d}")
    return "\n".join(L)


def output_schema(rub):
    """JSON schema for structured outputs, derived from the same contract."""
    dims = {d["id"]: d for d in rub["dimensions"]}
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "enum": dims["material"]["values"]},
                "donor": {"type": "string"},
                "shared": {"type": "string"},
                "usable": {"type": "string", "enum": dims["usable"]["values"]},
                "recruitment": {"type": "string", "enum": dims["recruitment"]["values"]},
                "evidence": {"type": "string"},
                "verdict": {"type": "string", "enum": rub["verdict"]["values"]},
                "rationale": {"type": "string"},
            },
            "required": ["material", "donor", "shared", "usable", "recruitment",
                         "evidence", "verdict", "rationale"],
            "additionalProperties": False,
        },
    }


def user_message(item):
    """Passage, target, and the pre-computed overlap table.

    Overlaps are extracted from exactly the text shown, so the judge is never told about
    a donor it cannot see.
    """
    tail = item["prefix"][-PREFIX_CHARS:]
    truncated = len(item["prefix"]) > PREFIX_CHARS
    words = [w.lower() for w in WORD_RE.findall(tail)]
    ov = overlaps(item["target"], words)
    if ov:
        rows = "\n".join(
            "  {w:<18} shares \"{s}\" ({n}) — {a} of {w}, {b} of {t}; "
            "{e} edit(s) apart overall".format(
                w=o["word"], s=o["shared"], n=o["n"], a=o["in_donor"],
                b=o["in_target"], t=item["target"], e=o["edits"])
            for o in ov)
    else:
        rows = "  none — no word in the passage shares 3 or more consecutive letters."
    return (f"PASSAGE:\n{'…' if truncated else ''}{tail} ___\n\n"
            f"TARGET: {item['target']}\n\n"
            f"OVERLAPS (computed exhaustively — do not search for others):\n{rows}")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def client():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  note: ANTHROPIC_API_KEY unset — relying on an `ant auth login` profile",
              file=sys.stderr)
    return anthropic.Anthropic()


def request_params(rub, item, model, effort="off"):
    """One judging request.

    The rubric goes in a cached system block: it is identical across every item, so it is
    written once and read at a tenth of the price thereafter. Structured outputs make the
    eight rubric fields a schema rather than prose to parse. No thinking parameter --
    Haiku 4.5 predates adaptive thinking and rejects `effort`.
    """
    p = {
        "model": model,
        # Headroom, not a target -- max_tokens is a cap, so a generous value costs
        # nothing. It matters because on Sonnet 5 adaptive thinking is ON BY DEFAULT and
        # max_tokens caps thinking + response TOGETHER: at 512 the thinking consumed the
        # whole budget and no text block was emitted at all. 1536 because one item in 115
        # truncated at 1024; max_tokens is a cap, so headroom is free.
        "max_tokens": 1536,
        "system": [{"type": "text", "text": system_prompt(rub),
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "output_config": {"format": output_schema(rub)},
        "messages": [{"role": "user", "content": user_message(item)}],
    }
    # Thinking is OFF by default here. It bills as output, output dominates this run's
    # cost, and this is rubric-driven classification against an explicit written standard
    # -- the case where deliberation buys least. `--effort low|medium|high` turns it back
    # on if a calibration run ever shows the judgment needs it.
    if model in ("claude-sonnet-5", "claude-opus-5"):
        if effort == "off":
            p["thinking"] = {"type": "disabled"}
        else:
            p["thinking"] = {"type": "adaptive"}
            p["output_config"]["effort"] = effort
    return p


def judge_one(cl, rub, item, model, effort="off"):
    resp = cl.messages.create(**request_params(rub, item, model, effort))
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("truncated at max_tokens — raise it or lower --effort")
    if resp.stop_reason == "refusal":
        return {"verdict": "unsure", "rationale": "refused", "material": "none",
                "donor": "", "shared": "", "usable": "unsure",
                "recruitment": "none", "evidence": ""}, resp.usage
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"no text block (stop_reason={resp.stop_reason}, "
                           f"blocks={[b.type for b in resp.content]})")
    v = json.loads(text)
    missing = set(rub["output_contract"]["fields"]) - set(v)
    if missing:
        raise RuntimeError(f"schema not honoured, missing {sorted(missing)}")
    return v, resp.usage


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def mode_pool(args):
    items = load_corpus()
    pool, null_n = build_pool(items, report_null=True)
    JUDGE.mkdir(parents=True, exist_ok=True)
    out = JUDGE / "candidates.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "target", "year", "is_future", "prefix"])
        for i, it in enumerate(pool):
            w.writerow([i, it["target"], it["year"], it["is_future"], it["prefix"]])
    n = len(items)
    print(f"  corpus items with a recoverable prefix : {n}")
    print(f"  selected by the pre-filter             : {len(pool)}  ({100*len(pool)/n:.1f}%)")
    print(f"  permutation null (false-positive rate) : {null_n}  ({100*null_n/n:.1f}%)")
    print(f"  lift over chance                       : {len(pool)/max(null_n,1):.2f}x")
    print(f"\n  wrote {out}")


def _labelled_items():
    """The hand-labelled prompts, joined to their human verdict. Calibration set."""
    sheet = ANALYSIS / "scaffold_prompt_labels.csv"
    log = ANALYSIS / "hand_labeling" / "frame_labels.jsonl"
    if not sheet.exists() or not log.exists():
        sys.exit("missing the hand-labelled set — run scaffold_subset.py and label_frames.py")
    hand = {}
    for line in log.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                hand[r["id"]] = r
            except json.JSONDecodeError:
                pass
    out = []
    for r in csv.DictReader(open(sheet)):
        h = hand.get(r["id"])
        if not h or h.get("recruits") is None:
            continue
        out.append({
            "id": r["id"],
            "target": r["target_word"].strip().lower(),
            "prefix": r["prefix_tail"],
            "human": bool(h["recruits"]),
        })
    return out


def mode_calibrate(args):
    rub = load_rubric()
    items = _labelled_items()
    if args.n:
        items = items[:args.n]
    cl = client()
    JUDGE.mkdir(parents=True, exist_ok=True)
    out_path = JUDGE / f"calibration_{args.model}.jsonl"

    cm = Counter()
    errs = Counter()
    dims = defaultdict(Counter)
    tok_in = tok_out = tok_cache = tok_write = 0
    with open(out_path, "w") as fh:
        for k, it in enumerate(items, 1):
            try:
                verdict, usage = judge_one(cl, rub, it, args.model, args.effort)
            except Exception as e:                       # noqa: BLE001
                errs[type(e).__name__ + ": " + str(e)[:90]] += 1
                continue
            tok_in += usage.input_tokens
            tok_out += usage.output_tokens
            tok_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
            tok_cache += getattr(usage, "cache_read_input_tokens", 0) or 0
            j = verdict["verdict"] == "scaffolded"
            cm[(j, it["human"])] += 1
            dims["material"][verdict["material"]] += 1
            dims["recruitment"][verdict["recruitment"]] += 1
            fh.write(json.dumps({**it, "judge": verdict}) + "\n")
            if k % 25 == 0:
                print(f"    {k}/{len(items)}…", file=sys.stderr)

    tp, fp, fn, tn = cm[(1, 1)], cm[(1, 0)], cm[(0, 1)], cm[(0, 0)]
    n = tp + fp + fn + tn
    if not n:
        sys.exit("no items scored")
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    ceiling = load_rubric()["calibration"]["ceiling"]

    print(f"\n  {args.model} vs the human labels  (n={n})")
    print(f"    agreement {(tp+tn)/n:.3f}   precision {pr:.3f}   recall {rc:.3f}   F1 {f1:.3f}")
    print(f"    annotator self-agreement (ceiling)  {ceiling:.3f}")
    print(f"    share of ceiling reached            {((tp+tn)/n)/ceiling:.1%}")
    print(f"\n    | | human: recruits | human: no |")
    print(f"    | judge: scaffolded | {tp} | {fp} |")
    print(f"    | judge: not        | {fn} | {tn} |")
    for d in ("material", "recruitment"):
        print(f"\n    {d}: " + ", ".join(f"{k}={v}" for k, v in dims[d].most_common()))
    scored = tp + fp + fn + tn
    print(f"\n    tokens/item: {tok_in/scored:.0f} fresh in, {tok_cache/scored:.0f} cache-read,"
          f" {tok_out/scored:.0f} out   (cache writes {tok_write})")
    if tok_cache == 0:
        print("    ! cache never read — the corpus run would cost ~3x the estimate")
    pin, pout = PRICES[args.model]
    per = ((tok_in + 0.1 * tok_cache) * pin + tok_out * pout) / scored / 1e6
    print(f"    measured ${per:.5f}/item -> corpus (50350, batched): ${per*50350*0.5:.0f}")
    if errs:
        print("    errors:")
        for e, c in errs.most_common():
            print(f"      {c}x  {e}")
    print(f"    wrote {out_path}")
    if ((tp + tn) / n) > ceiling:
        print("\n    NOTE: agreement exceeds the annotator's own self-agreement. That is a"
              "\n    warning, not a success — it suggests fitting one annotator's idiosyncrasy.")


PRICES = {  # $ per million tokens, (input, output); see the claude-api skill
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),      # introductory, through 2026-08-31
    "claude-opus-5": (5.00, 25.00),
}
CACHE_MIN = {"claude-haiku-4-5": 4096, "claude-sonnet-5": 1024, "claude-opus-5": 512}


def mode_cost(args):
    rub = load_rubric()
    sp = system_prompt(rub)
    sys_tok = len(sp) // 4                      # rough; --calibrate reports measured usage
    items = load_corpus()
    pool, _ = build_pool(items)
    print(f"  rubric ≈ {sys_tok} tokens; pool = {len(pool)} items; corpus = {len(items)}")
    print(f"\n  {'model':<20}{'caches?':>9}{'pool':>10}{'corpus':>10}   (Batch API, 50% off)")
    for m, (pin, pout) in PRICES.items():
        caches = sys_tok >= CACHE_MIN[m]
        per_in = (sys_tok * (0.1 if caches else 1.0) + 200)
        per = (per_in * pin + 150 * pout) / 1e6 * 0.5
        print(f"  {m:<20}{('yes' if caches else 'NO'):>9}"
              f"{'$' + format(per * len(pool), '.0f'):>10}"
              f"{'$' + format(per * len(items), '.0f'):>10}")
    for m in PRICES:
        if sys_tok < CACHE_MIN[m]:
            print(f"\n  ! {m} needs a {CACHE_MIN[m]}-token prefix to cache; the rubric is "
                  f"{sys_tok}.\n    Below that the rubric is re-billed in full on every "
                  f"item — a ~10x input cost.")


def mode_submit(args):
    """Batch API: half price, and 50k items is far under the 100k-per-batch cap."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    rub = load_rubric()
    items = load_corpus()
    pool = items if args.all else build_pool(items)[0]
    if args.n:
        pool = pool[:args.n]
    cl = client()

    # Pre-warm the cache before submitting. A cache entry only becomes readable once the
    # first response has begun, so without this the batch's initial burst of concurrent
    # requests all miss and pay full price for the ~5.2k-token rubric. A max_tokens=0
    # request runs prefill, writes the entry, and returns immediately with no output
    # tokens billed. The 1h TTL is the longest available, so a long batch will let it
    # lapse and re-write occasionally -- that is a handful of writes at ~$0.02 each, not
    # a per-item cost, so it does not need managing.
    warm = request_params(rub, pool[0], args.model, args.effort)
    warm["max_tokens"] = 0
    warm.pop("output_config", None)          # rejected alongside max_tokens=0
    try:
        cl.messages.create(**warm)
        print("  cache pre-warmed")
    except Exception as e:                   # noqa: BLE001
        print(f"  ! pre-warm failed ({e}); the first requests will pay full price")

    # Each batch request carries its own copy of the system prompt, so the 256 MB
    # per-batch cap binds long before the 100k-request cap: ~10 KB per request means
    # roughly 25k requests fits. Shard well under that and submit in sequence.
    # Size the shard from a REAL serialized request, not an estimate. A character-count
    # approximation undershot the JSON body by ~7% and put the largest shard at 213 MB
    # against a 256 MB cap -- a single long citation could have failed the submission.
    # Target 150 MB so the margin absorbs the spread in passage length.
    per_req = len(json.dumps(request_params(rub, pool[0], args.model, args.effort))) + 256
    shard = max(1, min(SHARD_MAX, int(150 * 1024 * 1024 / per_req)))
    JUDGE.mkdir(parents=True, exist_ok=True)
    ids = []
    man_path = JUDGE / "batches.json"

    def save():
        # Written after EVERY shard, not once at the end. A failure partway through would
        # otherwise leave the earlier batches running with no record of their ids -- work
        # that is billed and cannot be collected without hunting through batches.list().
        man_path.write_text(json.dumps(
            {"model": args.model, "prefiltered": not args.all, "n_items": len(pool),
             "shard_size": shard, "batch_ids": ids}, indent=2))

    for start in range(0, len(pool), shard):
        chunk = pool[start:start + shard]
        reqs = [Request(custom_id=f"item-{start + i}",
                        params=MessageCreateParamsNonStreaming(
                            **request_params(rub, it, args.model, args.effort)))
                for i, it in enumerate(chunk)]
        try:
            b = cl.messages.batches.create(requests=reqs)
        except Exception as e:                              # noqa: BLE001
            save()
            sys.exit(f"\n  ! shard {len(ids) + 1} failed to submit: {e}\n"
                     f"  {len(ids)} shard(s) are already running and recorded in {man_path}.\n"
                     f"  Collect those first, then resubmit the rest from item {start}.")
        ids.append(b.id)
        save()
        print(f"  shard {len(ids)}: {len(reqs)} requests -> {b.id}")
    print(f"\n  {len(pool)} requests across {len(ids)} batch(es) on {args.model}")
    print(f"  manifest: {JUDGE / 'batches.json'}")
    print(f"  collect with: local/bin/python scaffold_judge.py --collect all")


def mode_collect(args):
    cl = client()
    man_path = JUDGE / "batches.json"
    if args.collect == "all":
        if not man_path.exists():
            sys.exit(f"no manifest at {man_path} — pass an explicit batch id")
        man = json.loads(man_path.read_text())
        batch_ids, prefiltered = man["batch_ids"], man["prefiltered"]
    else:
        batch_ids = [args.collect]
        prefiltered = json.loads(man_path.read_text())["prefiltered"] if man_path.exists() else True
    pending = []
    for bid in batch_ids:
        b = cl.messages.batches.retrieve(bid)
        print(f"  {bid}: {b.processing_status}  {b.request_counts}")
        if b.processing_status != "ended":
            pending.append(bid)
    if pending:
        print(f"\n  {len(pending)} batch(es) still running — rerun when they end")
        return
    items = load_corpus()
    pool = items if not prefiltered else build_pool(items)[0]
    JUDGE.mkdir(parents=True, exist_ok=True)
    out = JUDGE / "verdicts.jsonl"
    n, failed = 0, []
    with open(out, "w") as fh:
        for bid in batch_ids:
            for res in cl.messages.batches.results(bid):
                if res.result.type != "succeeded":
                    failed += 1
                    continue
                idx = int(res.custom_id.split("-")[1])
                msg = res.result.message
                text = next((x.text for x in msg.content if x.type == "text"), None)
                # A missing text block is a FAILURE, not an empty verdict. Defaulting to
                # "{}" here wrote 23 records whose `judge` was an empty object, which read
                # downstream as a judgment rather than as a hole and crashed --analyze.
                if text is None:
                    failed.append((idx, "no text block"))
                    continue
                try:
                    v = json.loads(text)
                except json.JSONDecodeError:
                    failed.append((idx, "unparseable JSON"))
                    continue
                absent = [f for f in FIELDS if f not in v]
                if absent:
                    failed.append((idx, "missing " + ",".join(absent)))
                    continue
                it = pool[idx]
                fh.write(json.dumps({
                    # Content-hashed, and the SAME scheme scaffold_subset.py uses, so
                    # verdicts join to the hand labels and the neighborhood sheets without
                    # depending on pool order. Positional ids would re-point on any
                    # resampling -- that failure has already cost this project 25 labels.
                    "item_id": item_id("P", it["target"], it["prefix"]),
                    "target": it["target"],
                    "year": it["year"],
                    "is_future": it["is_future"],
                    "judge": v}) + "\n")
                n += 1
    if failed:
        print(f"  {len(failed)} request(s) did not return a usable verdict "
              f"({100 * len(failed) / max(n + len(failed), 1):.3f}%)")
        why = Counter(r for _, r in failed)
        for r, c in why.most_common():
            print(f"      {c:>4}  {r}")
        (JUDGE / "failed.json").write_text(json.dumps(
            [{"idx": i, "target": pool[i]["target"], "reason": r} for i, r in failed],
            indent=2))
        print(f"      listed in {JUDGE / 'failed.json'}")
    print(f"  wrote {n} verdicts to {out}")


def mode_cache_check(args):
    """Two identical requests; the second must read what the first wrote.

    Prompt caching is a request parameter, not an account setting, but this settles it
    empirically rather than by argument. Costs a fraction of a cent.
    """
    rub = load_rubric()
    item = {"target": "cholinergic",
            "prefix": "I suggest the words 'adrenergic' and '"}
    cl = client()
    for i in (1, 2):
        _, u = judge_one(cl, rub, item, args.model, args.effort)
        w = getattr(u, "cache_creation_input_tokens", 0) or 0
        r = getattr(u, "cache_read_input_tokens", 0) or 0
        print(f"  call {i}: input {u.input_tokens:>5}   cache_write {w:>6}   cache_read {r:>6}")
    print()
    if r:
        pin = PRICES[args.model][0]
        print(f"  CACHING IS ON. The second call read {r} tokens at 0.1x "
              f"(${r * pin * 0.1 / 1e6:.5f} instead of ${r * pin / 1e6:.5f}).")
        print(f"  Over 50,350 items that is the difference between ~$62 and ~$170.")
    else:
        print("  CACHING IS NOT ENGAGING — do not submit the corpus run.")
        print(f"  The rubric is {len(system_prompt(rub)) // 4} tokens; {args.model} needs "
              f"{CACHE_MIN[args.model]}. If that is not the cause, check the dashboard.")


def mode_analyze(args):
    """The flagged post-run check: does the withheld interaction show up unprompted?

    The judge is never told that affix material needs a frame and stem material does not.
    If the corpus verdicts reproduce that pattern anyway, it is evidence for the claim.
    Reading this table is NOT a licence to edit the rubric -- doing so would convert the
    finding into an instruction and destroy exactly what makes it evidence.
    """
    path = JUDGE / "verdicts.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run --collect first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Defensive even though --collect now drops incomplete verdicts: a file written by an
    # older collect is still on disk in most working copies, and a crash here is a worse
    # failure than a reported count of holes.
    holes = [r for r in rows if any(f not in r.get("judge", {}) for f in FIELDS)]
    if holes:
        print(f"  ! {len(holes)} record(s) missing rubric fields — skipped.\n"
              f"    Re-run --collect to rewrite the file without them.\n")
        rows = [r for r in rows if r not in holes]
    FRAMES = ("coordination", "naming", "contrast_analogy")
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        j = r["judge"]
        mat = "stem" if j["material"] in ("stem", "both") else j["material"]
        framed = "frame" if j["recruitment"] in FRAMES else j["recruitment"]
        c = cell[(mat, framed)]
        c[0] += 1
        c[1] += j["verdict"] == "scaffolded"
    print(f"  {len(rows)} verdicts\n")
    print(f"    {'material':<10}{'recruitment':<14}{'N':>8}{'scaffolded':>12}")
    for k in sorted(cell):
        n, s_ = cell[k]
        print(f"    {k[0]:<10}{k[1]:<14}{n:>8}{s_/n:>12.3f}")
    print("\n  This table is DESCRIPTIVE. It is not a test of the withheld interaction,")
    print("  and must not be reported as one. Rubric section 6 defines inert material as")
    print("  not scaffolded, so the recruitment=none rows are true by definition, and a")
    print("  lookup on (material, usable, recruited) reproduces ~99% of these verdicts.")
    print("  The withheld claim is about model recall; testing it means joining these")
    print("  verdicts to recall outcomes or to hand annotation on `item_id`.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="build candidates, report the null")
    ap.add_argument("--calibrate", action="store_true", help="judge the hand-labelled items")
    ap.add_argument("--cost", action="store_true", help="price the run (offline)")
    ap.add_argument("--submit", action="store_true", help="submit the pool via Batch API")
    ap.add_argument("--collect", metavar="BATCH_ID", help="retrieve batch results")
    ap.add_argument("--cache-check", action="store_true",
                    help="two identical calls; prove the second reads the first's cache")
    ap.add_argument("--analyze", action="store_true",
                    help="post-run: does the withheld material x frame interaction appear?")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES))
    ap.add_argument("--n", type=int, help="cap the number of items (smoke tests)")
    ap.add_argument("--effort", default="off",
                    choices=["off", "low", "medium", "high"],
                    help="thinking depth; 'off' disables thinking entirely")
    ap.add_argument("--all", action="store_true",
                    help="judge every cloze item — no pre-filter at all")
    ap.add_argument("--show-prompt", action="store_true", help="print the system prompt")
    args = ap.parse_args()

    if args.show_prompt:
        print(system_prompt(load_rubric()))
    elif args.pool:
        mode_pool(args)
    elif args.cost:
        mode_cost(args)
    elif args.calibrate:
        mode_calibrate(args)
    elif args.submit:
        mode_submit(args)
    elif args.collect:
        mode_collect(args)
    elif args.cache_check:
        mode_cache_check(args)
    elif args.analyze:
        mode_analyze(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
