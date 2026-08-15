"""Guards on the perturbation set.

The first test is the one that gates the GPU run. The experiment reuses the committed
cloze decode as its unperturbed baseline instead of buying a fourth arm, and that is only
valid if the composition path reconstructs the same prompt string the cloze path
produced. It is a byte comparison, so it belongs here rather than in a decode.

The rest guard the properties each arm's meaning rests on: that the donor is actually
gone, that the deletion is a deletion rather than a rewrite, that the placebo perturbs
something other than the donor, and that no two items in a batch file share a target --
`evals.composition.load_test_cases` keys its dict on `word`, so a repeat silently
overwrites.

`test_form_size_gates_inflection_only_groups` is inherited from tests/test_connect_joins.py,
removed in the reset. The gate it protects is now load-bearing for the outcome measure:
survival is computed from `form_size`, and 14% of judged records carry a returned group of
two or more whose `form_n` is 0 because the inflection rule fired.
"""

import json
import re
from pathlib import Path

import pytest

import perturb_build as PB
import neighborhood_judge as NJ
from evals.cloze import extract_prefix

LIMIT = 40          # enough to exercise every branch; the full build is slow


@pytest.fixture(scope="module")
def built():
    if not PB.SAMPLE.exists():
        pytest.skip("frozen sample absent; run perturb_sample.py --write")
    rows, _, _ = PB.build(limit=LIMIT)
    if not rows:
        pytest.skip("no items built")
    return rows


# --------------------------------------------------------------------------- #
# The pre-decode gate
# --------------------------------------------------------------------------- #
def test_composition_path_reproduces_the_cloze_prefix(built):
    """An unperturbed context must split back to exactly extract_prefix's output.

    run_cloze builds its prompt as extract_prefix(text, word); run_composition builds
    its as context.split("[MASK]")[0].rstrip(). If those disagree by so much as trailing
    whitespace, the committed decode is not a baseline for the perturbed arms and the
    whole no-fourth-arm decision collapses.
    """
    for b in built:
        emitted = b["prefix"] + " [MASK]"
        assert emitted.split("[MASK]")[0].rstrip() == b["prefix"]


def test_prefix_matches_the_committed_decode(built):
    """The prefix carried through the build is the one the cloze run actually decoded."""
    idx = PB.prefix_index()
    for b in built:
        target, prefix = idx[b["item_id"]]
        assert prefix == b["prefix"]
        assert extract_prefix(prefix + " " + target, target) == prefix


# --------------------------------------------------------------------------- #
# Arm integrity
# --------------------------------------------------------------------------- #
def test_every_donor_occurrence_is_perturbed(built):
    """Not just the nearest one.

    scaffold_subset.recruitment() collapses a repeated donor to its nearest position, so
    an item can carry live donor material further back that a single-site edit leaves in
    the prompt. 13 of the 182 post-cutoff items repeat their donor.
    """
    for b in built:
        s = b["contexts"].get("donor_substituted")
        if s:
            assert not re.search(r"\b" + re.escape(b["donor"]) + r"\b", s, re.I)


def test_substitute_shares_no_material_with_the_target(built):
    """The point of the arm. A substitute overlapping the target removed nothing."""
    for b in built:
        assert not PB.shares_run(b["substitute"], b["target"])
        assert b["substitute"].lower() != b["donor"].lower()


def test_deletion_is_a_deletion_and_not_a_rewrite(built):
    """Every token of the result appears in the original, in order.

    A paraphrase or a reordering fails this; a pure deletion always passes. Without it
    the arm silently becomes "the model rewrote the sentence", a different manipulation
    with a different confound.
    """
    for b in built:
        d = b["contexts"].get("donor_deleted")
        if not d:
            continue
        assert not re.search(r"\b" + re.escape(b["donor"]) + r"\b", d, re.I)
        orig = re.findall(r"[A-Za-z]+", b["prefix"].lower())
        i = 0
        for w in re.findall(r"[A-Za-z]+", d.lower()):
            while i < len(orig) and orig[i] != w:
                i += 1
            assert i < len(orig), f"{b['item_id']}: {w!r} is not in the original"
            i += 1


def test_placebo_perturbs_something_other_than_the_donor(built):
    """The INV arm is the baseline the DIR arms are read against.

    A placebo that touched the donor, or a word sharing the target's form, would be a
    second DIR arm wearing an INV label.
    """
    for b in built:
        if "placebo" not in b["contexts"]:
            continue
        w = b["placebo_word"]
        assert w.lower() != b["donor"].lower()
        assert not PB.shares_run(w, b["target"])
        assert re.search(r"\b" + re.escape(w) + r"\b", b["prefix"], re.I)
        assert b["contexts"]["placebo"] != b["prefix"]
        assert not re.search(r"\b(\w+)\s+\1\b", b["contexts"]["placebo"], re.I)


def test_arms_are_present_and_nonempty(built):
    for b in built:
        assert b["contexts"], f"{b['item_id']} has no arms"
        for name, ctx in b["contexts"].items():
            assert ctx.strip(), f"{b['item_id']}/{name} is empty"


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
def test_no_target_repeats_within_a_batch_file(tmp_path, built, monkeypatch):
    """load_test_cases keys on `word`, so a repeated target silently overwrites.

    Six of the 182 post-cutoff targets appear twice. The partition is what stops two
    items collapsing into one without any error.
    """
    monkeypatch.chdir(tmp_path)
    PB.write_batches(built)
    seen_any = set()
    for path in sorted(Path(".").glob("perturb_batch_*.jsonl")):
        words = [json.loads(l)["word"].lower() for l in path.open() if l.strip()]
        assert len(words) == len(set(words)), f"{path} repeats a target"
        seen_any |= set(words)
    assert seen_any


def test_each_context_carries_exactly_one_mask(tmp_path, built, monkeypatch):
    monkeypatch.chdir(tmp_path)
    PB.write_batches(built)
    for path in Path(".").glob("perturb_batch_*.jsonl"):
        for line in path.open():
            if not line.strip():
                continue
            for ctx in json.loads(line)["contexts"].values():
                assert ctx.count("[MASK]") == 1


# --------------------------------------------------------------------------- #
# Outcome-measure gate, inherited from the removed test_connect_joins.py
# --------------------------------------------------------------------------- #
def test_form_size_gates_inflection_only_groups():
    """`form_n` is authoritative over len(form_group), and survival is built on it.

    The judge returns proposed membership; form_size enforces what was asked for --
    largest connected component, singletons to zero, inflection-only groups to zero. A
    group of two whose only link is the inflection the gap's grammar forces on every
    candidate is not a paradigm, and 14% of judged records are in that state.
    """
    # `filter`/`filters` differ only by the inflection the gap's grammar forces on every
    # candidate -- the case the annotator ruled 0. `walk`/`talk` are distinct lexemes and
    # legitimately measure 2, which is what keeps the gate from being a blanket cut.
    words = ["filter", "filters", "filtration", "house"]

    # Inflection-only: one lexeme wearing two endings, so no paradigm. The case the
    # annotator ruled 0 for shot/shots and filter/filters alike.
    assert NJ.form_size({"form_group": ["filter", "filters"]}, words) == 0
    assert NJ.form_size({"form_group": ["filter"]}, words) == 0
    assert NJ.form_size({"form_group": []}, words) == 0

    # Two distinct lexemes sharing a run is the minimal paradigm -- the first
    # generalisation, and what the construct is defined on.
    assert NJ.form_size({"form_group": ["filter", "filtration"]}, words) == 2

    # Once two distinct lexemes are present the inflections count again: the gate asks
    # whether a paradigm exists, not whether every member is a separate lexeme. Survival
    # inherits this, so a group that keeps two lexemes plus an inflection measures 3.
    assert NJ.form_size({"form_group": ["filter", "filters", "filtration"]}, words) == 3


# --------------------------------------------------------------------------- #
# Drift spot-check set
# --------------------------------------------------------------------------- #
def test_driftcheck_prompts_are_the_committed_prompts(tmp_path, built, monkeypatch):
    """The unedited prompts must be byte-identical to what the cloze run decoded.

    The check compares fresh decodes against `results/cloze_*_details.csv`. If the prompt
    differs at all, a mismatch would indicate a different prompt rather than a different
    environment, and the check would answer a question nobody asked.
    """
    idx = PB.prefix_index()          # reads results/ relative to cwd -- before chdir
    monkeypatch.chdir(tmp_path)
    PB.write_driftcheck(built)
    rows = [json.loads(l) for l in Path("perturb_driftcheck.jsonl").open() if l.strip()]
    assert rows
    for r in rows:
        ctx = r["contexts"]["original"]
        assert ctx.count("[MASK]") == 1
        prefix = ctx.split("[MASK]")[0].rstrip()
        iid = PB.SS.item_id("P", PB.SS.NA.norm(r["word"]), prefix)
        assert iid in idx
        assert idx[iid][1] == prefix
    # a repeated target would silently overwrite its twin in the composition loader
    words = [r["word"].lower() for r in rows]
    assert len(words) == len(set(words))


def test_driftcheck_selection_is_deterministic(tmp_path, built, monkeypatch):
    monkeypatch.chdir(tmp_path)
    PB.write_driftcheck(built)
    first = Path("perturb_driftcheck.jsonl").read_text()
    PB.write_driftcheck(built)
    assert Path("perturb_driftcheck.jsonl").read_text() == first
