#!/usr/bin/env python3
"""Build the perturbed prompts for the scaffold-perturbation experiment.

A scaffolded prompt contains a DONOR -- a word the scaffold judge named as supplying
form material the model could use to build the target (`clearing` for `clearance`,
`displacement` for `recruitment`). The experiment removes that material and re-decodes.

THREE ARMS, in CheckList terms (Ribeiro et al. 2020):

  donor_substituted   DIR. The donor is replaced by a word of matched meaning that
                      shares no letters with the target. Meaning stays, form goes.
  donor_deleted       DIR. The phrase the judge quoted as recruiting the donor is
                      removed outright.
  placebo             INV. A non-donor content word is substituted under the identical
                      procedure. Nothing about the target's form material changes, so
                      nothing should happen -- and if something does, the two DIR arms
                      are measuring sensitivity to being edited rather than to losing
                      the donor.

There is no `original` arm. Decoding is deterministic constrained beam search
(do_sample=False, num_beams=100, length_penalty=0.0) over pinned weights, so
results/cloze_{model}_details.csv already IS the unperturbed condition. What a
re-decode would have caught is a mismatch between the cloze path (extract_prefix ->
cloze_prompt) and the composition path ([MASK] split -> parts[0].rstrip() ->
cloze_prompt); that is a string identity, checked on CPU in tests, not something worth
buying GPU time for. What remains unverified is that the numerical stack is unchanged
since the original run -- an assumption about run provenance, bounded by a ~30-item
drift spot-check rather than a fourth arm.

THE SCREEN is what makes the substitution arm mean anything, so it is enforced in code
and every rejection is counted:

  * shares no substring of 3+ characters with the target  -- the point of the arm; a
    substitute that still overlaps the target has not removed the scaffold
  * attested in the period lexicon, as given or as a lemma  -- a period-plausibility
    check. It is deliberately NOT the grammaticality gate: the proposer sees the
    sentence and returns the surface form that fits the slot, which is a judgement about
    this passage rather than something a spelling rule can supply
  * within 1.5 orders of magnitude of the original's Brown frequency  -- a
    rare-for-common swap changes how predictable the region is
  * for the placebo, a word that really occurs in the passage and is not the donor

A model-authored deletion must additionally be a genuine deletion: every token of the
result appears in the original, in order. A paraphrase fails that subsequence test, so
the arm cannot quietly become "the model rewrote the sentence".

Items with no admissible substitute are REPORTED, not forced, and every rejection is
counted. A forced bad substitute silently converts a DIR arm into a grammaticality
probe.

Emits composition-format JSONL (one record per item, one context per arm, each ending
in [MASK]) plus a metadata sidecar. evals/composition.load_test_cases keys its dict on
`word`, so a repeated target silently overwrites; targets are partitioned across batch
files so none repeats within a file.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import neighborhood_judge as NJ
import scaffold_subset as SS
from evals.cloze import extract_prefix

ANALYSIS = Path("analysis")
RESULTS = Path("results")
SAMPLE = ANALYSIS / "perturb_sample_ids.txt"
SCAFFOLD_VERDICTS = ANALYSIS / "judge" / "verdicts.jsonl"

MIN_OVERLAP = 3          # a shared run of this length is what the judge calls material
MIN_CONTENT_LEN = 4      # below this a word is not a placebo candidate
# Allowed |log10| gap between the original word's Brown frequency and its replacement's.
# 1.0 dex rejected 1,030 candidates -- the largest single cause of screen failure -- on a
# 1M-word 1961 corpus where most period and technical vocabulary sits at or near zero
# count, so the band was measuring corpus sparsity as much as frequency mismatch. 1.5 dex
# still excludes a rare-for-common swap (the thing the control exists to prevent) while
# recovering items the corpus simply has no evidence about.
FREQ_BAND_DEX = 1.5


# --------------------------------------------------------------------------- #
# Shared-substring test
# --------------------------------------------------------------------------- #
def shares_run(a, b, n=MIN_OVERLAP):
    """Do `a` and `b` share any run of >= n characters?

    This is the same relation the scaffold judge was shown as its OVERLAPS table, so the
    screen rejects exactly what the judge would have been able to call material.
    """
    a, b = a.lower(), b.lower()
    if len(a) < n or len(b) < n:
        return False
    return any(a[i:i + n] in b for i in range(len(a) - n + 1))


# --------------------------------------------------------------------------- #
# Inflection transfer
# --------------------------------------------------------------------------- #
INFLECTIONS = ("ings", "ing", "edly", "ed", "es", "s", "ly", "er", "est")


def split_inflection(word):
    """(stem, suffix) using the longest inflectional ending that leaves a real stem."""
    w = word.lower()
    for suf in INFLECTIONS:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)], suf
    return w, ""


def inflect_like(lemma, suffix):
    """Surface forms of `lemma` carrying `suffix`, cheapest spelling rules first.

    Deliberately generates candidates rather than deciding: every form is then required
    to appear in the attestation set, so a wrong guess is dropped by evidence instead of
    being reasoned about. `-ing` on a consonant-final stem yields both `runing` and
    `running` here; only the attested one survives.
    """
    if not suffix:
        return [lemma]
    out = []
    if suffix in ("ing", "ings", "ed", "er", "est"):
        if lemma.endswith("e"):
            out.append(lemma[:-1] + suffix)
        out.append(lemma + suffix)
        if len(lemma) >= 3 and lemma[-1] not in "aeiou" and lemma[-2] in "aeiou":
            out.append(lemma + lemma[-1] + suffix)
    elif suffix in ("s", "es"):
        out.extend([lemma + "s", lemma + "es"])
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            out.append(lemma[:-1] + "ies")
    else:
        out.append(lemma + suffix)
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


# --------------------------------------------------------------------------- #
# Candidate proposal
# --------------------------------------------------------------------------- #
def wordnet_synonyms(word):
    """Lemmas sharing a synset with `word`, grouped by part of speech.

    WordNet is already loaded by neighborhood_analysis.load_resources(), so this adds no
    new dependency and keeps the whole build local and deterministic -- no API call, no
    model output to validate, nothing for a reviewer to take on trust.
    """
    from nltk.corpus import wordnet as wn
    stem, _ = split_inflection(word)
    out = defaultdict(list)
    for probe in (word.lower(), stem):
        for syn in wn.synsets(probe):
            for lem in syn.lemmas():
                name = lem.name().replace("_", "-").lower()
                if "-" in name or not name.isalpha():
                    continue
                if name in (word.lower(), stem):
                    continue
                if name not in out[syn.pos()]:
                    out[syn.pos()].append(name)
    return out


def donor_pos(word):
    """The part of speech WordNet most often assigns to `word`, or None."""
    from nltk.corpus import wordnet as wn
    stem, _ = split_inflection(word)
    for probe in (word.lower(), stem):
        syns = wn.synsets(probe)
        if syns:
            return Counter(s.pos() for s in syns).most_common(1)[0][0]
    return None


def attested(word, attest):
    """Period plausibility: the word itself, or its lemma, is in the attestation set."""
    w = word.lower()
    if w in attest:
        return True
    stem, suf = split_inflection(w)
    return bool(suf) and stem in attest


def freq_dex(word, brown):
    import math
    return math.log10(brown[word.lower()] + 1)


_BROWN = None


def brown_freq():
    """Memoised Brown frequency distribution. Building it costs ~2s."""
    global _BROWN
    if _BROWN is None:
        from nltk.probability import FreqDist
        from nltk.corpus import brown as brown_corpus
        _BROWN = FreqDist(w.lower() for w in brown_corpus.words())
    return _BROWN


def screen(candidates, donor, target, attest, brown, reject, ranked=False,
           pre_inflected=False):
    """Best admissible replacement for `donor` among `candidates`, or None.

    The screen is the same whatever proposed the candidates, which is the point: the
    proposer supplies meaning, the screen supplies the guarantees. Ordered by frequency
    proximity to the donor, so the chosen word perturbs the region's predictability least
    while still removing the form material. Every rejection increments `reject` -- a
    screen whose failures are not counted cannot be reported.
    """
    _, suffix = split_inflection(donor)
    dstem = split_inflection(donor)[0]
    scored = []
    for lemma in candidates:
        lemma = lemma.lower().strip()
        if not lemma.isalpha():
            continue
        # A word cannot be its own substitute, and a same-stem "synonym" leaves the
        # donor's form material in place -- which is the one thing the arm must remove.
        if lemma == donor.lower() or split_inflection(lemma)[0] == dstem:
            reject["identity_or_same_stem"] += 1
            continue
        if shares_run(lemma, target):
            reject["overlaps_target"] += 1
            continue
        if pre_inflected:
            # The proposer saw the sentence and returned the surface form that belongs in
            # the slot. Attestation then does the job it should always have done -- is
            # this a word of the period -- rather than doubling as an inflection oracle.
            # It was doing the latter badly: mechanical inflection turned 1,528 attested
            # lemmas into 1,529 attested surfaces while rejecting 836 candidates.
            if not attested(lemma, attest):
                reject["not_attested"] += 1
                continue
            surface = lemma
        else:
            forms = [f for f in inflect_like(lemma, suffix) if f in attest]
            if not forms:
                reject["inflected_form_unattested"] += 1
                continue
            surface = forms[0]
        if surface == donor.lower() or shares_run(surface, target):
            reject["overlaps_target"] += 1
            continue
        gap = abs(freq_dex(surface, brown) - freq_dex(donor, brown))
        if gap > FREQ_BAND_DEX:
            reject["frequency_band"] += 1
            continue
        scored.append((gap, surface))
    if not scored:
        return None
    # An in-context proposer returns candidates best-first, and that ordering encodes
    # sense disambiguation the screen cannot recover. Re-sorting by frequency proximity
    # threw it away: `degree` in "a new degree of freedom" took `sort` over the better
    # candidates because `sort` sat closer in the Brown band. Frequency is a filter, not
    # a ranker -- honour the proposer's order and take the first admissible candidate.
    # WordNet's ordering carries no such signal, so the fallback still ranks by gap.
    if ranked:
        return scored[0][1]
    scored.sort()
    return scored[0][1]


def choose_substitute(donor, target, attest, brown, reject, proposals=None,
                      pre_inflected=False):
    """Screened replacement for `donor`, preferring in-context proposals.

    WordNet is the fallback, not the intended source. Measured on 40 items it produced a
    semantically wrong substitute at an unacceptable rate -- `representation` -> `theatrical`
    in a genomics passage, `degree` -> `stage` in "a new degree of freedom" -- because
    synonyms without sense disambiguation pick whichever synset happens to be listed, and
    Lesk did not fix it (it resolves that same `degree` to `academic_degree`). A wrong-sense
    substitute breaks the sentence, which silently converts a DIR arm into a grammaticality
    probe. In-context proposals come from `--submit`/`--collect`.
    """
    if proposals:
        got = screen(proposals, donor, target, attest, brown, reject, ranked=True,
                     pre_inflected=pre_inflected)
        # If the proposer saw the sentence and everything it offered failed the screen,
        # that is evidence the slot is hard -- usually because the donor itself overlaps
        # the target (`ileostomy` contains `stoma`), so every near-synonym does too.
        # WordNet does not know more about the sentence than the model did; letting it
        # rescue the item produced "a patient with an surgeon" and rewrote "Ileostomy
        # Association" to "Surgeon Association". Report the item instead.
        return got
    pos = donor_pos(donor)
    if pos is None:
        reject["donor_not_in_wordnet"] += 1
        return None
    cands = wordnet_synonyms(donor).get(pos, [])
    if not cands:
        reject["no_synonyms"] += 1
        return None
    return screen(cands, donor, target, attest, brown, reject)


# --------------------------------------------------------------------------- #
# Perturbation operations
# --------------------------------------------------------------------------- #
def substitute_all(prefix, word, replacement):
    """Replace every whole-word occurrence, preserving capitalisation of the first letter.

    Every occurrence, not the nearest: scaffold_subset.recruitment() collapses a repeated
    donor to its nearest position, so an item can carry live donor material further back
    that a single-site edit would leave in the prompt. 13 of the 182 repeat their donor.
    """
    pat = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

    def sub(m):
        got = m.group(0)
        return replacement.capitalize() if got[:1].isupper() else replacement

    new, n = pat.subn(sub, prefix)
    return new, n


def delete_evidence(prefix, evidence, donor):
    """Remove the phrase the judge quoted as recruiting the donor.

    Falls back to deleting the donor's own clause when the quoted phrase cannot be found
    verbatim -- the judge quotes from the 320-character tail it was shown, and normalises
    whitespace, so an exact match is not guaranteed.
    """
    def ok(out):
        # A deletion that takes most of the prompt is not an ablation of the donor, it is
        # a different prompt. One clause_fallback removed all but a comma from a physics
        # citation. Require that the bulk of the passage survives, or drop to word-only.
        return out and len(out) >= 40 and len(out) >= 0.6 * len(prefix)

    if evidence:
        idx = prefix.lower().find(evidence.lower().strip())
        if idx != -1:
            out = re.sub(r"\s+", " ",
                         prefix[:idx] + prefix[idx + len(evidence.strip()):]).strip()
            if ok(out):
                return out, "evidence_phrase"
    parts = re.split(r"([,;:]|\band\b|\bbut\b)", prefix)
    keep = [p for p in parts if not re.search(r"\b" + re.escape(donor) + r"\b", p, re.I)]
    out = re.sub(r"\s+", " ", "".join(keep)).strip()
    if out != prefix and ok(out):
        return out, "clause_fallback"
    # No word-only fallback. Excising the bare donor leaves the sentence ungrammatical
    # ("introducing a new of freedom"), and a DIR arm that damages grammar measures
    # sensitivity to broken syntax rather than to the missing donor -- the same confound
    # that excluded the `move` arm. An item with no clean deletion is reported as
    # deletion-ineligible and still contributes to the substitution and placebo arms.
    return None, "no_clean_deletion"


def placebo_word(prefix, donor, target, brown, stops):
    """The non-donor content word the INV arm will replace, or None.

    Chosen deterministically and independently of whether a replacement exists, so the
    word can be named in a proposal request before any substitute is known. Matched to
    the donor on part of speech and frequency so the placebo perturbs a comparable amount
    of the prompt. Words sharing form with the target are excluded -- they are potential
    donors themselves, and editing one would be a second DIR arm wearing an INV label.
    """
    pos = donor_pos(donor)
    seen, cands = set(), []
    for w in re.findall(r"[A-Za-z]+", prefix):
        lw = w.lower()
        if lw in seen or lw in stops or len(lw) < MIN_CONTENT_LEN:
            continue
        seen.add(lw)
        if lw == donor.lower() or shares_run(lw, target) or shares_run(lw, donor):
            continue
        if donor_pos(lw) != pos:
            continue
        cands.append(lw)
    if not cands:
        return None
    cands.sort(key=lambda w: (abs(freq_dex(w, brown) - freq_dex(donor, brown)), w))
    return cands[0]


def verified_deletion(prefix, donor, proposed):
    """A model-authored deletion, or (None, reason) if it is not one.

    The model is asked to delete, not rewrite, and this checks that it did. Every token
    of the result must appear in the original in order -- a subsequence test, which a
    paraphrase or a reordering fails and a pure deletion always passes. Without it the
    arm would silently become "the model rewrote the sentence", which is a different
    manipulation with a different confound.
    """
    if not proposed or not proposed.strip():
        return None, "model_declined"
    out = re.sub(r"\s+", " ", proposed).strip()
    if re.search(r"\b" + re.escape(donor) + r"\b", out, re.I):
        return None, "donor_survives"
    # Excising a span mid-clause can weld the remainder together across the cut
    # ("consciousness,and purposes"). The join is the artefact, not the deletion.
    if re.search(r"[,;:][A-Za-z]", out):
        return None, "punctuation_artifact"
    orig = re.findall(r"[A-Za-z]+", prefix.lower())
    got = re.findall(r"[A-Za-z]+", out.lower())
    if not got:
        return None, "empty"
    i = 0
    for w in got:
        while i < len(orig) and orig[i] != w:
            i += 1
        if i == len(orig):
            return None, "not_a_subsequence"
        i += 1
    if len(out) < 40 or len(out) < 0.6 * len(prefix):
        return None, "too_much_removed"
    return out, "model_deletion"


def pick_placebo(prefix, donor, target, attest, brown, stops, reject,
                 chosen=None, proposals=None, pre_inflected=False):
    """(word, replacement) for the INV arm, or (None, None).

    The INV arm is what licenses reading the two DIR arms as anything other than
    sensitivity to being edited at all, so it goes through the same in-context proposer
    and the same screen as the donor substitution. Building it from the WordNet fallback
    -- the path measured to produce wrong-sense substitutes -- undercut both DIR arms
    rather than just weakening this one.
    """
    w = None
    if chosen:
        c = chosen.strip().lower()
        # The model's choice still has to satisfy the control the heuristic enforced:
        # a word actually in the passage, not the donor, sharing no form with the target.
        if (re.search(r"\b" + re.escape(c) + r"\b", prefix, re.I)
                and c != donor.lower() and not shares_run(c, target)):
            w = c
        else:
            reject["placebo_word_rejected"] += 1
    if w is None:
        w = placebo_word(prefix, donor, target, brown, stops)
        proposals = None          # proposals were for a word we are not using
    if not w:
        reject["no_placebo_candidate"] += 1
        return None, None
    rep = choose_substitute(w, target, attest, brown, reject, proposals,
                            pre_inflected=pre_inflected and proposals is not None)
    if not rep:
        return None, None
    # A substitute that lands beside its own twin ("a natural born" -> "a born born")
    # is a degenerate string, and the INV arm has to be the cleanest of the three: it is
    # the baseline against which the DIR arms are read.
    trial, _ = substitute_all(prefix, w, rep)
    if re.search(r"\b(\w+)\s+\1\b", trial, re.I):
        reject["placebo_adjacent_duplicate"] += 1
        return None, None
    return w, rep


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def load_sample():
    if not SAMPLE.exists():
        sys.exit(f"missing {SAMPLE} -- run perturb_sample.py --write first")
    rows = []
    with SAMPLE.open() as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("stratum\t"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) == 6:
                rows.append(dict(zip(
                    ("stratum", "item_id", "target", "year", "material", "recruitment"), p)))
    return rows


def load_verdicts():
    out = {}
    with SCAFFOLD_VERDICTS.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r["judge"]
    return out


def prefix_index():
    csv.field_size_limit(10 ** 7)
    idx = {}
    with (RESULTS / "cloze_typewriter_details.csv").open() as fh:
        for row in csv.DictReader(fh):
            target = row["target_word"]
            prefix = extract_prefix(row["text"], target)
            idx[SS.item_id("P", SS.NA.norm(target), prefix)] = (target, prefix)
    return idx


def locate_donors(limit=None):
    """Frozen-sample items whose judge-named donor is findable in the prefix.

    Shared by --submit and build() so the proposal batch and the assembled prompts are
    keyed to exactly the same item set; deriving them separately would let the two drift.
    """
    sample = load_sample()
    if limit:
        sample = sample[:limit]
    verdicts = load_verdicts()
    idx = prefix_index()
    out = []
    for row in sample:
        iid = row["item_id"]
        if iid not in idx:
            continue
        target, prefix = idx[iid]
        j = verdicts.get(iid, {})
        donors = [d.strip() for d in (j.get("donor") or "").split(",") if d.strip()]
        donors = [d for d in donors
                  if re.search(r"\b" + re.escape(d) + r"\b", prefix, re.I)]
        if not donors:
            continue
        out.append({**row, "target": target, "prefix": prefix,
                    "donor": donors[0], "evidence": j.get("evidence", "")})
    return out


def build(limit=None):
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))
    attest = NJ.attest_set()
    brown = brown_freq()

    items = locate_donors(limit=limit)
    proposals = load_proposals()
    n_sample = len(load_sample()[:limit] if limit else load_sample())

    built, skipped = [], []
    skipped.extend([("?", "donor_not_located")] * (n_sample - len(items)))
    reject = Counter()
    for row in items:
        iid, target, prefix = row["item_id"], row["target"], row["prefix"]
        donor = row["donor"]
        bundle = proposals.get((iid, "bundle")) or {}
        pre = bool(bundle)

        # Two-stage merge across proposal sets. Each list is screened under the regime it
        # was produced for -- the bundle returns surface forms fitted to the slot, the
        # earlier substitution-only run returns lemmas needing inflection -- so they
        # cannot be concatenated into one list, but they can be tried in turn. The
        # earlier set rescues 19 of the 40 items the bundle alone cannot place.
        rep = choose_substitute(donor, target, attest, brown, reject,
                                bundle.get("donor_replacements"), pre_inflected=True)
        if not rep:
            rep = choose_substitute(donor, target, attest, brown, reject,
                                    proposals.get((iid, "donor")), pre_inflected=False)
        if not rep:
            skipped.append((iid, "no_admissible_substitute"))
            continue
        sub_prefix, n_sub = substitute_all(prefix, donor, rep)

        # Per-arm availability: an item missing a clean deletion still carries the
        # substitution (DIR) and placebo (INV) arms. Arms are analysed per arm anyway,
        # so the alternative -- dropping the item outright -- would discard two good
        # observations to preserve a uniform n that the statistics never require.
        # Every deletion passes the same gate regardless of source. The regex fallback
        # removed the judge's evidence phrase without ever checking the donor went with
        # it, and left the donor standing in 3 of its 16 outputs -- an arm whose whole
        # premise is that the donor is gone.
        del_prefix, del_how = verified_deletion(prefix, donor, bundle.get("deletion"))
        if del_prefix is None:
            cand, how = delete_evidence(prefix, row["evidence"], donor)
            del_prefix, why = verified_deletion(prefix, donor, cand)
            del_how = how if del_prefix else f"{how}_rejected_{why}"

        pw, prep = pick_placebo(prefix, donor, target, attest, brown, stops, reject,
                                bundle.get("placebo_word"),
                                bundle.get("placebo_replacements"), pre_inflected=pre)
        if not pw:
            skipped.append((iid, "no_placebo"))
            continue
        pl_prefix, _ = substitute_all(prefix, pw, prep)

        built.append({
            "item_id": iid, "stratum": row["stratum"], "target": target,
            "year": row["year"], "material": row["material"],
            "recruitment": row["recruitment"],
            "donor": donor, "substitute": rep, "n_donor_sites": n_sub,
            "placebo_word": pw, "placebo_substitute": prep,
            "delete_method": del_how,
            "prefix": prefix,
            "contexts": {k: v for k, v in (
                ("donor_substituted", sub_prefix),
                ("donor_deleted", del_prefix),
                ("placebo", pl_prefix),
            ) if v},
        })
    return built, skipped, reject


# --------------------------------------------------------------------------- #
# In-context substitute proposal (batch API)
# --------------------------------------------------------------------------- #
PROPOSE_MODEL = "claude-opus-5"
PROPOSALS = ANALYSIS / "perturb" / "proposals.jsonl"
PROPOSE_MANIFEST = ANALYSIS / "perturb" / "batches.json"

PROPOSE_SYSTEM = """\
You prepare minimally-edited variants of a historical citation.

Follow the counterfactually-augmented data protocol of Kaushik et al. (2020): each
variant must retain internal coherence -- still the same sentence, about the same thing,
in the same register and period -- and avoid unnecessary changes.

You produce three things.

1. DONOR_REPLACEMENTS. Words that could stand in for the DONOR word.
   - each must share NO run of three or more letters with the FORBIDDEN word
   - give the exact surface form that belongs in the sentence, already inflected: if the
     donor is "prefers" give "chooses", not "choose"
   - must be ordinary English of the period shown; do not reuse the donor or a word
     built on its stem

2. PLACEBO_WORD and PLACEBO_REPLACEMENTS. Choose one OTHER content word from the
   passage -- not the donor -- that could be swapped for a synonym without changing what
   the passage says. Prefer one of comparable ordinariness to the donor. It must share no
   run of three or more letters with the FORBIDDEN word. Then give replacements for it
   under the same rules as above.

3. DELETION. The passage with the phrase that introduces the donor removed, so the
   donor no longer appears and what remains is grammatical and still reads as a sentence.
   Delete words; do not rewrite, reorder, or substitute any word. Keep as much of the
   passage as possible. If the donor cannot be removed without wrecking the sentence,
   return an empty string rather than a damaged passage.

Order every list best first. Give fewer rather than pad with words that break the
sentence. Do not include internal or system XML tags in your response."""

BUNDLE_SCHEMA = {
    "type": "object",
    "properties": {
        "donor_replacements": {"type": "array", "items": {"type": "string"}},
        "placebo_word": {"type": "string"},
        "placebo_replacements": {"type": "array", "items": {"type": "string"}},
        "deletion": {"type": "string"},
    },
    "required": ["donor_replacements", "placebo_word", "placebo_replacements",
                 "deletion"],
    "additionalProperties": False,
}


def propose_prompt(t):
    return (f"PASSAGE (it stops where a word was removed):\n...{t['prefix'][-320:]}\n\n"
            f"DONOR: {t['donor']}\n"
            f"FORBIDDEN (share no 3-letter run with this): {t['target']}\n")


def proposal_targets(limit=None):
    """One request per item covering all three arms.

    Every gap the screen leaves is a judgement about this sentence -- which surface form
    fits the slot, which other word is swappable, which phrase can be cut without
    wrecking the grammar -- so all three are asked of the model that can see the
    sentence, in one request. Splitting them across batches would pay the passage's input
    tokens once per arm and still leave the deletion arm mechanical.
    """
    return [{"item_id": it["item_id"], "role": "bundle", "donor": it["donor"],
             "prefix": it["prefix"], "target": it["target"]}
            for it in locate_donors(limit=limit)]


def mode_submit(args):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    items = proposal_targets(limit=args.limit)
    if args.only_missing:
        have = set(load_proposals())
        items = [t for t in items if (t["item_id"], t["role"]) not in have]
    print(f"{len(items)} (item, role) pairs need proposals: "
          + ", ".join(f"{r}={sum(1 for t in items if t['role'] == r)}"
                      for r in ("donor", "placebo")))
    if not items:
        sys.exit("nothing to request")
    cl = anthropic.Anthropic()
    # max_tokens caps THINKING PLUS RESPONSE TEXT, and on claude-opus-5 thinking is on
    # by default -- omitting the parameter runs adaptive, a reversal from Opus 4.8/4.7
    # where omitting it meant no thinking. The first run set max_tokens=300 with no
    # thinking field, so reasoning consumed the budget and the JSON truncated before its
    # closing brace on 351 of 360 items. Thinking is disabled here (this is a synonym
    # lookup, and the 9 collected results were already high quality) and the ceiling is
    # raised far past what the output needs. `disabled` is valid only at effort `high`
    # or below on this model; `high` is the default, so it is not set explicitly.
    # Structured outputs rather than a JSON-shaped instruction plus a tolerant parser:
    # the schema is enforced server-side, so a malformed or truncated object cannot be
    # returned at all. That removes the failure class that cost the first run, instead of
    # detecting it after the fact.
    reqs = [Request(custom_id=f"item-{i}",
                    params=MessageCreateParamsNonStreaming(**{
                        "model": PROPOSE_MODEL, "max_tokens": 2000,
                        "thinking": {"type": "disabled"},
                        "system": PROPOSE_SYSTEM,
                        "output_config": {"format": {"type": "json_schema",
                                                     "schema": BUNDLE_SCHEMA}},
                        "messages": [{"role": "user", "content": propose_prompt(it)}],
                    }))
            for i, it in enumerate(items)]
    PROPOSALS.parent.mkdir(parents=True, exist_ok=True)
    b = cl.messages.batches.create(requests=reqs)
    PROPOSE_MANIFEST.write_text(json.dumps(
        {"model": PROPOSE_MODEL, "n_items": len(items), "batch_ids": [b.id],
         "keys": [[t["item_id"], t["role"]] for t in items]}, indent=2))
    print(f"submitted {len(reqs)} proposal requests -> {b.id}")
    print(f"manifest: {PROPOSE_MANIFEST}")
    print("collect with: local/bin/python perturb_build.py --collect")


def parse_candidates(text):
    """(candidates, reason). Reason is None on success.

    Tries strict JSON first, then a brace-balanced scan that tolerates a preamble, then
    a bare list. Every failure names itself: the first version of this collector dropped
    351 of 360 items through three silent `continue`s, so nothing here fails quietly.
    """
    if not text.strip():
        return None, "empty_text"
    for m in re.finditer(r"\{", text):
        depth = 0
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
                    c = obj.get("candidates")
                    if isinstance(c, list) and c:
                        return [str(x) for x in c], None
                    break
    m = re.search(r"\[([^\[\]]*)\]", text, re.S)
    if m:
        got = [w.strip().strip('"\'') for w in m.group(1).split(",")]
        got = [w for w in got if w and w.isalpha()]
        if got:
            return got, None
    if "{" in text and "}" not in text:
        return None, "truncated_json"
    return None, "unparseable"


def _iter_results(cl, man):
    for bid in man["batch_ids"]:
        st = cl.messages.batches.retrieve(bid)
        if st.processing_status != "ended":
            sys.exit(f"batch {bid} is {st.processing_status}, not ended")
        for res in cl.messages.batches.results(bid):
            yield res


def mode_diagnose(args):
    """Re-read an already-billed batch and report why results were lost. Costs nothing."""
    import anthropic
    man = json.loads(PROPOSE_MANIFEST.read_text())
    cl = anthropic.Anthropic()
    kinds, stops, blocks, reasons = Counter(), Counter(), Counter(), Counter()
    samples = []
    for res in _iter_results(cl, man):
        kinds[res.result.type] += 1
        if res.result.type != "succeeded":
            continue
        msg = res.result.message
        stops[getattr(msg, "stop_reason", "?")] += 1
        for b in msg.content:
            blocks[b.type] += 1
        text = "".join(b.text for b in msg.content if b.type == "text")
        cands, why = parse_candidates(text)
        reasons["parsed_ok" if cands else (why or "unparseable")] += 1
        if not cands and len(samples) < 3:
            samples.append((res.custom_id, getattr(msg, "stop_reason", "?"),
                            [b.type for b in msg.content], repr(text[:400])))
    print(f"manifest: {man['n_items']} items, batches {man['batch_ids']}")
    for label, c in (("result.type", kinds), ("stop_reason", stops),
                     ("content blocks", blocks), ("parse outcome", reasons)):
        print(f"\n{label}:")
        for k, v in c.most_common():
            print(f"  {str(k):28} {v}")
    for cid, stop, btypes, text in samples:
        print(f"\n--- {cid}  stop={stop}  blocks={btypes}\n{text}")


def manifest_keys(man):
    """(item_id, role) per request index. `item_ids` is the pre-placebo manifest form."""
    if "keys" in man:
        return [tuple(k) for k in man["keys"]]
    return [(i, "donor") for i in man["item_ids"]]


def mode_collect(args):
    import anthropic
    man = json.loads(PROPOSE_MANIFEST.read_text())
    cl = anthropic.Anthropic()
    order = manifest_keys(man)
    rows, failures = {}, []
    for res in _iter_results(cl, man):
        idx = int(res.custom_id.split("-")[1])
        key = order[idx]
        if res.result.type != "succeeded":
            failures.append((key, f"result_{res.result.type}"))
            continue
        msg = res.result.message
        text = "".join(b.text for b in msg.content if b.type == "text")
        payload, why = parse_bundle(text)
        if payload is None:
            failures.append((key, f"{why}|stop={getattr(msg,'stop_reason','?')}"))
            continue
        rows[key] = payload
    # Merge rather than truncate: a partial collect must never destroy a good earlier one.
    rows = {**load_proposals(), **rows}
    PROPOSALS.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSALS.open("w") as fh:
        for (iid, role), c in rows.items():
            fh.write(json.dumps({"item_id": iid, "role": role, "payload": c}) + "\n")
    got = Counter(role for _, role in rows)
    print(f"wrote {PROPOSALS} ({len(rows)} pairs: "
          + ", ".join(f"{r}={got[r]}" for r in ("donor", "placebo")) + ")")
    if failures:
        print(f"\n{len(failures)} not collected:")
        for reason, n in Counter(r for _, r in failures).most_common():
            print(f"  {reason:40} {n}")
        fp = PROPOSALS.parent / "collect_failures.txt"
        fp.write_text("\n".join(f"{i}\t{role}\t{r}" for (i, role), r in failures))
        print(f"  keys written to {fp}")


def parse_bundle(text):
    """(payload_dict, reason). Reason is None on success.

    Structured outputs make the object well-formed at the wire level, so this is a
    guard rather than a recovery path -- but it stays tolerant, because the run that
    lost 351 items looked exactly like a well-formed run until someone counted.
    """
    if not text.strip():
        return None, "empty_text"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, "truncated_json" if "{" in text else "unparseable"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, "unparseable"
    if not isinstance(obj, dict) or "donor_replacements" not in obj:
        return None, "schema_mismatch"
    return obj, None


def load_proposals():
    """{(item_id, role): payload}.

    A `bundle` payload is the dict covering all three arms; a `donor` payload is the
    bare candidate list from the first, substitution-only run, retained so those items
    still build if a bundle request fails.
    """
    if not PROPOSALS.exists():
        return {}
    out = {}
    with PROPOSALS.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["item_id"], r.get("role", "donor"))] = r.get("payload",
                                                                r.get("candidates"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N built examples")
    ap.add_argument("--submit", action="store_true",
                    help="submit in-context substitute proposals as a batch")
    ap.add_argument("--collect", action="store_true", help="collect proposal batch")
    ap.add_argument("--diagnose", action="store_true",
                    help="re-read an already-billed batch and report losses; costs nothing")
    ap.add_argument("--only-missing", action="store_true",
                    help="submit only items absent from proposals.jsonl")
    ap.add_argument("--driftcheck", metavar="DIR", nargs="?", const="perturb_drift",
                    help="compare decoded unedited prompts in DIR against the committed "
                         "cloze run (default DIR: perturb_drift)")
    args = ap.parse_args()

    if args.driftcheck:
        return mode_driftcheck(args)
    if args.diagnose:
        return mode_diagnose(args)
    if args.submit:
        return mode_submit(args)
    if args.collect:
        return mode_collect(args)

    built, skipped, reject = build(limit=args.limit)
    total = len(built) + len(skipped)
    print(f"built {len(built)} of {total} items")
    if skipped:
        print("\nskipped:")
        for reason, n in Counter(r for _, r in skipped).most_common():
            print(f"  {reason:32} {n}")
    if reject:
        print("\nscreen rejections (candidate-level):")
        for reason, n in reject.most_common():
            print(f"  {reason:32} {n}")

    for b in built[: args.show]:
        print(f"\n--- {b['item_id']}  target={b['target']}  donor={b['donor']} "
              f"-> {b['substitute']}")
        print(f"  orig : ...{b['prefix'][-110:]}")
        print(f"  subst: ...{b['contexts']['donor_substituted'][-110:]}")
        print(f"  del  : ...{b['contexts']['donor_deleted'][-110:]}  [{b['delete_method']}]")
        print(f"  plac : ...{b['contexts']['placebo'][-110:]}  [{b['placebo_word']}"
              f"->{b['placebo_substitute']}]")

    if args.write:
        write_batches(built)


def write_batches(built):
    """Partition targets across batch files so no target repeats within a file.

    evals/composition.load_test_cases keys its dict on `word`, so two items sharing a
    target silently collapse to one. Six of the 182 post-cutoff targets appear twice.
    """
    batches, assigned = [], []
    for b in built:
        placed = False
        for i, seen in enumerate(assigned):
            if b["target"].lower() not in seen:
                seen.add(b["target"].lower())
                batches[i].append(b)
                placed = True
                break
        if not placed:
            assigned.append({b["target"].lower()})
            batches.append([b])

    meta = []
    for i, batch in enumerate(batches):
        # `perturb_batch_*`, not `perturb_*`: the sidecar is perturb_metadata.jsonl, and a
        # glob over the looser pattern picks it up as a batch file. It has no `word` key,
        # so the failure would land inside the composition loader rather than here.
        path = Path(f"perturb_batch_{chr(97 + i)}.jsonl")
        with path.open("w") as fh:
            for b in batch:
                fh.write(json.dumps({
                    "word": b["target"], "category": f"perturb_{b['stratum']}",
                    "year": b["year"],
                    "contexts": {k: v + " [MASK]" for k, v in b["contexts"].items()},
                }) + "\n")
        print(f"wrote {path} ({len(batch)} items)")
        for b in batch:
            meta.append({**{k: v for k, v in b.items() if k != "contexts"},
                         "batch": path.name})
    with open("perturb_metadata.jsonl", "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    print(f"wrote perturb_metadata.jsonl ({len(meta)} rows)")
    write_driftcheck(built)


DRIFT_N = 30
DRIFT_SEED = 20260814


def write_driftcheck(built):
    """Emit unedited prompts, to test that the committed decode is still a baseline.

    The experiment reuses `results/cloze_{model}_details.csv` as its unedited condition
    rather than decoding one, which assumes the numerical environment is unchanged since
    that run. That is an assumption about run provenance, not a measured property: a
    different GPU, CUDA version or floating-point reduction order can resolve beam ties
    and near-threshold ranks differently. These items measure it.

    Drawn evenly from both strata and given unique targets, since load_test_cases keys on
    the target and a repeat would silently drop one of the pair.
    """
    pools = defaultdict(list)
    for b in built:
        pools[b["stratum"]].append(b)
    rng = random.Random(DRIFT_SEED)
    for rows in pools.values():
        rows.sort(key=lambda b: b["item_id"])       # deterministic before shuffling
        rng.shuffle(rows)

    picked, seen = [], set()
    while len(picked) < DRIFT_N and any(pools.values()):
        for stratum in sorted(pools):                # alternate strata, don't fill one
            while pools[stratum] and len(picked) < DRIFT_N:
                b = pools[stratum].pop()
                if b["target"].lower() in seen:
                    continue                        # a repeat would overwrite its twin
                seen.add(b["target"].lower())
                picked.append(b)
                break

    path = Path("perturb_driftcheck.jsonl")
    with path.open("w") as fh:
        for b in picked:
            fh.write(json.dumps({
                "word": b["target"], "category": "driftcheck",
                "year": b["year"],
                "contexts": {"original": b["prefix"] + " [MASK]"},
            }) + "\n")
    print(f"wrote {path} ({len(picked)} items, "
          f"{dict(Counter(b['stratum'] for b in picked))})")


def committed_predictions(model):
    """item_id -> the ten words the committed cloze run recorded for that prompt."""
    src = RESULTS / f"cloze_{model}_details.csv"
    if not src.exists():
        return None
    csv.field_size_limit(10 ** 7)
    out = {}
    with src.open() as fh:
        for row in csv.DictReader(fh):
            target = row["target_word"]
            prefix = extract_prefix(row["text"], target)
            out[SS.item_id("P", SS.NA.norm(target), prefix)] = row["top_10_words"]
    return out


def mode_driftcheck(args):
    """Compare freshly decoded unedited prompts against the committed decode.

    The experiment reuses results/cloze_{model}_details.csv as its unedited condition
    instead of decoding one. That is valid only if the same prompt still produces the same
    ten words -- true under deterministic decoding over fixed weights, provided the
    numerical environment is unchanged since that run. Different hardware, CUDA version or
    floating-point reduction order can resolve beam ties and near-threshold ranks
    differently, and nothing so far has measured whether it did.

    A clean result means no drift was detected at this sample size, not that the two
    environments are identical: systematic drift would show up in nearly every item, but
    drift confined to a few near-threshold ranks could hide in 30.
    """
    src = Path(args.driftcheck)
    total_ok = total_bad = 0
    for model in ("talkie-base", "talkie-web", "typewriter"):
        path = src / f"composition_{model}_detailed.csv"
        if not path.exists():
            print(f"  {model:14} no decode at {path}")
            continue
        committed = committed_predictions(model)
        if committed is None:
            print(f"  {model:14} no committed cloze run to compare against")
            continue
        ok, bad, missing = 0, [], 0
        csv.field_size_limit(10 ** 7)
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("context_level") != "original":
                    continue
                iid = SS.item_id("P", SS.NA.norm(row["target_word"]), row["prefix"])
                was = committed.get(iid)
                if was is None:
                    missing += 1
                elif was == row["top_10_words"]:
                    ok += 1
                else:
                    bad.append((row["target_word"], was, row["top_10_words"]))
        total_ok += ok
        total_bad += len(bad)
        flag = "OK" if not bad else "DRIFT"
        print(f"  {model:14} {flag:6} reproduced {ok}, differed {len(bad)}"
              + (f", unmatched {missing}" if missing else ""))
        for target, was, now in bad[:3]:
            print(f"      {target}\n        committed: {was}\n        now      : {now}")

    print()
    if total_bad:
        print(f"  {total_bad} prompt(s) did not reproduce. The committed decode is NOT a\n"
              f"  valid unedited baseline on this hardware. Decode the unedited condition\n"
              f"  in full rather than comparing across numerics.")
    elif total_ok:
        print(f"  {total_ok} prompts reproduced exactly; no drift detected at this sample\n"
              f"  size. The committed decode stands as the unedited condition.")
    else:
        print("  nothing compared -- run the driftcheck decode first")


if __name__ == "__main__":
    main()
