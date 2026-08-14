#!/usr/bin/env python3
"""Terminal labelling interface for recruiting-frame annotation.

WHAT IS BEING LABELLED.  `scaffold_subset.py` grades a cloze prompt by whether it
contains a word sharing form with the target.  That rule is too coarse on its own: half
the graded pool turns out to be inert (the very-common-ending stratum recalls at 0.220
against an unscaffolded baseline of 0.215).  What separates real scaffolding from
coincidence is whether some *construction* recruits the donor into a parallel with the
slot.  The judgment is a **binary** -- recruited or not, with an explicit `?` for the
ambiguous cases -- plus optional free text describing the construction.  No fixed
taxonomy: the analysis needs the binary, and 108 items split across invented categories
would give cells too small to read.  What gets typed into the free-text field is what
tells us later whether a taxonomy is worth having at all.

WHAT THIS SHOWS, AND WHAT IT HIDES.  The prompt, the donor, the shared ending and the
donor's distance from the slot -- all properties of the *input*.  It shows **no model
output whatsoever**: no predictions, no rank, no model name.  That is the blinding that
matters here; a judgment about whether a prompt sets up a form cannot validate anything
if the answer is visible while making it.

THE INITIAL PASS is a spaCy dependency parse, not a keyword match.  Keyword matching
cannot tell whether the donor is *syntactically parallel to the slot*, and on this data it
plainly fails: it fires COORD on `southern` because the citation contains "friends, and
they were talking", and misses `mainland` in "If you're from the North Island, South
Islanders will tell you that they live on the ___", which is an obvious parallel frame.

Parsing the citation with the target filling the slot gives the relation directly.  On
Dale's sentence the slot comes back as `conj` headed by `adrenergic` -- the donor -- which
is coordination established structurally.  The guess is shown *with the evidence that
produced it* so it can be overruled on sight, and it is recorded separately from the
decision so its precision can be reported afterwards.

Using the target to fill the slot is not cheating: the target is the cloze answer key, a
property of the item, not model output.  The blinding that matters here is against the
model's predictions, and those never appear.

RESTARTABILITY.  Every decision is appended to analysis/hand_labeling/frame_labels.jsonl
and flushed immediately, so killing the terminal loses nothing.  On restart, already
labelled ids are skipped; re-labelling an item appends a new record and the last one
wins.  Nothing is ever overwritten in place.

Keys: [y] recruits  [n] does not  [?] ambiguous  [g] accept the guess
      [c] describe the construction  [k] note  [b] back  [s] skip  [q] save+quit

Run:  local/bin/python label_frames.py
      local/bin/python label_frames.py --all       # revisit finished items too
      local/bin/python label_frames.py --export    # write the tidy CSV, no UI
      local/bin/python label_frames.py --stats     # progress + initial-pass agreement
Needs spacy + en_core_web_sm (installed in local/); the UI itself is stdlib curses.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS = Path("analysis")
HAND = ANALYSIS / "hand_labeling"
LOG = HAND / "frame_labels.jsonl"
EXPORT = HAND / "frame_labels.csv"
BLIND_LOG = HAND / "frame_labels_blind.jsonl"
SEED = 20260806
SHEET = ANALYSIS / "scaffold_prompt_labels.csv"
KEY = ANALYSIS / "scaffold_prompt_labels_KEY.csv"

# NO FIXED TAXONOMY.  The judgment is a binary plus optional free text.
#
# An earlier version offered six numbered categories (COORD / ANALOGY / NAMING / LIST /
# CONTRAST / NONE).  That was a mistake on two counts.  It imposed in advance exactly the
# categories that were supposed to emerge from the citations; and the analysis has no use
# for them -- the moderator tables need one binary, "is the donor recruited", and 108
# scaffolded items split six ways gives cells of 5-20, which is not analysable.
#
# It also forced a call on genuinely ambiguous items: `positron` is coordination AND
# enumeration ("the neutron and ___" inside a deuteron/neutron series); `ordination` is
# naming AND contrast ("an antonym to classification" ... "propose the term").  Hence the
# explicit `?` verdict below -- recording uncertainty beats manufacturing a decision.
VERDICTS = {"y": True, "n": False, "?": None}

WORD_RE = re.compile(r"[A-Za-z]+")

# Verbs and nouns that mark a metalinguistic naming frame, where the slot is the name
# being coined rather than an ordinary argument.
NAMING_LEMMAS = {"call", "name", "term", "suggest", "propose", "christen", "dub",
                 "style", "designate", "denominate", "label"}
NAMING_NOUNS = {"word", "term", "name", "expression", "designation"}
CONTRAST_LEMMAS = {"unlike", "antonym", "opposite", "converse", "whereas", "contrast",
                   "rather", "against", "versus"}
ANALOGY_LEMMAS = {"similarly", "likewise", "analogue", "analogous", "analogy",
                  "counterpart", "correspond", "corresponding", "like", "same"}

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError:
            sys.exit("spaCy is required for the initial pass:\n"
                     "  local/bin/pip install spacy\n"
                     "  local/bin/python -m spacy download en_core_web_sm")
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _path_to_root(tok):
    seen, node = [], tok
    while True:
        seen.append(node)
        if node.head == node:
            return seen
        node = node.head


def dep_distance(a, b):
    """Edges between two tokens in the dependency tree (their lowest common ancestor)."""
    pa, pb = _path_to_root(a), _path_to_root(b)
    idx = {t.i: d for d, t in enumerate(pa)}
    for d, t in enumerate(pb):
        if t.i in idx:
            return idx[t.i] + d
    return 99


def analyze_frame(prefix, target, donors):
    """Initial pass over the dependency parse. Returns (category, evidence, features).

    Priority runs most-specific-first: a direct conjunct beats a shared head, which beats
    a mere marker in the neighbourhood. Every branch records the evidence that fired so
    the guess is auditable rather than oracular.
    """
    nlp = get_nlp()
    tgt = target.strip()
    # OED citations carry elided runs of dots ("Island,..South Islanders"); spaCy glues
    # them into one token and the donor then vanishes from the parse. Normalise first.
    text = re.sub(r"\.{2,}", ". ", prefix).strip()
    text = re.sub(r"\s+", " ", text)
    start = len(text) + 1
    doc = nlp(f"{text} {tgt}")

    # Locate the slot by character offset rather than by token match: hyphenated targets
    # ("open-ended") are split into several tokens and an equality test misses them.
    span = doc.char_span(start, start + len(tgt), alignment_mode="expand")
    slot = span.root if span is not None else None
    if slot is None:
        return "NONE", "slot not found in parse", {}

    dtoks = [t for t in doc if t.i != slot.i
             and (t.lower_ in donors or t.text.strip(".,;:'\"()").lower() in donors)]
    if not dtoks:
        return "NONE", "no donor token in parse", {}
    donor = min(dtoks, key=lambda t: (dep_distance(t, slot), abs(t.i - slot.i)))

    feats = {
        "dep_distance": dep_distance(donor, slot),
        "donor_pos": donor.pos_, "slot_pos": slot.pos_,
        "pos_match": donor.pos_ == slot.pos_,
        "linear_gap": slot.i - donor.i,
        "same_sentence": donor.sent == slot.sent,
    }

    def ev(s):
        return s[:70]

    # 1. direct coordination: "the neutron and ___" / "'adrenergic' and ___"
    if slot.dep_ == "conj" and slot.head.i == donor.i:
        return "COORD", ev(f"slot is conj of donor '{donor.text}'"), feats
    if donor.dep_ == "conj" and donor.head.i == slot.i:
        return "COORD", ev(f"donor is conj of slot"), feats
    # 2. sibling conjuncts of a common head -> enumeration if the series is long
    if slot.dep_ == "conj" and donor.dep_ == "conj" and slot.head.i == donor.head.i:
        n = sum(1 for t in doc if t.dep_ == "conj" and t.head.i == slot.head.i)
        return ("LIST" if n >= 2 else "COORD"), ev(f"co-conjuncts of '{slot.head.text}'"), feats

    anc = list(slot.ancestors)
    # 3. metalinguistic naming: slot is what is being called/termed/suggested
    for a in anc[:3]:
        if a.lemma_.lower() in NAMING_LEMMAS:
            return "NAMING", ev(f"governed by '{a.text}'"), feats
    for t in list(slot.children) + list(slot.head.children):
        if t.lemma_.lower() in NAMING_NOUNS:
            return "NAMING", ev(f"'{t.text}' in the slot's phrase"), feats

    # 4. contrast, on the path between donor and slot
    lo, hi = sorted((donor.i, slot.i))
    between = doc[lo:hi + 1]
    for t in between:
        if t.lemma_.lower() in CONTRAST_LEMMAS:
            return "CONTRAST", ev(f"'{t.text}' between donor and slot"), feats
    for t in between:
        if t.lemma_.lower() in ANALOGY_LEMMAS:
            return "ANALOGY", ev(f"'{t.text}' between donor and slot"), feats

    # 5. appositive / shared head -> parallel arguments of one predicate
    if slot.dep_ == "appos" or donor.dep_ == "appos":
        return "COORD", ev("appositive"), feats
    if slot.head.i == donor.head.i and feats["dep_distance"] <= 2:
        return "COORD", ev(f"share head '{slot.head.text}'"), feats

    return "NONE", ev(f"dep-distance {feats['dep_distance']}, no parallel"), feats


def load_items():
    if not SHEET.exists() or not KEY.exists():
        sys.exit(f"missing {SHEET} / {KEY} — run scaffold_subset.py first")
    key = {r["id"]: r for r in csv.DictReader(open(KEY))}
    items = []
    for r in csv.DictReader(open(SHEET)):
        k = key.get(r["id"], {})
        donors = [d for d in (k.get("stem_donors", "") + "|" +
                              k.get("affix_donors", "")).split("|") if d]
        cat, cue, feats = analyze_frame(r["prefix_tail"], r["target_word"], set(donors))
        items.append({
            "dep_distance": feats.get("dep_distance", ""),
            "donor_pos": feats.get("donor_pos", ""),
            "slot_pos": feats.get("slot_pos", ""),
            "pos_match": feats.get("pos_match", ""),
            "id": r["id"],
            "target_word": r["target_word"],
            "prefix": r["prefix_tail"],
            "donors": donors,
            "stem_donors": [d for d in k.get("stem_donors", "").split("|") if d],
            "affix_donors": [d for d in k.get("affix_donors", "").split("|") if d],
            "grade": k.get("grade", "?"),
            "domain": k.get("domain", "?"),
            "ending_types": k.get("ending_types", "?"),
            "ending_rarity": k.get("ending_rarity", "?"),
            "donor_distance": k.get("donor_distance", "?"),
            "locality": k.get("locality", "?"),
            "kind": k.get("kind", "?"),
            "guess_category": cat,
            "guess_cue": cue,
        })
    return items


def load_done(valid_ids=None):
    """Last record per id wins, so re-labelling just appends.

    ``valid_ids`` drops records whose item is not in the current sheet. The log is
    append-only and ids are content-derived, so a record for a citation that has since
    been resampled out is history rather than an error -- but counting it would inflate
    progress and corrupt the agreement figure.
    """
    done = {}
    if LOG.exists():
        with open(LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # a torn final line from a hard kill; ignore it
                done[rec["id"]] = rec
    if valid_ids is not None:
        done = {k: v for k, v in done.items() if k in valid_ids}
    return done


def append(rec, log=None):
    HAND.mkdir(parents=True, exist_ok=True)
    with open(log or LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def blind_subset(items, n):
    """Stratified re-label sample for the anchoring check.

    The main pass showed the labeller the parse's verdict, so agreement between the parse
    and the labels is not independent -- an interaction discovered there (affix-only
    donors recruiting at 0.65 with a frame vs 0.07 without) could be partly anchoring.
    This draws a balanced sample across route x frame, the four cells that carry that
    interaction, and re-labels them with the guess hidden.
    """
    import csv as _csv
    sub = {}
    path = Path("analysis/scaffold_subset.csv")
    if path.exists():
        with open(path) as f:
            for r in _csv.DictReader(f):
                if r["model"] == "talkie-base":
                    sub[r["item_id"]] = r
    cells = defaultdict(list)
    for it in items:
        s_ = sub.get(it["id"])
        if not s_ or s_["grade"] == "NONE":
            continue
        cells[(s_["route"], it["guess_category"] != "NONE")].append(it)
    rng = random.Random(SEED)
    per = max(1, n // max(1, len(cells)))
    out = []
    for k in sorted(cells, key=str):
        pool = sorted(cells[k], key=lambda x: x["id"])
        out.extend(rng.sample(pool, min(per, len(pool))))
    rng.shuffle(out)
    return out, {k: len(v) for k, v in cells.items()}


def compare_blind(items):
    """How much did seeing the guess move the labels?"""
    by_id = {i["id"]: i for i in items}
    anchored = load_done({i["id"] for i in items})
    blind = {}
    if BLIND_LOG.exists():
        for line in open(BLIND_LOG):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    blind[r["id"]] = r
                except json.JSONDecodeError:
                    pass
    both = [i for i in blind if i in anchored
            and blind[i]["recruits"] is not None
            and anchored[i]["recruits"] is not None]
    if not both:
        print("no overlapping blind labels yet")
        return
    same = sum(1 for i in both if blind[i]["recruits"] == anchored[i]["recruits"])
    print(f"blind vs anchored: {same}/{len(both)} agree = {same/len(both):.3f}")
    gs = sum(1 for i in both if anchored[i]["recruits"] == (by_id[i]["guess_category"] != "NONE"))
    gb = sum(1 for i in both if blind[i]["recruits"] == (by_id[i]["guess_category"] != "NONE"))
    print(f"  agreement WITH the parse — anchored {gs}/{len(both)} = {gs/len(both):.3f}, "
          f"blind {gb}/{len(both)} = {gb/len(both):.3f}")
    print(f"  anchoring inflation: {(gs-gb)/len(both):+.3f}")
    flipped = [i for i in both if blind[i]["recruits"] != anchored[i]["recruits"]]
    if flipped:
        print("\n  items that moved when the guess was hidden:")
        for i in flipped:
            print(f"    {by_id[i]['target_word']:<18} anchored="
                  f"{'Y' if anchored[i]['recruits'] else 'n'} "
                  f"blind={'Y' if blind[i]['recruits'] else 'n'} "
                  f"(guess said {by_id[i]['guess_category']})")


def export(items, done):
    HAND.mkdir(parents=True, exist_ok=True)
    cols = ["id", "target_word", "grade", "domain", "kind", "donors",
            "ending_types", "ending_rarity", "donor_distance", "locality",
            "dep_distance", "donor_pos", "slot_pos", "pos_match",
            "guess_category", "guess_cue", "guess_recruits",
            "recruits", "construction", "notes", "labelled_at"]
    with open(EXPORT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            rec = done.get(it["id"])
            if not rec:
                continue
            w.writerow({**it, **rec, "donors": "|".join(it["donors"])})
    return EXPORT


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def draw_prefix(win, y, x, width, height, prefix, donors, target):
    """Word-wrap the citation, reverse-video the donors, mark the slot."""
    import curses
    tokens = re.split(r"(\W+)", prefix + " ___")
    cy, cx = y, x
    for tok in tokens:
        if cy >= y + height:
            break
        if not tok:
            continue
        if cx + len(tok) > x + width:
            cy += 1
            cx = x
            if cy >= y + height:
                break
            if tok.strip() == "":
                continue
        attr = curses.A_NORMAL
        low = tok.lower()
        if low in donors:
            attr = curses.color_pair(2) | curses.A_BOLD
        elif tok == "___":
            attr = curses.color_pair(3) | curses.A_BOLD
        elif low == target.lower():
            attr = curses.color_pair(4) | curses.A_BOLD
        try:
            win.addstr(cy, cx, tok, attr)
        except Exception:
            pass
        cx += len(tok)
    return cy - y + 1


def run(stdscr, items, done, revisit, blind=False, log=None):
    import curses
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)      # chrome
    curses.init_pair(2, curses.COLOR_YELLOW, -1)    # donor
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)   # slot
    curses.init_pair(4, curses.COLOR_GREEN, -1)     # target
    curses.init_pair(5, curses.COLOR_RED, -1)       # warnings

    # Always hold the whole sheet, so navigation can move freely in both directions and
    # revisit anything. `revisit` only changes where the cursor starts.
    queue = items
    if not queue:
        return "All items already labelled. Use --all to revisit, --export for the CSV."

    i = 0 if revisit else next((n for n, it in enumerate(queue)
                                if it["id"] not in done), 0)
    msg = ""

    while True:
        it = queue[i]
        prev = done.get(it["id"])
        stdscr.erase()
        H, W = stdscr.getmaxyx()
        if H < 20 or W < 60:
            stdscr.addstr(0, 0, "terminal too small (need 60x20)")
            stdscr.refresh()
            if stdscr.getch() == ord("q"):
                return "quit"
            continue

        w = min(W - 2, 100)
        mark = "✓" if it["id"] in done else "·"
        head = (f" frame labelling{' [BLIND]' if blind else ''} — {mark} "
                f"{it['id']} — {i+1}/{len(queue)} — {len(done)}/{len(items)} done ")
        stdscr.addstr(0, 0, head.ljust(w), curses.color_pair(1) | curses.A_REVERSE)

        y = 2
        stdscr.addstr(y, 2, "TARGET   ", curses.color_pair(1))
        stdscr.addstr(y, 11, it["target_word"], curses.color_pair(4) | curses.A_BOLD)
        y += 1
        d = ", ".join(it["stem_donors"]) or "—"
        a = ", ".join(it["affix_donors"]) or "—"
        stdscr.addstr(y, 2, "DONORS   ", curses.color_pair(1))
        stdscr.addstr(y, 11, f"stem: {d}   affix: {a}"[:w - 12],
                      curses.color_pair(2))
        y += 1
        rar = it["ending_rarity"]
        rar_attr = curses.color_pair(5) if rar == "very-common" else curses.A_NORMAL
        stdscr.addstr(y, 2, "ENDING   ", curses.color_pair(1))
        stdscr.addstr(y, 11, f"{it['ending_types']} corpus types ({rar})", rar_attr)
        if rar == "very-common":
            stdscr.addstr(y, 11 + len(f"{it['ending_types']} corpus types ({rar})") + 2,
                          "← inert stratum", curses.color_pair(5))
        y += 1
        stdscr.addstr(y, 2, "DISTANCE ", curses.color_pair(1))
        stdscr.addstr(y, 11, f"{it['donor_distance']} words from slot ({it['locality']})"
                             f"   ·  dep-distance {it['dep_distance']}")
        y += 1
        stdscr.addstr(y, 2, "POS      ", curses.color_pair(1))
        pm = it["pos_match"]
        stdscr.addstr(y, 11, f"donor {it['donor_pos']} / slot {it['slot_pos']}",
                      curses.A_NORMAL if pm else curses.color_pair(5))
        if pm is False:
            stdscr.addstr(y, 11 + len(f"donor {it['donor_pos']} / slot {it['slot_pos']}") + 2,
                          "← mismatch, donor may be spurious", curses.color_pair(5))
        y += 2

        stdscr.addstr(y, 2, "PROMPT", curses.color_pair(1))
        y += 1
        used = draw_prefix(stdscr, y, 4, w - 6, max(3, H - y - 12),
                           it["prefix"], {d.lower() for d in it["donors"]},
                           it["target_word"])
        y += used + 1

        if not blind:
            g = "recruits" if it["guess_category"] != "NONE" else "no"
            stdscr.addstr(y, 2, "GUESS    ", curses.color_pair(1))
            stdscr.addstr(y, 11, g, curses.A_BOLD)
            if it["guess_cue"]:
                stdscr.addstr(y, 11 + len(g) + 2, f'← {it["guess_cue"]}'[:w - 14 - len(g)],
                              curses.color_pair(1))
            y += 1
        if prev and not blind:
            r = prev.get("recruits")
            shown = {True: "recruits", False: "no", None: "ambiguous"}.get(r, str(r))
            extra = " · ".join(filter(None, [prev.get("construction"), prev.get("notes")]))
            stdscr.addstr(y, 11, f"you said: {shown}"
                                 + (f" — {extra}" if extra else "")[:w - 30],
                          curses.color_pair(4))
            y += 1

        y = max(y + 1, H - 6)
        stdscr.addstr(y, 2, "Does the prompt RECRUIT the donor into the target's form?",
                      curses.color_pair(1) | curses.A_BOLD)
        try:
            stdscr.addstr(y + 1, 4,
                          "[y] yes    [n] no    [?] ambiguous    [g] accept guess"[:w - 6])
            stdscr.addstr(y + 2, 4,
                          "[c] describe the construction (free text)   [k] note"[:w - 6],
                          curses.color_pair(1))
            stdscr.addstr(y + 3, 4,
                          "←/→ move   [ ] ±10   Home/End   [u] next unlabelled   "
                          "[q] save+quit"[:w - 6],
                          curses.color_pair(1))
        except Exception:
            pass
        if msg:
            stdscr.addstr(H - 1, 2, msg[:w - 4], curses.color_pair(3))
        stdscr.refresh()

        ch = stdscr.getch()
        key = chr(ch) if 0 <= ch < 256 else ""
        msg = ""

        def commit(recruits, construction=None, notes=None):
            old = done.get(it["id"], {})
            rec = {"id": it["id"], "target_word": it["target_word"],
                   "recruits": recruits,
                   "construction": (old.get("construction", "") if construction is None
                                    else construction),
                   "notes": (old.get("notes", "") if notes is None else notes),
                   "guess_recruits": it["guess_category"] != "NONE",
                   "guess_category": it["guess_category"],
                   "guess_cue": it["guess_cue"],
                   "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            append(rec, log)
            done[it["id"]] = rec

        def advance():
            nonlocal i
            if i < len(queue) - 1:
                i += 1

        def move(delta):
            nonlocal i
            i = max(0, min(len(queue) - 1, i + delta))

        def next_unlabelled(direction=1):
            nonlocal i
            n = len(queue)
            for step in range(1, n + 1):
                j = (i + direction * step) % n
                if queue[j]["id"] not in done:
                    i = j
                    return True
            return False

        def ask(label):
            curses.echo()
            curses.curs_set(1)
            stdscr.addstr(H - 1, 2, " " * (w - 4))
            stdscr.addstr(H - 1, 2, label, curses.color_pair(3))
            try:
                val = stdscr.getstr(H - 1, 2 + len(label), 90).decode("utf-8").strip()
            except Exception:
                val = ""
            curses.noecho()
            curses.curs_set(0)
            return val

        if key == "q":
            return "saved"
        elif ch == curses.KEY_LEFT or key in ("b", ","):
            move(-1)
        elif ch == curses.KEY_RIGHT or key in ("s", "."):
            move(1)
        elif key == "[":
            move(-10)
        elif key == "]":
            move(10)
        elif ch == curses.KEY_HOME:
            i = 0
        elif ch == curses.KEY_END:
            i = len(queue) - 1
        elif key == "u":
            if not next_unlabelled(1):
                msg = "nothing left unlabelled"
        elif key == "U":
            if not next_unlabelled(-1):
                msg = "nothing left unlabelled"
        elif key in VERDICTS:
            v = VERDICTS[key]
            commit(v)
            msg = {True: "recorded: recruits", False: "recorded: does not recruit",
                   None: "recorded: ambiguous"}[v]
            advance()
        elif key == "g":
            commit(it["guess_category"] != "NONE")
            msg = f"accepted guess ({it['guess_category']})"
            advance()
        elif key == "c":
            val = ask("construction: ")
            if val:
                commit(done.get(it["id"], {}).get("recruits", True), construction=val)
                msg = f"construction: {val}"
        elif key == "k":
            val = ask("note: ")
            if val:
                commit(done.get(it["id"], {}).get("recruits", None), notes=val)
                msg = "note saved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="revisit already-labelled items")
    ap.add_argument("--export", action="store_true", help="write the CSV and exit")
    ap.add_argument("--stats", action="store_true", help="print progress and exit")
    ap.add_argument("--blind", action="store_true",
                    help="re-label a stratified subset with the parse's guess HIDDEN")
    ap.add_argument("--blind-n", type=int, default=40, help="size of the blind subset")
    ap.add_argument("--compare", action="store_true",
                    help="blind vs anchored agreement, and how much the guess moved them")
    args = ap.parse_args()

    items = load_items()
    if args.compare:
        compare_blind(items)
        return
    if args.blind:
        items, cells = blind_subset(items, args.blind_n)
        print(f"blind subset: {len(items)} items across {len(cells)} route x frame cells",
              file=sys.stderr)
        done = load_done({it["id"] for it in items})
        blind_done = {}
        if BLIND_LOG.exists():
            for line in open(BLIND_LOG):
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r["id"] in {i["id"] for i in items}:
                            blind_done[r["id"]] = r
                    except json.JSONDecodeError:
                        pass
        import curses
        res = curses.wrapper(run, items, blind_done, False, True, BLIND_LOG)
        print(res if isinstance(res, str) else "done")
        print(f"{len(blind_done)}/{len(items)} labelled blind -> {BLIND_LOG}")
        print("compare with: local/bin/python label_frames.py --compare")
        return
    done = load_done({it["id"] for it in items})

    if args.export or args.stats:
        if args.export:
            print(f"wrote {export(items, done)} ({len(done)}/{len(items)} labelled)")
        agree = sum(1 for it in items
                    if it["id"] in done and done[it["id"]]["recruits"] is not None
                    and done[it["id"]]["recruits"] == (it["guess_category"] != "NONE"))
        decided = sum(1 for r in done.values() if r["recruits"] is not None)
        print(f"{len(done)}/{len(items)} labelled")
        if done:
            from collections import Counter
            c = Counter({True: "recruits", False: "no", None: "ambiguous"}[r["recruits"]]
                        for r in done.values())
            for k, v in c.most_common():
                print(f"   {k:<10} {v}")
            if decided:
                print(f"initial pass agrees on {agree}/{decided} decided = "
                      f"{agree/decided:.3f}")
            free = [r["construction"] for r in done.values() if r.get("construction")]
            if free:
                print("   constructions described:")
                for t in sorted(set(free)):
                    print(f"     - {t}")
        return

    import curses
    result = curses.wrapper(run, items, done, args.all)
    print(result if isinstance(result, str) else "done")
    print(f"{len(done)}/{len(items)} labelled → {LOG}")
    print(f"export with: local/bin/python label_frames.py --export")


if __name__ == "__main__":
    main()
