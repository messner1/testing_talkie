#!/usr/bin/env python3
"""Terminal annotation interface for neighborhood outcomes.

WHAT IS BEING ANNOTATED.  Whether a model's top-10 predictions look like a *scratchpad* --
the model working a paradigm, assembling candidates out of shared material -- or like a
semantic field, a set of words that mean similar things but look nothing alike.  The first
is the signature of composition, the second of retrieval.

SHEET A -- internal cohesion (default mode).  Shows the ten predicted words and NOTHING
else: no target, no citation, no model identity, and in randomized order so predicted rank
leaks nothing.  This blinding is the whole point.  The analysis selects items by a property
of the *prompt* and measures a property of the *neighborhood*; if the annotator could see
the prompt they could infer whether the item was scaffolded, and the outcome measure would
be contaminated by the selection variable.  An earlier sheet made exactly that mistake.

The question is deliberately about the candidates' relation to EACH OTHER, not to the
target.  A neighborhood that enumerates a paradigm and misses the target is the informative
case -- composition that failed rather than composition that never happened -- and a
target-resemblance measure scores it as a null.

TWO AXES, COUNT-VALUED.  Sheet A asks two independent questions per neighborhood:

    FORM     -- how many of the ten belong to the largest group sharing FORM?
    MEANING  -- how many belong to the largest group sharing MEANING?

They are asked separately because a single "are these variants on a theme?" cannot
discriminate the two poles the study is about: `acetylergic / acetoergic / cholinergic` and
`nerve / impulse / synapse / fibre` are both variants on a theme, and collapsing them makes
the outcome measure blind to the distinction it exists to draw.

They are COUNTED rather than answered yes/no because mixtures are the norm, not the
exception -- most neighborhoods carry a formal group and a semantic group at once, of
different sizes:

    neurosis · neuro · neurotic · be · a · paralysis · neuralgia · paresis · psychosis · disease
        form 4 (neur-), meaning 6 (pathology), 2 filler

A binary scores that identically to a set where all ten are variants and to one where
exactly two are.  A majority rule would not help: it fixes a threshold inside the
annotation, where it can never be varied again.  Counting moves every threshold --
presence, >=3, majority -- into the analysis, where it can be reported as a curve and shown
to be robust.

THREE RULES THAT MAKE THE COUNT WELL-BEHAVED.

1. A group needs at least TWO members, so a count is 0 or 2..10 and never 1.  This makes
   "no formal cohesion" mean exactly `form_n == 0`, which is the branch Sheet B routes on.
2. Count the LARGEST SINGLE group, not the union of all of them.  Above, `neur-` gives 4 and
   `-sis` gives 4; ties do not matter.  The two counts may overlap and may sum past ten -- a
   word can be both form- and meaning-related, and that must not be constrained.
3. Shared material the SLOT'S GRAMMAR forces on every candidate is NOT form cohesion:

       progresses · goes · nears · is · becomes · continues · advances · reaches · proceeds
       -> form 0, meaning 9

   All ten end in `-s`, but every candidate for that slot would, however the model got
   there, so it carries no information about composition versus recall.  If removing the
   inflection leaves nothing shared, the count is 0.  Filler and function words (`be`, `a`,
   `the`, `and`) are simply in neither group.

4. A candidate whose meaning you can only get by PARSING IT does not join the meaning
   group.  `acetylcholide` and `hepton` look like they belong to a semantic field, but the
   only route to that reading is their morphology -- so counting them as shared meaning
   restates the form judgment instead of adding independent evidence, and makes the two
   axes non-independent exactly where they have to separate:

       cholinegic · acetonic · choline · acetylcholine · acetylcholic ·
       acetylcholide · cholic · acetylenic · aceto · acetylcholinic
       -> form 10, meaning ~5 (only the ones whose sense is known independently)

   Without this rule every scratchpad scores meaning 10 as well, and the measure collapses.
   The test is NOT whether the word is attested: that asks a blind annotator for an
   attestation judgment, which this project has twice found unreliable (the coined-word
   flag fired on scan noise -- `caſuall`, `lengtli`, `femaie` -- and the synthetic novelty
   oracle leaked ~5%), and it would penalise real period vocabulary you happen not to know.
   Ask only whether you can state the sense without reading the parts.  A real word you do
   not know self-excludes: you cannot place it in a meaning group either way.

   This makes the axes deliberately asymmetric -- form is judged on the surface, meaning
   requires knowledge you already had -- so meaning counts run lower in coinage-heavy
   neighborhoods.  That is the intended behaviour, not a defect to be corrected: it is what
   separates a scratchpad from a semantic field.

Neither axis is judged against the target, and neither needs the prompt: ten words are
enough to see whether they share shape or whether they are about the same thing.

SHEET B -- aptness for the slot (--sheet-b).  Shows the citation and target, and asks
whether the candidates are apt for THIS slot specifically.  Sheet A already establishes
whether they cohere semantically at all; B is the narrower question of whether that
coherence is the right one for the gap, which genuinely cannot be judged blind.  It is
therefore asked only where Sheet A found no formal cohesion -- the branch where retrieval
is the live hypothesis.

RESTARTABILITY.  Decisions append to analysis/hand_labeling/neighborhood_{A,B}.jsonl with
an fsync each, so a killed terminal loses nothing.  Ids are content hashes, so annotations
stay bound to the neighborhood judged even when the sampled pool changes.  Re-annotating an
item appends a new record; the last one wins.

Keys: Sheet A  [0]..[9] count  [x] 10  [?] unsure   [t] theme  [k] note
      Sheet B  [y] yes  [n] no  [?] unsure                    [k] note
      arrows or [b]/[s] move   [ ] +-10   [u] next unlabelled   [q] save + quit

On Sheet A each item takes two keystrokes: the first answers FORM, the second MEANING.
The prompt says which one it is waiting for, and [b]/left backs out of a half-answered
item without recording anything.

Run:  local/bin/python label_neighborhoods.py            # Sheet A
      local/bin/python label_neighborhoods.py --sheet-b  # Sheet B
      local/bin/python label_neighborhoods.py --stats
      local/bin/python label_neighborhoods.py --export
Standard library only.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS = Path("analysis")
# Overridable so a smoke test can exercise --stats/--export without writing anywhere near
# real annotation. Set NEIGHBORHOOD_LABEL_DIR to a scratch path. This exists because a
# throwaway test script once truncated the real log: the guard it used printed a warning
# and then ran anyway, which is not a guard.
_LABEL_DIR = os.environ.get("NEIGHBORHOOD_LABEL_DIR")
# Explicit None check, NOT `or`: Path("") is PosixPath("."), which is TRUTHY, so
# `Path(os.environ.get(..., "")) or default` silently resolves to the current directory
# whenever the variable is unset. That shipped once and wrote real annotation to the repo
# root instead of analysis/hand_labeling/.
HAND = Path(_LABEL_DIR) if _LABEL_DIR else ANALYSIS / "hand_labeling"
SHEET_A = ANALYSIS / "neighborhood_A_cohesion.csv"
SHEET_B = ANALYSIS / "neighborhood_B_aptness.csv"
LOG_A = HAND / "neighborhood_A.jsonl"
LOG_B = HAND / "neighborhood_B.jsonl"
EXPORT = HAND / "neighborhood_labels.csv"
KEY = ANALYSIS / "neighborhood_KEY.csv"
CALIB = ANALYSIS / "neighborhood_A_calibration.txt"
HELDOUT = ANALYSIS / "neighborhood_A_heldout.txt"
RUBRIC = Path("neighborhood_judge_rubric.json")
TARGET_POS = Path("cache/target_pos.json")
SUBSET = ANALYSIS / "scaffold_subset.csv"

VERDICTS = {"y": True, "n": False, "?": None}          # Sheet B: aptness is genuinely binary

# Sheet A: how many of the ten belong to the largest group. A group needs two members, so a
# valid count is 0 or 2..10 and never 1 -- which is what makes "no formal cohesion" mean
# exactly form_n == 0, the branch Sheet B routes on.
COUNTS = {**{str(d): d for d in range(10)}, "x": 10, "?": None}

# Thresholds the stats view reports the signature split at. The point of counting rather
# than asking for a binary is that no single k is baked in; these are a display default,
# not a decision, and the analysis re-cuts them freely.
KS = (1, 3, 5)

# Sentinel for "leave this field as it was". None cannot serve, because None is a real
# verdict here (unsure) and conflating the two would silently overwrite a considered
# "unsure" with a default every time a note was attached.
_KEEP = object()


def _say(v):
    if v is None:
        return "unsure"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    return "—"


csv.field_size_limit(10 ** 7)


def rubric_help():
    """The judge's rules, rendered from the SAME JSON the judge's prompt is built from.

    Not paraphrased: if the annotator and the judge are to be compared, they must be
    answering the same question, and a hand-written restatement would drift from the
    deployed instrument the first time either changed.
    """
    if not RUBRIC.exists():
        return ["(no rubric file found)"]
    r = json.loads(RUBRIC.read_text())
    out = ["THE TASK", ""] + [x for x in r["task"] if not x.startswith("REPORT")]
    d = r["dimension_1_form"]
    out += ["", "THE RULES", ""] + [f"* {x}" for x in d["rules"]]
    out += ["", d["note_on_kinds"], "", d["not_a_criterion"]]
    if r.get("context_frame"):
        out += ["", "WHAT YOU ARE TOLD ABOUT THE GAP", "", r["context_frame"]["preamble"], ""]
        out += [f"* {x}" for x in r["context_frame"]["rules"]]
    out += ["", "NOT A CRITERION", ""] + [f"* {x}" for x in r["non_criteria"]]
    return out


def _context_for(ids):
    """Slot category and date band per item — the same two facts the judge receives."""
    import csv as _csv
    tpos = json.loads(TARGET_POS.read_text()) if TARGET_POS.exists() else {}
    out = {}
    if not SUBSET.exists():
        return out
    for r in _csv.DictReader(open(SUBSET)):
        if r["nbr_id"] not in ids:
            continue
        y = r.get("year", "")
        yi = int(y) if y.isdigit() else None
        out[r["nbr_id"]] = {
            "slot_pos": tpos.get(r["target_word"].lower()) or "unknown",
            "year": y if yi else "?",
            "register": ("early modern" if yi and yi < 1700 else
                         "18th-19th century" if yi and yi < 1900 else
                         "modern" if yi else ""),
            "words": [w for w in (r["top_10_words"] or "").split("|") if w]}
    return out


def load_items(sheet_b=False):
    path = SHEET_B if sheet_b else SHEET_A
    if not path.exists():
        sys.exit(f"missing {path} — run:\n  local/bin/python scaffold_subset.py"
                 + (" --sheet-b" if sheet_b else ""))
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def load_done(log, valid_ids=None):
    """Last record per id wins. Records for items no longer in the sheet are history,
    not progress, so they are dropped rather than counted."""
    done = {}
    if log.exists():
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # torn final line from a hard kill
                done[rec["id"]] = rec
    if valid_ids is not None:
        done = {k: v for k, v in done.items() if k in valid_ids}
    return done


def append(rec, log):
    HAND.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def draw_wrapped(win, y, x, width, height, chunks):
    """Lay out (text, attr) chunks with word wrap; returns lines used."""
    import curses
    cy, cx = y, x
    for text, attr in chunks:
        for i, word in enumerate(text.split(" ")):
            if not word:
                continue
            piece = word if cx == x else " " + word
            if cx + len(piece) > x + width:
                cy += 1
                cx = x
                piece = word
                if cy >= y + height:
                    return cy - y
            try:
                win.addstr(cy, cx, piece, attr)
            except Exception:
                pass
            cx += len(piece)
    return cy - y + 1


def _pager(stdscr, lines, title):
    import curses
    top = 0
    while True:
        stdscr.erase()
        H, W = stdscr.getmaxyx()
        w = min(W - 2, 96)
        stdscr.addstr(0, 0, f" {title} — [j/k] scroll  [q] back ".ljust(w)[:w],
                      curses.color_pair(1) | curses.A_REVERSE)
        y = 2
        wrapped = []
        for ln in lines:
            if not ln:
                wrapped.append("")
                continue
            cur = ""
            for word in ln.split(" "):
                if len(cur) + len(word) + 1 > w - 4:
                    wrapped.append(cur); cur = word
                else:
                    cur = (cur + " " + word).strip()
            wrapped.append(cur)
        for ln in wrapped[top:top + H - 3]:
            try:
                stdscr.addstr(y, 2, ln[:w - 2],
                              curses.A_BOLD if ln[:1] not in (" ", "*", "") and ln.isupper()
                              else curses.A_NORMAL)
            except Exception:
                pass
            y += 1
        stdscr.refresh()
        ch = stdscr.getch()
        k = chr(ch) if 0 <= ch < 256 else ""
        if k == "q":
            return
        if k == "j" or ch == curses.KEY_DOWN:
            top = min(max(0, len(wrapped) - (H - 4)), top + 1)
        if k == "k" or ch == curses.KEY_UP:
            top = max(0, top - 1)


def run(stdscr, items, done, log, sheet_b, revisit, ctx=None):
    import curses
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)
    curses.init_pair(4, curses.COLOR_GREEN, -1)

    queue = items
    i = 0 if revisit else next((n for n, r in enumerate(queue)
                                if r["id"] not in done), 0)
    msg = ""

    while True:
        it = queue[i]
        prev = done.get(it["id"])
        stdscr.erase()
        H, W = stdscr.getmaxyx()
        if H < 18 or W < 58:
            stdscr.addstr(0, 0, "terminal too small (need 58x18)")
            stdscr.refresh()
            if stdscr.getch() == ord("q"):
                return "quit"
            continue
        w = min(W - 2, 100)

        mark = "✓" if it["id"] in done else "·"
        title = "aptness (Sheet B)" if sheet_b else "cohesion (Sheet A)"
        head = (f" neighborhood — {title} — {mark} {it['id']} — "
                f"{i+1}/{len(queue)} — {len(done)}/{len(items)} done ")
        stdscr.addstr(0, 0, head.ljust(w)[:w],
                      curses.color_pair(1) | curses.A_REVERSE)

        y = 2
        if sheet_b:
            stdscr.addstr(y, 2, "TARGET   ", curses.color_pair(1))
            stdscr.addstr(y, 11, it.get("target_word", ""),
                          curses.color_pair(4) | curses.A_BOLD)
            y += 2
            stdscr.addstr(y, 2, "CITATION", curses.color_pair(1))
            y += 1
            used = draw_wrapped(stdscr, y, 4, w - 6, max(3, (H - y) // 2),
                                [(it.get("prefix_tail", ""), curses.A_NORMAL),
                                 (" ___", curses.color_pair(3) | curses.A_BOLD)])
            y += used + 1

        info = (ctx or {}).get(it["id"])
        if info:
            stdscr.addstr(y, 2, "THE GAP REQUIRES", curses.color_pair(1))
            stdscr.addstr(y, 20, info["slot_pos"], curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(y + 1, 2, "TEXT DATES FROM", curses.color_pair(1))
            stdscr.addstr(y + 1, 20, f"{info['year']}  ({info['register']})",
                          curses.color_pair(4))
            y += 3
        label = "CANDIDATES" if not sheet_b else "PREDICTIONS"
        stdscr.addstr(y, 2, label, curses.color_pair(1))
        y += 1
        words = [c.strip() for c in it["candidates"].replace("|", " · ").split("·")]
        chunks = []
        for n, word in enumerate(words):
            chunks.append((word, curses.color_pair(2) | curses.A_BOLD))
            if n < len(words) - 1:
                chunks.append(("·", curses.color_pair(1)))
        used = draw_wrapped(stdscr, y, 4, w - 6, max(2, H - y - 8), chunks)
        y += used + 1

        if prev:
            if sheet_b:
                shown = f"you said: {_say(prev.get('apt'))}"
            else:
                shown = f"you said: {_say(prev.get('form'))}"
            extra = " · ".join(filter(None, [prev.get("theme"), prev.get("notes")]))
            stdscr.addstr(min(y, H - 8), 4,
                          (shown + (f" — {extra}" if extra else ""))[:w - 6],
                          curses.color_pair(4))

        y = H - 6
        if sheet_b:
            q = "Are these predictions APT for this slot?"
        else:
            q = "How many belong to the largest group sharing FORM?"
        stdscr.addstr(y, 2, q[:w - 4], curses.color_pair(1) | curses.A_BOLD)
        try:
            keys = ("[y] yes    [n] no    [?] unsure" if sheet_b else
                    "[0]…[9] count   [x] 10   [?] unsure    (a group needs 2+)")
            stdscr.addstr(y + 1, 4, keys[:w - 6])
            hint = ("[k] note" if sheet_b else
                    "[t] theme  [k] note  [r] rubric  [o] show overlaps")
            stdscr.addstr(y + 2, 4, hint[:w - 6], curses.color_pair(1))
            stdscr.addstr(y + 3, 4,
                          "←/→ move   [ ] ±10   [u] next unlabelled   [q] save+quit"[:w - 6],
                          curses.color_pair(1))
        except Exception:
            pass
        if msg:
            stdscr.addstr(H - 1, 2, msg[:w - 4], curses.color_pair(3))
        stdscr.refresh()

        ch = stdscr.getch()
        key = chr(ch) if 0 <= ch < 256 else ""
        msg = ""

        def commit(verdict, theme=None, notes=None):
            old = done.get(it["id"], {})
            rec = {"id": it["id"],
                   "theme": old.get("theme", "") if theme is None else theme,
                   "notes": old.get("notes", "") if notes is None else notes,
                   "sheet": "B" if sheet_b else "A",
                   "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if sheet_b:
                rec["apt"] = verdict
                rec["target_word"] = it.get("target_word", "")
            else:
                rec["form"] = verdict
            append(rec, log)
            done[it["id"]] = rec

        def move(d):
            nonlocal i
            i = max(0, min(len(queue) - 1, i + d))

        def ask(prompt):
            curses.echo()
            curses.curs_set(1)
            stdscr.addstr(H - 1, 2, " " * (w - 4))
            stdscr.addstr(H - 1, 2, prompt, curses.color_pair(3))
            try:
                val = stdscr.getstr(H - 1, 2 + len(prompt), 80).decode("utf-8").strip()
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
            nxt = next((n for n in range(i + 1, len(queue))
                        if queue[n]["id"] not in done), None)
            if nxt is None:
                nxt = next((n for n in range(0, i) if queue[n]["id"] not in done), None)
            if nxt is None:
                msg = "nothing left unlabelled"
            else:
                i = nxt
        elif sheet_b and key in VERDICTS:
            v = VERDICTS[key]
            commit(v)
            msg = f"recorded: {_say(v)}"
            move(1)
        elif not sheet_b and key in COUNTS:
            v = COUNTS[key]
            if v == 1:
                # A group of one is not a group. Rejecting 1 outright rather than silently
                # coercing it keeps `form_n == 0` an exact statement about the absence of
                # formal cohesion, which is what Sheet B routes on.
                msg = "a group needs two members — use 0 for none, or 2+"
            else:
                commit(v)
                msg = f"recorded: {_say(v)}"
                move(1)
        elif key == "t" and not sheet_b:
            val = ask("theme: ")
            if val:
                commit(done.get(it["id"], {}).get("form"), theme=val)
                msg = f"theme: {val}"
        elif key == "r":
            _pager(stdscr, rubric_help(), "rubric (identical to the judge's)")
        elif key == "o":
            info = (ctx or {}).get(it["id"])
            ws = (info or {}).get("words") or [c.strip() for c in
                                               it["candidates"].replace("|", "·").split("·")]
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from scaffold_judge import lcs_run
                rows = []
                for i in range(len(ws)):
                    for j in range(i + 1, len(ws)):
                        a, b = ws[i].lower(), ws[j].lower()
                        if len(a) < 3 or len(b) < 3:
                            continue
                        n, sub_ = lcs_run(a, b)
                        if n >= 3:
                            rows.append((n, f"{a} / {b}: '{sub_}' ({n})"))
                rows.sort(reverse=True)
                _pager(stdscr, ["COMPUTED OVERLAPS — the judge is shown these; you are not,",
                                "by default. Shown on request only.", ""] +
                       [t for _, t in rows] or ["(no pair shares three or more characters)"],
                       "overlaps")
            except Exception as e:                              # noqa: BLE001
                msg = f"overlaps unavailable: {e}"
        elif key == "k":
            val = ask("note: ")
            if val:
                f = "apt" if sheet_b else "form"
                commit(done.get(it["id"], {}).get(f), notes=val)
                msg = "note saved"


# --------------------------------------------------------------------------- #
def report(items, done, sheet_b):
    print(f"{len(done)}/{len(items)} labelled")
    if not done:
        return
    if sheet_b:
        c = Counter(_say(r.get("apt")) for r in done.values())
        for k, v in c.most_common():
            print(f"   {k:<8} {v}")
    else:
        vals = [r.get("form") for r in done.values()]
        print("\n   count distribution")
        print(f"   {'n':>4}{'items':>8}")
        for n in [0] + list(range(2, 11)):
            c = sum(v == n for v in vals)
            if c:
                print(f"   {n:>4}{c:>8}")
        u = sum(v is None for v in vals)
        if u:
            print(f"   {'?':>4}{u:>8}")
        known = [v for v in vals if v is not None]
        if known:
            print(f"\n   mean {sum(known)/len(known):.2f}   "
                  f"zero {sum(v == 0 for v in known)/len(known):.1%}")
            print(f"\n   {'k':>4}{'group >= k':>12}")
            for k in KS:
                print(f"   {k:>4}{sum(v >= k for v in known):>12}")
    themes = [r["theme"] for r in done.values() if r.get("theme")]
    if themes:
        print("   themes described:")
        for t in sorted(set(themes)):
            print(f"     - {t}")


def export(done_a, done_b):
    """Join both sheets to the key, so the analysis reads one tidy table."""
    HAND.mkdir(parents=True, exist_ok=True)
    key = {r["id"]: r for r in csv.DictReader(open(KEY))} if KEY.exists() else {}
    cols = ["id", "model", "grade", "route", "is_future", "domain", "target_word",
            "year", "rank", "hit", "locality", "ending_rarity",
            "form_n", "theme", "apt", "notes"]
    with open(EXPORT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rid in sorted(set(done_a) | set(done_b)):
            a, b = done_a.get(rid, {}), done_b.get(rid, {})
            # Form only. The meaning axis was retired in rubric v3.0 after a pilot showed
            # it did not discriminate (a semantic group of 3+ present at 0.959 both with and
            # without a scaffolded prompt) and was the less stable axis (test-retest 0.719
            # against 0.906). Older records keep their meaning counts on disk; nothing
            # reads them.
            w.writerow({**key.get(rid, {}), "id": rid, "form_n": a.get("form"),
                        "theme": a.get("theme", ""), "apt": b.get("apt"),
                        "notes": " · ".join(filter(None, [a.get("notes"),
                                                          b.get("notes")]))})
    return EXPORT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-b", action="store_true",
                    help="annotate aptness instead of cohesion (needs Sheet A first)")
    ap.add_argument("--all", action="store_true", help="start at the first item")
    ap.add_argument("--heldout", action="store_true",
                    help="the 20-item held-out set, with the judge's rubric and context")
    ap.add_argument("--calibration", action="store_true",
                    help="restrict the queue to the judge's calibration subset")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    if args.export:
        a = load_done(LOG_A)
        b = load_done(LOG_B)
        print(f"wrote {export(a, b)}  (A: {len(a)}, B: {len(b)})")
        return

    items = load_items(args.sheet_b)
    if args.heldout and not args.sheet_b:
        if not HELDOUT.exists():
            sys.exit(f"missing {HELDOUT}")
        want = {l.strip() for l in HELDOUT.read_text().splitlines() if l.strip()}
        items = [r for r in items if r["id"] in want]
        print(f"held-out set: {len(items)} items, rubric frozen — press [r] to read it")
    if args.calibration and not args.sheet_b:
        # Filters the QUEUE only -- same sheet, same ids, same log. Labels made here are
        # ordinary Sheet A labels and stay valid if the calibration subset is later
        # widened or discarded; nothing about them is specific to calibration.
        if not CALIB.exists():
            sys.exit(f"missing {CALIB} — run:\n"
                     f"  local/bin/python neighborhood_judge.py --calibration-set")
        want = {l.strip() for l in CALIB.read_text().splitlines() if l.strip()}
        items = [r for r in items if r["id"] in want]
        if not items:
            sys.exit(f"{CALIB} matched no rows in the sheet")
        print(f"calibration subset: {len(items)} of the full sheet")
    log = LOG_B if args.sheet_b else LOG_A
    done = load_done(log, {r["id"] for r in items})

    if args.stats:
        report(items, done, args.sheet_b)
        return

    import curses
    ctx = _context_for({r["id"] for r in items}) if (args.heldout and not args.sheet_b) else None
    res = curses.wrapper(run, items, done, log, args.sheet_b, args.all, ctx)
    print(res if isinstance(res, str) else "done")
    print(f"{len(done)}/{len(items)} labelled → {log}")
    if not args.sheet_b:
        print("next: local/bin/python scaffold_subset.py --sheet-b")


if __name__ == "__main__":
    main()
