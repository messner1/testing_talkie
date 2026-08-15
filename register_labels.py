#!/usr/bin/env python3
"""Editorial subject labels, and whether scaffolding concentrates in technical vocabulary.

WHY A SECOND LABEL.  The marker used elsewhere in this project (`technical_composition.is_technical`)
selects words carrying a scientific affix that are rare in a general reference corpus. It
selects on the word's ENDING -- and one of the two routes into being scaffolded is a
passage word sharing the target's ending. Marker and criterion are therefore linked by
construction, and any concentration measured with it is inflated. Editorial subject labels
carry no morphological information at all, so they break that link.

WHERE THE LABELS COME FROM.  OED sense definitions in `Hplm/historical-cloze` are prefixed
with the editorially assigned subject field, in the form "Medicine . An abnormality of...".
This module extracts that prefix, normalises away non-subject qualifiers, and sorts the
result into a scientific field, a non-scientific field, or neither.

THE BOUNDARY IS A JUDGEMENT, AND IT IS IN THE CODE SO IT CAN BE AUDITED.  The register
claim being tested is Halliday's: that scientific prose is dense in neoclassical
word-formation and nominalisation. That motivates drawing the line at the natural
sciences, medicine, engineering and mathematics, and putting the social sciences and
humanities on the other side of it -- Linguistics and Sociology are scholarly registers
but not the neoclassical-derivation register the claim is about. Labels that name a
region, a stylistic register or a structural note rather than a subject are excluded
entirely rather than assigned. `--audit` prints every label with its class and count.

NOTE ON PROVENANCE.  An earlier pass of this analysis was run ad hoc and never committed;
`results.md` quotes 1,559 parseable labels and 908 unambiguous ones from it. This module
reconstructs the extraction rather than reproducing it, so its counts are its own and are
reported as such. `analysis/register_labels.csv` lists every labelled item so the
classification can be checked by hand.

Run: local/bin/python register_labels.py [--audit] [--write]
CPU-only, deterministic. No API calls.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assoc as AS
import neighborhood_analysis as NA
import scaffold_subset as SS
from evals.cloze import extract_prefix

ANALYSIS = Path("analysis")
DATASET = "Hplm/historical-cloze"
SPLIT = "test"
OUT_CSV = ANALYSIS / "register_labels.csv"

# "Medicine . An abnormality ..." -- the label leads the definition and is followed by a
# space-padded full stop. Multi-word fields ("Electrical Engineering", "Christian Church")
# are captured whole.
LABEL_RE = re.compile(r'^\["?([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\s*\.\s')

# Qualifiers OED puts in front of a field name. They modify how generally the label
# applies, not which field it is, so they are stripped before classification.
QUALIFIERS = ("Chiefly ", "Originally ", "Also ", "Modern ", "Early ")

# --------------------------------------------------------------------------- #
# The register split -- the primary classification.
#
# Fields are assigned by the CHARACTERISTIC WORD-FORMATION OF THEIR VOCABULARY, not by
# subject matter and not by anything measured here. A field is `latinate` if its technical
# vocabulary is characteristically built from Latin and Greek combining forms and
# nominalising suffixes (-tion, -ity, -ism, -osis, -itis, -ology); `vernacular` if it is
# characteristically Germanic or French-vernacular and concrete. This is the property
# Halliday's account of scientific register actually names, and it is stated ahead of the
# outcome so the assignment cannot be tuned to it.
#
# The distinction matters because it cuts across subject matter. Philosophy, Grammar and
# Logic are not sciences but are thoroughly neoclassical in their lexis; Mining,
# Agriculture and Shipbuilding are technical trades whose vocabulary is largely native.
# A science-versus-everything-else split therefore mixes registers on both sides, which is
# why it is retained below only as a sensitivity comparison.
# --------------------------------------------------------------------------- #
LATINATE = {
    # natural science and medicine
    "Medicine", "Surgery", "Dentistry", "Obstetrics", "Ophthalmology", "Psychiatry",
    "Pathology", "Physiology", "Anatomy", "Pharmacology", "Immunology", "Homeopathy",
    "Homoeopathy", "Alternative Medicine", "Chemistry", "Physical Chemistry",
    "Organic Chemistry", "Biochemistry", "Physics", "Particle Physics", "Nuclear Physics",
    "Optics", "Mechanics", "Crystallography", "Materials Science", "Astronomy",
    "Meteorology", "Biology", "Cell Biology", "Molecular Biology", "Microbiology",
    "Botany", "Zoology", "Ecology", "Genetics", "Entomology", "Ornithology", "Ichthyology",
    "Conchology", "Embryology", "Cytology", "Taxonomy", "Natural History",
    "Plant Physiology", "Palaeontology", "Geology", "Geomorphology", "Mineralogy",
    "Soil Science", "Physical Geography", "Geography", "Archaeology", "Mathematics",
    "Geometry", "Statistics",
    # technology coined on neoclassical stems
    "Computing", "Electronics", "Engineering", "Electrical Engineering",
    "Nuclear Engineering", "Nuclear Technology", "Telephone Engineering", "Telephony",
    "Aeronautics", "Aviation", "Metallurgy", "Photography", "Cinematography",
    "Radio", "Television", "Sound Recording", "Broadcasting", "Science",
    # scholarly registers with neoclassical lexis but no claim to being sciences
    "Philosophy", "Scholastic Philosophy", "Ancient Greek Philosophy", "Ethics", "Logic",
    "Theology", "Ecclesiastical", "Ecclesiastical Law", "Church History",
    "Grammar", "Linguistics", "Phonetics", "Philology", "Semiotics", "Prosody",
    "Rhetoric", "Literary Criticism", "Psychology", "Social Psychology", "Psychoanalysis",
    "Sociology", "Cultural Anthropology", "Economics", "Political Economy", "Politics",
    "Law", "Scots Law", "Roman Law", "English Law", "Civil Law", "Education",
    "Library Science", "Astrology", "Alchemy", "Perspective",
}

VERNACULAR = {
    # sport and games
    "Sport", "Sporting", "Cricket", "Golf", "Boxing", "Baseball", "American Football",
    "Rugby", "Tennis", "Real Tennis", "Squash", "Bowls", "Curling", "Croquet", "Cycling",
    "Surfing", "Fencing", "Archery", "Wrestling", "Weightlifting", "Athletics",
    "Horse Racing", "Horse Riding", "Motor Racing", "Angling", "Hunting", "Falconry",
    "Australian Rules Football", "Chess", "Cards", "Bridge", "Contract Bridge", "Whist",
    "Dominoes", "Cribbage", "Faro", "Games",
    # crafts, trades and extractive industry
    "Cookery", "Brewing", "Needlework", "Knitting", "Weaving", "Carpentry", "Bookbinding",
    "Printing", "Typography", "Papermaking", "Watchmaking", "Hairdressing", "Tanning",
    "Leather Manufacturing", "Brickmaking", "Bleaching", "Coining", "Farriery", "Building",
    "Mining", "Coal Mining", "Agriculture", "Forestry", "Horticulture", "Fisheries",
    "Shipbuilding", "Surveying", "Town Planning", "Architecture",
    # arts and material culture
    "Music", "Early Music", "Jazz", "Ballet", "Theatre", "Film", "Painting", "Fine Art",
    "Printmaking", "Fashion", "Heraldry", "Armour", "Numismatics", "Science Fiction",
    # military, nautical, and the named churches as institutions
    "Military", "Gunnery", "Firearms", "Fortification", "Navy", "British Navy", "Nautical",
    "Christian Church", "Roman Catholic Church", "Anglican Church", "Buddhism",
    "Spiritualism", "Palmistry", "Hypnotism",
    # history and myth: narrative subjects, not derivational registers
    "History", "Roman History", "Ancient History", "Ancient Greek History",
    "English History", "French History", "British History", "Medieval History",
    "Classical Mythology", "Greek Mythology", "Roman Mythology",
    # commerce
    "Business", "Commerce", "Banking", "Bookkeeping", "Insurance", "Stock Market",
    "Finance",
}


# Latinate and neoclassical endings, used only to STRATIFY the register result -- never to
# assign a register. Separating the two is what shows the association is not merely the
# target's own ending sharing form with a passage word.
LATINATE_ENDING = re.compile(
    r"(tion|sion|ment|ity|ance|ence|ism|ist|ology|osis|itis|oma|ate|ic|ive|ous|al)$")


def _lat_share(rows, reg):
    """Share of targets in `reg`-assigned fields carrying a Latinate ending."""
    sub = [r for r in rows if r[5] == reg]
    return (sum(bool(LATINATE_ENDING.search(r[1] or "")) for r in sub) / len(sub)
            if sub else 0.0)


def register(label):
    """'latinate', 'vernacular', or None -- the primary classification."""
    n = normalise(label)
    if n is None:
        return None
    return "latinate" if n in LATINATE else "vernacular" if n in VERNACULAR else None


# --------------------------------------------------------------------------- #
# The science-versus-the-rest split, retained only as a sensitivity comparison.
# --------------------------------------------------------------------------- #
# Natural science, medicine, engineering, mathematics, computing.
SCIENTIFIC = {
    "Medicine", "Surgery", "Dentistry", "Obstetrics", "Ophthalmology", "Psychiatry",
    "Pathology", "Physiology", "Anatomy", "Pharmacology", "Immunology", "Homeopathy",
    "Homoeopathy", "Alternative Medicine",
    "Chemistry", "Physical Chemistry", "Organic Chemistry", "Biochemistry",
    "Physics", "Particle Physics", "Nuclear Physics", "Optics", "Mechanics",
    "Crystallography", "Materials Science", "Astronomy", "Meteorology",
    "Biology", "Cell Biology", "Molecular Biology", "Microbiology", "Botany", "Zoology",
    "Ecology", "Genetics", "Entomology", "Ornithology", "Ichthyology", "Conchology",
    "Embryology", "Cytology", "Taxonomy", "Natural History", "Plant Physiology",
    "Palaeontology", "Geology", "Geomorphology", "Mineralogy", "Soil Science",
    "Physical Geography", "Geography", "Archaeology",
    "Mathematics", "Geometry", "Statistics",
    "Computing", "Electronics", "Engineering", "Electrical Engineering",
    "Nuclear Engineering", "Nuclear Technology", "Telephone Engineering", "Telephony",
    "Aeronautics", "Aviation", "Metallurgy", "Mining", "Coal Mining", "Surveying",
    "Radio", "Television", "Sound Recording", "Broadcasting", "Photography",
    "Cinematography", "Science", "Agriculture", "Forestry", "Horticulture", "Fisheries",
}

# Arts, sport, games, humanities, commerce, religion, law, crafts and trades. Scholarly
# but not neoclassical-derivational registers (Linguistics, Sociology) sit here too.
NON_SCIENTIFIC = {
    "Music", "Early Music", "Ballet", "Theatre", "Film", "Painting", "Fine Art",
    "Printmaking", "Perspective", "Architecture", "Town Planning", "Fashion",
    "Literary Criticism", "Rhetoric", "Prosody", "Classical Mythology",
    "Greek Mythology", "Roman Mythology", "Heraldry", "Numismatics", "Armour",
    "Philosophy", "Scholastic Philosophy", "Ancient Greek Philosophy", "Ethics",
    "Logic", "Theology", "Ecclesiastical", "Ecclesiastical Law", "Christian Church",
    "Roman Catholic Church", "Anglican Church", "Church History", "Buddhism",
    "Spiritualism", "Astrology", "Alchemy", "Palmistry", "Hypnotism",
    "Linguistics", "Phonetics", "Grammar", "Philology", "Semiotics", "Education",
    "Library Science", "Psychology", "Social Psychology", "Psychoanalysis",
    "Sociology", "Cultural Anthropology", "Politics", "Political Economy",
    "Economics", "Finance", "Business", "Commerce", "Banking", "Bookkeeping",
    "Insurance", "Stock Market", "Law", "Scots Law", "Roman Law", "English Law",
    "Civil Law", "History", "Roman History", "Ancient History", "Ancient Greek History",
    "English History", "French History", "British History", "Medieval History",
    "Military", "Gunnery", "Firearms", "Fortification", "Navy", "British Navy",
    "Nautical", "Shipbuilding",
    "Sport", "Sporting", "Cricket", "Golf", "Boxing", "Baseball", "American Football",
    "Rugby", "Tennis", "Real Tennis", "Squash", "Bowls", "Curling", "Croquet",
    "Cycling", "Surfing", "Fencing", "Archery", "Wrestling", "Weightlifting",
    "Athletics", "Horse Racing", "Horse Riding", "Motor Racing", "Angling", "Hunting",
    "Falconry", "Australian Rules Football", "Chess", "Cards", "Bridge",
    "Contract Bridge", "Whist", "Dominoes", "Cribbage", "Faro", "Games",
    "Cookery", "Brewing", "Needlework", "Knitting", "Weaving", "Carpentry",
    "Bookbinding", "Printing", "Typography", "Papermaking", "Watchmaking",
    "Hairdressing", "Tanning", "Leather Manufacturing", "Brickmaking", "Bleaching",
    "Coining", "Farriery", "Building", "Jazz", "Science Fiction",
}

# Regional, stylistic or structural notes -- not subject fields. Excluded, not assigned.
NOT_A_SUBJECT = re.compile(
    r"^(North American|British|Scottish|Irish English|Australian|Australia|Canadian|"
    r"South African|Caribbean|Newfoundland|Phrases?|Proverb|Prov|Const|Spec|Obsolete|"
    r"Historical|Cant|Secret|Widely|Euphemistically|Disconnected|Drunk|Unfashionable|"
    r"Quarrelsomeness|Asparagus|Petrol|Print|Milieu|Beetle|Triad Society|"
    r"Cambridge University|Oxford University|Ordnance Survey|Geol|Phrase)$")


def normalise(label):
    """Strip qualifiers; return None if the label does not name a subject field."""
    for q in QUALIFIERS:
        if label.startswith(q):
            label = label[len(q):]
            break
    return None if NOT_A_SUBJECT.match(label) else label


def classify(label):
    """'technical', 'general', or None where the field is outside both lists."""
    n = normalise(label)
    if n is None:
        return None
    if n in SCIENTIFIC:
        return "technical"
    if n in NON_SCIENTIFIC:
        return "general"
    return None


def load_labelled():
    """[(item_id, target, year, raw_label, field, cls)] over the whole corpus.

    `item_id` is rebuilt exactly as `scaffold_judge.load_corpus` builds it, so the join to
    the scaffold verdicts is on the same key rather than on a re-derived one.
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=SPLIT)
    rows = []
    for r in ds:
        m = LABEL_RE.match(str(r["sense_descriptions"]))
        if not m:
            continue
        target = (r["word"] or "").strip().lower()
        iid = SS.item_id("P", target, extract_prefix(r["text"], target))
        rows.append((iid, target, r["sense_start_year"], m.group(1),
                     normalise(m.group(1)), register(m.group(1)),
                     classify(m.group(1))))
    return rows


def _cells(rows, verdicts, key, pos, neg):
    """(a, b, c, d) for `pos` vs `neg` on `key` x scaffolded, in assoc's convention.

    `unsure` verdicts are dropped, as they are everywhere else in this project: the
    contrast is a judged scaffold against a judged absence of one, and an abstention is
    neither.
    """
    cnt = Counter()
    for r in rows:
        iid, cls = r[0], r[key]
        if cls not in (pos, neg) or iid not in verdicts:
            continue
        v = verdicts[iid]["verdict"]
        if v in ("scaffolded", "not_scaffolded"):
            cnt[(cls, v == "scaffolded")] += 1
    return (cnt[(pos, True)], cnt[(neg, True)],
            cnt[(pos, False)], cnt[(neg, False)])


def _report(name, a, b, c, d, pos, neg):
    n = a + b + c + d
    print(f"\n  {name}, n = {n}\n")
    print(f"  {'class':<12}{'n':>6}{'scaffolded':>12}{'rate':>8}")
    for lab, k, tot in ((pos, a, a + c), (neg, b, b + d)):
        print(f"  {lab:<12}{tot:>6}{k:>12}{k / tot:>8.3f}" if tot else
              f"  {lab:<12}{'—':>6}")
    if not (a + c) or not (b + d):
        return
    l, orr, (lo, hi), g2 = AS.association(a, b, c, d)
    print(f"\n  lambda {l:.3f}   OR {orr:.2f} [{lo:.2f}, {hi:.2f}]   "
          f"G^2 = {g2:.2f} {AS.stars(g2)}   min expected "
          f"{AS.min_expected(a, b, c, d):.1f}")


def load_verdicts():
    path = ANALYSIS / "judge" / "verdicts.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} -- run scaffold_judge.py --collect all first")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            out[j["item_id"]] = j["judge"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true",
                    help="print every extracted label with its class and count")
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT_CSV}")
    args = ap.parse_args()

    rows = load_labelled()
    verdicts = load_verdicts()

    if args.audit:
        seen = Counter()
        for _, _, _, raw, field, reg, cls in rows:
            seen[(raw, field, reg or "excluded", cls or "excluded")] += 1
        print(f"{'count':>6}  {'raw label':<32}{'normalised':<26}{'register':<12}science")
        for (raw, field, reg, cls), c in seen.most_common():
            print(f"{c:>6}  {raw:<32}{str(field or '-'):<26}{reg:<12}{cls}")

    print(f"\n{len(rows)} items carry a parseable subject label")
    print(f"{sum(1 for r in rows if r[5])} fall in an assigned register class")
    print(f"{sum(1 for r in rows if r[5] and r[0] in verdicts)} of those join a "
          f"scaffold verdict")

    # Per field, so the reader sees what the binary is built from rather than taking it on
    # trust. This is the keyness idiom -- rank candidates by association strength -- and it
    # is what shows the split is not science against everything else.
    print("\n  scaffolding rate by subject field (n >= 15), with its register class\n")
    print(f"  {'field':<24}{'register':<12}{'n':>5}{'scaffolded':>12}{'rate':>8}")
    per, sc = Counter(), Counter()
    reg_of = {}
    for iid, _, _, _, field, reg, _ in rows:
        if not field or iid not in verdicts:
            continue
        v = verdicts[iid]["verdict"]
        if v in ("scaffolded", "not_scaffolded"):
            per[field] += 1
            sc[field] += (v == "scaffolded")
            reg_of[field] = reg or "unassigned"
    for field, n in per.most_common():
        if n >= 15:
            print(f"  {field:<24}{reg_of[field]:<12}{n:>5}{sc[field]:>12}"
                  f"{sc[field] / n:>8.3f}")

    a, b, c, d = _cells(rows, verdicts, 5, "latinate", "vernacular")
    _report("REGISTER -- neoclassical vs native word-formation", a, b, c, d,
            "latinate", "vernacular")

    # The crude figure is inflated by the target's OWN ending. A field assigned `latinate`
    # has more Latinate-ending targets in it, and sharing the target's ending is one of the
    # two routes into being scaffolded, so some of the association is mechanical. Splitting
    # on the target's ending separates the two: the register effect survives among targets
    # carrying no Latinate ending at all, which is the part that cannot be mechanical.
    print("\n  ADJUSTED -- the same contrast within each target-morphology stratum")
    print(f"\n  share of targets carrying a Latinate ending: "
          f"latinate fields {_lat_share(rows, 'latinate'):.3f}, "
          f"vernacular fields {_lat_share(rows, 'vernacular'):.3f}")
    for want, lab in ((True, "targets WITH a Latinate ending"),
                      (False, "targets WITHOUT one")):
        sub = [r for r in rows if bool(LATINATE_ENDING.search(r[1] or "")) is want]
        a2, b2, c2, d2 = _cells(sub, verdicts, 5, "latinate", "vernacular")
        _report(lab, a2, b2, c2, d2, "latinate", "vernacular")

    sa, sb, sc_, sd = _cells(rows, verdicts, 6, "technical", "general")
    _report("SENSITIVITY -- science vs the rest, the cut this replaces",
            sa, sb, sc_, sd, "technical", "general")

    print("\n  Two-way association only. The subject label is a property of the word, not "
          "of\n  the model, so there is no per-model or per-stratum second difference to "
          "take.")

    if args.write:
        ANALYSIS.mkdir(exist_ok=True)
        with OUT_CSV.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["item_id", "target", "year", "raw_label", "field", "register",
                        "science_class", "sc_verdict", "sc_material", "sc_donor"])
            for iid, t, y, raw, f, reg, cls in sorted(
                    rows, key=lambda r: (r[5] or "zz", r[4] or "", r[1])):
                j = verdicts.get(iid) or {}
                w.writerow([iid, t, y, raw, f or "", reg or "excluded",
                            cls or "excluded", j.get("verdict", ""),
                            j.get("material", ""), j.get("donor", "")])
        print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
