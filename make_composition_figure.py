#!/usr/bin/env python3
"""Sense Composition as a Source of Anachronism — one-page exhibit.

Layout: a Prompt row carrying the shared context and the cloze completion each slot is
scored against, then one row per model.  Colour encodes HOW EACH NEIGHBOUR RELATES TO THE
TARGET, because that is what separates the two routes into a slot.

  1. Real Post-Cutoff Term — `positron`, dated 1933 by the corpus's own OED years.  Both
     models see the identical citation.  Talkie-Base ranks it 36th while emitting `hepton`,
     `lithon`, `lithion`: three strings with no attestation anywhere.  Talkie-Web ranks it
     3rd amid `pion`, `muon`, `meson`, all of which postdate 1930.  Same slot, same prompt
     -- one model builds members of the paradigm, the other supplies real ones it should
     not have.  This is the column a contamination audit would flag, for both models, on
     identical evidence.

  2. Constructed Term — `thyminase`, Base rank 2, Web rank 3, both amid further coinages.
     The one synthetic item that survived every restriction: cueing, high context, and the
     Wiktionary novelty re-screen.  Retrieval is impossible here by construction.

  3. Constructed Term, Reduced Context — the same coinage at the probe set's MEDIUM level,
     which still specifies the referent exactly ("the progressive liberation of the free
     base from the nucleoside") but supplies no stem and no -ase exemplar.  Both models
     drop to zero built forms.  A memorised string would survive this; a built one cannot.

The two constructed columns use `-ase` while the natural column uses `-on`, because Base
has no usable natural `-ase` exhibit (its `synthase` neighbourhood is mostly function
words) and no cued `-on` coinage survived the re-screen.  The split appearing across two
unrelated affixes is a feature.

NOVELTY was audited against English Wiktionary on 2026-08-03 (batched API, then a
categories pass to drop foreign-language homographs).  Every chip marked NOVEL returned no
page in any language.  Two casualties from the first draft of the positron panel: `metron`
IS a real English noun (Category:en:Physics), marked MORPH; `protron` exists only under
Category:English misspellings -- a recognised typo of `proton`, not a lexeme -- and is
marked FILL with the other spelling variants.

POST is assigned from the corpus's own OED entry_start_year where it has one (`positron`
1933) and otherwise only where the date is textbook-certain (`meson` 1939, `muon` and
`pion` c.1947-51).  The twentieth-century enzyme names in the -ase panels are not
confidently datable and stay MORPH/SEM -- the conservative call.

Predictions that merely echo the prompt are outlined dashed; the cued target is outlined
solid.

Run: local/bin/python make_composition_figure.py
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGURES = Path("figures")

# The axis is HOW A NEIGHBOUR RELATES TO THE TARGET -- which is what separates the two
# routes into the slot.  Building leaves variants of the target's own form; retrieving
# leaves real words, and in the leakage-rich model, real words it should not have.
#
#   NOVEL  no attestation found -- the model built it (hepton, lithon, thyminease)
#   POST   a real word that postdates the cutoff -- the leakage signature (pion, muon)
#   MORPH  a real in-period word sharing the target's stem or affix (proton, argon)
#   SEM    a real in-period word of related meaning but unrelated form (helium, sugar)
#   FILL   function word, fragment, or spelling variant of the target
NOVEL, POST, MORPH, SEM, FILL = "NOVEL", "POST", "MORPH", "SEM", "FILL"

COLORS = {
    NOVEL: ("#E07B15", "#ffffff"),
    POST:  ("#6A4C93", "#ffffff"),
    MORPH: ("#F7D5AC", "#4a3418"),
    SEM:   ("#9CBBD8", "#12293f"),
    FILL:  ("#F0F0F0", "#9a9a9a"),
}
LEGEND = [(NOVEL, "no attestation found"),
          (POST,  "real, postdates cutoff"),
          (MORPH, "shares stem or affix"),
          (SEM,   "related meaning only"),
          (FILL,  "function word / variant")]

PROMPT_NAT = (
    "The unit of heavy hydrogen, the deuteron, or deuton, as it has been called, bids fair "
    "to rival in interest its recently found cousins, the neutron and ___")
PROMPT_CON = (
    "Where the proteinases split protein and the lipase splits fat, the enzyme effecting "
    "the removal of the pyrimidine base thymine from its sugar in the digest we propose to "
    "designate the ___")
PROMPT_RED = (
    "The progressive liberation of the free base from the nucleoside pointed to the action "
    "of a hitherto unrecognised ___")

# Verbatim top-10s.
#   natural   -> results/cloze_<model>_details.csv, target `positron` (OED 1933), the one
#                cloze item both models share; the same citation the synthetic set later
#                borrowed as `scintillon`'s family template.
#   con / red -> analysis/synthcomp_<model>_joined.csv, target `thyminase`, high / medium
#
# NOVELTY was audited against English Wiktionary on 2026-08-03 (batched API, then a
# categories pass).  Every chip marked NOVEL here returned no page in any language.  Two
# casualties from the first draft of this panel: `metron` IS a real English noun
# (Category:en:Physics) and is marked MORPH; `protron` exists only under
# "Category:English misspellings" -- a recognised typo of `proton`, not a lexeme -- and is
# marked FILL alongside the other spelling variants.
#
# POST is assigned from the corpus's own OED `entry_start_year` where it has one
# (`positron` 1933) and otherwise only where the date is textbook-certain: `meson` 1939,
# `muon` and `pion` c.1947-51.  The enzyme names in the -ase panels are 20th-century but
# not confidently datable, so they stay MORPH/SEM -- the conservative call.
PANELS = {
    ("talkie-base", "natural"): dict(
        cue="positron", rank="rank 36", prompt=PROMPT_NAT,
        words=[("the", FILL), ("proton", MORPH), ("helium", SEM), ("metron", MORPH),
               ("electron", MORPH), ("hepton", NOVEL), ("protron", FILL),
               ("lithon", NOVEL), ("argon", MORPH), ("lithion", NOVEL)]),
    ("talkie-web", "natural"): dict(
        cue="positron", rank="rank 3", prompt=PROMPT_NAT,
        words=[("the", FILL), ("proton", MORPH), ("positron", POST), ("pion", POST),
               ("muon", POST), ("meson", POST), ("pi", FILL), ("electron", MORPH),
               ("its", FILL), ("alpha", SEM)]),
    ("talkie-base", "synth"): dict(
        cue="thyminase", rank="rank 2", prompt=PROMPT_CON,
        words=[("pyrimidase", NOVEL), ("thyminase", NOVEL), ("thymine", MORPH),
               ("pyrimidine", SEM), ("thymase", NOVEL), ("thyminease", NOVEL),
               ("pyrimidinease", NOVEL), ("sugar", SEM), ("enzyme", SEM),
               ("nuclease", MORPH)]),
    ("talkie-web", "synth"): dict(
        cue="thyminase", rank="rank 3", prompt=PROMPT_CON,
        words=[("pyrimidinease", NOVEL), ("thyminease", NOVEL), ("thyminase", NOVEL),
               ("thymine", MORPH), ("thymidase", NOVEL), ("enzyme", SEM),
               ("thymidinease", NOVEL), ("purinease", NOVEL), ("glycosidase", MORPH),
               ("thymidases", NOVEL)]),
    ("talkie-base", "strip"): dict(
        cue="thyminase", rank="missed", prompt=PROMPT_RED,
        words=[("enzyme", SEM), ("ferment", SEM), ("oxidising", SEM), ("class", FILL),
               ("catalyst", SEM), ("type", FILL), ("oxidase", MORPH), ("factor", SEM),
               ("agent", SEM), ("base", SEM)]),
    ("talkie-web", "strip"): dict(
        cue="thyminase", rank="missed", prompt=PROMPT_RED,
        words=[("enzyme", SEM), ("type", FILL), ("nucleoside", SEM), ("class", FILL),
               ("base", SEM), ("hydrolase", MORPH), ("hydrolytic", SEM), ("group", FILL),
               ("specific", FILL), ("nucleotidase", MORPH)]),
}

ROWS = [("talkie-base", "Talkie-Base", "restricted, 1930"),
        ("talkie-web", "Talkie-Web", "unrestricted")]
COLS = [("natural", "Real Post-Cutoff Term", PROMPT_NAT, "positron"),
        ("synth",   "Constructed Term", PROMPT_CON, "thyminase"),
        ("strip",   "Constructed Term, Reduced Context", PROMPT_RED, "thyminase")]

FS = 9.5
CHAR_W = 0.60      # monospace advance as a fraction of font size


def prompt_words(prompt):
    """Word set of a prompt, for marking predictions that merely echo it."""
    return {w.strip("‘’'\",.;:()").lower() for w in prompt.split()}


def axis_width_pts(ax, fig):
    return ax.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi


def draw_chips(ax, words, echo, cue, fig):
    """Colour-coded word chips, wrapping to the panel width.

    Chip and line heights are derived from the axis height in points so the same
    routine works in a wide two-column layout and a narrow three-column one.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    width_pts = bbox.width * 72.0 / fig.dpi
    height_pts = bbox.height * 72.0 / fig.dpi
    chip_h = FS * 1.75 / height_pts
    line_h = FS * 2.45 / height_pts
    x, y = 0.0, 1.0 - line_h * 0.62
    for word, status in words:
        w = (len(word) + 1.6) * CHAR_W * FS / width_pts
        if x + w > 1.0 and x > 0:
            x, y = 0.0, y - line_h
        face, text_c = COLORS[status]
        is_echo = word.lower() in echo
        is_target = word.lower() == cue.lower()
        if is_target:                       # the cued word itself, recalled
            lw, ls, ec = 2.2, "solid", "#111111"
        elif is_echo:                       # merely copied out of the prompt
            lw, ls, ec = 1.5, (0, (1.4, 1.2)), "#555555"
        else:
            lw, ls, ec = 0.7, "solid", "#00000022"
        ax.add_patch(FancyBboxPatch(
            (x, y - chip_h / 2), w * 0.965, chip_h,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            linewidth=lw, linestyle=ls, edgecolor=ec,
            facecolor=face, transform=ax.transAxes, clip_on=False))
        ax.text(x + w * 0.482, y, word, transform=ax.transAxes,
                ha="center", va="center", fontsize=FS, family="monospace",
                color=text_c, fontweight="bold" if status in (NOVEL, POST) else "normal")
        x += w


def main():
    fig = plt.figure(figsize=(15.4, 5.9))
    fig.canvas.draw()

    fig.text(0.5, 0.985, "Sense Composition as a Source of Anachronism",
             ha="center", va="top", fontsize=18, fontweight="bold")

    left, right = 0.088, 0.988
    gap = 0.022
    col_w = (right - left - 2 * gap) / 3
    top = 0.855                       # top of the Prompt row
    prompt_h, row_h, row_gap = 0.170, 0.190, 0.038
    wrap_at = 70

    for ci, (_k, ctitle, _p, _c) in enumerate(COLS):
        cx = left + ci * (col_w + gap) + col_w / 2
        fig.text(cx, top + 0.028, ctitle, ha="center", va="bottom", fontsize=11.5,
                 fontweight="bold", color="#1a1a1a")

    # ---- Prompt row: the shared prompt + the cloze completion it is scored on ----
    fig.text(left - 0.030, top - prompt_h / 2, "Prompt", ha="center", va="center",
             fontsize=11, fontweight="bold", rotation=90, color="#1a1a1a")
    for ci, (_k, _t, prompt, completion) in enumerate(COLS):
        x0 = left + ci * (col_w + gap)
        ax = fig.add_axes([x0, top - prompt_h, col_w, prompt_h])
        ax.set_facecolor("#F6F6F6")
        for sp in ax.spines.values():
            sp.set_edgecolor("#DDDDDD")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.035, 0.88, "\n".join(textwrap.wrap(prompt, wrap_at)),
                transform=ax.transAxes, fontsize=8.0, color="#555555", style="italic",
                va="top", ha="left", linespacing=1.55)
        # the completion the slot is scored against
        ax.text(0.035, 0.14, "cloze:", transform=ax.transAxes,
                fontsize=7.6, color="#8a8a8a", va="center", ha="left")
        ax.text(0.115, 0.14, completion, transform=ax.transAxes, fontsize=9.5,
                family="monospace", fontweight="bold", va="center", ha="left",
                color="#111111")

    # ---- model rows ----
    for ri, (mkey, mlabel, msub) in enumerate(ROWS):
        y0 = top - prompt_h - (ri + 1) * (row_h + row_gap)
        fig.text(left - 0.030, y0 + row_h / 2, mlabel, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", rotation=90, color="#1a1a1a")
        fig.text(left - 0.014, y0 + row_h / 2, msub, ha="center", va="center",
                 fontsize=8, rotation=90, color="#8a8a8a")
        for ci, (ckey, _t, prompt, _c) in enumerate(COLS):
            p = PANELS[(mkey, ckey)]
            x0 = left + ci * (col_w + gap)
            ax = fig.add_axes([x0, y0, col_w, row_h])
            ax.set_facecolor("#FBFBFB")
            for sp in ax.spines.values():
                sp.set_edgecolor("#DDDDDD")
            ax.set_xticks([]); ax.set_yticks([])
            fig.text(x0 + col_w, y0 + row_h + 0.010, p["rank"], fontsize=9,
                     color="#333333", ha="right", va="bottom", fontweight="bold")
            inner = fig.add_axes([x0 + 0.005, y0 + 0.006, col_w - 0.010, row_h - 0.012])
            inner.patch.set_alpha(0)
            draw_chips(inner, p["words"], prompt_words(prompt), p["cue"], fig)

    lx, ly = left, 0.125
    for status, label in LEGEND:
        face, _ = COLORS[status]
        fig.patches.append(plt.Rectangle((lx, ly - 0.015), 0.013, 0.031, facecolor=face,
                                         edgecolor="#00000022", transform=fig.transFigure))
        fig.text(lx + 0.017, ly, label, fontsize=8.5, va="center", color="#333333")
        lx += 0.017 + 0.0052 * len(label) + 0.019
    fig.patches.append(plt.Rectangle((lx, ly - 0.015), 0.013, 0.031, facecolor="#ffffff",
                                     edgecolor="#555555", linewidth=1.5,
                                     linestyle=(0, (1.4, 1.2)), transform=fig.transFigure))
    fig.text(lx + 0.017, ly, "in prompt", fontsize=8.5, va="center", color="#333333")

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "composition_exhibit.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# Second figure: composition is scaffolding-gated, retrieval would not be.
# --------------------------------------------------------------------------- #
# The same coinage `thyminase` under the three context levels the synthetic sets were
# built with.  HIGH supplies the stem (thymine) and the pattern (proteinases, lipase).
# MEDIUM still specifies the referent precisely -- it describes the enzyme's function --
# but hands over no morphology.  LOW is a bare frame.  Composition tracks the *materials*,
# not the semantic specificity: at MEDIUM the model knows what is being asked for and
# answers with real category words instead of building one.
SCAFFOLD = {
    ("talkie-base", "high"): dict(rank="rank 2", words=[
        ("pyrimidase", NOVEL), ("thyminase", NOVEL), ("thymine", MORPH),
        ("pyrimidine", SEM), ("thymase", NOVEL), ("thyminease", NOVEL),
        ("pyrimidinease", NOVEL), ("sugar", SEM), ("enzyme", SEM), ("nuclease", MORPH)]),
    ("talkie-base", "medium"): dict(rank="missed", words=[
        ("enzyme", SEM), ("ferment", SEM), ("oxidising", SEM), ("class", FILL),
        ("catalyst", SEM), ("type", FILL), ("oxidase", MORPH), ("factor", SEM),
        ("agent", SEM), ("base", SEM)]),
    ("talkie-base", "low"): dict(rank="missed", words=[
        ("oil", SEM), ("and", FILL), ("fat", SEM), ("blood", SEM), ("solution", SEM),
        ("extract", SEM), ("product", SEM), ("acid", SEM), ("fatty", SEM),
        ("water", SEM)]),
    ("talkie-web", "high"): dict(rank="rank 3", words=[
        ("pyrimidinease", NOVEL), ("thyminease", NOVEL), ("thyminase", NOVEL),
        ("thymine", MORPH), ("thymidase", NOVEL), ("enzyme", SEM),
        ("thymidinease", NOVEL), ("purinease", NOVEL), ("glycosidase", MORPH),
        ("thymidases", NOVEL)]),
    ("talkie-web", "medium"): dict(rank="missed", words=[
        ("enzyme", SEM), ("type", FILL), ("nucleoside", SEM), ("class", FILL),
        ("base", SEM), ("hydrolase", MORPH), ("hydrolytic", SEM), ("group", FILL),
        ("specific", FILL), ("nucleotidase", MORPH)]),
    ("talkie-web", "low"): dict(rank="missed", words=[
        ("protein", SEM), ("dna", SEM), ("and", FILL), ("proteins", SEM),
        ("recombinant", SEM), ("water", SEM), ("extract", SEM), ("fractions", SEM),
        ("enzyme", SEM), ("antibody", SEM)]),
}
LEVELS = [
    ("high", "stem and pattern supplied",
     "Where the proteinases split protein and the lipase splits fat, the enzyme effecting "
     "the removal of the pyrimidine base thymine from its sugar in the digest we propose to "
     "designate the ___"),
    ("medium", "referent fully described, no morphology",
     "The progressive liberation of the free base from the nucleoside pointed to the action "
     "of a hitherto unrecognised ___"),
    ("low", "bare frame", "Addition of the purified ___"),
]
RATES = {"talkie-base": ["31%", "0%", "0%"], "talkie-web": ["62%", "17%", "0%"]}


def build_scaffolding_figure():
    fig = plt.figure(figsize=(15.2, 7.8))
    fig.canvas.draw()
    fig.text(0.5, 0.992, "Composition needs the materials — retrieval would not",
             ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.944,
             "One coinage, $\\it{thyminase}$, under the three context levels the probe set was "
             "built with. The middle row still says exactly which enzyme is meant; it just "
             "supplies no stem and no pattern.",
             ha="center", va="top", fontsize=10, color="#555555")

    left, right, top = 0.075, 0.985, 0.800
    col_w = (right - left) / 2 - 0.018
    row_h, row_gap = 0.135, 0.115
    for ci, mkey in enumerate(("talkie-base", "talkie-web")):
        cx = left + ci * (col_w + 0.036) + col_w / 2
        label = "Talkie-Base  (restricted, 1930)" if ci == 0 else "Talkie-Web  (unrestricted)"
        fig.text(cx, 0.884, label, ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#1a1a1a")

    for ri, (lvl, gloss, prompt) in enumerate(LEVELS):
        y0 = top - (ri + 1) * row_h - ri * row_gap
        fig.text(left - 0.058, y0 + row_h / 2, lvl.upper(), ha="center", va="center",
                 fontsize=11, fontweight="bold", rotation=90, color="#1a1a1a")
        fig.text(left - 0.040, y0 + row_h / 2, gloss, ha="center", va="center",
                 fontsize=7.6, rotation=90, color="#888888")
        for ci, mkey in enumerate(("talkie-base", "talkie-web")):
            p = SCAFFOLD[(mkey, lvl)]
            x0 = left + ci * (col_w + 0.036)
            ax = fig.add_axes([x0, y0, col_w, row_h])
            ax.set_facecolor("#FBFBFB")
            for s in ax.spines.values():
                s.set_edgecolor("#DDDDDD")
            ax.set_xticks([]); ax.set_yticks([])
            fig.text(x0, y0 + row_h + 0.036, "\n".join(textwrap.wrap(prompt, 126)),
                     fontsize=7.4, color="#8a8a8a", style="italic", va="bottom",
                     ha="left", linespacing=1.5)
            fig.text(x0, y0 + row_h + 0.011,
                     f"across all 42 cued coinages:  {RATES[mkey][ri]} carry a made-up relative",
                     fontsize=8.6, color="#444444", va="bottom", ha="left")
            fig.text(x0 + col_w, y0 + row_h + 0.011, p["rank"], fontsize=9,
                     color="#333333", ha="right", va="bottom", fontweight="bold")
            inner = fig.add_axes([x0 + 0.006, y0 + 0.008, col_w - 0.012, row_h - 0.016])
            inner.patch.set_alpha(0)
            draw_chips(inner, p["words"], prompt_words(prompt), "thyminase", fig)

    fig.text(left, 0.092,
             "A leaked string should survive the loss of morphological scaffolding — an "
             "associative cue (“in 1934 Dale coined the term ___”) would still reach it. A "
             "built one cannot: at MEDIUM the model is told precisely which enzyme is meant "
             "and answers with real category words\ninstead of constructing a name. "
             "Talkie-Base coins nothing at all below HIGH; Typewriter shows no gradient "
             "because it never composes here (4% / 12% / 9%).",
             fontsize=9, va="top", ha="left", color="#444444", linespacing=1.6)

    out = FIGURES / "scaffolding_gradient.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
    build_scaffolding_figure()
