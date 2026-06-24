#!/usr/bin/env python3
"""Probe cloze details for post-cutoff items the models still recall in top-100.

Two distinct phenomena, keyed on the two date columns:

  (a) post-cutoff SENSE  -> ``year`` (sense coinage date) > cutoff
        A new *sense* of an old word. Recall here mixes leakage with word-level
        compositional reasoning (the headword existed pre-cutoff; only the sense
        is new). See tt.md "aboard" (1936 metaphorical sense, 1458 entry).

  (b) post-cutoff WORD   -> ``entry_start_year`` (first attestation) > cutoff
        The whole headword is novel; it could not have been seen pre-cutoff in
        any sense. Recall here is the cleaner leakage-vs-inference question and
        is the seed pool for the synthetic compositionality sets. See tt.md
        positron (1933), synthase (1954), cholinergic (1934), exciton (1936).

In both cases we keep only rows actually recalled in top-100: ``0 < rank <= 100``.

Outputs (under analysis/):
  - postcutoff_<model>_senses.csv
  - postcutoff_<model>_words.csv
  - postcutoff_seedwords_<cutoff>.csv   (deduped word list, the Step-3 seed pool)
  - postcutoff_summary.md               (counts + strongest examples per model)

Run: local/bin/python probe_postcutoff.py
"""

import pandas as pd
from pathlib import Path

RESULTS = Path("results")
ANALYSIS = Path("analysis")

# model -> (details csv, cutoff year). Talkie=1930, Typewriter=1913 (evals/registry.py).
MODELS = {
    "talkie-base": (RESULTS / "cloze_talkie-base_details.csv", 1930),
    "talkie-web": (RESULTS / "cloze_talkie-web_details.csv", 1930),
    "typewriter": (RESULTS / "cloze_typewriter_details.csv", 1913),
}

# Which model's recalled post-cutoff WORDS seed the synthetic sets, per cutoff.
# Only the DATA-RESTRICTED models seed: talkie-base (1930) and typewriter (1913).
# talkie-web is unrestricted, so its post-cutoff recall is expected leakage, a
# different phenomenon — it is excluded from the seed pools (but still probed/reported).
SEED_MODEL = {1930: "talkie-base", 1913: "typewriter"}

# Columns surfaced in the example tables (mirrors tt.md's Context/Word/Date/Rank).
COLS = ["text", "target_word", "year", "entry_start_year", "rank", "top_10_words"]
TOPN_EXAMPLES = 15  # strongest examples per slice in the markdown summary


def load(path):
    df = pd.read_csv(path)
    # year / entry_start_year may be blank -> NaN floats; rank is int with -1 = absent.
    for c in ("year", "entry_start_year", "rank"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def recalled(df, k=100):
    return df[(df["rank"] > 0) & (df["rank"] <= k)]


def post_cutoff_senses(df, cutoff, k=100):
    r = recalled(df, k)
    return r[r["year"] > cutoff].sort_values("rank")


def post_cutoff_words(df, cutoff, k=100):
    r = recalled(df, k)
    return r[r["entry_start_year"] > cutoff].sort_values("rank")


def md_table(sub, n=TOPN_EXAMPLES):
    """Compact markdown table of the strongest (lowest-rank) examples."""
    rows = sub.head(n)
    if len(rows) == 0:
        return "_none_\n"
    out = ["| Context | Word | Sense yr | Entry yr | Rank |",
           "| --- | --- | --- | --- | --- |"]
    for _, r in rows.iterrows():
        ctx = str(r["text"]).replace("|", "\\|").replace("\n", " ")
        if len(ctx) > 110:
            ctx = ctx[:107] + "..."
        sy = "" if pd.isna(r["year"]) else int(r["year"])
        ey = "" if pd.isna(r["entry_start_year"]) else int(r["entry_start_year"])
        out.append(f"| {ctx} | **{r['target_word']}** | {sy} | {ey} | {int(r['rank'])} |")
    return "\n".join(out) + "\n"


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary = ["# Post-cutoff recall probe",
               "",
               "Items the models still rank in top-100 despite post-dating their cutoff.",
               "`year` = sense coinage date; `entry_start_year` = whole-word first attestation.",
               ""]

    # Collect seed-word pools per cutoff (unioned across models sharing a cutoff).
    seed_pool = {}  # cutoff -> {word: row dict}

    for model, (path, cutoff) in MODELS.items():
        if not path.exists():
            print(f"!! missing {path}, skipping {model}")
            continue
        df = load(path)
        senses = post_cutoff_senses(df, cutoff)
        words = post_cutoff_words(df, cutoff)

        senses[COLS].to_csv(ANALYSIS / f"postcutoff_{model}_senses.csv", index=False)
        words[COLS].to_csv(ANALYSIS / f"postcutoff_{model}_words.csv", index=False)

        n_recalled = len(recalled(df))
        print(f"{model} (cutoff {cutoff}): {n_recalled} recalled@100 | "
              f"post-cutoff senses={len(senses)} words={len(words)}")

        summary += [
            f"## {model}  (cutoff {cutoff})",
            "",
            f"- recalled in top-100: **{n_recalled}**",
            f"- of which post-cutoff **senses** (`year` > {cutoff}): **{len(senses)}**",
            f"- of which post-cutoff **words** (`entry_start_year` > {cutoff}): **{len(words)}**",
            "",
            f"### (a) Strongest post-cutoff SENSES — {model}",
            md_table(senses),
            f"### (b) Strongest post-cutoff WORDS — {model}",
            md_table(words),
        ]

        # Seed pool: ONLY the designated data-restricted seed model contributes.
        if SEED_MODEL.get(cutoff) != model:
            continue
        pool = seed_pool.setdefault(cutoff, {})
        for _, r in words.iterrows():
            w = str(r["target_word"]).strip().lower()
            entry_yr = None if pd.isna(r["entry_start_year"]) else int(r["entry_start_year"])
            rank = int(r["rank"])
            if w not in pool or rank < pool[w]["best_rank"]:
                pool[w] = {
                    "word": w,
                    "entry_start_year": entry_yr,
                    "best_rank": rank,
                    "example_context": str(r["text"]),
                    "models": set(),
                }
            pool[w]["models"].add(model)

    # Write per-cutoff deduped seed-word lists (Step-3 input). Seeded ONLY from the
    # data-restricted model per cutoff (web excluded — see SEED_MODEL).
    summary += [f"## Seed-word pools (post-cutoff WORDS from the restricted seed model only)",
                f"Seed models: {SEED_MODEL}. talkie-web excluded (unrestricted).", ""]
    for cutoff, pool in sorted(seed_pool.items()):
        rows = sorted(pool.values(), key=lambda d: d["best_rank"])
        seed_df = pd.DataFrame([{
            "word": d["word"],
            "entry_start_year": d["entry_start_year"],
            "best_rank": d["best_rank"],
            "models": "|".join(sorted(d["models"])),
            "example_context": d["example_context"],
        } for d in rows])
        out = ANALYSIS / f"postcutoff_seedwords_{cutoff}.csv"
        seed_df.to_csv(out, index=False)
        print(f"seed pool (cutoff {cutoff}): {len(seed_df)} distinct words -> {out}")
        summary += [
            f"- cutoff **{cutoff}**: **{len(seed_df)}** distinct post-cutoff words "
            f"recalled in top-100 -> `{out}`",
            f"  - sample: {', '.join(seed_df['word'].head(25).tolist())}",
            "",
        ]

    (ANALYSIS / "postcutoff_summary.md").write_text("\n".join(summary))
    print(f"\nWrote analysis/postcutoff_summary.md")


if __name__ == "__main__":
    main()
