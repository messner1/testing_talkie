"""Leakage is keyed on sense_start_year; entry_start_year is preserved (the fix_detailed fix).

Network-free: FakeBackend supplies top_k_words and model.eval(), so run_cloze runs without
any model or dataset download.
"""

import pandas as pd

from evals.cloze import run_cloze
from evals.prompts import PlainPrefixPrompt
from evals.registry import ModelSpec

from .fakes import FakeBackend

SPEC = ModelSpec(name="fake", family="talkie", cutoff_year=1930, prompt=PlainPrefixPrompt())


def test_is_future_follows_sense_year_and_entry_is_preserved(tmp_path):
    dataset = [
        # future by SENSE (1990) but past by entry (1450) -> is_future must be 1
        {"text": "a cat", "word": "cat", "sense_start_year": 1990, "entry_start_year": 1450},
        # past by SENSE (1500) but future by entry (1990) -> is_future must be 0
        {"text": "a dog", "word": "dog", "sense_start_year": 1500, "entry_start_year": 1990},
        # undated sense -> treated as past (is_future 0); entry still preserved
        {"text": "the", "word": "the", "sense_start_year": None, "entry_start_year": 1700},
    ]
    prefix = str(tmp_path / "cloze_fake")

    df = run_cloze(FakeBackend(), SPEC, dataset, output_prefix=prefix, ks=[10])

    # is_future is driven by sense_start_year, NOT entry_start_year.
    assert list(df["is_future"]) == [1, 0, 0]
    # The leakage `year` column holds sense_start_year (None -> NaN via pandas).
    assert df["year"].iloc[0] == 1990 and df["year"].iloc[1] == 1500
    assert pd.isna(df["year"].iloc[2])
    # entry_start_year is preserved as its own column for downstream use.
    assert list(df["entry_start_year"]) == [1450, 1990, 1700]


def test_details_csv_has_both_year_columns(tmp_path):
    dataset = [
        {"text": "a cat", "word": "cat", "sense_start_year": 1850, "entry_start_year": 1400},
    ]
    prefix = str(tmp_path / "cloze_fake")
    run_cloze(FakeBackend(), SPEC, dataset, output_prefix=prefix, ks=[10])

    df = pd.read_csv(f"{prefix}_details.csv")
    cols = list(df.columns)
    assert "year" in cols and "entry_start_year" in cols
    # entry_start_year sits right after year (fixed/ column order + the new column).
    assert cols.index("entry_start_year") == cols.index("year") + 1
