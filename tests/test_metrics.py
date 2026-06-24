"""Cloze and composition summary statistics."""

import math

import pandas as pd

from evals.metrics import compute_cloze_summary, compute_tron_summary


def test_compute_cloze_summary_recall_leakage_rnl():
    df = pd.DataFrame([
        {"is_future": 0, "correct@10": 1},
        {"is_future": 0, "correct@10": 0},  # past recall = 1/2
        {"is_future": 1, "correct@10": 1},  # future leakage = 1/1
    ])
    summary = compute_cloze_summary(df, cutoff_year=1930, ks=[10])
    assert summary["ks"] == [10]
    assert summary["past_samples"] == 2 and summary["future_samples"] == 1
    m = summary["metrics_by_k"]["10"]   # fixed/ schema: metrics_by_k, string keys
    assert m["recall"] == 0.5
    assert m["leakage"] == 1.0
    assert m["recall_normalized_leakage"] == 2.0
    assert m["past_correct"] == 1 and m["future_correct"] == 1


def test_cloze_summary_rnl_infinite_when_recall_zero():
    df = pd.DataFrame([
        {"is_future": 0, "correct@10": 0},
        {"is_future": 1, "correct@10": 1},
    ])
    m = compute_cloze_summary(df, 1930, [10])["metrics_by_k"]["10"]
    assert m["recall"] == 0.0
    assert math.isinf(m["recall_normalized_leakage"])


def test_compute_tron_summary_shape_and_recall():
    df = pd.DataFrame([
        {"category": "control", "context_level": "high", "rank": 1, "correct@10": 1},
        {"category": "control", "context_level": "high", "rank": -1, "correct@10": 0},
        {"category": "synthetic", "context_level": "low", "rank": 5, "correct@10": 1},
    ])
    out = compute_tron_summary(df, ks=[10])
    assert {"category", "context_level", "n_samples", "n_found",
            "recall@10"}.issubset(out.columns)
    control_high = out[(out["category"] == "control") & (out["context_level"] == "high")]
    assert control_high.iloc[0]["n_samples"] == 2
    assert control_high.iloc[0]["n_found"] == 1
    assert control_high.iloc[0]["recall@10"] == 0.5
