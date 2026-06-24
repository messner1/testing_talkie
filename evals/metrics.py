"""Summary-statistic computation for the cloze and composition evals.

Lifted from the legacy ``eval_leakage_tt_and_hf.py`` (recall/leakage/RNL) and
``eval_tron.py`` (category x context-level recall) so both harnesses share one copy.
"""

import pandas as pd


def compute_cloze_summary(df, cutoff_year, ks):
    """Compute recall / leakage / recall-normalised-leakage from cloze results.

    Args:
        df: per-sample DataFrame with ``is_future`` (0/1) and ``correct@k`` columns.
        cutoff_year: the model's knowledge cutoff (recorded in the summary).
        ks: list of k values.

    Returns:
        A JSON-serialisable summary dict (the schema produced by the legacy
        ``recalculate_summary.py`` / ``fixed/`` outputs). ``recall_normalized_leakage`` is
        the leakage (future-sample recall) divided by in-cutoff recall, or ``inf`` when
        recall is 0.
    """
    past = df[df['is_future'] == 0]
    future = df[df['is_future'] == 1]

    summary = {
        'cutoff_year': cutoff_year,
        'ks': list(ks),
        'total_samples': len(df),
        'past_samples': len(past),
        'future_samples': len(future),
        'metrics_by_k': {},
    }

    for k in ks:
        col = f'correct@{k}'
        past_correct = int(past[col].sum())
        future_correct = int(future[col].sum())
        recall = past_correct / len(past) if len(past) > 0 else 0.0
        leakage = future_correct / len(future) if len(future) > 0 else 0.0
        rnl = leakage / recall if recall > 0 else float('inf')
        overall = df[col].sum() / len(df) if len(df) > 0 else 0.0

        summary['metrics_by_k'][str(k)] = {
            'overall_accuracy': overall,
            'recall': recall,
            'leakage': leakage,
            'recall_normalized_leakage': rnl,
            'past_correct': past_correct,
            'future_correct': future_correct,
        }

    return summary


def _subset_summary(subset, category, context_level, ks):
    """Build one summary row for a (category, context_level) slice."""
    found = subset['rank'] > 0
    row = {
        'category': category,
        'context_level': context_level,
        'n_samples': len(subset),
        'n_found': int(found.sum()),
        'pct_found': 100 * found.sum() / len(subset) if len(subset) else 0.0,
        'mean_rank': subset[found]['rank'].mean() if found.any() else float('inf'),
        'median_rank': subset[found]['rank'].median() if found.any() else float('inf'),
    }
    for k in ks:
        row[f'recall@{k}'] = subset[f'correct@{k}'].mean() if len(subset) else 0.0
    return row


def compute_tron_summary(df, ks):
    """Compute composition recall by category x context-level (plus a per-category ALL row).

    Returns a DataFrame matching the legacy ``{prefix}_summary.csv`` schema.
    """
    categories = sorted(df['category'].unique())
    context_levels = ['high', 'medium', 'low']

    rows = []
    for category in categories:
        for context_level in context_levels:
            subset = df[(df['category'] == category) & (df['context_level'] == context_level)]
            if len(subset) == 0:
                continue
            rows.append(_subset_summary(subset, category, context_level, ks))

    return pd.DataFrame(rows)
