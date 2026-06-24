"""Synthetic "-tron" composition eval.

Reads a JSONL of test cases (word, category, year, contexts{high,medium,low} each
with a ``[MASK]``), runs the same constrained beam search as the cloze eval on the
prefix before ``[MASK]``, ranks the target word, and summarises recall by
category x context-level.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .metrics import compute_tron_summary


def load_test_cases(jsonl_path):
    """Load composition test cases from a JSONL file into an ordered dict."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Test cases file not found: {jsonl_path}")

    test_cases = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            test_cases[data['word']] = {
                'category': data['category'],
                'year': data.get('year'),
                'contexts': data['contexts'],
            }
    print(f"Loaded {len(test_cases)} test cases from {jsonl_path}")
    return test_cases


def run_composition(backend, spec, test_cases_path, output_prefix, ks):
    """Run composition eval; write ``{output_prefix}_detailed.csv`` + ``_summary.csv``.

    Args:
        backend: a loaded :class:`~evals.backends.base.EvalBackend`.
        spec: the :class:`~evals.registry.ModelSpec` (for prompt + first-token policy).
        test_cases_path: path to the test-case JSONL.
        output_prefix: path prefix for the two output files.
        ks: list of k values for recall@k.

    Returns:
        The per-example results DataFrame.
    """
    backend.model.eval()
    test_cases = load_test_cases(test_cases_path)

    print("Building token constraint lists...")
    continuation_ids, word_start_ids = backend.build_constraint_ids()
    print(f"Found {len(word_start_ids)} word-start, {len(continuation_ids)} continuation tokens")

    max_k = max(ks)
    results = []

    for word, info in tqdm(test_cases.items(), desc="Evaluating composition"):
        for context_level, context_text in info['contexts'].items():
            if '[MASK]' not in context_text:
                print(f"Warning: no [MASK] for {word} ({context_level})")
                continue

            parts = context_text.split('[MASK]')
            prefix = parts[0].rstrip()
            suffix = parts[1] if len(parts) > 1 else ""
            prompt = spec.prompt.cloze_prompt(prefix)

            top_words = backend.top_k_words(prompt, continuation_ids, word_start_ids, k=max_k)

            top_word_list = [w for w, _ in top_words]
            top_nlls = [-s for _, s in top_words]
            target_lower = word.strip().lower()
            rank = (top_word_list.index(target_lower) + 1
                    if target_lower in top_word_list else -1)

            row = {
                'target_word': word,
                'category': info['category'],
                'year': info['year'],
                'context_level': context_level,
                'text': context_text,
                'prefix': prefix,
                'suffix': suffix,
                'rank': rank,
                'target_nll': top_nlls[rank - 1] if rank > 0 else float('inf'),
            }
            for k in ks:
                row[f'correct@{k}'] = int(0 < rank <= k)
            row['top_10_words'] = '|'.join(top_word_list[:10])
            row['top_10_nlls'] = '|'.join(
                f"{n:.4f}" if n != float('inf') else 'inf' for n in top_nlls[:10])
            results.append(row)

    df = pd.DataFrame(results)
    summary_df = compute_tron_summary(df, ks)

    df.to_csv(f"{output_prefix}_detailed.csv", index=False)
    summary_df.to_csv(f"{output_prefix}_summary.csv", index=False)

    print(f"\nResults saved to {output_prefix}_detailed.csv and {output_prefix}_summary.csv")
    print(summary_df.to_string(index=False))
    return df
