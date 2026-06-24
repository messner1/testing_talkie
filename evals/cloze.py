"""HPLM cloze leakage eval.

For each dataset row we strip the target word to form a prefix, wrap it with the
spec's prompt strategy, run constrained beam search to get top-k single-word
completions, rank the target, and aggregate recall / leakage / RNL by cutoff year.
"""

import json
import re

import pandas as pd
from tqdm import tqdm

from .metrics import compute_cloze_summary


def extract_prefix(text, target_word):
    """Return the text up to the (last) occurrence of ``target_word``.

    Falls back to a word-boundary regex match, then to the full text if the target
    is not found. Shared by every model (this is tokenizer-independent).
    """
    text_lower = text.lower()
    target_lower = target_word.lower().strip()

    idx = text_lower.rfind(target_lower)
    if idx != -1:
        return text[:idx].rstrip()

    pattern = r'\b' + re.escape(target_lower) + r'\b'
    matches = list(re.finditer(pattern, text_lower))
    if matches:
        return text[:matches[-1].start()].rstrip()

    return text.rstrip()


def run_cloze(backend, spec, dataset, output_prefix, ks, cutoff_year=None):
    """Run the cloze eval and write ``{output_prefix}_details.csv`` + ``_summary.json``.

    Leakage is computed against ``sense_start_year`` (when this *sense* entered use) — the
    ``year`` column and ``is_future``. ``entry_start_year`` (when the headword first
    appeared) is preserved as its own column for downstream use but never drives leakage.

    Args:
        backend: a loaded :class:`~evals.backends.base.EvalBackend`.
        spec: the :class:`~evals.registry.ModelSpec` (for the prompt strategy).
        dataset: iterable of rows with ``text``, ``word``, ``sense_start_year``,
            ``entry_start_year`` keys.
        output_prefix: path prefix for the two output files.
        ks: list of k values for recall@k.
        cutoff_year: overrides ``spec.cutoff_year`` when provided.

    Returns:
        The per-sample results DataFrame.
    """
    cutoff_year = cutoff_year if cutoff_year is not None else spec.cutoff_year
    backend.model.eval()

    print("Building token constraint lists...")
    continuation_ids, word_start_ids = backend.build_constraint_ids()
    print(f"Found {len(word_start_ids)} word-start, {len(continuation_ids)} continuation tokens")
    print(f"Using vocab_size={backend.vocab_size}, eos_token_id={backend.eos_token_id}, "
          f"cutoff_year={cutoff_year}")

    max_k = max(ks)
    results = []

    for i in tqdm(range(len(dataset)), desc="Evaluating cloze"):
        sample = dataset[i]
        text = sample['text']
        target_word = sample['word']
        # Leakage is keyed on sense_start_year; entry_start_year is preserved only.
        sense_year = sample.get('sense_start_year')
        entry_year = sample.get('entry_start_year')

        prefix = extract_prefix(text, target_word)
        prompt = spec.prompt.cloze_prompt(prefix)

        top_words = backend.top_k_words(prompt, continuation_ids, word_start_ids, k=max_k)

        top_word_list = [w for w, _ in top_words]
        top_nlls = [-s for _, s in top_words]
        target_lower = target_word.strip().lower()
        rank = top_word_list.index(target_lower) + 1 if target_lower in top_word_list else -1

        is_future = sense_year is not None and sense_year > cutoff_year
        result = {
            'text': text,
            'target_word': target_word,
            'year': sense_year,
            'entry_start_year': entry_year,
            'rank': rank,
            'is_future': int(is_future),
            'top_10_words': '|'.join(top_word_list[:10]),
            'top_10_nlls': '|'.join(f"{n:.4f}" for n in top_nlls[:10]),
        }
        for k in ks:
            result[f'correct@{k}'] = int(rank != -1 and rank <= k)
        results.append(result)

    df = pd.DataFrame(results)
    summary = compute_cloze_summary(df, cutoff_year, ks)

    df.to_csv(f"{output_prefix}_details.csv", index=False)
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_prefix}_details.csv and {output_prefix}_summary.json")
    return df
