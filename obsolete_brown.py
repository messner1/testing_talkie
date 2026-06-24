#!/usr/bin/env python3
"""
Analyze model Recall@k performance on data subsets (obsolete senses + non-Brown words).

Current model set: Talkie-Base, Talkie-Web (cutoff 1930) and Typewriter (cutoff 1913).
Each model carries its OWN cutoff, because the two cutoff-dependent subsets —
  * obsolete senses   (sense_end_year   < cutoff)
  * non-Brown temporal split (sense_start_year >= cutoff)
are defined relative to the model's knowledge boundary. Typewriter's 1913 cutoff is
therefore honoured separately from the Talkie models' 1930, and every chart annotates
the per-model cutoff so the bars are not misread as a single shared experiment. The
Brown / non-Brown membership split itself is cutoff-independent (Brown-corpus vocabulary
only), so that contrast is directly comparable across all three.

Usage:
    # All three current models (Base+Web at 1930, Typewriter at 1913)
    python obsolete_brown.py \
        --base results/cloze_talkie-base_details.csv \
        --web  results/cloze_talkie-web_details.csv \
        --typewriter results/cloze_typewriter_details.csv \
        --cutoff 1930 --typewriter-cutoff 1913 \
        --output-dir figures/subsets

    # Talkie models only
    python obsolete_brown.py --base ... --web ... --cutoff 1930
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
import nltk
from nltk.corpus import brown

# Display label -> bar colour. Typewriter replaces the retired IT model.
COLORS = {'Base': '#1f77b4', 'Web': '#2ca02c', 'Typewriter': '#9467bd'}


def get_brown_vocabulary():
    """Get Brown corpus vocabulary."""
    try:
        brown.words()
    except LookupError:
        nltk.download('brown')
    return set(word.lower() for word in brown.words())


def align_results(df_results, dataset):
    """Align results with dataset using 'text' field."""
    ds_df = pd.DataFrame(dataset)
    text_map = {str(row['text']): idx for idx, row in ds_df.iterrows()}
    df_results = df_results.copy()
    df_results['dataset_idx'] = df_results['text'].apply(lambda x: text_map.get(str(x)))
    print(f"    Aligned {df_results['dataset_idx'].notna().sum()}/{len(df_results)} samples")
    return df_results


def compute_metrics(df, k):
    """Compute recall@k for a subset."""
    if len(df) == 0:
        return {'n_samples': 0, 'recall': np.nan}
    return {'n_samples': len(df), 'recall': df[f'correct@{k}'].mean()}


def compute_contribution(n_subset, recall_subset, n_total):
    """Compute contribution to overall performance."""
    return (n_subset * recall_subset) / n_total if n_total > 0 else 0.0


def save_contribution_summary(results, k_values, output_path, subset_names, model_cutoffs):
    """Save contribution summary CSV (one row per model x k)."""
    rows = []
    subset_a, subset_b = subset_names

    for model in results.keys():
        for k in k_values:
            a_contrib = results[model][subset_a][k]['contrib_recall']
            b_contrib = results[model][subset_b][k]['contrib_recall']
            total_recall = a_contrib + b_contrib

            n_a = results[model][subset_a][k]['n_samples']
            n_b = results[model][subset_b][k]['n_samples']
            n_total = n_a + n_b

            rows.append({
                'model': model,
                'cutoff': model_cutoffs[model],
                'k': k,
                f'{subset_a}_n': n_a,
                f'{subset_b}_n': n_b,
                f'{subset_a}_pct_samples': 100 * n_a / n_total if n_total else 0,
                f'{subset_b}_pct_samples': 100 * n_b / n_total if n_total else 0,
                f'{subset_a}_recall': results[model][subset_a][k]['recall'],
                f'{subset_b}_recall': results[model][subset_b][k]['recall'],
                f'{subset_a}_contrib_pct': 100 * a_contrib / total_recall if total_recall > 0 else 0,
                f'{subset_b}_contrib_pct': 100 * b_contrib / total_recall if total_recall > 0 else 0,
                'overall_recall': total_recall,
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  ✅ Saved: {output_path}")


def analyze_obsolete_senses(results_dict, dataset, model_cutoffs, k_values, output_dir):
    """Analyze obsolete senses (sense_end_year < cutoff), per model cutoff."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: OBSOLETE SENSES")
    print("=" * 80)

    ds_df = pd.DataFrame(dataset)

    results = {}
    obsolete_idx_by_model = {}
    csv_dir = output_dir / 'obsolete_senses'
    csv_dir.mkdir(parents=True, exist_ok=True)

    for model_name, df_results in results_dict.items():
        cutoff = model_cutoffs[model_name]
        # Cutoff-dependent subset, recomputed per model so Typewriter (1913) and
        # the Talkie models (1930) each get their own obsolete pool.
        obsolete_mask = ds_df['sense_end_year'].notna() & (ds_df['sense_end_year'] < cutoff)
        obsolete_indices = set(ds_df[obsolete_mask].index.tolist())
        extant_indices = set(ds_df[~obsolete_mask].index.tolist())
        obsolete_idx_by_model[model_name] = obsolete_indices

        results[model_name] = {'obsolete': {}, 'extant': {}}
        print(f"\n{model_name} Model (cutoff {cutoff}):")
        print(f"  Obsolete: {len(obsolete_indices)} ({100*len(obsolete_indices)/len(ds_df):.1f}%)"
              f" | Extant: {len(extant_indices)}")

        df_aligned = align_results(df_results, dataset)
        n_total = len(df_aligned)

        df_obsolete = df_aligned[df_aligned['dataset_idx'].isin(obsolete_indices)]
        df_extant = df_aligned[df_aligned['dataset_idx'].isin(extant_indices)]

        df_obsolete.drop(columns=['dataset_idx']).to_csv(
            csv_dir / f'{model_name.lower()}_obsolete_senses.csv', index=False)
        df_extant.drop(columns=['dataset_idx']).to_csv(
            csv_dir / f'{model_name.lower()}_extant_senses.csv', index=False)

        for k in k_values:
            obs_metrics = compute_metrics(df_obsolete, k)
            ext_metrics = compute_metrics(df_extant, k)
            results[model_name]['obsolete'][k] = {
                **obs_metrics,
                'contrib_recall': compute_contribution(obs_metrics['n_samples'], obs_metrics['recall'], n_total)
            }
            results[model_name]['extant'][k] = {
                **ext_metrics,
                'contrib_recall': compute_contribution(ext_metrics['n_samples'], ext_metrics['recall'], n_total)
            }

    save_contribution_summary(results, k_values,
                              output_dir / 'obsolete_senses_contribution_summary.csv',
                              ('obsolete', 'extant'), model_cutoffs)
    return results, obsolete_idx_by_model


def analyze_non_brown_words(results_dict, dataset, model_cutoffs, k_values, output_dir,
                            obsolete_idx_by_model):
    """Analyze non-Brown words with temporal breakdown and overlap (per model cutoff)."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: NON-BROWN CORPUS WORDS")
    print("=" * 80)

    ds_df = pd.DataFrame(dataset)
    brown_vocab = get_brown_vocabulary()
    print(f"Brown vocab: {len(brown_vocab)} words")

    # Brown membership is cutoff-independent -> shared across models.
    ds_df['word_lower'] = ds_df['word'].str.lower()
    non_brown_mask = ~ds_df['word_lower'].isin(brown_vocab)
    non_brown_indices = set(ds_df[non_brown_mask].index.tolist())
    brown_indices = set(ds_df[~non_brown_mask].index.tolist())
    print(f"Non-Brown: {len(non_brown_indices)} ({100*len(non_brown_indices)/len(ds_df):.1f}%)"
          f" | Brown: {len(brown_indices)}")

    results = {}
    overlap_rows = []
    csv_dir = output_dir / 'brown_corpus'
    csv_dir.mkdir(parents=True, exist_ok=True)

    for model_name, df_results in results_dict.items():
        cutoff = model_cutoffs[model_name]

        # Temporal split is cutoff-dependent -> per model.
        post_mask = ds_df['sense_start_year'].notna() & (ds_df['sense_start_year'] >= cutoff)
        non_brown_post = non_brown_indices & set(ds_df[post_mask].index.tolist())
        non_brown_pre = non_brown_indices - non_brown_post

        overlap = non_brown_indices & obsolete_idx_by_model[model_name]
        overlap_rows.append({
            'model': model_name, 'cutoff': cutoff,
            'total_samples': len(ds_df),
            'non_brown': len(non_brown_indices), 'brown': len(brown_indices),
            'obsolete': len(obsolete_idx_by_model[model_name]),
            f'non_brown_post_cutoff': len(non_brown_post),
            f'non_brown_pre_cutoff': len(non_brown_pre),
            'overlap_non_brown_obsolete': len(overlap),
            'pct_non_brown_obsolete': 100 * len(overlap) / len(non_brown_indices) if non_brown_indices else 0,
        })

        results[model_name] = {
            'non_brown': {}, 'brown': {},
            'non_brown_post': {}, 'non_brown_pre': {}
        }
        print(f"\n{model_name} Model (cutoff {cutoff}):")
        print(f"  Non-Brown post-{cutoff}: {len(non_brown_post)} | pre-{cutoff}: {len(non_brown_pre)}")

        df_aligned = align_results(df_results, dataset)
        n_total = len(df_aligned)

        df_non_brown = df_aligned[df_aligned['dataset_idx'].isin(non_brown_indices)]
        df_brown = df_aligned[df_aligned['dataset_idx'].isin(brown_indices)]
        df_nb_post = df_aligned[df_aligned['dataset_idx'].isin(non_brown_post)]
        df_nb_pre = df_aligned[df_aligned['dataset_idx'].isin(non_brown_pre)]

        for df, name in [(df_non_brown, 'non_brown'), (df_brown, 'brown'),
                         (df_nb_post, f'non_brown_post_{cutoff}'),
                         (df_nb_pre, f'non_brown_pre_{cutoff}')]:
            df.drop(columns=['dataset_idx']).to_csv(
                csv_dir / f'{model_name.lower()}_{name}.csv', index=False)

        n_non_brown = len(df_non_brown)

        for k in k_values:
            nb_metrics = compute_metrics(df_non_brown, k)
            br_metrics = compute_metrics(df_brown, k)
            nb_post_metrics = compute_metrics(df_nb_post, k)
            nb_pre_metrics = compute_metrics(df_nb_pre, k)

            results[model_name]['non_brown'][k] = {
                **nb_metrics,
                'contrib_recall': compute_contribution(nb_metrics['n_samples'], nb_metrics['recall'], n_total)
            }
            results[model_name]['brown'][k] = {
                **br_metrics,
                'contrib_recall': compute_contribution(br_metrics['n_samples'], br_metrics['recall'], n_total)
            }
            results[model_name]['non_brown_post'][k] = {
                **nb_post_metrics,
                'contrib_recall': compute_contribution(nb_post_metrics['n_samples'], nb_post_metrics['recall'], n_non_brown)
            }
            results[model_name]['non_brown_pre'][k] = {
                **nb_pre_metrics,
                'contrib_recall': compute_contribution(nb_pre_metrics['n_samples'], nb_pre_metrics['recall'], n_non_brown)
            }

    pd.DataFrame(overlap_rows).to_csv(output_dir / 'subset_overlap_statistics.csv', index=False)

    save_contribution_summary(results, k_values,
                              output_dir / 'brown_corpus_contribution_summary.csv',
                              ('non_brown', 'brown'), model_cutoffs)
    save_contribution_summary(results, k_values,
                              output_dir / 'non_brown_temporal_contribution_summary.csv',
                              ('non_brown_pre', 'non_brown_post'), model_cutoffs)
    return results


def plot_stacked_bars(results, k_values, output_path, subset_names, title,
                      model_cutoffs, ylabel="Recall@k"):
    """Plot stacked bar chart (per-model cutoffs annotated)."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    models = list(results.keys())
    subset_a, subset_b = subset_names

    bar_width = 0.25
    x = np.arange(len(k_values))

    for i, model in enumerate(models):
        a_contribs = [results[model][subset_a][k]['contrib_recall'] for k in k_values]
        b_contribs = [results[model][subset_b][k]['contrib_recall'] for k in k_values]
        offset = (i - 1) * bar_width if len(models) == 3 else (i - 0.5) * bar_width

        ax.bar(x + offset, a_contribs, bar_width, color=COLORS[model],
               alpha=0.55, edgecolor='black', linewidth=0.5,
               label=f'{model} ({subset_a.replace("_", " ").title()})')
        ax.bar(x + offset, b_contribs, bar_width, bottom=a_contribs, color=COLORS[model],
               alpha=1.0, edgecolor='black', linewidth=0.5,
               label=f'{model} ({subset_b.replace("_", " ").title()})')

    ax.set_xlabel('k (top-k predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(fontsize=8, loc='upper left', ncol=len(models))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Per-model sample-count annotation (subset sizes differ when cutoffs differ).
    lines = []
    for model in models:
        n_a = results[model][subset_a][k_values[0]]['n_samples']
        n_b = results[model][subset_b][k_values[0]]['n_samples']
        n_total = n_a + n_b
        lines.append(f"{model} (≤{model_cutoffs[model]}): "
                     f"{subset_a.replace('_', ' ')} n={n_a} ({100*n_a/n_total:.1f}%)")
    ax.text(0.98, 0.74, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Recall@k on data subsets')
    parser.add_argument('--base', required=True, help='Base model details CSV')
    parser.add_argument('--web', required=False, default=None, help='Web model details CSV (optional)')
    parser.add_argument('--typewriter', required=False, default=None,
                        help='Typewriter model details CSV (optional)')
    parser.add_argument('--cutoff', type=int, default=1930,
                        help='Cutoff for the Talkie models Base/Web (default: 1930)')
    parser.add_argument('--typewriter-cutoff', type=int, default=1913,
                        help='Cutoff for Typewriter (default: 1913)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 50, 100])
    parser.add_argument('--output-dir', default='figures/subsets', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset: Hplm/historical-cloze")
    dataset = load_dataset('Hplm/historical-cloze', split='test')
    print(f"Loaded {len(dataset)} samples")

    print("\nLoading model results...")
    results_dict = {}
    model_cutoffs = {}

    results_dict['Base'] = pd.read_csv(args.base)
    model_cutoffs['Base'] = args.cutoff
    print(f"  Base: {len(results_dict['Base'])} samples (cutoff {args.cutoff})")

    if args.web is not None:
        results_dict['Web'] = pd.read_csv(args.web)
        model_cutoffs['Web'] = args.cutoff
        print(f"  Web:  {len(results_dict['Web'])} samples (cutoff {args.cutoff})")

    if args.typewriter is not None:
        results_dict['Typewriter'] = pd.read_csv(args.typewriter)
        model_cutoffs['Typewriter'] = args.typewriter_cutoff
        print(f"  Typewriter: {len(results_dict['Typewriter'])} samples (cutoff {args.typewriter_cutoff})")

    # Analysis 1: Obsolete senses
    obsolete_results, obsolete_idx_by_model = analyze_obsolete_senses(
        results_dict, dataset, model_cutoffs, args.k_values, output_dir)

    plot_stacked_bars(obsolete_results, args.k_values,
                      output_dir / 'obsolete_senses_comparison.png',
                      ('obsolete', 'extant'),
                      'Contribution to Recall@k: Obsolete vs Extant Senses',
                      model_cutoffs)

    # Analysis 2: Non-Brown words
    brown_results = analyze_non_brown_words(
        results_dict, dataset, model_cutoffs, args.k_values, output_dir, obsolete_idx_by_model)

    plot_stacked_bars(brown_results, args.k_values,
                      output_dir / 'brown_corpus_comparison.png',
                      ('non_brown', 'brown'),
                      'Contribution to Recall@k: Non-Brown vs Brown Words',
                      model_cutoffs)

    plot_stacked_bars(brown_results, args.k_values,
                      output_dir / 'non_brown_temporal_comparison.png',
                      ('non_brown_pre', 'non_brown_post'),
                      'Non-Brown Temporal Split: Contribution to Recall@k',
                      model_cutoffs,
                      ylabel='Contribution to Recall@k (within Non-Brown)')

    print("\n" + "=" * 80)
    print("✅ Complete! Outputs in:", output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
