#!/usr/bin/env python3
"""
Create temporal performance plot with sample distribution overlay.
Automatically finds optimal bin size that naturally divides the data range and aligns with cutoff.

Usage:
    python plot_temporal_with_support.py \
        --base results_base_fixed.csv \
        --it results_it_fixed.csv \
        --web results_web_fixed.csv \
        --cutoff 1930 \
        --bin-size 50 \
        --metric recall \
        --k 100 \
        --output figures/temporal_with_support.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def adjusted_mrr(ranks, k=100):
    """
    Compute MRR with not-found samples assigned rank k+1.
    This penalizes models that fail to find the target.
    """
    if len(ranks) == 0:
        return np.nan
    adjusted = [r if r > 0 else (k + 1) for r in ranks]
    return np.mean([1.0 / r for r in adjusted])


def find_divisors(n):
    """Find all divisors of n."""
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def find_optimal_binning(min_year, max_year, cutoff_year, requested_bin_size, tolerance=10):
    """
    Find optimal bin size and start year that:
    1. Makes (cutoff_year - start_year) evenly divisible by bin_size
    2. Bin_size is close to requested_bin_size
    3. Covers the full data range
    
    Returns: (bin_size, start_year, bin_edges)
    """
    best_solution = None
    best_diff = float('inf')
    
    # Try different start years (decades and centuries)
    for start_year in range((min_year // 100) * 100, min_year + 50, 10):
        span_to_cutoff = cutoff_year - start_year
        
        if span_to_cutoff <= 0:
            continue
        
        # Find all divisors of span_to_cutoff
        divisors = find_divisors(span_to_cutoff)
        
        # Filter divisors to those in reasonable range
        valid_divisors = [d for d in divisors if 10 <= d <= 200]
        
        for bin_size in valid_divisors:
            diff = abs(bin_size - requested_bin_size)
            
            if diff <= tolerance and diff < best_diff:
                # Calculate total bins needed to cover data range
                total_span = max_year - start_year
                n_bins_total = int(np.ceil(total_span / bin_size))
                
                # Verify cutoff alignment
                n_bins_pre = span_to_cutoff // bin_size
                cutoff_check = start_year + n_bins_pre * bin_size
                
                if cutoff_check == cutoff_year:
                    best_solution = {
                        'bin_size': bin_size,
                        'start_year': start_year,
                        'n_bins_pre': n_bins_pre,
                        'n_bins_total': n_bins_total
                    }
                    best_diff = diff
    
    if best_solution is None:
        raise ValueError(f"Could not find valid binning within tolerance {tolerance} years of {requested_bin_size}")
    
    bin_size = best_solution['bin_size']
    start_year = best_solution['start_year']
    n_bins_total = best_solution['n_bins_total']
    n_bins_pre = best_solution['n_bins_pre']
    
    # Generate bin edges
    bin_edges = [start_year + i * bin_size for i in range(n_bins_total + 1)]
    
    # Verify cutoff alignment
    cutoff_bin = start_year + n_bins_pre * bin_size
    
    print(f"\n  Bin size optimization:")
    print(f"    Requested: {requested_bin_size} years")
    print(f"    Selected:  {bin_size} years")
    print(f"    Start year: {start_year}")
    print(f"    Span to cutoff: {cutoff_year - start_year} years")
    print(f"    Bins before cutoff: {n_bins_pre}")
    print(f"    Total bins: {n_bins_total}")
    print(f"    Cutoff boundary: {start_year} + {n_bins_pre} × {bin_size} = {cutoff_bin}")
    print(f"    Cutoff requested: {cutoff_year}")
    print(f"    Alignment check: {'✓ ALIGNED' if cutoff_bin == cutoff_year else '✗ NOT ALIGNED'}")
    
    return bin_size, start_year, bin_edges


def compute_metrics_by_bin(df, bin_edges, k=100):
    """
    Compute Recall@k and Adjusted MRR@k for each year bin.
    Bins are defined by bin_edges.
    
    Returns DataFrame with metrics per bin.
    """
    results = []
    
    # Process dated samples only
    df_valid = df[df['year'].notna()].copy()
    
    # Assign each sample to a bin
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Samples in this bin [bin_start, bin_end)
        mask = (df_valid['year'] >= bin_start) & (df_valid['year'] < bin_end)
        group = df_valid[mask]
        
        if len(group) == 0:
            continue
        
        recall = group[f'correct@{k}'].mean() if f'correct@{k}' in group.columns else np.nan
        adj_mrr = adjusted_mrr(group['rank'].tolist(), k=k)
        n_samples = len(group)
        std_err = group[f'correct@{k}'].std() / np.sqrt(n_samples) if n_samples > 0 else 0
        
        results.append({
            'year_bin': bin_start,
            'bin_end': bin_end,
            'recall': recall,
            'adj_mrr': adj_mrr,
            'n_samples': n_samples,
            'std_err': std_err
        })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('year_bin')
    return df_result


def plot_temporal_with_support(metrics_dict, cutoff_year, bin_size, 
                               metric_col, k, output_path, min_samples=5,
                               min_year=None):
    """
    Create temporal plot with three overlaid information streams.
    """
    # Set up figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    # Model colors and markers
    colors = {'Base': '#1f77b4', 'IT': '#ff7f0e', 'Web': '#2ca02c'}
    markers = {'Base': 'o', 'IT': 's', 'Web': '^'}
    
    # Metric label
    metric_labels = {
        'recall': f'Recall@{k}',
        'adj_mrr': f'Adjusted MRR@{k}'
    }
    metric_label = metric_labels.get(metric_col, metric_col)
    
    # --- LAYER 1: Background sample distribution bars ---
    # Use first model's distribution (should be same for all)
    first_model = list(metrics_dict.keys())[0]
    df_samples = metrics_dict[first_model]
    
    # Create second y-axis for sample counts
    ax2 = ax1.twinx()
    
    # Plot bars - each bar spans from year_bin to bin_end
    for _, row in df_samples.iterrows():
        bar_start = row['year_bin']
        bar_width = row['bin_end'] - row['year_bin']
        
        ax2.bar(
            bar_start, 
            row['n_samples'], 
            width=bar_width * 0.95,  # 95% to leave small gap
            align='edge',
            color='lightsteelblue',
            alpha=0.25,
            edgecolor='lightsteelblue',
            linewidth=0.5,
            zorder=0
        )
    
    # Configure secondary axis (sample counts)
    ax2.set_ylabel(
        'Number of Samples per Bin', 
        fontsize=13, 
        fontweight='bold', 
        color='gray'
    )
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=10)
    ax2.set_ylim(0, df_samples['n_samples'].max() * 1.2)
    ax2.grid(False)  # No grid for secondary axis
    
    # --- LAYER 2: Cutoff line and shading ---
    ax1.axvline(
        x=cutoff_year, 
        color='red', 
        linestyle='--', 
        linewidth=2.5, 
        alpha=0.8, 
        zorder=1
    )
    
  
    # Shade post-cutoff region (very subtle)
    max_year = df_samples['bin_end'].max()
    ax1.axvspan(cutoff_year, max_year, alpha=0.08, color='red', zorder=0)
    
    # --- LAYER 3: Model performance lines with error bars ---
    all_years = []
    
    for model_name, metrics_df in metrics_dict.items():
        # Filter by minimum sample size
        df_plot = metrics_df[metrics_df['n_samples'] >= min_samples].copy()
        
        if len(df_plot) == 0:
            print(f"Warning: No bins with >= {min_samples} samples for {model_name}")
            continue
        
        # Use bin midpoints for x-coordinates
        x_vals = (df_plot['year_bin'].values + df_plot['bin_end'].values) / 2
        y_vals = df_plot[metric_col].values
        err_vals = df_plot['std_err'].values * 1.96  # 95% confidence interval
        
        all_years.extend(df_plot['year_bin'].values)
        
        # Plot line with error bars
        ax1.errorbar(
            x_vals, 
            y_vals,
            yerr=err_vals,
            marker=markers[model_name],
            color=colors[model_name],
            label=f'{model_name}',
            linewidth=2.5,
            markersize=10,
            capsize=5,
            capthick=1.5,
            alpha=0.9,
            zorder=2
        )
    
    # --- Configure x-axis limits ---
    if min_year is None:
        min_year = min(all_years)
    
    ax1.set_xlim(min_year - bin_size * 0.5, max_year)
    
    # --- Configure primary axis (performance metrics) ---
    ax1.set_xlabel(
        f'Year ({bin_size}-year bins, aligned to cutoff)', 
        fontsize=14, 
        fontweight='bold'
    )
    ax1.set_ylabel(
        metric_label, 
        fontsize=14, 
        fontweight='bold'
    )
    ax1.set_title(
        f'{metric_label} by Time Period', 
        fontsize=16, 
        fontweight='bold', 
        pad=20
    )
    ax1.set_ylim(0, min(1.0, ax1.get_ylim()[1] * 1.1))
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.grid(True, alpha=0.3, zorder=0)
    
    # --- Legend (only model lines) ---
    ax1.legend(
        loc='upper left', 
        fontsize=12, 
        framealpha=0.95
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create temporal performance plot with sample distribution overlay.'
    )
    parser.add_argument('--base', type=str, required=True,
                        help='Path to Base model fixed CSV')
    parser.add_argument('--it', type=str, required=True,
                        help='Path to IT model fixed CSV')
    parser.add_argument('--web', type=str, required=True,
                        help='Path to Web model fixed CSV')
    parser.add_argument('--cutoff', type=int, default=1930,
                        help='Cutoff year (default: 1930)')
    parser.add_argument('--bin-size', type=int, default=50,
                        help='Requested bin size in years (will auto-adjust for alignment, default: 50)')
    parser.add_argument('--metric', type=str, default='recall',
                        choices=['recall', 'adj_mrr'],
                        help='Metric to plot (default: recall)')
    parser.add_argument('--k', type=int, default=100,
                        help='k value for recall@k (default: 100)')
    parser.add_argument('--output', type=str, default='temporal_with_support.png',
                        help='Output path (default: temporal_with_support.png)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Minimum samples per bin to plot (default: 5)')
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum year to display (default: auto)')
    parser.add_argument('--tolerance', type=int, default=10,
                        help='Maximum deviation from requested bin size (default: 10 years)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    if output_dir != Path('.') and output_dir != Path(''):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSVs
    print("Loading data...")
    df_base = pd.read_csv(args.base)
    df_it = pd.read_csv(args.it)
    df_web = pd.read_csv(args.web)
    
    print(f"  Base: {len(df_base)} samples")
    print(f"  IT:   {len(df_it)} samples")
    print(f"  Web:  {len(df_web)} samples")
    
    # Find temporal range across all models
    all_years = pd.concat([
        df_base[df_base['year'].notna()]['year'],
        df_it[df_it['year'].notna()]['year'],
        df_web[df_web['year'].notna()]['year']
    ])
    
    min_data_year = int(all_years.min())
    max_data_year = int(all_years.max())
    
    print(f"\n  Data temporal range: {min_data_year} - {max_data_year}")
    
    # Find optimal binning
    bin_size, start_year, bin_edges = find_optimal_binning(
        min_data_year, 
        max_data_year, 
        args.cutoff, 
        args.bin_size,
        tolerance=args.tolerance
    )
    
    # Compute metrics by bin
    print(f"\nComputing metrics...")
    metrics_base = compute_metrics_by_bin(df_base, bin_edges, k=args.k)
    metrics_it = compute_metrics_by_bin(df_it, bin_edges, k=args.k)
    metrics_web = compute_metrics_by_bin(df_web, bin_edges, k=args.k)
    
    print(f"  Base: {len(metrics_base)} bins")
    print(f"  IT:   {len(metrics_it)} bins")
    print(f"  Web:  {len(metrics_web)} bins")
    
    # Verify and display bins around cutoff
    print(f"\n  Bin boundaries around cutoff ({args.cutoff}):")
    for _, row in metrics_base.iterrows():
        bin_start = int(row['year_bin'])
        bin_end = int(row['bin_end'])
        
        if abs(bin_start - args.cutoff) <= bin_size * 2 or abs(bin_end - args.cutoff) <= bin_size * 2:
            pre_post = "PRE " if bin_end <= args.cutoff else "POST" if bin_start >= args.cutoff else "SPAN"
            is_boundary = (bin_end == args.cutoff) or (bin_start == args.cutoff)
            boundary_marker = " ← CUTOFF BOUNDARY" if is_boundary else ""
            print(f"    {pre_post}: [{bin_start}, {bin_end}){boundary_marker}")
    
    metrics_dict = {
        'Base': metrics_base,
        'IT': metrics_it,
        'Web': metrics_web
    }
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_temporal_with_support(
        metrics_dict,
        cutoff_year=args.cutoff,
        bin_size=bin_size,
        metric_col=args.metric,
        k=args.k,
        output_path=args.output,
        min_samples=args.min_samples,
        min_year=args.min_year
    )
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for model_name, metrics_df in metrics_dict.items():
        pre = metrics_df[metrics_df['bin_end'] <= args.cutoff]
        post = metrics_df[metrics_df['year_bin'] >= args.cutoff]
        
        # Weighted means
        if len(pre) > 0:
            pre_mean = (pre[args.metric] * pre['n_samples']).sum() / pre['n_samples'].sum()
        else:
            pre_mean = np.nan
            
        if len(post) > 0:
            post_mean = (post[args.metric] * post['n_samples']).sum() / post['n_samples'].sum()
        else:
            post_mean = np.nan
        
        print(f"\n{model_name} Model:")
        print(f"  Pre-{args.cutoff}:  {pre_mean:.4f} (n={int(pre['n_samples'].sum()) if len(pre) > 0 else 0})")
        print(f"  Post-{args.cutoff}: {post_mean:.4f} (n={int(post['n_samples'].sum()) if len(post) > 0 else 0})")
        
        if not np.isnan(pre_mean) and not np.isnan(post_mean):
            ratio = post_mean / pre_mean
            print(f"  Post/Pre ratio: {ratio:.3f}")
    
    print("\n" + "="*80)
    print("✅ Done!")


if __name__ == '__main__':
    main()
