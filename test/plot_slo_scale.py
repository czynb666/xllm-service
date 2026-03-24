"""Plot SLO Scale Sensitivity from single-scale JSON results.

Reads JSON files produced by slo_scale_test.py (one per slo_scale + algorithm),
groups by algorithm_name, and plots:
  1. SLO Scale vs Attainment (TTFT / TPOT / E2E) — one subplot per algorithm
  2. Algorithm comparison — E2E attainment on same axes

Usage:
    python plot_slo_scale.py results/                         # scan directory
    python plot_slo_scale.py slo_scale_1.0_xllm.json slo_scale_2.0_xllm.json
    python plot_slo_scale.py results/ --output-dir plots/
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(paths):
    """Load JSON result files. paths can be files or directories."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, 'slo_scale_*.json'))))
        elif os.path.isfile(p):
            files.append(p)
    # Group by algorithm_name: {algo: {scale: result_dict}}
    grouped = defaultdict(dict)
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        algo = data['config']['algorithm_name']
        scale = data['config']['slo_scale']
        grouped[algo][scale] = data['result']
    return grouped


def plot_per_algorithm(grouped, output_dir):
    """One figure per algorithm: TTFT / TPOT / E2E attainment vs scale."""
    for algo, scale_results in sorted(grouped.items()):
        scales = sorted(scale_results.keys())
        e2e = [scale_results[s]['e2e_attainment'] for s in scales]
        ttft = [scale_results[s]['ttft_attainment'] for s in scales]
        tpot = [scale_results[s]['tpot_attainment'] for s in scales]

        plt.figure(figsize=(8, 5))
        plt.plot(scales, e2e, 's-', lw=2.5, ms=8, color='#2196F3',
                 label='E2E')
        plt.plot(scales, ttft, 'o--', lw=1.5, ms=6, color='#4CAF50',
                 label='TTFT')
        plt.plot(scales, tpot, '^--', lw=1.5, ms=6, color='#FF9800',
                 label='TPOT')
        plt.xlabel('SLO Scale', fontsize=12)
        plt.ylabel('SLO Attainment (%)', fontsize=12)
        plt.title(f'SLO Attainment — {algo}', fontsize=14)
        plt.xticks(scales, [f'{s}x' for s in scales])
        plt.ylim(0, 105)
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()

        path = os.path.join(output_dir, f'slo_attainment_{algo}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def plot_algorithm_comparison(grouped, output_dir):
    """E2E attainment comparison across algorithms on same axes."""
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0',
              '#795548', '#607D8B', '#E91E63']
    markers = ['s', 'o', '^', 'D', 'v', 'P', 'X', 'h']

    plt.figure(figsize=(8, 5))
    for idx, (algo, scale_results) in enumerate(sorted(grouped.items())):
        scales = sorted(scale_results.keys())
        e2e = [scale_results[s]['e2e_attainment'] for s in scales]
        plt.plot(scales, e2e,
                 marker=markers[idx % len(markers)],
                 linestyle='-', lw=2, ms=8,
                 color=colors[idx % len(colors)],
                 label=algo)

    plt.xlabel('SLO Scale', fontsize=12)
    plt.ylabel('E2E SLO Attainment (%)', fontsize=12)
    plt.title('E2E SLO Attainment Comparison', fontsize=14)
    # Use x ticks from the algorithm with most scales
    all_scales = sorted(set(s for sr in grouped.values() for s in sr))
    plt.xticks(all_scales, [f'{s}x' for s in all_scales])
    plt.ylim(0, 105)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, 'slo_e2e_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot SLO scale sensitivity from JSON results')
    parser.add_argument('paths', nargs='+',
                        help='JSON files or directories containing them')
    parser.add_argument('--output-dir', default='.',
                        help='Directory for output plots (default: cwd)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    grouped = load_results(args.paths)
    if not grouped:
        print("No result files found.")
        return

    print(f"Loaded {sum(len(v) for v in grouped.values())} result(s) "
          f"from {len(grouped)} algorithm(s): {sorted(grouped.keys())}")

    plot_per_algorithm(grouped, args.output_dir)
    if len(grouped) > 1:
        plot_algorithm_comparison(grouped, args.output_dir)


if __name__ == "__main__":
    main()
