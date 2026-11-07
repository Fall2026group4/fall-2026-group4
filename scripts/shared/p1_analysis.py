#!/usr/bin/env python3
"""Baseline-vs-P1 tables and figure from results/openwebtext/p1_sparsity_control_pilot/validation_evaluation_long.csv.

Run from repository root:  python scripts/shared/p1_analysis.py

Adds a dominance reading for L0-unmatched pairs (added after the run; it uses no tuning and is reported
alongside the pre-registered verdict, never instead of it):
  * 'better at lower-or-equal L0' : P1 uses no more latents and the metric is better -> improvement despite mismatch
  * 'worse at higher-or-equal L0' : P1 uses no fewer latents and the metric is worse  -> regression despite mismatch
  * 'trade-off (needs L0 curve)'  : fewer latents and worse metric, or more latents and better metric
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
P1 = REPO / 'results/openwebtext/p1_sparsity_control_pilot'
COLOR = {'relu': '#2a78d6', 'topk': '#eb6834', 'jumprelu': '#1baf7a'}
MARKER = {'relu': 'o', 'topk': 's', 'jumprelu': '^'}
LABEL = {'relu': 'ReLU', 'topk': 'TopK', 'jumprelu': 'JumpReLU (surrogate)'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
TABLE = ['owt_val_l0', 'owt_val_mse', 'owt_val_nmse', 'owt_val_ev', 'owt_val_cosine', 'owt_val_dead_fraction',
         'owt_delta_ce', 'owt_kl', 'got_geographic_logistic_gap', 'got_numerical_logistic_gap',
         'got_cities_pair_accuracy_change', 'repe_logistic_gap', 'repe_diffmean_gap', 'repe_direction_cosine',
         'repe_persona_shift_cosine_final']


def dominance(row):
    if row['lower_is_better'] is None or pd.isna(row['lower_is_better']) or str(row['verdict']).startswith('no clear change'):
        return ''
    lower = str(row['lower_is_better']) == 'True'
    better = row['difference_p1_minus_seed42'] < 0 if lower else row['difference_p1_minus_seed42'] > 0
    if row['l0_matched_2pct']:
        return 'L0 matched'
    fewer = row['p1_l0'] <= row['baseline_l0']
    if better and fewer:
        return 'better at lower-or-equal L0'
    if not better and not fewer:
        return 'worse at higher-or-equal L0'
    return 'trade-off (needs L0 curve)'


def main():
    c = pd.read_csv(P1 / 'baseline_vs_p1.csv')
    c = c[c.p1_variant == 'p1_selected'].copy()
    c['dominance'] = c.apply(dominance, axis=1)
    c.to_csv(P1 / 'baseline_vs_p1_with_dominance.csv', index=False)
    w = c[c.metric.isin(TABLE)].pivot_table(index=['layer', 'architecture'], columns='metric',
                                            values=['baseline_seed42', 'p1'], aggfunc='first')
    w.columns = [f'{m}__{"base" if v == "baseline_seed42" else "p1"}' for v, m in w.columns]
    w = w[[f'{m}__{s}' for m in TABLE for s in ('base', 'p1')]]
    w.reset_index().to_csv(P1 / 'baseline_vs_p1_table.csv', index=False)

    # figure: OWT EV vs L0 (baseline seeds + P1), and dead fraction, per layer
    long = pd.read_csv(P1 / 'validation_evaluation_long.csv')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, layer in zip(axes, (0, 2, 6, 10)):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7.5)
        g = long[long.layer == layer]
        for a in ('relu', 'topk', 'jumprelu'):
            b = g[(g.architecture == a) & (g.variant == 'baseline')]
            p = g[(g.architecture == a) & (g.variant == 'p1_selected')]
            ax.scatter(b.owt_val_l0, b.owt_val_ev, color=COLOR[a], marker=MARKER[a], s=34, facecolors='none', lw=1.3,
                       label=f'{LABEL[a]} baseline (3 seeds)')
            ax.scatter(p.owt_val_l0, p.owt_val_ev, color=COLOR[a], marker=MARKER[a], s=70, edgecolors=INK, lw=0.8,
                       label=f'{LABEL[a]} P1 (seed 42)', zorder=3)
        ax.axvspan(128 * 0.95, 128 * 1.05, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.set_title(f'layer {layer}', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('actual L0 (OWT validation)', color=INK2, fontsize=8)
    axes[0].set_ylabel('explained variance (OWT validation)', color=INK2, fontsize=8)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(0.005, 0.93), ncol=3, fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.suptitle('P1 pilot vs uninterrupted baseline: reconstruction at the actual L0 reached', x=0.01, y=0.99, ha='left',
                 va='top', color=INK, fontsize=11)
    fig.text(0.01, 0.01, 'Validation split. Hollow: baseline seeds 42/1042/2042; filled: P1 (seed 42, pairs with baseline seed 42). '
             'Grey band: 128 ± 5 %. Layer-0 JumpReLU baseline seed 42 (L0 ~75) and P1 (L0 112) are outside the target band.',
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.82))
    fig.savefig(REPO / 'results/openwebtext/figures/p1_vs_baseline_ev_l0.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(c[c.verdict != 'no clear change'].groupby(['architecture', 'dominance', 'verdict']).size().to_string())


if __name__ == '__main__':
    main()
