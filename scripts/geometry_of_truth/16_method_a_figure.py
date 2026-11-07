#!/usr/bin/env python3
"""Figure for Method A at layer 6 (reads results/geometry_of_truth/method_a_layer6/per_sae.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
COLOR = {'relu': '#2a78d6', 'topk': '#eb6834', 'jumprelu': '#1baf7a'}
MARKER = {'relu': 'o', 'topk': 's', 'jumprelu': '^'}
LABEL = {'relu': 'ReLU', 'topk': 'TopK', 'jumprelu': 'JumpReLU'}
VAR = ['base', 'owt_only', 'method_a']
VNAME = ['base SAE', 'OWT-only\ncontrol', 'Method A\n(GoT + replay)']
PANELS = [('got_geographic_indep_mlp_gap_mean', 'GoT geographic: independent MLP probe gap\n(primary outcome; lower is better)'),
          ('got_geographic_indep_logistic_gap', 'GoT geographic: independent logistic probe gap'),
          ('got_geographic_val_relative_sq_error', 'GoT reconstruction error (relative)'),
          ('splice_owt_delta_ce', 'OWT next-token loss increase when spliced')]
d = pd.read_csv(REPO / 'results/geometry_of_truth/method_a_layer6/per_sae.csv')
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)
for ax, (col, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    for i, a in enumerate(('relu', 'topk', 'jumprelu')):
        g = d[d.architecture == a]
        for s in (42, 1042, 2042):
            gs = g[g.seed == s].set_index('variant').reindex(VAR)[col]
            ax.plot([x + (i - 1) * 0.12 for x in range(3)], gs.values, color=COLOR[a], lw=0.8, alpha=0.45)
        m = g.groupby('variant')[col].mean().reindex(VAR)
        ax.plot([x + (i - 1) * 0.12 for x in range(3)], m.values, color=COLOR[a], lw=2.2, marker=MARKER[a], ms=7,
                label=LABEL[a], zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(VNAME, fontsize=8, color=INK2)
    ax.set_title(title, color=INK, fontsize=9, loc='left')
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc='upper right', fontsize=8, frameon=False, labelcolor=INK2, ncol=3)
fig.suptitle('Method A at layer 6: GoT reconstruction fine-tuning with OWT replay vs base SAE and OWT-only control',
             x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
fig.text(0.01, 0.01, 'Validation only. Thick lines: mean of seeds 42/1042/2042; thin lines: individual seeds (paired). '
         'Probes fitted on original activations of train groups B; fine-tuning used groups A.', fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
fig.savefig(REPO / 'results/geometry_of_truth/figures/method_a_layer6.png', dpi=160, facecolor=SURFACE)
