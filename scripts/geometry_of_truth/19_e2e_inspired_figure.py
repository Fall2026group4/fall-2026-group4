#!/usr/bin/env python3
"""Figure for the E2E-inspired layer-6 experiment (reads results/geometry_of_truth/e2e_inspired_layer6/per_sae.csv).
Hollow markers: L0 NOT matched to the base (OWT-validation L0 > 5 % above base) - not creditable comparisons."""
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
M = ['base', 'A', 'B', 'C']
MN = ['base SAE', 'A: recon\n(GoT + replay)', 'B: + output KL', 'C: + KL + down-\nstream recon']
PANELS = [('splice_owt_kl', 'OWT: KL(original || spliced) per token'),
          ('splice_got_statement_kl', 'GoT statements: KL per token'),
          ('downstream_nmse_mean', 'Downstream NMSE, layers 7-11 (OWT)'),
          ('got_geographic_indep_mlp_gap_mean', 'GoT geographic: independent MLP probe gap')]
d = pd.read_csv(REPO / 'results/geometry_of_truth/e2e_inspired_layer6/per_sae.csv')
base_l0 = d[d.method == 'base'].set_index(['architecture', 'seed'])['owt_val_l0']
d['l0_matched'] = [abs(r.owt_val_l0 - base_l0[(r.architecture, r.seed)]) / base_l0[(r.architecture, r.seed)] <= 0.05 for r in d.itertuples()]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), facecolor=SURFACE)
for ax, (col, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    for i, a in enumerate(('relu', 'topk', 'jumprelu')):
        g = d[d.architecture == a].groupby('method').agg(v=(col, 'mean'), sd=(col, 'std'), ok=('l0_matched', 'all')).reindex(M)
        x = [k + (i - 1) * 0.1 for k in range(4)]
        ax.plot(x, g.v, color=COLOR[a], lw=2, zorder=2, label=LABEL[a])
        ax.errorbar(x, g.v, yerr=g.sd, fmt='none', ecolor=COLOR[a], elinewidth=1, capsize=2.5, zorder=2)
        for xx, (_, r) in zip(x, g.iterrows()):
            ax.scatter(xx, r.v, marker=MARKER[a], s=55, zorder=3, color=COLOR[a] if r.ok else SURFACE, edgecolors=COLOR[a], lw=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(MN, fontsize=7.5, color=INK2)
    ax.set_title(title, color=INK, fontsize=9.5, loc='left')
ax0 = axes[0]
ax0.scatter([], [], marker='o', facecolors=SURFACE, edgecolors=INK2, label='hollow: L0 > 5 % above base (not matched)')
h, lab = ax0.get_legend_handles_labels()
fig.legend(h, lab, loc='upper right', fontsize=8, frameon=False, labelcolor=INK2, ncol=4)
fig.suptitle('E2E-inspired fine-tuning at layer 6 (not an exact reproduction of Braun et al. 2024)', x=0.01, y=0.99, ha='left',
             va='top', color=INK, fontsize=11)
fig.text(0.01, 0.01, 'Validation only; mean ± sd over seeds 42/1042/2042. All panels: lower is better. Splicing metrics are not steering results.',
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.88))
fig.savefig(REPO / 'results/geometry_of_truth/figures/e2e_inspired_layer6.png', dpi=160, facecolor=SURFACE)
