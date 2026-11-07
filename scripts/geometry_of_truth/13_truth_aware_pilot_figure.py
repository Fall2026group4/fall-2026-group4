#!/usr/bin/env python3
"""Figure for the truth-aware pilot (reads results/geometry_of_truth/truth_aware_pilot/per_sae.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID, SURFACE, ACCENT = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb', '#2a78d6'
ORDER = ['base', 'reconstruction_only', 'random_teacher', 'truth_aware']
NAMES = ['base SAE', 'recon-only', 'random teacher', 'truth-aware']
PANELS = [('got_geographic_TEACHER_gap_objective_only', 'Teacher probe (training target)\nobjective attainment only'),
          ('got_geographic_indep_logistic_gap', 'Independent logistic probe (B)\ncos with teacher 0.64'),
          ('got_geographic_indep_mlp_gap_mean', 'Independent MLP probe (B)\nmean of 5 probe seeds'),
          ('got_geographic_val_relative_sq_error', 'GoT reconstruction error\n(relative squared error)')]

d = pd.read_csv(REPO / 'results/geometry_of_truth/truth_aware_pilot/per_sae.csv')
fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), facecolor=SURFACE)
for ax, (col, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    for i, v in enumerate(ORDER):
        g = d[d.variant == v]
        ax.scatter([i] * len(g), g[col], color=INK2, s=22, zorder=3, label='one seed' if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [g[col].mean()] * 2, color=ACCENT, lw=2.5, zorder=4, label='mean of 3 seeds' if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(NAMES, rotation=20, ha='right', fontsize=8, color=INK2)
    ax.set_title(title, color=INK, fontsize=9, loc='left')
axes[0].set_ylabel('AUROC gap on GoT geographic validation\n(original − reconstructed; lower is better)', color=INK2, fontsize=8)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc='upper right', fontsize=8, frameon=False, labelcolor=INK2, ncol=2)
fig.suptitle('Truth-aware pilot (layer 6, TopK): the training-probe gain does not carry over to the independent MLP probe',
             x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
fig.text(0.01, 0.01, 'Validation split only. Probes fitted on original activations of train groups B; the SAEs were fine-tuned on groups A.',
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
fig.savefig(REPO / 'results/geometry_of_truth/figures/truth_aware_pilot.png', dpi=160, facecolor=SURFACE)
