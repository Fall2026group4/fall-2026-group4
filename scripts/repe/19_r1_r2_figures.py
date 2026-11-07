#!/usr/bin/env python3
"""Figures for R1 (and later R2). Run from repository root after 16_r1_evaluation.py:
    python scripts/repe/19_r1_r2_figures.py --r1
Writes results/repe/figures/r1_honesty_preservation_layer6.png (refuses to overwrite)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R1 = REPO / 'results/repe/r1_honesty_preservation_layer6'
FIG = REPO / 'results/repe/figures'
VARIANTS = ('base', 'method_a', 'r_recon', 'r_samepersona', 'r_honesty')
COLORS = ('#888888', '#b07aa1', '#4e79a7', '#f28e2b', '#e15759')


def r1():
    out = FIG / 'r1_honesty_preservation_layer6.png'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    df = pd.read_csv(R1 / 'per_sae.csv')
    panels = [('val_B_mlp_gap', 'B-MLP AUROC gap, validation (primary; lower better)'),
              ('cw_B_mlp_gap', 'B-MLP AUROC gap, cross-wording (lower better)'),
              ('val_pair_direction_cosine', 'Pair-difference direction cosine, validation (higher better)'),
              ('val_l0', 'RepE validation-prompt L0 (guard: within 5 % of base)')]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (col, title) in zip(axes, panels):
        for i, arch in enumerate(('relu', 'topk', 'jumprelu')):
            for j, (v, c) in enumerate(zip(VARIANTS, COLORS)):
                d = df[(df.architecture == arch) & (df.variant == v)][col]
                x = i * (len(VARIANTS) + 1) + j
                ax.bar(x, d.mean(), color=c, label=v if i == 0 else None)
                ax.scatter([x] * len(d), d, color='k', s=8, zorder=3)
        ax.set_xticks([i * (len(VARIANTS) + 1) + 2 for i in range(3)], ['ReLU', 'TopK', 'JumpReLU'])
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle('R1: RepE honesty-preservation fine-tuning, GPT-2 Small layer 6 (bars = mean of 3 seeds, dots = seeds; '
                 'validation only)', fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--r1', action='store_true', required=True)
    ap.parse_args()
    r1()
