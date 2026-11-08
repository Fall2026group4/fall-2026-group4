#!/usr/bin/env python3
"""Figures for R1 (and later R2). Run from repository root after 16_r1_evaluation.py:
    python scripts/repe/19_r1_r2_figures.py --r1
    python scripts/repe/19_r1_r2_figures.py --r2      # after 17_r2_sae_in_the_loop_steering.py --validation
Writes results/repe/figures/r1_honesty_preservation_layer6.png and r2_sae_in_the_loop_validation.png (refuse to
overwrite)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R1 = REPO / 'results/repe/r1_honesty_preservation_layer6'
R2 = REPO / 'results/repe/r2_sae_in_the_loop_steering_layer6'
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


def r2():
    out = FIG / 'r2_sae_in_the_loop_validation.png'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    T = pd.read_csv(R2 / 'validation_targets.csv')
    S = pd.read_csv(R2 / 'validation_summary.csv')
    pos = pd.read_json(R2 / 'validation_run_info.json', typ='series')['positive_control_E']['val']
    T = T[(T.sae == 'base') & (T.set == 'val') & (T.direction == 'U->H') & (T.error_kept == True) & T['kind'].isna()]  # noqa: E712
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for i, arch in enumerate(('relu', 'topk', 'jumprelu')):
        ax = axes[i]
        for k, c in zip((1, 4, 16), ('#4e79a7', '#f28e2b', '#e15759')):
            g = T[(T.architecture == arch) & (T.k == k)].groupby('alpha').E
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=c, marker='o', capsize=3, label=f'k={k}')
        ax.axhline(0, color='k', lw=0.6)
        ax.axhline(pos, color='g', ls='--', lw=0.8, label='positive control')
        ax.set_title(f'{arch}: E vs alpha (U->H, error kept; mean +- sd of 3 seeds)', fontsize=9)
        ax.set_xlabel('alpha')
        ax.legend(fontsize=7)
    ax = axes[3]
    b = S[(S.sae == 'base') & (S.set == 'val')]
    for j, (col, lab, c) in enumerate((('E_alpha1', 'target', '#e15759'), ('C2_q95', 'C2 random features q95', '#888888'),
                                       ('C3_q95', 'C3 matched-norm dirs q95', '#bbbbbb'),
                                       ('E_unrelated', 'unrelated features', '#59a14f'))):
        x = [i * 5 + j for i in range(3)]
        ax.bar(x, [b[b.architecture == a][col].mean() for a in ('relu', 'topk', 'jumprelu')], color=c, label=lab)
    ax.set_xticks([1.5, 6.5, 11.5], ['ReLU', 'TopK', 'JumpReLU'])
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title('alpha = 1, mean over seeds and k (validation)', fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle('R2: SAE-in-the-loop clamp of honesty features, GPT-2 Small layer 6 (VALIDATION, descriptive; '
                 'E = fraction of honest-persona output shift recovered)', fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--r1', action='store_true')
    g.add_argument('--r2', action='store_true')
    a = ap.parse_args()
    r1() if a.r1 else r2()
