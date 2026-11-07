#!/usr/bin/env python3
"""Figures for the 200-control causal test (RepE) and the Gemma-2-2B steering sweep.

Formerly scripts/professor_review/plot_causal_and_gemma.py. Reads and writes the dataset directories:
  results/repe/steering/causal_random_controls/{causal_controls_200,causal_summary}.csv -> .../figures/
  results/gemma/steering_sweep_layer11/gemma_steering_sweep*.csv                      -> .../figures/
Use --out-root DIR to write the figures under DIR instead (verification).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CAUSAL = Path('results/repe/steering/causal_random_controls')
GEMMA = Path('results/gemma/steering_sweep_layer11')
OUT_ROOT = Path('.')


def fig_path(base, name):
    p = OUT_ROOT / base / 'figures' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def causal():
    df = pd.read_csv(CAUSAL / 'causal_controls_200.csv')
    summ = pd.read_csv(CAUSAL / 'causal_summary.csv')
    g = df[(df.intervention_kind == 'control') & (df.control_group == 'gaussian')]['effect_E']
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(g, bins=30, color='#bbbbbb', label=f'200 Gaussian random directions (same norm)')
    raw = summ[summ.intervention == 'raw_honesty_direction'].iloc[0]
    ax.axvline(raw.effect_E, color='#C44E52', lw=2, label=f'raw honesty direction (p={raw.vs_gaussian_p_empirical:.3f})')
    ax.axvspan(raw.E_ci95_low, raw.E_ci95_high, color='#C44E52', alpha=0.15)
    for arch, c in (('relu', '#4C72B0'), ('topk', '#DD8452'), ('jumprelu', '#55A868')):
        r = summ[(summ.intervention == 'sae_reconstruction_direction') & (summ.sae_architecture == arch)
                 & (summ.sae_variant == 'pretrained')].iloc[0]
        ax.axvline(r.effect_E, color=c, ls='--', label=f'{arch} SAE-reconstruction direction (p={r.vs_gaussian_p_empirical:.2f})')
    ax.set_xlabel('E = fraction of honest-persona output shift recovered (test, 90 statements)')
    ax.set_ylabel('count')
    pc = summ[summ.intervention_kind == 'positive_control'].iloc[0]
    ax.set_title(f'Task C — GPT-2 L6 final-token intervention, norm 0.22 '
                 f'(positive control: activation patch E={pc.effect_E:.3f}, off-scale)', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path(CAUSAL, 'causal_vs_200_gaussian_controls.png'), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, arch in enumerate(('relu', 'topk', 'jumprelu')):
        for j, var in enumerate(('pretrained', 'reconstruction_only', 'truth_aware')):
            ax = axes[i, j]
            c = df[(df.intervention_kind == 'control') & (df.control_group == f'{arch}_{var}__feat1')]['effect_E']
            ax.hist(c, bins=30, color='#bbbbbb')
            t = summ[(summ.intervention == 'sae_top1_feature') & (summ.sae_architecture == arch) & (summ.sae_variant == var)].iloc[0]
            ax.axvline(t.effect_E, color='#C44E52', lw=2)
            ax.set_title(f'{arch} / {var}\ntop-1 feature p={t.vs_random_features_p_empirical:.3f}', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('E')
    fig.suptitle('Task C — top-1 SAE honesty feature (red) vs 200 random alive features of the same SAE (grey)')
    fig.tight_layout()
    fig.savefig(fig_path(CAUSAL, 'top1_feature_vs_200_random_features.png'), dpi=160)
    plt.close(fig)


def gemma():
    p = GEMMA / 'gemma_steering_sweep.csv'
    if not p.exists():
        return
    s = pd.read_csv(p)
    allr = pd.read_csv(GEMMA / 'gemma_steering_sweep_all.csv')
    for split in ('val', 'test'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for col, mode in enumerate(('segment', 'final')):
            base = s[(s.split == split) & (s.direction == 'none')].iloc[0]
            for row, metric, lab in ((0, 'mean_LD', 'mean logit(TRUE) - logit(FALSE)'),
                                     (1, 'judged_true_pct', '% judged TRUE')):
                ax = axes[row, col]
                rnd = allr[(allr.split == split) & (allr['mode'] == mode) & (allr.direction == 'random')]
                for cid, g in rnd.groupby('control_id'):
                    g = pd.concat([g, pd.DataFrame([{**base.to_dict(), 'alpha_act_norm': 0.0}])]).sort_values('alpha_act_norm')
                    ax.plot(g['alpha_act_norm'], g[metric], color='#cccccc', lw=0.8)
                for d, c in (('bare', '#C44E52'), ('context', '#4C72B0')):
                    g = s[(s.split == split) & (s['mode'] == mode) & (s.direction == d)]
                    g = pd.concat([g, pd.DataFrame([{**base.to_dict(), 'alpha_act_norm': 0.0}])]).sort_values('alpha_act_norm')
                    ax.plot(g['alpha_act_norm'], g[metric], marker='o', color=c, label=f'{d} truth direction')
                ax.axvline(0, color='k', lw=0.5)
                ax.set_xscale('symlog', linthresh=0.01)
                ax.set_ylabel(lab)
                ax.set_title(f'positions: {mode}', fontsize=9)
                if row == 1:
                    ax.set_xlabel('signed strength α (× mean activation norm), symlog')
        axes[0, 0].plot([], [], color='#cccccc', label='20 random directions (same norm)')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'Task D — Gemma-2-2B layer 11 truth-direction steering, {split} split')
        fig.tight_layout()
        fig.savefig(fig_path(GEMMA, f'gemma_steering_{split}.png'), dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', type=Path, default=Path('.'))
    OUT_ROOT = ap.parse_args().out_root
    causal()
    gemma()
