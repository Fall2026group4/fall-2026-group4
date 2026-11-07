#!/usr/bin/env python3
"""Tasks A/B analysis: original vs consistent-protocol reconstruction, matched vs unmatched sparsity.

Inputs (read-only):
  results/openwebtext/reconstruction_evaluation/layers_2_6_10/test_reconstruction_metrics.csv (original, mixed procedures)
  results/professor_review_corrections/owt_consistent_protocol/test_metrics_all_runs.csv     (corrected)
  results/openwebtext/sae_sparsity_sweep/layer_*/comparison.csv                               (1000-step sweep)
Outputs:
  reconstruction_comparison.csv, sparsity_comparison.csv, sparsity_matched_cells.csv, figures/*.png

Matching criterion used throughout: |actual L0 - target| / target <= 0.10 (the stricter of the two
criteria found in the repository; the 0.20 criterion is reported alongside).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path('results/professor_review_corrections')
FIG = OUT / 'figures'
ARCH_ORDER = ['relu', 'topk', 'jumprelu']
COLORS = {'relu': '#4C72B0', 'topk': '#DD8452', 'jumprelu': '#55A868'}
TOL = 0.10


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    orig = pd.read_csv('results/openwebtext/reconstruction_evaluation/layers_2_6_10/test_reconstruction_metrics.csv')
    new = pd.read_csv(OUT / 'owt_consistent_protocol' / 'test_metrics_all_runs.csv')
    assert len(new) == 27, f'expected 27 corrected runs, found {len(new)}'
    assert (new['test_test_tokens'] == 12700).all() and (orig['test_tokens'] == 12700).all()

    agg = new.groupby(['layer', 'architecture']).agg(
        corrected_mse_mean=('test_mse', 'mean'), corrected_mse_sd=('test_mse', 'std'),
        corrected_cosine_mean=('test_mean_cosine_similarity', 'mean'),
        corrected_cosine_sd=('test_mean_cosine_similarity', 'std'),
        corrected_ev_mean=('test_explained_variance', 'mean'), corrected_ev_sd=('test_explained_variance', 'std'),
        corrected_l0_mean=('test_actual_l0', 'mean'), corrected_l0_sd=('test_actual_l0', 'std'),
        corrected_seed42_mse=('test_mse', 'first'), n_seeds=('seed', 'nunique')).reset_index()
    s42 = new[new.seed == 42].set_index(['layer', 'architecture'])
    o = orig.rename(columns={'mse': 'original_mse', 'mean_cosine_similarity': 'original_cosine',
                             'explained_variance': 'original_ev', 'actual_l0': 'original_l0',
                             'training_procedure': 'original_procedure'})
    rc = o[['layer', 'architecture', 'original_procedure', 'test_tokens', 'original_mse', 'original_cosine',
            'original_ev', 'original_l0']].merge(agg, on=['layer', 'architecture'])
    rc['corrected_seed42_mse'] = [s42.loc[(r.layer, r.architecture), 'test_mse'] for r in rc.itertuples()]
    rc['delta_ev_corrected_minus_original'] = rc['corrected_ev_mean'] - rc['original_ev']
    rc['delta_mse_rel'] = (rc['corrected_mse_mean'] - rc['original_mse']) / rc['original_mse']
    rc['original_within_corrected_seed_range'] = [
        new[(new.layer == r.layer) & (new.architecture == r.architecture)]['test_explained_variance'].between(-1, 1).all()
        and (new[(new.layer == r.layer) & (new.architecture == r.architecture)]['test_explained_variance'].min()
             <= r.original_ev <= new[(new.layer == r.layer) & (new.architecture == r.architecture)]['test_explained_variance'].max())
        for r in rc.itertuples()]
    rc['corrected_l0_rel_error'] = (rc['corrected_l0_mean'] - 128).abs() / 128
    rc['corrected_l0_matched_10pct'] = new.groupby(['layer', 'architecture'])['test_l0_rel_error'].max().values <= TOL
    # within-layer ranking
    for col, prefix in (('original_ev', 'original'), ('corrected_ev_mean', 'corrected')):
        rc[f'{prefix}_rank_within_layer'] = rc.groupby('layer')[col].rank(ascending=False).astype(int)
    rc.to_csv(OUT / 'reconstruction_comparison.csv', index=False)
    print(rc[['layer', 'architecture', 'original_procedure', 'original_ev', 'corrected_ev_mean', 'corrected_ev_sd',
              'original_l0', 'corrected_l0_mean', 'original_rank_within_layer', 'corrected_rank_within_layer']]
          .round(4).to_string(index=False))

    # ---------------- sparsity table
    sweep = []
    for layer in range(12):
        d = pd.read_csv(f'results/openwebtext/sae_sparsity_sweep/layer_{layer}/comparison.csv')
        d['layer'] = layer
        sweep.append(d)
    sweep = pd.concat(sweep)
    sp = pd.DataFrame(dict(source='1000-step sweep (validation split)', layer=sweep['layer'],
                           architecture=sweep['architecture'], target_l0=sweep['target_l0'], seed=sweep['seed'],
                           steps=sweep['steps'], actual_l0=sweep['l0'], mse=sweep['mse'], cosine=sweep['cosine'],
                           explained_variance=sweep['explained_variance']))
    sp = pd.concat([sp,
                    pd.DataFrame(dict(source='original 15k (test split, mixed procedure)', layer=orig['layer'],
                                      architecture=orig['architecture'], target_l0=128, seed=42, steps=15000,
                                      actual_l0=orig['actual_l0'], mse=orig['mse'],
                                      cosine=orig['mean_cosine_similarity'],
                                      explained_variance=orig['explained_variance'])),
                    pd.DataFrame(dict(source='corrected 15k (test split, uninterrupted)', layer=new['layer'],
                                      architecture=new['architecture'], target_l0=128, seed=new['seed'], steps=15000,
                                      actual_l0=new['test_actual_l0'], mse=new['test_mse'],
                                      cosine=new['test_mean_cosine_similarity'],
                                      explained_variance=new['test_explained_variance']))], ignore_index=True)
    sp['l0_rel_error'] = (sp['actual_l0'] - sp['target_l0']).abs() / sp['target_l0']
    sp['matched_10pct'] = sp['l0_rel_error'] <= 0.10
    sp['matched_20pct'] = sp['l0_rel_error'] <= 0.20
    sp['matching_table'] = np.where(sp['matched_10pct'], 'matched', 'unmatched')
    sp.to_csv(OUT / 'sparsity_comparison.csv', index=False)

    # cells (source, layer, target, seed) where all three architectures are matched -> directly comparable
    cells = []
    for key, g in sp.groupby(['source', 'layer', 'target_l0', 'seed']):
        if set(g['architecture']) != set(ARCH_ORDER):
            continue
        ok = bool(g['matched_10pct'].all())
        row = dict(zip(['source', 'layer', 'target_l0', 'seed'], key), all_three_matched=ok)
        for _, r in g.iterrows():
            row[f"{r['architecture']}_l0"] = r['actual_l0']
            row[f"{r['architecture']}_ev"] = r['explained_variance']
        row['best_ev_architecture'] = g.loc[g['explained_variance'].idxmax(), 'architecture']
        cells.append(row)
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / 'sparsity_matched_cells.csv', index=False)
    s = sp[sp.source.str.startswith('1000')].groupby(['target_l0', 'architecture']).agg(
        matched_10pct=('matched_10pct', 'sum'), matched_20pct=('matched_20pct', 'sum'), runs=('matched_10pct', 'size'))
    print('\nSweep matching counts:\n', s.to_string())
    m = cells[cells.source.str.startswith('1000')]
    print('\nSweep cells with all three architectures matched (10%):',
          int(m.all_three_matched.sum()), 'of', len(m))
    print(m[m.all_three_matched].groupby('target_l0')['best_ev_architecture'].value_counts().to_string())

    # ---------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, layer in zip(axes, (2, 6, 10)):
        sub = rc[rc.layer == layer].set_index('architecture').loc[ARCH_ORDER]
        x = np.arange(3)
        ax.bar(x - 0.18, sub['original_ev'], 0.36, label='original (mixed procedure, 1 run)', color='#bbbbbb')
        ax.bar(x + 0.18, sub['corrected_ev_mean'], 0.36, yerr=sub['corrected_ev_sd'], capsize=4,
               label='corrected (uninterrupted, mean ± sd, 3 seeds)', color=[COLORS[a] for a in ARCH_ORDER])
        ax.set_xticks(x, ARCH_ORDER)
        ax.set_title(f"Layer {layer} ({sub['original_procedure'].iloc[0]})", fontsize=10)
        lo = min(sub['original_ev'].min(), sub['corrected_ev_mean'].min())
        ax.set_ylim(lo - 0.03, min(1.0, max(sub['original_ev'].max(), sub['corrected_ev_mean'].max()) + 0.02))
        ax.set_ylabel('Explained variance (OWT test, 12 700 tokens)')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Task A — OWT reconstruction: original vs consistent-protocol checkpoints (target L0=128)')
    fig.tight_layout()
    fig.savefig(FIG / 'taskA_owt_original_vs_corrected_ev.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    sw = sp[sp.source.str.startswith('1000')]
    for ax, t in zip(axes, (32, 64, 128)):
        for a in ARCH_ORDER:
            g = sw[(sw.target_l0 == t) & (sw.architecture == a)].sort_values('layer')
            ax.plot(g['layer'], g['actual_l0'], marker='o', color=COLORS[a], label=a)
        ax.axhspan(t * 0.9, t * 1.1, color='green', alpha=0.12, label='±10% band')
        ax.axhline(t, color='k', lw=0.8, ls='--')
        ax.set_title(f'target L0 = {t}')
        ax.set_xlabel('GPT-2 layer')
        ax.set_yscale('log')
    axes[0].set_ylabel('actual L0 (validation, log scale)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Task B — 1000-step sweep: actual vs target L0 (outside the band = not comparable)')
    fig.tight_layout()
    fig.savefig(FIG / 'taskB_sweep_target_vs_actual_l0.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
