#!/usr/bin/env python3
"""All-layer OWT baseline: uninterrupted protocol, 12 layers x 3 architectures x 3 seeds, L0 target 128.

Reads only (CPU, no training):
  results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/test_metrics_all_runs.csv
Writes (under --out-root, default repository root):
  results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/
      all_layers_per_run.csv                    one row per run, all metrics + sparsity flags + checkpoint path
      all_layers_summary.csv                    layer x architecture: mean and sd across seeds
      all_layers_matched_architecture_comparison.csv
                                                within-layer comparison, ONLY where all 3 architectures are matched
  results/openwebtext/figures/all_layers_{explained_variance,nmse,cosine,actual_l0}.png

Definitions (OWT test split, 12 700 non-BOS tokens per layer):
  mse        = sum (x - x_hat)^2 / (tokens * 768)                  (raw scale; never averaged across layers)
  nmse       = sum (x - x_hat)^2 / sum (x - mean_test(x))^2 = 1 - explained_variance
  cosine     = mean over tokens of cos(x, x_hat)
  actual_l0  = mean number of non-zero latents per token
Sparsity matching: |actual_l0 - 128| / 128 <= 0.10 (repository standard); the 20 % criterion is reported alongside.
A (layer, architecture) cell is matched only if every seed is matched. Architectures are compared within a layer only
when all three cells are matched; unmatched layers are listed but not ranked.
Seeds share initialization and batch order across architectures (train_one seeds both), so architecture
differences are also reported as paired per-seed differences.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

UNI = Path('results/openwebtext/reconstruction_evaluation/uninterrupted_protocol')
FIG = Path('results/openwebtext/figures')
CKPT = 'results/openwebtext/convergence/layer_{layer}/{arch}_l0_128_uninterrupted_seed{seed}/checkpoint_step_15000.pt'
ARCHS = ['relu', 'topk', 'jumprelu']
LABEL = {'relu': 'ReLU', 'topk': 'TopK', 'jumprelu': 'JumpReLU (surrogate)'}
# Reference categorical palette slots 1-3 (validated all-pairs, light mode); markers give a non-colour encoding.
COLOR = {'relu': '#2a78d6', 'topk': '#eb6834', 'jumprelu': '#1baf7a'}
MARKER = {'relu': 'o', 'topk': 's', 'jumprelu': '^'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
TARGET, TOL = 128, 0.10
OUT_ROOT = Path('.')


def out(rel):
    p = OUT_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def per_run(raw):
    df = pd.DataFrame({
        'layer': raw['layer'].astype(int), 'architecture': raw['architecture'], 'seed': raw['seed'].astype(int),
        'steps': raw['steps'], 'target_l0': raw['target_l0'], 'test_tokens': raw['test_test_tokens'],
        'mse': raw['test_mse'], 'nmse': 1.0 - raw['test_explained_variance'],
        'cosine': raw['test_mean_cosine_similarity'], 'explained_variance': raw['test_explained_variance'],
        'actual_l0': raw['test_actual_l0'],
        'val_explained_variance': raw['val_explained_variance'], 'val_l0': raw['val_l0'],
        'train_seconds': raw['train_seconds'],
        'gpu_peak_allocated_mb': raw.get('gpu_peak_allocated_mb'), 'gpu_peak_reserved_mb': raw.get('gpu_peak_reserved_mb'),
    })
    df['l0_rel_error'] = (df['actual_l0'] - TARGET).abs() / TARGET
    df['matched_10pct'] = df['l0_rel_error'] <= TOL
    df['matched_20pct'] = df['l0_rel_error'] <= 0.20
    recorded = raw['checkpoint_path'] if 'checkpoint_path' in raw else pd.Series([np.nan] * len(raw))
    df['checkpoint_path'] = [r if isinstance(r, str) and r else CKPT.format(layer=l, arch=a, seed=s)
                             for r, l, a, s in zip(recorded, df.layer, df.architecture, df.seed)]
    df['checkpoint_exists'] = [Path(p).exists() for p in df['checkpoint_path']]
    finite = np.isfinite(df[['mse', 'explained_variance', 'cosine', 'actual_l0']].to_numpy()).all(axis=1)
    df['run_ok'] = finite & df['checkpoint_exists'] & (df['test_tokens'] == 12700)
    df['sparsity_flag'] = np.where(df['matched_10pct'], '', np.where(
        df['matched_20pct'], 'UNMATCHED (10-20% off target)', 'UNMATCHED (>20% off target)'))
    return df.sort_values(['layer', 'architecture', 'seed']).reset_index(drop=True)


def summarize(df):
    g = df.groupby(['layer', 'architecture'])
    s = g.agg(n_seeds=('seed', 'nunique'),
              mse_mean=('mse', 'mean'), mse_sd=('mse', 'std'),
              nmse_mean=('nmse', 'mean'), nmse_sd=('nmse', 'std'),
              cosine_mean=('cosine', 'mean'), cosine_sd=('cosine', 'std'),
              ev_mean=('explained_variance', 'mean'), ev_sd=('explained_variance', 'std'),
              l0_mean=('actual_l0', 'mean'), l0_sd=('actual_l0', 'std'),
              l0_rel_error_max=('l0_rel_error', 'max'),
              n_seeds_matched_10pct=('matched_10pct', 'sum'),
              train_seconds_mean=('train_seconds', 'mean')).reset_index()
    s['cell_matched_10pct'] = s['n_seeds_matched_10pct'] == s['n_seeds']
    return s


def compare(df, s):
    rows = []
    for layer, sl in s.groupby('layer'):
        comparable = bool(sl['cell_matched_10pct'].all()) and set(sl.architecture) == set(ARCHS)
        best = sl.loc[sl['ev_mean'].idxmax(), 'architecture']
        for r in sl.itertuples():
            row = dict(layer=layer, architecture=r.architecture, comparable_all_three_matched=comparable,
                       ev_mean=r.ev_mean, ev_sd=r.ev_sd, nmse_mean=r.nmse_mean, l0_mean=r.l0_mean,
                       cell_matched_10pct=r.cell_matched_10pct)
            if comparable:
                row['rank_by_ev'] = int(sl['ev_mean'].rank(ascending=False)[sl.architecture == r.architecture].iloc[0])
                row['best_architecture'] = best
                a = df[(df.layer == layer) & (df.architecture == best)].set_index('seed')['explained_variance']
                b = df[(df.layer == layer) & (df.architecture == r.architecture)].set_index('seed')['explained_variance']
                d = (a - b).dropna()
                row['paired_ev_gap_to_best_mean'] = d.mean()
                row['paired_ev_gap_to_best_sd'] = d.std() if len(d) > 1 else np.nan
                row['seeds_where_best_is_higher'] = f'{int((d > 0).sum())}/{len(d)}' if r.architecture != best else ''
                # descriptive: best wins on every seed AND mean gap > 2 x paired sd
                row['gap_consistent_across_seeds'] = (r.architecture != best and (d > 0).all()
                                                      and d.mean() > 2 * (d.std() if len(d) > 1 else np.inf))
            rows.append(row)
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xticks(range(12))
    ax.set_xlabel('GPT-2 Small layer (hook_resid_pre)', color=INK2, fontsize=9)


def line_figure(s, df, col, ylabel, title, fname, log=False):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    style(ax)
    for i, a in enumerate(ARCHS):
        g = s[s.architecture == a].sort_values('layer')
        x = g['layer'] + (i - 1) * 0.08
        ax.plot(x, g[f'{col}_mean'], color=COLOR[a], lw=2, marker=MARKER[a], ms=7, label=LABEL[a], zorder=3)
        ax.errorbar(x, g[f'{col}_mean'], yerr=g[f'{col}_sd'], fmt='none', ecolor=COLOR[a], elinewidth=1.2, capsize=3, zorder=2)
        um = g[~g['cell_matched_10pct']]
        ax.scatter(um['layer'] + (i - 1) * 0.08, um[f'{col}_mean'], s=140, facecolors='none', edgecolors=INK,
                   linewidths=1.2, zorder=4,
                   label='L0 not matched (>10 % off 128)' if i == 0 and not s['cell_matched_10pct'].all() else None)
    if log:
        ax.set_yscale('log')
        ticks = [0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in ticks]))
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    h, lab = ax.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda k: lab[k].startswith('L0 not matched'))  # series first, flag last
    ax.legend([h[k] for k in order], [lab[k] for k in order], fontsize=8, frameon=False, labelcolor=INK2)
    fig.text(0.01, 0.01, 'Mean ± sd over seeds 42/1042/2042; OWT test, 12 700 tokens; d_sae 3072, target L0 128, 15k steps, '
             'uninterrupted protocol.', fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out(FIG / fname), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def l0_figure(s, fname):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    style(ax)
    ax.axhspan(TARGET * 0.9, TARGET * 1.1, color='#d9d8d2', alpha=0.6, lw=0, label='±10 % of target (matched)')
    ax.axhline(TARGET, color=INK2, lw=1, ls='--')
    for i, a in enumerate(ARCHS):
        g = s[s.architecture == a].sort_values('layer')
        x = g['layer'] + (i - 1) * 0.08
        ax.plot(x, g['l0_mean'], color=COLOR[a], lw=2, marker=MARKER[a], ms=7, label=LABEL[a], zorder=3)
        ax.errorbar(x, g['l0_mean'], yerr=g['l0_sd'], fmt='none', ecolor=COLOR[a], elinewidth=1.2, capsize=3)
    ax.set_yticks([70, 80, 90, 100, 115.2, 128, 140.8])
    ax.set_yticklabels(['70', '80', '90', '100', '115.2 (−10 %)', '128', '140.8 (+10 %)'])
    ax.set_ylim(65, 150)
    ax.set_ylabel('Actual L0 on test (mean ± sd over seeds)', color=INK2, fontsize=9)
    ax.set_title('Actual vs target sparsity (target L0 = 128)', color=INK, fontsize=11, loc='left')
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out(FIG / fname), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', type=Path, default=Path('.'))
    OUT_ROOT = ap.parse_args().out_root
    raw = pd.read_csv(UNI / 'test_metrics_all_runs.csv')
    assert not raw.duplicated(['layer', 'architecture', 'seed']).any(), 'duplicate runs'
    df = per_run(raw)
    s = summarize(df)
    c = compare(df, s)
    df.to_csv(out(UNI / 'all_layers_per_run.csv'), index=False)
    s.to_csv(out(UNI / 'all_layers_summary.csv'), index=False)
    c.to_csv(out(UNI / 'all_layers_matched_architecture_comparison.csv'), index=False)
    line_figure(s, df, 'ev', 'Explained variance (test)', 'Reconstruction quality by layer: explained variance',
                'all_layers_explained_variance.png')
    line_figure(s, df, 'nmse', 'Normalized MSE = 1 − EV (log scale)', 'Reconstruction error by layer: normalized MSE',
                'all_layers_nmse.png', log=True)
    line_figure(s, df, 'cosine', 'Mean cosine(x, x̂) (test)', 'Reconstruction direction by layer: cosine similarity',
                'all_layers_cosine.png')
    l0_figure(s, 'all_layers_actual_l0.png')
    n_expected = 12 * 3 * 3
    print(f'runs: {len(df)} / {n_expected} expected; ok: {int(df.run_ok.sum())}; '
          f'unmatched (10%): {int((~df.matched_10pct).sum())}')
    print(s[['layer', 'architecture', 'n_seeds', 'ev_mean', 'ev_sd', 'nmse_mean', 'cosine_mean', 'l0_mean',
             'n_seeds_matched_10pct']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
