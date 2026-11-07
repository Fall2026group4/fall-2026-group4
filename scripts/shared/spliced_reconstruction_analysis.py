#!/usr/bin/env python3
"""Summaries, matched-sparsity comparison, unified cross-dataset table and figures for Phase 0
(scripts/shared/spliced_reconstruction_evaluation.py). CPU only. Run from repository root.

Rules: runs > 10 % off L0 128 on OWT test (layer 0 JumpReLU seed 42) are excluded from means and rankings and listed
in excluded_unmatched_runs.csv. A layer is ranked only if all three architectures have all three seeds matched.
"consistent" = the leader is better on every seed AND the mean paired difference > 2 x its sd.
All numbers are validation-split numbers.
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
D = {k: REPO / f'results/{v}/spliced_evaluation' for k, v in
     [('owt', 'openwebtext'), ('got', 'geometry_of_truth'), ('repe', 'repe'), ('shared', 'shared')]}
PROBE = {'got': REPO / 'results/geometry_of_truth/all_layer_frozen_probe_evaluation/per_run.csv',
         'repe': REPO / 'results/repe/all_layer_frozen_probe_evaluation/per_run.csv'}
ARCHS = ['relu', 'topk', 'jumprelu']
LABEL = {'relu': 'ReLU', 'topk': 'TopK', 'jumprelu': 'JumpReLU (surrogate)'}
COLOR = {'relu': '#2a78d6', 'topk': '#eb6834', 'jumprelu': '#1baf7a'}   # reference palette slots 1-3
MARKER = {'relu': 'o', 'topk': 's', 'jumprelu': '^'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
KEYS = ['layer', 'architecture', 'seed']

# (dataset, column, lower_is_better, label) -- primary causal outcomes ranked at matched sparsity
PRIMARY = [('owt', 'owt__delta_ce', True, 'OWT next-token loss increase (nats/token)'),
           ('owt', 'owt__kl_mean', True, 'OWT KL(original || spliced) per token'),
           ('got', 'got__statement_kl_mean', True, 'GoT statement KL per token'),
           ('repe', 'repe_final__final_kl_mean', True, 'RepE final-position KL (final-token splice)'),
           ('repe', 'repe_final__persona_shift_cosine', False, 'RepE persona-shift cosine (final-token splice)'),
           ('repe', 'repe_all__persona_shift_cosine', False, 'RepE persona-shift cosine (all-position splice)')]


def load(key):
    d = pd.read_csv(D[key] / 'per_run.csv')
    c = pd.read_csv(D[key] / 'layer_controls.csv')
    return d, c


def wide(d):
    """One row per run: SAE columns plus the same columns for the norm-matched random-error control (suffix _rand)."""
    s = d[d.intervention == 'sae_reconstruction'].set_index(KEYS)
    r = d[d.intervention == 'norm_matched_random_error'].set_index(KEYS)
    num = [c for c in s.columns if '__' in c and not c.startswith('owt_val_recon__')]
    return s.join(r[num].add_suffix('_rand')).reset_index()


def compare(df, col, lower_better):
    rows = []
    for layer, sl in df.groupby('layer'):
        cells = sl.groupby('architecture')
        eligible = set(sl.architecture) == set(ARCHS) and all(len(g) == 3 and g.matched_10pct.all() for _, g in cells)
        means = cells[col].mean()
        best = means.idxmin() if lower_better else means.idxmax()
        for arch, g in cells:
            row = {'layer': layer, 'metric': col, 'architecture': arch, 'mean': g[col].mean(), 'sd': g[col].std(),
                   'ranking_eligible': eligible}
            if eligible:
                a = sl[sl.architecture == best].set_index('seed')[col]
                b = g.set_index('seed')[col]
                diff = (b - a) if lower_better else (a - b)   # > 0: leader better
                row.update(best_architecture=best, paired_diff_mean=diff.mean(), paired_diff_sd=diff.std(),
                           seeds_leader_better=f'{int((diff > 0).sum())}/3' if arch != best else '',
                           difference_consistent=bool(arch != best and (diff > 0).all() and diff.mean() > 2 * diff.std()))
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
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xticks(range(12))
    ax.set_xlabel('GPT-2 Small layer spliced (hook_resid_pre)', color=INK2, fontsize=8)


def panel(ax, df, col, title, ylabel, rand=True, hline=None, log=False):
    style(ax)
    m = df[df.matched_10pct]
    for i, a in enumerate(ARCHS):
        g = m[m.architecture == a].groupby('layer')
        x = np.array(sorted(m.layer.unique())) + (i - 1) * 0.1
        mu, sd = g[col].mean(), g[col].std()
        ax.plot(x, mu.values, color=COLOR[a], lw=2, marker=MARKER[a], ms=5.5, label=LABEL[a], zorder=4)
        ax.errorbar(x, mu.values, yerr=sd.values, fmt='none', ecolor=COLOR[a], elinewidth=1, capsize=2, zorder=3)
        if rand and f'{col}_rand' in m:
            ax.plot(x, g[f'{col}_rand'].mean().values, color=COLOR[a], lw=1.2, ls=':', marker=MARKER[a], ms=4,
                    mfc='none', zorder=2, label=f'{LABEL[a]}: random error, same norm')
    if hline is not None:
        ax.axhline(hline, color=INK2, lw=0.8)
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v:g}' if str(f'{v:g}').lstrip('0.').startswith(('2', '5')) else ''))
    ax.set_title(title, color=INK, fontsize=9.5, loc='left')
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)


def figure(panels, path, title, note):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(max(9.0, 5.4 * n), 4.9), facecolor=SURFACE, squeeze=False)
    for ax, (df, col, t, yl, kw) in zip(axes[0], panels):
        panel(ax, df, col, t, yl, **kw)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(0.005, 0.93), fontsize=7.2, frameon=False, labelcolor=INK2,
               ncol=3)
    fig.suptitle(title, x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
    fig.text(0.01, 0.01, textwrap.fill(note, 170), fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    out = D['shared']
    (out / 'figures').mkdir(exist_ok=True)
    W = {}
    for key in ('owt', 'got', 'repe'):
        d, c = load(key)
        w = wide(d)
        if key == 'owt':
            mean_abl = c[c.intervention == 'mean_ablation'].set_index('layer')['owt__ce_spliced']
            w['owt__loss_recovered'] = ((w.layer.map(mean_abl) - w['owt__ce_spliced'])
                                        / (w.layer.map(mean_abl) - w['owt__ce_original']))
            w['owt__loss_recovered_rand'] = ((w.layer.map(mean_abl) - w['owt__ce_spliced_rand'])
                                             / (w.layer.map(mean_abl) - w['owt__ce_original']))
            w['owt__sae_vs_random_delta_ce_ratio'] = w['owt__delta_ce'] / w['owt__delta_ce_rand']
        if key == 'got':
            for nm in ('cities', 'neg_cities', 'larger_than', 'smaller_than'):
                orig = c[c.intervention == 'identity'].set_index('layer')[f'got__{nm}_truth_pair_accuracy']
                w[f'got__{nm}_pair_accuracy_change'] = w[f'got__{nm}_truth_pair_accuracy'] - w.layer.map(orig)
        W[key] = w
        w.to_csv(D[key] / 'per_run_wide.csv', index=False)
        w[~w.matched_10pct].to_csv(D[key] / 'excluded_unmatched_runs.csv', index=False)
        num = [x for x in w.columns if '__' in x]
        s = (w[w.matched_10pct].groupby(['layer', 'architecture'])[num].agg(['mean', 'std']))
        s.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in s.columns]
        s.insert(0, 'n_seeds_used', w[w.matched_10pct].groupby(['layer', 'architecture']).size())
        s.reset_index().to_csv(D[key] / 'summary.csv', index=False)
        c.to_csv(D[key] / 'layer_controls_summary.csv', index=False)

    comp = pd.concat([compare(W[k], col, lb).assign(dataset=k) for k, col, lb, _ in PRIMARY], ignore_index=True)
    comp.to_csv(out / 'matched_architecture_comparison.csv', index=False)

    # ---- unified table (one row per run) with validation-split frozen-probe gaps from the all-layer evaluation
    u = W['owt'][KEYS + ['matched_10pct', 'sparsity_flag', 'owt_val_recon__explained_variance', 'owt_val_recon__l0',
                         'owt_val_recon__dead_fraction', 'owt__delta_ce', 'owt__kl_mean', 'owt__loss_recovered',
                         'owt__sae_vs_random_delta_ce_ratio', 'owt__top1_agreement']]
    u = u.merge(W['got'][KEYS + ['got__statement_kl_mean', 'got__all_final_token_kl_mean', 'got__cities_pair_accuracy_change',
                                 'got__larger_than_pair_accuracy_change']], on=KEYS)
    u = u.merge(W['repe'][KEYS + ['repe_final__final_kl_mean', 'repe_final__persona_shift_cosine',
                                  'repe_all__persona_shift_cosine', 'repe_final__persona_next_token_effect_corr']], on=KEYS)
    for key, path in PROBE.items():
        p = pd.read_csv(path)
        p = p[p.eval_set == 'validation']
        for dom, g in p.groupby('domain'):
            g = g.set_index(KEYS)
            u = u.join(g[['logistic_auroc_gap', 'diffmean_auroc_gap']].add_prefix(f'{dom}_val_'), on=KEYS)
    u.to_csv(out / 'unified_comparison.csv', index=False)

    # ---- how reconstruction relates to causal effects (matched runs; pooled across layers = confounded by depth)
    m = u[u.matched_10pct]
    rel = []
    pairs = [('owt_val_recon__explained_variance', y) for y in
             ['owt__delta_ce', 'owt__loss_recovered', 'got__statement_kl_mean', 'repe_final__final_kl_mean',
              'repe_final__persona_shift_cosine']]
    pairs += [('owt__kl_mean', 'repe_final__persona_shift_cosine'), ('owt__kl_mean', 'honesty_val_diffmean_auroc_gap'),
              ('owt__kl_mean', 'geographic_val_logistic_auroc_gap'), ('repe_final__persona_shift_cosine', 'honesty_val_diffmean_auroc_gap')]
    for x, y in pairs:
        f = m[[x, y, 'layer']].dropna()
        rho, pv = spearmanr(f[x], f[y])
        within = [spearmanr(g[x], g[y])[0] for _, g in f.groupby('layer') if len(g) >= 6]
        rel.append({'x': x, 'y': y, 'n': len(f), 'pooled_spearman': rho, 'pooled_p': pv,
                    'within_layer_mean_spearman': float(np.nanmean(within)), 'n_layers_within': len(within)})
    pd.DataFrame(rel).to_csv(out / 'reconstruction_vs_causal_correlations.csv', index=False)

    note = ('Validation split; mean ± sd over sparsity-matched seeds (42/1042/2042); layer 0 JumpReLU seed 42 excluded. '
            'Dotted: the same run with its SAE error replaced by a random error of equal per-token norm.')
    figure([(W['owt'], 'owt__delta_ce', 'Next-token loss increase', 'Δ CE (nats/token)', {'log': True}),
            (W['owt'], 'owt__loss_recovered', 'Loss recovered vs mean-ablation', 'fraction', {'hline': 1.0})],
           out / 'figures' / 'owt_spliced_loss.png', 'OWT: replacing activations with SAE reconstructions', note)
    figure([(W['got'], 'got__statement_kl_mean', 'GoT statements: KL per token (all-position splice)', 'KL (nats)', {'log': True}),
            (W['got'], 'got__cities_pair_accuracy_change', 'GoT cities: change in true-vs-false preference',
             'Δ pairwise accuracy (spliced − original)', {'hline': 0.0})],
           out / 'figures' / 'got_spliced_effects.png', 'GoT: effect of splicing on the model’s predictions', note)
    figure([(W['repe'], 'repe_final__persona_shift_cosine', 'Final-token splice', 'cos(persona shift original, spliced)', {}),
            (W['repe'], 'repe_all__persona_shift_cosine', 'All-position splice', 'cos(persona shift original, spliced)', {})],
           out / 'figures' / 'repe_spliced_persona_shift.png',
           'RepE: is the honest-vs-untruthful output shift preserved after splicing?', note)
    print(comp[comp.ranking_eligible & (comp.architecture == comp.best_architecture)]
          .groupby(['dataset', 'metric']).best_architecture.value_counts().to_string())


if __name__ == '__main__':
    main()
