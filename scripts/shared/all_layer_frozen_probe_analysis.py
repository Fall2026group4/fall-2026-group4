#!/usr/bin/env python3
"""Summaries, matched-sparsity architecture comparison, reconstruction-vs-truth analysis and figures for the outputs
of ``scripts/shared/all_layer_frozen_probe_evaluation.py``. CPU only; reads per_run.csv, writes next to it.

Run from repository root:  python scripts/shared/all_layer_frozen_probe_analysis.py --dataset got|repe

Rules (fixed before looking at the truth/honesty numbers):
* Runs whose OWT-test L0 is > 10 % off 128 are excluded from every mean and ranking (layer 0 JumpReLU seed 42).
  Their numbers stay in per_run.csv with ``sparsity_flag`` set.
* A layer is ranked only if all 3 architectures have all 3 seeds matched, and only where that probe is informative
  (its validation AUROC on ORIGINAL activations >= 0.70; decided per probe). Rankings use the TEST set and seeds as pairs.
* "consistent" = the leader has the smaller gap on every seed AND the mean paired difference > 2 x its sd.
"""
import argparse
import textwrap
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIRS = {'got': REPO / 'results/geometry_of_truth', 'repe': REPO / 'results/repe'}
SUB = 'all_layer_frozen_probe_evaluation'
OWT = REPO / 'results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_per_run.csv'
ARCHS = ['relu', 'topk', 'jumprelu']
LABEL = {'relu': 'ReLU', 'topk': 'TopK', 'jumprelu': 'JumpReLU (surrogate)'}
COLOR = {'relu': '#2a78d6', 'topk': '#eb6834', 'jumprelu': '#1baf7a'}   # reference palette slots 1-3
MARKER = {'relu': 'o', 'topk': 's', 'jumprelu': '^'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
MIN_DIRECTION_NORM = 1e-4
METRICS = ['mse', 'nmse', 'explained_variance', 'relative_sq_error', 'cosine', 'l0',
           'logistic_auroc_reconstructed', 'logistic_auroc_gap', 'logistic_retained_signal',
           'diffmean_auroc_reconstructed', 'diffmean_auroc_gap', 'diffmean_retained_signal',
           'direction_cosine', 'direction_norm_ratio']


def load(dataset):
    d = pd.read_csv(DIRS[dataset] / SUB / 'per_run.csv')
    owt = pd.read_csv(OWT)[['layer', 'architecture', 'seed', 'explained_variance', 'nmse']]
    d = d.merge(owt.rename(columns={'explained_variance': 'owt_test_ev', 'nmse': 'owt_test_nmse'}),
                on=['layer', 'architecture', 'seed'], how='left')
    o = pd.read_csv(DIRS[dataset] / SUB / 'original_probes_by_layer.csv')
    o['logistic_informative'] = o['validation_logistic_auroc'] >= 0.70
    o['diffmean_informative'] = o['validation_diffmean_auroc'] >= 0.70
    d = d.merge(o[['domain', 'layer', 'logistic_informative', 'diffmean_informative', 'direction_norm']], on=['domain', 'layer'])
    # GoT numerical layer 0: the class means are identical (norm exactly 0 in float64); the 2026-11-07 run computed
    # them in float32 and stored ~3e-5 of rounding noise. Direction metrics are undefined there -> mask.
    undefined = d['direction_norm'] < MIN_DIRECTION_NORM
    d.loc[undefined, ['direction_cosine', 'direction_norm_ratio']] = np.nan
    for pr in ('logistic', 'diffmean'):
        # Share of the original above-chance AUROC that survives reconstruction (only meaningful if informative).
        d[f'{pr}_retained_signal'] = (d[f'{pr}_auroc_reconstructed'] - 0.5) / (d[f'{pr}_auroc_original'] - 0.5)
        d.loc[~d[f'{pr}_informative'], f'{pr}_retained_signal'] = np.nan
    return d


def summarize(d):
    m = d[d.matched_10pct]
    agg = {f'{c}_{s}': (c, s) for c in METRICS for s in ('mean', 'std')}
    s = m.groupby(['domain', 'eval_set', 'layer', 'architecture']).agg(n_seeds_used=('seed', 'nunique'), **agg).reset_index()
    s = s.rename(columns={c: c.replace('_std', '_sd') for c in s.columns})
    allseeds = d.groupby(['domain', 'eval_set', 'layer', 'architecture']).matched_10pct.all().rename('cell_all_seeds_matched')
    orig = d.groupby(['domain', 'eval_set', 'layer']).agg(logistic_auroc_original=('logistic_auroc_original', 'first'),
                                                          diffmean_auroc_original=('diffmean_auroc_original', 'first'),
                                                          informative_layer=('informative_layer', 'first'),
                                                          diffmean_informative=('diffmean_informative', 'first'))
    return s.join(allseeds, on=['domain', 'eval_set', 'layer', 'architecture']).join(orig, on=['domain', 'eval_set', 'layer'])


def compare(d, s, eval_set='test'):
    rows = []
    for (domain, layer), sl in s[s.eval_set == eval_set].groupby(['domain', 'layer']):
        eligible = bool(sl.cell_all_seeds_matched.all() and set(sl.architecture) == set(ARCHS))
        for pr in ('logistic', 'diffmean'):
            informative = bool(d[(d.domain == domain) & (d.layer == layer)][f'{pr}_informative'].iloc[0])
            col = f'{pr}_auroc_gap'
            if sl[f'{col}_mean'].isna().all():
                continue
            best = sl.loc[sl[f'{col}_mean'].idxmin(), 'architecture']
            for r in sl.itertuples():
                row = {'domain': domain, 'eval_set': eval_set, 'layer': layer, 'probe': pr, 'architecture': r.architecture,
                       'gap_mean': getattr(r, f'{col}_mean'), 'gap_sd': getattr(r, f'{col}_sd'),
                       'nmse_mean': r.nmse_mean, 'probe_informative_on_original_val': informative,
                       'ranking_eligible': eligible and informative}
                if eligible and informative:
                    a = d[(d.domain == domain) & (d.eval_set == eval_set) & (d.layer == layer) & (d.architecture == best)].set_index('seed')[col]
                    b = d[(d.domain == domain) & (d.eval_set == eval_set) & (d.layer == layer) & (d.architecture == r.architecture)].set_index('seed')[col]
                    diff = (b - a).dropna()   # >0: this architecture loses more than the leader
                    row.update(best_architecture=best,
                               rank_by_gap=int(sl[f'{col}_mean'].rank()[sl.architecture == r.architecture].iloc[0]),
                               paired_excess_gap_mean=diff.mean(), paired_excess_gap_sd=diff.std(),
                               seeds_where_leader_better=f'{int((diff > 0).sum())}/{len(diff)}' if r.architecture != best else '',
                               difference_consistent=(r.architecture != best and (diff > 0).all()
                                                      and diff.mean() > 2 * diff.std()))
                rows.append(row)
    return pd.DataFrame(rows)


def relation(d, s, eval_set='test'):
    """How reconstruction quality relates to truth/honesty preservation (descriptive Spearman correlations)."""
    m0 = d[d.matched_10pct & (d.eval_set == eval_set)]
    rows = []

    def add(scope, x, y, frame, note=''):
        f = frame[[x, y]].dropna()
        if len(f) >= 4 and f[x].nunique() > 1 and f[y].nunique() > 1:
            rho, p = spearmanr(f[x], f[y])
            rows.append({'scope': scope, 'x': x, 'y': y, 'n': len(f), 'spearman_rho': rho, 'p_value': p, 'note': note})

    for domain, mdd in m0.groupby('domain'):
        for pr in ('logistic', 'diffmean'):
            g = f'{pr}_auroc_gap'
            md = mdd[mdd[f'{pr}_informative']]
            inf_layers = set(md.layer)
            add(f'{domain}: all informative layers, matched runs', 'nmse', g, md,
                'pooled across layers: mixes the layer effect with the SAE effect')
            add(f'{domain}: all informative layers, matched runs', 'owt_test_nmse', g, md)
            add(f'{domain}: all informative layers, matched runs', 'cosine', g, md)
            add(f'{domain}: all informative layers, matched runs', 'direction_cosine', g, md)
            # within layer: does the better reconstruction (across archs/seeds) preserve more? 9 runs per layer
            for layer, ml in md.groupby('layer'):
                add(f'{domain}: layer {layer} (within layer)', 'nmse', g, ml)
            # layer means
            lm = s[(s.domain == domain) & (s.eval_set == eval_set) & s.layer.isin(inf_layers) & s.cell_all_seeds_matched]
            add(f'{domain}: cell means (layer x arch)', 'nmse_mean', f'{g}_mean', lm)
    return pd.DataFrame(rows)


def style(ax, xlabel=True):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xticks(range(12))
    if xlabel:
        ax.set_xlabel('GPT-2 Small layer (hook_resid_pre, final token)', color=INK2, fontsize=8)


def layer_panels(s, d, col, ylabel, title, path, eval_set='test', original=None, zero_line=False, inf_col='informative_layer'):
    domains = sorted(s.domain.unique())
    fig, axes = plt.subplots(1, len(domains), figsize=(max(9.0, 5.6 * len(domains)), 5.0), facecolor=SURFACE, squeeze=False)
    for ax, domain in zip(axes[0], domains):
        style(ax)
        sd = s[(s.domain == domain) & (s.eval_set == eval_set)]
        inf = sd.groupby('layer')[inf_col].first()
        for L in inf[~inf].index:
            ax.axvspan(L - 0.5, L + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        if original is not None:
            o = sd.groupby('layer')[original].first()
            ax.plot(o.index, o.values, color=INK2, lw=1.5, ls='--', marker='D', ms=4, label='Original activations', zorder=3)
        for i, a in enumerate(ARCHS):
            g = sd[sd.architecture == a].sort_values('layer')
            x = g.layer + (i - 1) * 0.1
            ax.plot(x, g[f'{col}_mean'], color=COLOR[a], lw=2, marker=MARKER[a], ms=6, label=LABEL[a], zorder=4)
            ax.errorbar(x, g[f'{col}_mean'], yerr=g[f'{col}_sd'], fmt='none', ecolor=COLOR[a], elinewidth=1, capsize=2.5, zorder=3)
            um = g[~g.cell_all_seeds_matched]
            ax.scatter(um.layer + (i - 1) * 0.1, um[f'{col}_mean'], s=130, facecolors='none', edgecolors=INK, lw=1.1, zorder=5,
                       label='Cell has an unmatched seed (excluded)' if i == 2 and len(um) else None)
        if zero_line:
            ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f'{domain}', color=INK, fontsize=10, loc='left')
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(0.005, 0.935), fontsize=7.5, frameon=False, labelcolor=INK2, ncol=min(len(lab), 5))
    fig.suptitle(title, x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
    fig.text(0.01, 0.01, textwrap.fill(f'{eval_set} split; mean ± sd over matched seeds (42/1042/2042). Grey bands: layers where '
             'the original probe is not informative (val AUROC < 0.70).', 150), fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def scatter(d, path, eval_set='test'):
    m = d[d.matched_10pct & (d.eval_set == eval_set) & d.informative_layer]
    domains = sorted(m.domain.unique())
    fig, axes = plt.subplots(1, len(domains), figsize=(max(9.0, 5.6 * len(domains)), 5.0), facecolor=SURFACE, squeeze=False)
    for ax, domain in zip(axes[0], domains):
        style(ax, xlabel=False)
        ax.grid(axis='x', color=GRID, lw=0.8)
        md = m[m.domain == domain]
        for a in ARCHS:
            g = md[md.architecture == a]
            ax.scatter(g.nmse, g.logistic_auroc_gap, color=COLOR[a], marker=MARKER[a], s=34, edgecolors=SURFACE, lw=0.8,
                       label=LABEL[a], zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f'Normalized MSE on {domain} {eval_set} activations (1 − EV)', color=INK2, fontsize=8)
        ax.set_ylabel('Frozen logistic probe AUROC gap (original − reconstructed)', color=INK2, fontsize=8)
        ax.set_title(domain, color=INK, fontsize=10, loc='left')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(0.005, 0.935), fontsize=7.5, frameon=False, labelcolor=INK2, ncol=3)
    fig.suptitle('Reconstruction error vs truth-information loss (one point per SAE run)', x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
    fig.text(0.01, 0.01, 'Informative layers only, sparsity-matched runs only. Each point is one layer × architecture × seed.',
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('got', 'repe'), required=True)
    a = ap.parse_args()
    base = DIRS[a.dataset]
    out, fig = base / SUB, base / 'figures'
    fig.mkdir(exist_ok=True)
    d = load(a.dataset)
    s = summarize(d)
    c = pd.concat([compare(d, s, es) for es in d.eval_set.unique()], ignore_index=True)
    r = pd.concat([relation(d, s, es).assign(eval_set=es) for es in d.eval_set.unique()], ignore_index=True)
    s.to_csv(out / 'summary.csv', index=False)
    c.to_csv(out / 'matched_architecture_comparison.csv', index=False)
    r.to_csv(out / 'reconstruction_vs_truth_correlations.csv', index=False)
    d[~d.matched_10pct].to_csv(out / 'excluded_unmatched_runs.csv', index=False)
    tag = 'truth' if a.dataset == 'got' else 'honesty'
    layer_panels(s, d, 'logistic_auroc_reconstructed', 'AUROC', f'Frozen logistic {tag} probe: AUROC on SAE reconstructions vs originals',
                 fig / 'all_layers_frozen_probe_auroc.png', original='logistic_auroc_original')
    layer_panels(s, d, 'logistic_auroc_gap', 'AUROC gap (original − reconstructed)', f'{tag.capitalize()} information lost by reconstruction (frozen logistic probe)',
                 fig / 'all_layers_auroc_gap.png', zero_line=True)
    layer_panels(s, d, 'diffmean_auroc_gap', 'AUROC gap (original − reconstructed)', f'{tag.capitalize()} information lost (frozen mass-mean direction)',
                 fig / 'all_layers_diffmean_auroc_gap.png', zero_line=True, inf_col='diffmean_informative')
    layer_panels(s, d, 'nmse', 'Normalized MSE (1 − EV)', f'Reconstruction error on {a.dataset.upper()} final-token activations',
                 fig / 'all_layers_reconstruction_nmse.png')
    layer_panels(s, d, 'direction_cosine', 'cosine(original, re-estimated)', 'Direction fidelity: mass-mean direction re-estimated on reconstructions',
                 fig / 'all_layers_direction_cosine.png')
    scatter(d, fig / 'reconstruction_vs_truth_scatter.png')
    print(s[s.eval_set == 'test'].groupby(['domain', 'layer', 'architecture'])[['nmse_mean', 'logistic_auroc_gap_mean', 'direction_cosine_mean']].first().round(3).to_string())


if __name__ == '__main__':
    main()
