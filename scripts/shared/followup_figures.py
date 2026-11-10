#!/usr/bin/env python3
"""Figures and summary tables for the 2026-11-10 follow-up experiments. CPU only; reads saved results.

Run from repository root after the experiments:
    python scripts/shared/followup_figures.py
    python scripts/shared/followup_figures.py --root results --out results/shared/followup_figures   (defaults)

Each figure is made only if its inputs exist; missing inputs are listed, never invented.
  F1 graded_positive_control      results/repe/steering/sae_in_the_loop/graded_positive_control.csv
  F2 latent_mediation             results/repe/steering/sae_in_the_loop/summary.csv (part B)
  F3 repe_direction_steering      results/repe/steering/sae_in_the_loop/summary.csv (part C)
  F4 probe_gap_vs_random_error    results/shared/frozen_probe_error_decomposition/*/per_run.csv
  F5 truth_gap_vs_prompt_l0       results/shared/sparsity_matched_truth_preservation/{got,repe}/per_run.csv
  F6 got_causal_steering          results/geometry_of_truth/causal_truth_steering/summary.csv
  F7 loss_recovered_public_vs_inhouse
                                  results/shared/public_sae_spliced_evaluation/*/{per_run,layer_controls}.csv and
                                  results/openwebtext/spliced_evaluation/{per_run,layer_controls}.csv
Writes <out>/<name>.svg and .png, and <out>/tables.md (numbers rounded from the CSVs). Refuses to overwrite.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARCH_COLORS = {'relu': '#1f77b4', 'topk': '#d95f02', 'jumprelu': '#1b9e77',
               'public_relu': '#6a3d9a', 'public_topk': '#b15928'}
ARCHS = ('relu', 'topk', 'jumprelu')


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'svg.fonttype': 'none'})
    return plt


def save(fig, out, name, made):
    fig.tight_layout()
    fig.savefig(out / f'{name}.svg')
    fig.savefig(out / f'{name}.png', dpi=160)
    made.append(name)


def arch_legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=ARCH_COLORS.get(a, '#222222'), label=a or 'raw direction')
         for a in ('',) + ARCHS]
    ax.legend(handles=h, frameon=False, fontsize=7, loc='best')


def seed_mean(df, keys, col):
    g = df.groupby(keys)[col]
    return g.mean().rename('mean').to_frame().join(g.std(ddof=1).rename('sd')).join(g.size().rename('n')).reset_index()


# ---------------------------------------------------------------- figures
def f1_graded(root, out, plt, made, tables):
    p = root / 'repe/steering/sae_in_the_loop/graded_positive_control.csv'
    if not p.exists():
        return [str(p)]
    d = pd.read_csv(p)
    layers = sorted(d['layer'].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3), sharey=True, squeeze=False)
    for ax, L in zip(axes[0], layers):
        s = d[d['layer'] == L].sort_values('lam')
        ax.plot(s['lam'], s['effect_E'], 'o-', color='#14213d', label='activation patch x lambda')
        ax.plot(s['lam'], s['control_q95'], '--', color='#9aa3ad', label='95th pct of random controls')
        mdd = s['minimum_detectable_dose'].iloc[0]
        if np.isfinite(mdd):
            ax.axvline(mdd, color='#d62728', lw=0.8)
        ax.set_xscale('log')
        ax.set_title(f'layer {L}; detectable from lambda = {mdd:g}')
        ax.set_xlabel('lambda (share of the full patch)')
    axes[0][0].set_ylabel('E (share of honest shift recovered)')
    axes[0][0].legend(frameon=False, fontsize=8)
    save(fig, out, 'F1_graded_positive_control', made)
    tables.append(('Graded positive control (RepE, SAE-in-the-loop script)',
                   d[['layer', 'lam', 'effect_E', 'control_q95', 'p_upper', 'minimum_detectable_dose']]))
    return []


def f2_mediation(root, out, plt, made, tables):
    p = root / 'repe/steering/sae_in_the_loop/summary.csv'
    if not p.exists():
        return [str(p)]
    s = pd.read_csv(p)
    b = s[s['part'] == 'B'].drop_duplicates(['layer', 'sae_architecture', 'sae_seed', 'intervention']).copy()
    if b.empty or 'share_of_activation_patch' not in b:
        return [f'{p} (no part B rows)']
    order = [i for i in ('latent_all', 'error_term', 'latent_top1', 'latent_top10', 'latent_top50') if i in set(b['intervention'])]
    agg = seed_mean(b, ['layer', 'sae_architecture', 'intervention'], 'share_of_activation_patch')
    layers = sorted(agg['layer'].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.2), sharey=True, squeeze=False)
    w = 0.8 / len(ARCHS)
    for ax, L in zip(axes[0], layers):
        for k, arch in enumerate(ARCHS):
            a = agg[(agg['layer'] == L) & (agg['sae_architecture'] == arch)].set_index('intervention').reindex(order)
            x = np.arange(len(order)) + (k - 1) * w
            ax.bar(x, a['mean'], w, yerr=a['sd'].fillna(0), color=ARCH_COLORS[arch], label=arch, capsize=2)
        ax.axhline(0, color='#555555', lw=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace('latent_', '') for o in order], rotation=30)
        ax.set_title(f'layer {L}')
    axes[0][0].set_ylabel('share of the full activation patch\n(mean +- sd over 3 SAE seeds)')
    axes[0][0].legend(frameon=False, fontsize=8)
    save(fig, out, 'F2_latent_mediation', made)
    t = s[s['part'] == 'B'][['layer', 'intervention', 'sae_architecture', 'sae_seed', 'control_group_kind', 'effect_E',
                             'share_of_activation_patch', 'p_upper', 'p_bh']]
    tables.append(('SAE-in-the-loop latent patching (RepE)', t))
    return []


def f3_steering(root, out, plt, made, tables):
    p = root / 'repe/steering/sae_in_the_loop/summary.csv'
    if not p.exists():
        return [str(p)]
    s = pd.read_csv(p)
    c = s[s['part'] == 'C'].copy()
    if c.empty:
        return [f'{p} (no part C rows)']
    c = c.drop_duplicates(['layer', 'intervention', 'sae_architecture', 'sae_seed', 'sign'])
    c['sae_architecture'] = c['sae_architecture'].fillna('')
    agg = seed_mean(c, ['layer', 'intervention', 'sae_architecture', 'sign'], 'effect_E')
    layers = sorted(agg['layer'].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.8 * len(layers), 3.4), sharey=True, squeeze=False)
    labels = []
    for ax, L in zip(axes[0], layers):
        a = agg[agg['layer'] == L]
        keys = [('raw_honesty_direction', '')] + [(i, ar) for i in ('sae_reconstruction_direction', 'sae_top1_direction',
                                                                    'sae_top10_direction') for ar in ARCHS]
        labels = []
        for k, (i, ar) in enumerate(keys):
            for sign, mk in ((1, 'o'), (-1, 'x')):
                r = a[(a['intervention'] == i) & (a['sae_architecture'] == ar) & (a['sign'] == sign)]
                if len(r):
                    ax.errorbar(r['mean'], [k], xerr=r['sd'].fillna(0), fmt=mk, color=ARCH_COLORS.get(ar, '#222222'))
            labels.append('raw direction' if i.startswith('raw') else f"{ar} {i.replace('sae_', '').replace('_direction', '')}")
        ax.axvline(0, color='#555555', lw=0.6, ls='--')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f'layer {L}  (o: +s, x: -s)')
        ax.set_xlabel('E')
    arch_legend(axes[0][0])
    save(fig, out, 'F3_repe_direction_steering', made)
    tables.append(('Direction steering, both signs (RepE)', c[['layer', 'intervention', 'sae_architecture', 'sae_seed',
                                                               'sign', 'effect_E', 'p_two_sided', 'p_bh']]))
    return []


def f4_probe(root, out, plt, made, tables):
    base = root / 'shared/frozen_probe_error_decomposition'
    runs = {p.parent.name: pd.read_csv(p) for p in sorted(base.glob('*/per_run.csv'))} if base.exists() else {}
    if not runs:
        return [str(base / '*/per_run.csv')]
    allr = pd.concat([d.assign(run=k) for k, d in runs.items()], ignore_index=True)
    domains = sorted(allr['domain'].unique())
    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 3.3), sharey=True, squeeze=False)
    for ax, dom in zip(axes[0], domains):
        d = allr[allr['domain'] == dom]
        for arch, g in d.groupby('architecture'):
            m = g.groupby('layer')[['logistic_auroc_gap', 'logistic_noise_gap_mean']].mean()
            col = ARCH_COLORS.get(arch, '#444444')
            ax.plot(m.index, m['logistic_auroc_gap'], 'o-', color=col, label=f'{arch}: SAE error')
            ax.plot(m.index, m['logistic_noise_gap_mean'], ':', color=col, label=f'{arch}: random error, same size')
        ax.set_title(dom)
        ax.set_xlabel('layer')
    axes[0][0].set_ylabel('frozen-probe AUROC gap (lower is better)')
    axes[0][0].legend(frameon=False, fontsize=7)
    save(fig, out, 'F4_probe_gap_vs_random_error', made)
    cols = ['run', 'domain', 'layer', 'architecture', 'logistic_auroc_gap', 'logistic_noise_gap_mean', 'logistic_excess_gap',
            'logistic_signal_retention', 'logistic_noise_inflation', 'logistic_error_anisotropy', 'informative_layer']
    t = allr[[c for c in cols if c in allr]].groupby(['run', 'domain', 'layer', 'architecture'], as_index=False).mean(numeric_only=True)
    tables.append(('Frozen-probe gap: SAE error vs random error of the same size (validation)', t))
    return []


def f5_sparsity(root, out, plt, made, tables):
    base = root / 'shared/sparsity_matched_truth_preservation'
    files = [p for p in (base / 'got/per_run.csv', base / 'repe/per_run.csv') if p.exists()]
    if not files:
        return [str(base / '{got,repe}/per_run.csv')]
    d = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    inf = d[d['informative_layer']]
    show_layers = [L for L in (6, 10) if L in set(inf['layer'])] or sorted(inf['layer'].unique())[:2]
    domains = sorted(inf['domain'].unique())
    fig, axes = plt.subplots(len(show_layers), len(domains), figsize=(3.6 * len(domains), 3 * len(show_layers)),
                             squeeze=False, sharex=True)
    for i, L in enumerate(show_layers):
        for j, dom in enumerate(domains):
            ax = axes[i][j]
            s = inf[(inf['layer'] == L) & (inf['domain'] == dom)]
            for arch in ARCHS:
                a = s[s['architecture'] == arch]
                tr = a[a['truncation'] == 'top_m'].groupby('m')[['l0', 'logistic_auroc_gap']].mean()
                nat = a[a['truncation'] == 'natural'][['l0', 'logistic_auroc_gap']].mean()
                ax.plot(tr['l0'], tr['logistic_auroc_gap'], '-', color=ARCH_COLORS[arch], label=arch)
                ax.plot([nat['l0']], [nat['logistic_auroc_gap']], '*', ms=10, color=ARCH_COLORS[arch])
            ax.set_xscale('log')
            ax.set_title(f'{dom}, layer {L}')
            if i == len(show_layers) - 1:
                ax.set_xlabel('prompt L0 (latents per token)')
            if j == 0:
                ax.set_ylabel('frozen-probe AUROC gap')
    axes[0][0].legend(frameon=False, fontsize=8, title='star: natural L0', title_fontsize=7)
    save(fig, out, 'F5_truth_gap_vs_prompt_l0', made)
    for p in files:
        mc = p.parent / 'matched_comparison.csv'
        if mc.exists():
            m = pd.read_csv(mc)
            if len(m):
                m = m[(m['metric'] == 'logistic_auroc_gap') & m['informative_layer']]
                tables.append((f'Matched-L0 comparison ({p.parent.name}, logistic gap, informative layers)',
                               m[['domain', 'layer', 'm', 'comparison', 'mean_diff', 'sd_diff', 'all_seeds_agree']]))
    return []


def f6_got(root, out, plt, made, tables):
    p = root / 'geometry_of_truth/causal_truth_steering/summary.csv'
    if not p.exists():
        return [str(p)]
    s = pd.read_csv(p)
    s['sae_architecture'] = s['sae_architecture'].fillna('')
    c = s[(s['part'] == 'C')]
    agg = seed_mean(c, ['layer', 'intervention', 'sae_architecture', 'sign'], 'effect')
    layers = sorted(agg['layer'].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.6), sharey=True, squeeze=False)
    for ax, L in zip(axes[0], layers):
        a = agg[agg['layer'] == L]
        keys = [('mass_mean_direction', ''), ('logistic_probe_direction', '')] + [
            (i, ar) for i in ('sae_reconstruction_direction', 'sae_top1_direction', 'sae_top10_direction') for ar in ARCHS]
        labels = []
        for k, (i, ar) in enumerate(keys):
            for sign, mk in ((1, 'o'), (-1, 'x')):
                r = a[(a['intervention'] == i) & (a['sae_architecture'] == ar) & (a['sign'] == sign)]
                if len(r):
                    ax.errorbar(r['mean'], [k], xerr=r['sd'].fillna(0), fmt=mk, color=ARCH_COLORS.get(ar, '#222222'))
            labels.append(i.replace('_direction', '') if not ar else f"{ar} {i.replace('sae_', '').replace('_direction', '')}")
        q = s[(s['layer'] == L) & (s['part'] == 'C') & (s['control_group'].str.endswith('__gaussian'))]['control_q95']
        if len(q):
            ax.axvline(q.iloc[0], color='#9aa3ad', ls='--', lw=0.8)
        ax.axvline(0, color='#555555', lw=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f'layer {L}  (o: +s, x: -s)')
        ax.set_xlabel('change in log P(true) - log P(false)')
    arch_legend(axes[0][0])
    save(fig, out, 'F6_got_causal_steering', made)
    tables.append(('Geometry of Truth causal steering (cities, test pairs)',
                   s[['layer', 'part', 'intervention', 'sae_architecture', 'sae_seed', 'sign', 'detail', 'effect',
                      'ci95_low', 'ci95_high', 'p_upper', 'p_two_sided', 'p_bh']]))
    return []


def _loss_recovered(per_run, controls):
    mean_ce = controls[controls['intervention'] == 'mean_ablation'].set_index('layer')['owt__ce_spliced']
    r = per_run[per_run['intervention'] == 'sae_reconstruction'].copy()
    r['loss_recovered_pct'] = 100 * (r['layer'].map(mean_ce) - r['owt__ce_spliced']) / (r['layer'].map(mean_ce) - r['owt__ce_original'])
    return r


def f7_public(root, out, plt, made, tables):
    inh = root / 'openwebtext/spliced_evaluation'
    pub = sorted((root / 'shared/public_sae_spliced_evaluation').glob('*/per_run.csv'))
    if not pub or not (inh / 'per_run.csv').exists():
        return [str(root / 'shared/public_sae_spliced_evaluation/*/per_run.csv')]
    frames = []
    a = _loss_recovered(pd.read_csv(inh / 'per_run.csv'), pd.read_csv(inh / 'layer_controls.csv'))
    frames.append(a.assign(group=a['architecture']))
    for p in pub:
        b = _loss_recovered(pd.read_csv(p), pd.read_csv(p.parent / 'layer_controls.csv'))
        frames.append(b.assign(group=p.parent.name))
    d = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for g, s in d.groupby('group'):
        m = s.groupby('layer')['loss_recovered_pct'].mean()
        ax.plot(m.index, m.values, 'o-' if g.startswith('public') else '.-', color=ARCH_COLORS.get(g, '#444444'), label=g)
    ax.set_xlabel('layer')
    ax.set_ylabel('OWT loss recovered vs mean ablation (%)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Public SAEs are not sparsity-matched: external reference only', fontsize=8)
    save(fig, out, 'F7_loss_recovered_public_vs_inhouse', made)
    t = d.groupby(['group', 'layer'], as_index=False)[['loss_recovered_pct', 'owt__kl_mean']].mean()
    tables.append(('OWT loss recovered vs mean ablation, public and in-house SAEs (validation)', t))
    return []


def write_tables(tables, path):
    lines = ['# Follow-up results: tables generated from the saved CSVs', '',
             'Every number below is read from a results file; nothing is typed by hand.', '']
    for title, df in tables:
        lines += [f'## {title}', '']
        d = df.copy()
        for c in d.columns:
            if d[c].dtype.kind == 'f':
                d[c] = d[c].round(4)
        lines.append('| ' + ' | '.join(map(str, d.columns)) + ' |')
        lines.append('|' + '---|' * len(d.columns))
        for _, r in d.iterrows():
            lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r.values) + ' |')
        lines.append('')
    path.write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, default=Path('results/shared/followup_figures'))
    a = ap.parse_args()
    if (a.out / 'tables.md').exists():
        raise SystemExit(f'{a.out}/tables.md exists; refusing to overwrite (use a new --out).')
    a.out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    made, missing, tables = [], [], []
    for f in (f1_graded, f2_mediation, f3_steering, f4_probe, f5_sparsity, f6_got, f7_public):
        missing += f(a.root, a.out, plt, made, tables)
        plt.close('all')
    write_tables(tables, a.out / 'tables.md')
    print('figures made:', ', '.join(made) if made else 'none')
    if missing:
        print('inputs not found (figure skipped):\n  ' + '\n  '.join(missing))


if __name__ == '__main__':
    main()
