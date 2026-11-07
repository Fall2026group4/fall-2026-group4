#!/usr/bin/env python3
"""Validation-only evaluation and analysis of the layer-6 TopK KL-weight ablation (script 20).

Run from repository root:  python scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py

Evaluates 15 SAEs in one context with scripts/shared/layer6_faithfulness_eval.py: base, w=0 (Method A), w=0.25,
w=0.50 (new) and w=1 (Method B), seeds 42/1042/2042. Base/A/B rows must equal the pre-training reproduction
(verification) within 1e-4. Applies the pre-registered criteria; writes tables, criteria and figures.
Outputs: results/geometry_of_truth/kl_weight_ablation_layer6_topk/{per_sae.csv, summary.csv, paired_differences.csv,
criteria.json, eval_info.json}; figures results/geometry_of_truth/figures/kl_weight_ablation_*.png
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402

W = [0.0, 0.25, 0.5, 1.0]
WNAME = {0.0: 'A (w=0)', 0.25: 'w=0.25', 0.5: 'w=0.50', 1.0: 'B (w=1)'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
SEEDC = {42: '#2a78d6', 1042: '#eb6834', 2042: '#1baf7a'}      # reference palette slots 1-3 (identity = seed)
SEEDM = {42: 'o', 1042: 's', 2042: '^'}
PRIMARY = ['got_geographic_indep_mlp_gap_mean', 'splice_owt_kl', 'splice_owt_delta_ce']
SECONDARY = ['got_numerical_indep_mlp_gap_mean', 'got_geographic_indep_logistic_gap', 'got_numerical_indep_logistic_gap',
             'repe_logistic_gap', 'repe_diffmean_gap', 'repe_direction_cosine', 'splice_repe_persona_shift_cosine',
             'owt_val_mse', 'owt_val_ev', 'owt_val_cosine', 'owt_val_dead_fraction', 'downstream_nmse_mean',
             'splice_got_statement_kl', 'splice_got_cities_pref_change', 'got_geographic_val_relative_sq_error']
L0 = ['owt_val_l0', 'got_geographic_val_l0', 'got_numerical_val_l0', 'repe_val_l0']


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def consistent_lower(d):
    return bool((d < 0).all() and -d.mean() > 2 * d.std())


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    s20 = load('scripts/geometry_of_truth/20_kl_weight_ablation.py', 's20')
    ev6 = load('scripts/shared/layer6_faithfulness_eval.py', 'ev6')
    OUT = s20.OUT
    if (OUT / 'per_sae.csv').exists():
        raise SystemExit('per_sae.csv exists; refusing to overwrite.')
    base, A, B, new = s20.paths()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    ctx = ev6.build_context('cuda')
    items = [('base', None, s, base[s]) for s in s20.SEEDS] + [('A', 0.0, s, A[s]) for s in s20.SEEDS]
    items += [(f'w{f:.2f}', f, s, new[(f, s)]) for f in s20.FRACTIONS for s in s20.SEEDS] + [('B', 1.0, s, B[s]) for s in s20.SEEDS]
    rows = []
    for name, w, s, p in items:
        sae, _ = fm.load_sae(p, 'cuda', architecture='topk')
        sae.eval()
        rows.append({'variant': name, 'kl_fraction': w, 'seed': s, 'checkpoint': Path(p).relative_to(REPO).as_posix(),
                     'checkpoint_sha256': sha(p), **ev6.evaluate_sae(sae, ctx)})
        print(f'{name} seed {s} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_sae.csv', index=False)
    metrics = [c for c in df.columns if c not in ('variant', 'kl_fraction', 'seed', 'checkpoint', 'checkpoint_sha256')]
    # consistency with the pre-training reproduction of base/A/B
    rep = pd.read_csv(OUT / 'reproduction_of_saved_results.csv').set_index(['method', 'seed'])
    cur = df[df.variant.isin(['base', 'A', 'B'])].set_index(['variant', 'seed'])[metrics]
    cons = float((cur.values - rep.loc[cur.index, metrics].values).__abs__().max())

    summ = df.groupby('variant', sort=False)[metrics].agg(['mean', 'std'])
    summ.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False)
    piv = {v: df[df.variant == v].set_index('seed')[metrics] for v in df.variant.unique()}
    pd_rows = []
    for v in ('w0.25', 'w0.50', 'B'):
        for ref in ('A', 'B') if v != 'B' else ('A',):
            d = piv[v] - piv[ref]
            for m in metrics:
                pd_rows.append({'comparison': f'{v} - {ref}', 'metric': m, 'mean': d[m].mean(), 'sd': d[m].std(),
                                **{f'seed{s}': d[m][s] for s in s20.SEEDS}})
    pd.DataFrame(pd_rows).to_csv(OUT / 'paired_differences.csv', index=False)

    gap, kl, ce = PRIMARY
    a_, b_ = piv['A'], piv['B']
    endpoint_truth_diff = float((b_[gap] - a_[gap]).mean())
    crit = {'consistency_with_verification_max_abs_diff': cons, 'endpoint_mean_gap_B_minus_A': endpoint_truth_diff,
            'endpoints_differ_in_truth': endpoint_truth_diff > 0.02}
    for v in ('w0.25', 'w0.50'):
        M = piv[v]
        behav = consistent_lower(M[kl] - a_[kl]) and consistent_lower(M[ce] - a_[ce])
        tni = float((M[gap] - a_[gap]).mean()) <= 0.02
        chord = a_[gap] + (b_[gap] - a_[gap]) * (a_[kl] - M[kl]) / (a_[kl] - b_[kl])
        trade = bool((M[gap] < chord).all())
        guards = {'l0_128_all_datasets': bool(all((M[c] == 128.0).all() for c in L0)),
                  'owt_ev_ge_A_minus_0.02': bool((M['owt_val_ev'] >= a_['owt_val_ev'] - 0.02).all()),
                  'cities_ge_A_minus_0.018': bool(M['splice_got_cities_pref_change'].mean() >= a_['splice_got_cities_pref_change'].mean() - 0.018),
                  'repe_diffmean_le_A_plus_0.02': bool(M['repe_diffmean_gap'].mean() <= a_['repe_diffmean_gap'].mean() + 0.02),
                  'geo_logistic_le_A_plus_0.02': bool(M['got_geographic_indep_logistic_gap'].mean() <= a_['got_geographic_indep_logistic_gap'].mean() + 0.02)}
        if not crit['endpoints_differ_in_truth']:
            verdict = 'INCONCLUSIVE (endpoints do not differ in truth by > 0.02; no measurable trade-off to improve)'
        elif behav and tni and trade and all(guards.values()):
            verdict = 'IMPROVES TRADE-OFF'
        else:
            verdict = 'DOES NOT IMPROVE TRADE-OFF'
        crit[v] = {'BEHAV': bool(behav), 'TRUTH_NONINFERIOR': bool(tni), 'TRADE_below_chord_all_seeds': trade, 'guards': guards,
                   'verdict': verdict,
                   'per_seed': {f'seed{s}': {'gap': float(M[gap][s]), 'gap_A': float(a_[gap][s]), 'gap_B': float(b_[gap][s]),
                                             'chord_gap': float(chord[s]), 'owt_kl': float(M[kl][s]), 'kl_A': float(a_[kl][s]),
                                             'kl_B': float(b_[kl][s])} for s in s20.SEEDS},
                   'fraction_of_B_gain_retained': {
                       'owt_kl': ((a_[kl] - M[kl]) / (a_[kl] - b_[kl])).round(3).to_dict(),
                       'owt_ce': ((a_[ce] - M[ce]) / (a_[ce] - b_[ce])).round(3).to_dict(),
                       'got_statement_kl': ((a_['splice_got_statement_kl'] - M['splice_got_statement_kl'])
                                            / (a_['splice_got_statement_kl'] - b_['splice_got_statement_kl'])).round(3).to_dict()},
                   'mean_paired_diff_vs_A': {m: float((M[m] - a_[m]).mean()) for m in PRIMARY},
                   'sd_paired_diff_vs_A': {m: float((M[m] - a_[m]).std()) for m in PRIMARY}}
    (OUT / 'criteria.json').write_text(json.dumps(crit, indent=2))

    # ---- figures: truth gap vs output KL (seed-level, paired lines); behaviour and reconstruction vs w
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    style(ax)
    for s in s20.SEEDS:
        pts = [(piv[v][kl][s], piv[v][gap][s]) for v in ('A', 'w0.25', 'w0.50', 'B')]
        ax.plot(*zip(*pts), color=SEEDC[s], lw=1.2, alpha=0.7)
        ax.plot([pts[0][0], pts[3][0]], [pts[0][1], pts[3][1]], color=SEEDC[s], lw=0.8, ls=':', alpha=0.8)
        for (x, y), v in zip(pts, ('A', 'w0.25', 'w0.50', 'B')):
            ax.scatter(x, y, color=SEEDC[s], marker=SEEDM[s], s=46, zorder=3, edgecolors=SURFACE, lw=0.8,
                       label=f'seed {s}' if v == 'A' else None)
            if s == 42:
                ax.annotate(WNAME[{'A': 0.0, 'w0.25': .25, 'w0.50': .5, 'B': 1.0}[v]], (x, y), xytext=(5, 6),
                            textcoords='offset points', fontsize=7.5, color=INK2)
    mean_pts = [(piv[v][kl].mean(), piv[v][gap].mean()) for v in ('A', 'w0.25', 'w0.50', 'B')]
    ax.plot(*zip(*mean_pts), color=INK, lw=2, marker='D', ms=6, label='mean of 3 seeds', zorder=4)
    ax.set_xlabel('OWT validation output KL(original || spliced), nats/token (lower = better behaviour)', color=INK2, fontsize=8)
    ax.set_ylabel('GoT geographic independent MLP AUROC gap (lower = better)', color=INK2, fontsize=8)
    ax.set_title('KL-weight ablation, layer 6 TopK: truth gap vs output KL\n(dotted: each seed\'s A-B chord)', color=INK,
                 fontsize=10, loc='left')
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(REPO / 'results/geometry_of_truth/figures/kl_weight_ablation_truth_vs_kl.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)

    panels = [(kl, 'OWT output KL ↓'), (ce, 'OWT spliced CE increase ↓'), ('splice_got_statement_kl', 'GoT statement KL ↓'),
              (gap, 'GoT geo MLP gap ↓'), ('owt_val_ev', 'OWT validation EV ↑'), ('downstream_nmse_mean', 'Downstream NMSE L7-11 ↓')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for ax, (m, t) in zip(axes.flat, panels):
        style(ax)
        for s in s20.SEEDS:
            ax.plot(W, [piv[v][m][s] for v in ('A', 'w0.25', 'w0.50', 'B')], color=SEEDC[s], marker=SEEDM[s], lw=1.2, ms=5,
                    label=f'seed {s}')
        ax.plot(W, [piv[v][m].mean() for v in ('A', 'w0.25', 'w0.50', 'B')], color=INK, lw=2, marker='D', ms=5, label='mean')
        ax.set_xticks(W)
        ax.set_xticklabels(['A\n(0)', '0.25', '0.50', 'B\n(1)'], fontsize=8, color=INK2)
        ax.set_title(t, color=INK, fontsize=9.5, loc='left')
    axes[0][0].legend(fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.suptitle('KL-weight ablation (layer 6 TopK, L0 = 128): metrics vs KL fraction of Method B (validation)', x=0.01,
                 ha='left', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPO / 'results/geometry_of_truth/figures/kl_weight_ablation_metrics_vs_w.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)
    (OUT / 'eval_info.json').write_text(json.dumps({'script_sha256': sha(__file__), 'data': 'validation only', 'test_rows_used': False,
                                                    'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                    'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('BEHAV', 'TRUTH_NONINFERIOR', 'TRADE_below_chord_all_seeds', 'verdict')}) for k, v in crit.items()}, indent=2))


if __name__ == '__main__':
    main()
