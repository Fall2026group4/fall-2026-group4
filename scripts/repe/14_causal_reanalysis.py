#!/usr/bin/env python3
"""Reanalysis of the pre-registered RepE causal test (scripts/repe/13_causal_random_controls.py). CPU only.

Run from repository root:
    python scripts/repe/14_causal_reanalysis.py

Reads only the SAVED outputs of the 2026-11-04 test run; no model is run and no test row is re-scored:
    results/repe/steering/causal_random_controls/causal_controls_200.csv      (one row per intervention)
    results/repe/steering/causal_random_controls/causal_summary.csv           (targets, CIs)
    results/repe/steering/causal_random_controls/per_statement_kl_targets.npz (per-statement KL of the targets)

What it adds (all EXPLORATORY: these analyses were not in the preregistration of script 13):
  1. Two-sided empirical p-values. Script 13 tests only "target better than controls". A target that is WORSE
     than nearly every control (E far below the control distribution) is also distinguishable from controls.
        p_upper = (1 + #{controls >= target}) / (1 + n)      (script 13's p, reproduced and checked)
        p_lower = (1 + #{controls <= target}) / (1 + n)
        p_two   = min(1, 2 * min(p_upper, p_lower))
  2. Benjamini-Hochberg adjustment over the family of 27 SAE targets (9 SAEs x {reconstruction direction,
     top-1 feature, top-10 features}), for the one-sided and the two-sided p-values. The raw honesty direction
     is the single primary target of script 13 and is reported unadjusted, separately.
  3. A power statement: the smallest p that n controls can give, the smallest effect that reaches p <= 0.05
     against each control group (minimum detectable effect, MDE), the MDE as a share of the positive control,
     and how many controls a single true effect would need to survive correction over the family.
  4. Paired bootstrap (over statements) of E(raw direction) - E(SAE target): does an SAE direction recover
     less of the persona shift than the raw direction it approximates?
  5. Whether random SAE-feature directions are more disruptive than Gaussian directions (control-group
     distributions compared; Kolmogorov-Smirnov).

Outputs go to results/repe/steering/causal_reanalysis/. Existing outputs are never overwritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('results/repe/steering/causal_random_controls')
OUT = Path('results/repe/steering/causal_reanalysis')
ALPHA = 0.05
N_BOOT = 2000
BOOT_SEED = 0
KEY = {'sae_reconstruction_direction': 'recon_dir', 'sae_top1_feature': 'top1', 'sae_top10_features': 'top10'}


# ---------------------------------------------------------------- pure functions (unit-tested)
def empirical_p_upper(target, controls):
    """Script 13's one-sided p: share of controls at least as large as the target (with the +1 correction)."""
    controls = np.asarray(controls, dtype=float)
    return (1 + int(np.sum(controls >= target))) / (1 + len(controls))


def empirical_p_lower(target, controls):
    controls = np.asarray(controls, dtype=float)
    return (1 + int(np.sum(controls <= target))) / (1 + len(controls))


def empirical_p_two_sided(target, controls):
    return min(1.0, 2 * min(empirical_p_upper(target, controls), empirical_p_lower(target, controls)))


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (same order as the input). NaNs are not allowed."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p.copy()
    if not np.isfinite(p).all():
        raise ValueError('bh_adjust: non-finite p-value')
    m = p.size
    order = np.argsort(p, kind='stable')
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def min_attainable_p(n_controls):
    return 1.0 / (n_controls + 1)


def controls_needed_bonferroni(n_tests, alpha=ALPHA):
    """Smallest n with 1/(n+1) <= alpha/n_tests: below this, no single test can pass Bonferroni."""
    return int(math.ceil(n_tests / alpha - 1 - 1e-9))


def min_floor_discoveries_for_bh(n_tests, n_controls, alpha=ALPHA):
    """Smallest k such that k targets all at the p floor pass BH (k * alpha / n_tests >= floor)."""
    return int(math.ceil(min_attainable_p(n_controls) * n_tests / alpha - 1e-9))


def minimum_detectable_effect(controls, alpha=ALPHA):
    """Smallest target value with one-sided empirical p <= alpha against these controls.

    p <= alpha  <=>  #{controls >= target} <= alpha*(n+1) - 1.  With k = floor(alpha*(n+1) - 1) controls
    allowed above, the target must exceed the (k+1)-th largest control; that control value is returned
    (any target strictly above it is detectable). NaN if even zero controls above cannot reach alpha.
    """
    c = np.sort(np.asarray(controls, dtype=float))[::-1]
    k = math.floor(alpha * (len(c) + 1) - 1 + 1e-9)
    if k < 0:
        return float('nan')
    return float(c[k]) if k < len(c) else float(c[-1])


def recovery(kl_after, kl_before):
    return 1.0 - float(np.sum(kl_after)) / float(np.sum(kl_before))


def paired_bootstrap_difference(kl_a, kl_b, kl_before, n=N_BOOT, seed=BOOT_SEED):
    """E(a) - E(b) with a 95 % bootstrap CI over statements (same resamples for a and b)."""
    kl_a, kl_b, kl_before = (np.asarray(v, dtype=float) for v in (kl_a, kl_b, kl_before))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl_before), size=(n, len(kl_before)))
    diff = (1.0 - kl_a[idx].sum(1) / kl_before[idx].sum(1)) - (1.0 - kl_b[idx].sum(1) / kl_before[idx].sum(1))
    point = recovery(kl_a, kl_before) - recovery(kl_b, kl_before)
    return point, float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- analysis
def analyse(src=SRC):
    df = pd.read_csv(src / 'causal_controls_200.csv')
    summ = pd.read_csv(src / 'causal_summary.csv')
    kls = np.load(src / 'per_statement_kl_targets.npz')
    ctrl = df[df['intervention_kind'] == 'control']
    groups = {g: d['effect_E'].to_numpy() for g, d in ctrl.groupby('control_group')}
    n_controls = {g: len(v) for g, v in groups.items()}
    if len(set(n_controls.values())) != 1:
        raise ValueError(f'control groups differ in size: {n_controls}')
    n = next(iter(n_controls.values()))
    pc = summ[summ['intervention_kind'] == 'positive_control'].iloc[0]
    kl0 = kls['baseline_kl']

    rows = []
    for _, t in summ[summ['intervention_kind'] == 'target'].iterrows():
        c = groups[t['control_group']]
        saved = t['vs_gaussian_p_empirical'] if t['control_group'] == 'gaussian' else t['vs_random_features_p_empirical']
        p_up = empirical_p_upper(t['effect_E'], c)
        if abs(p_up - saved) > 1e-12:
            raise ValueError(f"saved p-value not reproduced for {t['intervention']} {t['control_group']}: {p_up} vs {saved}")
        is_raw = t['intervention'] == 'raw_honesty_direction'
        name = 'raw_direction' if is_raw else f"{t['sae_architecture']}_{t['sae_variant']}__{KEY[t['intervention']]}"
        if is_raw:
            d_pt = d_lo = d_hi = np.nan
        else:
            d_pt, d_lo, d_hi = paired_bootstrap_difference(kls['raw_direction'], kls[name], kl0)
        rows.append({
            'target': name, 'intervention': t['intervention'],
            'sae_architecture': '' if is_raw else t['sae_architecture'], 'sae_variant': '' if is_raw else t['sae_variant'],
            'control_group': t['control_group'], 'n_controls': len(c),
            'effect_E': t['effect_E'], 'E_ci95_low': t['E_ci95_low'], 'E_ci95_high': t['E_ci95_high'],
            'control_mean': c.mean(), 'control_sd': c.std(ddof=1),
            'z_vs_controls': (t['effect_E'] - c.mean()) / c.std(ddof=1),
            'controls_above': int((c >= t['effect_E']).sum()), 'controls_below': int((c <= t['effect_E']).sum()),
            'p_upper_script13': p_up, 'p_lower': empirical_p_lower(t['effect_E'], c),
            'p_two_sided': empirical_p_two_sided(t['effect_E'], c),
            'E_raw_minus_E_target': d_pt, 'diff_ci95_low': d_lo, 'diff_ci95_high': d_hi,
            'cosine_with_raw_direction': t['cosine_with_raw_direction'],
            'family': 'primary' if is_raw else 'sae_27',
        })
    res = pd.DataFrame(rows)
    fam = res['family'] == 'sae_27'
    for col in ('p_upper_script13', 'p_two_sided'):
        res[f'{col}_bh'] = np.nan
        res.loc[fam, f'{col}_bh'] = bh_adjust(res.loc[fam, col].to_numpy())
    res['direction_of_effect'] = np.where(res['effect_E'] > 0, 'towards honest output', 'away from honest output')

    # control groups vs Gaussian
    from scipy.stats import ks_2samp
    g0 = groups['gaussian']
    crow = []
    for g, c in sorted(groups.items()):
        ks = ks_2samp(c, g0) if g != 'gaussian' else None
        crow.append({'control_group': g, 'n': len(c), 'mean_E': c.mean(), 'sd_E': c.std(ddof=1),
                     'q05': np.percentile(c, 5), 'q95': np.percentile(c, 95), 'min': c.min(), 'max': c.max(),
                     'share_negative': float((c < 0).mean()),
                     'mde_alpha_0.05': minimum_detectable_effect(c),
                     'mde_share_of_positive_control': minimum_detectable_effect(c) / pc['effect_E'],
                     'ks_stat_vs_gaussian': np.nan if ks is None else ks.statistic,
                     'ks_p_vs_gaussian': np.nan if ks is None else ks.pvalue})
    cdf = pd.DataFrame(crow)

    m = int(fam.sum())
    power = {
        'n_controls_per_group': n, 'smallest_attainable_p': min_attainable_p(n),
        'family_size_sae_targets': m, 'alpha': ALPHA,
        'bonferroni_threshold': ALPHA / m,
        'single_test_can_pass_bonferroni_with_these_controls': min_attainable_p(n) <= ALPHA / m,
        'controls_needed_for_one_test_to_pass_bonferroni': controls_needed_bonferroni(m),
        'targets_at_p_floor_needed_to_pass_bh': min_floor_discoveries_for_bh(m, n),
        'positive_control_E': float(pc['effect_E']),
        'positive_control_E_ci95': [float(pc['E_ci95_low']), float(pc['E_ci95_high'])],
        'raw_direction_E': float(res.loc[~fam, 'effect_E'].iloc[0]),
        'raw_direction_share_of_positive_control': float(res.loc[~fam, 'effect_E'].iloc[0] / pc['effect_E']),
        'mde_gaussian_alpha_0.05': minimum_detectable_effect(g0),
        'mde_gaussian_share_of_positive_control': minimum_detectable_effect(g0) / float(pc['effect_E']),
        'median_target_ci_half_width': float(((res['E_ci95_high'] - res['E_ci95_low']) / 2).median()),
        'gaussian_control_sd': float(g0.std(ddof=1)),
        'n_test_statements': int(len(kl0)),
        'sae_targets_negative_E': int((res.loc[fam, 'effect_E'] < 0).sum()),
        'sae_targets_two_sided_p_le_0.05': int((res.loc[fam, 'p_two_sided'] <= ALPHA).sum()),
        'sae_targets_two_sided_bh_le_0.05': int((res.loc[fam, 'p_two_sided_bh'] <= ALPHA).sum()),
        'sae_targets_one_sided_p_le_0.05': int((res.loc[fam, 'p_upper_script13'] <= ALPHA).sum()),
        'sae_targets_one_sided_bh_le_0.05': int((res.loc[fam, 'p_upper_script13_bh'] <= ALPHA).sum()),
    }
    return res, cdf, power, groups


def figure(res, groups, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'': '#222222', 'relu': '#1f77b4', 'topk': '#d95f02', 'jumprelu': '#1b9e77'}
    order = res.sort_values(['family', 'intervention', 'sae_architecture', 'sae_variant']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    for i, r in order.iterrows():
        c = groups[r['control_group']]
        lo, hi = np.percentile(c, [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color='#c9ced6', lw=7, solid_capstyle='butt', zorder=1)
        col = colors[r['sae_architecture']]
        ax.plot([r['E_ci95_low'], r['E_ci95_high']], [i, i], color=col, lw=1.4, zorder=2)
        ax.plot(r['effect_E'], i, 'o', color=col, ms=5, zorder=3,
                mfc=col if r['p_two_sided'] <= ALPHA else 'white')
    ax.axvline(0, color='#555555', lw=0.8, ls='--')
    labels = [('raw honesty direction' if r['family'] == 'primary' else
               f"{r['sae_architecture']} / {r['sae_variant']} / {KEY[r['intervention']]}") for _, r in order.iterrows()]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('E = share of the honest-persona output shift recovered (higher is better)')
    ax.set_title('RepE causal test, GPT-2 Small layer 6, 90 test statements\n'
                 'dot + line: target and its 95 % bootstrap CI; grey bar: central 95 % of its 200 controls;\n'
                 'filled dot: two-sided empirical p <= 0.05 (unadjusted)', fontsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if (a.out / 'reanalysis_targets.csv').exists():
        raise SystemExit(f'{a.out}/reanalysis_targets.csv exists; refusing to overwrite.')
    res, cdf, power, groups = analyse()
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / 'reanalysis_targets.csv', index=False)
    cdf.to_csv(a.out / 'control_group_summary.csv', index=False)
    info = {'script': 'scripts/repe/14_causal_reanalysis.py', 'script_sha256': sha256(__file__),
            'inputs_sha256': {f: sha256(SRC / f) for f in ('causal_controls_200.csv', 'causal_summary.csv',
                                                           'per_statement_kl_targets.npz')},
            'status': 'EXPLORATORY (post hoc; not in the preregistration of scripts/repe/13)',
            'bootstrap': f'{N_BOOT} resamples over statements, seed {BOOT_SEED}', **power}
    (a.out / 'power_statement.json').write_text(json.dumps(info, indent=2))
    figure(res, groups, a.out / 'figures' / 'causal_effects_forest.svg')
    cols = ['target', 'effect_E', 'z_vs_controls', 'p_upper_script13', 'p_two_sided', 'p_two_sided_bh',
            'E_raw_minus_E_target', 'diff_ci95_low', 'diff_ci95_high']
    print(res[cols].round(4).to_string(index=False))
    print(cdf.round(5).to_string(index=False))
    print(json.dumps(power, indent=2))


if __name__ == '__main__':
    main()
