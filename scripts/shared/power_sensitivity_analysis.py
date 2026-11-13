#!/usr/bin/env python3
"""Sensitivity (minimum detectable effect) of the existing GoT and M4 comparisons (2026-11-13; remediation S2/S7).

Uses only committed result files. Nothing is re-run, no checkpoint, cache or test row is read:
  * results/shared/m5_confirmatory_test/m5_rows.csv and comparisons.csv  (GoT, TEST, already scored once in M5);
  * results/shared/pca_capacity_matched/got_paired.csv                   (GoT, VALIDATION, paired SAE - PCA CIs);
  * results/repe/sae_patch_decomposition/decomposition_summary.csv       (M4 causal, VALIDATION).

Definitions (two-sided alpha = 0.05, power = 0.80):
  * statement-level SE of a quantity = (95 % grouped-bootstrap CI width) / (2 * 1.96);
  * statement-level MDE = (z_0.975 + z_0.80) * SE = 2.80 * SE;
  * seed-level MDE of a between-family comparison (the R2 rule's Welch test on 3 vs 3 seeds) =
    (t_0.975,df + t_0.80,df) * sqrt(sd_new^2/n_new + sd_ref^2/n_ref), df = Welch df stored in comparisons.csv.
  * The statement-level SE of an SAE-vs-SAE paired difference is approximated by the paired SAE-vs-PCA SE from PCA-CM
    (same statements, same bootstrap). The two reconstructions in an SAE-vs-SAE pair are probably more correlated, so
    this is an approximate (probably conservative) value. It is labelled as such.
  * M4: preservation = T_rec / T_full. Its MDE uses the stored CI. A row is flagged when T_full's CI low is below
    25 % of T_full's mean, because the ratio is then unstable.
These are sensitivity estimates of what the existing designs can resolve, not post-hoc power for observed effects.

Run: python scripts/shared/power_sensitivity_analysis.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/shared/power_sensitivity'
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
log = logging.getLogger('power')


def se_from_ci(lo, hi):
    return (np.asarray(hi, float) - np.asarray(lo, float)) / (2 * stats.norm.ppf(0.975))


def got_tables():
    rows = pd.read_csv(REPO / 'results/shared/m5_confirmatory_test/m5_rows.csv')
    g = rows[rows.got_cities__refit_logistic_gap.notna()].copy()
    g['frozen_se'] = se_from_ci(g.got_cities__frozen_logistic_gap_ci_low, g.got_cities__frozen_logistic_gap_ci_high)
    g['refit_se'] = se_from_ci(g.got_cities__refit_logistic_gap_ci_low, g.got_cities__refit_logistic_gap_ci_high)
    fam = g.groupby(['layer', 'family', 'name']).agg(
        n_seeds=('seed', 'size'), refit_gap_mean=('got_cities__refit_logistic_gap', 'mean'),
        refit_gap_seed_sd=('got_cities__refit_logistic_gap', 'std'), refit_gap_stmt_se=('refit_se', 'mean'),
        frozen_gap_mean=('got_cities__frozen_logistic_gap', 'mean'),
        frozen_gap_seed_sd=('got_cities__frozen_logistic_gap', 'std'), frozen_gap_stmt_se=('frozen_se', 'mean'),
        original_auroc=('got_cities__original_auroc', 'first')).reset_index()
    fam['refit_gap_stmt_mde'] = Z * fam.refit_gap_stmt_se
    fam['frozen_gap_stmt_mde'] = Z * fam.frozen_gap_stmt_se
    comp = pd.read_csv(REPO / 'results/shared/m5_confirmatory_test/comparisons.csv')
    c = comp[comp.metric.str.startswith('got_cities') & (comp.rule == 'R2 (all pairs + Welch)')].copy()
    se = np.sqrt(c.new_sd ** 2 / 3 + c.ref_sd ** 2 / 3)
    tsum = stats.t.ppf(0.975, c.welch_df) + stats.t.ppf(0.80, c.welch_df)
    c['seed_se'] = se
    c['seed_mde'] = tsum * se
    pp = REPO / 'results/shared/pca_capacity_matched/got_paired.csv'
    if pp.exists():
        p = pd.read_csv(pp)
        p['paired_se'] = se_from_ci(p.diff_refit_logistic_gap_ci_low, p.diff_refit_logistic_gap_ci_high)
        stmt = p[p.reference == 'capacity_k128'].groupby('layer').paired_se.median()
        c['stmt_paired_se_approx'] = c.layer.map(stmt)
        c.loc[~c.metric.str.contains('refit'), 'stmt_paired_se_approx'] = np.nan   # only the refit gap was paired
        c['stmt_mde_approx'] = Z * c.stmt_paired_se_approx
        c['combined_mde_approx'] = np.sqrt(c.seed_mde ** 2 + c.stmt_mde_approx ** 2)
    c['abs_diff'] = c['diff'].abs()
    c['detectable_at_seed_mde'] = c.abs_diff >= c.seed_mde
    keep = ['layer', 'comparison', 'metric', 'cls', 'ref_mean', 'new_mean', 'diff', 'ref_sd', 'new_sd', 'welch_df',
            'seed_se', 'seed_mde', 'stmt_paired_se_approx', 'stmt_mde_approx', 'combined_mde_approx', 'abs_diff',
            'detectable_at_seed_mde']
    return fam, c[[k for k in keep if k in c]]


def m4_table():
    d = pd.read_csv(REPO / 'results/repe/sae_patch_decomposition/decomposition_summary.csv')
    d = d[(d.kind == 'persona') & (d.template_set == 'T2_T4')].copy()
    d['pres_se'] = se_from_ci(d.preservation_rec_ci_low, d.preservation_rec_ci_high)
    d['pres_mde'] = Z * d.pres_se
    d['T_full_se'] = se_from_ci(d.T_full_ci_low, d.T_full_ci_high)
    d['T_full_mde_nats'] = Z * d.T_full_se
    d['T_full_ci_low_over_mean'] = d.T_full_ci_low / d.T_full_mean
    d['ratio_unstable_flag'] = d.T_full_ci_low_over_mean < 0.25
    a = d.groupby(['site', 'architecture']).agg(
        n_seeds=('seed', 'size'), n_pairs=('n_pairs', 'first'), T_full_mean=('T_full_mean', 'mean'),
        T_full_ci_low=('T_full_ci_low', 'mean'), T_full_ci_high=('T_full_ci_high', 'mean'),
        T_full_mde_nats=('T_full_mde_nats', 'mean'), T_full_ci_low_over_mean=('T_full_ci_low_over_mean', 'mean'),
        preservation_mean=('preservation_rec', 'mean'), preservation_seed_sd=('preservation_rec', 'std'),
        preservation_ci_halfwidth=('pres_se', lambda s: float(np.mean(s) * stats.norm.ppf(0.975))),
        preservation_stmt_mde=('pres_mde', 'mean'), ratio_unstable_any_seed=('ratio_unstable_flag', 'any')).reset_index()
    return a


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for f in ('got_family_sensitivity.csv', 'got_comparison_mde.csv', 'm4_causal_sensitivity.csv', 'summary.json'):
        if (OUT / f).exists():
            raise SystemExit(f'{OUT / f} exists; refusing to overwrite.')
    OUT.mkdir(parents=True, exist_ok=True)
    fam, comp = got_tables()
    m4 = m4_table()
    fam.to_csv(OUT / 'got_family_sensitivity.csv', index=False)
    comp.to_csv(OUT / 'got_comparison_mde.csv', index=False)
    m4.to_csv(OUT / 'm4_causal_sensitivity.csv', index=False)
    trained = fam[fam.name != 'topk_frozen_decoder']
    s = dict(
        z_sum=float(Z),
        got_refit_gap_stmt_mde_range=[float(trained.refit_gap_stmt_mde.min()), float(trained.refit_gap_stmt_mde.max())],
        got_frozen_gap_stmt_mde_range=[float(trained.frozen_gap_stmt_mde.min()), float(trained.frozen_gap_stmt_mde.max())],
        got_refit_gap_seed_sd_range=[float(trained.refit_gap_seed_sd.min()), float(trained.refit_gap_seed_sd.max())],
        got_comparison_seed_mde_range=[float(comp.seed_mde.min()), float(comp.seed_mde.max())],
        got_comparisons_detectable_at_seed_mde=int(comp.detectable_at_seed_mde.sum()),
        got_comparisons_total=int(len(comp)),
        m4_preservation_stmt_mde_range=[float(m4.preservation_stmt_mde.min()), float(m4.preservation_stmt_mde.max())],
        m4_T_full_range_nats=[float(m4.T_full_mean.min()), float(m4.T_full_mean.max())],
        m4_ratio_unstable_rows=int(m4.ratio_unstable_any_seed.sum()))
    if 'combined_mde_approx' in comp:
        r = comp[comp.metric == 'got_cities__refit_logistic_gap']
        s['got_refit_comparison_combined_mde_approx_range'] = [float(r.combined_mde_approx.min()),
                                                               float(r.combined_mde_approx.max())]
    (OUT / 'summary.json').write_text(json.dumps(s, indent=2))
    log.info(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
