#!/usr/bin/env python3
"""TSP pilot analysis: the pre-registered decision rule (2026-11-14). Reads only pilot/eval outputs.

Paired by seed (every arm starts from the same R2 TopK seed). For arm X vs reference R and metric m (lower = better
unless stated):
  d_s = m(X, s) - m(R, s), s in {42, 1042, 2042}
  se_seed = sd(d) / sqrt(3)
  v_stmt  = variance over the shared grouped-bootstrap replicates (validation cities) of the seed-mean difference
            (available for the gap metrics of stage 1)
  se      = sqrt(se_seed^2 + v_stmt);  df = 2 (conservative: n_seeds - 1);  95 % CI = mean(d) +/- t(0.975, 2) * se
Co-primary: frozen_logistic_gap and refit_logistic_gap (stage 1, group-B readers), Holm over the 2 for tsp vs ctrl.
A metric passes when 3/3 seed differences favour X, the CI excludes 0 in X's favour and Holm p < 0.05.
Outputs: pilot/analysis/ (refuses to overwrite).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2] / 'results/geometry_of_truth/truth_subspace_preservation'
SEEDS = (42, 1042, 2042)
PRIMARY = ('frozen_logistic_gap', 'refit_logistic_gap')
SECONDARY_GOT = ('refit_mlp_gap', 'frozen_massmean_gap', 'refit_sparse_z_gap', 'refit_knn_gap', 'error_term_auroc',
                 'task_nmse')
BOOT_KEYS = ('frozen_logistic_gap', 'refit_logistic_gap', 'frozen_massmean_gap')
LADDER = ('owt_nmse', 'splice_delta_ce', 'splice_kl_mean', 'splice_sae_vs_noise_delta_ce_ratio', 'owt_l0',
          'repe_persona__refit_logistic_gap', 'repe_persona__frozen_logistic_gap', 'repe_neutral__refit_logistic_gap',
          'rlm__pair_dir_cosine', 'rlm__pair_norm_retention', 'm4__preservation_rec', 'repe_persona__task_l0',
          'got_cities__refit_logistic_gap')
HIGHER_BETTER = {'rlm__pair_dir_cosine', 'm4__preservation_rec'}
COMPARISONS = [('tsp', 'ctrl'), ('tsp', 'rand'), ('tsp', 'pca'), ('ctrl', 'base'), ('ctrl', 'r4mse'),
               ('rand', 'ctrl'), ('pca', 'ctrl')]
FAM = dict(tsp='tsp_tsp', ctrl='tsp_ctrl', rand='tsp_rand', pca='tsp_pca', base='r2_base', r4mse='r4_mse')
T2 = stats.t.ppf(0.975, 2)


def compare(vals_x, vals_r, boot_x=None, boot_r=None, higher_better=False):
    d = np.asarray(vals_x) - np.asarray(vals_r)
    se_seed = d.std(ddof=1) / np.sqrt(len(d))
    v_stmt = float(np.var(np.mean(boot_x, 0) - np.mean(boot_r, 0), ddof=1)) if boot_x is not None else 0.0
    se = float(np.sqrt(se_seed ** 2 + v_stmt))
    m = float(d.mean())
    lo, hi = m - T2 * se, m + T2 * se
    p = float(2 * stats.t.sf(abs(m) / se, 2)) if se > 0 else 0.0
    better = (d > 0) if higher_better else (d < 0)
    ok_ci = (lo > 0) if higher_better else (hi < 0)
    return dict(mean_diff=m, ci_low=lo, ci_high=hi, se_seed=float(se_seed), v_stmt=v_stmt, p=p,
                seeds_better=int(better.sum()), ci_excludes_0_favourably=bool(ok_ci), per_seed=d.round(5).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=Path, default=ROOT / 'pilot')
    a = ap.parse_args()
    out = a.pilot / 'analysis'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    ev = a.pilot / 'eval'
    g = pd.read_csv(ev / 'got_independent.csv')
    boots = np.load(ev / 'got_independent_bootstrap.npz')
    lad = pd.read_csv(ev / 'ladder_layer8.csv') if (ev / 'ladder_layer8.csv').exists() else None

    def vals(df, fam, col):
        x = df[df.family == fam].set_index('seed').loc[list(SEEDS), col]
        return x.to_numpy(dtype=float)

    rows = []
    for x, r in COMPARISONS:
        for m in PRIMARY + SECONDARY_GOT:
            if m not in g.columns:
                continue
            bx = br = None
            if m in BOOT_KEYS:
                bx = np.stack([boots[f'{FAM[x]}|{s}|{m}'] for s in SEEDS])
                br = np.stack([boots[f'{FAM[r]}|{s}|{m}'] for s in SEEDS])
            res = compare(vals(g, FAM[x], m), vals(g, FAM[r], m), bx, br, m == 'error_term_auroc' and False)
            rows.append(dict(arm=x, ref=r, source='independent_got', metric=m, primary=(m in PRIMARY),
                             mean_x=float(vals(g, FAM[x], m).mean()), mean_ref=float(vals(g, FAM[r], m).mean()), **res))
        if lad is not None:
            for m in LADDER:
                if m in lad.columns:
                    res = compare(vals(lad, FAM[x], m), vals(lad, FAM[r], m), higher_better=m in HIGHER_BETTER)
                    rows.append(dict(arm=x, ref=r, source='ladder', metric=m, primary=False,
                                     mean_x=float(vals(lad, FAM[x], m).mean()), mean_ref=float(vals(lad, FAM[r], m).mean()),
                                     **res))
    df = pd.DataFrame(rows)
    pr = df[(df.arm == 'tsp') & (df.ref == 'ctrl') & df.primary].copy()
    ps = pr.p.to_numpy()
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        holm[i] = run
    pr['p_holm'] = holm
    pr['pass'] = (pr.seeds_better == 3) & pr.ci_excludes_0_favourably & (pr.p_holm < 0.05)

    def spec(metric):
        s = df[(df.arm == 'tsp') & df.ref.isin(['rand', 'pca']) & (df.metric == metric)]
        return bool(len(s) == 2 and (s.seeds_better == 3).all() and (s.mean_diff < 0).all())

    def lad_val(fam, m):
        return float(vals(lad, FAM[fam], m).mean()) if lad is not None and m in lad.columns else float('nan')
    guards = dict(
        owt_nmse_le_1_02x_ctrl=lad_val('tsp', 'owt_nmse') <= 1.02 * lad_val('ctrl', 'owt_nmse'),
        splice_dce_le_1_05x_ctrl=lad_val('tsp', 'splice_delta_ce') <= 1.05 * lad_val('ctrl', 'splice_delta_ce'),
        persona_refit_gap_le_ctrl_plus_0_02=lad_val('tsp', 'repe_persona__refit_logistic_gap')
        <= lad_val('ctrl', 'repe_persona__refit_logistic_gap') + 0.02,
        m4_preservation_ge_ctrl_minus_0_02=lad_val('tsp', 'm4__preservation_rec') >= lad_val('ctrl', 'm4__preservation_rec') - 0.02,
        l0_owt_128=abs(lad_val('tsp', 'owt_l0') - 128) < 1e-6)
    verdict = {}
    for _, row in pr.iterrows():
        verdict[row.metric] = dict(pass_rule=bool(row['pass']), p_holm=float(row.p_holm), specific_vs_controls=spec(row.metric))
    p_info = verdict.get('refit_logistic_gap', {}).get('pass_rule', False)
    p_read = verdict.get('frozen_logistic_gap', {}).get('pass_rule', False)
    g_ok = all(guards.values())
    if p_info and g_ok and verdict['refit_logistic_gap']['specific_vs_controls']:
        overall = 'INFORMATION IMPROVEMENT (validation pilot; truth-specific)'
    elif p_info or p_read:
        overall = ('READOUT-ONLY IMPROVEMENT' if not p_info else 'INFORMATION IMPROVEMENT') + \
            (' (not specific to the truth subspace)' if not verdict['refit_logistic_gap' if p_info else 'frozen_logistic_gap']['specific_vs_controls'] else '') + \
            ('' if g_ok else ' WITH A GUARD FAILURE (trade-off)')
    else:
        overall = 'NOT SUPPORTED'
    out.mkdir(parents=True)
    df.to_csv(out / 'comparisons.csv', index=False)
    pr.to_csv(out / 'primary.csv', index=False)
    (out / 'verdict.json').write_text(json.dumps(dict(primary=verdict, guards=guards, overall=overall), indent=2,
                                                 default=bool))
    print(overall)
    print(pr[['metric', 'mean_x', 'mean_ref', 'mean_diff', 'ci_low', 'ci_high', 'seeds_better', 'p_holm', 'pass']].to_string())
    print(json.dumps(guards, default=bool))


if __name__ == '__main__':
    main()
