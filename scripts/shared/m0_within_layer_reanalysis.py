#!/usr/bin/env python3
"""M0: within-layer reanalysis of EXISTING validation results for the 108 uninterrupted OWT SAEs.

Pre-registered in results/shared/m0_within_layer_reanalysis/preregistration.json (written before this script ran).
No model, no SAE, no activation and no test row is loaded: inputs are the existing validation CSVs only.

Questions
  SQ5  Within a layer, does reconstruction error predict (a) loss of GoT-cities truth readout and (b) splice damage?
  SQ1  Architecture comparison per layer, with the L0 difference made explicit (L0 adjustment, effective width).
  Claim corrections: pooled-across-layer correlations versus within-layer ones; splicing SAE-vs-random ratio range.

Run from repository root:
    python scripts/shared/m0_within_layer_reanalysis.py
"""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/shared/m0_within_layer_reanalysis'
UNIFIED = REPO / 'results/shared/spliced_evaluation/unified_comparison.csv'
GOT = REPO / 'results/geometry_of_truth/all_layer_frozen_probe_evaluation/per_run.csv'
OWT = REPO / 'results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_per_run.csv'
SWEEP = 'results/openwebtext/sae_sparsity_sweep/layer_*/comparison.csv'
N_PERM, N_BOOT, SEED = 5000, 2000, 20261108
ARCHS = ['relu', 'topk', 'jumprelu']
OUTPUT_FILES = ['merged_validation_table.csv', 'within_layer_regressions.csv', 'within_layer_spearman.csv',
                'architecture_by_layer.csv', 'architecture_pairwise_l0_adjusted.csv', 'l0_elasticity_from_sweep.csv',
                'pooled_vs_within_corrections.csv', 'splice_ratio_range.json', 'summary.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- data

def load():
    u = pd.read_csv(UNIFIED)
    g = pd.read_csv(GOT)
    g = g[(g.domain == 'geographic') & (g.eval_set == 'validation')]
    assert len(u) == 108 and len(g) == 108, (len(u), len(g))
    assert set(pd.read_csv(GOT).eval_set) == {'validation', 'test'}
    g = g.rename(columns={'relative_sq_error': 'got_rel_sq_error', 'nmse': 'got_nmse', 'cosine': 'got_cosine',
                          'l0': 'got_l0', 'logistic_auroc_original': 'got_logistic_orig',
                          'diffmean_auroc_original': 'got_diffmean_orig', 'logistic_auroc_gap': 'got_logistic_gap',
                          'diffmean_auroc_gap': 'got_diffmean_gap', 'informative_layer': 'got_informative'})
    keep = ['layer', 'architecture', 'seed', 'got_rel_sq_error', 'got_nmse', 'got_cosine', 'got_l0', 'got_logistic_orig',
            'got_diffmean_orig', 'got_logistic_gap', 'got_diffmean_gap', 'got_informative', 'n']
    d = u.merge(g[keep], on=['layer', 'architecture', 'seed'], how='inner', validate='one_to_one')
    assert len(d) == 108
    # consistency: the GoT gap recorded in the unified table is the same validation number
    assert np.allclose(d.got_logistic_gap, d.geographic_val_logistic_auroc_gap)
    o = pd.read_csv(OWT)[['layer', 'architecture', 'seed', 'val_explained_variance', 'val_l0']]
    d = d.merge(o, on=['layer', 'architecture', 'seed'], validate='one_to_one')
    d['owt_val_nmse'] = 1 - d.owt_val_recon__explained_variance
    d['owt_val_l0'] = d.owt_val_recon__l0
    d['live_latents'] = 3072 * (1 - d.owt_val_recon__dead_fraction)
    d['log_owt_nmse'] = np.log(d.owt_val_nmse)
    d['log_got_rel_err'] = np.log(d.got_rel_sq_error)
    d['log_owt_l0'] = np.log(d.owt_val_l0)
    d['owt_delta_ce'] = d.owt__delta_ce
    d['included'] = d.matched_10pct.astype(bool)
    return d


# ------------------------------------------------------------------------------------- regression

def design(df, pred, arch_fe):
    cols = [df[pred].to_numpy(float)[:, None]]
    layers = sorted(df.layer.unique())
    cols.append(np.stack([(df.layer == L).to_numpy(float) for L in layers], 1))           # layer fixed effects
    if arch_fe:
        cols.append(np.stack([(df.architecture == a).to_numpy(float) for a in ARCHS[1:]], 1))
    cols.append(df.log_owt_l0.to_numpy(float)[:, None])
    return np.concatenate(cols, 1)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = r @ r / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), r, dof


def fit(df, pred, out, arch_fe, rng):
    X, y = design(df, pred, arch_fe), df[out].to_numpy(float)
    beta, se, _, dof = ols(X, y)
    t_obs = beta[0] / se[0]
    # Freedman-Lane: permute residuals of the reduced (nuisance-only) model within layer
    Xr = X[:, 1:]
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fitted, res = Xr @ br, y - Xr @ br
    idx_by_layer = [np.where(df.layer.to_numpy() == L)[0] for L in sorted(df.layer.unique())]
    exceed = 0
    for _ in range(N_PERM):
        perm = np.arange(len(y))
        for ix in idx_by_layer:
            perm[ix] = rng.permutation(ix)
        b, s, _, _ = ols(X, fitted + res[perm])
        exceed += abs(b[0] / s[0]) >= abs(t_obs)
    p_perm = (1 + exceed) / (1 + N_PERM)
    boots = []
    for _ in range(N_BOOT):                     # resample runs within layer
        ix = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx_by_layer])
        b, *_ = np.linalg.lstsq(X[ix], y[ix], rcond=None)
        boots.append(b[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # standardised effect: change in outcome (in within-layer sd units) per within-layer sd of the predictor
    wsd_x = df.groupby('layer')[pred].transform(lambda v: v - v.mean()).std()
    wsd_y = df.groupby('layer')[out].transform(lambda v: v - v.mean()).std()
    return dict(beta=beta[0], se=se[0], t=t_obs, dof=dof, p_perm=p_perm, ci95_low=lo, ci95_high=hi,
                beta_std=beta[0] * wsd_x / wsd_y, within_layer_sd_pred=wsd_x, within_layer_sd_out=wsd_y, n=len(y))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------------------- architecture + L0

def sweep_elasticity():
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / SWEEP)))])
    s['log_nmse'] = np.log(1 - s.explained_variance)
    s['log_l0'] = np.log(s.actual_l0)
    rows = []
    for (L, a), g in s.groupby(['layer', 'architecture']):
        g = g[g.target_l0.isin([64, 128])]
        slope = np.polyfit(g.log_l0, g.log_nmse, 1)[0] if len(g) == 2 else np.nan
        rows.append(dict(layer=L, architecture=a, elasticity_64_128=slope, steps=int(g.steps.iloc[0]),
                         l0_points=';'.join(f'{v:.1f}' for v in g.actual_l0)))
    return pd.DataFrame(rows)


def architecture_tables(d, el):
    inc = d[d.included]
    metrics = ['owt_val_nmse', 'owt_val_l0', 'got_l0', 'live_latents', 'owt_delta_ce', 'owt__kl_mean',
               'got__statement_kl_mean', 'got_rel_sq_error', 'got_logistic_gap', 'got_diffmean_gap']
    agg = inc.groupby(['layer', 'architecture'])[metrics].agg(['mean', 'std', 'count'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    agg = agg.reset_index().merge(el, on=['layer', 'architecture'], how='left')
    # L0-adjusted NMSE: move each SAE to L0 = 128 along its sweep elasticity (local, 1000-step sweep)
    agg['owt_val_nmse_l0adj_128'] = agg.owt_val_nmse_mean * np.exp(
        -agg.elasticity_64_128 * (np.log(agg.owt_val_l0_mean) - np.log(128)))
    pair = []
    for L, g in agg.groupby('layer'):
        g = g.set_index('architecture')
        for a, b in [('relu', 'topk'), ('relu', 'jumprelu'), ('topk', 'jumprelu')]:
            if a not in g.index or b not in g.index:
                continue
            ra, rb = inc[(inc.layer == L) & (inc.architecture == a)], inc[(inc.layer == L) & (inc.architecture == b)]
            log_ratio = np.log(g.loc[a, 'owt_val_nmse_mean'] / g.loc[b, 'owt_val_nmse_mean'])
            log_l0_ratio = np.log(g.loc[a, 'owt_val_l0_mean'] / g.loc[b, 'owt_val_l0_mean'])
            welch = stats.ttest_ind(np.log(ra.owt_val_nmse), np.log(rb.owt_val_nmse), equal_var=False)
            adj = np.log(g.loc[a, 'owt_val_nmse_l0adj_128'] / g.loc[b, 'owt_val_nmse_l0adj_128'])
            winner_raw = a if log_ratio < 0 else b
            # break-even elasticity: the elasticity at which the L0 difference alone would explain the NMSE gap
            breakeven = (-log_ratio / log_l0_ratio) if abs(log_l0_ratio) > 1e-9 else np.nan
            explained = (np.sign(log_ratio) == -np.sign(log_l0_ratio)) and not np.isnan(breakeven)
            pair.append(dict(
                layer=L, a=a, b=b, n_a=len(ra), n_b=len(rb),
                nmse_a=g.loc[a, 'owt_val_nmse_mean'], nmse_b=g.loc[b, 'owt_val_nmse_mean'],
                nmse_rel_diff_pct=100 * (np.exp(log_ratio) - 1), welch_p_log_nmse=welch.pvalue,
                l0_a=g.loc[a, 'owt_val_l0_mean'], l0_b=g.loc[b, 'owt_val_l0_mean'],
                l0_rel_diff_pct=100 * (np.exp(log_l0_ratio) - 1),
                lower_nmse_raw=winner_raw,
                breakeven_elasticity=breakeven if explained else np.nan,
                nmse_rel_diff_pct_l0adj=100 * (np.exp(adj) - 1),
                lower_nmse_l0adj=a if adj < 0 else b,
                ranking_flips_after_adj=(adj < 0) != (log_ratio < 0),
                live_latents_a=g.loc[a, 'live_latents_mean'], live_latents_b=g.loc[b, 'live_latents_mean'],
                delta_ce_a=g.loc[a, 'owt_delta_ce_mean'], delta_ce_b=g.loc[b, 'owt_delta_ce_mean'],
                got_gap_a=g.loc[a, 'got_logistic_gap_mean'], got_gap_b=g.loc[b, 'got_logistic_gap_mean'],
                got_l0_a=g.loc[a, 'got_l0_mean'], got_l0_b=g.loc[b, 'got_l0_mean']))
    return agg, pd.DataFrame(pair)


# --------------------------------------------------------------------------------------- pooled

def pooled_vs_within(d):
    inc = d[d.included]
    specs = [  # (x, y, scope_filter, claim it corrects)
        ('owt_val_recon__explained_variance', 'repe_final__persona_shift_cosine', None,
         'old RepE: reconstruction tracks honesty preservation (HISTORICAL, length-confounded task)'),
        ('owt__kl_mean', 'honesty_val_diffmean_auroc_gap', None,
         'old RepE: rho = 0.86 between splice KL and honesty gap (HISTORICAL, length-confounded task)'),
        ('owt_val_recon__explained_variance', 'owt__delta_ce', None, 'reconstruction predicts splice damage'),
        ('owt_val_recon__explained_variance', 'got__statement_kl_mean', None, 'reconstruction predicts GoT splice KL'),
        ('got_rel_sq_error', 'got_logistic_gap', 'got_informative', 'task reconstruction error predicts truth-readout loss'),
        ('owt_val_nmse', 'got_logistic_gap', 'got_informative', 'OWT reconstruction error predicts truth-readout loss'),
        ('got_cosine', 'got_logistic_gap', 'got_informative', 'task cosine predicts truth-readout loss'),
        ('owt__kl_mean', 'got_logistic_gap', 'got_informative', 'splice KL predicts truth-readout loss'),
    ]
    rows = []
    for x, y, filt, claim in specs:
        s = inc if filt is None else inc[inc[filt].astype(bool)]
        pr, pp = stats.spearmanr(s[x], s[y])
        w = [stats.spearmanr(g[x], g[y])[0] for _, g in s.groupby('layer') if g[x].nunique() > 1]
        # within-layer pooled: rank-correlate after removing the layer mean (demeaned Spearman)
        xd = s[x] - s.groupby('layer')[x].transform('mean')
        yd = s[y] - s.groupby('layer')[y].transform('mean')
        dr, dp = stats.spearmanr(xd, yd)
        rows.append(dict(x=x, y=y, claim=claim, n=len(s), n_layers=s.layer.nunique(),
                         pooled_spearman=pr, pooled_p=pp, within_layer_mean_spearman=np.mean(w),
                         within_layer_min=np.min(w), within_layer_max=np.max(w),
                         layer_demeaned_spearman=dr, layer_demeaned_p=dp))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ main

def main():
    for f in OUTPUT_FILES:
        if (OUT / f).exists():
            raise SystemExit(f'{OUT / f} exists; refusing to overwrite.')
    prereg = OUT / 'preregistration.json'
    if not prereg.exists():
        raise SystemExit('preregistration.json missing: pre-register before running.')
    pr = json.loads(prereg.read_text())
    if pr['hashes']['script'] != sha(__file__):
        raise SystemExit('script hash differs from the pre-registered hash; refusing to run.')
    rng = np.random.default_rng(SEED)
    d = load()
    d.to_csv(OUT / 'merged_validation_table.csv', index=False)
    inc = d[d.included]

    # ---- primary and secondary within-layer regressions
    primary = [('log_owt_nmse', 'got_logistic_gap'), ('log_got_rel_err', 'got_logistic_gap'),
               ('log_owt_nmse', 'owt_delta_ce'), ('log_got_rel_err', 'owt_delta_ce')]
    secondary_out = ['got_diffmean_gap', 'got__statement_kl_mean', 'owt__kl_mean']
    rows = []
    for arch_fe, family in [(True, 'primary (layer FE + architecture FE + log L0)'),
                            (False, 'secondary (layer FE + log L0, no architecture FE)')]:
        fam_rows = []
        for pred, out in primary + [('log_owt_nmse', o) for o in secondary_out]:
            df = inc[inc.got_informative.astype(bool)] if out.startswith('got_') and 'gap' in out else inc
            r = fit(df.reset_index(drop=True), pred, out, arch_fe, rng)
            r.update(model=family, predictor=pred, outcome=out, holm_family=(arch_fe and (pred, out) in primary),
                     layers=f'{df.layer.min()}-{df.layer.max()}')
            fam_rows.append(r)
        rows += fam_rows
    reg = pd.DataFrame(rows)
    m = reg.holm_family
    reg['p_holm'] = np.nan
    reg.loc[m, 'p_holm'] = holm(reg.loc[m, 'p_perm'])
    reg.to_csv(OUT / 'within_layer_regressions.csv', index=False)

    # ---- per-layer Spearman (descriptive, n = 8-9 per layer)
    sp = []
    for L, g in inc.groupby('layer'):
        for x, y in [('owt_val_nmse', 'owt_delta_ce'), ('owt_val_nmse', 'got_logistic_gap'),
                     ('got_rel_sq_error', 'got_logistic_gap'), ('owt_val_nmse', 'got_diffmean_gap')]:
            if y.startswith('got_') and not bool(g.got_informative.iloc[0]):
                continue
            r, p = stats.spearmanr(g[x], g[y])
            sp.append(dict(layer=L, x=x, y=y, n=len(g), spearman=r, p=p))
    pd.DataFrame(sp).to_csv(OUT / 'within_layer_spearman.csv', index=False)

    # ---- architecture tables
    el = sweep_elasticity()
    el.to_csv(OUT / 'l0_elasticity_from_sweep.csv', index=False)
    agg, pair = architecture_tables(d, el)
    agg.to_csv(OUT / 'architecture_by_layer.csv', index=False)
    pair.to_csv(OUT / 'architecture_pairwise_l0_adjusted.csv', index=False)

    # ---- claim corrections
    pv = pooled_vs_within(d)
    pv.to_csv(OUT / 'pooled_vs_within_corrections.csv', index=False)
    ratio = inc.owt__sae_vs_random_delta_ce_ratio
    rr = dict(n=int(len(ratio)), min=float(ratio.min()), max=float(ratio.max()), median=float(ratio.median()),
              n_below_1=int((ratio < 1).sum()),
              runs_below_1=inc.loc[ratio < 1, ['layer', 'architecture', 'seed']].astype(str).agg('/'.join, 1).tolist(),
              by_layer_min=inc.groupby('layer').owt__sae_vs_random_delta_ce_ratio.min().round(3).to_dict(),
              by_layer_max=inc.groupby('layer').owt__sae_vs_random_delta_ce_ratio.max().round(3).to_dict())
    (OUT / 'splice_ratio_range.json').write_text(json.dumps(rr, indent=2))

    summary = dict(
        stage='M0', evidence='validation reanalysis of existing results (no new model runs)',
        n_runs=int(len(d)), n_included=int(d.included.sum()),
        excluded=d.loc[~d.included, ['layer', 'architecture', 'seed', 'owt_val_l0']].to_dict('records'),
        primary=reg[reg.holm_family][['predictor', 'outcome', 'beta', 'ci95_low', 'ci95_high', 'beta_std', 'p_perm',
                                      'p_holm', 'n']].to_dict('records'),
        ranking_flips_after_l0_adjustment=int(pair.ranking_flips_after_adj.sum()),
        input_sha256={str(p.relative_to(REPO)): sha(p) for p in [UNIFIED, GOT, OWT]},
        script_sha256=sha(__file__), n_perm=N_PERM, n_boot=N_BOOT, seed=SEED)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option('display.width', 250)
    print(reg[['model', 'predictor', 'outcome', 'n', 'beta', 'ci95_low', 'ci95_high', 'beta_std', 'p_perm', 'p_holm']]
          .round(4).to_string(index=False))
    print(pair[['layer', 'a', 'b', 'nmse_rel_diff_pct', 'l0_rel_diff_pct', 'nmse_rel_diff_pct_l0adj',
                'lower_nmse_raw', 'lower_nmse_l0adj', 'breakeven_elasticity']].round(3).to_string(index=False))
    print(pv.round(3).to_string(index=False))
    print(json.dumps(rr, indent=1))


if __name__ == '__main__':
    main()
