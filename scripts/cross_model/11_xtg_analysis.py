#!/usr/bin/env python3
"""XT-G / DX analysis (pre-registered rules; written and hash-frozen before any XT-G output value was inspected).

Inputs: results/cross_model/xtg_truth_preservation/runs/<model>_<tag>/{per_cell,dx,repe_extras}.csv (+ original_by_layer).
Rules (preregistration.json `dx_rules`), per SAE x task cell:
  READOUT_SHIFT    raw frozen gap >= 0.05 AND affine-aligned frozen gap <= 0.5 x raw AND affine <= refit gap + 0.02
  INFORMATION_LOSS refit logistic gap CI low > 0 AND refit MLP gap CI low > 0
  LOSS_BEYOND_PCA  SAE refit logistic gap - error-matched-PCA refit logistic gap > 0
  SUPPRESSION      error_truth_enrichment CI low > 1
Implementation notes fixed BEFORE reading outputs (recorded in the XT-G AMENDMENTS as deviations):
  * The runner did not store bootstrap arrays, so (i) BH uses normal-approximation p-values from the stored 95 % CIs
    (se = (hi - lo) / 3.92; as pca_cm_posthoc_multiplicity), intersection-union (max p) for two-part rules; (ii) the
    LOSS_BEYOND_PCA "paired bootstrap CI" is replaced by z = (g_sae - g_pca) / sqrt(se_sae^2 + se_pca^2), which ignores
    the positive SAE/PCA covariance and is therefore conservative; label requires z-test p_BH < 0.05 and diff > 0.
  * Each SAE's error-matched PCA reference is recovered exactly: k_err = il.pca_k_for_val_nmse(xv, mu, V,
    owt_val_nmse_sae) with the runner's own owt_refs (PCA rows were de-duplicated by (kind, k) in the runner, A3).
  * Ceiling (A2): cells with original AUROC >= 0.99 (dx_ceiling_flag) are flagged; AUROC-gap rules are reported but
    the d' refit retention is the interpretable metric there.
BH within (model, task, rule) across layers and SAEs. Cross-model: descriptive at depth-matched triplets only
(headroom-normalised refit gap = gap / (orig AUROC - 0.5); SAE - PCA gap). Sparsity mechanism (batch 2, tag b2):
Spearman of refit gap and enrichment vs measured L0 within (layer, width).
Run: python scripts/cross_model/11_xtg_analysis.py --tag v1 [--runs gemma_main,gpt2_main] [--ladder gemma_b2]
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

REPO = Path(__file__).resolve().parents[2]
XT = REPO / 'results/cross_model/xtg_truth_preservation'
TRIP = {'T_B': {'gpt2': [3, 6, 9], 'gemma': [5, 12, 19]}, 'T_A': {'gpt2': [1, 8, 11], 'gemma': [1, 16, 23]}}
KEYS = ['model', 'layer', 'family', 'sae_id', 'seed', 'source', 'task']


def p_from_ci(est, lo, hi, null=0.0):
    se = (hi - lo) / 3.92
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (est - null) / se
    return np.where(se > 0, 2 * norm.sf(np.abs(z)), np.where(est == null, 1.0, 0.0)), se


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[o[i]] * m / (i + 1))
        q[o[i]] = prev
    return q


def load(run):
    d = XT / 'runs' / run
    pc, dx = pd.read_csv(d / 'per_cell.csv'), pd.read_csv(d / 'dx.csv')
    m = pc.merge(dx[KEYS + [c for c in dx.columns if c.startswith('dx_')]], on=KEYS, how='left', validate='1:1')
    return m


def k_err_map(df):
    """(model, layer, sae_id, seed) -> error-matched PCA k, recomputed with the runner's own references."""
    spec = importlib.util.spec_from_file_location('xtg10', REPO / 'scripts/cross_model/10_truth_preservation.py')
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    from src.evaluation import information_levels as il
    out = {}
    for model, g in df[df.source == 'sae'].groupby('model'):
        layers = sorted(g.layer.unique())
        refs = R.owt_refs(model, layers)
        for r in g.drop_duplicates(['layer', 'sae_id', 'seed']).itertuples():
            mu, _, V, xv = refs[r.layer]
            out[(model, r.layer, r.sae_id, r.seed)] = int(il.pca_k_for_val_nmse(xv, mu, V, r.owt_val_nmse_sae))
    return out


def rules(df, kmap):
    S = df[df.source == 'sae'].copy()
    P = df[df.source == 'pca_err'][['model', 'layer', 'task', 'pca_k', 'refit_logistic_gap', 'refit_logistic_gap_ci_low',
                                    'refit_logistic_gap_ci_high']].drop_duplicates(['model', 'layer', 'task', 'pca_k'])
    S['k_err'] = [kmap[(r.model, r.layer, r.sae_id, r.seed)] for r in S.itertuples()]
    S = S.merge(P.rename(columns={'pca_k': 'k_err', 'refit_logistic_gap': 'pca_refit_gap',
                                  'refit_logistic_gap_ci_low': 'pca_lo', 'refit_logistic_gap_ci_high': 'pca_hi'}),
                on=['model', 'layer', 'task', 'k_err'], how='left', validate='m:1')
    assert S.pca_refit_gap.notna().all(), 'an SAE has no error-matched PCA row'
    raw, aff = S.dx_raw_frozen_gap, S.dx_affine_frozen_gap
    S['READOUT_SHIFT'] = (raw >= 0.05) & (aff <= 0.5 * raw) & (aff <= S.refit_logistic_gap + 0.02)
    p1, se1 = p_from_ci(S.refit_logistic_gap, S.refit_logistic_gap_ci_low, S.refit_logistic_gap_ci_high)
    p2, _ = p_from_ci(S.refit_mlp_gap, S.refit_mlp_gap_ci_low, S.refit_mlp_gap_ci_high)
    S['p_INFORMATION_LOSS'] = np.maximum(np.where(S.refit_logistic_gap > 0, p1, 1.0), np.where(S.refit_mlp_gap > 0, p2, 1.0))
    S['INFORMATION_LOSS_ci'] = (S.refit_logistic_gap_ci_low > 0) & (S.refit_mlp_gap_ci_low > 0)
    se_p = (S.pca_hi - S.pca_lo) / 3.92
    S['sae_minus_pca_gap'] = S.refit_logistic_gap - S.pca_refit_gap
    se_d = np.sqrt(se1 ** 2 + se_p ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        S['p_LOSS_BEYOND_PCA'] = np.where(se_d > 0, 2 * norm.sf(np.abs(S.sae_minus_pca_gap / se_d)), 1.0)
    pe, _ = p_from_ci(S.dx_error_truth_enrichment, S.dx_error_truth_enrichment_ci_low, S.dx_error_truth_enrichment_ci_high, 1.0)
    S['p_SUPPRESSION'] = np.where(S.dx_error_truth_enrichment > 1, pe, 1.0)
    S['SUPPRESSION_ci'] = S.dx_error_truth_enrichment_ci_low > 1
    for rule in ('INFORMATION_LOSS', 'LOSS_BEYOND_PCA', 'SUPPRESSION'):
        S[f'q_{rule}'] = S.groupby(['model', 'task'])[f'p_{rule}'].transform(lambda p: bh(p.to_numpy()))
    S['INFORMATION_LOSS'] = S.INFORMATION_LOSS_ci & (S.q_INFORMATION_LOSS < 0.05)
    S['LOSS_BEYOND_PCA'] = (S.sae_minus_pca_gap > 0) & (S.q_LOSS_BEYOND_PCA < 0.05)
    S['SUPPRESSION'] = S.SUPPRESSION_ci & (S.q_SUPPRESSION < 0.05)
    S['ceiling'] = S.dx_ceiling_flag.fillna(False).astype(bool)
    return S


def cross_model(S, orig):
    rows = []
    for tname, t in TRIP.items():
        for slot in range(3):
            for task in sorted(S.task.unique()):
                for model in ('gpt2', 'gemma'):
                    g = S[(S.model == model) & (S.layer == t[model][slot]) & (S.task == task)]
                    for fam, gg in g.groupby('family'):
                        o = gg.dx_orig_auroc.mean()
                        rows.append(dict(triplet=tname, slot=slot, task=task, model=model, layer=t[model][slot],
                                         family=fam, n_sae=len(gg), orig_auroc=o, ceiling=bool(gg.ceiling.any()),
                                         refit_gap=gg.refit_logistic_gap.mean(),
                                         headroom_norm_gap=gg.refit_logistic_gap.mean() / (o - 0.5) if o > 0.5 else np.nan,
                                         sae_minus_pca=gg.sae_minus_pca_gap.mean(),
                                         dprime_refit_retention=gg.dx_dprime_refit_retention.mean(),
                                         share_info_loss=gg.INFORMATION_LOSS.mean(),
                                         share_beyond_pca=gg.LOSS_BEYOND_PCA.mean(),
                                         share_readout_shift=gg.READOUT_SHIFT.mean()))
    return pd.DataFrame(rows)


def sparsity(S):
    rows = []
    for (model, L, task), g in S.groupby(['model', 'layer', 'task']):
        g = g.assign(width=g.d_sae)
        for w, gw in g.groupby('width'):
            if len(gw) >= 4:
                for m in ('refit_logistic_gap', 'dx_error_truth_enrichment', 'sae_minus_pca_gap'):
                    rho, p = spearmanr(gw.owt_val_l0_sae, gw[m])
                    rows.append(dict(model=model, layer=L, task=task, d_sae=int(w), n=len(gw), metric=m, spearman=rho, p=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--runs', default='gemma_main,gpt2_main')
    ap.add_argument('--ladder', default='')
    a = ap.parse_args()
    out = XT / f'analysis_{a.tag}'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    runs = [r for r in a.runs.split(',') if (XT / 'runs' / r / 'per_cell.csv').exists()]
    df = pd.concat([load(r) for r in runs], ignore_index=True)
    S = rules(df, k_err_map(df))
    orig = {r: pd.read_csv(XT / 'runs' / r / 'original_by_layer.csv') for r in runs}
    C = cross_model(S, orig)
    out.mkdir(parents=True)
    S.to_csv(out / 'cells_rules.csv', index=False)
    C.to_csv(out / 'cross_model_descriptive.csv', index=False)
    fam = S.groupby(['model', 'task', 'family'])[['READOUT_SHIFT', 'INFORMATION_LOSS', 'LOSS_BEYOND_PCA', 'SUPPRESSION',
                                                   'ceiling']].agg(['sum', 'count'])
    fam.to_csv(out / 'rule_counts.csv')
    info = dict(runs=runs, n_sae_cells=len(S), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                approximations=['BH p from stored 95 % CIs (normal approx)', 'LOSS_BEYOND_PCA z-test ignoring SAE/PCA covariance (conservative)'])
    if a.ladder:
        L = load(a.ladder)
        SL = rules(L, k_err_map(L))
        SL.to_csv(out / 'ladder_cells_rules.csv', index=False)
        sparsity(SL).to_csv(out / 'sparsity_mechanism.csv', index=False)
        info['ladder'] = a.ladder
    (out / 'info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
