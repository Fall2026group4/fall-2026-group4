#!/usr/bin/env python3
"""AUDIT-S5 (sensitivity, NOT confirmatory): is the frozen-probe loss on SAE reconstructions explained by the
error's marginal distribution, or is the SAE error aligned with the label?

Problem (audit 2026-11-13): M2's only perturbation reference is ISOTROPIC noise with the per-row norm of the SAE
error. The standardised logistic probe up-weights low-variance directions, so isotropic noise is an unusually
damaging reference, and "the collapse is reproduced by noise" may depend on that choice.

References (frozen ORIGINAL probes are linear: score(x + v) = score(x) + w . v), VALIDATION rows, TRAIN-fitted probes:
  R0 SAE reconstruction x_hat (reproduces M2 frozen gaps)
  R1 isotropic, ||eta_i|| = ||e_i||                       (M2's reference, recomputed with 20 draws)
  R2 row-shuffled SAE error: x_i + e_pi(i)                (exact marginal of the SAE error, alignment with x_i removed;
                                                           20 permutations)
  R3 Gaussian with the SAE-error covariance Sigma_e (estimated on the task's TRAIN rows), rescaled to ||e_i|| (20 draws)
for the frozen logistic and frozen mass-mean readers, GoT cities+neg_cities and RepE-LM persona, all 108 base SAEs
(layers 0-11) and the R2 TopK/BatchTopK/ReLU SAEs (layers 1, 8, 11).
Rule (fixed before running): a cell shows LABEL-ALIGNED loss if the paired grouped-bootstrap 95 % CI (B = 500) of
gap(R0) - gap(R2) is > 0; BH over the cells of each task x reader. Reported per layer as counts.
Outputs: results/shared/audit_sensitivity/s5_noise_references/ (refuses to overwrite). No TEST row is read.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

spec = importlib.util.spec_from_file_location('ladder', REPO / 'scripts/shared/sae_ladder_evaluation.py')
LAD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LAD)
M2 = LAD.M2

OUT = REPO / 'results/shared/audit_sensitivity/s5_noise_references'
M2_CELLS = REPO / 'results/shared/m2_four_level_faithfulness/per_cell.csv'
N_DRAW, N_DRAW_BOOT, B, SEED = 20, 10, 500, 20261116
TASKS = ('got_cities', 'repe_persona')
R2_ARMS = ('topk', 'batchtopk', 'relu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.mkdir(parents=True)
    t0, device = time.time(), 'cuda'
    tasks = M2.load_tasks()
    m2 = pd.read_csv(M2_CELLS)
    rows, checks = [], []
    layers = [8] if a.smoke else list(range(12))
    for L in layers:
        rng0 = np.random.default_rng(M2.SEED + L)
        origs = {tn: M2.original_side(t, L, rng0) for tn, t in tasks.items()}
        srcs = LAD.sources(L, ['old'] + (['r2'] if L in (1, 8, 11) else []))
        srcs = [s for s in srcs if s['family'] == 'old' or s['name'] in R2_ARMS]
        if a.smoke:
            srcs = [s for s in srcs if s['seed'] == 42]
        for tn in TASKS:
            t = tasks[tn]
            tr, va = t['split'] == 'train', t['split'] == 'validation'
            y, X = t['y'][va], t['X'][L][va].astype(np.float64)
            o = origs[tn]
            if o['auroc']['logistic'] < M2.INFORMATIVE:
                continue
            rb = np.random.default_rng(SEED + L)
            bidx = il.bootstrap_indices(t['groups'][va], B, rb)
            readers = dict(logistic=(o['a'], o['scores']['logistic']), massmean=(o['direction'], o['scores']['massmean']))
            ob = {k: np.array([il.auroc(y[i], v[1][i]) for i in bidx]) for k, v in readers.items()}
            for i, src in enumerate(srcs):
                sae = LAD.load(src, L, device)
                H, _ = LAD.encode_np(sae, t['X'][L], device)
                Htr = H[tr].astype(np.float64)
                del sae
                torch.cuda.empty_cache()
                E = X - H[va].astype(np.float64)
                Etr = t['X'][L][tr].astype(np.float64) - Htr
                nrm = np.linalg.norm(E, axis=1, keepdims=True)
                cov = np.cov(Etr.T)
                Lc = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0])
                rng = np.random.default_rng(SEED + 1000 * L + i)
                perms = [rng.permutation(len(E)) for _ in range(N_DRAW)]
                iso = rng.standard_normal((N_DRAW,) + E.shape)
                iso = iso / np.linalg.norm(iso, axis=-1, keepdims=True) * nrm[None]
                gau = rng.standard_normal((N_DRAW,) + E.shape) @ Lc.T
                gau = gau / np.linalg.norm(gau, axis=-1, keepdims=True) * nrm[None]
                key = (src['family'], src['name'], src['seed'])
                r = dict(layer=L, task=tn, family=key[0], name=key[1], seed=key[2])
                for rd, (w, s0) in readers.items():
                    s_sae = s0 - E @ w                          # score(x_hat) = score(x) - w.e, since x_hat = x - e
                    s_shuf = [s0 - E[p] @ w for p in perms]     # x_i - e_pi(i): same error marginal, alignment removed
                    s_iso = [s0 + v @ w for v in iso]           # symmetric references: sign immaterial
                    s_gau = [s0 + v @ w for v in gau]
                    auc0 = o['auroc'][rd]
                    r[f'{rd}_gap_sae'] = auc0 - il.auroc(y, s_sae)
                    r[f'{rd}_gap_isotropic'] = auc0 - np.mean([il.auroc(y, s) for s in s_iso])
                    r[f'{rd}_gap_shuffled_error'] = auc0 - np.mean([il.auroc(y, s) for s in s_shuf])
                    r[f'{rd}_gap_error_cov_gaussian'] = auc0 - np.mean([il.auroc(y, s) for s in s_gau])
                    sb = np.array([il.auroc(y[j], s_sae[j]) for j in bidx])
                    shb = np.mean([[il.auroc(y[j], (s0 - E[p] @ w)[j]) for j in bidx] for p in perms[:N_DRAW_BOOT]], 0)
                    diff = (ob[rd] - sb) - (ob[rd] - shb)
                    r[f'{rd}_sae_minus_shuffled'] = r[f'{rd}_gap_sae'] - r[f'{rd}_gap_shuffled_error']
                    r[f'{rd}_sae_minus_shuffled_ci_low'], r[f'{rd}_sae_minus_shuffled_ci_high'] = il.ci(diff)
                    r[f'{rd}_sae_minus_shuffled_p'] = il.p_greater(diff)
                if key[0] == 'old':
                    ref = m2[(m2.layer == L) & (m2.task == tn) & (m2.source == 'sae') & (m2.architecture == key[1]) & (m2.seed == key[2])]
                    if len(ref):
                        checks.append(dict(layer=L, task=tn, name=key[1], seed=key[2],
                                           abs_diff_logistic=abs(r['logistic_gap_sae'] - float(ref.frozen_logistic_gap.iloc[0])),
                                           abs_diff_massmean=abs(r['massmean_gap_sae'] - float(ref.frozen_massmean_gap.iloc[0]))))
                rows.append(r)
            print(f'L{L} {tn} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    for tn in df.task.unique():
        for rd in ('logistic', 'massmean'):
            m = df.task == tn
            df.loc[m, f'{rd}_sae_minus_shuffled_q'] = il.bh(df.loc[m, f'{rd}_sae_minus_shuffled_p'].to_numpy())
            df.loc[m, f'{rd}_label_aligned'] = (df.loc[m, f'{rd}_sae_minus_shuffled_ci_low'] > 0) & \
                                               (df.loc[m, f'{rd}_sae_minus_shuffled_q'] < 0.05)
    df.to_csv(a.out / 'per_cell.csv', index=False)
    agg = df.groupby(['task', 'layer', 'family']).agg(
        n=('seed', 'size'), logistic_gap_sae=('logistic_gap_sae', 'mean'), logistic_gap_isotropic=('logistic_gap_isotropic', 'mean'),
        logistic_gap_shuffled=('logistic_gap_shuffled_error', 'mean'), logistic_gap_errcov=('logistic_gap_error_cov_gaussian', 'mean'),
        logistic_label_aligned=('logistic_label_aligned', 'sum'),
        massmean_gap_sae=('massmean_gap_sae', 'mean'), massmean_gap_isotropic=('massmean_gap_isotropic', 'mean'),
        massmean_gap_shuffled=('massmean_gap_shuffled_error', 'mean'), massmean_gap_errcov=('massmean_gap_error_cov_gaussian', 'mean'),
        massmean_label_aligned=('massmean_label_aligned', 'sum')).reset_index()
    agg.to_csv(a.out / 'by_layer.csv', index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(a.out / 'reproduction_checks.csv', index=False)
    info = dict(test_rows_used=0, n_cells=len(df), max_abs_diff_vs_m2_logistic=float(ck.abs_diff_logistic.max()) if len(ck) else None,
                max_abs_diff_vs_m2_massmean=float(ck.abs_diff_massmean.max()) if len(ck) else None,
                label_aligned_counts={f'{tn}/{rd}': int(df[df.task == tn][f'{rd}_label_aligned'].sum())
                                      for tn in df.task.unique() for rd in ('logistic', 'massmean')},
                runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
