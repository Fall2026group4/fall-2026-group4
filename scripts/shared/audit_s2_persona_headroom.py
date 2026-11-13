#!/usr/bin/env python3
"""AUDIT-S2 (sensitivity, NOT confirmatory): is the SAE's persona-information loss larger than its loss on the
neutral one-token control once each task's decodable headroom is taken into account?

Problem (audit 2026-11-13): the paper's "part of the mid-layer persona loss is persona-specific" compares RAW re-trained
logistic gaps (AUROC_orig - AUROC_refit). The persona contrast is far more decodable than the neutral contrast
(L6: 0.83 vs 0.62), so raw gaps are not comparable.

Measure (pre-registered in results/shared/audit_sensitivity/preregistration.json):
  normalised loss  NL_task = (AUROC_orig - AUROC_refit) / (AUROC_orig - 0.5)    (1 - retained share of the signal)
  difference       Delta_NL = NL_persona - NL_neutral, per SAE, then averaged over the SAEs of a layer (and per family)
Uncertainty: paired grouped bootstrap over validation statement groups (norm_group), B = 1000, the same resample for
both tasks and all SAEs of a layer (persona and neutral rows come from the same validation statements).
Rule: a layer shows persona-SPECIFIC loss only if the 95 % percentile CI of the layer-mean Delta_NL is > 0 and the
one-sided bootstrap p survives Benjamini-Hochberg over layers 1-11 (q < 0.05). Layers where the neutral original
AUROC is < 0.60 (headroom < 0.10) are flagged LOW PRECISION.
Sources: the 108 base SAEs (layers 0-11) and the R2 TopK / BatchTopK / ReLU SAEs (layers 1, 8, 11). Probes and
originals use the unchanged M2 functions (fit on TRAIN, scored on VALIDATION); the re-fit gaps are checked against
the committed M2 per_cell.csv. No TEST row is read.

Outputs: results/shared/audit_sensitivity/s2_persona_headroom/ (refuses to overwrite).
Run from the repository root:  python scripts/shared/audit_s2_persona_headroom.py [--smoke --out DIR]
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

OUT = REPO / 'results/shared/audit_sensitivity/s2_persona_headroom'
M2_CELLS = REPO / 'results/shared/m2_four_level_faithfulness/per_cell.csv'
B, BOOT_SEED = 1000, 20261114
R2_ARMS = ('topk', 'batchtopk', 'relu')


def bh(p):
    return il.bh(p)


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
    tp, tn = tasks['repe_persona'], tasks['repe_neutral']
    m2 = pd.read_csv(M2_CELLS)
    layers = [8] if a.smoke else list(range(1, 12))
    per_sae, layer_rows, checks = [], [], []
    for L in layers:
        rng = np.random.default_rng(M2.SEED + L)
        origs = {k: M2.original_side(t, L, rng) for k, t in tasks.items()}          # same stream as M2
        vp, vn = tp['split'] == 'validation', tn['split'] == 'validation'
        gp, gn = tp['groups'][vp], tn['groups'][vn]
        u = np.unique(np.concatenate([gp, gn]))
        assert set(np.unique(gp)) == set(np.unique(gn)) == set(u), 'persona and neutral statements differ'
        rb = np.random.default_rng(BOOT_SEED + L)
        draws = [rb.integers(len(u), size=len(u)) for _ in range(B)]
        mem_p = {c: np.flatnonzero(gp == g_) for c, g_ in enumerate(u)}
        mem_n = {c: np.flatnonzero(gn == g_) for c, g_ in enumerate(u)}
        ip = [np.concatenate([mem_p[c] for c in d]) for d in draws]
        in_ = [np.concatenate([mem_n[c] for c in d]) for d in draws]
        yp, yn = tp['y'][vp], tn['y'][vn]
        op, on = origs['repe_persona']['scores']['logistic'], origs['repe_neutral']['scores']['logistic']
        op_b = np.array([il.auroc(yp[i], op[i]) for i in ip])
        on_b = np.array([il.auroc(yn[i], on[i]) for i in in_])
        o_auc_p, o_auc_n = origs['repe_persona']['auroc']['logistic'], origs['repe_neutral']['auroc']['logistic']
        srcs = LAD.sources(L, ['old'] + (['r2'] if L in (1, 8, 11) else []))
        srcs = [s for s in srcs if s['family'] == 'old' or s['name'] in R2_ARMS]
        if a.smoke:
            srcs = [s for s in srcs if s['seed'] == 42]
        nl_b = {}
        for src in srcs:
            sae = LAD.load(src, L, device)
            sc = {}
            for k, t in (('persona', tp), ('neutral', tn)):
                H, _ = LAD.encode_np(sae, t['X'][L], device)
                tr = t['split'] == 'train'
                pipe, _ = il.fit_logistic(H[tr], t['y'][tr], t['groups'][tr])
                sc[k] = pipe.decision_function(H[t['split'] == 'validation'])
            del sae
            torch.cuda.empty_cache()
            rp_auc, rn_auc = il.auroc(yp, sc['persona']), il.auroc(yn, sc['neutral'])
            gap_p, gap_n = o_auc_p - rp_auc, o_auc_n - rn_auc
            nl_p, nl_n = gap_p / (o_auc_p - 0.5), gap_n / (o_auc_n - 0.5)
            rp_b = np.array([il.auroc(yp[i], sc['persona'][i]) for i in ip])
            rn_b = np.array([il.auroc(yn[i], sc['neutral'][i]) for i in in_])
            d_b = (op_b - rp_b) / (op_b - 0.5) - (on_b - rn_b) / (on_b - 0.5)
            key = (src['family'], src['name'], src['seed'])
            nl_b[key] = d_b
            per_sae.append(dict(layer=L, family=key[0], name=key[1], seed=key[2], orig_auroc_persona=o_auc_p,
                                orig_auroc_neutral=o_auc_n, refit_gap_persona=gap_p, refit_gap_neutral=gap_n,
                                raw_gap_diff=gap_p - gap_n, nl_persona=nl_p, nl_neutral=nl_n, delta_nl=nl_p - nl_n,
                                delta_nl_ci_low=float(np.nanpercentile(d_b, 2.5)),
                                delta_nl_ci_high=float(np.nanpercentile(d_b, 97.5))))
            if key[0] == 'old':
                for task, gap in (('repe_persona', gap_p), ('repe_neutral', gap_n)):
                    ref = m2[(m2.layer == L) & (m2.task == task) & (m2.source == 'sae') & (m2.architecture == key[1])
                             & (m2.seed == key[2])]
                    if len(ref):
                        checks.append(dict(layer=L, task=task, name=key[1], seed=key[2], recomputed=gap,
                                           m2=float(ref.refit_logistic_gap.iloc[0]),
                                           abs_diff=abs(gap - float(ref.refit_logistic_gap.iloc[0]))))
            print(f"L{L} {key}: NL persona {nl_p:.3f} neutral {nl_n:.3f} ({time.time() - t0:.0f}s)", flush=True)
        ps = pd.DataFrame([r for r in per_sae if r['layer'] == L])
        for fam, sel in (('old', ps.family == 'old'), ('r2', ps.family == 'r2')):
            if not sel.any():
                continue
            keys = [(r.family, r.name, r.seed) for r in ps[sel].itertuples()]
            mb = np.nanmean([nl_b[k] for k in keys], 0)
            layer_rows.append(dict(layer=L, family=fam, n_saes=len(keys), orig_auroc_persona=o_auc_p,
                                   orig_auroc_neutral=o_auc_n, low_precision=bool(o_auc_n < 0.60),
                                   raw_gap_persona=ps[sel].refit_gap_persona.mean(),
                                   raw_gap_neutral=ps[sel].refit_gap_neutral.mean(),
                                   nl_persona=ps[sel].nl_persona.mean(), nl_neutral=ps[sel].nl_neutral.mean(),
                                   delta_nl=ps[sel].delta_nl.mean(), ci_low=float(np.nanpercentile(mb, 2.5)),
                                   ci_high=float(np.nanpercentile(mb, 97.5)),
                                   p_greater=float((1 + (mb <= 0).sum()) / (1 + np.isfinite(mb).sum())),
                                   p_less=float((1 + (mb >= 0).sum()) / (1 + np.isfinite(mb).sum())),
                                   saes_delta_positive=int((ps[sel].delta_nl > 0).sum())))
    pd.DataFrame(per_sae).to_csv(a.out / 'per_sae.csv', index=False)
    lr = pd.DataFrame(layer_rows)
    for fam in lr.family.unique():
        m = lr.family == fam
        lr.loc[m, 'q_greater_bh'] = bh(lr.loc[m, 'p_greater'].to_numpy())
        lr.loc[m, 'q_less_bh'] = bh(lr.loc[m, 'p_less'].to_numpy())
    lr['persona_specific'] = (lr.ci_low > 0) & (lr.q_greater_bh < 0.05)
    lr['neutral_loss_larger'] = (lr.ci_high < 0) & (lr.q_less_bh < 0.05)
    lr.to_csv(a.out / 'by_layer.csv', index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(a.out / 'reproduction_checks.csv', index=False)
    info = dict(B=B, boot_seed=BOOT_SEED, test_rows_used=0, max_abs_diff_vs_m2=float(ck.abs_diff.max()) if len(ck) else None,
                layers_persona_specific_old=lr[(lr.family == 'old') & lr.persona_specific].layer.tolist(),
                layers_neutral_larger_old=lr[(lr.family == 'old') & lr.neutral_loss_larger].layer.tolist(),
                runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))


if __name__ == '__main__':
    main()
