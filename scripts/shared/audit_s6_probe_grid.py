#!/usr/bin/env python3
"""AUDIT-S6 (sensitivity, NOT confirmatory): do the re-trained-probe conclusions depend on the narrow C grid?

Problem (audit 2026-11-13): every logistic probe selects C from {0.01, 0.1, 1} by grouped 3-fold CV on TRAIN; the
original persona probe picks the boundary value 0.01 at 11 of 12 layers.

Protocol: the same selection procedure (il.select_C, grouped 3-fold CV on TRAIN, ties -> smaller C) with the extended
grid C in {1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}, for the original probe and for the probe re-fitted
on each SAE reconstruction; scored on VALIDATION. Sources: old ReLU/TopK/JumpReLU-3072 and R2 TopK/BatchTopK/ReLU-12288
(3 seeds) at layers 1, 8, 11; tasks repe_persona (1, 8, 11) and got_cities (8, 11). The R2 rule is re-applied to
C1-C3 on the re-fit gap; a class change versus the committed ladder is flagged GRID-SENSITIVE. A selected C at either
end of the extended grid is reported. No TEST row is read.
Outputs: results/shared/audit_sensitivity/s6_probe_grid/ (refuses to overwrite).
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
spec = importlib.util.spec_from_file_location('r2c', REPO / 'scripts/shared/r2_comparisons.py')
R2C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2C)

OUT = REPO / 'results/shared/audit_sensitivity/s6_probe_grid'
LADDER = REPO / 'results/openwebtext/r2_scaled_saes/ladder'
GRID = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ARMS = {('old', 'topk'), ('old', 'relu'), ('old', 'jumprelu'), ('r2', 'topk'), ('r2', 'relu'), ('r2', 'batchtopk')}
COMP = {'C1': (('r2', 'topk'), ('old', 'topk')), 'C2': (('r2', 'relu'), ('old', 'relu')),
        'C3': (('r2', 'batchtopk'), ('old', 'jumprelu'))}
TASK_LAYERS = {'repe_persona': (1, 8, 11), 'got_cities': (8, 11)}


def fit_ext(X, y, g):
    C = il.select_C(X, y, g, il.logistic, grid=GRID)
    return il.logistic(C).fit(X, y), C


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
    lad = pd.concat([pd.read_csv(p) for p in sorted(LADDER.glob('ladder_layer*.csv'))], ignore_index=True)
    rows, orig_rows = [], []
    for L in ((8,) if a.smoke else (1, 8, 11)):
        srcs = [s for s in LAD.sources(L, ['old', 'r2']) if (s['family'], s['name']) in ARMS
                and (not a.smoke or s['seed'] == 42)]
        tl = [tn for tn, ls in TASK_LAYERS.items() if L in ls]
        origs = {}
        for tn in tl:
            t = tasks[tn]
            tr, va = t['split'] == 'train', t['split'] == 'validation'
            X = t['X'][L]
            pipe, C = fit_ext(X[tr], t['y'][tr], t['groups'][tr])
            auc = il.auroc(t['y'][va], pipe.decision_function(X[va]))
            C0 = il.select_C(X[tr], t['y'][tr], t['groups'][tr], il.logistic)
            origs[tn] = auc
            orig_rows.append(dict(layer=L, task=tn, C_extended=C, C_default=C0, auroc_extended=auc,
                                  auroc_default=il.auroc(t['y'][va], il.logistic(C0).fit(X[tr], t['y'][tr]).decision_function(X[va])),
                                  at_grid_edge=bool(C in (GRID[0], GRID[-1]))))
        for src in srcs:
            sae = LAD.load(src, L, device)
            key = (src['family'], src['name'], src['seed'])
            r = dict(layer=L, family=key[0], name=key[1], seed=key[2])
            for tn in tl:
                t = tasks[tn]
                tr, va = t['split'] == 'train', t['split'] == 'validation'
                H, _ = LAD.encode_np(sae, t['X'][L], device)
                pipe, C = fit_ext(H[tr], t['y'][tr], t['groups'][tr])
                r[f'{tn}__refit_C_extended'] = C
                r[f'{tn}__refit_at_grid_edge'] = bool(C in (GRID[0], GRID[-1]))
                r[f'{tn}__refit_logistic_gap'] = origs[tn] - il.auroc(t['y'][va], pipe.decision_function(H[va]))
            del sae
            torch.cuda.empty_cache()
            rows.append(r)
            print(f'L{L} {key} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / 'per_sae.csv', index=False)
    pd.DataFrame(orig_rows).to_csv(a.out / 'original_probes.csv', index=False)
    if a.smoke:
        return
    comp = []
    for tn, ls in TASK_LAYERS.items():
        m = f'{tn}__refit_logistic_gap'
        for L in ls:
            dm, dn = df[df.layer == L], lad[lad.layer == L]
            for cid, (new, ref) in COMP.items():
                sel = lambda x, arm: x[(x.family == arm[0]) & (x.name == arm[1])].sort_values('seed')[m]
                re = R2C.rule(sel(dm, new), sel(dm, ref), False)
                rn = R2C.rule(sel(dn, new), sel(dn, ref), False)
                comp.append(dict(layer=L, comparison=cid, metric=m, default_grid_cls=rn['cls'], default_grid_diff=rn['diff'],
                                 extended_grid_cls=re['cls'], extended_grid_diff=re['diff'], ci_low=re['ci95_low'],
                                 ci_high=re['ci95_high'], grid_sensitive=bool(rn['cls'] != re['cls'])))
    cp = pd.DataFrame(comp)
    cp.to_csv(a.out / 'comparisons_default_vs_extended.csv', index=False)
    edge = {c: int(df[c].sum()) for c in df.columns if c.endswith('at_grid_edge')}
    info = dict(test_rows_used=0, grid=GRID, n_grid_sensitive=int(cp.grid_sensitive.sum()), refit_at_edge=edge,
                original_at_edge=int(pd.DataFrame(orig_rows).at_grid_edge.sum()), runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))


if __name__ == '__main__':
    main()
