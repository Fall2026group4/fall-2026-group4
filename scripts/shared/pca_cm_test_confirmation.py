#!/usr/bin/env python3
"""PCA-CM-T: TEST confirmation of the capacity-matched PCA comparison (rules frozen 2026-11-13; run 2026-11-21).

Frozen protocol: results/shared/pca_capacity_matched_test_protocol/{README.md, preregistration.json}.
* GoT cities + neg_cities; readers fitted on TRAIN rows (M5 / PCA-CM `M2.evaluate_source`, unchanged).
* Every SAE in the M5 source list at layers 8 and 11, exactly as PCA-CM (same source order, same bootstrap and
  reader seeds), against PCA fitted on OWT TRAIN at
    - capacity_k128   (k = 128),
    - task_l0_matched (k = round(SAE GoT task L0) taken from the committed VALIDATION `got_paired.csv`, never TEST).
* Statistic: per seed, paired grouped-bootstrap 95 % CI of (SAE re-fit gap - PCA re-fit gap), 1,000 resamples.
* Families P1/P2/P3 and the Holm rule (normal approximation, intersection-union over seeds) as in
  `pca_cm_posthoc_multiplicity.py`.

Modes (outputs refuse to be overwritten):
  --validate DIR  eval split = VALIDATION; must reproduce results/shared/pca_capacity_matched/got_paired.csv
                  (the implementation rule: "smoke-tested on VALIDATION before it reads a TEST row").
  --test          eval split = TEST; refuses to run unless addendum_pre_test.json exists and its hashes (this
                  script, the frozen protocol, the validation reproduction) match.
The OWT part of PCA-CM is not repeated (not in the frozen families). Nothing is trained.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _load('scripts/shared/pca_capacity_matched.py', 'pcacm')
M5, LAD, M2 = CM.M5, CM.LAD, CM.M2
VAL = REPO / 'results/shared/pca_capacity_matched'
PROT = REPO / 'results/shared/pca_capacity_matched_test_protocol'
REFS = ('capacity_k128', 'task_l0_matched')
ALPHA = 0.05
FAMILIES = {'P2': ['L8 r2 relu', 'L11 r2 relu', 'L8 resjb'],
            'P3': ['L8 old relu', 'L8 r2 relu', 'L8 resjb', 'L11 old jumprelu', 'L11 old relu',
                   'L11 r2 batchtopk', 'L11 r2 relu', 'L11 resjb']}
FAMILY_REF = {'P2': 'capacity_k128', 'P3': 'task_l0_matched'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell_key(layer, family, name):
    return f'L{layer} {family}' if family == 'resjb' else f'L{layer} {family} {name}'


def holm(p, alpha=ALPHA):
    order, m = np.argsort(p), len(p)
    rej = np.zeros(m, dtype=bool)
    for i, ix in enumerate(order):
        if p[ix] <= alpha / (m - i):
            rej[ix] = True
        else:
            break
    return rej


def run_split(eval_split, task_k):
    """Return got_paired rows (capacity_k128 and task_l0_matched only) for eval_split in ('val', 'test')."""
    import torch
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    got, n_eval = M5.got_cities(eval_split)
    pca_stats, n_pca = M2.owt_pca(False)
    paired, info = [], dict(eval_split=eval_split, n_got_eval_rows=n_eval, n_pca_train_tokens=int(n_pca))
    for L in CM.GOT_LAYERS:
        mu, cov = pca_stats[L]
        _, V = il.pca_basis(mu, cov)
        orig = M2.original_side(got, L, np.random.default_rng(M2.SEED + L))
        bidx = il.bootstrap_indices(got['groups'][got['split'] == 'validation'], M2.N_BOOT,
                                    np.random.default_rng(M2.SEED + 100 + L))
        Xg = got['X'][L]
        info[f'L{L}_original_auroc'] = orig['auroc']['logistic']
        cache = {}

        def pca_eval(k):
            if k not in cache:
                Hp = il.pca_reconstruct(Xg.astype(np.float64), mu, V, k).astype(np.float32)
                cache[k] = M2.evaluate_source(dict(), Xg, Hp, None, got['y'], got['split'], got['groups'], orig,
                                              bidx, CM.SEED + 1000 * L + 7 * k)
            return cache[k]

        # identical source order to PCA-CM, so the per-source reader seed SEED + 1000 L + i is identical
        for i, src in enumerate(LAD.sources(L, list(CM.SOURCES))):
            sae = LAD.load(src, L, device)
            H, Z = LAD.encode_np(sae, Xg, device)
            rs, bs = M2.evaluate_source(dict(), Xg, H, Z, got['y'], got['split'], got['groups'], orig, bidx,
                                        CM.SEED + 1000 * L + i)
            key = (L, src['family'], src['name'], int(src['seed']))
            refs = {'capacity_k128': CM.K_CAP, 'task_l0_matched': task_k[key]}
            for ref, k in refs.items():
                rp, bp = pca_eval(k)
                d = bs['refit_logistic_gap'] - bp['refit_logistic_gap']
                lo, hi = il.ci(d)
                paired.append(dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'], reference=ref,
                                   pca_k=k, sae_task_l0=rs['task_l0'], sae_refit_logistic_gap=rs['refit_logistic_gap'],
                                   pca_refit_logistic_gap=rp['refit_logistic_gap'],
                                   diff_refit_logistic_gap=rs['refit_logistic_gap'] - rp['refit_logistic_gap'],
                                   diff_refit_logistic_gap_ci_low=lo, diff_refit_logistic_gap_ci_high=hi,
                                   checkpoint_sha256=sha(src['path'])))
            print(f"L{L} {src['family']} {src['name']} {src['seed']} ({time.time()-t0:.0f}s)", flush=True)
            del sae
            torch.cuda.empty_cache()
    info['runtime_s'] = time.time() - t0
    return pd.DataFrame(paired), info


def cells_and_families(p):
    p = p[p.name != 'topk_frozen_decoder'].copy()
    se = (p.diff_refit_logistic_gap_ci_high - p.diff_refit_logistic_gap_ci_low) / 3.92
    p['z'] = p.diff_refit_logistic_gap / se
    keys = ['layer', 'family', 'name', 'reference']
    cells = p.groupby(keys).agg(n_seeds=('seed', 'count'), diff_mean=('diff_refit_logistic_gap', 'mean'),
                                sae_gap_mean=('sae_refit_logistic_gap', 'mean'),
                                pca_gap_mean=('pca_refit_logistic_gap', 'mean'),
                                z_weakest=('z', 'max'), ci_high_max=('diff_refit_logistic_gap_ci_high', 'max'),
                                ci_low_min=('diff_refit_logistic_gap_ci_low', 'min'),
                                pca_k=('pca_k', lambda s: ';'.join(map(str, sorted(set(s)))))).reset_index()
    cells['p_cell'] = np.where(cells.z_weakest < 0, 2 * norm.sf(cells.z_weakest.abs()), 1.0)
    cells['cls'] = [CM.classify(g.diff_refit_logistic_gap_ci_low, g.diff_refit_logistic_gap_ci_high)
                    for _, g in p.groupby(keys)]
    cells['cell'] = [cell_key(r.layer, r.family, r.name) for r in cells.itertuples()]
    fam = {}
    c128 = cells[cells.reference == 'capacity_k128']
    fam['P1'] = dict(rule='no trained cell at capacity_k128 classified SAE WORSE', n_cells=len(c128),
                     n_sae_worse=int((c128.cls == 'SAE WORSE').sum()),
                     sae_worse_cells=c128[c128.cls == 'SAE WORSE'].cell.tolist(),
                     confirmed=bool((c128.cls == 'SAE WORSE').sum() == 0))
    cells['family_member'] = ''
    for F, members in FAMILIES.items():
        sub = cells[(cells.reference == FAMILY_REF[F]) & cells.cell.isin(members)].copy()
        assert sorted(sub.cell) == sorted(members), (F, sorted(sub.cell))
        sub['holm_reject'] = holm(sub.p_cell.to_numpy())
        sub['confirmed'] = (sub.ci_high_max < 0) & sub.holm_reject
        fam[F] = dict(reference=FAMILY_REF[F], n_cells=len(sub), n_confirmed=int(sub.confirmed.sum()),
                      cells={r.cell: dict(diff_mean=r.diff_mean, ci_high_max=r.ci_high_max, p_cell=r.p_cell,
                                          holm_reject=bool(r.holm_reject), confirmed=bool(r.confirmed))
                             for r in sub.itertuples()})
        for r in sub.itertuples():
            cells.loc[r.Index, 'family_member'] = (cells.loc[r.Index, 'family_member'] + ' ' + F).strip()
    return cells, fam


def main():
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument('--validate', type=Path)
    gx.add_argument('--test', action='store_true')
    a = ap.parse_args()
    pr = json.loads((PROT / 'preregistration.json').read_text())
    for rel, h in pr['hashes'].items():
        if sha(REPO / rel) != h:
            raise SystemExit(f'{rel}: hash differs from the frozen protocol; refusing to run.')
    vp = pd.read_csv(VAL / 'got_paired.csv')
    task_k = {(int(r.layer), r.family, r.name, int(r.seed)): int(r.pca_k)
              for r in vp[vp.reference == 'task_l0_matched'].itertuples()}
    out = a.validate if a.validate else PROT / 'test'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    if a.test:
        add = PROT / 'addendum_pre_test.json'
        if not add.exists():
            raise SystemExit('addendum_pre_test.json missing: validate and freeze before reading TEST.')
        ad = json.loads(add.read_text())
        for rel, h in ad['hashes'].items():
            if sha(REPO / rel) != h:
                raise SystemExit(f'{rel}: hash differs from the addendum; refusing to run.')
        if not ad.get('validation_reproduced'):
            raise SystemExit('validation reproduction not recorded as passed; refusing to run.')
    paired, info = run_split('test' if a.test else 'val', task_k)
    out.mkdir(parents=True)
    paired.to_csv(out / 'got_paired.csv', index=False)
    cells, fam = cells_and_families(paired)
    cells.to_csv(out / 'cells.csv', index=False)
    info.update(mode='test' if a.test else 'validate', script_sha256=sha(__file__), n_paired=len(paired),
                test_label=pr['test_label'] if a.test else 'VALIDATION reproduction')
    if not a.test:      # reproduction check against the committed PCA-CM validation outputs
        m = paired.merge(vp, on=['layer', 'family', 'name', 'seed', 'reference'], suffixes=('', '_v'))
        cols = ['pca_k', 'sae_refit_logistic_gap', 'pca_refit_logistic_gap', 'diff_refit_logistic_gap_ci_low',
                'diff_refit_logistic_gap_ci_high']
        info['reproduction'] = {c: float(np.max(np.abs(m[c] - m[c + '_v']))) for c in cols}
        info['reproduction_rows'] = [len(m), int(vp.reference.isin(REFS).sum())]
        info['validation_reproduced'] = bool(len(m) == vp.reference.isin(REFS).sum()
                                             and max(info['reproduction'].values()) < 1e-6)
    (out / 'families.json').write_text(json.dumps(fam, indent=2, default=float))
    (out / 'checks.json').write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != 'reproduction_rows'}, default=float, indent=1))
    print(json.dumps({F: {k: v for k, v in d.items() if k != 'cells'} for F, d in fam.items()}, default=float))


if __name__ == '__main__':
    main()
