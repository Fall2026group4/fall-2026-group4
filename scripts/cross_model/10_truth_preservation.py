#!/usr/bin/env python3
"""XT-G / DX: truth and honesty representation preservation, one protocol for GPT-2 and Gemma (2026-11-21).

Pre-registration: results/cross_model/xtg_truth_preservation/preregistration.json.
Unchanged project code does the probing:
  * scripts/shared/m2_four_level_faithfulness.py: original_side, evaluate_source (got_cities, repe_persona, repe_neutral)
  * scripts/repe/26_repe_lm_representation_preservation.py: original_side, evaluate (synonym, held-out wording T5/T6,
    neutral, pair-direction geometry)
New code: task loading for Gemma (same splits), SAE/PCA sources for both models, and the pre-registered DX
diagnostics (affine / Procrustes alignment fitted on unlabeled OWT-VAL windows, CKA, truth-projection fidelity,
error enrichment along the truth direction). VALIDATION only: no TEST row is loaded.

Run: python scripts/cross_model/10_truth_preservation.py --model gemma --saes canonical --layers all --tag main
     python scripts/cross_model/10_truth_preservation.py --model gpt2 --saes resjb,r2,old --layers triplets --tag main
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
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

OUT = REPO / 'results/cross_model/xtg_truth_preservation'
GACT = REPO / 'data/gemma/activations'
TRIP = {'gpt2': [1, 3, 6, 8, 9, 11], 'gemma': [1, 5, 12, 16, 19, 23]}
NL = {'gpt2': 12, 'gemma': 26}
SEED = 20261121


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M2 = _load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
S26 = _load('scripts/repe/26_repe_lm_representation_preservation.py', 's26')
XB = _load('scripts/cross_model/02_matched_splicing.py', 'xb')


# ------------------------------------------------------------------------------------------------ tasks
FRESH = {'got_companies': 'companies_true_false', 'got_common_claim': 'common_claim_true_false'}


def add_fresh(tasks, layers, path_of):
    """Amendment A2: fresh GoT families, DEV rows only (the rows Gemma's extraction used); an internal 60/40
    group split (entity / statement groups, seed 20261121) gives train / validation. TEST rows are never read."""
    for tn, fam in FRESH.items():
        rows = pd.read_csv(GACT / f'got_fresh/{fam}_dev_rows.csv')
        g = rows.group_id.astype(str).to_numpy()
        u = np.unique(g)
        tr_groups = set(np.random.default_rng(SEED).permutation(u)[: int(round(0.6 * len(u)))])
        split = np.array(['train' if x in tr_groups else 'validation' for x in g])
        tasks[tn] = dict(X={L: np.load(path_of(fam, L)).astype(np.float32) for L in layers},
                         y=rows.label.to_numpy(), split=split, groups=g)
    return tasks


def gemma_tasks(layers):
    rows = []
    for name in ('cities', 'neg_cities'):
        df = pd.read_csv(REPO / f'data/got/{name}.csv')
        sp = pd.read_csv(REPO / f'data/got/splits/{name}_splits.csv')
        assert (sp.row_id.to_numpy() == np.arange(len(df))).all() and (sp.label.to_numpy() == df.label.to_numpy()).all()
        rows.append((name, sp))
    split = np.concatenate([s.split.to_numpy() for _, s in rows])
    keep = np.isin(split, ['train', 'val'])
    X = {L: np.concatenate([np.load(GACT / f'got/{n}_final_layer{L}.npy') for n, _ in rows]).astype(np.float32)[keep]
         for L in layers}
    tasks = {'got_cities': dict(X=X, y=np.concatenate([s.label.to_numpy() for _, s in rows])[keep],
                                split=np.where(split[keep] == 'val', 'validation', split[keep]),
                                groups=np.concatenate([s.group_id.to_numpy() for _, s in rows])[keep])}
    assert 'test' not in set(tasks['got_cities']['split'])
    d = pd.read_csv(REPO / 'data/repe/repe_lm/prompts.csv')
    acts = {L: np.load(GACT / f'repe_lm/final_layer{L}.npy').astype(np.float32) for L in layers}
    for name, m in [('repe_persona', d.template.isin(['T1', 'T2', 'T3', 'T4']) & d.variant.isin(['H', 'U'])),
                    ('repe_neutral', d.template.isin(['N1', 'N2']))]:
        idx = np.flatnonzero(m.to_numpy())
        tasks[name] = dict(X={L: acts[L][idx] for L in layers}, y=d.label.to_numpy()[idx],
                           split=d.split.to_numpy()[idx], groups=d.norm_group.to_numpy()[idx])
    add_fresh(tasks, layers, lambda fam, L: GACT / f'got_fresh/{fam}_dev_final_layer{L}.npy')
    return tasks, acts


def gpt2_tasks(layers):
    tasks = M2.load_tasks()
    add_fresh(tasks, layers, lambda fam, L: REPO / f'data/got/activations/gpt2_small/fresh_dev_all_layers/{fam}_dev_final_layer{L}.npy')
    acts = {L: np.load(REPO / f'data/repe/activations/repe_lm/final_layer{L}.npy').astype(np.float32) for L in layers}
    return tasks, acts


def owt_refs(model, layers):
    """{L: (mu, eigvals, V, X_owt_val)} from each model's own OWT train/val windows (same documents)."""
    out = {}
    if model == 'gemma':
        for L in layers:
            z = np.load(GACT / f'owt/train_pca_layer{L}.npz')
            out[L] = (z['mu'], z['eigvals'], z['eigvecs'].astype(np.float64),
                      np.load(GACT / f'owt/val_layer{L}.npy').astype(np.float32))
        return out
    stats, _ = M2.owt_pca()
    for L in layers:
        mu, cov = stats[L]
        w, V = il.pca_basis(mu, cov)
        out[L] = (mu, w, V, M2.owt_val_layer(L))
    return out


# ------------------------------------------------------------------------------------------------ DX
def fit_affine(Hs, Xs, lam_rel=1e-3):
    """Ridge map x ~ [x_hat, 1] A on unlabeled OWT windows."""
    Hs, Xs = Hs.astype(np.float64), Xs.astype(np.float64)
    hm, xm = Hs.mean(0), Xs.mean(0)
    Hc, Xc = Hs - hm, Xs - xm
    G = Hc.T @ Hc / len(Hc)
    lam = lam_rel * np.trace(G) / G.shape[0]
    A = np.linalg.solve(G + lam * np.eye(G.shape[0]), Hc.T @ Xc / len(Hc))
    return lambda H: ((H - hm) @ A + xm).astype(np.float32)


def fit_procrustes(Hs, Xs):
    Hs, Xs = Hs.astype(np.float64), Xs.astype(np.float64)
    hm, xm = Hs.mean(0), Xs.mean(0)
    U, _, Vt = np.linalg.svd((Hs - hm).T @ (Xs - xm), full_matrices=False)
    R = U @ Vt
    return lambda H: ((H - hm) @ R + xm).astype(np.float32)


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def dx_cell(meta, t, H, orig, aff, proc, boot_idx):
    tr, va = t['split'] == 'train', t['split'] == 'validation'
    X, y = t['X_L'], t['y']
    yv, Xv, Hv = y[va], X[va], H[va]
    r = dict(meta)
    base = orig['auroc']['logistic']
    s_raw = orig['pipe'].decision_function(Hv)
    s_aff = orig['pipe'].decision_function(aff(Hv))
    s_pro = orig['pipe'].decision_function(proc(Hv))
    r['dx_raw_frozen_gap'] = base - il.auroc(yv, s_raw)
    r['dx_affine_frozen_gap'] = base - il.auroc(yv, s_aff)
    r['dx_procrustes_frozen_gap'] = base - il.auroc(yv, s_pro)
    ob = il.boot_auroc(yv, orig['scores']['logistic'], boot_idx)
    for k, s in (('affine', s_aff), ('procrustes', s_pro)):
        r[f'dx_{k}_frozen_gap_ci_low'], r[f'dx_{k}_frozen_gap_ci_high'] = il.ci(ob - il.boot_auroc(yv, s, boot_idx))
    r['dx_cka'] = cka(Xv.astype(np.float64), Hv.astype(np.float64))
    w = orig['direction'] / np.linalg.norm(orig['direction'])
    px, ph = Xv @ w, Hv @ w
    r['dx_truth_proj_corr'] = float(np.corrcoef(px, ph)[0, 1])
    r['dx_truth_proj_var_ratio'] = float(ph.var() / px.var())
    E = (Xv - Hv).astype(np.float64)
    d = X.shape[1]
    enr = (E @ w) ** 2 / ((E ** 2).sum(1) / d)
    r['dx_error_truth_enrichment'] = float(enr.mean())
    bs = np.array([enr[i].mean() for i in boot_idx])
    r['dx_error_truth_enrichment_ci_low'], r['dx_error_truth_enrichment_ci_high'] = il.ci(bs)
    # A2: non-saturating separation (d') of the logistic scores: original, frozen on x_hat, refit on x_hat
    def dprime(sc):
        a, b = sc[yv == 1], sc[yv == 0]
        return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))
    rp, _ = il.fit_logistic(H[tr], y[tr], t['groups'][tr])
    r['dx_dprime_orig'] = dprime(orig['scores']['logistic'])
    r['dx_dprime_frozen'] = dprime(s_raw)
    r['dx_dprime_refit'] = dprime(rp.decision_function(Hv))
    r['dx_dprime_refit_retention'] = r['dx_dprime_refit'] / r['dx_dprime_orig']
    r['dx_orig_auroc'] = base
    r['dx_ceiling_flag'] = bool(base >= 0.99)
    # also on the label-contrast direction refit on reconstructions (does the SAE keep *a* truth direction?)
    wh = il.massmean_direction(H[tr], y[tr])
    r['dx_massmean_dir_cosine'] = float(w @ (wh / np.linalg.norm(wh)))
    return r


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gpt2', 'gemma'])
    ap.add_argument('--saes', required=True)
    ap.add_argument('--layers', default='all')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    run = f'{a.model}_{a.tag}'
    out = OUT / ('smoke' if a.smoke else 'runs') / run
    # A2: one done-marker per invocation's layer list (the queue runs one layer per process to bound host RAM)
    out.mkdir(parents=True, exist_ok=True)
    layers = (list(range(NL[a.model])) if a.layers == 'all' else TRIP[a.model] if a.layers == 'triplets'
              else [int(x) for x in a.layers.split(',')])
    if a.smoke:
        layers = layers[:1]
    done = set()
    if (out / 'per_cell.csv').exists():
        done = set(pd.read_csv(out / 'per_cell.csv').layer.unique())
    t0 = time.time()
    tasks, repe_acts = (gemma_tasks if a.model == 'gemma' else gpt2_tasks)(layers)
    refs = owt_refs(a.model, layers)
    d, idx, y_r, y_hhp, g_r, sid, neardup, _ = S26.load_rows()
    which = a.saes.split(',')
    dev = 'cuda'
    for L in layers:
        if L in done:
            continue
        rng = np.random.default_rng(SEED + L)
        origs = {tn: M2.original_side(t, L, rng) for tn, t in tasks.items()}
        S26.G.clear()
        S26.G.update(idx=idx, y=y_r, y_hhp=y_hhp, g=g_r, sid=sid, neardup=neardup, X=repe_acts[L])
        S26.G['orig'] = S26.original_side(repe_acts[L])
        orow = dict(layer=L, **{f'{tn}_orig_{k}_auroc': v for tn, o in origs.items() for k, v in o['auroc'].items()},
                    **{f'repe_orig_{k}_auroc': v for k, v in S26.G['orig']['auc'].items()})
        mu, w, V, xv = refs[L]
        specs = XB.sae_specs(a.model, L, which)
        # A3: GPT-2 scope (old in-house: seed 42 only, all seeds already in M2 under the same code; R2: the three
        # main architectures only); PCA references are de-duplicated by (kind, k) within a layer.
        specs = [sp for sp in specs if not (sp['family'].startswith('old_') and sp['seed'] != 42)
                 and not (sp['family'] == 'r2' and sp['sae_id'].split('/')[-1] not in ('topk', 'batchtopk', 'relu'))]
        seen_pca = set()
        jobs, s26jobs = [], []
        bidx = {tn: il.bootstrap_indices(t['groups'][t['split'] == 'validation'], M2.N_BOOT,
                                         np.random.default_rng(SEED + 100 + L)) for tn, t in tasks.items()}
        dx = []
        for s in specs:
            s['layer'] = L
            sae = XB.load_sae(a.model, s, dev)

            def enc(X, parts=8):
                outs = [sae(torch.as_tensor(np.ascontiguousarray(c)).to(dev)) for c in np.array_split(X, parts)]
                return (np.concatenate([o[0].float().cpu().numpy() for o in outs]),
                        np.concatenate([o[1].cpu().numpy() for o in outs]))
            Ho, l0o = enc(xv)
            nm = M2.nmse(xv, Ho)
            l0_owt = float(l0o.mean())
            k_err = il.pca_k_for_val_nmse(xv, mu, V, nm)
            k_l0 = int(max(1, round(l0_owt)))
            for kind, k in (('sae', None), ('pca_err', k_err), ('pca_l0', k_l0)):
                if kind != 'sae':
                    if (kind, k) in seen_pca:
                        continue
                    seen_pca.add((kind, k))
                recon = (lambda X: enc(X)) if kind == 'sae' else \
                    (lambda X, k=k: (il.pca_reconstruct(X.astype(np.float64), mu, V, k).astype(np.float32),
                                     np.full(len(X), k)))
                Ho_k = Ho if kind == 'sae' else recon(xv)[0]
                aff, proc = fit_affine(Ho_k, xv), fit_procrustes(Ho_k, xv)
                meta = dict(model=a.model, layer=L, depth=(L / 12) if a.model == 'gpt2' else (L + 1) / 26,
                            family=s['family'], sae_id=s['sae_id'], seed=s['seed'], source=kind, pca_k=k,
                            owt_val_nmse_sae=nm, owt_val_l0_sae=l0_owt, d_sae=sae.d_sae,
                            owt_val_nmse_source=M2.nmse(xv, Ho_k))
                for tn, t in tasks.items():
                    X = t['X'][L]
                    H, Z = recon(X)
                    jobs.append((dict(meta, task=tn), X, H, None, t, tn))
                    if True:   # A2: DX on every task
                        dx.append(dx_cell(dict(meta, task=tn), dict(t, X_L=X), H, origs[tn], aff, proc, bidx[tn]))
                Hr, l0r = recon(repe_acts[L])
                s26jobs.append((dict(meta), Hr, l0r))
            del sae
            torch.cuda.empty_cache()
        res = Parallel(n_jobs=a.jobs, backend='loky', max_nbytes='1M')(
            delayed(M2.evaluate_source)(m, X, H, Z, t['y'], t['split'], t['groups'], M2.strip(origs[tn]), bidx[tn],
                                        SEED + 1000 * L + i)
            for i, (m, X, H, Z, t, tn) in enumerate(jobs))
        rows = [r for r, _ in res]
        r26 = Parallel(n_jobs=a.jobs, backend='threading')(       # S26 reads its module-level state G
            delayed(S26.evaluate)(m, Hr, None, l0r, np.nan) for (m, Hr, l0r) in s26jobs)
        _append(out / 'per_cell.csv', pd.DataFrame(rows))
        _append(out / 'dx.csv', pd.DataFrame(dx))
        _append(out / 'repe_extras.csv', pd.DataFrame(r26))
        _append(out / 'original_by_layer.csv', pd.DataFrame([orow]))
        print(f'layer {L}: {len(specs)} SAEs, {len(jobs)} cells ({(time.time() - t0) / 60:.1f} min)', flush=True)
    (out / f"DONE_layers_{a.layers.replace(',', '-')}.json").write_text(json.dumps(dict(args=vars(a), layers=layers, minutes=(time.time() - t0) / 60,
                                                   script_sha256=XB.ps.sha256_file(Path(__file__))), indent=1))


def _s26_eval(state, m, Hr, l0r):
    S26.G.clear()
    S26.G.update(state)
    return S26.evaluate(m, Hr, None, l0r, np.nan)


def _append(path, df):
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


if __name__ == '__main__':
    main()
