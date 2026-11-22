#!/usr/bin/env python3
"""TSP-D: root-cause diagnostics for the GoT truth-preservation failures (2026-11-14). VALIDATION only; no training.

Questions (Failure A: truth-aware fine-tuning; Failure B: R2 did not improve GoT truth; Failure C: R4):
  D1  Is the GoT truth subspace S (estimated on GoT-train group A only) LOW-VARIANCE on OWT, the SAE training data?
  D2  Does each SAE lose more of the signal inside S than inside a same-rank random or label-free PCA subspace?
  D3  Is truth still decodable from the SAE ERROR alone (information displaced into the residual)?
  D4  Group-B-only readers: frozen logistic (fitted on original activations) and re-fit logistic / MLP on x_hat.
  D5  Truth-aware pilot (L6 TopK): error inside the teacher's direction vs inside S-orthogonal directions.

Independence: S and the teacher direction come from group A; every probe is fitted on group-B TRAIN rows only;
every number is on VALIDATION cities (disjoint from both). Test rows are dropped on load (asserted).
Outputs: results/geometry_of_truth/truth_subspace_preservation/diagnostics/ (refuses to overwrite).
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
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import information_levels as il  # noqa: E402
from src.sae import truth_subspace as ts  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/geometry_of_truth/truth_subspace_preservation/diagnostics'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
R4 = REPO / 'results/openwebtext/r4_kl_finetune/checkpoints'
TA = REPO / 'results/geometry_of_truth/sae_finetuning/truth_aware_pilot_layer6_topk'
SEEDS = (42, 1042, 2042)
RANK, SEED = 8, 20261114


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_got(layers):
    """GoT geographic (cities + neg_cities): train/val only, with polarity and the A/B half of each train group."""
    got03 = _mod('scripts/geometry_of_truth/03_original_truth_probes.py', 'got03')
    got03.ACT_DIR, got03.SPLIT_DIR = REPO / got03.ACT_DIR, REPO / got03.SPLIT_DIR
    parts = got03.load_domain(got03.DOMAINS['geographic'])
    split = np.concatenate([p['splits'] for p in parts])
    keep = np.isin(split, ['train', 'val'])
    pol = np.concatenate([np.full(len(p['labels']), int(p['name'] == 'neg_cities')) for p in parts])[keep]
    groups = np.concatenate([p['groups'] for p in parts])[keep]
    ab = pd.read_csv(REPO / 'data/got/splits/train_AB_groups_truth_aware_pilot.csv')
    ab = dict(zip(ab[ab.domain == 'geographic'].group_id.astype(str), ab[ab.domain == 'geographic'].half))
    sp = np.where(split[keep] == 'val', 'validation', split[keep])
    half = np.array([ab.get(g, '') if s == 'train' else '' for g, s in zip(groups, sp)])
    assert set(np.unique(half[sp == 'train'])) == {'A', 'B'}, 'every train group must be in A or B'
    X = {L: np.concatenate([p['layers'][L].numpy() for p in parts]).astype(np.float32)[keep] for L in layers}
    t = dict(X=X, y=np.concatenate([p['labels'] for p in parts])[keep], split=sp, groups=groups, pol=pol, half=half)
    assert 'test' not in set(sp)
    tr, va = sp == 'train', sp == 'validation'
    assert not (set(groups[tr]) & set(groups[va]))
    assert not (set(groups[half == 'A']) & set(groups[half == 'B']))
    return t


def owt_cov(L):
    o3 = _mod('scripts/openwebtext/03_training_sae.py', 'o3')
    x = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', L).double()
    return torch.cov(x.T).numpy()


def sources(L):
    out = []
    if L in (8, 11):
        out += [dict(family='old', name=a, seed=s, path=Path(uninterrupted_checkpoint(L, a, s)), arch=a)
                for a in ('topk', 'relu', 'jumprelu') for s in SEEDS]
        out += [dict(family='r2', name=n, seed=s, path=R2 / f'layer_{L}/{n}_seed{s}/checkpoint_final.pt', arch=None)
                for n in ('topk', 'batchtopk', 'relu') for s in SEEDS]
        out += [dict(family=f'r4_{a}', name='topk', seed=s, path=R4 / f'layer_{L}/{a}_topk_seed{s}/checkpoint.pt',
                     arch=None) for a in ('kl', 'mse') for s in SEEDS]
    if L == 6:
        out += [dict(family='base_l6', name='topk', seed=s, path=Path(uninterrupted_checkpoint(6, 'topk', s)),
                     arch='topk') for s in SEEDS]
        out += [dict(family=f'ta_{v}', name='topk', seed=s, path=TA / f'{v}_seed{s}/finetuned_checkpoint.pt',
                     arch='topk') for v in ('reconstruction_only', 'truth_aware', 'random_teacher') for s in SEEDS]
    return [s for s in out if s['path'].exists()]


@torch.no_grad()
def reconstruct(sae, X, device):
    H, Z = [], []
    for b in torch.as_tensor(X).split(2048):
        h, z, _ = sae(b.to(device))
        H.append(h.float().cpu())
        Z.append(z.float().cpu())
    return torch.cat(H).numpy(), torch.cat(Z).numpy()


def b_readers(Xtr, ytr, gtr, Xva, yva):
    pipe, C = il.fit_logistic(Xtr, ytr, gtr)
    return dict(logistic=il.auroc(yva, pipe.decision_function(Xva)),
                mlp=il.auroc(yva, il.mlp_scores(Xtr, ytr, Xva))), pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', default='6,8,11')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    layers = [int(x) for x in a.layers.split(',')]
    device, t0 = ('cuda' if torch.cuda.is_available() else 'cpu'), time.time()
    t = load_got(layers)
    y, sp, g, pol, half = t['y'], t['split'], t['groups'], t['pol'], t['half']
    A, B, VA = (sp == 'train') & (half == 'A'), (sp == 'train') & (half == 'B'), sp == 'validation'
    rows, sub_rows = [], []
    for L in layers:
        X = t['X'][L]
        P, info = ts.estimate_truth_subspace(X[A], y[A], pol[A], g[A], r=RANK, seed=SEED + L)
        Pr, Pp = ts.random_subspace(X.shape[1], P.shape[1], SEED + L), ts.pca_subspace(X[A], P.shape[1])
        w_teacher = ts.logistic_raw_weight(X[A], y[A], C=0.1)
        w_teacher = (w_teacher / np.linalg.norm(w_teacher))[:, None]
        Q, _ = np.linalg.qr(np.concatenate([P, np.random.default_rng(SEED).standard_normal((768, 768 - P.shape[1]))], 1))
        Pperp = Q[:, P.shape[1]:]
        cov = owt_cov(L)
        got_cov = np.cov(X[A].astype(np.float64).T)
        orig, pipeB = b_readers(X[B], y[B], g[B], X[VA], y[VA])
        sub_rows.append(dict(layer=L, rank=P.shape[1], n_A=int(A.sum()), n_B=int(B.sum()), n_val=int(VA.sum()),
                             owt_var_share_truth=ts.variance_share(cov, P), owt_var_share_random=ts.variance_share(cov, Pr),
                             owt_var_share_pca=ts.variance_share(cov, Pp),
                             got_var_share_truth=ts.variance_share(got_cov, P),
                             owt_var_share_teacher_dir=ts.variance_share(cov, w_teacher),
                             orig_B_logistic_auroc=orig['logistic'], orig_B_mlp_auroc=orig['mlp'],
                             truth_subspace_only_B_logistic_auroc=b_readers(X[B] @ P, y[B], g[B], X[VA] @ P, y[VA])[0]['logistic'],
                             **{f'subspace_{k}': v for k, v in info.items() if k != 'singular_values'}))
        srcs = sources(L)[: (3 if a.smoke else None)]
        for s in srcs:
            sae, _ = fm.load_sae(s['path'], device, architecture=s['arch'])
            H, Z = reconstruct(sae.eval(), X, device)
            E = X - H
            c = X[A].mean(0)
            r = dict(layer=L, family=s['family'], name=s['name'], seed=s['seed'],
                     val_l0=float((Z[VA] > 0).sum(1).mean()),
                     val_nmse=float(((E[VA]) ** 2).sum() / ((X[VA] - X[VA].mean(0)) ** 2).sum()),
                     relerr_truth=ts.subspace_rel_error(X[VA], H[VA], P, c),
                     relerr_random=ts.subspace_rel_error(X[VA], H[VA], Pr, c),
                     relerr_pca=ts.subspace_rel_error(X[VA], H[VA], Pp, c),
                     relerr_perp=ts.subspace_rel_error(X[VA], H[VA], Pperp, c),
                     relerr_teacher_dir=ts.subspace_rel_error(X[VA], H[VA], w_teacher, c))
            Hc = H[VA] + (E[VA].mean(0) - 0)          # remove the mean (bias) error: error centred over validation rows
            for nm, Q in (('truth', P), ('random', Pr), ('pca', Pp), ('perp', Pperp), ('teacher_dir', w_teacher)):
                r[f'relerr_centred_{nm}'] = ts.subspace_rel_error(X[VA], Hc, Q, c)
                r[f'bias_share_{nm}'] = 1.0 - r[f'relerr_centred_{nm}'] / r[f'relerr_{nm}']
            rec, _ = b_readers(H[B], y[B], g[B], H[VA], y[VA])
            err, _ = b_readers(E[B], y[B], g[B], E[VA], y[VA])
            r.update(refit_B_logistic_gap=orig['logistic'] - rec['logistic'], refit_B_mlp_gap=orig['mlp'] - rec['mlp'],
                     frozen_B_logistic_gap=orig['logistic'] - il.auroc(y[VA], pipeB.decision_function(H[VA])),
                     error_only_B_logistic_auroc=err['logistic'], error_only_B_mlp_auroc=err['mlp'])
            rows.append(r)
            print(f"L{L} {s['family']} {s['seed']}: relerr truth {r['relerr_truth']:.3f} rand {r['relerr_random']:.3f} "
                  f"pca {r['relerr_pca']:.3f}; refit-B gap {r['refit_B_logistic_gap']:.3f}; err-only "
                  f"{r['error_only_B_logistic_auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            del sae
    a.out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.out / 'per_sae.csv', index=False)
    pd.DataFrame(sub_rows).to_csv(a.out / 'subspace_by_layer.csv', index=False)
    (a.out / 'checks.json').write_text(json.dumps(dict(test_rows_used=0, rank=RANK, seed=SEED, layers=layers,
                                                       n_sources=len(rows), runtime_s=time.time() - t0), indent=2))


if __name__ == '__main__':
    main()
