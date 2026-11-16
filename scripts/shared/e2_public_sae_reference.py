#!/usr/bin/env python3
"""E2: external reference with the public gpt2-small-res-jb SAEs (results/shared/e2_public_sae_reference/README.md).

  --check     adapter equivalence (in-house class with mapped weights vs sae_lens' own forward on 4096 OWT
              validation rows, max abs diff < 1e-4) and SHA-256 hashes; writes checks.json
  --analyse   E2-H1..H5 from the ladder CSVs written by scripts/shared/sae_ladder_evaluation.py; writes
              hypotheses.csv and summary.json. Layers 8 and 10 are the pre-registered layers; layers 1 and 11 are a
              labelled extension (R1, 2026-11-10).
VALIDATION only; no test row is read.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / 'results/shared/e2_public_sae_reference'
PREREG_LAYERS = (8, 10)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(layers):
    lad = _load('scripts/shared/sae_ladder_evaluation.py', 'lad')
    o3 = lad.O3
    from sae_lens import SAE
    res = {}
    for L in layers:
        x = o3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L)[:4096]
        ours = lad.load_resjb(L, 'cpu')
        ref = SAE.from_pretrained('gpt2-small-res-jb', f'blocks.{L}.hook_resid_pre', device='cpu')
        ref = ref[0] if isinstance(ref, tuple) else ref
        with torch.no_grad():
            a = ours(x)[0]
            b = ref.decode(ref.encode(x))
        res[L] = dict(max_abs_diff=float((a - b).abs().max()), pass_=bool((a - b).abs().max() < 1e-4),
                      weights_sha256=sha(lad.resjb_path(L)))
        print(L, res[L], flush=True)
    files = ['scripts/shared/e2_public_sae_reference.py', 'scripts/shared/sae_ladder_evaluation.py',
             'scripts/shared/m2_four_level_faithfulness.py', 'scripts/repe/24_sae_patch_decomposition.py',
             'scripts/shared/spliced_reconstruction_evaluation.py', 'results/shared/e2_public_sae_reference/README.md']
    out = dict(adapter=res, hashes={f: sha(REPO / f) for f in files}, test_rows_used=0)
    p = OUT / 'checks.json'
    if p.exists():
        raise SystemExit(f'{p} exists; refusing to overwrite.')
    p.write_text(json.dumps(out, indent=2))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r in range(n - 1, -1, -1):
        prev = min(prev, p[o[r]] * n / (r + 1))
        q[o[r]] = prev
    return q


def analyse(ladder_dirs):
    files = sorted(p for d in ladder_dirs for p in Path(d).glob('ladder_layer*.csv') if not p.name.endswith('.partial.csv'))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.drop_duplicates(subset=['family', 'layer', 'name', 'seed'])
    rj = df[df.family == 'resjb'].set_index('layer')
    ours = df[df.family == 'old']
    rows = []
    for L, r in rj.iterrows():
        scope = 'pre-registered' if L in PREREG_LAYERS else 'extension (R1)'
        for task, tl in (('got_cities', 'GoT cities'), ('repe_persona', 'RepE-LM persona')):
            if task == 'got_cities' and L < 4:
                continue
            f, n = r[f'{task}__frozen_logistic_gap'], r[f'{task}__noise_frozen_logistic_gap']
            rows.append(dict(layer=L, scope=scope, hypothesis='E2-H1', task=tl, sae=f, noise=n,
                             holds=bool(n >= 0.5 * f), rule='noise frozen gap >= 0.5 x SAE frozen gap'))
            rows.append(dict(layer=L, scope=scope, hypothesis='E2-H2', task=tl, sae=r[f'{task}__refit_logistic_gap'],
                             ci_low=r[f'{task}__refit_logistic_gap_ci_low'], p=r[f'{task}__refit_logistic_gap_p'],
                             rule='re-fit gap > 0 with CI low > 0 (BH over layers x tasks)'))
            rows.append(dict(layer=L, scope=scope, hypothesis='E2-H3', task=tl, sae=r[f'{task}__refit_logistic_gap'],
                             pca=r[f'{task}__pca_refit_logistic_gap'],
                             holds=bool(r[f'{task}__refit_logistic_gap'] >= r[f'{task}__pca_refit_logistic_gap'] - 0.02),
                             rule='SAE re-fit gap >= PCA re-fit gap - 0.02 (PCA at the SAE OWT validation NMSE, as in M2)'))
        if pd.notna(r.get('m4__preservation_rec', np.nan)):
            rows.append(dict(layer=L, scope=scope, hypothesis='E2-H4', task='RepE-LM persona (M4 site)',
                             sae=r['m4__preservation_rec'], ci_high=r['m4__preservation_rec_ci_high'],
                             above_controls=bool(r['m4__T_rec_above_controls_q95']),
                             holds=bool(r['m4__preservation_rec_ci_high'] < 0.80 and r['m4__T_rec_above_controls_q95']),
                             rule='preservation CI high < 0.80 and T_rec above matched-norm and shuffle controls'))
        o = ours[ours.layer == L]
        rows.append(dict(layer=L, scope=scope, hypothesis='E2-H5 (descriptive)', task='OWT',
                         sae=r['splice_delta_ce'], resjb_nmse=r['owt_nmse'], resjb_l0=r['owt_l0'],
                         ours_delta_ce_range=f"{o.splice_delta_ce.min():.4f}-{o.splice_delta_ce.max():.4f}",
                         ours_nmse_range=f"{o.owt_nmse.min():.4f}-{o.owt_nmse.max():.4f}"))
    h = pd.DataFrame(rows)
    m = h.hypothesis == 'E2-H2'
    h.loc[m, 'q_bh'] = bh(h.loc[m, 'p'].to_numpy())
    h.loc[m, 'holds'] = (h.loc[m, 'ci_low'] > 0) & (h.loc[m, 'q_bh'] < 0.05)
    for f in ('hypotheses.csv',):
        if (OUT / f).exists():
            raise SystemExit(f'{OUT / f} exists; refusing to overwrite.')
    h.to_csv(OUT / 'hypotheses.csv', index=False)
    print(h.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--analyse', type=Path, nargs='+')
    ap.add_argument('--layers', default='1,8,10,11')
    a = ap.parse_args()
    if a.check:
        check([int(x) for x in a.layers.split(',')])
    if a.analyse:
        analyse(a.analyse)


if __name__ == '__main__':
    main()
