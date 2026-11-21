#!/usr/bin/env python3
"""TSP pilot evaluation (2026-11-14). VALIDATION only; no test row is read.

Stage 1, INDEPENDENT GoT readers (primary): M2's unchanged original_side / evaluate_source with the GoT-train
group-A rows relabelled 'teacher_A', so every probe (frozen logistic and mass-mean fitted on ORIGINAL group-B
activations; re-fit logistic / MLP / kNN / sparse-z on x_hat of group B) is fitted on rows the subspace never saw.
The grouped bootstrap resamples (validation cities) are shared by every source, so arm differences are paired.

Stage 2, everything else: the unchanged ladder (scripts/shared/sae_ladder_evaluation.py) run on the same sources:
OWT reconstruction, OWT splicing (dCE, KL, noise ratio), RepE-LM persona / neutral M2 readers, RLM-RP, M4 causal
share at the L8 final-token site. Its GoT columns use probes fitted on ALL GoT-train rows (A included) and are
therefore NOT independent of the subspace; they are kept but never used for a criterion.

Sources (layer 8): R2 TopK base (3 seeds), R4 MSE control (3 seeds; same tokens, no GoT rows), TSP arms
ctrl / tsp / rand / pca (3 seeds each). Outputs: .../truth_subspace_preservation/pilot/eval/ (refuses to overwrite).
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

ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
LAYER, SEEDS, ARMS = 8, (42, 1042, 2042), ('ctrl', 'tsp', 'rand', 'pca')
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
R4 = REPO / 'results/openwebtext/r4_kl_finetune/checkpoints'
N_BOOT, SEED = 1000, 20261114


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sources(pilot, seeds=SEEDS):
    out = [dict(family='r2_base', name='topk', seed=s, path=R2 / f'layer_{LAYER}/topk_seed{s}/checkpoint_final.pt',
                arch=None) for s in seeds]
    out += [dict(family='r4_mse', name='topk', seed=s, path=R4 / f'layer_{LAYER}/mse_topk_seed{s}/checkpoint.pt',
                 arch=None) for s in seeds]
    out += [dict(family=f'tsp_{a}', name='topk', seed=s, path=pilot / f'checkpoints/layer_{LAYER}/{a}_topk_seed{s}/checkpoint.pt',
                 arch=None) for a in ARMS for s in seeds]
    missing = [str(s['path']) for s in out if not s['path'].exists()]
    if missing:
        raise SystemExit(f'missing checkpoints: {missing[:3]}')
    return out


def independent_got(srcs, device):
    D = _mod('scripts/geometry_of_truth/24_truth_subspace_diagnostics.py', 'tspd')
    M2 = _mod('scripts/shared/m2_four_level_faithfulness.py', 'm2')
    t = D.load_got([LAYER])
    split = t['split'].copy()
    split[(split == 'train') & (t['half'] == 'A')] = 'teacher_A'         # excluded from every probe fit
    assert set(split) == {'train', 'validation', 'teacher_A'}
    X, y, g = t['X'][LAYER], t['y'], t['groups']
    task = dict(X={LAYER: X}, y=y, split=split, groups=g)
    orig = M2.original_side(task, LAYER, np.random.default_rng(SEED))
    bidx = il.bootstrap_indices(g[split == 'validation'], N_BOOT, np.random.default_rng(SEED + 1))
    rows, boots = [], {}
    for i, s in enumerate(srcs):
        sae, _ = fm.load_sae(s['path'], device, architecture=s['arch'])
        H, Z = D.reconstruct(sae.eval(), X, device)
        r, b = M2.evaluate_source(dict(family=s['family'], seed=s['seed']), X, H, Z, y, split, g, orig, bidx, SEED + i)
        rows.append(r)
        for k, v in b.items():
            boots[f"{s['family']}|{s['seed']}|{k}"] = v
        print(f"[indep] {s['family']} {s['seed']}: refit-B logistic {r['refit_logistic_gap']:.4f} mlp "
              f"{r['refit_mlp_gap']:.4f} frozen {r['frozen_logistic_gap']:.4f} l0 {r.get('task_l0', float('nan')):.1f}",
              flush=True)
        del sae
    return pd.DataFrame(rows), boots, dict(orig_auroc=orig['auroc'], n_val=int((split == 'validation').sum()),
                                           n_B_train=int((split == 'train').sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=Path, default=ROOT / 'pilot')
    ap.add_argument('--skip-ladder', action='store_true')
    ap.add_argument('--seeds', default='42,1042,2042')
    ap.add_argument('--light', action='store_true', help='smoke: ladder reconstruction and splicing only')
    a = ap.parse_args()
    out = a.pilot / 'eval'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    out.mkdir(parents=True)
    device, t0 = 'cuda', time.time()
    srcs = sources(a.pilot, tuple(int(x) for x in a.seeds.split(',')))
    df, boots, info = independent_got(srcs, device)
    df.to_csv(out / 'got_independent.csv', index=False)
    np.savez_compressed(out / 'got_independent_bootstrap.npz', **boots)
    info['stage1_runtime_s'] = time.time() - t0
    if not a.skip_ladder:
        LD = _mod('scripts/shared/sae_ladder_evaluation.py', 'ladder')
        LD.sources = lambda L, which: srcs                                   # same harness, new source list
        sys.argv = ['ladder', '--layer', str(LAYER), '--sources', 'custom', '--out', str(out / 'ladder_layer8.csv')] + (['--light'] if a.light else [])
        LD.main()
    info['runtime_s'] = time.time() - t0
    info['test_rows_used'] = 0
    (out / 'eval_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
