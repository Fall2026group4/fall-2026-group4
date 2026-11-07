#!/usr/bin/env python3
"""All-layer truth (GoT) / honesty (RepE) preservation of the 108 uninterrupted OWT SAEs, with frozen probes.

Run from repository root:
    python scripts/shared/all_layer_frozen_probe_evaluation.py --dataset got
    python scripts/shared/all_layer_frozen_probe_evaluation.py --dataset repe

For every GPT-2 layer (hook_resid_pre, final token), every architecture and every seed:
  1. Reconstruct the cached ORIGINAL activations with the layer's SAE (no fitting of any kind).
  2. Reconstruction metrics on each evaluation set (src/evaluation/faithfulness_metrics.py).
  3. Probes are fit on ORIGINAL train activations only; their hyper-parameters are selected on ORIGINAL validation
     activations only; they are then applied unchanged (frozen) to the reconstructions:
       * logistic  : StandardScaler + LogisticRegression, C from the grid of GoT script 03 (same selection rule).
       * diffmean  : unit mass-mean direction mean(x|1) - mean(x|0) on original train (RepE script 03 probe).
     auroc_gap = auroc(original) - auroc(reconstructed) on the same rows.
  4. Direction fidelity: the mass-mean direction re-estimated on reconstructed TRAIN rows vs the original one.
  5. Paired grouped-bootstrap 95 % CI of the test gap (groups: GoT group_id, RepE statement_id).

Test rows are only scored; no choice (C, informative layers, checkpoints) depends on them. The protocol is written to
``protocol.json`` before any SAE is evaluated. Existing results are never overwritten (the script refuses).
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

LAYERS = range(12)
ARCHS = ('relu', 'topk', 'jumprelu')
SEEDS = (42, 1042, 2042)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)      # = scripts/geometry_of_truth/03_original_truth_probes.py
PROBE_SEED = 42
INFORMATIVE_VAL_AUROC = 0.70                 # fixed before this run (docs/research_gaps_and_plan.md)
N_BOOT = 1000
OUT = {'got': REPO / 'results/geometry_of_truth/all_layer_frozen_probe_evaluation',
       'repe': REPO / 'results/repe/all_layer_frozen_probe_evaluation'}
PER_RUN_FLAGS = REPO / 'results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_per_run.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_got():
    """Return {domain: dict(X={layer: array}, y, split, groups)} using GoT script 03's validated loader."""
    spec = importlib.util.spec_from_file_location('got03', REPO / 'scripts/geometry_of_truth/03_original_truth_probes.py')
    got03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(got03)
    got03.ACT_DIR = REPO / got03.ACT_DIR
    got03.SPLIT_DIR = REPO / got03.SPLIT_DIR
    out = {}
    for domain, names in got03.DOMAINS.items():
        parts = got03.load_domain(names)  # checks labels, splits, group leakage, shapes
        split = np.concatenate([p['splits'] for p in parts])
        out[domain] = {
            'X': {l: np.concatenate([p['layers'][l].numpy() for p in parts]).astype(np.float32) for l in LAYERS},
            'y': np.concatenate([p['labels'] for p in parts]),
            'split': np.where(split == 'val', 'validation', split),
            'groups': np.concatenate([p['groups'] for p in parts]),
        }
    return out, {'validation': 'validation', 'test': 'test'}


def load_repe():
    main = torch.load(REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt', map_location='cpu', weights_only=True)
    cw = torch.load(REPO / 'data/repe/activations/cross_wording/repe_final_token.pt', map_location='cpu', weights_only=True)
    ids = np.array(main['statement_ids'])
    split = np.array(main['splits'])
    if any(len(set(split[ids == i])) > 1 for i in np.unique(ids)):
        raise ValueError('RepE: a statement crosses splits')
    cw_ids = np.array([f'cw{i}' for i in cw['statement_ids']])
    d = {
        'X': {l: np.concatenate([main['activations'][l].numpy(), cw['activations'][l].numpy()]).astype(np.float32)
              for l in LAYERS},
        'y': np.concatenate([main['persona_labels'].numpy(), cw['persona_labels'].numpy()]),
        # cross-wording rows are a separate held-out wording set; never used for fitting or selection.
        'split': np.concatenate([split, np.repeat('cross_wording', len(cw_ids))]),
        'groups': np.concatenate([ids.astype(str), cw_ids]),
    }
    return {'honesty': d}, {'validation': 'validation', 'test': 'test', 'cross_wording': 'cross_wording'}


def fit_original_probes(X, y, split):
    """Probes on ORIGINAL activations: fit on train, C chosen on validation (ties -> smallest C)."""
    tr, va = split == 'train', split == 'validation'
    best = None
    for C in C_GRID:
        p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=PROBE_SEED))
        p.fit(X[tr], y[tr])
        auc = fm.safe_auroc(y[va], p.predict_proba(X[va])[:, 1])
        if best is None or auc > best[0] + 1e-15:
            best = (auc, C, p)
    raw_direction = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
    # Undefined when the class means coincide exactly (GoT numerical, layer 0: identical final-token embeddings).
    direction = None if np.linalg.norm(raw_direction) == 0 else raw_direction / np.linalg.norm(raw_direction)
    return {'C': best[1], 'logistic': best[2], 'direction': direction, 'raw_direction': raw_direction}


def scores(probes, X):
    d = probes['direction']
    return {'logistic': probes['logistic'].decision_function(X),
            'diffmean': np.full(len(X), np.nan) if d is None else X @ d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('got', 'repe'), required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = OUT[a.dataset]
    if (out / 'per_run.csv').exists():
        raise SystemExit(f'{out}/per_run.csv exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    data, eval_sets = load_got() if a.dataset == 'got' else load_repe()
    flags = pd.read_csv(PER_RUN_FLAGS).set_index(['layer', 'architecture', 'seed'])
    ckpts = {(l, ar, s): uninterrupted_checkpoint(l, ar, s) for l in LAYERS for ar in ARCHS for s in SEEDS}

    protocol = {
        'dataset': a.dataset, 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'scripts/shared/all_layer_frozen_probe_evaluation.py', 'script_sha256': sha(__file__),
        'metrics_module_sha256': sha(fm.__file__),
        'hook': 'blocks.{layer}.hook_resid_pre, final token (same hook as SAE training)',
        'saes': '108 uninterrupted OWT SAEs (12 layers x relu/topk/jumprelu x seeds 42/1042/2042), d_sae 3072, L0 128',
        'checkpoint_sha256': {f'{l}/{ar}/{s}': sha(p) for (l, ar, s), p in ckpts.items()},
        'probes': {'logistic': f'StandardScaler+LogisticRegression, C in {C_GRID} selected on original validation AUROC',
                   'diffmean': 'unit mass-mean direction on original train'},
        'fit_rows': 'train (original activations only)', 'selection_rows': 'validation (original activations only)',
        'scored_sets': list(eval_sets), 'test_used_for_selection': False,
        'informative_layer_rule': f'original validation AUROC (logistic) >= {INFORMATIVE_VAL_AUROC}',
        'gap': 'auroc(original) - auroc(reconstructed), same rows',
        'bootstrap': f'{N_BOOT} paired grouped resamples of the test set, seed 0',
        'sparsity_rule': 'run matched if |L0_owt_test - 128|/128 <= 0.10; cell ranking-eligible only if all 3 seeds matched',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    orig_rows, run_rows = [], []
    t0 = time.time()
    for domain, d in data.items():
        y, split, groups = d['y'], d['split'], d['groups']
        tr = split == 'train'
        for layer in LAYERS:
            X = d['X'][layer]
            if not np.isfinite(X).all():
                raise ValueError(f'{domain} layer {layer}: non-finite activations')
            probes = fit_original_probes(X, y, split)
            s_orig = scores(probes, X)
            o = {'domain': domain, 'layer': layer, 'C': probes['C'],
                 'train_n': int(tr.sum()), 'direction_norm': float(np.linalg.norm(probes['raw_direction']))}
            for sname in eval_sets:
                m = split == sname
                o[f'{sname}_n'] = int(m.sum())
                for pr in ('logistic', 'diffmean'):
                    o[f'{sname}_{pr}_auroc'] = fm.safe_auroc(y[m], s_orig[pr][m])
            o['informative'] = o['validation_logistic_auroc'] >= INFORMATIVE_VAL_AUROC
            orig_rows.append(o)
            for arch in ARCHS:
                for seed in SEEDS:
                    model, _ = fm.load_sae(ckpts[(layer, arch, seed)], a.device)
                    X_hat, l0, _ = fm.reconstruct(model, X, a.device)
                    X_hat = X_hat.numpy()
                    s_rec = scores(probes, X_hat)
                    if probes['direction'] is None:
                        fid = {'direction_cosine': np.nan, 'direction_norm_ratio': np.nan}
                    else:
                        fid = fm.direction_fidelity(probes['raw_direction'],
                                                    X_hat[tr & (y == 1)].mean(0) - X_hat[tr & (y == 0)].mean(0))
                    f = flags.loc[(layer, arch, seed)]
                    for sname in eval_sets:
                        m = split == sname
                        r = {'domain': domain, 'layer': layer, 'architecture': arch, 'seed': seed, 'eval_set': sname,
                             'owt_test_l0': f['actual_l0'], 'sparsity_flag': f['sparsity_flag'],
                             'matched_10pct': bool(f['matched_10pct']),
                             'checkpoint_path': ckpts[(layer, arch, seed)].relative_to(REPO).as_posix()}
                        r.update(fm.reconstruction_metrics(X[m], X_hat[m], l0[torch.as_tensor(m)]))
                        for pr in ('logistic', 'diffmean'):
                            r[f'{pr}_auroc_original'] = o[f'{sname}_{pr}_auroc']
                            r[f'{pr}_auroc_reconstructed'] = fm.safe_auroc(y[m], s_rec[pr][m])
                            r[f'{pr}_auroc_gap'] = r[f'{pr}_auroc_original'] - r[f'{pr}_auroc_reconstructed']
                            if sname == 'test' and np.isfinite(s_orig[pr]).all():
                                lo, hi = fm.grouped_bootstrap_gap(y[m], s_orig[pr][m], s_rec[pr][m], groups[m], N_BOOT)
                                r[f'{pr}_gap_ci95_low'], r[f'{pr}_gap_ci95_high'] = lo, hi
                        r.update(fid)
                        r['informative_layer'] = o['informative']
                        run_rows.append(r)
            print(f'{domain} layer {layer:2d} C={probes["C"]:g} '
                  f'val AUROC={o["validation_logistic_auroc"]:.3f} ({time.time() - t0:.0f}s)', flush=True)

    pd.DataFrame(orig_rows).to_csv(out / 'original_probes_by_layer.csv', index=False)
    pd.DataFrame(run_rows).to_csv(out / 'per_run.csv', index=False)
    protocol['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    protocol['runtime_seconds'] = round(time.time() - t0, 1)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
