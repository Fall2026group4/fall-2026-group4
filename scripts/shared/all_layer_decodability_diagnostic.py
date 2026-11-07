#!/usr/bin/env python3
"""DIAGNOSTIC (not a faithfulness metric): is truth/honesty information destroyed by reconstruction, or still present
but moved off the frozen probe's direction?

For each of the 108 uninterrupted SAEs, a logistic probe (same C as the frozen original probe for that layer) is
re-fit on RECONSTRUCTED TRAIN activations and scored on RECONSTRUCTED VALIDATION activations. Test rows are never
read. Compare with the frozen-probe validation AUROC in per_run.csv:
  * frozen low, re-fit high  -> information still decodable, but not where the original probe reads it
  * both low                 -> information removed by the SAE

Run from repository root:  python scripts/shared/all_layer_decodability_diagnostic.py --dataset got|repe
"""
import argparse
import importlib.util
import sys
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

spec = importlib.util.spec_from_file_location('ev', REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('got', 'repe'), required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = ev.OUT[a.dataset] / 'decodability_diagnostic_validation.csv'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    data, _ = ev.load_got() if a.dataset == 'got' else ev.load_repe()
    orig = pd.read_csv(ev.OUT[a.dataset] / 'original_probes_by_layer.csv').set_index(['domain', 'layer'])
    rows = []
    for domain, d in data.items():
        y, split = d['y'], d['split']
        tr, va = split == 'train', split == 'validation'
        for layer in ev.LAYERS:
            C = float(orig.loc[(domain, layer), 'C'])
            X = d['X'][layer]
            for arch in ev.ARCHS:
                for seed in ev.SEEDS:
                    model, _ = fm.load_sae(ev.uninterrupted_checkpoint(layer, arch, seed), a.device)
                    X_hat = fm.reconstruct(model, X, a.device)[0].numpy()
                    p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=ev.PROBE_SEED))
                    p.fit(X_hat[tr], y[tr])
                    rows.append({'domain': domain, 'layer': layer, 'architecture': arch, 'seed': seed, 'C': C,
                                 'original_val_auroc': orig.loc[(domain, layer), 'validation_logistic_auroc'],
                                 'refit_on_reconstruction_val_auroc': fm.safe_auroc(y[va], p.decision_function(X_hat[va]))})
            print(domain, layer, flush=True)
    df = pd.DataFrame(rows)
    pr = pd.read_csv(ev.OUT[a.dataset] / 'per_run.csv')
    pr = pr[pr.eval_set == 'validation'][['domain', 'layer', 'architecture', 'seed', 'logistic_auroc_reconstructed',
                                         'sparsity_flag', 'matched_10pct', 'informative_layer']]
    df = df.merge(pr.rename(columns={'logistic_auroc_reconstructed': 'frozen_probe_val_auroc'}),
                  on=['domain', 'layer', 'architecture', 'seed'])
    df['note'] = 'diagnostic: probe re-fit on reconstructed train; measures decodability, not faithfulness'
    df.to_csv(out, index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
