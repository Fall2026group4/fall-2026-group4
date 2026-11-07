#!/usr/bin/env python3
"""GoT layer 10 (professor-review Task E): reproduce the legacy GoT layer-10 result and add a frozen-probe comparison.

Legacy pipeline (scripts/prepare_geometry_data.py -> cache_geometry_activations.py ->
run_geometry_probes.py -> run_geometry_sae.py -> run_geometry_faithfulness.py):
  cities.csv, city-grouped 70/30 split (seed 42), last-token hook_resid_pre (BOS prepended),
  SAELens gpt2-small-res-jb blocks.10.hook_resid_pre, StandardScaler + LogisticRegression(max_iter=2000, seed 42).
  The reported "reconstructed" probe was RE-TRAINED on reconstructions.

This script re-runs that pipeline from data/got/cities.csv (the legacy data/raw and data/processed
files are absent) using the legacy helper functions, then reports three probes on the same test rows:
  original      : probe trained on original train activations, scored on original test activations
  retrained     : (legacy) probe trained on reconstructed train, scored on reconstructed test
  frozen        : the ORIGINAL probe (unchanged weights, scaler) scored on reconstructed test
Plus paired bootstrap CIs for AUROC differences and a robustness check over 5 extra city-split seeds.

Run from repository root:  python scripts/geometry_of_truth/10_layer10_frozen_probe.py
(Formerly scripts/professor_review/got_layer10_frozen_probe.py.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from src.models.gpt2_activations import load_gpt2_small, extract_last_token_residuals  # noqa: E402
from src.sae.geometry_sae import load_geometry_sae, SAE_RELEASE, SAE_ID  # noqa: E402

OUT = Path('results/geometry_of_truth/layer10_frozen_probe')
LAYER = 10
SEED = 42
EXTRA_SPLIT_SEEDS = (1, 2, 3, 4, 5)


def legacy_probe(seed=SEED):
    return Pipeline([('scaler', StandardScaler()),
                     ('classifier', LogisticRegression(max_iter=2000, random_state=seed))])


def city_split(df, seed):
    cities = df['city'].drop_duplicates().tolist()
    tr, te = train_test_split(cities, test_size=0.30, random_state=seed, shuffle=True)
    is_train = df['city'].isin(tr).values
    assert set(df.loc[is_train, 'city']).isdisjoint(set(df.loc[~is_train, 'city']))
    return is_train


def scores(probe, X, y):
    p = probe.predict_proba(X)[:, 1]
    return dict(accuracy=accuracy_score(y, (p > 0.5).astype(int)), auroc=roc_auc_score(y, p)), p


def paired_bootstrap(y, p_a, p_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('data/got/cities.csv')
    y = df['label'].values.astype(int)

    model = load_gpt2_small(device=device)
    acts = extract_last_token_residuals(model, df['statement'].tolist(), batch_size=32, device=device)
    X = acts[:, LAYER, :].float()
    del model

    sae = load_geometry_sae(device=device)
    sae.eval()
    with torch.inference_mode():
        feats = sae.encode(X.to(device))
        Xr = sae.decode(feats).float().cpu()
    l0 = (feats > 0).sum(-1).float().cpu().numpy()
    Xn, Xrn = X.numpy(), Xr.numpy()

    rows, recon_rows = [], []
    for split_seed in (SEED,) + EXTRA_SPLIT_SEEDS:
        tr = city_split(df, split_seed)
        te = ~tr
        orig = legacy_probe().fit(Xn[tr], y[tr])
        retr = legacy_probe().fit(Xrn[tr], y[tr])
        m_o, p_o = scores(orig, Xn[te], y[te])
        m_r, p_r = scores(retr, Xrn[te], y[te])
        m_f, p_f = scores(orig, Xrn[te], y[te])
        ci_f = paired_bootstrap(y[te], p_o, p_f)
        ci_r = paired_bootstrap(y[te], p_o, p_r)
        base = dict(layer=LAYER, sae=f'{SAE_RELEASE}/{SAE_ID}', split_seed=split_seed,
                    n_train=int(tr.sum()), n_test=int(te.sum()))
        rows += [
            dict(base, probe='original_probe_on_original', **m_o, auroc_gap_vs_original=0.0),
            dict(base, probe='retrained_probe_on_reconstruction(legacy)', **m_r,
                 auroc_gap_vs_original=m_o['auroc'] - m_r['auroc'],
                 gap_ci95_low=ci_r[0], gap_ci95_high=ci_r[1]),
            dict(base, probe='frozen_original_probe_on_reconstruction', **m_f,
                 auroc_gap_vs_original=m_o['auroc'] - m_f['auroc'],
                 gap_ci95_low=ci_f[0], gap_ci95_high=ci_f[1]),
        ]
        err = ((Xn[te] - Xrn[te]) ** 2)
        recon_rows.append(dict(base, mse=float(err.mean()),
                               cosine=float(torch.nn.functional.cosine_similarity(X[te], Xr[te]).mean()),
                               explained_variance=float(1 - err.sum() / ((Xn[te] - Xn[te].mean(0)) ** 2).sum()),
                               mean_l0=float(l0[te].mean())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'got_layer10_probe_comparison.csv', index=False)
    pd.DataFrame(recon_rows).to_csv(OUT / 'got_layer10_reconstruction.csv', index=False)
    legacy = pd.read_csv('results/geometry_of_truth/sae_faithfulness_results.csv').iloc[0]
    main_rows = res[res['split_seed'] == SEED].set_index('probe')
    check = dict(
        legacy_raw_accuracy=float(legacy['raw_accuracy']),
        reproduced_raw_accuracy=float(main_rows.loc['original_probe_on_original', 'accuracy']),
        legacy_raw_auroc=float(legacy['raw_auroc']),
        reproduced_raw_auroc=float(main_rows.loc['original_probe_on_original', 'auroc']),
        legacy_reconstructed_accuracy=float(legacy['reconstructed_accuracy']),
        reproduced_retrained_accuracy=float(main_rows.loc['retrained_probe_on_reconstruction(legacy)', 'accuracy']),
        legacy_reconstructed_auroc=float(legacy['reconstructed_auroc']),
        reproduced_retrained_auroc=float(main_rows.loc['retrained_probe_on_reconstruction(legacy)', 'auroc']))
    (OUT / 'legacy_reproduction_check.json').write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    print(res.round(4).to_string(index=False))
    print(pd.DataFrame(recon_rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
