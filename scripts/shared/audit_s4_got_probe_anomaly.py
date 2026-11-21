#!/usr/bin/env python3
"""AUDIT-S4 (diagnostic, NOT confirmatory): why do non-linear probes read GoT "truth" at layers 1-3 when the linear
probe is at chance (M2: L1 kNN 0.72 vs logistic 0.52; L2 MLP 0.83 vs logistic 0.59)?

Original activations only (no SAE); TRAIN fit, VALIDATION score; no TEST row is read. Uses the M2 probe functions
(src/evaluation/information_levels.py) unchanged.

Diagnostics per layer 0-11 (pre-registered in results/shared/audit_sensitivity/preregistration.json):
  D1 pooled cities+neg_cities (reproduces M2's original AUROCs);
  D2 cities only and neg_cities only (probes fit and scored within one polarity);
  D3 pooled, train labels permuted across training rows (null; 5 permutations): must be ~0.5;
  D4 cross-polarity transfer: fit on cities, score neg_cities with labels flipped (a probe for "truth" transfers; a
     probe for the surface pairing does not);
  D5 text-only baselines with no model: bag-of-words (unigram counts) logistic and kNN on the statements, pooled and
     per polarity;
  D6 "negation x country-prior" baseline: per-polarity country label prior estimated on TRAIN.
Interpretation rules (fixed before running):
  * pooled non-linear >= 0.65 while BOTH per-polarity non-linear <= 0.60  -> POOLING ARTEFACT;
  * D5 text baseline >= 0.65 on validation -> LEXICAL SHORTCUT available without the model;
  * D3 null outside [0.45, 0.55] -> PIPELINE LEAK (would invalidate the probes).
  POST HOC (added after seeing the layer-1 smoke output, disclosed in the pre-registration):
  * per-polarity logistic >= 0.65 while pooled logistic <= 0.60 -> POOLED-LINEAR SUPPRESSION (the pooled label is
    truth XOR negation, which no linear probe can read).
Outputs: results/shared/audit_sensitivity/s4_got_probe_anomaly/ (refuses to overwrite).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

OUT = REPO / 'results/shared/audit_sensitivity/s4_got_probe_anomaly'
spec = importlib.util.spec_from_file_location('got03', REPO / 'scripts/geometry_of_truth/03_original_truth_probes.py')
G3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G3)
G3.ACT_DIR, G3.SPLIT_DIR = REPO / G3.ACT_DIR, REPO / G3.SPLIT_DIR
SEED = 20261115


def probes(Xtr, ytr, gtr, Xva):
    pipe, _ = il.fit_logistic(Xtr, ytr, gtr)
    return dict(logistic=pipe.decision_function(Xva), mlp=il.mlp_scores(Xtr, ytr, Xva), knn=il.knn_scores(Xtr, ytr, Xva))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--layers', default='0-11')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.mkdir(parents=True)
    t0 = time.time()
    lo, hi = map(int, a.layers.split('-'))
    parts = G3.load_domain(G3.DOMAINS['geographic'])
    names = [p['name'] for p in parts]
    assert names == ['cities', 'neg_cities'], names
    pol = np.concatenate([np.full(len(p['labels']), i) for i, p in enumerate(parts)])
    y = np.concatenate([p['labels'] for p in parts]).astype(int)
    split = np.concatenate([p['splits'] for p in parts])
    grp = np.concatenate([p['groups'] for p in parts])
    text = pd.concat([pd.read_csv(REPO / f'data/got/{n}.csv') for n in names], ignore_index=True)
    assert len(text) == len(y) and (text.label.to_numpy() == y).all()
    keep = np.isin(split, ['train', 'val'])
    tr, va = (split == 'train'), (split == 'val')
    rows = []
    rng = np.random.default_rng(SEED)
    # ---------- D5/D6 text-only baselines (layer-independent)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(token_pattern=r"[A-Za-z']+").fit(text.statement[tr])
    W = cv.transform(text.statement).toarray().astype(np.float32)
    for scope, m in (('pooled', np.ones(len(y), bool)), ('cities', pol == 0), ('neg_cities', pol == 1)):
        s = probes(W[tr & m], y[tr & m], grp[tr & m], W[va & m])
        for fam, sc in s.items():
            rows.append(dict(layer=-1, diagnostic='D5_bag_of_words', scope=scope, family=fam, auroc=il.auroc(y[va & m], sc)))
    for scope, k in (('cities', 0), ('neg_cities', 1)):
        m = pol == k
        prior = pd.Series(y[tr & m]).groupby(text.country[tr & m].to_numpy()).mean()
        sc = text.country[va & m].map(prior).fillna(y[tr & m].mean()).to_numpy()
        rows.append(dict(layer=-1, diagnostic='D6_country_prior', scope=scope, family='lookup', auroc=il.auroc(y[va & m], sc)))
    # ---------- per layer
    for L in range(lo, hi + 1):
        X = np.concatenate([p['layers'][L].numpy() for p in parts]).astype(np.float32)
        s = probes(X[tr], y[tr], grp[tr], X[va])
        for fam, sc in s.items():
            rows.append(dict(layer=L, diagnostic='D1_pooled', scope='pooled', family=fam, auroc=il.auroc(y[va], sc)))
        for scope, k in (('cities', 0), ('neg_cities', 1)):
            m = pol == k
            s = probes(X[tr & m], y[tr & m], grp[tr & m], X[va & m])
            for fam, sc in s.items():
                rows.append(dict(layer=L, diagnostic='D2_per_polarity', scope=scope, family=fam, auroc=il.auroc(y[va & m], sc)))
        # D3 permutation null (labels shuffled across training rows)
        for rep in range(5):
            yp = y.copy()
            yp[tr] = rng.permutation(y[tr])
            s = probes(X[tr], yp[tr], grp[tr], X[va])
            for fam, sc in s.items():
                rows.append(dict(layer=L, diagnostic='D3_permuted_null', scope=f'perm{rep}', family=fam, auroc=il.auroc(y[va], sc)))
        # D4 cross-polarity transfer: fit on cities, score neg_cities (labels flipped back to "affirmative clause true")
        m0, m1 = pol == 0, pol == 1
        s = probes(X[tr & m0], y[tr & m0], grp[tr & m0], X[va & m1])
        for fam, sc in s.items():
            rows.append(dict(layer=L, diagnostic='D4_transfer_cities_to_neg', scope='neg_cities_flipped', family=fam,
                             auroc=il.auroc(1 - y[va & m1], sc)))
        print(f'L{L} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / 'diagnostics.csv', index=False)
    # ---------- verdicts per layer
    v = []
    for L in range(lo, hi + 1):
        x = df[df.layer == L]
        pooled = x[x.diagnostic == 'D1_pooled'].set_index('family').auroc
        per = x[x.diagnostic == 'D2_per_polarity'].groupby('family').auroc.max()
        null = x[x.diagnostic == 'D3_permuted_null'].auroc
        nl = max(pooled['mlp'], pooled['knn'])
        v.append(dict(layer=L, pooled_logistic=pooled['logistic'], pooled_nonlinear_max=nl,
                      per_polarity_nonlinear_max=max(per['mlp'], per['knn']), per_polarity_logistic_max=per['logistic'],
                      null_min=null.min(), null_max=null.max(),
                      pooling_artefact=bool(nl >= 0.65 and max(per['mlp'], per['knn']) <= 0.60),
                      pipeline_leak=bool(null.min() < 0.45 or null.max() > 0.55),
                      # POST HOC (added after the layer-1 smoke run; see the pre-registration disclosure):
                      pooled_linear_suppression_posthoc=bool(per['logistic'] >= 0.65 and pooled['logistic'] <= 0.60)))
    vd = pd.DataFrame(v)
    vd.to_csv(a.out / 'verdicts_by_layer.csv', index=False)
    bow = df[df.diagnostic == 'D5_bag_of_words']
    info = dict(test_rows_used=0, bag_of_words_max_auroc=float(bow.auroc.max()),
                lexical_shortcut=bool(bow.auroc.max() >= 0.65), pooling_artefact_layers=vd[vd.pooling_artefact].layer.tolist(),
                pipeline_leak_layers=vd[vd.pipeline_leak].layer.tolist(),
                pooled_linear_suppression_layers_posthoc=vd[vd.pooled_linear_suppression_posthoc].layer.tolist(),
                runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
