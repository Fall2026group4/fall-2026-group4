#!/usr/bin/env python3
"""XB amendment A4, pre-registered comparison: does the GPT-2 - Gemma difference change between the original
1024-token shared span and the 128-token shared span? (2026-11-21; written and hashed before any A4 output exists.)

Reuses the frozen XB analysis functions unchanged (03_matched_splicing_analysis.py: load_runs, per_doc_table,
stats_from_sums, boot, holm, TRIP, XMETRICS, NB, SEED). Both trees are bootstrapped with the IDENTICAL document
resample matrix (np.random.default_rng(SEED).integers(0, n_docs, (NB, n_docs)), as 03 does for every run); the
documents and their rank order are the same in both trees (A4 manifests are re-cuts of the frozen manifests), so the
context change is paired by document.

Per (corpus, triplet, slot, GPT-2 family, Gemma family, metric in XMETRICS), with SAE tracks averaged over seeds per
resample as in 03:
  d_ctx    = Gemma - GPT-2 at that context (1024: xb_matched_splicing/runs; 128: xb_matched_splicing/a4_ctx128/runs)
  change   = d_128 - d_1024                      (raw)
  shrink   = |d_1024| - |d_128|                  (> 0: the cross-model difference moves toward 0 at 128 tokens)
  label    = SHRINKS if shrink CI low > 0; GROWS if CI high < 0; else NO CLEAR CHANGE; two-sided bootstrap p,
             Holm over all rows of this table (the A4 comparison family).
Only complete runs (DONE.json) of both models in both trees enter. Output (refuses to overwrite):
results/cross_model/xb_matched_splicing/a4_ctx128/context_comparison_<tag>/{context_change.csv, info.json}

Run: python scripts/cross_model/07_a4_context_comparison.py --tag a4
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
XB_DIR = REPO / 'results/cross_model/xb_matched_splicing'
TREES = {1024: XB_DIR, 128: XB_DIR / 'a4_ctx128'}


def load03():
    spec = importlib.util.spec_from_file_location('xb03', REPO / 'scripts/cross_model/03_matched_splicing_analysis.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tree_boots(A, root):
    """{(model, corpus, layer, family): {metric: [NB] array averaged over that family's SAEs}} for complete shared runs."""
    A.XB = root
    out, meta = {}, {}
    for name, df, sx in A.load_runs('runs'):
        if not bool(df.complete.iloc[0]) or df.context.iloc[0] != 'shared':
            continue
        model, corpus = df.model.iloc[0], df.corpus.iloc[0]
        w = A.per_doc_table(df)
        ranks = np.sort(w['rank'].unique())
        idxs = np.random.default_rng(A.SEED).integers(0, len(ranks), size=(A.NB, len(ranks)))
        acc = {}
        for (L, sid, seed), g in w.groupby(['layer', 'sae_id', 'seed']):
            g = g.set_index('rank').loc[ranks].reset_index()
            sumx = np.stack([sx[f'{r}_{L}'] for r in ranks]) if sx is not None else None
            keys = [k for k in A.XMETRICS if k != 'ev' or sumx is not None]
            b = A.boot(g, sumx, idxs, keys)
            acc.setdefault((model, corpus, int(L), g.family.iloc[0]), []).append(b)
        for k, bl in acc.items():
            out[k] = {m: np.mean([b[m] for b in bl], 0) for m in bl[0]}
            meta[k] = dict(n_docs=len(ranks), n_sae=len(bl), ranks_sha=hashlib.sha256(ranks.tobytes()).hexdigest())
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    out = TREES[128] / f'context_comparison_{a.tag}'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    A = load03()
    B = {ctx: tree_boots(A, root) for ctx, root in TREES.items()}
    rows = []
    for corpus in ('pile', 'owt_xm'):
        for tname, t in A.TRIP.items():
            for slot in range(3):
                Lg2, Lgm = t['gpt2'][slot], t['gemma'][slot]
                fams2 = {k[3] for k in B[128][0] if k[:3] == ('gpt2', corpus, Lg2)}
                famsm = {k[3] for k in B[128][0] if k[:3] == ('gemma', corpus, Lgm)}
                for fa in sorted(fams2):
                    for fb in sorted(famsm):
                        keys = {c: (('gpt2', corpus, Lg2, fa), ('gemma', corpus, Lgm, fb)) for c in TREES}
                        if not all(k in B[c][0] for c in TREES for k in keys[c]):
                            continue
                        for c in TREES:        # pairing check: same documents in the same order in both trees
                            for k in keys[c]:
                                assert B[c][1][k]['ranks_sha'] == B[128][1][keys[128][0]]['ranks_sha'], k
                        for m in A.XMETRICS:
                            if not all(m in B[c][0][k] for c in TREES for k in keys[c]):
                                continue
                            d = {c: B[c][0][keys[c][1]][m] - B[c][0][keys[c][0]][m] for c in TREES}
                            change, shrink = d[128] - d[1024], np.abs(d[1024]) - np.abs(d[128])
                            lo, hi = np.quantile(shrink, [0.025, 0.975])
                            p = min(1.0, 2 * min((shrink <= 0).mean(), (shrink >= 0).mean()) + 1 / A.NB)
                            rows.append(dict(corpus=corpus, triplet=tname, slot=slot, gpt2_layer=Lg2, gemma_layer=Lgm,
                                             gpt2_family=fa, gemma_family=fb, metric=m,
                                             diff_1024=float(d[1024].mean()), diff_128=float(d[128].mean()),
                                             change=float(change.mean()), change_lo=float(np.quantile(change, 0.025)),
                                             change_hi=float(np.quantile(change, 0.975)), shrink=float(shrink.mean()),
                                             shrink_lo=float(lo), shrink_hi=float(hi), p_boot=p,
                                             label='SHRINKS' if lo > 0 else 'GROWS' if hi < 0 else 'NO CLEAR CHANGE'))
    R = pd.DataFrame(rows)
    if len(R):
        R['p_holm'] = A.holm(R.p_boot.to_numpy())
        R['holm_label'] = np.where(R.p_holm < 0.05, R.label, 'NO CLEAR CHANGE (Holm)')
    out.mkdir(parents=True)
    R.to_csv(out / 'context_change.csv', index=False)
    info = dict(n_rows=len(R), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                counts=R.holm_label.value_counts().to_dict() if len(R) else {})
    (out / 'info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
