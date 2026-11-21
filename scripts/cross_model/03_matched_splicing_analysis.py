#!/usr/bin/env python3
"""XB step 3: analysis of the matched splicing runs (pre-registered rules; 2026-11-21).

Inputs: results/cross_model/xb_matched_splicing/runs/<run>/per_doc.csv (+ sum_x.npz).
Outputs (refuse to overwrite unless --tag differs): results/cross_model/xb_matched_splicing/analysis_<tag>/
  * summary_by_sae.csv   per (model, corpus, context, layer, SAE): token-weighted metrics + 95 % doc-bootstrap CIs
  * h1_by_sae.csv        XB_H1 (dCE_sae > dCE_random), paired doc bootstrap, Holm within (model, corpus, context)
  * cross_model.csv      XB_H2 paired (same documents) gemma - gpt2 differences of normalised metrics per triplet slot
                         and SAE track, with the matched-stratum flag
  * depth_trend.csv      XB_H3 Spearman of dCE_rel / ratio vs depth per model (canonical / public track)
  * gate_v_m6.json       V_M6 (if the Gemma own-1024 Pile run exists), V_IDENT, V_REPEAT
Run: python scripts/cross_model/03_matched_splicing_analysis.py --tag v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
XB = REPO / 'results/cross_model/xb_matched_splicing'
D_MODEL = {'gpt2': 768, 'gemma': 2304}
TRIP = {'T_B': {'gpt2': [3, 6, 9], 'gemma': [5, 12, 19]}, 'T_A': {'gpt2': [1, 8, 11], 'gemma': [1, 16, 23]}}
NB, SEED = 1000, 0


def load_runs(which='runs'):
    out = []
    for d in sorted((XB / which).glob('*/')):
        f = d / 'per_doc.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        model, corpus, context = d.name.split('_')[0], '_'.join(d.name.split('_')[1:-2]), d.name.split('_')[-2]
        df['model'], df['corpus'], df['context'], df['run'] = model, corpus, context, d.name
        df['complete'] = (d / 'DONE.json').exists()
        sx = np.load(d / 'sum_x.npz') if (d / 'sum_x.npz').exists() else None
        out.append((d.name, df, sx))
    return out


def per_doc_table(df):
    """Wide per-(layer, sae, seed, doc) table with SAE, random and mean-ablation sums side by side."""
    sae = df[df.cond == 'sae'].copy()
    rnd = df[df.cond == 'random'][['rank', 'layer', 'sae_id', 'seed', 'ce_new_sum', 'kl_sum', 'top1_agree']]
    rnd = rnd.rename(columns={'ce_new_sum': 'ce_rand_sum', 'kl_sum': 'kl_rand_sum', 'top1_agree': 'top1_rand'})
    mab = df[df.cond == 'mean_ablation'][['rank', 'layer', 'ce_new_sum', 'kl_sum']]
    mab = mab.rename(columns={'ce_new_sum': 'ce_mean_sum', 'kl_sum': 'kl_mean_sum'})
    w = sae.merge(rnd, on=['rank', 'layer', 'sae_id', 'seed'], validate='1:1').merge(mab, on=['rank', 'layer'],
                                                                                   validate='m:1')
    return w


def stats_from_sums(g, idx=None, sumx=None, rank_pos=None):
    """Token-weighted metrics for one SAE's per-doc rows g (sorted by rank); idx = resample of row positions."""
    if idx is None:
        idx = np.arange(len(g))
    s = {c: g[c].to_numpy()[idx].sum() for c in ('n_pred', 'ce_orig_sum', 'ce_new_sum', 'ce_rand_sum', 'ce_mean_sum',
                                                 'kl_sum', 'kl_mean_sum', 'top1_agree', 'n_tok', 'sse', 'sum_sq',
                                                 'cos_sum', 'l0_sum')}
    n = s['n_pred']
    ce0 = s['ce_orig_sum'] / n
    d_sae = (s['ce_new_sum'] - s['ce_orig_sum']) / n
    d_rnd = (s['ce_rand_sum'] - s['ce_orig_sum']) / n
    d_mean = (s['ce_mean_sum'] - s['ce_orig_sum']) / n
    out = dict(ce_orig=ce0, dce_sae=d_sae, dce_random=d_rnd, dce_mean=d_mean, kl_sae=s['kl_sum'] / n,
               kl_mean=s['kl_mean_sum'] / n, top1=s['top1_agree'] / n,
               loss_recovered=(d_mean - d_sae) / d_mean, ratio_sae_random=d_sae / d_rnd,
               excess_rel=(d_sae - d_rnd) / ce0, dce_rel=d_sae / ce0, kl_over_kl_mean=s['kl_sum'] / s['kl_mean_sum'],
               cos=s['cos_sum'] / s['n_tok'], rel_sq_error=s['sse'] / s['sum_sq'], l0=s['l0_sum'] / s['n_tok'])
    if sumx is not None:
        sx = sumx[idx].sum(0)
        tot = s['sum_sq'] - float(sx @ sx) / s['n_tok']
        out['ev'] = 1 - s['sse'] / tot
        out['nmse'] = 1 - out['ev']
    return out


def boot(g, sumx, idxs, keys):
    vals = {k: [] for k in keys}
    for idx in idxs:
        st = stats_from_sums(g, idx, sumx)
        for k in keys:
            vals[k].append(st[k])
    return {k: np.array(v) for k, v in vals.items()}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * p[j]))
        adj[j] = run
    return adj


METRICS = ['ce_orig', 'dce_sae', 'dce_random', 'dce_mean', 'kl_sae', 'kl_mean', 'top1', 'loss_recovered',
           'ratio_sae_random', 'excess_rel', 'dce_rel', 'kl_over_kl_mean', 'cos', 'rel_sq_error', 'l0', 'ev', 'nmse']
XMETRICS = ['loss_recovered', 'ratio_sae_random', 'excess_rel', 'dce_rel', 'kl_over_kl_mean', 'ev']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--which', default='runs', choices=['runs', 'smoke'])
    a = ap.parse_args()
    out = XB / f'analysis_{a.tag}' if a.which == 'runs' else XB / f'smoke/analysis_{a.tag}'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    runs = load_runs(a.which)
    summ, h1rows, boots, gates = [], [], {}, {'V_IDENT': [], 'V_REPEAT': []}
    for name, df, sx in runs:
        model = df.model.iloc[0]
        for cond, key in (('identity', 'V_IDENT'), ('original_repeat', 'V_REPEAT')):
            c = df[df.cond == cond]
            if len(c):
                gates[key].append(dict(run=name, max_abs_dce=float(((c.ce_new_sum - c.ce_orig_sum) / c.n_pred).abs().max()),
                                       max_kl=float((c.kl_sum / c.n_pred).max())))
        w = per_doc_table(df)
        ranks = np.sort(w['rank'].unique())
        n_docs = len(ranks)
        idxs = rng.integers(0, n_docs, size=(NB, n_docs)) if n_docs else None
        rng = np.random.default_rng(SEED)   # same resample matrix for every run (paired across models)
        for (L, sid, seed), g in w.groupby(['layer', 'sae_id', 'seed']):
            g = g.set_index('rank').loc[ranks].reset_index()
            sumx = np.stack([sx[f'{r}_{L}'] for r in ranks]) if sx is not None else None
            pt = stats_from_sums(g, None, sumx)
            b = boot(g, sumx, idxs, METRICS if sumx is not None else METRICS[:-2])
            boots[(name, L, sid, seed)] = b
            d_doc = ((g.ce_new_sum - g.ce_rand_sum) / g.n_pred).to_numpy()
            dmeans = np.array([d_doc[i].mean() for i in idxs])
            p = min(1.0, 2 * min((dmeans <= 0).mean(), (dmeans >= 0).mean()) + 1 / NB)
            d_sae = int(g.d_sae.iloc[0])
            row = dict(run=name, model=model, corpus=df.corpus.iloc[0], context=df.context.iloc[0],
                       complete=bool(df.complete.iloc[0]), layer=int(L), depth=float(g.depth.iloc[0]),
                       family=g.family.iloc[0], sae_id=sid, seed=int(seed), n_docs=n_docs, d_sae=d_sae,
                       d_sae_over_d=d_sae / D_MODEL[model])
            for k, v in pt.items():
                row[k] = v
                if k in b:
                    row[f'{k}_lo'], row[f'{k}_hi'] = np.quantile(b[k], [0.025, 0.975])
            row['l0_over_d'] = row['l0'] / D_MODEL[model]
            row['active_frac'] = row['l0'] / d_sae
            summ.append(row)
            lo, hi = np.quantile(dmeans, [0.025, 0.975])
            h1rows.append(dict(run=name, model=model, corpus=row['corpus'], context=row['context'], layer=int(L),
                               family=row['family'], sae_id=sid, seed=int(seed), mean_d=float(d_doc.mean()),
                               ci_lo=lo, ci_hi=hi, p_boot=p,
                               verdict='supported' if lo > 0 else ('contradicted' if hi < 0 else 'inconclusive')))
    S = pd.DataFrame(summ)
    H = pd.DataFrame(h1rows)
    if len(H):
        H['p_holm'] = H.groupby(['model', 'corpus', 'context'])['p_boot'].transform(lambda p: holm(p.to_numpy()))
        H['holm_supported'] = (H.p_holm < 0.05) & (H.mean_d > 0)
    S.to_csv(out / 'summary_by_sae.csv', index=False)
    H.to_csv(out / 'h1_by_sae.csv', index=False)

    # XB_H2: paired cross-model differences per triplet slot; SAE tracks averaged over seeds per resample
    xrows = []
    if len(S):
        for corpus in S.corpus.unique():
            for context in ('shared',):
                for tname, t in TRIP.items():
                    for slot in range(3):
                        Lg2, Lgm = t['gpt2'][slot], t['gemma'][slot]
                        A = S[(S.model == 'gpt2') & (S.corpus == corpus) & (S.context == context) & (S.layer == Lg2)]
                        B = S[(S.model == 'gemma') & (S.corpus == corpus) & (S.context == context) & (S.layer == Lgm)]
                        for fa, ga in A.groupby('family'):
                            for fb, gb in B.groupby('family'):
                                def agg(rows, m):
                                    return np.mean([boots[(r.run, r.layer, r.sae_id, r.seed)][m] for r in rows.itertuples()], 0)
                                r = dict(corpus=corpus, context=context, triplet=tname, slot=slot, gpt2_layer=Lg2,
                                         gemma_layer=Lgm, gpt2_family=fa, gemma_family=fb, gpt2_n_sae=len(ga),
                                         gemma_sae=';'.join(gb.sae_id))
                                l0d_ratio = gb.l0_over_d.mean() / ga.l0_over_d.mean()
                                w_ratio = gb.d_sae_over_d.mean() / ga.d_sae_over_d.mean()
                                r.update(l0d_ratio=l0d_ratio, width_ratio=w_ratio,
                                         matched=bool(0.85 <= l0d_ratio <= 1.15 and 0.85 <= w_ratio <= 1.15))
                                for m in XMETRICS:
                                    if m not in boots[(ga.iloc[0].run, ga.iloc[0].layer, ga.iloc[0].sae_id, ga.iloc[0].seed)]:
                                        continue
                                    da, db = agg(ga, m), agg(gb, m)
                                    diff = db - da
                                    r[f'{m}_gpt2'], r[f'{m}_gemma'] = ga[m].mean(), gb[m].mean()
                                    r[f'{m}_diff'] = gb[m].mean() - ga[m].mean()
                                    r[f'{m}_diff_lo'], r[f'{m}_diff_hi'] = np.quantile(diff, [0.025, 0.975])
                                    excl = r[f'{m}_diff_lo'] > 0 or r[f'{m}_diff_hi'] < 0
                                    r[f'{m}_label'] = ('cross-model difference' if (excl and r['matched']) else
                                                       'unmatched, descriptive' if not r['matched'] else 'no clear difference')
                                xrows.append(r)
    pd.DataFrame(xrows).to_csv(out / 'cross_model.csv', index=False)

    # XB_H3 depth trend
    trows = []
    for (model, corpus, context, fam), g in S.groupby(['model', 'corpus', 'context', 'family']):
        g = g.groupby('layer').mean(numeric_only=True).reset_index()
        if len(g) >= 5:
            for m in ('dce_rel', 'ratio_sae_random', 'loss_recovered'):
                rho, p = spearmanr(g.depth, g[m])
                trows.append(dict(model=model, corpus=corpus, context=context, family=fam, metric=m, n_layers=len(g),
                                  spearman=rho, p=p))
    pd.DataFrame(trows).to_csv(out / 'depth_trend.csv', index=False)

    # V_M6
    m6 = pd.read_csv(XB / 'inputs/m6/summary_by_layer.csv')
    g6 = S[(S.model == 'gemma') & (S.corpus == 'pile') & (S.context == 'own') &
           (S.family == 'gemmascope_16k_canonical') & (S.layer.isin([5, 12, 19]))]
    vm6 = []
    for r in g6.itertuples():
        ref = m6[m6.layer == r.layer].iloc[0]
        vm6.append(dict(layer=r.layer, dce=r.dce_sae, dce_m6=ref.delta_ce_sae, kl=r.kl_sae, kl_m6=ref.kl_sae,
                        ev=r.ev, ev_m6=ref.explained_variance,
                        within=bool(abs(r.dce_sae - ref.delta_ce_sae) <= 0.005 and abs(r.kl_sae - ref.kl_sae) <= 0.005
                                    and abs(r.ev - ref.explained_variance) <= 0.001)))
    gates['V_M6'] = vm6
    (out / 'gates.json').write_text(json.dumps(gates, indent=1, default=float))
    print(json.dumps(gates, indent=1, default=float))
    print(f'{len(S)} SAE rows; {len(xrows)} cross-model rows -> {out}')


if __name__ == '__main__':
    main()
