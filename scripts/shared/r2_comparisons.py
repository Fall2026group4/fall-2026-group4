#!/usr/bin/env python3
"""R2 pre-registered comparisons (results/openwebtext/r2_scaled_saes/README.md section 1.4).

Reads the ladder CSVs (scripts/shared/sae_ladder_evaluation.py) for layers 1, 8, 11 and applies the rule:
IMPROVEMENT if every new seed beats every reference seed (all pairs) AND the Welch 95 % interval of the mean
difference excludes 0; REGRESSION for the mirror image; otherwise NO CLEAR CHANGE. SAEs whose OWT validation L0 is
outside 128 +- 10 % are excluded from comparisons (reported). VALIDATION only.

Run: python scripts/shared/r2_comparisons.py --ladder-dir DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

COMPARISONS = [  # id, (family, name) new, (family, name) reference, description
    ('C1', ('r2', 'topk'), ('old', 'topk'), 'new TopK-12288 vs old TopK-3072'),
    ('C2', ('r2', 'relu'), ('old', 'relu'), 'new ReLU-12288 vs old ReLU-3072'),
    ('C3', ('r2', 'batchtopk'), ('old', 'jumprelu'), 'new BatchTopK-12288 vs old JumpReLU-3072'),
    ('A1', ('r2', 'topk_w3072'), ('old', 'topk'), 'data+recipe at equal width: new TopK-3072 vs old TopK-3072'),
    ('A2', ('r2', 'topk'), ('r2', 'topk_w3072'), 'width: new TopK-12288 vs new TopK-3072'),
]
# metric, higher_is_better, level
METRICS = [('owt_nmse', False, 'reconstruction'), ('owt_explained_variance', True, 'reconstruction'),
           ('owt_cosine', True, 'reconstruction'), ('owt_dead_fraction', False, 'reconstruction'),
           ('splice_delta_ce', False, 'behaviour'), ('splice_kl_mean', False, 'behaviour'),
           ('splice_sae_vs_noise_delta_ce_ratio', False, 'behaviour'),
           ('got_cities__frozen_logistic_gap', False, 'information'), ('got_cities__refit_logistic_gap', False, 'information'),
           ('got_cities__sae_minus_pca_refit_gap', False, 'information'),
           ('repe_persona__frozen_logistic_gap', False, 'information'),
           ('repe_persona__refit_logistic_gap', False, 'information'),
           ('repe_persona__sae_minus_pca_refit_gap', False, 'information'),
           ('repe_neutral__refit_logistic_gap', False, 'information'),
           ('rlm__pair_dir_cosine', True, 'representation'), ('rlm__frozen_heldout_gap', False, 'representation'),
           ('m4__preservation_rec', True, 'causal'), ('m4__preservation_err', False, 'causal')]
GOT_INFORMATIVE = (4, 5, 6, 7, 8, 9, 10, 11)
T_LOW, T_HIGH = 0.9 * 128, 1.1 * 128


def rule(new, ref, higher):
    new, ref = np.asarray(new, float), np.asarray(ref, float)
    if len(new) < 2 or len(ref) < 2 or np.isnan(new).any() or np.isnan(ref).any():
        return dict(cls='NOT EVALUATED')
    d = new.mean() - ref.mean()
    sn, sr = new.var(ddof=1) / len(new), ref.var(ddof=1) / len(ref)
    se = np.sqrt(sn + sr)
    dof = (sn + sr) ** 2 / ((sn ** 2) / (len(new) - 1) + (sr ** 2) / (len(ref) - 1)) if se > 0 else np.inf
    h = stats.t.ppf(0.975, dof) * se if se > 0 else 0.0
    lo, hi = d - h, d + h
    better = (new[:, None] > ref[None, :]) if higher else (new[:, None] < ref[None, :])
    worse = (new[:, None] < ref[None, :]) if higher else (new[:, None] > ref[None, :])
    if better.all() and ((lo > 0) if higher else (hi < 0)):
        cls = 'IMPROVEMENT'
    elif worse.all() and ((hi < 0) if higher else (lo > 0)):
        cls = 'REGRESSION'
    else:
        cls = 'NO CLEAR CHANGE'
    return dict(cls=cls, ref_mean=ref.mean(), ref_sd=ref.std(ddof=1), new_mean=new.mean(), new_sd=new.std(ddof=1),
                diff=d, ci95_low=lo, ci95_high=hi, welch_df=dof,
                pct_change=100 * d / ref.mean() if ref.mean() else np.nan,
                improvement_pct=(100 * d / abs(ref.mean()) * (1 if higher else -1)) if ref.mean() else np.nan,
                pairs_better=int(better.sum()), n_pairs=int(better.size),
                new_values=[round(x, 6) for x in new], ref_values=[round(x, 6) for x in ref])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    files = [p for p in sorted(a.ladder_dir.glob('ladder_layer*.csv')) if not p.name.endswith('.partial.csv')]
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if df.duplicated(['layer', 'family', 'name', 'seed']).any():
        raise SystemExit('duplicate SAE rows across ladder files')
    df['matched_l0'] = df.owt_l0.between(T_LOW, T_HIGH)
    excluded = df[~df.matched_l0 & df.family.isin(['old', 'r2'])]
    rows, crit = [], {}
    for L in sorted(df.layer.unique()):
        d = df[(df.layer == L) & df.matched_l0]
        for cid, new, ref, desc in COMPARISONS:
            nv = d[(d.family == new[0]) & (d.name == new[1])].sort_values('seed')
            rv = d[(d.family == ref[0]) & (d.name == ref[1])].sort_values('seed')
            for m, hib, lev in METRICS:
                if m not in d or (m.startswith('got_cities') and L not in GOT_INFORMATIVE) or (m.startswith('m4') and nv[m].isna().all()):
                    continue
                res = rule(nv[m], rv[m], hib)
                rows.append(dict(layer=L, comparison=cid, description=desc, metric=m, level=lev, higher_is_better=hib,
                                 n_new=len(nv), n_ref=len(rv), **{k: v for k, v in res.items()}))
        # random-decoder baseline (descriptive): seed 42 trained vs frozen decoder
        tr = d[(d.family == 'r2') & (d.name == 'topk') & (d.seed == 42)]
        fz = df[(df.layer == L) & (df.family == 'r2') & (df.name == 'topk_frozen_decoder')]
        if len(tr) and len(fz):
            crit[f'layer{L}_random_baseline'] = {m: dict(trained=float(tr[m].iloc[0]), frozen_decoder=float(fz[m].iloc[0]))
                                                 for m, _, _ in METRICS if m in tr and pd.notna(tr[m].iloc[0])}
    comp = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ('comparisons.csv', 'criteria.json', 'excluded_unmatched.csv'):
        if (a.out / f).exists():
            raise SystemExit(f'{a.out / f} exists; refusing to overwrite.')
    comp.to_csv(a.out / 'comparisons.csv', index=False)
    excluded.to_csv(a.out / 'excluded_unmatched.csv', index=False)
    crit['counts'] = comp.groupby(['comparison', 'cls']).size().unstack(fill_value=0).to_dict('index')
    crit['n_cells'] = int(len(comp))
    (a.out / 'criteria.json').write_text(json.dumps(crit, indent=2, default=float))
    print(comp.pivot_table(index=['comparison', 'metric'], columns='layer', values='cls', aggfunc='first').to_string())


if __name__ == '__main__':
    main()
