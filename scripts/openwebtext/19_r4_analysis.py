#!/usr/bin/env python3
"""R4 pre-registered comparisons K1-K4 (results/openwebtext/r4_kl_finetune/README.md), VALIDATION.

Reads the R4 ladder (families r4_kl, r4_mse) and the R2 ladder (family r2, name topk) at layers 8 and 11.
Rule (IMP-L), per layer and metric, paired by seed: better on all 3 seeds AND |mean paired difference| > 2 sd.
Guard: mean OWT EV of kl >= mean OWT EV of mse - 0.02; if it fails, an improving cell is NOT CREDITABLE.
Classes: PRELIMINARY IMPROVEMENT / NO IMPROVEMENT / REGRESSION / NOT CREDITABLE.

Run: python scripts/openwebtext/19_r4_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/openwebtext/r4_kl_finetune'
R2L = REPO / 'results/openwebtext/r2_scaled_saes/ladder'
CMP = [  # id, new family, reference family, metrics (metric, higher_is_better)
    ('K1', 'r4_kl', 'r4_mse', [('splice_delta_ce', False), ('splice_kl_mean', False)]),
    ('K2', 'r4_kl', 'r4_mse', [('repe_persona__refit_logistic_gap', False), ('got_cities__refit_logistic_gap', False),
                               ('repe_persona__frozen_logistic_gap', False), ('got_cities__frozen_logistic_gap', False)]),
    ('K3', 'r4_kl', 'r4_mse', [('m4__preservation_rec', True)]),
    ('K4', 'r4_kl', 'r2', [('splice_delta_ce', False), ('splice_kl_mean', False)]),
]
DESCRIPTIVE = [('owt_explained_variance', True), ('owt_l0', None), ('rlm__pair_dir_cosine', True),
               ('repe_persona__sae_minus_pca_refit_gap', False)]


def main():
    files = [p for p in sorted((OUT / 'ladder').glob('ladder_layer*.csv')) if not p.name.endswith('.partial.csv')]
    files += [p for p in sorted(R2L.glob('ladder_layer*_r2_resjb.csv'))]
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df[(df.name == 'topk') & df.family.isin(['r4_kl', 'r4_mse', 'r2']) & df.layer.isin([8, 11])]
    rows, guard = [], {}
    for L in (8, 11):
        d = df[df.layer == L].set_index(['family', 'seed'])
        ev = {f: d.loc[f].owt_explained_variance.mean() for f in ('r4_kl', 'r4_mse')}
        guard[L] = dict(ev_kl=ev['r4_kl'], ev_mse=ev['r4_mse'], ok=bool(ev['r4_kl'] >= ev['r4_mse'] - 0.02))
        for cid, new, ref, metrics in CMP:
            for m, hib in metrics + [(x, h) for x, h in DESCRIPTIVE if cid == 'K1']:
                if m not in d:
                    continue
                n, r = d.loc[new][m].sort_index(), d.loc[ref][m].sort_index()
                assert list(n.index) == list(r.index) == [42, 1042, 2042]
                diff = (n - r).to_numpy(float)
                mean, sd = diff.mean(), diff.std(ddof=1)
                rec = dict(layer=L, comparison=cid, metric=m, new=new, ref=ref, new_mean=n.mean(), ref_mean=r.mean(),
                           new_sd=n.std(ddof=1), ref_sd=r.std(ddof=1), diff_mean=mean, diff_sd=sd,
                           new_values=n.round(6).tolist(), ref_values=r.round(6).tolist())
                if hib is None or (m, hib) in DESCRIPTIVE:
                    rec['cls'] = 'descriptive'
                else:
                    good = diff > 0 if hib else diff < 0
                    bad = diff < 0 if hib else diff > 0
                    big = abs(mean) > 2 * sd
                    rec['improvement_pct'] = 100 * mean / abs(r.mean()) * (1 if hib else -1)
                    if good.all() and big:
                        rec['cls'] = 'PRELIMINARY IMPROVEMENT' if guard[L]['ok'] else 'NOT CREDITABLE'
                    elif bad.all() and big:
                        rec['cls'] = 'REGRESSION'
                    else:
                        rec['cls'] = 'NO IMPROVEMENT'
                rows.append(rec)
    out = pd.DataFrame(rows)
    for f in ('comparisons.csv', 'criteria.json'):
        if (OUT / f).exists():
            raise SystemExit(f'{OUT / f} exists; refusing to overwrite.')
    out.to_csv(OUT / 'comparisons.csv', index=False)
    (OUT / 'criteria.json').write_text(json.dumps(dict(guard=guard, rule='IMP-L paired by seed; guard EV(kl) >= EV(mse) - 0.02',
                                                       counts={f'{k[0]}|{k[1]}': int(v) for k, v in out.groupby(['comparison', 'cls']).size().items()}),
                                                  indent=2, default=str))
    pd.set_option('display.width', 250)
    print(out[['layer', 'comparison', 'metric', 'ref_mean', 'new_mean', 'diff_mean', 'diff_sd', 'cls']].round(4).to_string())
    print(guard)


if __name__ == '__main__':
    main()
