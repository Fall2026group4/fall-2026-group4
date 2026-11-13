#!/usr/bin/env python3
"""X0 analysis: token exposure within the R2 runs (results/openwebtext/r2_scaled_saes/scaling/README.md).

Reads the light ladder rows of the 10M/25M/50M checkpoints (scaling/ladder_layer*_r2scaling.csv) and the committed
ladder rows of the final checkpoints (ladder/ladder_layer*_r2_resjb.csv; same harness, same validation data).
Paired-by-seed rule per family x layer x step: IMPROVES if all seeds improve and |mean paired diff| > 2 sd; WORSENS
mirror; else NO CLEAR CHANGE. Rows with OWT L0 outside 128 +- 10 % are excluded from comparisons (reported).
50M -> final is labelled 'exposure + LR decay'. VALIDATION, descriptive. Outputs refuse to be overwritten.

Run: python scripts/shared/r2_scaling_analysis.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
R2 = REPO / 'results/openwebtext/r2_scaled_saes'
OUT = R2 / 'scaling'
BUDGET = {'r2_10M': 10e6, 'r2_25M': 25e6, 'r2_50M': 50e6, 'r2': 99_999_744}
STEPS = [('r2_10M', 'r2_25M', 'exposure only (constant LR)'), ('r2_25M', 'r2_50M', 'exposure only (constant LR)'),
         ('r2_50M', 'r2', 'exposure + LR decay (confounded)')]
NAMES = ('topk', 'batchtopk', 'relu', 'topk_w3072')
METRICS = [('owt_nmse', False), ('owt_explained_variance', True), ('owt_cosine', True), ('owt_dead_fraction', False),
           ('splice_delta_ce', False), ('splice_kl_mean', False), ('splice_sae_vs_noise_delta_ce_ratio', False)]
L0_LO, L0_HI = 0.9 * 128, 1.1 * 128
log = logging.getLogger('x0')


def load():
    sc = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('ladder_layer*_r2scaling.csv'))], ignore_index=True)
    fin = pd.concat([pd.read_csv(p) for p in sorted((R2 / 'ladder').glob('ladder_layer*_r2_resjb.csv'))], ignore_index=True)
    fin = fin[(fin.family == 'r2') & fin.name.isin(NAMES)]
    df = pd.concat([sc, fin[sc.columns.intersection(fin.columns)]], ignore_index=True)
    df['tokens'] = df.family.map(BUDGET)
    df['matched_l0'] = df.owt_l0.between(L0_LO, L0_HI)
    return df


def paired(a, b, higher):
    d = (b - a) * (1 if higher else -1)            # positive = improvement
    if len(d) < 2 or d.isna().any():
        return 'NOT EVALUATED', np.nan, np.nan
    mu, sd = d.mean(), d.std(ddof=1)
    if (d > 0).all() and abs(mu) > 2 * sd:
        return 'IMPROVES', mu, sd
    if (d < 0).all() and abs(mu) > 2 * sd:
        return 'WORSENS', mu, sd
    return 'NO CLEAR CHANGE', mu, sd


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', force=True)
    outs = [OUT / f for f in ('per_checkpoint.csv', 'comparisons.csv', 'summary.json')]
    for f in outs:
        if f.exists():
            raise SystemExit(f'{f} exists; refusing to overwrite.')
    df = load()
    df.to_csv(outs[0], index=False)
    rows = []
    for (L, n), g in df.groupby(['layer', 'name']):
        for a_, b_, lab in STEPS:
            A = g[g.family == a_].set_index('seed').sort_index()
            B = g[g.family == b_].set_index('seed').sort_index()
            seeds = A.index.intersection(B.index)
            ok = A.loc[seeds, 'matched_l0'] & B.loc[seeds, 'matched_l0']
            for m, hib in METRICS:
                if not ok.all():
                    cls, mu, sd = 'EXCLUDED (L0 outside 128 +- 10 %)', np.nan, np.nan
                else:
                    cls, mu, sd = paired(A.loc[seeds, m], B.loc[seeds, m], hib)
                rows.append(dict(layer=L, name=n, step=f'{a_}->{b_}', step_type=lab, metric=m, higher_is_better=hib,
                                 from_mean=A.loc[seeds, m].mean(), to_mean=B.loc[seeds, m].mean(),
                                 improvement_mean=mu, improvement_sd=sd,
                                 pct_change=100 * (B.loc[seeds, m].mean() - A.loc[seeds, m].mean()) / A.loc[seeds, m].mean(),
                                 cls=cls, n_seeds=len(seeds)))
    comp = pd.DataFrame(rows)
    comp.to_csv(outs[1], index=False)
    counts = comp.groupby(['step', 'metric', 'cls']).size().unstack(fill_value=0)
    summ = dict(evidence='VALIDATION, descriptive', n_checkpoint_rows=int(len(df)),
                unmatched_l0=df[~df.matched_l0][['layer', 'family', 'name', 'seed', 'owt_l0']].to_dict('records'),
                counts={f'{s}|{m}': r.to_dict() for (s, m), r in counts.iterrows()})
    outs[2].write_text(json.dumps(summ, indent=2, default=float) + '\n')
    log.info('\n%s', counts.to_string())
    figure(df)


def figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    for j, (m, lab) in enumerate((('owt_nmse', 'OWT NMSE (lower is better)'),
                                  ('splice_delta_ce', 'spliced ΔCE, nats (lower is better)'), ('owt_l0', 'OWT L0'))):
        for i, L in enumerate(sorted(df.layer.unique())):
            ax = axes[i, j]
            for n in NAMES:
                g = df[(df.layer == L) & (df.name == n)].groupby('tokens')[m]
                mu, sd = g.mean(), g.std()
                ax.errorbar(mu.index / 1e6, mu.values, yerr=sd.values, marker='o', capsize=2, label=n)
            if m == 'owt_l0':
                ax.axhspan(L0_LO, L0_HI, color='0.9')
            ax.axvspan(80, 100, color='0.95')
            ax.set_xscale('log')
            ax.set_title(f'L{L}: {lab}', fontsize=9)
            if i == 2:
                ax.set_xlabel('training tokens (M); shaded = LR decay phase')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('X0 (VALIDATION): R2 checkpoints vs token exposure; width, recipe, seed and data stream fixed', fontsize=10)
    fig.tight_layout()
    fdir = OUT / 'figures'
    fdir.mkdir(exist_ok=True)
    for ext in ('png', 'svg', 'pdf'):
        p = fdir / f'x0_exposure_curve.{ext}'
        if p.exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
        fig.savefig(p, dpi=150)


if __name__ == '__main__':
    main()
