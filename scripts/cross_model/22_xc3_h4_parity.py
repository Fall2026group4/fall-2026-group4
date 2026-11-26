#!/usr/bin/env python3
"""XC-3 H4 (pre-registered, descriptive): cross-model parity at depth-matched pairs from the saved TEST per-pair KLs.

For each depth pair (GPT-2 L, Gemma L): E_raw, E_rec, E_patch and retention R = E_rec / E_raw per model, and
R_gemma - R_gpt2 with a statement-clustered bootstrap that resamples the SAME statements for both models (2000, seed 0).
The difference is computed only where H1 holds in both models (pre-registration). No GPU; reads results only.
Run: python scripts/cross_model/22_xc3_h4_parity.py
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = REPO / 'results/cross_model/xc3_steering_parity/test'
PAIRS = [(3, 5), (6, 12), (9, 19)]


def per_statement(npz, keys):
    sid = npz['statement_id'].astype(str)
    u, inv = np.unique(sid, return_inverse=True)
    return u, {k: np.bincount(inv, npz[k], len(u)) for k in keys}


def main():
    out = D / 'h4_parity.json'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    res = json.loads((D / 'results.json').read_text())
    h1 = {(r['model'], r['layer']): r['H1_raw_beats_controls'] for r in res['rows']}
    data = {m: np.load(D / f'{m}_test_per_pair.npz') for m in ('gpt2', 'gemma')}
    rng = np.random.default_rng(0)
    rows = []
    for lg, lm in PAIRS:
        ka = ['kl_HU', f'L{lg}_raw', f'L{lg}_rec', f'L{lg}_patch']
        kb = ['kl_HU', f'L{lm}_raw', f'L{lm}_rec', f'L{lm}_patch']
        ua, A = per_statement(data['gpt2'], ka)
        ub, B = per_statement(data['gemma'], kb)
        assert (ua == ub).all(), 'statement sets differ between models'
        E = lambda S, k, idx=slice(None): 1 - S[k][idx].sum(-1) / S['kl_HU'][idx].sum(-1)
        r = dict(gpt2_layer=lg, gemma_layer=lm, n_statements=len(ua),
                 gpt2=dict(E_raw=float(E(A, ka[1])), E_rec=float(E(A, ka[2])), E_patch=float(E(A, ka[3]))),
                 gemma=dict(E_raw=float(E(B, kb[1])), E_rec=float(E(B, kb[2])), E_patch=float(E(B, kb[3]))),
                 H1_both=bool(h1[('gpt2', lg)] and h1[('gemma', lm)]))
        for m in ('gpt2', 'gemma'):
            r[m]['R'] = r[m]['E_rec'] / r[m]['E_raw']
        if r['H1_both']:
            idx = rng.integers(0, len(ua), (2000, len(ua)))
            Ra = E(A, ka[2], idx) / E(A, ka[1], idx)
            Rb = E(B, kb[2], idx) / E(B, kb[1], idx)
            d = Rb - Ra
            r['R_gemma_minus_gpt2'] = float(r['gemma']['R'] - r['gpt2']['R'])
            r['R_diff_ci95'] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
            dr = E(B, kb[1], idx) - E(A, ka[1], idx)
            r['E_raw_gemma_minus_gpt2'] = float(r['gemma']['E_raw'] - r['gpt2']['E_raw'])
            r['E_raw_diff_ci95'] = [float(np.quantile(dr, .025)), float(np.quantile(dr, .975))]
        else:
            r['R_gemma_minus_gpt2'] = 'not computed (H1 not met in both models)'
        rows.append(r)
    out.write_text(json.dumps(dict(evidence='descriptive (H4, pre-registered)', rows=rows), indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
