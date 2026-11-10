"""Independent, read-only verification of the M5 confirmatory TEST pass.

Written 2026-11-10 for the final defense audit (docs/final_defense/EXPERIMENT_AUDIT.md). It reads only committed
result files, never a checkpoint, an activation cache or a test row, and writes nothing.

Checks:
1. every hash in results/shared/m5_confirmatory_test/preregistration.json against the current file;
2. completeness of m5_rows.csv: expected SAE families per layer, no duplicate (layer, family, name, seed) keys,
   no duplicate checkpoint hashes, split == 'test', OWT L0 of compared SAEs within 128 +- 10 %;
3. re-application of the pre-registered R2 rule (all 9 seed pairs better AND Welch 95 % CI excludes 0) to every
   R2-rule row of comparisons.csv from its stored per-seed values, and of the IMP-L paired rule to K1/K4;
4. per-family mean and sd of the headline metrics, printed.

Run from the repository root:  python scripts/shared/m5_independent_verification.py
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
M5 = REPO / 'results/shared/m5_confirmatory_test'
HIGHER_IS_BETTER = {'owt_explained_variance', 'owt_cosine'}
EXPECTED = {1: 23, 8: 29, 11: 29}  # 9 old + 13 r2 + 1 resjb (+ 6 r4 at L8/L11)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def values(s):
    return [float(v) for v in ast.literal_eval(s.replace('np.float64', ''))]


def r2_rule(new, ref, higher):
    new, ref = np.asarray(new), np.asarray(ref)
    d = new.mean() - ref.mean()
    v1, v2 = new.var(ddof=1) / len(new), ref.var(ddof=1) / len(ref)
    se = np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(new) - 1) + v2 ** 2 / (len(ref) - 1)) if se > 0 else 1.0
    h = stats.t.ppf(0.975, dof) * se if se > 0 else 0.0
    lo, hi = d - h, d + h
    better = np.array([(x > y) if higher else (x < y) for x in new for y in ref])
    worse = np.array([(x < y) if higher else (x > y) for x in new for y in ref])
    if better.all() and ((lo > 0) if higher else (hi < 0)):
        return 'IMPROVEMENT'
    if worse.all() and ((hi < 0) if higher else (lo > 0)):
        return 'REGRESSION'
    return 'NO CLEAR CHANGE'


def paired_rule(new, ref):
    d = np.asarray(new) - np.asarray(ref)
    m, s = d.mean(), d.std(ddof=1)
    if (d < 0).all() and abs(m) > 2 * s:
        return 'IMPROVEMENT'
    if (d > 0).all() and abs(m) > 2 * s:
        return 'REGRESSION'
    return 'NO CLEAR CHANGE'


def main():
    failures = []
    pre = json.loads((M5 / 'preregistration.json').read_text())
    print('1. Pre-registration hashes')
    for rel, h in pre['hashes'].items():
        ok = sha(REPO / rel) == h
        note = '' if ok else ('  (expected: results were appended after the run)' if rel.endswith('README.md') else '')
        print(f"   {'OK  ' if ok else 'DIFF'} {rel}{note}")
        if not ok and not rel.endswith('README.md'):
            failures.append(f'hash {rel}')

    rows = pd.read_csv(M5 / 'm5_rows.csv')
    print('2. Row completeness')
    n_per_layer = rows.groupby('layer').size().to_dict()
    print('   rows per layer', n_per_layer, 'expected', EXPECTED)
    if n_per_layer != EXPECTED:
        failures.append('row count')
    dups = int(rows.duplicated(['layer', 'family', 'name', 'seed']).sum())
    dup_ck = int(rows.checkpoint_sha256.duplicated().sum())
    print('   duplicate keys', dups, '| duplicate checkpoint hashes', dup_ck, '| splits', sorted(rows.split.unique()))
    if dups or dup_ck or set(rows.split) != {'test'}:
        failures.append('duplicates or split')
    cmp_rows = rows[(rows.family != 'resjb') & (rows.name != 'topk_frozen_decoder')]
    lo, hi = cmp_rows.owt_l0.min(), cmp_rows.owt_l0.max()
    print(f'   OWT test L0 of compared SAEs: {lo:.2f} - {hi:.2f} (window 115.2 - 140.8)')
    if lo < 115.2 or hi > 140.8:
        failures.append('L0 window')
    checks = json.loads((M5 / 'checks.json').read_text())
    print('   checks.json:', {k: checks[k] for k in ('split', 'n_saes', 'test_rows_used', 'n_got_eval_rows')})
    if checks['script_sha256'] != pre['hashes']['scripts/shared/m5_confirmatory_test.py']:
        failures.append('script hash at run time')

    print('3. Decision rules re-applied')
    comp = pd.read_csv(M5 / 'comparisons.csv')
    mism = 0
    for _, r in comp.iterrows():
        new, ref = values(r.new_values), values(r.ref_values)
        cls = r2_rule(new, ref, r.metric in HIGHER_IS_BETTER) if r.rule.startswith('R2') else paired_rule(new, ref)
        if cls != r.cls:
            mism += 1
            print('   MISMATCH', r.layer, r.comparison, r.metric, cls, r.cls)
    print(f'   {len(comp)} rows re-checked, {mism} mismatches')
    if mism:
        failures.append('rule mismatch')
    print(comp.groupby(['comparison', 'cls']).size().unstack(fill_value=0).to_string())

    print('4. Family means (sd) on TEST')
    cols = ['owt_nmse', 'owt_explained_variance', 'owt_cosine', 'owt_l0', 'owt_dead_fraction', 'splice_delta_ce',
            'splice_kl_mean', 'got_cities__refit_logistic_gap', 'got_cities__sae_minus_pca_refit_gap']
    g = rows.groupby(['layer', 'family', 'name'])[cols]
    out = g.mean().round(4).astype(str) + ' (' + g.std().round(4).fillna(0).astype(str) + ')'
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(out.to_string())

    print('RESULT:', 'ALL CHECKS PASS' if not failures else f'FAILURES: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
