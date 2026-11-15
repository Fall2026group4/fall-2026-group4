#!/usr/bin/env python3
"""R1/R2 data preparation: split the RepE TRAIN statements into halves A (SAE fine-tuning, R2 feature selection is
separate and uses all train) and B (independent evaluation probes), and record the validation exclusion list.

Run from repository root:
    python scripts/repe/14_prepare_r1_r2_splits_and_wording.py --splits     # writes the A/B split + verification

Pre-registered in results/repe/r1_honesty_preservation_layer6/README.md (§2, §5):
  * grouping on the normalised statement (lower-case, non-alphanumerics -> space, collapsed whitespace), so the
    duplicate train statements 436/509 ("The sun/Sun is a star.") fall into the same half;
  * seed 20261107; balanced by factual label (per label, half of the permuted groups -> A, the rest -> B);
  * validation statement 326 (near-duplicate of train statement 589) is excluded from every R1/R2 validation score.
Only train statements are assigned. Validation and test rows are not modified.

The held-out-wording TEST set (§2 "needs approval") is DEFERRED: the user asked on 2026-11-07 to preserve the final
test split, so `--heldout-wording` refuses to run until that is approved and recorded.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / 'data/repe/persona_pairs.csv'
CROSS = REPO / 'data/repe/cross_wording_validation.csv'
SPLIT_CSV = REPO / 'data/repe/splits/r1_train_AB_groups.csv'
OUT = REPO / 'results/repe/r1_honesty_preservation_layer6'
SEED = 20261107
EXCLUDED_VALIDATION = (326,)   # near-duplicate of train 589 ("... 776 B.C." vs "... 776 BC.")


def norm(t):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', t.lower())).strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_split():
    p = pd.read_csv(PAIRS)
    s = p.drop_duplicates('statement_id')[['statement_id', 'statement', 'factual_label', 'split']].copy()
    s['norm_group'] = s.statement.map(norm)
    tr = s[s.split == 'train'].copy()
    if tr.groupby('norm_group').factual_label.nunique().max() != 1:
        raise SystemExit('a normalised group has mixed factual labels')
    groups = tr.groupby('norm_group').factual_label.first().sort_index()
    rng = np.random.default_rng(SEED)
    half = {}
    for lab in (0, 1):
        g = groups[groups == lab].index.to_numpy()
        g = g[rng.permutation(len(g))]
        n_a = len(g) // 2
        half.update({x: 'A' for x in g[:n_a]})
        half.update({x: 'B' for x in g[n_a:]})
    tr['half'] = tr.norm_group.map(half)
    return s, tr.sort_values('statement_id')[['statement_id', 'statement', 'factual_label', 'norm_group', 'half']]


def verify(s, tr):
    other = s[s.split != 'train']
    gA, gB = set(tr.norm_group[tr.half == 'A']), set(tr.norm_group[tr.half == 'B'])
    n_val = s[s.split == 'validation']
    cw = pd.read_csv(CROSS)
    res = {
        'split_file': SPLIT_CSV.relative_to(REPO).as_posix(), 'seed': SEED,
        'statements_A': int((tr.half == 'A').sum()), 'statements_B': int((tr.half == 'B').sum()),
        'rows_A': int(2 * (tr.half == 'A').sum()), 'rows_B': int(2 * (tr.half == 'B').sum()),
        'factual_label_mean_A': float(tr.factual_label[tr.half == 'A'].mean()),
        'factual_label_mean_B': float(tr.factual_label[tr.half == 'B'].mean()),
        'A_B_group_overlap': len(gA & gB),
        'train_vs_val_test_exact_normalised_overlap': len((gA | gB) & set(other.statement.map(norm))),
        'val_or_test_statements_in_split_file': int(tr.statement_id.isin(other.statement_id).sum()),
        'all_train_statements_assigned': bool(set(tr.statement_id) == set(s.statement_id[s.split == 'train'])),
        'duplicate_436_509_same_half': bool(tr.set_index('statement_id').loc[[436, 509], 'half'].nunique() == 1),
        'excluded_validation_statements': list(EXCLUDED_VALIDATION),
        'excluded_are_validation': bool(set(EXCLUDED_VALIDATION) <= set(n_val.statement_id)),
        'validation_statements_scored': int(n_val.statement_id.nunique() - len(EXCLUDED_VALIDATION)),
        'cross_wording_statements_scored': int(cw.statement_id.nunique() - cw.statement_id.isin(EXCLUDED_VALIDATION).any()),
        'cross_wording_statements_are_validation': bool(set(cw.statement_id) <= set(n_val.statement_id)),
        'persona_pairs_sha256': sha(PAIRS), 'cross_wording_sha256': sha(CROSS),
    }
    res['passes'] = bool(res['A_B_group_overlap'] == 0 and res['train_vs_val_test_exact_normalised_overlap'] == 0
                         and res['val_or_test_statements_in_split_file'] == 0 and res['all_train_statements_assigned']
                         and res['duplicate_436_509_same_half'] and res['excluded_are_validation']
                         and res['cross_wording_statements_are_validation']
                         and abs(res['factual_label_mean_A'] - res['factual_label_mean_B']) < 0.02
                         and min(res['statements_A'], res['statements_B']) >= 200)
    return res


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--splits', action='store_true')
    g.add_argument('--verify', action='store_true', help='re-derive the split and compare with the saved file')
    g.add_argument('--heldout-wording', action='store_true')
    a = ap.parse_args()
    if a.heldout_wording:
        raise SystemExit('Held-out-wording TEST set is deferred (test split preserved, 2026-11-07); needs explicit approval.')
    s, tr = make_split()
    if a.verify:
        saved = pd.read_csv(SPLIT_CSV)
        same = saved[['statement_id', 'half']].reset_index(drop=True).equals(tr[['statement_id', 'half']].reset_index(drop=True))
        res = verify(s, saved)
        res['rederived_split_identical'] = same
        print(json.dumps(res, indent=2))
        sys.exit(0 if (same and res['passes']) else 1)
    for p in (SPLIT_CSV, OUT / 'split_verification.json'):
        if p.exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
    res = verify(s, tr)
    if not res['passes']:
        print(json.dumps(res, indent=2))
        raise SystemExit('split verification failed; nothing written')
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(SPLIT_CSV, index=False)
    res['split_file_sha256'] = sha(SPLIT_CSV)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'split_verification.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
