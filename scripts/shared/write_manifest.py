#!/usr/bin/env python3
"""Project-wide experiment manifest: docs/experiment_manifest.csv.

Formerly scripts/professor_review/write_manifest.py, which wrote a hard-coded manifest for the
2026-11-04 professor-review experiments. That original output is preserved, byte-identical, as
docs/experiment_manifest_2026-11-04.csv and is the provenance record (its `script_sha256_16` column is
the hash of each script *at run time*).

This script builds the living manifest from that historical record:
  * keeps every original column, renaming `script_sha256_16` -> `script_sha256_16_at_run`
  * adds `script_path_current` and `outputs_current` (post-migration locations, 2026-11-06)
  * adds `script_sha256_16_current` so any later edit to a script is visible

Future experiments append rows with `append_experiment(...)`; they are never overwritten.
Run from repository root:  python scripts/shared/write_manifest.py
"""
import hashlib
from pathlib import Path

import pandas as pd

HISTORICAL = Path('docs/experiment_manifest_2026-11-04.csv')
MANIFEST = Path('docs/experiment_manifest.csv')

SCRIPT_MOVES = {
    'scripts/professor_review/analyze_owt_corrections.py': 'scripts/openwebtext/14_sparsity_matching_analysis.py',
    'scripts/professor_review/got_layer10_frozen_probe.py': 'scripts/geometry_of_truth/10_layer10_frozen_probe.py',
    'scripts/professor_review/finetuning_repe_got_eval.py': 'scripts/shared/finetuning_heldout_eval.py',
    'src/tests/test_professor_review.py': 'src/tests/test_faithfulness_metrics.py',
}
OUTPUTS_CURRENT = {
    'A': 'results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/; '
         'results/openwebtext/convergence/layer_{2,6,10}/*_l0_128_uninterrupted_seed*/',
    'B': 'results/openwebtext/layer_comparison/sparsity_comparison.csv, sparsity_matched_cells.csv',
    'C': 'results/repe/steering/causal_random_controls/',
    'D': 'results/gemma/steering_sweep_layer11/',
    'E': 'results/geometry_of_truth/layer10_frozen_probe/, results/geometry_of_truth/got_faithfulness_frozen_probes.csv',
    'F/G': 'results/repe/representation_preservation_val_test.csv, '
           'results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv, '
           'results/geometry_of_truth/got_faithfulness_frozen_probes.csv',
    'I': '',
}


def sha16(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else 'MISSING'


def append_experiment(row: dict, manifest: Path = MANIFEST):
    """Append one experiment row (dict of manifest columns). Existing rows are never modified."""
    row = dict(row)
    row.setdefault('script_path_current', row.get('script', ''))
    row['script_sha256_16_at_run'] = sha16(row['script_path_current'])
    row['script_sha256_16_current'] = row['script_sha256_16_at_run']
    df = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(manifest, index=False)


def build():
    if MANIFEST.exists():
        raise SystemExit(f'{MANIFEST} exists; refusing to rebuild (append with append_experiment instead)')
    df = pd.read_csv(HISTORICAL, dtype=str, keep_default_na=False)
    df = df.rename(columns={'script_sha256_16': 'script_sha256_16_at_run'})
    df['script_path_current'] = df['script'].map(lambda s: SCRIPT_MOVES.get(s, s))
    df['outputs_current'] = df['task'].map(OUTPUTS_CURRENT)
    df['script_sha256_16_current'] = df['script_path_current'].map(sha16)
    df['migrated'] = '2026-11-06: outputs moved from results/professor_review_corrections/ (see docs/migration_manifest.csv)'
    df.to_csv(MANIFEST, index=False)
    print(df[['task', 'script_path_current', 'script_sha256_16_at_run', 'script_sha256_16_current']].to_string(index=False))


if __name__ == '__main__':
    build()
