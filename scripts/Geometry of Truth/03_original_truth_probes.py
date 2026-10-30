#!/usr/bin/env python3
"""Train original-activation truth probes on grouped Geometry of Truth splits.

Run from repository root:
    python "scripts/Geometry of Truth/03_original_truth_probes.py"

Uses train and validation only. Does NOT read or score test rows.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('data/got')
ACT_DIR = ROOT / 'activations' / 'gpt2_small'
SPLIT_DIR = ROOT / 'splits'
OUT_DIR = Path('results/got/original_truth_probes')
DOMAINS = {
    'geographic': ('cities', 'neg_cities'),
    'numerical': ('larger_than', 'smaller_than'),
}
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
SEED = 42


def load_domain(names):
    parts = []
    for name in names:
        data = torch.load(ACT_DIR / f'{name}_final_token.pt', map_location='cpu', weights_only=False)
        splits = pd.read_csv(SPLIT_DIR / f'{name}_splits.csv')
        row_ids = np.asarray(data['row_ids'], dtype=np.int64)
        labels = np.asarray(data['labels'], dtype=np.int64)
        if len(row_ids) != len(splits) or len(set(row_ids)) != len(row_ids):
            raise ValueError(f'{name}: row count or IDs invalid')
        aligned = splits.set_index('row_id').reindex(row_ids)
        if aligned.isna().any().any():
            raise ValueError(f'{name}: missing split assignments')
        if not np.array_equal(labels, aligned['label'].to_numpy(dtype=np.int64)):
            raise ValueError(f'{name}: labels do not match activation file')
        if not set(aligned['split'].unique()).issubset({'train', 'val', 'test'}):
            raise ValueError(f'{name}: invalid split names')
        layers = {int(k): v for k, v in data['activations'].items()}
        if set(layers) != set(range(12)):
            raise ValueError(f'{name}: expected layers 0..11, found {sorted(layers)}')
        for layer, tensor in layers.items():
            if tensor.shape != (len(row_ids), 768):
                raise ValueError(f'{name}, layer {layer}: unexpected shape {tensor.shape}')
        parts.append({'name': name, 'labels': labels, 'splits': aligned['split'].to_numpy(),
                      'groups': aligned['group_id'].astype(str).to_numpy(), 'layers': layers})

    # Assert no group crosses splits across datasets within the domain.
    group_to_split = {}
    for part in parts:
        for group, split in zip(part['groups'], part['splits']):
            if group in group_to_split and group_to_split[group] != split:
                raise ValueError(f'Group leakage detected: {group}')
            group_to_split[group] = split
    return parts


def safe_auc(labels, scores):
    if np.unique(labels).size < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    chosen_rows = []
    for domain, names in DOMAINS.items():
        print(f'\nDomain: {domain} ({", ".join(names)})', flush=True)
        parts = load_domain(names)
        y = np.concatenate([p['labels'] for p in parts])
        split = np.concatenate([p['splits'] for p in parts])
        source = np.concatenate([np.repeat(p['name'], len(p['labels'])) for p in parts])
        train = split == 'train'
        val = split == 'val'
        # Test rows are never passed to fit(), predict_proba(), or metrics.
        print(f'Train={train.sum()}, validation={val.sum()}, test reserved={(split == "test").sum()}')
        if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
            raise ValueError(f'{domain}: both train and validation must contain both labels')
        for layer in range(12):
            X = np.concatenate([p['layers'][layer].numpy() for p in parts], axis=0)
            if not np.isfinite(X[train | val]).all():
                raise ValueError(f'{domain} layer {layer}: non-finite activations')
            layer_results = []
            for C in C_VALUES:
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=C, max_iter=3000, random_state=SEED),
                )
                probe.fit(X[train], y[train])
                scores = probe.predict_proba(X[val])[:, 1]
                row = {'domain': domain, 'layer': layer, 'C': C,
                       'validation_auroc': safe_auc(y[val], scores),
                       'train_n': int(train.sum()), 'val_n': int(val.sum())}
                for name in names:
                    mask = val & (source == name)
                    row[f'{name}_val_auroc'] = safe_auc(y[mask], probe.predict_proba(X[mask])[:, 1])
                all_rows.append(row)
                layer_results.append(row)
            # Choose C using overall validation AUROC, then smallest C on ties.
            best = sorted(layer_results, key=lambda r: (-r['validation_auroc'], r['C']))[0]
            chosen_rows.append(best)
            print(f'  Layer {layer:2d}: val AUROC={best["validation_auroc"]:.4f}, C={best["C"]:g}', flush=True)

    pd.DataFrame(all_rows).to_csv(OUT_DIR / 'validation_all_C.csv', index=False)
    pd.DataFrame(chosen_rows).to_csv(OUT_DIR / 'validation_best_C_by_layer.csv', index=False)
    summary = {
        'selection_basis': 'validation AUROC only',
        'test_evaluated': False,
        'domains': DOMAINS,
        'C_grid': C_VALUES,
        'random_seed': SEED,
        'note': 'Best C selected independently per layer; probe weights are not saved. Refit on train with selected C for later frozen-probe comparison, using same original activation data and split.',
    }
    (OUT_DIR / 'run_info.json').write_text(json.dumps(summary, indent=2))
    print(f'\nSaved validation results to {OUT_DIR}')
    print('Test set untouched. No SAE evaluation performed.')


if __name__ == '__main__':
    main()
