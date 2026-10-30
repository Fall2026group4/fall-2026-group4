#!/usr/bin/env python3
"""Stage 5: Test truth decodability from SAE-reconstructed activations.

Runs only on train and validation data. Never fits or evaluates on test.
Run from the project root with the existing dl-venv.
"""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STAGE4 = Path('scripts/Geometry of Truth/04_sae_truth_preservation.py')
OUT = Path('results/got/sae_diagnostic_truth_probes')
PREVIOUS = Path('results/got/sae_truth_preservation/validation_truth_preservation.csv')
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)


def load_stage4():
    if not STAGE4.is_file():
        raise FileNotFoundError(STAGE4)
    spec = importlib.util.spec_from_file_location('got_stage4', STAGE4)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(4)
    s4 = load_stage4()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sae_module = s4.load_sae_module()
    previous = pd.read_csv(PREVIOUS)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print('Device:', device, '| Stage 5 diagnostic probes | validation only', flush=True)

    for domain, names in s4.DOMAINS.items():
        for layer in s4.LAYERS:
            X, y, split, source = s4.load_domain(names, layer)
            train, val = split == 'train', split == 'val'
            if not train.any() or not val.any() or np.any(split == 'test'):
                raise ValueError('Invalid train/validation selection')
            for arch in s4.ARCHITECTURES:
                sae = s4.load_sae(sae_module, layer, arch, device)
                # SAE weights stay frozen. Reconstruct train and validation separately.
                X_train_rec, _ = s4.reconstruct(sae, X[train], device)
                X_val_rec, _ = s4.reconstruct(sae, X[val], device)
                del sae

                options = []
                for C in C_VALUES:
                    probe = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, random_state=s4.SEED),
                    )
                    probe.fit(X_train_rec, y[train])
                    scores = probe.predict_proba(X_val_rec)[:, 1]
                    options.append((s4.safe_auc(y[val], scores), C, scores))

                # Select C on validation, as in Stage 3. This is exploratory.
                best_auc, best_C, best_scores = sorted(
                    options, key=lambda item: (-item[0], item[1])
                )[0]
                match = previous[
                    (previous['domain'] == domain)
                    & (previous['layer'] == layer)
                    & (previous['architecture'] == arch)
                ]
                if len(match) != 1:
                    raise ValueError(f'Missing Stage 4 baseline: {domain} {layer} {arch}')
                baseline = match.iloc[0]
                row = {
                    'domain': domain, 'layer': layer, 'architecture': arch,
                    'train_n': int(train.sum()), 'validation_n': int(val.sum()),
                    'best_C': best_C,
                    'original_auroc': float(baseline['original_auroc']),
                    'frozen_probe_on_sae_auroc': float(baseline['sae_auroc']),
                    'sae_trained_probe_auroc': best_auc,
                    'diagnostic_gain_over_frozen': best_auc - float(baseline['sae_auroc']),
                }
                for name in names:
                    mask = source[val] == name
                    row[f'{name}_diagnostic_auroc'] = s4.safe_auc(y[val][mask], best_scores[mask])
                rows.append(row)
                print(
                    f'{domain:10s} L{layer:2d} {arch:8s} | '
                    f'original={row["original_auroc"]:.4f} | '
                    f'frozen→SAE={row["frozen_probe_on_sae_auroc"]:.4f} | '
                    f'SAE-trained→SAE={best_auc:.4f} | C={best_C:g}',
                    flush=True,
                )

    pd.DataFrame(rows).to_csv(OUT / 'validation_diagnostic_probes.csv', index=False)
    with (OUT / 'run_info.json').open('w') as f:
        json.dump({
            'training': 'SAE reconstructions of original train split only',
            'evaluation': 'SAE reconstructions of validation split only',
            'SAE_weights': 'frozen', 'test_evaluated': False,
            'C_selection': 'validation AUROC (exploratory)',
            'C_values': list(C_VALUES), 'seed': s4.SEED,
        }, f, indent=2)
    print('\nSaved:', OUT / 'validation_diagnostic_probes.csv')
    print('Test set remains untouched.')


if __name__ == '__main__':
    main()
