#!/usr/bin/env python3
"""Corrected OWT protocol: uninterrupted 15000-step training for layers 2, 6, 10.

Professor-review correction (Task A). The original 15000-step checkpoints were
produced by two different procedures:
  * layer 6  : scratch 0->5000 (scripts 05/06/07), then 5000->15000 with the
               Adam state and sparsity-controller state RESTORED (script 08) but
               a re-seeded RNG stream.
  * layers 2,10 : 1000-step sweep weights (script 03) warm-started 1000->15000
               with a FRESH Adam state and a RESET sparsity controller (script 11).
Cross-layer comparisons therefore mix optimisation procedures.

Chosen protocol (applied identically to every layer/architecture/seed):
  a single uninterrupted run from random initialisation to 15000 steps, one Adam
  optimiser for the whole run, one sparsity controller for the whole run, one RNG
  stream seeded once. This is the procedure all other variants approximate, it
  needs no checkpoint surgery, and it is cheap (about 1 min per run on an A10G).

The training loop is the ORIGINAL `train_one` from scripts/openwebtext/03_training_sae.py
(imported, not re-implemented), called with steps=15000, so loss, sparsity controller,
gradient clipping and decoder normalisation are unchanged.

Test evaluation uses the ORIGINAL `evaluate` from 10_reconstruction_evaluation.py.
Original checkpoints and results are never modified.

Outputs (since the 2026-11-06 migration into the OWT dataset directories):
  metrics    results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/test_metrics_all_runs.csv
  config     results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/run_config.json
  checkpoint results/openwebtext/convergence/layer_{L}/{arch}_l0_{target}_uninterrupted_seed{S}/checkpoint_step_{steps}.pt
  smoke      results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/smoke/
Runs already present in test_metrics_all_runs.csv (same layer/architecture/seed) are skipped, so extending
to more layers (e.g. --layers 0 1 3 4 5 7 8 9 11) never repeats completed runs.

Run from repository root:
    python scripts/openwebtext/13_consistent_protocol_retrain.py            # full run
    python scripts/openwebtext/13_consistent_protocol_retrain.py --smoke    # 200-step check
"""
import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path('.')
TRAIN_SCRIPT = Path('scripts/openwebtext/03_training_sae.py')
EVAL_SCRIPT = Path('scripts/openwebtext/10_reconstruction_evaluation.py')
DATA = Path('data/activations/gpt2_small')
OUT = Path('results/openwebtext/reconstruction_evaluation/uninterrupted_protocol')
CKPT_ROOT = Path('results/openwebtext/convergence')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', nargs='+', type=int, default=[2, 6, 10])
    p.add_argument('--architectures', nargs='+', default=['relu', 'topk', 'jumprelu'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 1042, 2042])
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--target-l0', type=int, default=128)
    p.add_argument('--smoke', action='store_true', help='200 steps, layer 6, seed 42, separate dir')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required; refusing to silently train on CPU.')
    out = OUT
    if a.smoke:
        a.layers, a.seeds, a.steps = [6], [42], 200
        out = OUT / 'smoke'
    out.mkdir(parents=True, exist_ok=True)

    base = load(TRAIN_SCRIPT, 'owt_train')
    ev = load(EVAL_SCRIPT, 'owt_eval')
    device = torch.device('cuda')
    torch.set_num_threads(4)

    rows_path = out / 'test_metrics_all_runs.csv'
    rows = pd.read_csv(rows_path).to_dict('records') if rows_path.exists() else []
    done = {(r['layer'], r['architecture'], r['seed']) for r in rows}

    for layer in a.layers:
        train = validation = None
        for seed in a.seeds:
            for arch in a.architectures:
                if (layer, arch, seed) in done:
                    print(f'SKIP done layer={layer} arch={arch} seed={seed}', flush=True)
                    continue
                if train is None:
                    train = base.load_split(DATA, 'train', layer)
                    validation = base.load_split(DATA, 'validation', layer)
                    print(f'layer {layer}: train {tuple(train.shape)} val {tuple(validation.shape)}', flush=True)
                # Same hyper-parameters as the original sweep (03 defaults), only steps/seed/layer differ.
                args = argparse.Namespace(
                    layer=layer, features=3072, batch_size=256, steps=a.steps, lr=3e-4, seed=seed,
                    temperature=0.1, initial_sparsity_coeff=0.001, min_sparsity_coeff=1e-7,
                    max_sparsity_coeff=100.0, adjust_every=25, adjust_rate=0.5, match_tolerance=0.2,
                    log_every=2500)
                t0 = time.time()
                model, val_metrics = base.train_one(arch, a.target_l0, train, validation, args, device)
                train_seconds = time.time() - t0
                model.eval()
                for prm in model.parameters():
                    prm.requires_grad_(False)
                test = ev.evaluate(model, layer, device)
                row = dict(layer=layer, architecture=arch, seed=seed, steps=a.steps,
                           training_procedure='uninterrupted_from_scratch_single_optimizer',
                           target_l0=a.target_l0,
                           val_mse=val_metrics['mse'], val_cosine=val_metrics['cosine'],
                           val_explained_variance=val_metrics['explained_variance'], val_l0=val_metrics['l0'],
                           **{f'test_{k}': v for k, v in test.items()},
                           train_seconds=round(train_seconds, 1))
                row['test_l0_rel_error'] = abs(row['test_actual_l0'] - a.target_l0) / a.target_l0
                rows.append(row)
                if a.smoke:
                    ck = out / f'layer_{layer}' / f'{arch}_seed{seed}_step{a.steps}.pt'
                else:
                    ck = (CKPT_ROOT / f'layer_{layer}' / f'{arch}_l0_{a.target_l0}_uninterrupted_seed{seed}'
                          / f'checkpoint_step_{a.steps}.pt')
                if ck.exists():
                    raise SystemExit(f'refusing to overwrite existing checkpoint {ck}')
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model.cpu().state_dict(),
                            'config': {**vars(args), 'architecture': arch, 'target_l0': a.target_l0,
                                       'd_in': 768, 'procedure': row['training_procedure']},
                            'validation_metrics': val_metrics, 'test_metrics': test}, ck)
                pd.DataFrame(rows).to_csv(rows_path, index=False)
                print(f"DONE layer={layer} arch={arch} seed={seed} {train_seconds:.0f}s "
                      f"test_mse={test['mse']:.5f} EV={test['explained_variance']:.4f} "
                      f"L0={test['actual_l0']:.1f}", flush=True)
                del model
                torch.cuda.empty_cache()
        del train, validation

    # Accumulate provenance: a later invocation (e.g. extending to more layers) must not erase earlier ones.
    cfg_path = out / 'run_config.json'
    prev = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    invocations = prev.get('invocations') or (
        [{k: prev[k] for k in ('layers', 'architectures', 'seeds', 'steps') if k in prev}] if prev else [])
    invocations.append({'layers': a.layers, 'architectures': a.architectures, 'seeds': a.seeds,
                        'steps': a.steps, 'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
    cfg_path.write_text(json.dumps({
        'protocol': 'uninterrupted_from_scratch_single_optimizer',
        'layers': sorted({int(r['layer']) for r in rows}),
        'architectures': sorted({r['architecture'] for r in rows}),
        'seeds': sorted({int(r['seed']) for r in rows}), 'steps': a.steps,
        'invocations': invocations,
        'target_l0': a.target_l0, 'features': 3072, 'batch_size': 256, 'lr': 3e-4,
        'optimizer': 'Adam(default betas), grad-clip 10, decoder column renorm each step',
        'sparsity_controller': 'original multiplicative controller from 03_training_sae.py',
        'train_split': 'OWT train (800 docs, 101600 non-BOS tokens)',
        'validation_split': 'OWT validation (100 docs, 12700 tokens) - reported only, not used for selection',
        'test_split': 'OWT test (100 docs, 12700 tokens)',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
    }, indent=2))
    print('Saved', rows_path, flush=True)


if __name__ == '__main__':
    main()
