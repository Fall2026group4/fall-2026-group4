#!/usr/bin/env python3
"""Controlled ReLU SAE convergence test using the original adaptive L0 controller.

Run from repository root:
  python scripts/openwebtext/06_relu_convergence.py

Does not overwrite the original sparsity sweep or TopK convergence outputs.
"""
import csv
import importlib.util
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

BASELINE_SCRIPT = Path('scripts/openwebtext/03_training_sae.py')
OUTPUT = Path('results/openwebtext/convergence/layer_6/relu_l0_128')
LAYER, TARGET_L0, FEATURES = 6, 128, 3072
STEPS, BATCH_SIZE, LR, SEED = 5000, 256, 3e-4, 42
EVAL_EVERY = 250
INITIAL_SPARSITY_COEFF = 0.001
MIN_SPARSITY_COEFF = 1e-7
MAX_SPARSITY_COEFF = 100.0
ADJUST_EVERY = 25
ADJUST_RATE = 0.5
TEMPERATURE = 0.1


def load_baseline_module():
    if not BASELINE_SCRIPT.is_file():
        raise FileNotFoundError(f'Missing original training script: {BASELINE_SCRIPT}')
    spec = importlib.util.spec_from_file_location('original_sae_training', BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_curve(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: CSV saved, skipping plot.', flush=True)
        return
    xs = [r['step'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [r['validation_mse'] for r in rows], marker='o', label='Validation MSE')
    ax.axvline(1000, linestyle='--', color='gray', label='Original baseline endpoint (1000)')
    ax.set(xlabel='Training steps', ylabel='Validation reconstruction MSE',
           title='ReLU SAE convergence — GPT-2 layer 6, target L0=128')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'validation_mse_curve.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [r['validation_l0'] for r in rows], marker='o', label='Actual validation L0')
    ax.axhline(TARGET_L0, linestyle='--', color='gray', label='Target L0=128')
    ax.set(xlabel='Training steps', ylabel='Mean active features per token',
           title='ReLU sparsity convergence — GPT-2 layer 6')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'validation_l0_curve.png', dpi=170)
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this experiment')
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f'Output directory is not empty; refusing overwrite: {OUTPUT}')
    base = load_baseline_module()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    data_root = Path('data/activations/gpt2_small')
    train = base.load_split(data_root, 'train', LAYER)
    validation = base.load_split(data_root, 'validation', LAYER)
    print(f'Train {tuple(train.shape)}, validation {tuple(validation.shape)}', flush=True)
    model = base.SAE(train.shape[1], FEATURES, 'relu', TARGET_L0,
                     temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    coefficient = INITIAL_SPARSITY_COEFF
    ema_l0 = None
    settings = dict(layer=LAYER, architecture='relu', target_l0=TARGET_L0,
                    features=FEATURES, steps=STEPS, batch_size=BATCH_SIZE,
                    lr=LR, seed=SEED, eval_every=EVAL_EVERY,
                    initial_sparsity_coeff=INITIAL_SPARSITY_COEFF,
                    min_sparsity_coeff=MIN_SPARSITY_COEFF,
                    max_sparsity_coeff=MAX_SPARSITY_COEFF,
                    adjust_every=ADJUST_EVERY, adjust_rate=ADJUST_RATE,
                    temperature=TEMPERATURE, data_root=str(data_root),
                    original_script=str(BASELINE_SCRIPT))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'config.json').write_text(json.dumps(settings, indent=2))
    rows = []
    for step in range(1, STEPS + 1):
        ids = torch.randint(len(train), (BATCH_SIZE,))
        x = train[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        rec, z, _ = model(x)
        mse = F.mse_loss(rec, x)
        penalty = z.sum(dim=-1).mean()
        loss = mse + coefficient * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            batch_l0 = (z > 0).sum(dim=-1).float().mean().item()
        ema_l0 = batch_l0 if ema_l0 is None else 0.9 * ema_l0 + 0.1 * batch_l0
        if step % ADJUST_EVERY == 0:
            ratio = max(ema_l0, 1.0) / TARGET_L0
            coefficient *= math.exp(ADJUST_RATE * max(-2.0, min(2.0, math.log(ratio))))
            coefficient = max(MIN_SPARSITY_COEFF,
                              min(MAX_SPARSITY_COEFF, coefficient))
        if step % EVAL_EVERY == 0:
            metrics = base.evaluate(model, validation, device, BATCH_SIZE)
            row = dict(step=step, train_batch_mse=mse.item(),
                       train_batch_l0=batch_l0,
                       sparsity_coefficient=coefficient,
                       validation_mse=metrics['mse'],
                       validation_cosine=metrics['cosine'],
                       validation_explained_variance=metrics['explained_variance'],
                       validation_l0=metrics['l0'],
                       validation_l0_relative_error=abs(metrics['l0'] - TARGET_L0) / TARGET_L0)
            rows.append(row)
            with (OUTPUT / 'history.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"step={step:4d} train_batch_mse={row['train_batch_mse']:.5f} "
                  f"val_mse={row['validation_mse']:.5f} "
                  f"val_l0={row['validation_l0']:.2f} "
                  f"coeff={coefficient:.5g}", flush=True)
            if step in (1000, STEPS):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': settings, 'step': step,
                            'sparsity_coefficient': coefficient,
                            'ema_l0': ema_l0,
                            'validation_metrics': metrics},
                           OUTPUT / f'checkpoint_step_{step}.pt')
    save_curve(rows)
    print(f'Finished. Outputs saved in {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
