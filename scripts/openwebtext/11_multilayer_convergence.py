#!/usr/bin/env python3
"""Warm-start layer 2 and 10 SAEs from 1000-step sweep weights to 15000 steps.

Run from repository root:
    python scripts/openwebtext/11_multilayer_convergence.py

Optional: --layers 2 --architectures relu; --reset-run is deliberately unsupported.
Each run saves its own checkpoints and can resume after interruption.
Original sweep and layer-6 results are never modified.

Caveat: sweep checkpoints lack optimizer, RNG and sparsity controller state.
This is a warm start with reset optimizer/controller, not an exact uninterrupted run.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

BASELINE = Path('scripts/openwebtext/03_training_sae.py')
SWEEP = Path('results/openwebtext/sae_sparsity_sweep')
OUT = Path('results/openwebtext/convergence')
DATA = Path('data/activations/gpt2_small')
START = 1000
END = 15000
EVAL_EVERY = 500
SAVE_EVERY = 2500


def load_baseline():
    spec = importlib.util.spec_from_file_location('baseline_sae', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_history(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def train_one(base, layer, arch, device):
    source = SWEEP / f'layer_{layer}' / 'l0_128' / f'{arch}.pt'
    if not source.exists():
        raise FileNotFoundError(source)
    output = OUT / f'layer_{layer}' / f'{arch}_l0_128_warmstart_1000_to_15000'
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / f'checkpoint_step_{END}.pt'
    if final_path.exists():
        print(f'SKIP completed: layer={layer} arch={arch}', flush=True)
        return

    initial = torch.load(source, map_location='cpu', weights_only=False)
    cfg = initial['config']
    if (cfg['architecture'] != arch or int(cfg['layer']) != layer
            or int(cfg['target_l0']) != 128 or int(cfg['steps']) != START):
        raise ValueError(f'Unexpected sweep configuration: {source}')

    batch = int(cfg['batch_size'])
    features = int(cfg['features'])
    target = int(cfg['target_l0'])
    temp = float(cfg.get('temperature', 0.1))
    train = base.load_split(DATA, 'train', layer)
    validation = base.load_split(DATA, 'validation', layer)
    model = base.SAE(train.shape[1], features, arch, target, temperature=temp).to(device)
    model.load_state_dict(initial['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    coefficient = None if arch == 'topk' else float(cfg['initial_sparsity_coeff'])
    ema_l0 = None
    step_start = START
    history_path = output / 'history.csv'
    history = []

    run_cfg = {
        'source_checkpoint': str(source),
        'original_config': cfg,
        'start_step': START,
        'end_step': END,
        'evaluation_interval': EVAL_EVERY,
        'training_method': 'warm start from sweep weights; Adam/controller reset at step 1000',
    }
    config_path = output / 'config.json'
    if config_path.exists():
        old_cfg = json.loads(config_path.read_text())
        if old_cfg != run_cfg:
            raise RuntimeError(f'Configuration mismatch in {output}')
    else:
        config_path.write_text(json.dumps(run_cfg, indent=2))

    resume_paths = sorted(output.glob('checkpoint_step_*.pt'),
                          key=lambda p: int(p.stem.split('_')[-1]))
    if resume_paths:
        checkpoint = torch.load(resume_paths[-1], map_location='cpu', weights_only=False)
        step_start = int(checkpoint['step'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        coefficient = checkpoint['sparsity_coefficient']
        ema_l0 = checkpoint['ema_l0']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        history = [r for r in read_history(history_path) if int(r['step']) <= step_start]
        save_history(history_path, history)
        print(f'RESUME layer={layer} arch={arch} from step={step_start}', flush=True)
    else:
        # Independent deterministic stream per layer and architecture.
        arch_index = {'relu': 0, 'topk': 1, 'jumprelu': 2}[arch]
        seed = int(cfg['seed']) + 10000 + layer * 10 + arch_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f'WARM START layer={layer} arch={arch} from step=1000', flush=True)

    print(f'  train={tuple(train.shape)} validation={tuple(validation.shape)}', flush=True)
    for step in range(step_start + 1, END + 1):
        ids = torch.randint(len(train), (batch,))
        x = train[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        rec, z, pre = model(x)
        mse = F.mse_loss(rec, x)
        if arch == 'relu':
            penalty = z.sum(dim=-1).mean()
        elif arch == 'jumprelu':
            penalty = torch.sigmoid((pre - model.log_threshold.exp()) / temp).sum(dim=-1).mean()
        else:
            penalty = None
        loss = mse if penalty is None else mse + coefficient * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            batch_l0 = (z > 0).sum(dim=-1).float().mean().item()
        if arch != 'topk':
            ema_l0 = batch_l0 if ema_l0 is None else 0.9 * ema_l0 + 0.1 * batch_l0
            if step % int(cfg.get('adjust_every', 25)) == 0:
                ratio = max(ema_l0, 1.0) / target
                rate = float(cfg.get('adjust_rate', 0.5))
                coefficient *= math.exp(rate * max(-2.0, min(2.0, math.log(ratio))))
                coefficient = max(float(cfg.get('min_sparsity_coeff', 1e-7)),
                                  min(float(cfg.get('max_sparsity_coeff', 100.0)), coefficient))
        if step % EVAL_EVERY == 0:
            metrics = base.evaluate(model, validation, device, batch)
            row = dict(step=step, train_batch_mse=mse.item(), train_batch_l0=batch_l0,
                       validation_mse=metrics['mse'], validation_cosine=metrics['cosine'],
                       validation_explained_variance=metrics['explained_variance'],
                       validation_l0=metrics['l0'],
                       sparsity_coefficient=coefficient if coefficient is not None else 0)
            history.append(row)
            save_history(history_path, history)
            print(f"  step={step:5d} val_mse={row['validation_mse']:.5f} "
                  f"val_l0={row['validation_l0']:.2f}", flush=True)
            if step % SAVE_EVERY == 0 or step == END:
                state = dict(model_state_dict=model.state_dict(),
                             optimizer_state_dict=optimizer.state_dict(),
                             config=run_cfg, step=step, validation_metrics=metrics,
                             sparsity_coefficient=coefficient, ema_l0=ema_l0,
                             torch_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state_all())
                torch.save(state, output / f'checkpoint_step_{step}.pt')
    print(f'FINISHED layer={layer} arch={arch}: {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', nargs='+', type=int, choices=[2, 10], default=[2, 10])
    p.add_argument('--architectures', nargs='+',
                   choices=['relu', 'topk', 'jumprelu'],
                   default=['relu', 'topk', 'jumprelu'])
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4)
    base = load_baseline()
    device = torch.device('cuda')
    for layer in args.layers:
        for arch in args.architectures:
            train_one(base, layer, arch, device)
    print('All requested runs complete.', flush=True)


if __name__ == '__main__':
    main()
