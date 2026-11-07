#!/usr/bin/env python3
"""Resume layer-6 SAE convergence checkpoints from step 5000 to 15000.

Run from repository root:
  python scripts/openwebtext/08_resume_convergence.py --architecture topk
  python scripts/openwebtext/08_resume_convergence.py --architecture relu
  python scripts/openwebtext/08_resume_convergence.py --architecture jumprelu

Outputs are separate; originals remain untouched. Historical RNG state was not
saved at step 5000, so these are continuation runs, not exact uninterrupted replays.
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
ROOT = Path('results/openwebtext/convergence/layer_6')
DATA = Path('data/activations/gpt2_small')
START, END, EVERY = 5000, 15000, 500

def load_baseline():
    spec = importlib.util.spec_from_file_location('baseline_sae', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--architecture', required=True, choices=['topk', 'relu', 'jumprelu'])
    args = p.parse_args()
    arch = args.architecture
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    original = ROOT / f'{arch}_l0_128'
    checkpoint_path = original / 'checkpoint_step_5000.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output = ROOT / f'{arch}_l0_128_resume_5000_to_15000'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite: {output}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    assert checkpoint['step'] == START, 'Unexpected starting step'
    assert cfg['architecture'] == arch and cfg['layer'] == 6 and cfg['target_l0'] == 128
    batch = int(cfg['batch_size'])
    features = int(cfg['features'])
    target = int(cfg['target_l0'])
    temp = float(cfg.get('temperature', 0.1))
    base = load_baseline()
    torch.set_num_threads(4)
    device = torch.device('cuda')
    train = base.load_split(DATA, 'train', 6)
    validation = base.load_split(DATA, 'validation', 6)
    model = base.SAE(train.shape[1], features, arch, target, temperature=temp).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Distinct deterministic RNG stream; original run's state was not saved.
    torch.manual_seed(int(cfg['seed']) + 5000)
    torch.cuda.manual_seed_all(int(cfg['seed']) + 5000)
    coefficient = checkpoint.get('sparsity_coefficient', None)
    ema_l0 = checkpoint.get('ema_l0', None)
    if arch != 'topk' and (coefficient is None or ema_l0 is None):
        raise ValueError('Missing adaptive sparsity controller state')
    output.mkdir(parents=True, exist_ok=False)
    run_cfg = dict(source_checkpoint=str(checkpoint_path), original_config=cfg,
                   start_step=START, end_step=END, eval_every=EVERY,
                   rng_note='RNG reset to original seed + 5000 because original state was not saved')
    (output / 'config.json').write_text(json.dumps(run_cfg, indent=2))
    rows = []
    print(f'{arch}: train={tuple(train.shape)}, validation={tuple(validation.shape)}', flush=True)
    for step in range(START + 1, END + 1):
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
            ema_l0 = 0.9 * ema_l0 + 0.1 * batch_l0
            if step % int(cfg.get('adjust_every', 25)) == 0:
                ratio = max(ema_l0, 1.0) / target
                rate = float(cfg.get('adjust_rate', 0.5))
                coefficient *= math.exp(rate * max(-2.0, min(2.0, math.log(ratio))))
                coefficient = max(float(cfg.get('min_sparsity_coeff', 1e-7)),
                                  min(float(cfg.get('max_sparsity_coeff', 100.0)), coefficient))
        if step % EVERY == 0:
            metrics = base.evaluate(model, validation, device, batch)
            row = dict(step=step, train_batch_mse=mse.item(), train_batch_l0=batch_l0,
                       validation_mse=metrics['mse'], validation_cosine=metrics['cosine'],
                       validation_explained_variance=metrics['explained_variance'],
                       validation_l0=metrics['l0'], sparsity_coefficient=coefficient if coefficient is not None else 0)
            rows.append(row)
            with (output / 'history.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"step={step:5d} val_mse={row['validation_mse']:.5f} "
                  f"val_l0={row['validation_l0']:.2f}", flush=True)
            if step % 2500 == 0 or step == END:
                state = dict(model_state_dict=model.state_dict(),
                             optimizer_state_dict=optimizer.state_dict(), config=run_cfg,
                             step=step, validation_metrics=metrics,
                             sparsity_coefficient=coefficient, ema_l0=ema_l0,
                             torch_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state_all())
                torch.save(state, output / f'checkpoint_step_{step}.pt')
    print(f'Finished: {output}', flush=True)

if __name__ == '__main__':
    main()
