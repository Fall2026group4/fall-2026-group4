#!/usr/bin/env python3
"""P1 pilot (PREPARED 2026-11-07, NOT YET RUN): improved sparsity control + validation checkpoint selection.

Scope (approved design, training awaits explicit approval):
    layers 0, 2, 6, 10 x relu / topk / jumprelu x seed 42, target L0 128, 15 000 steps  ->  12 runs.

Everything not listed below is identical to the uninterrupted baseline (scripts/openwebtext/13 -> 03 train_one):
same SAE class, init, Adam lr 3e-4, batch 256, grad clip 10, decoder renorm, same seed and the same batch-index RNG
stream (torch.manual_seed(seed) then randint each step), so baseline seed 42 is the paired comparison.

Changes (fixed before any P1 run):
 1. Controller annealing on a fixed train subset: steps 1-10 000 use the original EMA controller. From 10 000 to
    12 000, every 250 steps, L0 is measured on a fixed 12 800-token TRAIN subset (no validation data) and the
    coefficient is corrected multiplicatively: c <- c * (L0 / target) ** CORRECTION_GAIN.
 2. Controller freeze: the sparsity coefficient is frozen from step 12 000 to 15 000.
 3. Scale-aware JumpReLU threshold initialisation: each latent's initial threshold is set to the
    (1 - target / d_sae) quantile of its pre-activations on 4 096 train tokens (initial expected L0 ~= target),
    instead of the fixed 0.1. ReLU and TopK initialisation unchanged.
 4. Validation checkpoint selection: OWT VALIDATION is evaluated at steps 10k, 11k, ..., 15k (EV, MSE, cosine, L0,
    dead-latent fraction). Selected checkpoint = highest validation EV among checkpoints with |L0 - 128| / 128 <= 0.05;
    if none qualifies, the checkpoint with the smallest L0 error. Both the selected and the final checkpoint are saved.
 No test data is read. Downstream P1 evaluation (GoT/RepE frozen probes, spliced effects) is validation-only.

Outputs
  metrics     results/openwebtext/p1_sparsity_control_pilot/{runs.csv, validation_curves.csv, run_config.json}
  checkpoints results/openwebtext/convergence/layer_{L}/{arch}_l0_128_p1_seed{S}/checkpoint_{selected,step_15000}.pt
The script refuses to overwrite any checkpoint or results file.

Run from repository root (after approval):
    python scripts/openwebtext/16_p1_sparsity_control_pilot.py --dry-run   # list runs / check paths, no training
    python scripts/openwebtext/16_p1_sparsity_control_pilot.py
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data/activations/gpt2_small'
OUT = REPO / 'results/openwebtext/p1_sparsity_control_pilot'
CKPT_ROOT = REPO / 'results/openwebtext/convergence'
LAYERS, ARCHS, SEEDS = (0, 2, 6, 10), ('relu', 'topk', 'jumprelu'), (42,)
TARGET, STEPS, FEATURES = 128, 15000, 3072
ANNEAL_START, FREEZE_AT, ANNEAL_EVERY, CORRECTION_GAIN = 10000, 12000, 250, 1.0
EVAL_STEPS = tuple(range(10000, STEPS + 1, 1000))
SELECT_TOL = 0.05
CONTROL_TOKENS, INIT_TOKENS = 12800, 4096
# Baseline hyper-parameters (scripts/openwebtext/13_consistent_protocol_retrain.py)
BASE = dict(features=FEATURES, batch_size=256, steps=STEPS, lr=3e-4, temperature=0.1, initial_sparsity_coeff=0.001,
            min_sparsity_coeff=1e-7, max_sparsity_coeff=100.0, adjust_every=25, adjust_rate=0.5, log_every=2500)


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def scale_aware_threshold_init(model, x_init, target):
    """Set each JumpReLU threshold to the (1 - target/d_sae) quantile of that latent's initial pre-activations."""
    pre = model.encoder(x_init - model.output_bias)
    q = 1.0 - target / pre.shape[1]
    thr = torch.quantile(pre.float(), q, dim=0).clamp_min(1e-3)
    model.log_threshold.copy_(thr.log())
    return float((pre > thr).sum(-1).float().mean())


@torch.no_grad()
def measure(model, x, device, batch=4096):
    """Validation metrics with the shared metric definitions + dead-latent fraction."""
    import sys
    sys.path.insert(0, str(REPO))
    from src.evaluation import faithfulness_metrics as fm
    model.eval()
    x_hat, l0, active = fm.reconstruct(model, x, device, batch)
    m = fm.reconstruction_metrics(x, x_hat, l0)
    m['dead_fraction'] = fm.dead_fraction(active)
    model.train()
    return m


def select(curve):
    ok = [r for r in curve if abs(r['l0'] - TARGET) / TARGET <= SELECT_TOL]
    if ok:
        return max(ok, key=lambda r: r['explained_variance']), 'max val EV within 5% L0'
    return min(curve, key=lambda r: abs(r['l0'] - TARGET)), 'no checkpoint within 5% L0: closest L0'


def train_p1(sae_cls, arch, train, val, seed, device):
    """Baseline loop (03 train_one) + changes 1-4. Returns selected state, final state, curve, info."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    a = argparse.Namespace(**BASE)
    model = sae_cls(train.shape[1], a.features, arch, TARGET, temperature=a.temperature).to(device)
    info = {}
    if arch == 'jumprelu':
        # uses a separate generator so the training batch stream stays identical to the baseline
        g = torch.Generator().manual_seed(seed + 7)
        idx = torch.randint(len(train), (INIT_TOKENS,), generator=g)
        info['init_expected_l0'] = scale_aware_threshold_init(model, train[idx].to(device), TARGET)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    coefficient = a.initial_sparsity_coeff if arch != 'topk' else 0.0
    g = torch.Generator().manual_seed(seed + 11)
    control_x = train[torch.randint(len(train), (CONTROL_TOKENS,), generator=g)].to(device)
    ema_l0, curve, states = None, [], {}
    for step in range(1, STEPS + 1):
        ids = torch.randint(len(train), (a.batch_size,))
        x = train[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        rec, z, pre = model(x)
        mse = F.mse_loss(rec, x)
        if arch == 'relu':
            penalty = z.sum(dim=-1).mean()
        elif arch == 'jumprelu':
            penalty = torch.sigmoid((pre - model.log_threshold.exp()) / a.temperature).sum(dim=-1).mean()
        else:
            penalty = mse.new_zeros(())
        (mse + coefficient * penalty).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            batch_l0 = (z > 0).sum(dim=-1).float().mean().item()
        ema_l0 = batch_l0 if ema_l0 is None else 0.9 * ema_l0 + 0.1 * batch_l0
        if arch != 'topk':
            if step < ANNEAL_START and step % a.adjust_every == 0:          # original controller
                ratio = max(ema_l0, 1.0) / TARGET
                coefficient *= math.exp(a.adjust_rate * max(-2.0, min(2.0, math.log(ratio))))
            elif ANNEAL_START <= step < FREEZE_AT and step % ANNEAL_EVERY == 0:   # change 1
                with torch.no_grad():
                    l0_ctrl = float((model(control_x)[1] > 0).sum(-1).float().mean())
                coefficient *= (max(l0_ctrl, 1.0) / TARGET) ** CORRECTION_GAIN
            # step >= FREEZE_AT: frozen (change 2)
            coefficient = max(a.min_sparsity_coeff, min(a.max_sparsity_coeff, coefficient))
        if step in EVAL_STEPS:                                              # change 4
            m = measure(model, val, device)
            m.update(step=step, sparsity_coefficient=coefficient)
            curve.append(m)
            states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step == 1 or step % a.log_every == 0:
            print(f'  step {step:5d} mse={mse.item():.5f} L0={batch_l0:.1f} coeff={coefficient:.5g}', flush=True)
    chosen, rule = select(curve)
    info.update(selection_rule=rule, selected_step=chosen['step'])
    return states[chosen['step']], states[STEPS], curve, info, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    plan = [(l, ar, s) for l in LAYERS for ar in ARCHS for s in SEEDS]
    paths = {k: CKPT_ROOT / f'layer_{k[0]}' / f'{k[1]}_l0_{TARGET}_p1_seed{k[2]}' for k in plan}
    clash = [str(p) for p in paths.values() if p.exists()] + [str(OUT / 'runs.csv')] * (OUT / 'runs.csv').exists()
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    if args.dry_run:
        for k, p in paths.items():
            print(k, '->', p.relative_to(REPO))
        print(f'{len(plan)} runs planned; nothing trained.')
        return
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required.')
    device = torch.device('cuda')
    base = load_module('scripts/openwebtext/03_training_sae.py', 'owt03')
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for layer in LAYERS:
        train = base.load_split(DATA, 'train', layer)
        val = base.load_split(DATA, 'validation', layer)
        for arch in ARCHS:
            for seed in SEEDS:
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                sel, fin, curve, info, a = train_p1(base.SAE, arch, train, val, seed, device)
                secs = time.time() - t0
                d = paths[(layer, arch, seed)]
                d.mkdir(parents=True, exist_ok=False)
                cfg = {**vars(a), 'layer': layer, 'seed': seed, 'architecture': arch, 'target_l0': TARGET, 'd_in': 768,
                       'procedure': 'p1_anneal_freeze_valselect', 'anneal_start': ANNEAL_START, 'freeze_at': FREEZE_AT,
                       'anneal_every': ANNEAL_EVERY, 'correction_gain': CORRECTION_GAIN, 'select_tol': SELECT_TOL,
                       'jumprelu_threshold_init': 'per-latent (1 - k/d_sae) quantile' if arch == 'jumprelu' else 'n/a'}
                chosen = next(r for r in curve if r['step'] == info['selected_step'])
                final = curve[-1]
                torch.save({'model_state_dict': sel, 'config': cfg, 'validation_metrics': chosen, 'selection': info},
                           d / 'checkpoint_selected.pt')
                torch.save({'model_state_dict': fin, 'config': cfg, 'validation_metrics': final}, d / f'checkpoint_step_{STEPS}.pt')
                for r in curve:
                    curves.append({'layer': layer, 'architecture': arch, 'seed': seed, **r})
                rows.append({'layer': layer, 'architecture': arch, 'seed': seed, 'train_seconds': round(secs, 1),
                             'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                             **info, **{f'selected_val_{k}': v for k, v in chosen.items()},
                             **{f'final_val_{k}': v for k, v in final.items()},
                             'checkpoint_dir': d.relative_to(REPO).as_posix()})
                pd.DataFrame(rows).to_csv(OUT / 'runs.csv', index=False)
                pd.DataFrame(curves).to_csv(OUT / 'validation_curves.csv', index=False)
                print(f'DONE layer={layer} {arch} seed={seed} {secs:.0f}s selected step {info["selected_step"]} '
                      f'val EV={chosen["explained_variance"]:.4f} L0={chosen["l0"]:.1f} dead={chosen["dead_fraction"]:.3f}', flush=True)
        del train, val
    (OUT / 'run_config.json').write_text(json.dumps({
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'layers': LAYERS, 'architectures': ARCHS,
        'seeds': SEEDS, 'target_l0': TARGET, 'steps': STEPS, 'eval_steps': EVAL_STEPS, 'test_data_used': False,
        'baseline_for_comparison': 'results/openwebtext/convergence/layer_{L}/{arch}_l0_128_uninterrupted_seed42',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, indent=2))


if __name__ == '__main__':
    main()
