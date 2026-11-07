#!/usr/bin/env python3
"""Exploratory sparsity-target SAE training on cached GPT-2 residual activations.

ReLU/JumpReLU target L0 via adaptive sparsity pressure; matching is NOT guaranteed.
JumpReLU uses a surrogate gradient, not the published JumpReLU estimator.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SAE(nn.Module):
    def __init__(self, d_in, d_sae, architecture, k, threshold=0.1, temperature=0.1):
        super().__init__()
        self.architecture = architecture
        self.k = k
        self.temperature = temperature
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.normal_(self.decoder.weight, std=1 / (d_in**0.5))
        self.normalize_decoder()
        if architecture == 'jumprelu':
            self.log_threshold = nn.Parameter(torch.full((d_sae,), math.log(threshold)))

    def forward(self, x):
        pre = self.encoder(x - self.output_bias)
        if self.architecture == 'relu':
            z = F.relu(pre)
        elif self.architecture == 'topk':
            positives = F.relu(pre)
            values, indices = positives.topk(self.k, dim=-1)
            z = torch.zeros_like(positives).scatter(-1, indices, values)
        else:
            threshold = self.log_threshold.exp()
            hard = (pre > threshold).to(pre.dtype)
            soft = torch.sigmoid((pre - threshold) / self.temperature)
            gate = hard.detach() - soft.detach() + soft
            z = F.relu(pre) * gate
        reconstructed = F.linear(z, self.decoder.weight) + self.output_bias
        return reconstructed, z, pre

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


def load_split(root, split, layer):
    paths = sorted((root / split).glob('chunk_*.pt'))
    if not paths:
        raise FileNotFoundError(f'No chunks in {root / split}')
    key = f'blocks.{layer}.hook_resid_pre'
    pieces = []
    for path in paths:
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        for i, length in enumerate(chunk['lengths']):
            if int(length) > 1:
                pieces.append(chunk['activations'][key][i, 1:int(length)].float())
    if not pieces:
        raise ValueError(f'No non-BOS tokens in {root / split}')
    data = torch.cat(pieces)
    if not torch.isfinite(data).all():
        raise ValueError(f'Non-finite activations in {split}')
    return data


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    mean = data.mean(dim=0).to(device)
    error = variance = cosine = l0 = 0.0
    for batch in data.split(batch_size):
        x = batch.to(device)
        rec, z, _ = model(x)
        error += (rec - x).square().sum().item()
        variance += (x - mean).square().sum().item()
        cosine += F.cosine_similarity(rec, x, dim=-1, eps=1e-8).sum().item()
        l0 += (z > 0).sum(dim=-1).sum().item()
    return {
        'mse': error / data.numel(),
        'cosine': cosine / len(data),
        'explained_variance': 1 - error / max(variance, 1e-12),
        'l0': l0 / len(data),
    }


def train_one(architecture, target, train, validation, args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SAE(train.shape[1], args.features, architecture, target, temperature=args.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    coefficient = args.initial_sparsity_coeff if architecture != 'topk' else 0.0
    ema_l0 = None
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(train), (args.batch_size,))
        x = train[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        rec, z, pre = model(x)
        mse = F.mse_loss(rec, x)
        if architecture == 'relu':
            # L1 penalizes positive feature magnitude; actual L0 is measured separately.
            penalty = z.sum(dim=-1).mean()
        elif architecture == 'jumprelu':
            # Differentiable estimate of active feature count; experimental surrogate.
            penalty = torch.sigmoid((pre - model.log_threshold.exp()) / args.temperature).sum(dim=-1).mean()
        else:
            penalty = mse.new_zeros(())
        loss = mse + coefficient * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            batch_l0 = (z > 0).sum(dim=-1).float().mean().item()
        ema_l0 = batch_l0 if ema_l0 is None else 0.9 * ema_l0 + 0.1 * batch_l0
        if architecture != 'topk' and step % args.adjust_every == 0:
            # Multiplicative controller: more active features -> stronger penalty.
            ratio = max(ema_l0, 1.0) / target
            coefficient *= math.exp(args.adjust_rate * max(-2.0, min(2.0, math.log(ratio))))
            coefficient = max(args.min_sparsity_coeff, min(args.max_sparsity_coeff, coefficient))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f'  step {step:4d}/{args.steps} mse={mse.item():.5f} '
                  f'L0={batch_l0:.1f} coeff={coefficient:.5g}', flush=True)
    metrics = evaluate(model, validation, device, args.batch_size)
    metrics.update({
        'architecture': architecture, 'target_l0': target, 'actual_l0': metrics['l0'],
        'l0_relative_error': abs(metrics['l0'] - target) / target,
        'l0_matched': abs(metrics['l0'] - target) / target <= args.match_tolerance,
        'sparsity_coefficient': coefficient, 'layer': args.layer,
        'steps': args.steps, 'features': args.features, 'seed': args.seed,
    })
    return model, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type=int, default=6, choices=range(12))
    p.add_argument('--targets', nargs='+', type=int, default=[32, 64, 128])
    p.add_argument('--architectures', nargs='+', choices=['relu', 'topk', 'jumprelu'],
                   default=['relu', 'topk', 'jumprelu'])
    p.add_argument('--data-root', type=Path, default=Path('data/activations/gpt2_small'))
    p.add_argument('--output-root', type=Path, default=Path('results/openwebtext/sae_sparsity_sweep'))
    p.add_argument('--features', type=int, default=3072)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--initial-sparsity-coeff', type=float, default=0.001)
    p.add_argument('--min-sparsity-coeff', type=float, default=1e-7)
    p.add_argument('--max-sparsity-coeff', type=float, default=100.0)
    p.add_argument('--adjust-every', type=int, default=25)
    p.add_argument('--adjust-rate', type=float, default=0.5)
    p.add_argument('--match-tolerance', type=float, default=0.2)
    p.add_argument('--log-every', type=int, default=200)
    args = p.parse_args()
    if not args.targets or any(t <= 0 or t > args.features for t in args.targets):
        p.error('Targets must be between 1 and --features')
    if not torch.cuda.is_available():
        p.error('CUDA GPU required')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    print(f'Loading cached layer {args.layer} activations...', flush=True)
    train = load_split(args.data_root, 'train', args.layer)
    validation = load_split(args.data_root, 'validation', args.layer)
    print(f'Train: {tuple(train.shape)} | Validation: {tuple(validation.shape)}', flush=True)
    output = args.output_root / f'layer_{args.layer}'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for target in args.targets:
        for architecture in args.architectures:
            print(f'\nTraining {architecture} with target L0={target}...', flush=True)
            model, metrics = train_one(architecture, target, train, validation, args, device)
            results.append(metrics)
            target_dir = output / f'l0_{target}'
            target_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.cpu().state_dict(),
                'config': {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                           'architecture': architecture, 'd_in': train.shape[1], 'target_l0': target},
                'validation_metrics': metrics,
            }, target_dir / f'{architecture}.pt')
            print(f'VALIDATION: {metrics}', flush=True)
            if not metrics['l0_matched']:
                print('WARNING: L0 target NOT reached; do not treat this as matched-sparsity evidence.', flush=True)
            del model
            torch.cuda.empty_cache()
            with (output / 'comparison.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    (output / 'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                  for k, v in vars(args).items()}, indent=2))
    print(f'\nSaved checkpoints and comparison CSV to {output}', flush=True)
    print('EXPLORATORY ONLY: matching may fail; JumpReLU uses an approximate surrogate gradient.', flush=True)


if __name__ == '__main__':
    main()
