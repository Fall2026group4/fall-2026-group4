#!/usr/bin/env python3
"""Evaluate layer 2, 6 and 10 SAEs on held-out OpenWebText activations.

Run from repository root:
    python scripts/openwebtext/10_reconstruction_evaluation.py

Reads nine existing 15000-step checkpoints. Does not train or modify models.
Writes separate combined outputs, preserving the earlier layer-6 evaluation.
"""
import csv
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

BASELINE = Path('scripts/openwebtext/03_training_sae.py')
DATA = Path('data/activations/gpt2_small')
CONVERGENCE = Path('results/openwebtext/convergence')
OUTPUT = Path('results/openwebtext/reconstruction_evaluation/layers_2_6_10')
LAYERS = (2, 6, 10)
ARCHITECTURES = ('relu', 'topk', 'jumprelu')
STEP = 15000
TARGET_L0 = 128
BATCH_SIZE = 256


def load_baseline():
    spec = importlib.util.spec_from_file_location('baseline_sae', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(layer, arch):
    if layer == 6:
        folder = f'{arch}_l0_128_resume_5000_to_15000'
    else:
        folder = f'{arch}_l0_128_warmstart_1000_to_15000'
    return CONVERGENCE / f'layer_{layer}' / folder / f'checkpoint_step_{STEP}.pt'


def iter_test_batches(layer):
    chunk_files = sorted((DATA / 'test').glob('chunk_*.pt'))
    if not chunk_files:
        raise FileNotFoundError('No test activation chunks found')
    hook = f'blocks.{layer}.hook_resid_pre'
    for path in chunk_files:
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        activations = chunk['activations'][hook].float()
        tokens = chunk['tokens']
        lengths = chunk['lengths']
        selected = []
        for index, length in enumerate(lengths):
            length = int(length)
            if int(tokens[index, 0]) != 50256:
                raise ValueError(f'Unexpected first token in {path}, document {index}')
            if length > 1:
                selected.append(activations[index, 1:length])
        if selected:
            x = torch.cat(selected, dim=0)
            for batch in x.split(BATCH_SIZE):
                yield batch


@torch.inference_mode()
def evaluate(model, layer, device):
    squared_error = 0.0
    sum_cosine = 0.0
    sum_l0 = 0.0
    total_tokens = 0
    total_elements = 0
    sum_x = None
    sum_x_squared = 0.0

    for batch in iter_test_batches(layer):
        x = batch.to(device)
        reconstructed, z, _ = model(x)
        squared_error += (x - reconstructed).square().sum().item()
        sum_cosine += F.cosine_similarity(x, reconstructed, dim=-1).sum().item()
        sum_l0 += (z != 0).sum().item()
        total_tokens += x.shape[0]
        total_elements += x.numel()
        batch_sum = x.double().sum(dim=0).cpu()
        sum_x = batch_sum if sum_x is None else sum_x + batch_sum
        sum_x_squared += x.double().square().sum().item()

    if total_tokens == 0:
        raise RuntimeError(f'No test tokens for layer {layer}')
    total_variance = sum_x_squared - sum_x.square().sum().item() / total_tokens
    return {
        'test_tokens': total_tokens,
        'mse': squared_error / total_elements,
        'mean_cosine_similarity': sum_cosine / total_tokens,
        'explained_variance': 1 - squared_error / total_variance if total_variance > 0 else float('nan'),
        'actual_l0': sum_l0 / total_tokens,
    }


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = load_baseline()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Device:', device, flush=True)
    rows = []

    for layer in LAYERS:
        for arch in ARCHITECTURES:
            path = checkpoint_path(layer, arch)
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            if int(ckpt['step']) != STEP:
                raise ValueError(f'Unexpected training step in {path}: {ckpt["step"]}')
            state = ckpt['model_state_dict']
            d_sae, d_in = state['encoder.weight'].shape
            if d_in != 768 or d_sae != 3072:
                raise ValueError(f'Unexpected SAE dimensions in {path}: {d_in}, {d_sae}')
            # Use the exact SAE class used during training, not a reimplementation.
            model = baseline.SAE(d_in, d_sae, arch, TARGET_L0, temperature=0.1).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            print(f'\nEvaluating layer={layer} architecture={arch}...', flush=True)
            metrics = evaluate(model, layer, device)
            row = {
                'layer': layer,
                'architecture': arch,
                'training_step': STEP,
                'training_procedure': 'optimizer_resumed' if layer == 6 else 'warmstart_optimizer_reset',
                **metrics,
            }
            rows.append(row)
            print(f"  tokens={metrics['test_tokens']} mse={metrics['mse']:.6f} "
                  f"cosine={metrics['mean_cosine_similarity']:.6f} "
                  f"explained_variance={metrics['explained_variance']:.6f} "
                  f"l0={metrics['actual_l0']:.2f}", flush=True)
            del model

    csv_path = OUTPUT / 'test_reconstruction_metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUTPUT / 'test_reconstruction_metrics.json'
    json_path.write_text(json.dumps(rows, indent=2))
    print('\nSaved:', csv_path, '\nSaved:', json_path, flush=True)
    print('Note: compare architecture MSE within a layer; activation scales vary across layers.')


if __name__ == '__main__':
    main()
