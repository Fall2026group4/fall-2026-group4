
#!/usr/bin/env python3
"""Extract final-token GPT-2 Small residual-stream activations for Geometry of Truth.

Reads existing data/got/{cities,neg_cities,larger_than,smaller_than}.csv.
Does not train probes, split data, or train SAEs.

Run from repository root:
    python scripts/got/01_extract_got_activations.py
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer

DATASETS = ('cities', 'neg_cities', 'larger_than', 'smaller_than')


def extract_dataset(model, name, input_dir, output_dir, batch_size, device):
    source = input_dir / f'{name}.csv'
    if not source.is_file():
        raise FileNotFoundError(f'Missing dataset: {source}')
    df = pd.read_csv(source)
    if not {'statement', 'label'}.issubset(df.columns):
        raise ValueError(f'{source} must contain statement and label columns')
    if df['statement'].isna().any() or df['label'].isna().any():
        raise ValueError(f'{source} has missing statements or labels')
    if not df['label'].isin([0, 1]).all():
        raise ValueError(f'{source} has nonbinary labels')

    layer_names = [f'blocks.{layer}.hook_resid_pre' for layer in range(model.cfg.n_layers)]
    names = set(layer_names)
    gathered = {layer: [] for layer in range(model.cfg.n_layers)}
    lengths = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        # Tokenize independently with BOS, matching OpenWebText extraction.
        sequences = [model.to_tokens(str(s), prepend_bos=True, move_to_device=False)[0]
                     for s in batch['statement']]
        seq_lengths = torch.tensor([seq.numel() for seq in sequences], dtype=torch.long)
        max_length = int(seq_lengths.max())
        # Right padding is safe for a causal transformer at the selected non-padding positions.
        # Pad tokens cannot influence any earlier real token.
        padded = torch.full((len(sequences), max_length),
                            model.tokenizer.eos_token_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :seq.numel()] = seq
        padded = padded.to(device)
        final_indices = (seq_lengths - 1).to(device)
        row_indices = torch.arange(len(sequences), device=device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                padded, names_filter=lambda n: n in names,
                return_type=None,
            )
            for layer, hook in enumerate(layer_names):
                vectors = cache[hook][row_indices, final_indices, :]
                gathered[layer].append(vectors.detach().to('cpu', dtype=torch.float32))
        lengths.extend(seq_lengths.tolist())
        print(f'  {name}: {min(start + batch_size, len(df))}/{len(df)} statements', flush=True)

    activations = {layer: torch.cat(parts, dim=0) for layer, parts in gathered.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{name}_final_token.pt'
    torch.save({
        'dataset': name,
        'model': 'gpt2-small',
        'hook': 'blocks.{layer}.hook_resid_pre',
        'position': 'final_nonpadding_token',
        'prepend_bos': True,
        'row_ids': torch.arange(len(df), dtype=torch.long),
        'labels': torch.tensor(df['label'].to_numpy(), dtype=torch.long),
        'token_lengths': torch.tensor(lengths, dtype=torch.long),
        'activations': activations,
    }, output_file)
    print(f'Saved {output_file}; layer shapes: {activations[0].shape}; layers: {len(activations)}')
    # Original CSVs are left unchanged. row_ids map to their original zero-based row order.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('data/got'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/got/activations/gpt2_small'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    print(f'Loading frozen GPT-2 Small on {args.device}...')
    model = HookedTransformer.from_pretrained('gpt2-small', device=args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f'Extracting final-token residual activations from {model.cfg.n_layers} layers')
    for name in DATASETS:
        extract_dataset(model, name, args.input_dir, args.output_dir, args.batch_size, args.device)
    print('All four datasets extracted. No train/validation/test split or probes created.')


if __name__ == '__main__':
    main()
