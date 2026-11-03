import argparse
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/repe/persona_pairs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/repe/activations/gpt2_small"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    df = pd.read_csv(args.input)

    required = {
        "statement_id", "split", "persona",
        "persona_label", "factual_label", "prompt"
    }
    assert required.issubset(df.columns)
    assert df["prompt"].is_unique
    assert not df[list(required)].isna().any().any()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading frozen GPT-2 Small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layer_names = [
        f"blocks.{layer}.hook_resid_pre"
        for layer in range(model.cfg.n_layers)
    ]
    names = set(layer_names)

    gathered = {layer: [] for layer in range(model.cfg.n_layers)}
    lengths = []

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]

        sequences = [
            model.to_tokens(
                str(prompt),
                prepend_bos=True,
                move_to_device=False,
            )[0]
            for prompt in batch["prompt"]
        ]

        seq_lengths = torch.tensor(
            [seq.numel() for seq in sequences],
            dtype=torch.long,
        )

        max_length = int(seq_lengths.max())
        padded = torch.full(
            (len(sequences), max_length),
            model.tokenizer.eos_token_id,
            dtype=torch.long,
        )

        for i, seq in enumerate(sequences):
            padded[i, :seq.numel()] = seq

        padded = padded.to(device)
        final_indices = (seq_lengths - 1).to(device)
        row_indices = torch.arange(len(sequences), device=device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                padded,
                names_filter=lambda n: n in names,
                return_type=None,
            )

            for layer, hook in enumerate(layer_names):
                vectors = cache[hook][row_indices, final_indices, :]
                gathered[layer].append(
                    vectors.detach().to("cpu", dtype=torch.float32)
                )

        lengths.extend(seq_lengths.tolist())

        print(
            f"Extracted {min(start + args.batch_size, len(df))}/{len(df)} prompts",
            flush=True,
        )

    activations = {
        layer: torch.cat(parts, dim=0)
        for layer, parts in gathered.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "repe_final_token.pt"

    torch.save(
        {
            "model": "gpt2-small",
            "hook": "blocks.{layer}.hook_resid_pre",
            "position": "final_nonpadding_token",
            "prepend_bos": True,
            "row_ids": torch.arange(len(df), dtype=torch.long),
            "statement_ids": df["statement_id"].tolist(),
            "splits": df["split"].tolist(),
            "personas": df["persona"].tolist(),
            "persona_labels": torch.tensor(
                df["persona_label"].to_numpy(), dtype=torch.long
            ),
            "factual_labels": torch.tensor(
                df["factual_label"].to_numpy(), dtype=torch.long
            ),
            "token_lengths": torch.tensor(lengths, dtype=torch.long),
            "activations": activations,
        },
        output_file,
    )

    print(f"Saved: {output_file}")
    print(f"Layer 6 shape: {activations[6].shape}")
    print(f"Total layers: {len(activations)}")
    print("=== REPE STAGE 2 COMPLETE ===")


if __name__ == "__main__":
    main()
