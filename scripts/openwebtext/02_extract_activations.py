
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer


def extract_split(model, split, args, hooks):
    data_path = Path("data/openwebtext") / f"{split}.parquet"
    df = pd.read_parquet(data_path)

    output_dir = Path(args.output_dir) / "gpt2_small" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for start in tqdm(
        range(0, len(df), args.batch_size),
        desc=f"Extracting {split}",
    ):
        batch = df.iloc[start:start + args.batch_size]
        texts = batch["text"].tolist()

        # Tokenize each document separately.
        # Keep the beginning-of-sequence token.
        token_lists = [
            model.to_tokens(
                text,
                prepend_bos=True,
                truncate=True,
                move_to_device=False,
            )[0, :args.context_length]
            for text in texts
        ]

        # Pad within each batch.
        lengths = [len(tokens) for tokens in token_lists]
        max_length = max(lengths)

        tokens = torch.full(
            (len(token_lists), max_length),
            model.tokenizer.eos_token_id,
            dtype=torch.long,
        )

        for i, token_ids in enumerate(token_lists):
            tokens[i, :len(token_ids)] = token_ids

        tokens = tokens.to(args.device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hooks,
                return_type=None,
            )

        chunk_id = start // args.batch_size
        filename = f"chunk_{chunk_id:05d}.pt"
        output_path = output_dir / filename

        # Save on CPU to free GPU memory.
        payload = {
            "document_ids": batch["document_id"].tolist(),
            "tokens": tokens.cpu(),
            "lengths": lengths,
            "activations": {
                hook: cache[hook].detach().cpu().to(torch.float16)
                for hook in hooks
            },
        }

        torch.save(payload, output_path)

        manifest.append({
            "file": filename,
            "documents": len(batch),
            "tokens": sum(lengths),
        })

        del cache, payload, tokens

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "model": "gpt2-small",
            "split": split,
            "hooks": hooks,
            "context_length": args.context_length,
            "chunks": manifest,
        }, f, indent=2)

    print(f"Completed {split}: {len(df)} documents")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--output-dir", default="data/activations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print("Loading frozen GPT-2 Small...")

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=args.device,
    )
    model.eval()

    hooks = [
        f"blocks.{layer}.hook_resid_pre"
        for layer in range(model.cfg.n_layers)
    ]

    print(f"Extracting {len(hooks)} layers")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")

    for split in args.splits:
        extract_split(model, split, args, hooks)

    print("\nSTAGE 2 COMPLETE")


if __name__ == "__main__":
    main()
