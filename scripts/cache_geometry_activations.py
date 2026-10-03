"""Stage 1: Cache GPT-2 residual activations for Geometry of Truth."""

from pathlib import Path

import pandas as pd
import torch

from src.models.gpt2_activations import (
    extract_last_token_residuals,
    load_gpt2_small,
)


SEED = 42
DEVICE = "cpu"
BATCH_SIZE = 32

DATA_DIR = Path("data/processed/geometry_of_truth")
OUTPUT_DIR = DATA_DIR / "activations"


def cache_split(model, split_name):
    """Cache GPT-2 activations for one dataset split."""

    input_path = DATA_DIR / f"{split_name}.csv"

    df = pd.read_csv(input_path)

    texts = df["statement"].tolist()
    labels = torch.tensor(
        df["label"].values,
        dtype=torch.long,
    )

    print(f"\nProcessing {split_name} split...")
    print(f"Examples: {len(texts)}")

    activations = extract_last_token_residuals(
        model=model,
        texts=texts,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    assert activations.shape[0] == len(df)
    assert activations.shape[1] == 12
    assert activations.shape[2] == 768

    output_path = OUTPUT_DIR / f"{split_name}_activations.pt"

    torch.save(
        {
            "activations": activations,
            "labels": labels,
        },
        output_path,
    )

    print(f"{split_name} activation shape: {tuple(activations.shape)}")
    print(f"Saved: {output_path}")

    return output_path


def main():
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading GPT-2 Small...")
    model = load_gpt2_small(device=DEVICE)

    train_path = cache_split(
        model,
        "train",
    )

    test_path = cache_split(
        model,
        "test",
    )

    print("\nStage 1 complete.")
    print(f"Train cache: {train_path}")
    print(f"Test cache: {test_path}")


if __name__ == "__main__":
    main()