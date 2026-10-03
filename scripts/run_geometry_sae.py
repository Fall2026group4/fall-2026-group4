"""Stage 3: Encode Geometry of Truth activations with a pretrained SAE."""

from pathlib import Path

import pandas as pd
import torch

from src.sae.geometry_sae import (
    SAE_ID,
    SAE_LAYER,
    SAE_RELEASE,
    encode_residual_activations,
    load_geometry_sae,
    select_sae_layer,
)


ACTIVATION_DIR = Path(
    "data/processed/geometry_of_truth/activations"
)

OUTPUT_DIR = Path(
    "data/processed/geometry_of_truth/sae_features"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

BATCH_SIZE = 64


def process_split(
    sae,
    split_name,
):
    """Encode one cached Geometry split with the pretrained SAE."""

    cache_path = (
        ACTIVATION_DIR
        / f"{split_name}_activations.pt"
    )

    cache = torch.load(
        cache_path,
        map_location="cpu",
    )

    cached_activations = cache["activations"]
    labels = cache["labels"]

    residuals = select_sae_layer(
        cached_activations,
        layer=SAE_LAYER,
    )

    print(
        f"\n{split_name.upper()} residual shape:",
        tuple(residuals.shape),
    )

    features = encode_residual_activations(
        sae=sae,
        activations=residuals,
        batch_size=BATCH_SIZE,
    )

    print(
        f"{split_name.upper()} SAE feature shape:",
        tuple(features.shape),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}_sae_features.pt"
    )

    torch.save(
        {
            "features": features,
            "labels": labels,
            "layer": SAE_LAYER,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
        },
        output_path,
    )

    print(f"Saved: {output_path}")

    return {
        "split": split_name,
        "n_examples": features.shape[0],
        "n_features": features.shape[1],
        "mean_active_features": (
            (features > 0)
            .sum(dim=1)
            .float()
            .mean()
            .item()
        ),
        "mean_feature_activation": (
            features.mean().item()
        ),
    }


def main():
    print("Stage 3 — SAE Encoding")
    print(f"Release: {SAE_RELEASE}")
    print(f"SAE ID: {SAE_ID}")
    print(f"Layer: {SAE_LAYER}")

    print("\nLoading pretrained SAE...")

    sae = load_geometry_sae(
        device="cpu",
    )

    train_summary = process_split(
        sae,
        "train",
    )

    test_summary = process_split(
        sae,
        "test",
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        [
            train_summary,
            test_summary,
        ]
    )

    summary_path = (
        RESULTS_DIR
        / "sae_encoding_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\nStage 3 complete.")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()