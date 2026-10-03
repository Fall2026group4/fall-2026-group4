"""Stage 4B: Measure SAE reconstruction faithfulness for Geometry of Truth."""

from pathlib import Path

import pandas as pd
import torch

from src.evaluation.geometry_sae_faithfulness import (
    evaluate_reconstructed_probe,
    reconstruct_from_sae_features,
)
from src.sae.geometry_sae import load_geometry_sae


FEATURE_DIR = Path(
    "data/processed/geometry_of_truth/sae_features"
)

BASELINE_RESULTS = Path(
    "results/geometry_of_truth/baseline_probe_results.csv"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

TARGET_LAYER = 10
BATCH_SIZE = 64
SEED = 42


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Stage 4B — SAE Reconstruction Faithfulness")

    train_cache = torch.load(
        FEATURE_DIR / "train_sae_features.pt",
        map_location="cpu",
    )

    test_cache = torch.load(
        FEATURE_DIR / "test_sae_features.pt",
        map_location="cpu",
    )

    train_features = train_cache["features"]
    train_labels = train_cache["labels"]

    test_features = test_cache["features"]
    test_labels = test_cache["labels"]

    print(
        "Train SAE feature shape:",
        tuple(train_features.shape),
    )

    print(
        "Test SAE feature shape:",
        tuple(test_features.shape),
    )

    print("\nLoading pretrained SAE...")
    sae = load_geometry_sae(device="cpu")

    print("\nDecoding train features...")
    train_reconstructed = reconstruct_from_sae_features(
        sae=sae,
        features=train_features,
        batch_size=BATCH_SIZE,
    )

    print(
        "Train reconstructed shape:",
        tuple(train_reconstructed.shape),
    )

    print("\nDecoding test features...")
    test_reconstructed = reconstruct_from_sae_features(
        sae=sae,
        features=test_features,
        batch_size=BATCH_SIZE,
    )

    print(
        "Test reconstructed shape:",
        tuple(test_reconstructed.shape),
    )

    print("\nEvaluating reconstructed residual probe...")

    reconstructed_metrics = evaluate_reconstructed_probe(
        train_reconstructed=train_reconstructed,
        train_labels=train_labels,
        test_reconstructed=test_reconstructed,
        test_labels=test_labels,
        seed=SEED,
    )

    baseline_df = pd.read_csv(
        BASELINE_RESULTS
    )

    baseline_row = baseline_df.loc[
        baseline_df["layer"] == TARGET_LAYER
    ].iloc[0]

    raw_accuracy = float(
        baseline_row["accuracy"]
    )

    raw_auroc = float(
        baseline_row["auroc"]
    )

    reconstructed_accuracy = (
        reconstructed_metrics["accuracy"]
    )

    reconstructed_auroc = (
        reconstructed_metrics["auroc"]
    )

    accuracy_gap = (
        raw_accuracy
        - reconstructed_accuracy
    )

    auroc_gap = (
        raw_auroc
        - reconstructed_auroc
    )

    results = pd.DataFrame(
        [
            {
                "layer": TARGET_LAYER,
                "raw_accuracy": raw_accuracy,
                "reconstructed_accuracy": reconstructed_accuracy,
                "accuracy_gap": accuracy_gap,
                "raw_auroc": raw_auroc,
                "reconstructed_auroc": reconstructed_auroc,
                "auroc_gap": auroc_gap,
            }
        ]
    )

    output_path = (
        RESULTS_DIR
        / "sae_faithfulness_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nStage 4B complete.")
    print(f"Results saved: {output_path}")

    print(
        f"Raw Layer-{TARGET_LAYER} Accuracy: "
        f"{raw_accuracy:.4f}"
    )

    print(
        f"Reconstructed Accuracy: "
        f"{reconstructed_accuracy:.4f}"
    )

    print(
        f"Accuracy Gap: "
        f"{accuracy_gap:.4f}"
    )

    print(
        f"Raw Layer-{TARGET_LAYER} AUROC: "
        f"{raw_auroc:.4f}"
    )

    print(
        f"Reconstructed AUROC: "
        f"{reconstructed_auroc:.4f}"
    )

    print(
        f"AUROC Faithfulness Gap: "
        f"{auroc_gap:.4f}"
    )


if __name__ == "__main__":
    main()