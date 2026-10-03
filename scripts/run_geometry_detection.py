"""Stage 4A: Detect truth-associated SAE features and evaluate on test data."""

from pathlib import Path

import torch

from src.evaluation.geometry_sae_detection import (
    evaluate_selected_features_on_test,
    rank_features_on_train,
)


FEATURE_DIR = Path(
    "data/processed/geometry_of_truth/sae_features"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

TOP_N = 50


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Stage 4A — SAE Feature Detection")

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
        "Train SAE shape:",
        tuple(train_features.shape),
    )

    print(
        "Test SAE shape:",
        tuple(test_features.shape),
    )

    print("\nRanking features using TRAIN only...")

    selected_features = rank_features_on_train(
        train_features=train_features,
        train_labels=train_labels,
        top_n=TOP_N,
    )

    train_output = (
        RESULTS_DIR
        / "sae_top_features_train.csv"
    )

    selected_features.to_csv(
        train_output,
        index=False,
    )

    print(
        f"Saved training-selected features: "
        f"{train_output}"
    )

    print("\nEvaluating selected features on TEST...")

    test_results = evaluate_selected_features_on_test(
        selected_features=selected_features,
        test_features=test_features,
        test_labels=test_labels,
    )

    test_results = test_results.sort_values(
        "test_auroc",
        ascending=False,
    ).reset_index(drop=True)

    test_output = (
        RESULTS_DIR
        / "sae_feature_test_results.csv"
    )

    test_results.to_csv(
        test_output,
        index=False,
    )

    print(
        f"Saved held-out test results: "
        f"{test_output}"
    )

    print("\nTop 10 TRAIN-selected features on TEST:")
    print(
        test_results.head(10).to_string(
            index=False
        )
    )

    best = test_results.iloc[0]

    print("\nStage 4A complete.")
    print(
        f"Best held-out feature: "
        f"{int(best['feature_id'])}"
    )
    print(
        f"Test AUROC: "
        f"{best['test_auroc']:.4f}"
    )


if __name__ == "__main__":
    main()