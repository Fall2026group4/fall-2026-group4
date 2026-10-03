"""Stage 2: Train and evaluate Geometry of Truth baseline probes."""

from pathlib import Path

import torch

from src.evaluation.geometry_probes import evaluate_linear_probes


SEED = 42

ACTIVATION_DIR = Path(
    "data/processed/geometry_of_truth/activations"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading cached GPT-2 activations...")

    train_cache = torch.load(
        ACTIVATION_DIR / "train_activations.pt",
        map_location="cpu",
    )

    test_cache = torch.load(
        ACTIVATION_DIR / "test_activations.pt",
        map_location="cpu",
    )

    train_activations = train_cache["activations"]
    train_labels = train_cache["labels"]

    test_activations = test_cache["activations"]
    test_labels = test_cache["labels"]

    print(
        "Train shape:",
        tuple(train_activations.shape),
    )

    print(
        "Test shape:",
        tuple(test_activations.shape),
    )

    print("\nTraining layer-wise baseline probes...\n")

    results_df, _ = evaluate_linear_probes(
        train_activations=train_activations,
        train_labels=train_labels,
        test_activations=test_activations,
        test_labels=test_labels,
        seed=SEED,
    )

    output_path = (
        RESULTS_DIR
        / "baseline_probe_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    best_row = results_df.loc[
        results_df["auroc"].idxmax()
    ]

    print("\nStage 2 complete.")
    print(f"Results saved: {output_path}")

    print(
        f"Best layer by AUROC: "
        f"{int(best_row['layer'])}"
    )

    print(
        f"Accuracy: "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"AUROC: "
        f"{best_row['auroc']:.4f}"
    )


if __name__ == "__main__":
    main()