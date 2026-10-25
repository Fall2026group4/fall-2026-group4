from pathlib import Path

import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geometry_of_truth"
    / "sae_features"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "geometry_of_truth"
)

SEED = 42
TOP_K_VALUES = [1, 5, 20]


def main():
    print("Geometry Top-K Feature Recovery")
    print("=" * 40)

    train_cache = torch.load(
        FEATURE_DIR / "train_sae_features.pt",
        map_location="cpu",
    )
    test_cache = torch.load(
        FEATURE_DIR / "test_sae_features.pt",
        map_location="cpu",
    )

    train_features = train_cache["features"].numpy()
    train_labels = train_cache["labels"].numpy()

    test_features = test_cache["features"].numpy()
    test_labels = test_cache["labels"].numpy()

    ranking = pd.read_csv(
        RESULTS_DIR / "sae_top_features_train.csv"
    )

    print(f"Train shape: {train_features.shape}")
    print(f"Test shape:  {test_features.shape}")
    print(f"Ranked features available: {len(ranking)}")

    rows = []

    for k in TOP_K_VALUES:
        feature_ids = (
            ranking
            .head(k)["feature_id"]
            .astype(int)
            .tolist()
        )

        x_train = train_features[:, feature_ids]
        x_test = test_features[:, feature_ids]

        probe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        probe.fit(
            x_train,
            train_labels,
        )

        predictions = probe.predict(x_test)
        probabilities = probe.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(
            test_labels,
            predictions,
        )

        auroc = roc_auc_score(
            test_labels,
            probabilities,
        )

        rows.append(
            {
                "top_k": k,
                "test_accuracy": accuracy,
                "test_auroc": auroc,
                "feature_ids": ",".join(
                    str(fid) for fid in feature_ids
                ),
            }
        )

        print(
            f"Top-{k}: "
            f"accuracy={accuracy:.4f}, "
            f"AUROC={auroc:.4f}"
        )

    results = pd.DataFrame(rows)

    output_path = (
        RESULTS_DIR
        / "geometry_topk_feature_recovery.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
