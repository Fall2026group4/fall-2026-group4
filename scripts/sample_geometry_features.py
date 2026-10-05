"""Sample ~50 Layer-10 SAE features across activation-frequency buckets.

Uses TRAIN data only.

Outputs:
1. feature_interpretation_sample.csv
2. feature_top_activations.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEED = 42
N_FEATURES = 50
N_BUCKETS = 5
FEATURES_PER_BUCKET = N_FEATURES // N_BUCKETS
TOP_K_EXAMPLES = 5

FEATURE_PATH = Path(
    "data/processed/geometry_of_truth/sae_features/train_sae_features.pt"
)

TRAIN_DATA_PATH = Path(
    "data/processed/geometry_of_truth/train.csv"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

SAMPLE_OUTPUT = (
    RESULTS_DIR / "feature_interpretation_sample.csv"
)

TOP_ACTIVATIONS_OUTPUT = (
    RESULTS_DIR / "feature_top_activations.csv"
)


def main():
    rng = np.random.default_rng(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Phase 3 — Sampling Layer-10 SAE features"
    )

    # ---------------------------------------------------------
    # Load TRAIN SAE features
    # ---------------------------------------------------------

    cache = torch.load(
        FEATURE_PATH,
        map_location="cpu",
    )

    features = cache["features"].float()

    print(
        "Train feature matrix:",
        tuple(features.shape),
    )

    # ---------------------------------------------------------
    # Activation statistics
    # ---------------------------------------------------------

    active = features > 0

    activation_frequency = (
        active.float().mean(dim=0)
    )

    mean_activation = (
        features.mean(dim=0)
    )

    active_count = (
        active.sum(dim=0)
    )

    stats = pd.DataFrame(
        {
            "feature_id": np.arange(
                features.shape[1]
            ),
            "active_count": (
                active_count.numpy()
            ),
            "activation_frequency": (
                activation_frequency.numpy()
            ),
            "mean_activation": (
                mean_activation.numpy()
            ),
        }
    )

    # Only features that activate at least once.
    stats = stats.loc[
        stats["active_count"] > 0
    ].copy()

    stats = stats.sort_values(
        [
            "activation_frequency",
            "feature_id",
        ]
    ).reset_index(drop=True)

    print(
        "Non-zero features:",
        len(stats),
    )

    # ---------------------------------------------------------
    # Create five rank-based frequency buckets
    # ---------------------------------------------------------

    bucket_names = [
        "very_low",
        "low",
        "medium",
        "high",
        "very_high",
    ]

    bucket_indices = np.array_split(
        np.arange(len(stats)),
        N_BUCKETS,
    )

    stats["frequency_bucket"] = ""

    for name, indices in zip(
        bucket_names,
        bucket_indices,
    ):
        stats.loc[
            indices,
            "frequency_bucket",
        ] = name

    # ---------------------------------------------------------
    # Sample 10 features from each bucket
    # ---------------------------------------------------------

    sampled_parts = []

    for bucket in bucket_names:
        bucket_df = stats.loc[
            stats["frequency_bucket"]
            == bucket
        ].copy()

        if len(bucket_df) < FEATURES_PER_BUCKET:
            raise ValueError(
                f"Not enough features in bucket "
                f"{bucket}: {len(bucket_df)}"
            )

        selected_indices = rng.choice(
            bucket_df.index.to_numpy(),
            size=FEATURES_PER_BUCKET,
            replace=False,
        )

        sampled = bucket_df.loc[
            selected_indices
        ].copy()

        sampled_parts.append(
            sampled
        )

    sample = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    sample = sample.sort_values(
        [
            "frequency_bucket",
            "activation_frequency",
        ]
    ).reset_index(drop=True)

    sample.to_csv(
        SAMPLE_OUTPUT,
        index=False,
    )

    print(
        f"Sampled features: "
        f"{len(sample)}"
    )

    print(
        f"Saved: {SAMPLE_OUTPUT}"
    )

    # ---------------------------------------------------------
    # Top-activation check
    # ---------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    if len(train_df) != features.shape[0]:
        raise ValueError(
            "TRAIN rows do not match SAE feature rows."
        )

    top_rows = []

    for _, feature_row in sample.iterrows():
        feature_id = int(
            feature_row["feature_id"]
        )

        values = features[
            :,
            feature_id,
        ]

        top_k = min(
            TOP_K_EXAMPLES,
            values.shape[0],
        )

        top_values, top_indices = torch.topk(
            values,
            k=top_k,
        )

        for rank, (
            row_index,
            activation,
        ) in enumerate(
            zip(
                top_indices.tolist(),
                top_values.tolist(),
            ),
            start=1,
        ):
            metadata = train_df.iloc[
                row_index
            ]

            top_rows.append(
                {
                    "feature_id": feature_id,
                    "frequency_bucket": (
                        feature_row[
                            "frequency_bucket"
                        ]
                    ),
                    "activation_frequency": (
                        feature_row[
                            "activation_frequency"
                        ]
                    ),
                    "rank": rank,
                    "activation": activation,
                    "statement": metadata[
                        "statement"
                    ],
                    "label": metadata[
                        "label"
                    ],
                    "city": metadata[
                        "city"
                    ],
                    "country": metadata[
                        "country"
                    ],
                    "correct_country": metadata[
                        "correct_country"
                    ],
                }
            )

    top_activations = pd.DataFrame(
        top_rows
    )

    top_activations.to_csv(
        TOP_ACTIVATIONS_OUTPUT,
        index=False,
    )

    print(
        f"Top-activation rows: "
        f"{len(top_activations)}"
    )

    print(
        f"Saved: "
        f"{TOP_ACTIVATIONS_OUTPUT}"
    )

    print(
        "\nBucket counts:"
    )

    print(
        sample[
            "frequency_bucket"
        ].value_counts()
    )

    print(
        "\nPhase 3 sampling complete."
    )


if __name__ == "__main__":
    main()