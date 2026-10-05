"""Build one-row-per-feature interpretation table for Phase 3."""

from pathlib import Path

import pandas as pd


LABELS_PATH = Path(
    "results/geometry_of_truth/feature_interpretation_with_labels.csv"
)

TOP_ACTIVATIONS_PATH = Path(
    "results/geometry_of_truth/feature_top_activations.csv"
)

OUTPUT_PATH = Path(
    "results/geometry_of_truth/feature_interpretation_review_table.csv"
)


def main():
    labels = pd.read_csv(
        LABELS_PATH
    )

    activations = pd.read_csv(
        TOP_ACTIVATIONS_PATH
    )

    print(
        f"Label rows: {len(labels)}"
    )

    print(
        f"Top-activation rows: {len(activations)}"
    )

    # ---------------------------------------------------------
    # Pivot top 5 activating statements into one row per feature
    # ---------------------------------------------------------

    statement_pivot = (
        activations.pivot(
            index="feature_id",
            columns="rank",
            values="statement",
        )
        .rename(
            columns={
                1: "top_1_statement",
                2: "top_2_statement",
                3: "top_3_statement",
                4: "top_4_statement",
                5: "top_5_statement",
            }
        )
        .reset_index()
    )

    activation_pivot = (
        activations.pivot(
            index="feature_id",
            columns="rank",
            values="activation",
        )
        .rename(
            columns={
                1: "top_1_activation",
                2: "top_2_activation",
                3: "top_3_activation",
                4: "top_4_activation",
                5: "top_5_activation",
            }
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Merge labels + top activations
    # ---------------------------------------------------------

    review = labels.merge(
        statement_pivot,
        on="feature_id",
        how="left",
    )

    review = review.merge(
        activation_pivot,
        on="feature_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Add human-review columns
    # ---------------------------------------------------------

    review["top_activation_check"] = ""
    review["review_notes"] = ""

    preferred_columns = [
        "feature_id",
        "frequency_bucket",
        "active_count",
        "activation_frequency",
        "mean_activation",
        "neuronpedia_label",
        "explanation_model",
        "n_explanations",
        "top_1_statement",
        "top_1_activation",
        "top_2_statement",
        "top_2_activation",
        "top_3_statement",
        "top_3_activation",
        "top_4_statement",
        "top_4_activation",
        "top_5_statement",
        "top_5_activation",
        "top_activation_check",
        "review_notes",
        "neuronpedia_url",
    ]

    review = review[
        preferred_columns
    ]

    review = review.sort_values(
        [
            "frequency_bucket",
            "activation_frequency",
            "feature_id",
        ]
    ).reset_index(drop=True)

    review.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Features in review table: "
        f"{len(review)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nPhase 3 interpretation table complete."
    )


if __name__ == "__main__":
    main()