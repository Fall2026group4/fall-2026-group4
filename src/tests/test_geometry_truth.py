"""Tests for src/evaluation/geometry_truth.py.

Pure pandas/torch-free logic (dataset split, feature statistics, candidate
ranking), so these run without downloading GPT-2 or the SAE weights - same
philosophy as test_detection.py and test_steering.py.
"""

from __future__ import annotations

import pandas as pd
import torch

from src.evaluation.geometry_truth import (
    compute_feature_statistics,
    rank_candidate_features,
    split_true_false,
)


def test_split_true_false_separates_by_label():
    df = pd.DataFrame(
        {
            "statement": ["a", "b", "c", "d"],
            "label": [1, 0, 1, 0],
        }
    )

    true_df, false_df = split_true_false(df)

    assert sorted(true_df["statement"]) == ["a", "c"]
    assert sorted(false_df["statement"]) == ["b", "d"]


def test_compute_feature_statistics_computes_mean_and_frequency_differences():
    # 2 true examples, 2 false examples, 3 features.
    true_features = torch.tensor(
        [
            [1.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
        ]
    )
    false_features = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )

    stats = compute_feature_statistics(true_features, false_features)

    assert list(stats["feature_id"]) == [0, 1, 2]

    # Feature 0: fires on every true example, never on false -> fully separating.
    row0 = stats.loc[stats["feature_id"] == 0].iloc[0]
    assert row0["true_mean_activation"] == 1.0
    assert row0["false_mean_activation"] == 0.0
    assert row0["mean_activation_difference"] == 1.0
    assert row0["true_activation_frequency"] == 1.0
    assert row0["false_activation_frequency"] == 0.0

    # Feature 2: fires on both groups but with a large magnitude gap.
    row2 = stats.loc[stats["feature_id"] == 2].iloc[0]
    assert row2["mean_activation_difference"] == 4.0


def test_rank_candidate_features_returns_top_n_each_direction():
    stats = pd.DataFrame(
        {
            "feature_id": [0, 1, 2, 3],
            "mean_activation_difference": [5.0, -5.0, 1.0, -1.0],
        }
    )

    top_true, top_false = rank_candidate_features(stats, top_n=2)

    # Most positive differences (true-associated) first.
    assert list(top_true["feature_id"]) == [0, 2]
    # Most negative differences (false-associated) first.
    assert list(top_false["feature_id"]) == [1, 3]


def test_rank_candidate_features_respects_top_n():
    stats = pd.DataFrame(
        {
            "feature_id": range(10),
            "mean_activation_difference": [float(i) for i in range(10)],
        }
    )

    top_true, top_false = rank_candidate_features(stats, top_n=3)

    assert len(top_true) == 3
    assert len(top_false) == 3
