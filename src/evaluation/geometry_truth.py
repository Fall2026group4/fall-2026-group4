"""Geometry of Truth evaluation helpers.

Reusable functions for loading the Geometry of Truth dataset
and preparing true/false SAE evaluation data.
"""

import pandas as pd


def load_geometry_dataset(path):
    """Load the Geometry of Truth CSV dataset."""
    df = pd.read_csv(path)
    return df


def split_true_false(df):
    """Split the Geometry of Truth dataset into true and false statements."""
    true_df = df[df["label"] == 1].copy()
    false_df = df[df["label"] == 0].copy()
    return true_df, false_df


def compute_feature_statistics(true_features, false_features):
    """Compute mean activation and firing frequency for true and false statements."""

    true_mean = true_features.float().mean(dim=0)
    false_mean = false_features.float().mean(dim=0)

    true_frequency = (true_features > 0).float().mean(dim=0)
    false_frequency = (false_features > 0).float().mean(dim=0)

    mean_activation_difference = true_mean - false_mean
    frequency_difference = true_frequency - false_frequency

    stats = pd.DataFrame({
        "feature_id": range(true_features.shape[1]),
        "true_mean_activation": true_mean.cpu().numpy(),
        "false_mean_activation": false_mean.cpu().numpy(),
        "mean_activation_difference": mean_activation_difference.cpu().numpy(),
        "true_activation_frequency": true_frequency.cpu().numpy(),
        "false_activation_frequency": false_frequency.cpu().numpy(),
        "activation_frequency_difference": frequency_difference.cpu().numpy(),
    })

    return stats


def rank_candidate_features(stats, top_n=10):
    """Return the strongest true-associated and false-associated SAE features."""

    true_candidates = (
        stats.sort_values("mean_activation_difference", ascending=False)
        .head(top_n)
        .copy()
    )

    false_candidates = (
        stats.sort_values("mean_activation_difference", ascending=True)
        .head(top_n)
        .copy()
    )

    return true_candidates, false_candidates
