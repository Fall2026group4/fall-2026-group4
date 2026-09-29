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

def compute_feature_statistics(feature_matrix, labels):
    """Compute mean SAE activation for true and false statements."""

    true_features = feature_matrix[labels == 1]
    false_features = feature_matrix[labels == 0]

    true_mean = true_features.mean(axis=0)
    false_mean = false_features.mean(axis=0)

    stats = pd.DataFrame({
        "feature_id": range(feature_matrix.shape[1]),
        "true_mean": true_mean,
        "false_mean": false_mean,
    })

    stats["mean_difference"] = (
        stats["true_mean"] - stats["false_mean"]
    )

    return stats
