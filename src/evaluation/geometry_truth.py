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
