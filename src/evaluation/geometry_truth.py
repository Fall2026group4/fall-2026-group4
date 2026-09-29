"""Geometry of Truth evaluation helpers.

Reusable functions for loading the Geometry of Truth dataset
and preparing true/false SAE evaluation data.
"""

import pandas as pd

def load_geometry_dataset(path):
    """Load the Geometry of Truth CSV dataset."""
    df = pd.read_csv(path)
    return df
