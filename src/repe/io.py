"""Paths and saving helpers for the RepE analysis (baseline and SAE stages)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "repe"
SPLIT_PATH = DATA_DIR / "facts_split.csv"
PAIRS_SPLIT_PATH = DATA_DIR / "honesty_pairs_split.csv"
ACTIVATIONS_DIR = DATA_DIR / "activations"

TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "repe"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "repe"


def save_table(df, name, index=False):
    """Save a table to results/tables/repe/<name>.csv and return the path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path