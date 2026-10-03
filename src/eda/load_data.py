from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "repe" / "facts_true_false.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "repe" / "facts_clean.csv"

TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "eda"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "eda"


def load_raw(path=RAW_PATH):
    """Load the CSV exactly as downloaded, without any cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download facts_true_false.csv from the RepE repo "
            "(andyzoujm/representation-engineering, data/facts/) into data/repe/."
        )
    return pd.read_csv(path)


def load_clean(path=CLEAN_PATH):
    """Load the cleaned dataset written by step 1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run step 1 first: python scripts/run_eda.py --step 1")
    return pd.read_csv(path)


def save_table(df, name, index=False):
    """Save a table to results/tables/eda/<name>.csv and return the path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path