"""Paths, dataset names and saving helpers for the Geometry of Truth part."""

from pathlib import Path

import pandas as pd

import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "got" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "got" / "clean"
TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "got"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "got"

DATASETS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]

# Each topic has an affirmative dataset and its negated / reversed twin, built on the same subjects.
FAMILY = {
    "cities": "cities", "neg_cities": "cities",
    "sp_en_trans": "translation", "neg_sp_en_trans": "translation",
    "larger_than": "numbers", "smaller_than": "numbers",
}


def load_raw(name):
    """One dataset exactly as downloaded."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download it first (step 0.0).")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_clean(name):
    """One dataset after step 0.1's cleaning."""
    path = CLEAN_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run step 1 first: python scripts/run_got_eda.py --step 1")
    return pd.read_csv(path)


def save_table(df, name, index=False):
    """Save to results/tables/got/<name>.csv."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    logger.info(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path


# Added for steps 8-10
REVIEW_PATH = TABLES_DIR / "got_s0_08_label_review_sample.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "got" / "split"


# Added for stage 1 onwards
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "got" / "activations"
DIRECTIONS_DIR = PROJECT_ROOT / "data" / "got" / "directions"