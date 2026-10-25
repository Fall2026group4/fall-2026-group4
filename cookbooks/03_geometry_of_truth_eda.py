"""Geometry of Truth Dataset - Exploratory Data Analysis.

Explores the Geometry of Truth "cities" dataset for use in the SAE-Faithful
project: inspects dataset structure, true/false statement labels, data
quality, and label/country balance, ahead of later concept-detection and
causal-faithfulness experiments (see cookbooks/04_geometry_of_truth_baseline.py).

Run from anywhere:
    python cookbooks/03_geometry_of_truth_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "results" / "figures"
RESULTS_DIR = REPO_ROOT / "results"

DATASET_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv"


def load_cities(url: str = DATASET_URL) -> pd.DataFrame:
    cities = pd.read_csv(url)
    print(cities.head())
    return cities


def describe_dataset(cities: pd.DataFrame) -> None:
    print("Shape:", cities.shape)
    print("\nColumns:")
    print(cities.columns.tolist())
    print("\nData types:")
    print(cities.dtypes)
    print("\nMissing values:")
    print(cities.isnull().sum())
    print("\nLabel distribution:")
    print(cities["label"].value_counts())
    print("\nLabel proportions:")
    print(cities["label"].value_counts(normalize=True))


def add_text_features(cities: pd.DataFrame) -> pd.DataFrame:
    cities["char_length"] = cities["statement"].str.len()
    cities["word_count"] = cities["statement"].str.split().str.len()
    print("Character length summary:")
    print(cities["char_length"].describe())
    print("\nWord count summary:")
    print(cities["word_count"].describe())
    print("\nAverage word count by label:")
    print(cities.groupby("label")["word_count"].mean())
    return cities


def check_duplicates_and_coverage(cities: pd.DataFrame) -> None:
    print("Duplicate rows:", cities.duplicated().sum())
    print("Duplicate statements:", cities["statement"].duplicated().sum())
    print("\nUnique cities:", cities["city"].nunique())
    print("Unique stated countries:", cities["country"].nunique())
    print("Unique correct countries:", cities["correct_country"].nunique())


def plot_label_and_length(cities: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cities["label"].value_counts().sort_index().plot(kind="bar", ax=axes[0])
    axes[0].set_title("True vs False Statement Balance")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["False (0)", "True (1)"], rotation=0)

    cities["word_count"].plot(kind="hist", bins=8, ax=axes[1])
    axes[1].set_title("Statement Word Count Distribution")
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "geometry_of_truth_label_and_length.svg", bbox_inches="tight")
    plt.close()


def plot_length_by_label(cities: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    cities[cities["label"] == 0]["word_count"].plot(kind="hist", bins=8, alpha=0.6, label="False")
    cities[cities["label"] == 1]["word_count"].plot(kind="hist", bins=8, alpha=0.6, label="True")
    plt.title("Word Count Distribution by Truth Label")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(FIGURES_DIR / "geometry_of_truth_length_by_label.svg", bbox_inches="tight")
    plt.close()


def show_examples(cities: pd.DataFrame, n: int = 10) -> None:
    false_examples = cities[cities["label"] == 0].head(n)
    true_examples = cities[cities["label"] == 1].head(n)
    cols = ["statement", "city", "country", "correct_country"]
    print("True examples:")
    print(true_examples[cols].to_string(index=False))
    print("\nFalse examples:")
    print(false_examples[cols].to_string(index=False))


def check_city_pairing(cities: pd.DataFrame) -> None:
    city_counts = cities.groupby("city").size()
    print("Statements per city:")
    print(city_counts.value_counts().sort_index())

    paired_check = cities.groupby("city")["label"].nunique()
    print("\nCities with both true and false labels:", (paired_check == 2).sum())
    print("Total unique cities:", cities["city"].nunique())


def plot_top_countries(cities: pd.DataFrame, top_n: int = 15) -> pd.Series:
    top_correct_countries = cities["correct_country"].value_counts().head(top_n)
    print(f"Top {top_n} correct countries by number of cities:")
    print(top_correct_countries)

    plt.figure(figsize=(10, 5))
    top_correct_countries.plot(kind="bar")
    plt.title(f"Top {top_n} Correct Countries in the Geometry of Truth Cities Dataset")
    plt.xlabel("Country")
    plt.ylabel("Number of Cities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "geometry_of_truth_top_countries.svg", bbox_inches="tight")
    plt.close()
    return top_correct_countries


def country_concentration(cities: pd.DataFrame) -> tuple[float, float]:
    country_counts = cities["correct_country"].value_counts()
    top_2_share = country_counts.head(2).sum() / country_counts.sum()
    top_5_share = country_counts.head(5).sum() / country_counts.sum()
    print(f"Share of cities from top 2 countries: {top_2_share:.2%}")
    print(f"Share of cities from top 5 countries: {top_5_share:.2%}")
    return top_2_share, top_5_share


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cities = load_cities()
    describe_dataset(cities)
    cities = add_text_features(cities)
    check_duplicates_and_coverage(cities)

    plot_label_and_length(cities)
    plot_length_by_label(cities)
    show_examples(cities)
    check_city_pairing(cities)
    plot_top_countries(cities)
    top_2_share, top_5_share = country_concentration(cities)

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "original_columns",
                "derived_text_features",
                "missing_values",
                "duplicate_rows",
                "duplicate_statements",
                "true_statements",
                "false_statements",
                "unique_cities",
                "unique_correct_countries",
                "mean_word_count",
                "top_2_country_share",
                "top_5_country_share",
            ],
            "value": [
                len(cities),
                5,
                2,
                cities.isnull().sum().sum(),
                cities.duplicated().sum(),
                cities["statement"].duplicated().sum(),
                (cities["label"] == 1).sum(),
                (cities["label"] == 0).sum(),
                cities["city"].nunique(),
                cities["correct_country"].nunique(),
                cities["word_count"].mean(),
                top_2_share,
                top_5_share,
            ],
        }
    )
    print(summary)
    summary.to_csv(RESULTS_DIR / "geometry_of_truth_eda_summary.csv", index=False)
    print("\nSummary saved to", RESULTS_DIR / "geometry_of_truth_eda_summary.csv")
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
