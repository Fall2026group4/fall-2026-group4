"""Exploratory data analysis on an OpenWebText sample.

Streams 1,000 OpenWebText documents, checks data quality, and compares
document length against GPT-2's context window. Saves figures as vector
SVGs (not PNG - see Instructor Review #2) and summary tables as CSVs.

Run from anywhere:
    python cookbooks/01_data_exploration.py
"""

from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "results" / "figures"
TABLES_DIR = REPO_ROOT / "results" / "tables"

SAMPLE_SIZE = 1000


def load_sample(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Stream a sample of OpenWebText documents without downloading the full dataset."""
    from datasets import load_dataset

    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    sample = list(islice(dataset, sample_size))
    df = pd.DataFrame(sample)
    print("OpenWebText sample loaded successfully.")
    print("Sample size:", len(df))
    return df


def check_data_quality(df: pd.DataFrame) -> tuple[int, int]:
    empty_documents = int((df["text"].str.strip() == "").sum())
    duplicate_documents = int(df["text"].duplicated().sum())
    print("Empty documents:", empty_documents)
    print("Duplicate documents:", duplicate_documents)
    return empty_documents, duplicate_documents


def plot_word_counts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["word_count"], bins=50, edgecolor="black")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of Document Lengths in OpenWebText Sample")
    plt.savefig(FIGURES_DIR / "eda_word_count_distribution.svg", bbox_inches="tight")
    plt.close()


def plot_token_counts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["token_count"], bins=50, edgecolor="black")
    plt.xlabel("Number of GPT-2 Tokens")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of GPT-2 Token Lengths in OpenWebText Sample")
    plt.savefig(FIGURES_DIR / "eda_token_count_distribution.svg", bbox_inches="tight")
    plt.close()


def plot_tokens_vs_context_limit(df: pd.DataFrame, context_limit: int = 1024) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["token_count"], bins=50, edgecolor="black")
    plt.axvline(
        x=context_limit,
        linestyle="--",
        linewidth=2,
        label=f"GPT-2 Context Limit ({context_limit} tokens)",
    )
    plt.xlabel("Number of GPT-2 Tokens")
    plt.ylabel("Number of Documents")
    plt.title("OpenWebText Token Lengths vs GPT-2 Context Limit")
    plt.legend()
    plt.savefig(FIGURES_DIR / "eda_token_vs_context_limit.svg", bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample()

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    print("\nFirst document preview:")
    print(df["text"].iloc[0][:300])

    empty_documents, duplicate_documents = check_data_quality(df)

    df["word_count"] = df["text"].str.split().str.len()
    print("\nMean word count:", round(df["word_count"].mean(), 2))
    print("Median word count:", df["word_count"].median())
    print("Minimum word count:", df["word_count"].min())
    print("Maximum word count:", df["word_count"].max())
    plot_word_counts(df)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("\nGPT-2 tokenizer loaded successfully.")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("GPT-2 maximum context length:", tokenizer.model_max_length, "tokens")

    example_text = df["text"].iloc[0][:100]
    token_ids = tokenizer.encode(example_text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("\nOriginal text:")
    print(example_text)
    print("\nGPT-2 tokens:")
    print(tokens)
    print("\nNumber of tokens:", len(tokens))

    df["token_count"] = df["text"].apply(
        lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
    print("\nMean token count:", round(df["token_count"].mean(), 2))
    print("Median token count:", df["token_count"].median())
    print("Minimum token count:", df["token_count"].min())
    print("Maximum token count:", df["token_count"].max())
    plot_token_counts(df)

    gpt2_max_length = tokenizer.model_max_length
    within_limit = int((df["token_count"] <= gpt2_max_length).sum())
    beyond_limit = int((df["token_count"] > gpt2_max_length).sum())
    print(f"\nDocuments within 1024 tokens: {within_limit}")
    print(f"Documents exceeding 1024 tokens: {beyond_limit}")
    plot_tokens_vs_context_limit(df, context_limit=gpt2_max_length)

    quality_df = pd.DataFrame(
        [
            {
                "empty_documents": empty_documents,
                "duplicate_documents": duplicate_documents,
                "sample_size": len(df),
            }
        ]
    )
    quality_df.to_csv(TABLES_DIR / "eda_data_quality.csv", index=False)

    df["word_count"].describe().to_csv(TABLES_DIR / "eda_word_count_stats.csv")
    df["token_count"].describe().to_csv(TABLES_DIR / "eda_token_count_stats.csv")

    compliance_df = pd.DataFrame(
        [
            {
                "within_1024_tokens": within_limit,
                "beyond_1024_tokens": beyond_limit,
                "pct_beyond": round(100 * beyond_limit / len(df), 2),
            }
        ]
    )
    compliance_df.to_csv(TABLES_DIR / "eda_context_window_compliance.csv", index=False)

    print("\nTables saved to", TABLES_DIR)
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
