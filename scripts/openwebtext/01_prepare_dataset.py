
"""
Stage 1: OpenWebText Dataset Preparation

SAE Faithfulness Research
Models: GPT-2 Small and Gemma 2 2B

Tasks:
1. Load OpenWebText using streaming
2. Clean and filter documents
3. Collect a reproducible sample
4. Split into train, validation, test
5. Save datasets and metadata

No LLM training or activation extraction occurs here.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset


def clean_text(text):
    """Normalize whitespace without changing document content."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def collect_documents(config):
    """Stream OpenWebText and collect a deterministic pilot sample."""

    settings = config["dataset"]

    target = settings["sample_documents"]
    min_chars = settings["min_characters"]
    max_chars = settings["max_characters"]
    seed = settings["random_seed"]

    print("Loading OpenWebText in streaming mode...")

    dataset = load_dataset(
        settings["name"],
        split="train",
        streaming=True,
    )

    # Shuffle using a fixed seed for reproducibility.
    # Streaming shuffle is buffer-based, not a global shuffle.
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=10000,
    )

    documents = []
    seen = set()

    for row in dataset:
        text = clean_text(row.get(settings["text_column"], ""))

        if not min_chars <= len(text) <= max_chars:
            continue

        # Exact-text deduplication
        if text in seen:
            continue

        seen.add(text)

        documents.append({
            "document_id": f"owt_{len(documents):06d}",
            "text": text,
            "character_count": len(text),
        })

        if len(documents) % 100 == 0:
            print(f"Collected {len(documents)}/{target} documents")

        if len(documents) >= target:
            break

    if len(documents) < target:
        raise RuntimeError(
            f"Only collected {len(documents)} of {target} documents."
        )

    return documents


def split_documents(documents, config):
    """Split documents without overlapping document IDs."""

    settings = config["dataset"]
    ratios = settings["split"]

    assert abs(
        ratios["train"] + ratios["validation"] + ratios["test"] - 1
    ) < 1e-8

    rng = random.Random(settings["random_seed"])
    rng.shuffle(documents)

    n = len(documents)

    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["validation"])

    return {
        "train": documents[:train_end],
        "validation": documents[train_end:val_end],
        "test": documents[val_end:],
    }


def save_results(splits, config):
    """Save each split and a dataset summary."""

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": config["dataset"]["name"],
        "random_seed": config["dataset"]["random_seed"],
        "splits": {},
    }

    for name, records in splits.items():
        dataframe = pd.DataFrame(records)

        output_path = output_dir / f"{name}.parquet"
        dataframe.to_parquet(output_path, index=False)

        summary["splits"][name] = {
            "documents": len(dataframe),
            "characters": int(dataframe["character_count"].sum()),
            "file": str(output_path),
        }

        print(f"Saved {name}: {len(dataframe)} documents")

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Metadata saved: {metadata_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/openwebtext/dataset.yaml",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    documents = collect_documents(config)

    splits = split_documents(documents, config)

    summary = save_results(splits, config)

    print("\n=== STAGE 1 COMPLETE ===")

    for name, info in summary["splits"].items():
        print(f"{name}: {info['documents']} documents")


if __name__ == "__main__":
    main()
