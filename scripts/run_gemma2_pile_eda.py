from datasets import load_dataset
from pathlib import Path
from collections import Counter
import csv
import json
import hashlib
import statistics

# Gemma 2 2B + The Pile
# Stage 0: Exploratory Data Analysis

SAMPLE_SIZE = 200
SHUFFLE_BUFFER = 500
SEED = 42

OUTPUT_DIR = Path("results/gemma2_pile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading The Pile in streaming mode...")

dataset = load_dataset(
    "monology/pile-uncopyrighted-parquet",
    split="train",
    streaming=True,
)

# Shuffle a streaming buffer so EDA is not based only on the first rows
sample = dataset.shuffle(
    seed=SEED,
    buffer_size=SHUFFLE_BUFFER
).take(SAMPLE_SIZE)

rows = []
source_counts = Counter()
seen_hashes = set()

empty_count = 0
duplicate_count = 0

char_lengths = []
word_lengths = []

print(f"Analyzing {SAMPLE_SIZE} streamed documents...")

for i, row in enumerate(sample, start=1):
    text = row.get("text", "") or ""
    meta = row.get("meta", {}) or {}

    source = meta.get("pile_set_name", "Unknown")

    char_length = len(text)
    word_length = len(text.split())

    if not text.strip():
        empty_count += 1

    text_hash = hashlib.sha1(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()

    if text_hash in seen_hashes:
        duplicate_count += 1
    else:
        seen_hashes.add(text_hash)

    source_counts[source] += 1
    char_lengths.append(char_length)
    word_lengths.append(word_length)

    rows.append(
        {
            "document_id": i,
            "source": source,
            "char_length": char_length,
            "word_count": word_length,
            "empty": not bool(text.strip()),
        }
    )

# Overall summary

summary = {
    "sample_size": len(rows),
    "unique_sources": len(source_counts),
    "empty_documents": empty_count,
    "exact_duplicate_documents": duplicate_count,
    "mean_char_length": round(statistics.mean(char_lengths), 2),
    "median_char_length": round(statistics.median(char_lengths), 2),
    "min_char_length": min(char_lengths),
    "max_char_length": max(char_lengths),
    "mean_word_count": round(statistics.mean(word_lengths), 2),
    "median_word_count": round(statistics.median(word_lengths), 2),
    "min_word_count": min(word_lengths),
    "max_word_count": max(word_lengths),
}

# Save document-level EDA
with open(
    OUTPUT_DIR / "pile_eda_sample.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Save source distribution
with open(
    OUTPUT_DIR / "pile_source_counts.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["source", "document_count"])

    for source, count in source_counts.most_common():
        writer.writerow([source, count])

# Save summary
with open(
    OUTPUT_DIR / "pile_eda_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print("\n=== THE PILE EDA SUMMARY ===")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTop sources:")
for source, count in source_counts.most_common(10):
    print(f"{source}: {count}")

print("\nEDA outputs saved to:")
print(OUTPUT_DIR)