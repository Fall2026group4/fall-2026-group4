from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
from collections import Counter, defaultdict
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


# Load Gemma 2 tokenizer


print("Loading Gemma 2 2B tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b"
)

print("Gemma tokenizer loaded successfully.")
print("Vocabulary size:", tokenizer.vocab_size)


# Load The Pile


print("\nLoading The Pile in streaming mode...")

dataset = load_dataset(
    "monology/pile-uncopyrighted-parquet",
    split="train",
    streaming=True,
)

# Use a small shuffle buffer to avoid excessive EC2 memory use
sample = dataset.shuffle(
    seed=SEED,
    buffer_size=SHUFFLE_BUFFER
).take(SAMPLE_SIZE)


# Containers for EDA


rows = []

source_counts = Counter()

source_token_counts = defaultdict(list)

seen_hashes = set()

empty_count = 0
duplicate_count = 0

char_lengths = []
word_lengths = []
token_lengths = []

print(f"Analyzing {SAMPLE_SIZE} streamed documents...")


# Analyze documents


for i, row in enumerate(sample, start=1):

    text = row.get("text", "") or ""
    meta = row.get("meta", {}) or {}

    source = meta.get("pile_set_name", "Unknown")

    # Basic text statistics
    char_length = len(text)
    word_count = len(text.split())

    # Gemma tokenizer
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    token_count = len(token_ids)

    # Empty document check
    is_empty = not bool(text.strip())

    if is_empty:
        empty_count += 1

    # Exact duplicate check
    text_hash = hashlib.sha1(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()

    if text_hash in seen_hashes:
        duplicate_count += 1
    else:
        seen_hashes.add(text_hash)

    # Store statistics
    source_counts[source] += 1

    char_lengths.append(char_length)
    word_lengths.append(word_count)
    token_lengths.append(token_count)

    source_token_counts[source].append(token_count)

    rows.append(
        {
            "document_id": i,
            "source": source,
            "char_length": char_length,
            "word_count": word_count,
            "token_count": token_count,
            "empty": is_empty,
        }
    )


# Overall EDA summary


summary = {
    "sample_size": len(rows),
    "unique_sources": len(source_counts),
    "empty_documents": empty_count,
    "exact_duplicate_documents": duplicate_count,

    "mean_char_length": round(
        statistics.mean(char_lengths), 2
    ),
    "median_char_length": round(
        statistics.median(char_lengths), 2
    ),
    "min_char_length": min(char_lengths),
    "max_char_length": max(char_lengths),

    "mean_word_count": round(
        statistics.mean(word_lengths), 2
    ),
    "median_word_count": round(
        statistics.median(word_lengths), 2
    ),
    "min_word_count": min(word_lengths),
    "max_word_count": max(word_lengths),

    "mean_token_count": round(
        statistics.mean(token_lengths), 2
    ),
    "median_token_count": round(
        statistics.median(token_lengths), 2
    ),
    "min_token_count": min(token_lengths),
    "max_token_count": max(token_lengths),
}


# Source-level token statistics


source_stats = []

for source, counts in source_token_counts.items():

    source_stats.append(
        {
            "source": source,
            "document_count": len(counts),
            "mean_token_count": round(
                statistics.mean(counts), 2
            ),
            "median_token_count": round(
                statistics.median(counts), 2
            ),
            "min_token_count": min(counts),
            "max_token_count": max(counts),
        }
    )

source_stats.sort(
    key=lambda x: x["document_count"],
    reverse=True
)

# Save document-level results


with open(
    OUTPUT_DIR / "pile_eda_sample.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)


# Save source counts


with open(
    OUTPUT_DIR / "pile_source_counts.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source",
            "document_count"
        ]
    )

    for source, count in source_counts.most_common():

        writer.writerow(
            [
                source,
                count
            ]
        )


# Save source-level token statistics


with open(
    OUTPUT_DIR / "pile_source_token_stats.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "source",
        "document_count",
        "mean_token_count",
        "median_token_count",
        "min_token_count",
        "max_token_count",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(source_stats)


# Save summary JSON


with open(
    OUTPUT_DIR / "pile_eda_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# Print results


print("\n=== GEMMA 2 + THE PILE EDA SUMMARY ===")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTop sources:")

for source, count in source_counts.most_common(10):
    print(f"{source}: {count}")

print("\nMean Gemma token count by source:")

for stat in source_stats[:10]:

    print(
        f"{stat['source']}: "
        f"{stat['mean_token_count']} tokens "
        f"(n={stat['document_count']})"
    )

print("\nEDA outputs saved to:")
print(OUTPUT_DIR)