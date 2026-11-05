from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt


# Gemma 2 2B + The Pile
# EDA Visualizations and Summary


RESULTS_DIR = Path("results/gemma2_pile")
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = RESULTS_DIR / "pile_eda_sample.csv"
SOURCE_FILE = RESULTS_DIR / "pile_source_token_stats.csv"
TRUNCATION_FILE = RESULTS_DIR / "pile_truncation_summary.csv"
SUMMARY_FILE = RESULTS_DIR / "pile_eda_summary.json"


# Load document-level sample


token_counts = []

with open(
    SAMPLE_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        token_counts.append(
            int(row["token_count"])
        )


# Load source-level statistics


source_names = []
source_document_counts = []
source_mean_tokens = []

with open(
    SOURCE_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        source_names.append(row["source"])
        source_document_counts.append(
            int(row["document_count"])
        )
        source_mean_tokens.append(
            float(row["mean_token_count"])
        )


# Load truncation statistics


thresholds = []
truncation_percentages = []

with open(
    TRUNCATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        thresholds.append(
            int(row["token_threshold"])
        )
        truncation_percentages.append(
            float(row["percent_over_threshold"])
        )


# Load overall summary


with open(
    SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:

    summary = json.load(f)


# Figure 1: Gemma token-length distribution


plt.figure(figsize=(10, 6))

plt.hist(
    token_counts,
    bins=30
)

plt.xlabel("Gemma 2 Token Count")
plt.ylabel("Number of Documents")
plt.title(
    "Gemma 2 Token-Length Distribution\n"
    "Initial The Pile Sample"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gemma_token_length_distribution.png",
    dpi=200
)

plt.close()


# Figure 2: Pile source distribution


plt.figure(figsize=(10, 7))

plt.barh(
    source_names[::-1],
    source_document_counts[::-1]
)

plt.xlabel("Documents in Sample")
plt.ylabel("Pile Source")
plt.title(
    "Source Distribution in Initial The Pile Sample"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "pile_source_distribution.png",
    dpi=200
)

plt.close()


# Figure 3: Context-length / truncation risk


threshold_labels = [
    f">{threshold:,}"
    for threshold in thresholds
]

plt.figure(figsize=(9, 6))

plt.bar(
    threshold_labels,
    truncation_percentages
)

plt.xlabel("Gemma Token Threshold")
plt.ylabel("Documents Above Threshold (%)")
plt.title(
    "Gemma 2 Context-Length / Truncation Risk"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gemma_truncation_risk.png",
    dpi=200
)

plt.close()


# Create EDA README


README_FILE = RESULTS_DIR / "README.md"

with open(
    README_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("# Gemma 2 2B + The Pile — Exploratory Data Analysis\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This analysis characterizes an initial streamed sample "
        "of The Pile before Gemma 2 2B model-level experiments. "
        "The goal is to identify data-quality issues, source "
        "heterogeneity, token-length variation, and potential "
        "context-length constraints.\n\n"
    )

    f.write("## Sample\n\n")
    f.write(
        f"- Documents analyzed: {summary['sample_size']}\n"
    )
    f.write(
        f"- Pile sources represented: {summary['unique_sources']}\n"
    )
    f.write(
        f"- Empty documents: {summary['empty_documents']}\n"
    )
    f.write(
        "- Exact duplicate documents: "
        f"{summary['exact_duplicate_documents']}\n\n"
    )

    f.write("## Document Length\n\n")
    f.write(
        f"- Mean word count: {summary['mean_word_count']}\n"
    )
    f.write(
        f"- Median word count: {summary['median_word_count']}\n"
    )
    f.write(
        f"- Mean Gemma token count: {summary['mean_token_count']}\n"
    )
    f.write(
        f"- Median Gemma token count: {summary['median_token_count']}\n"
    )
    f.write(
        f"- Minimum Gemma token count: {summary['min_token_count']}\n"
    )
    f.write(
        f"- Maximum Gemma token count: {summary['max_token_count']}\n\n"
    )

    f.write("## Context-Length / Truncation Risk\n\n")

    for threshold, percentage in zip(
        thresholds,
        truncation_percentages
    ):
        f.write(
            f"- {percentage}% of sampled documents exceed "
            f"{threshold:,} Gemma tokens.\n"
        )

    f.write("\n## Key Findings\n\n")

    f.write(
        "1. The Pile is highly heterogeneous across text sources.\n"
    )

    f.write(
        "2. Document length varies substantially across sources, "
        "with some technical and scientific sources producing "
        "much longer Gemma token sequences.\n"
    )

    f.write(
        "3. A meaningful fraction of documents exceed common "
        "context-length thresholds, so preprocessing must use "
        "an explicit truncation or chunking policy.\n"
    )

    f.write(
        "4. No empty documents or exact duplicates were found "
        "in this initial sample.\n\n"
    )

    f.write("## Limitation\n\n")

    f.write(
        "These findings are based on an initial streamed sample "
        "of 200 documents and should be treated as exploratory "
        "rather than population-level estimates for the full Pile.\n\n"
    )

    f.write("## Recommended Next Step\n\n")

    f.write(
        "Define a consistent Gemma-compatible context-length, "
        "truncation, or chunking strategy before beginning "
        "activation-level model analysis.\n"
    )

print("EDA visualizations created successfully.")

print("\nSaved figures:")
print(
    FIGURES_DIR / "gemma_token_length_distribution.png"
)
print(
    FIGURES_DIR / "pile_source_distribution.png"
)
print(
    FIGURES_DIR / "gemma_truncation_risk.png"
)

print("\nEDA summary saved to:")
print(README_FILE)