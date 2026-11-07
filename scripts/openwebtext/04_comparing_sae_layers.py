
#!/usr/bin/env python3
"""
Generate three separate PNG tables:
1. ReLU SAE
2. TopK SAE
3. JumpReLU SAE

Each table covers GPT-2 Small layers 0-11
at target L0 values 32, 64, and 128.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------------------
# PATHS AND CONFIGURATION
# -------------------------------------------

INPUT_DIR = Path(
    "results/openwebtext/sae_sparsity_sweep"
)

OUTPUT_DIR = Path(
    "results/openwebtext/layer_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURES = {
    "relu": "ReLU SAE",
    "topk": "TopK SAE",
    "jumprelu": "JumpReLU SAE",
}

HEADERS = [
    "GPT-2 Layer",
    "Target L0",
    "MSE ↓",
    "Cosine ↑",
    "Explained Variance ↑",
    "Actual L0",
    "L0 Matched",
]


# -------------------------------------------
# LOAD ALL LAYER RESULTS
# -------------------------------------------

frames = []

for layer in range(12):
    folder = INPUT_DIR / f"layer_{layer}"

    csv_path = folder / "comparison.csv"

    if not csv_path.exists():
        csv_path = folder / "metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing results for layer {layer}"
        )

    frame = pd.read_csv(csv_path)
    frame["layer"] = layer
    frames.append(frame)

df = pd.concat(frames, ignore_index=True)

required = [
    "layer",
    "architecture",
    "target_l0",
    "mse",
    "cosine",
    "explained_variance",
    "l0",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for column in [
    "layer",
    "target_l0",
    "mse",
    "cosine",
    "explained_variance",
    "l0",
]:
    df[column] = pd.to_numeric(
        df[column], errors="raise"
    )

df["architecture"] = (
    df["architecture"].astype(str).str.lower()
)

df["l0_error"] = (
    (df["l0"] - df["target_l0"]).abs()
    / df["target_l0"]
)

df["matched"] = df["l0_error"] <= 0.10

df = df.sort_values(
    ["architecture", "layer", "target_l0"]
).reset_index(drop=True)

if len(df) != 108:
    raise ValueError(
        f"Expected 108 configurations, found {len(df)}"
    )

if df.duplicated(
    ["layer", "target_l0", "architecture"]
).any():
    raise ValueError("Duplicate experiment configurations")

df.to_csv(
    OUTPUT_DIR / "all_layers_comparison.csv",
    index=False,
)


# -------------------------------------------
# GENERATE ONE TABLE PER SAE ARCHITECTURE
# -------------------------------------------

def create_architecture_table(architecture, title):

    subset = df[
        df["architecture"] == architecture
    ].copy()

    subset = subset.sort_values(
        ["layer", "target_l0"]
    ).reset_index(drop=True)

    if len(subset) != 36:
        raise ValueError(
            f"Expected 36 rows for {architecture}, "
            f"found {len(subset)}"
        )

    table_data = []

    for _, row in subset.iterrows():

        table_data.append([
            str(int(row["layer"])),
            str(int(row["target_l0"])),
            f'{row["mse"]:.3f}',
            f'{row["cosine"]:.3f}',
            f'{row["explained_variance"]:.3%}',
            f'{row["l0"]:.3f}',
            "Yes" if row["matched"] else "No",
        ])

    # Compact figure with minimal empty space.
    fig, ax = plt.subplots(
        figsize=(14, 13.5)
    )

    fig.patch.set_facecolor("white")
    ax.axis("off")

    fig.text(
        0.5, 0.965,
        f"{title} — Layer-wise Reconstruction",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#172b4d",
    )

    fig.text(
        0.5, 0.936,
        "GPT-2 Small | OpenWebText | "
        "Layers 0–11 | Target L0: 32, 64, 128",
        ha="center",
        va="center",
        fontsize=11,
        color="#64748b",
    )

    table = ax.table(
        cellText=table_data,
        colLabels=HEADERS,
        cellLoc="center",
        colLoc="center",
        colWidths=[
            0.13, 0.12, 0.12, 0.12,
            0.20, 0.15, 0.16,
        ],
        bbox=[0.025, 0.075, 0.95, 0.825],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Header styling
    for col in range(len(HEADERS)):

        cell = table[(0, col)]
        cell.set_facecolor("#172b4d")
        cell.set_edgecolor("#172b4d")
        cell.set_linewidth(1.0)
        cell.set_text_props(
            color="white",
            weight="bold",
            fontsize=10,
        )

    # Data row styling
    for row_index in range(len(subset)):

        record = subset.iloc[row_index]

        layer = int(record["layer"])
        target = int(record["target_l0"])
        matched = bool(record["matched"])

        # Alternate background by GPT-2 layer.
        background = (
            "#ffffff"
            if layer % 2 == 0
            else "#f1f5f9"
        )

        # Highlight the lowest MSE among the
        # three target-L0 runs for this layer.
        layer_rows = subset[
            subset["layer"] == layer
        ]

        best_index = layer_rows["mse"].idxmin()
        is_best = row_index == best_index

        if is_best:
            background = "#dcfce7"

        for col in range(len(HEADERS)):

            cell = table[(row_index + 1, col)]

            cell.set_facecolor(background)
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.65)

            cell.set_text_props(
                color="#1e293b",
                weight="bold" if is_best else "normal",
                fontsize=10,
            )

        if not matched:
            table[
                (row_index + 1, 6)
            ].set_text_props(
                color="#b91c1c",
                weight="bold",
            )

        # Strong border between layers.
        if row_index % 3 == 0:
            for col in range(len(HEADERS)):
                table[
                    (row_index + 1, col)
                ].set_linewidth(1.3)

    fig.text(
        0.5, 0.035,
        "Green = lowest reconstruction MSE within "
        "that layer (across sparsity targets). "
        "Red 'No' = actual L0 differs from target "
        "by more than 10%.",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )

    output_path = (
        OUTPUT_DIR / f"{architecture}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# -------------------------------------------
# GENERATE ALL THREE TABLES
# -------------------------------------------

for architecture, title in ARCHITECTURES.items():
    create_architecture_table(
        architecture,
        title,
    )

print("\n===== COMPLETE =====")
print("Generated three separate PNG tables.")
print(f"Results folder: {OUTPUT_DIR.resolve()}")
