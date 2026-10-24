"""Stage 6 figure utilities for Geometry of Truth results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_baseline_probe_figure(
    results_csv,
    output_path,
):
    """Plot layer-wise Accuracy and AUROC for raw GPT-2 residuals."""

    df = pd.read_csv(
        results_csv
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["layer"],
        df["accuracy"],
        marker="o",
        label="Accuracy",
    )

    ax.plot(
        df["layer"],
        df["auroc"],
        marker="o",
        label="AUROC",
    )

    ax.set_xlabel(
        "GPT-2 Layer"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Geometry of Truth — Baseline Probe Performance"
    )

    ax.set_xticks(
        df["layer"]
    )

    ax.set_ylim(
        0.45,
        1.0,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_faithfulness_figure(
    results_csv,
    output_path,
):
    """Compare raw and SAE-reconstructed probe performance."""

    df = pd.read_csv(
        results_csv
    )

    row = df.iloc[0]

    metrics = [
        "Accuracy",
        "AUROC",
    ]

    raw_values = [
        row["raw_accuracy"],
        row["raw_auroc"],
    ]

    reconstructed_values = [
        row["reconstructed_accuracy"],
        row["reconstructed_auroc"],
    ]

    x = range(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        raw_values,
        width=width,
        label="Raw residual",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        reconstructed_values,
        width=width,
        label="SAE reconstructed",
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        metrics
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        "SAE Reconstruction Faithfulness — Layer 10"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_steering_dose_response(
    results_csv,
    output_path,
):
    """Plot truth-probe score shift across steering strengths."""

    df = pd.read_csv(
        results_csv
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for condition, group in df.groupby(
        "condition"
    ):
        group = group.sort_values(
            "multiplier"
        )

        ax.plot(
            group["multiplier"],
            group["mean_probe_score_shift"],
            marker="o",
            label=condition,
        )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Steering Multiplier"
    )

    ax.set_ylabel(
        "Mean Truth-Probe Score Shift"
    )

    ax.set_title(
        "Geometry of Truth — Steering Dose Response"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_random_control_figure(
    random_results_csv,
    summary_csv,
    output_path,
):
    """Plot random SAE steering null distribution and candidate effect."""

    random_df = pd.read_csv(
        random_results_csv
    )

    summary_df = pd.read_csv(
        summary_csv
    )

    candidate_shift = float(
        summary_df.iloc[0][
            "candidate_probe_shift"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        random_df[
            "mean_probe_score_shift"
        ],
        bins=20,
        alpha=0.75,
        edgecolor="black",
    )

    ax.axvline(
        candidate_shift,
        linewidth=2,
        label=(
            f"Feature 19579 "
            f"({candidate_shift:.3f})"
        ),
    )

    ax.set_xlabel(
        "Mean Truth-Probe Score Shift"
    )

    ax.set_ylabel(
        "Number of Random SAE Features"
    )

    ax.set_title(
        "Feature 19579 vs 100 Random SAE Steering Controls"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_all_geometry_figures(
    results_dir,
    figure_dir,
):
    """Generate all Stage 6 Geometry SVG figures."""

    results_dir = Path(
        results_dir
    )

    figure_dir = Path(
        figure_dir
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_baseline_probe_figure(
        results_dir
        / "baseline_probe_results.csv",
        figure_dir
        / "geometry_baseline_probe_performance.svg",
    )

    save_faithfulness_figure(
        results_dir
        / "sae_faithfulness_results.csv",
        figure_dir
        / "geometry_sae_faithfulness.svg",
    )

    save_steering_dose_response(
        results_dir
        / "sae_steering_results.csv",
        figure_dir
        / "geometry_steering_dose_response.svg",
    )

    save_random_control_figure(
        results_dir
        / "sae_random_control_results.csv",
        results_dir
        / "sae_random_control_summary.csv",
        figure_dir
        / "geometry_random_control_null.svg",
    )