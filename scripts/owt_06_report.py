"""Stage 6: Per-layer plots, the sparsity-fidelity scatter, dead/dense
latent counts, and the KL dose-response figure.

Note on figure format: the project spec's repo-structure listing marks
`results/owt/figures/` as PNGs, but every other part of this project
switched to vector SVG after Instructor Review #2 flagged PNG results
figures. This script saves SVG to stay consistent with that project-wide
rule - flag this to the team/Dr. Jafari if PNG was actually intended here.

Run: python scripts/owt_06_report.py --config configs/owt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config_path: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.owt.data import load_config

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)

    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fidelity_path = tables_dir / "fidelity.csv"
    if not fidelity_path.exists():
        raise FileNotFoundError(f"{fidelity_path} not found - run owt_02_fidelity.py first.")
    fidelity_df = pd.read_csv(fidelity_path)

    wikitext_path = tables_dir / "fidelity_wikitext.csv"
    wikitext_df = pd.read_csv(wikitext_path) if wikitext_path.exists() else None

    feature_stats_path = tables_dir / "feature_stats.csv"
    feature_stats_df = pd.read_csv(feature_stats_path) if feature_stats_path.exists() else None

    steering_path = tables_dir / "steering_cost.csv"
    steering_df = pd.read_csv(steering_path) if steering_path.exists() else None

    sae_types = sorted(fidelity_df["sae_type"].unique())

    # 1. Per-layer FVU, L0, loss recovered (one line per SAE type).
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for sae_type in sae_types:
        sub = fidelity_df[fidelity_df["sae_type"] == sae_type].sort_values("layer")
        axes[0].plot(sub["layer"], sub["fvu"], marker="o", label=sae_type)
        axes[1].plot(sub["layer"], sub["l0"], marker="o", label=sae_type)
        axes[2].plot(sub["layer"], sub["loss_recovered_pct"], marker="o", label=f"{sae_type} (OWT)")
        if wikitext_df is not None:
            wt_sub = wikitext_df[wikitext_df["sae_type"] == sae_type].sort_values("layer")
            axes[2].plot(
                wt_sub["layer"], wt_sub["loss_recovered_pct"],
                marker="x", linestyle="--", label=f"{sae_type} (WikiText)",
            )

    axes[0].set_title("FVU by layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Fraction of Variance Unexplained")
    axes[0].legend()

    axes[1].set_title("L0 by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean active latents per token")
    axes[1].legend()

    axes[2].set_title("Loss recovered by layer")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("% CE loss recovered")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "owt_per_layer_fidelity.svg", bbox_inches="tight")
    plt.close()
    print("Saved: owt_per_layer_fidelity.svg")

    # 2. Sparsity-fidelity scatter: one point per SAE type and layer.
    plt.figure(figsize=(7, 6))
    for sae_type in sae_types:
        sub = fidelity_df[fidelity_df["sae_type"] == sae_type]
        plt.scatter(sub["l0"], sub["loss_recovered_pct"], label=sae_type, s=60)
        for _, row in sub.iterrows():
            plt.annotate(str(int(row["layer"])), (row["l0"], row["loss_recovered_pct"]), fontsize=7)
    plt.xlabel("L0 (mean active latents per token)")
    plt.ylabel("% CE loss recovered")
    plt.title("Sparsity vs Fidelity (labels are layer index)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "owt_sparsity_fidelity_scatter.svg", bbox_inches="tight")
    plt.close()
    print("Saved: owt_sparsity_fidelity_scatter.svg")

    # 3. Dead/dense latent counts per layer.
    if feature_stats_df is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(len(sae_types), 1)
        for i, sae_type in enumerate(sae_types):
            sub = feature_stats_df[feature_stats_df["sae_type"] == sae_type].sort_values("layer")
            offset = (i - (len(sae_types) - 1) / 2) * width
            ax.bar(sub["layer"] + offset, sub["dead_latents"], width=width, label=f"{sae_type} dead")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Dead latent count")
        ax.set_title("Dead latents per layer")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "owt_dead_dense_latents.svg", bbox_inches="tight")
        plt.close()
        print("Saved: owt_dead_dense_latents.svg")
    else:
        print("Skipping dead/dense figure (run owt_03_feature_stats.py first).")

    # 4. KL dose-response: KL vs alpha per layer, random directions +
    # truth/honesty directions overlaid.
    if steering_df is not None:
        is_extra = ~steering_df["direction"].str.startswith("random_feature_")
        random_mean = (
            steering_df[~is_extra]
            .groupby(["sae_type", "layer", "alpha"], as_index=False)["kl_divergence"]
            .mean()
        )
        extra = steering_df[is_extra]

        layers_with_data = sorted(random_mean["layer"].unique())
        fig, ax = plt.subplots(figsize=(8, 5))
        for sae_type in sae_types:
            sub = random_mean[random_mean["sae_type"] == sae_type]
            for layer in layers_with_data:
                layer_sub = sub[sub["layer"] == layer].sort_values("alpha")
                if len(layer_sub):
                    ax.plot(
                        layer_sub["alpha"], layer_sub["kl_divergence"],
                        alpha=0.3, color="gray", linewidth=1,
                    )
        for _, direction_name in enumerate(extra["direction"].unique() if len(extra) else []):
            sub = extra[extra["direction"] == direction_name].groupby("alpha", as_index=False)["kl_divergence"].mean()
            ax.plot(sub["alpha"], sub["kl_divergence"], marker="o", linewidth=2, label=direction_name)

        ax.set_xlabel("Steering strength (alpha)")
        ax.set_ylabel("Mean KL divergence from clean")
        ax.set_title("Steering cost: KL dose-response\n(gray = random directions, all layers; colored = shared truth/honesty directions)")
        if len(extra):
            ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "owt_kl_dose_response.svg", bbox_inches="tight")
        plt.close()
        print("Saved: owt_kl_dose_response.svg")
    else:
        print("Skipping KL dose-response figure (run owt_04_steering_cost.py first).")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)
