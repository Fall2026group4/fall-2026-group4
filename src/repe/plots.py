"""Figures for the RepE analysis. Each function saves a PNG and returns its path."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.repe.io import FIGURES_DIR, PROJECT_ROOT

METHOD_STYLE = {
    "repe_pca": ("RepE PCA", "#4C72B0"),
    "mean_diff": ("Mean difference", "#55A868"),
    "logistic": ("Logistic probe", "#C44E52"),
}


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path


def plot_baseline_by_layer(results, band, n_test_pairs, metric="auroc", name=None):
    """Test score per layer for each method, with 95% CIs and the random-direction band.

    metric: "auroc" (main) or "pair_accuracy".
    """
    low, high = ("auroc_low", "auroc_high") if metric == "auroc" else ("pair_acc_low", "pair_acc_high")
    name = name or f"repe_s2_honesty_by_layer_{metric}"
    reads = [r for r in ["last", "mean"] if r in set(results["read"])]
    titles = {"last": "Read at the last token", "mean": "Averaged over statement tokens"}
    fig, axes = plt.subplots(1, len(reads), figsize=(6.5 * len(reads), 4.5), sharey=True, squeeze=False)

    for ax, read in zip(axes[0], reads):
        b = band[band["read"] == read].sort_values("position_index")
        x = b["position_index"].values
        ax.fill_between(x, b[f"random_{metric}_low"], b[f"random_{metric}_high"], color="#BBBBBB", alpha=0.5,
                        label="Random directions (95% band)")
        for method, (label, color) in METHOD_STYLE.items():
            r = results[(results["read"] == read) & (results["method"] == method)].sort_values("position_index")
            ax.plot(r["position_index"], r[metric], marker="o", ms=4, color=color, label=label)
            ax.fill_between(r["position_index"], r[low], r[high], color=color, alpha=0.15)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(b["position"].values, fontsize=8)
        ax.set_xlabel("Layer (resid_pre; 'final' = output of layer 11)")
        ax.set_title(titles[read])
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel(f"Test {'AUROC' if metric == 'auroc' else 'pair accuracy'} (0.5 = chance)")
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Honest vs dishonest framing, original RepE wording ({n_test_pairs} test pairs)")
    return save_figure(fig, name)


SET_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452"]


def plot_within_set(within, band, method="mean_diff", name="repe_s3_within_set_by_layer"):
    """AUROC by layer inside each prompt set; grey = random band of a length-matched set."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    matched = [s for s in band["prompt_set"].unique() if s.startswith("matched")]
    ref = matched[0] if matched else band["prompt_set"].iloc[0]
    b = band[band["prompt_set"] == ref].sort_values("position_index")
    ax.fill_between(b["position_index"], b["random_auroc_low"], b["random_auroc_high"],
                    color="#BBBBBB", alpha=0.5, label=f"Random directions ({ref})")
    for color, (set_name, r) in zip(SET_COLORS, within[within["method"] == method].groupby("prompt_set", sort=False)):
        r = r.sort_values("position_index")
        style = "-" if set_name.startswith("matched") else "--"
        ax.plot(r["position_index"], r["auroc"], style, marker="o", ms=3.5, color=color, label=set_name)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(b["position_index"])
    ax.set_xticklabels(b["position"], fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUROC (0.5 = chance)")
    ax.set_title(f"Honesty signal inside each prompt set ({method}); solid = length-matched")
    ax.legend(fontsize=7, loc="lower right")
    return save_figure(fig, name)


def plot_transfer_heatmap(transfer_df, layers, method="mean_diff", name="repe_s3_transfer_heatmap"):
    """Train wording (rows) x test wording (columns), AUROC averaged over the given layers."""
    import numpy as np

    t = transfer_df[(transfer_df["method"] == method) & (transfer_df["position"].isin(layers))]
    order = list(dict.fromkeys(t["source"]))
    m = t.pivot_table(index="source", columns="target", values="auroc", aggfunc="mean").loc[order, order]
    fig, ax = plt.subplots(figsize=(1.3 * len(order) + 3, 1.0 * len(order) + 2))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.3, vmax=1.0)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    labels = ["m: " + s[len("matched_"):].replace("_", "/") if s.startswith("matched_") else s for s in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Fitted on")
    ax.set_title(f"Wording transfer, {method}, mean AUROC over {layers[0]}\u2013{layers[-1]}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test AUROC")
    return save_figure(fig, name)


def plot_transfer_from_original(transfer_df, method="mean_diff", name="repe_s3_transfer_from_original"):
    """Directions fitted on the ORIGINAL wording, tested on every other set, by layer."""
    t = transfer_df[(transfer_df["source"] == "original") & (transfer_df["method"] == method)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for color, (target, r) in zip(SET_COLORS, t.groupby("target", sort=False)):
        r = r.sort_values("position_index")
        ax.plot(r["position_index"], r["auroc"], marker="o", ms=3.5, color=color, label=f"tested on {target}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    first = t[t["target"] == t["target"].iloc[0]].sort_values("position_index")
    ax.set_xticks(first["position_index"])
    ax.set_xticklabels(first["position"], fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUROC (0.5 = chance)")
    ax.set_title(f"Fitted on 'honest / untruthful', tested on other wordings ({method})")
    ax.legend(fontsize=7, loc="lower right")
    return save_figure(fig, name)