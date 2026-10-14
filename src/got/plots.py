"""Figures for the Geometry of Truth part. Each function saves a PNG and returns its path."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.got.io import FIGURES_DIR, PROJECT_ROOT

TRUE_COLOR, FALSE_COLOR = "#4C72B0", "#DD8452"


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Step 2: Label balance
# ---------------------------------------------------------------------------

def plot_label_balance(table, name="got_s0_02_label_balance"):
    """True vs false statements per dataset, twins side by side."""
    t = table[table["dataset"] != "ALL"].reset_index(drop=True)
    x = range(len(t))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars_t = ax.bar([i - width / 2 for i in x], t["n_true"], width, color=TRUE_COLOR, label="True")
    bars_f = ax.bar([i + width / 2 for i in x], t["n_false"], width, color=FALSE_COLOR, label="False")
    for bars in (bars_t, bars_f):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{d}\n{p:.0f}% true" for d, p in zip(t["dataset"], t["percent_true"])], fontsize=9)
    ax.set_ylabel("Number of statements")
    ax.set_ylim(0, t[["n_true", "n_false"]].values.max() * 1.15)
    ax.set_title("Label balance per dataset (after cleaning)")
    ax.legend()
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 3: Length and the length shortcut
# ---------------------------------------------------------------------------

def plot_token_lengths(frames, shortcut, name="got_s0_03_token_lengths"):
    """GPT-2 tokens per statement, true vs false, one panel per dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    auc = shortcut[shortcut["measure"] == "token_len"].set_index("dataset")["auroc_best_direction"]
    for ax, (dataset, df) in zip(axes.T.ravel(), frames.items()):
        lo, hi = int(df["token_len"].min()), int(df["token_len"].max())
        for label, label_name, color in [(1, "True", TRUE_COLOR), (0, "False", FALSE_COLOR)]:
            vals = df.loc[df["label"] == label, "token_len"]
            ax.hist(vals, bins=range(lo, hi + 2), alpha=0.55, color=color,
                    label=f"{label_name} (mean {vals.mean():.1f})")
        ax.set_title(f"{dataset}  (length-only AUROC {auc[dataset]:.2f})", fontsize=10)
        ax.set_xlabel("GPT-2 tokens")
        ax.set_ylabel("Statements")
        ax.legend(fontsize=7)
    fig.suptitle("Statement length by label; AUROC near 0.5 = length gives no shortcut")
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 4: Formatting and the last token
# ---------------------------------------------------------------------------

def plot_last_tokens(last, datasets, top_n=3, name="got_s0_04_last_tokens"):
    """Share of statements ending in each of the most common last tokens, per dataset."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#999999"]
    left = [0.0] * len(datasets)
    seen = set()
    for rank in range(top_n + 1):
        widths, labels = [], []
        for dataset in datasets:
            d = last[last["dataset"] == dataset].sort_values("count", ascending=False).reset_index(drop=True)
            if rank < top_n:
                share = d["percent"].iloc[rank] if rank < len(d) else 0.0
                labels.append(d["last_token"].iloc[rank] if rank < len(d) else "")
            else:
                share = 100 - d["percent"].head(top_n).sum()
                labels.append("other")
            widths.append(share)
        label = f"#{rank + 1} most common" if rank < top_n else "all other tokens"
        ax.barh(datasets, widths, left=left, color=colors[rank], label=label if label not in seen else None)
        seen.add(label)
        for y, (w, l, lab) in enumerate(zip(widths, left, labels)):
            if w >= 6:
                ax.text(l + w / 2, y, f"{lab} {w:.0f}%", ha="center", va="center", color="white", fontsize=8)
        left = [a + b for a, b in zip(left, widths)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of statements")
    ax.set_title("Last GPT-2 token of each statement (where activations are read)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 5: Subject structure
# ---------------------------------------------------------------------------

def plot_subjects(summary, name="got_s0_05_subjects"):
    """Left: number of subjects. Right: statements per subject and share with both labels."""
    s = summary.reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = ax1.bar(s["dataset"], s["n_subjects"], color="#8172B2")
    for bar, v in zip(bars, s["n_subjects"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax1.set_title("Distinct subjects per dataset")
    ax1.set_ylabel("Subjects")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    bars = ax2.bar(s["dataset"], s["percent_subjects_with_both_labels"], color="#55A868")
    for bar, v, m in zip(bars, s["percent_subjects_with_both_labels"], s["statements_per_subject_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}%\n({m:.1f}/subj)", ha="center",
                 va="bottom", fontsize=8)
    ax2.set_ylim(0, 115)
    ax2.set_title("Subjects with BOTH a true and a false statement")
    ax2.set_ylabel("% of subjects")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 6: Shortcut cues
# ---------------------------------------------------------------------------

def plot_shortcuts(table, datasets, name="got_s0_06_shortcuts"):
    """AUROC of each surface cue per dataset; dashed line = no shortcut (0.5)."""
    cues = list(dict.fromkeys(table["cue"]))
    colors = ["#999999", "#C44E52", "#4C72B0", "#55A868", "#8172B2"]
    width = 0.8 / len(cues)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, cue in enumerate(cues):
        sub = table[table["cue"] == cue].set_index("dataset")
        xs = [j + (i - (len(cues) - 1) / 2) * width for j, d in enumerate(datasets) if d in sub.index]
        vals = [sub.loc[d, "auroc_best_direction"] for d in datasets if d in sub.index]
        ax.bar(xs, vals, width, color=colors[i % len(colors)], label=cue)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0.65, color="#C44E52", linestyle=":", linewidth=0.8)
    ax.text(len(datasets) - 0.5, 0.655, "high risk", color="#C44E52", fontsize=8, ha="right")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUROC of the cue alone (best direction)")
    ax.set_title("Can a surface cue alone predict true/false? (0.5 = no)")
    ax.legend(fontsize=8, ncol=len(cues), loc="upper left")
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 7: Number properties
# ---------------------------------------------------------------------------

def plot_numbers(by_n1, frames, name="got_s0_07_numbers"):
    """Left: % true by first number. Right: gap between the numbers, true vs false."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for (dataset, part), color in zip(by_n1.groupby("dataset", sort=False), ["#4C72B0", "#C44E52"]):
        ax1.plot(part["n1"], part["percent_true"], marker="o", ms=3, color=color, label=dataset)
    ax1.axhline(50, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("First number (n1)")
    ax1.set_ylabel("% of statements that are true")
    ax1.set_title("The first number alone reveals the label")
    ax1.legend(fontsize=8)

    df = next(iter(frames.values()))
    hi = int(df["abs_diff"].max())
    for label, label_name, color in [(1, "True", TRUE_COLOR), (0, "False", FALSE_COLOR)]:
        vals = df.loc[df["label"] == label, "abs_diff"]
        ax2.hist(vals, bins=range(0, hi + 2), alpha=0.55, color=color, label=f"{label_name} (mean {vals.mean():.1f})")
    ax2.set_xlabel("Gap between the numbers |n1 \u2212 n2|")
    ax2.set_ylabel("Statements")
    ax2.set_title(f"Gap size by label ({next(iter(frames))})")
    ax2.legend(fontsize=8)
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 8: Manual label review
# ---------------------------------------------------------------------------

def plot_review(table, name="got_s0_08_label_review"):
    """Manual review outcome per dataset."""
    t = table[table["dataset"] != "ALL"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = [0] * len(t)
    for answer, color in [("yes", "#55A868"), ("unsure", "#CCB974"), ("no", "#C44E52")]:
        ax.barh(t["dataset"], t[answer], left=left, color=color, label=answer)
        left = [a + b for a, b in zip(left, t[answer])]
    ax.invert_yaxis()
    ax.set_xlabel("Reviewed statements")
    ax.set_title("Manual label review: do you agree with the label?")
    ax.legend(title="Agree?", fontsize=8, loc="lower right")
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 9: Train/test split by subject
# ---------------------------------------------------------------------------

def plot_split(checks, name="got_s0_09_split"):
    """Train and test sizes per dataset, with % true in each."""
    x = range(len(checks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.3))
    bars = [ax.bar([i - width / 2 for i in x], checks["train"], width, color="#4C72B0", label="train"),
            ax.bar([i + width / 2 for i in x], checks["test"], width, color="#DD8452", label="test")]
    for bar_set, col in zip(bars, ["percent_true_train", "percent_true_test"]):
        for bar, pct in zip(bar_set, checks[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}\n{pct:.0f}% T",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(checks["dataset"], fontsize=9)
    ax.set_ylim(0, checks["train"].max() * 1.25)
    ax.set_ylabel("Statements")
    ax.set_title("Subject-level 70/30 split (T = % true)")
    ax.legend()
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 10: Does GPT-2 know these facts?
# ---------------------------------------------------------------------------

def plot_knowledge(summary, name="got_s0_10_gpt2_knowledge"):
    """Left: pair accuracy (paired datasets). Right: statement-level AUROC; numbers also within n1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    paired = summary.dropna(subset=["pair_accuracy"])
    bars = ax1.bar(paired["dataset"], paired["pair_accuracy"], color="#55A868")
    for bar, v in zip(bars, paired["pair_accuracy"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("True version more likely than its false twin")
    ax1.set_ylabel("Pair accuracy (0.5 = chance)")
    ax1.tick_params(axis="x", rotation=20, labelsize=8)

    x = range(len(summary))
    width = 0.38
    ax2.bar([i - width / 2 for i in x], summary["auroc_logprob"], width, color="#4C72B0", label="AUROC")
    within = summary["auroc_within_n1"].fillna(0)
    ax2.bar([i + width / 2 for i in x], within, width, color="#8172B2", label="AUROC within n1 (numbers)")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(summary["dataset"], rotation=20, fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("AUROC (0.5 = no preference)")
    ax2.set_title("GPT-2 sentence likelihood vs label")
    ax2.legend(fontsize=8)
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Stage 2: baseline per dataset and layer
# ---------------------------------------------------------------------------

METHOD_STYLE = {"mass_mean": ("Mass-mean", "#4C72B0"), "logistic": ("Logistic", "#C44E52"),
                "pca": ("PCA (unsupervised)", "#55A868")}
N1_SHORTCUT = 0.841  # first-number-only AUROC from EDA step 7


def plot_baseline(results, band, datasets, read="last", name=None):
    """Test AUROC by layer for each dataset: methods, random band, and the numbers shortcut."""
    name = name or f"got_s2_truth_by_layer_{read}"
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, dataset in zip(axes.T.ravel(), datasets):
        b = band[(band["dataset"] == dataset) & (band["read"] == read)].sort_values("position_index")
        x = b["position_index"].values
        ax.fill_between(x, b["random_low"], b["random_high"], color="#BBBBBB", alpha=0.5, label="Random band")
        for method, (label, color) in METHOD_STYLE.items():
            r = results[(results["dataset"] == dataset) & (results["read"] == read) &
                        (results["method"] == method)].sort_values("position_index")
            ax.plot(r["position_index"], r["auroc"], marker="o", ms=3, color=color, label=label)
            ax.fill_between(r["position_index"], r["auroc_low"], r["auroc_high"], color=color, alpha=0.12)
        if dataset in ("larger_than", "smaller_than"):
            r = results[(results["dataset"] == dataset) & (results["read"] == read) &
                        (results["method"] == "mass_mean")].sort_values("position_index")
            ax.plot(r["position_index"], r["auroc_within_n1"], ":", color="#4C72B0", label="Mass-mean within n1")
            ax.axhline(N1_SHORTCUT, color="#C44E52", linestyle="--", linewidth=0.8, label="First-number shortcut")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(b["position"], fontsize=7)
        ax.set_ylim(0.2, 1.02)
        ax.set_title(dataset, fontsize=10)
    axes[0][0].set_ylabel("Test AUROC")
    axes[1][0].set_ylabel("Test AUROC")
    handles, labels = axes[1][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(f"Truth is linearly readable? (read: {read} token)", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Stage 3: transfer, negation, joint training
# ---------------------------------------------------------------------------

def plot_transfer_heatmap(transfer_df, layers, datasets, method="mass_mean", name="got_s3_transfer_heatmap"):
    """Fitted on (rows) x tested on (columns), AUROC averaged over the given layers. Red < 0.5 = flipped."""
    import numpy as np

    t = transfer_df[(transfer_df["method"] == method) & (transfer_df["position"].isin(layers))]
    m = t.pivot_table(index="source", columns="target", values="auroc", aggfunc="mean").loc[datasets, datasets]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Fitted on")
    ax.set_title(f"Cross-dataset transfer ({method}, mean AUROC {layers[0]}\u2013{layers[-1]}); < 0.5 = flipped")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test AUROC")
    return save_figure(fig, name)


def plot_negation(transfer_df, method="mass_mean", name="got_s3_negation_transfer"):
    """Does a direction survive negation / reversal? Fitted on one twin, tested on the other, by layer."""
    pairs = [("cities", "neg_cities"), ("sp_en_trans", "neg_sp_en_trans"), ("larger_than", "smaller_than")]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (a, b), color in zip(pairs, colors):
        for src, tgt, style in [(a, b, "-"), (b, a, "--")]:
            r = transfer_df[(transfer_df["source"] == src) & (transfer_df["target"] == tgt) &
                            (transfer_df["method"] == method)].sort_values("position_index")
            ax.plot(r["position_index"], r["auroc"], style, marker="o", ms=3, color=color, label=f"{src} \u2192 {tgt}")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(r["position_index"])
    ax.set_xticklabels(r["position"], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUROC (below 0.5 = direction flipped)")
    ax.set_title(f"Negation / reversal transfer ({method})")
    ax.legend(fontsize=7, ncol=3, loc="lower center")
    return save_figure(fig, name)