"""Plotting functions for the RepE EDA. Each one saves a PNG and returns its path."""

import matplotlib

matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt

from src.eda.load_data import FIGURES_DIR, PROJECT_ROOT


def save_figure(fig, name):
    """Save a figure to results/figures/eda/<name>.png and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Step 2: Label balance and usable pairs
# ---------------------------------------------------------------------------

def plot_label_balance(n_raw, n_clean, n_true, n_false, name="repe_02_label_balance"):
    """Left: true vs false counts. Right: how the raw file becomes analysis prompts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: label balance
    bars = ax1.bar(["True (1)", "False (0)"], [n_true, n_false], color=["#4C72B0", "#DD8452"])
    for bar, count in zip(bars, [n_true, n_false]):
        pct = 100 * count / (n_true + n_false)
        ax1.text(bar.get_x() + bar.get_width() / 2, count, f"{count}\n({pct:.1f}%)",
                 ha="center", va="bottom")
    ax1.set_title("Label balance (clean dataset)")
    ax1.set_ylabel("Number of statements")
    ax1.set_ylim(0, max(n_true, n_false) * 1.2)

    # Right panel: from raw rows to prompts
    stages = ["Raw rows", "After cleaning", "True statements\n= pairs", "Prompts\n(2 per pair)"]
    values = [n_raw, n_clean, n_true, 2 * n_true]
    bars = ax2.bar(stages, values, color=["#999999", "#55A868", "#4C72B0", "#8172B2"])
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    ax2.set_title("From raw file to honest/untruthful prompts")
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, max(values) * 1.15)

    return save_figure(fig, name)



# ---------------------------------------------------------------------------
# Step 3: Length statistics
# ---------------------------------------------------------------------------

def plot_length_distribution(df, name="repe_03_length_distribution"):
    """Histograms of character, word and GPT-2 token counts, true vs false."""
    measures = [("char_len", "Characters"), ("word_len", "Words"), ("token_len", "GPT-2 tokens")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (col, title) in zip(axes, measures):
        lo, hi = df[col].min(), df[col].max()
        bins = min(30, int(hi - lo) + 1)
        for label, name_, color in [(1, "True", "#4C72B0"), (0, "False", "#DD8452")]:
            values = df.loc[df["label"] == label, col]
            ax.hist(values, bins=bins, range=(lo, hi + 1), alpha=0.55, color=color,
                    label=f"{name_} (mean {values.mean():.1f})")
            ax.axvline(values.mean(), color=color, linestyle="--", linewidth=1.2)
        ax.set_title(f"{title} per statement")
        ax.set_xlabel(title)
        ax.set_ylabel("Number of statements")
        ax.legend(fontsize=8)

    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 4: Formatting and the last token
# ---------------------------------------------------------------------------

def plot_last_tokens(table, top_n=10, name="repe_04_last_tokens"):
    """Most common final GPT-2 tokens, split into true and false statements."""
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(top) + 1)))
    ax.barh(top["last_token"], top["count_true"], color="#4C72B0", label="True")
    ax.barh(top["last_token"], top["count_false"], left=top["count_true"],
            color="#DD8452", label="False")
    for y, (total, pct) in enumerate(zip(top["count"], top["percent"])):
        ax.text(total, y, f" {total} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_title(f"Last GPT-2 token of each statement (top {len(top)})")
    ax.set_xlabel("Number of statements")
    ax.set_xlim(0, top["count"].max() * 1.25)
    ax.legend(loc="lower right")
    return save_figure(fig, name)



# ---------------------------------------------------------------------------
# Step 5: Honest/untruthful pair construction checks
# ---------------------------------------------------------------------------

def plot_pair_lengths(check, name="repe_05_pair_lengths"):
    """Left: prompt length in tokens, honest vs untruthful. Right: length difference per pair."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    lo = int(check[["honest_tokens", "untruthful_tokens"]].min().min())
    hi = int(check[["honest_tokens", "untruthful_tokens"]].max().max())
    bins = range(lo, hi + 2)
    ax1.hist(check["honest_tokens"], bins=bins, alpha=0.55, color="#4C72B0",
             label=f"Honest (mean {check['honest_tokens'].mean():.1f})")
    ax1.hist(check["untruthful_tokens"], bins=bins, alpha=0.55, color="#C44E52",
             label=f"Untruthful (mean {check['untruthful_tokens'].mean():.1f})")
    ax1.set_title("Prompt length in GPT-2 tokens")
    ax1.set_xlabel("Tokens per prompt")
    ax1.set_ylabel("Number of prompts")
    ax1.legend(fontsize=8)

    counts = check["length_difference"].value_counts().sort_index()
    bars = ax2.bar([str(d) for d in counts.index], counts.values, color="#8172B2")
    for bar, value in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    ax2.set_title("Untruthful minus honest length, per pair")
    ax2.set_xlabel("Extra tokens in the untruthful prompt")
    ax2.set_ylabel("Number of pairs")
    ax2.set_ylim(0, counts.max() * 1.15)

    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 6: Word-level cues (possible shortcuts)
# ---------------------------------------------------------------------------

def plot_word_cues(top_words, categories, name="repe_06_word_cues"):
    """Left: words most typical of true vs false. Right: cue categories by label."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(5, 0.28 * len(top_words) + 1)),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    words = top_words.sort_values("z")
    colors = ["#4C72B0" if z > 0 else "#DD8452" for z in words["z"]]
    ax1.barh(words["word"], words["z"], color=colors)
    ax1.axvline(0, color="black", linewidth=0.8)
    for x in (-1.96, 1.96):
        ax1.axvline(x, color="grey", linestyle=":", linewidth=1)
    ax1.set_title("Words most typical of true (blue) vs false (orange)")
    ax1.set_xlabel("z-score of log-odds (dotted lines = \u00b11.96)")

    x = range(len(categories))
    width = 0.38
    ax2.bar([i - width / 2 for i in x], categories["percent_true"], width, color="#4C72B0", label="True")
    ax2.bar([i + width / 2 for i in x], categories["percent_false"], width, color="#DD8452", label="False")
    for i, p in zip(x, categories["fisher_p"]):
        top = max(categories["percent_true"].iloc[i], categories["percent_false"].iloc[i])
        ax2.text(i, top + 0.5, f"p={p:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([c.replace("_", "\n") for c in categories["category"]])
    ax2.set_ylabel("% of statements containing the cue")
    ax2.set_title("Cue categories by label")
    ax2.legend()

    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 7: Topic coverage
# ---------------------------------------------------------------------------

def plot_topics(table, name="repe_07_topics"):
    """Statements per topic, stacked by label, with % true written on each bar."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(t) + 1)))
    ax.barh(t["topic"], t["count_true"], color="#4C72B0", label="True")
    ax.barh(t["topic"], t["count_false"], left=t["count_true"], color="#DD8452", label="False")
    for y, (total, pct) in enumerate(zip(t["count"], t["percent_true_in_topic"])):
        ax.text(total, y, f" {total}  ({pct:.0f}% true)", va="center", fontsize=9)
    ax.set_title("Statements per topic (keyword-based)")
    ax.set_xlabel("Number of statements")
    ax.set_xlim(0, t["count"].max() * 1.35)
    ax.legend(loc="lower right")
    return save_figure(fig, name)



# ---------------------------------------------------------------------------
# Step 8: Label quality
# ---------------------------------------------------------------------------

def plot_label_review(summary, name="repe_08_label_review"):
    """Manual review outcome: do you agree with the dataset's label?"""
    s = summary[summary["group"] != "all"]
    groups = ["true", "false"]
    answers = [("yes", "#55A868"), ("unsure", "#CCB974"), ("no", "#C44E52")]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    left = [0, 0]
    for answer, color in answers:
        vals = [int(s[(s["group"] == g) & (s["answer"] == answer)]["count"].sum()) for g in groups]
        ax.barh(["Labeled true", "Labeled false"], vals, left=left, color=color, label=answer)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=9)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_title("Manual label review: do you agree with the label?")
    ax.set_xlabel("Number of reviewed statements")
    ax.legend(title="Agree?", loc="lower right")
    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 9: Train/test split
# ---------------------------------------------------------------------------

def plot_split_balance(summary, topics, split_df, name="repe_09_split_balance"):
    """Labels, topics and lengths in train vs test."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [0.8, 1.4, 1]})
    colors = {"train": "#4C72B0", "test": "#DD8452"}

    ax = axes[0]
    x = range(2)
    width = 0.38
    for i, split_name in enumerate(["train", "test"]):
        row = summary[summary["split"] == split_name].iloc[0]
        vals = [row["n_true"], row["n_false"]]
        bars = ax.bar([j + (i - 0.5) * width for j in x], vals, width, color=colors[split_name],
                      label=f"{split_name} ({row['n_pairs']} pairs)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["True", "False"])
    ax.set_title("Statements per split")
    ax.set_ylim(0, summary[["n_true", "n_false"]].values.max() * 1.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    t = topics.iloc[::-1]
    y = range(len(t))
    ax.barh([i + 0.2 for i in y], t["percent_train"], 0.4, color=colors["train"], label="train")
    ax.barh([i - 0.2 for i in y], t["percent_test"], 0.4, color=colors["test"], label="test")
    ax.set_yticks(list(y))
    ax.set_yticklabels(t["topic"])
    ax.set_xlabel("% of split")
    ax.set_title("Topic mix per split")
    ax.legend(fontsize=8)

    ax = axes[2]
    lo, hi = split_df["token_len"].min(), split_df["token_len"].max()
    for split_name in ["train", "test"]:
        vals = split_df.loc[split_df["split"] == split_name, "token_len"]
        ax.hist(vals, bins=range(lo, hi + 2), density=True, alpha=0.55, color=colors[split_name],
                label=f"{split_name} (mean {vals.mean():.1f})")
    ax.set_title("Statement length per split")
    ax.set_xlabel("GPT-2 tokens")
    ax.set_ylabel("Share of statements")
    ax.legend(fontsize=8)

    return save_figure(fig, name)


# ---------------------------------------------------------------------------
# Step 10: Does GPT-2 know these facts?
# ---------------------------------------------------------------------------

def plot_gpt2_knowledge(scored, summary, name="repe_10_gpt2_knowledge"):
    """Left: GPT-2 scores for true vs false TEST statements. Right: test AUROC by method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    part = scored[(scored["format"] == "few_shot") & (scored["split"] == "test")]
    lo, hi = part["score"].min(), part["score"].max()
    for label, label_name, color in [(1, "True", "#4C72B0"), (0, "False", "#DD8452")]:
        vals = part.loc[part["label"] == label, "score"]
        ax1.hist(vals, bins=25, range=(lo, hi), alpha=0.55, color=color,
                 label=f"{label_name} (mean {vals.mean():.2f})")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("GPT-2 score on test statements (few-shot)")
    ax1.set_xlabel("log P(' True') \u2212 log P(' False')")
    ax1.set_ylabel("Number of statements")
    ax1.legend(fontsize=8)

    test = summary[summary["split"] == "test"].set_index("format")
    order = [f for f in ["length_only", "zero_shot", "few_shot"] if f in test.index]
    colors = {"length_only": "#999999", "zero_shot": "#8172B2", "few_shot": "#55A868"}
    bars = ax2.bar(order, test.loc[order, "auroc"], color=[colors[f] for f in order])
    for bar, v in zip(bars, test.loc[order, "auroc"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Test AUROC")
    ax2.set_title("Can GPT-2 tell true from false?")
    ax2.legend(fontsize=8)

    return save_figure(fig, name)