"""Analysis functions for the RepE EDA. Each section matches one EDA step."""

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Data integrity
# ---------------------------------------------------------------------------

def normalize_statement(text):
    """Lowercase, trim, collapse spaces, drop trailing punctuation.

    Two statements with the same normalized form are near-duplicates,
    e.g. "The sky is blue." and "the sky is blue".
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".!? ")


def integrity_report(df):
    """Check structure, missing values, labels and duplicates.

    Returns (summary_table, duplicates_table).
    """
    statements = df["statement"] if "statement" in df.columns else pd.Series(dtype=str)
    labels = df["label"] if "label" in df.columns else pd.Series(dtype=float)

    is_empty = statements.astype(str).str.strip().isin(["", "nan"])
    valid_labels = labels.isin([0, 1])

    norm = statements.dropna().map(normalize_statement)
    exact_dup = statements.duplicated(keep=False) & statements.notna()
    near_dup = norm.duplicated(keep=False).reindex(df.index, fill_value=False)

    # Near-duplicate groups whose copies carry DIFFERENT labels (contradictions).
    groups = pd.DataFrame({"norm": norm, "label": labels}).dropna()
    label_counts = groups.groupby("norm")["label"].nunique()
    conflicting_norms = set(label_counts[label_counts > 1].index)

    summary = pd.DataFrame([
        ("n_rows", len(df)),
        ("columns", ", ".join(df.columns)),
        ("dtypes", ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())),
        ("missing_statement", int(statements.isna().sum())),
        ("missing_label", int(labels.isna().sum())),
        ("empty_statement", int((is_empty & statements.notna()).sum())),
        ("invalid_label_values", int((~valid_labels & labels.notna()).sum())),
        ("unique_label_values", ", ".join(map(str, sorted(labels.dropna().unique())))),
        ("exact_duplicate_rows", int(exact_dup.sum())),
        ("near_duplicate_rows", int(near_dup.sum())),
        ("conflicting_label_groups", len(conflicting_norms)),
    ], columns=["check", "value"])

    dup_rows = df[near_dup].copy()
    dup_rows["normalized"] = norm[near_dup]
    dup_rows["exact_duplicate"] = exact_dup[near_dup]
    dup_rows["conflicting_labels"] = dup_rows["normalized"].isin(conflicting_norms)
    dup_rows = dup_rows.sort_values(["normalized", "label"]).reset_index(names="original_row")

    return summary, dup_rows


def clean_dataset(df):
    """Return a clean copy plus a log of what was removed and why.

    Rules, applied in order:
      1. drop rows with a missing/empty statement or a label that is not 0/1
      2. drop every copy of statements whose near-duplicates have conflicting labels
      3. keep the first copy of remaining near-duplicates
    """
    log = []
    out = df.copy()
    out["statement"] = out["statement"].astype(str).str.strip()

    bad = out["statement"].isin(["", "nan"]) | df["statement"].isna() | ~out["label"].isin([0, 1])
    log.append(("missing_or_invalid", int(bad.sum())))
    out = out[~bad].copy()
    out["label"] = out["label"].astype(int)

    out["normalized"] = out["statement"].map(normalize_statement)
    n_labels = out.groupby("normalized")["label"].transform("nunique")
    conflict = n_labels > 1
    log.append(("conflicting_label_duplicates", int(conflict.sum())))
    out = out[~conflict]

    dup = out["normalized"].duplicated(keep="first")
    log.append(("redundant_duplicates", int(dup.sum())))
    out = out[~dup]

    out = out[["statement", "label"]].reset_index(drop=True)
    log.append(("rows_kept", len(out)))
    return out, pd.DataFrame(log, columns=["step", "rows"])




# ---------------------------------------------------------------------------
# Step 2: Label balance and usable pairs
# ---------------------------------------------------------------------------

def label_balance(df):
    """Count true/false statements and how many honest/untruthful pairs they give.

    Only TRUE statements become pairs (one honest prompt + one untruthful
    prompt each), so n_true is the real size of the honesty analysis.
    """
    n = len(df)
    n_true = int((df["label"] == 1).sum())
    n_false = int((df["label"] == 0).sum())

    table = pd.DataFrame([
        ("total_statements", n, 100.0),
        ("true_statements", n_true, round(100 * n_true / n, 2)),
        ("false_statements", n_false, round(100 * n_false / n, 2)),
        ("true_to_false_ratio", round(n_true / n_false, 3) if n_false else None, None),
        ("usable_pairs", n_true, None),
        ("total_prompts", 2 * n_true, None),
        ("train_pairs_70pct", int(round(0.7 * n_true)), None),
        ("test_pairs_30pct", n_true - int(round(0.7 * n_true)), None),
    ], columns=["metric", "value", "percent"], dtype=object)
    return table



# ---------------------------------------------------------------------------
# Step 3: Length statistics
# ---------------------------------------------------------------------------

LENGTH_COLUMNS = ["char_len", "word_len", "token_len"]


def load_gpt2_tokenizer():
    """GPT-2's own tokenizer, so token counts match what the model sees.

    Downloads a small file the first time (needs internet once).
    """
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained("gpt2")


def add_length_columns(df, tokenizer):
    """Add character, word and GPT-2 token counts for each statement."""
    df = df.copy()
    df["char_len"] = df["statement"].str.len()
    df["word_len"] = df["statement"].str.split().str.len()
    df["token_len"] = [len(tokenizer.encode(s)) for s in df["statement"]]
    return df


def length_stats(df):
    """Mean, spread and range of each length measure: overall and per label."""
    rows = []
    groups = [("all", df), ("true", df[df["label"] == 1]), ("false", df[df["label"] == 0])]
    for group_name, part in groups:
        for col in LENGTH_COLUMNS:
            s = part[col]
            rows.append({
                "group": group_name, "measure": col, "count": len(s),
                "mean": round(s.mean(), 2), "std": round(s.std(), 2),
                "min": int(s.min()), "q25": s.quantile(0.25), "median": s.median(),
                "q75": s.quantile(0.75), "max": int(s.max()),
            })
    return pd.DataFrame(rows)


def length_shortcut_test(df):
    """Can length ALONE predict the label? If yes, a probe could cheat with it.

    auroc: 0.5 = length says nothing about truth; far from 0.5 = shortcut risk.
    auroc_best_direction folds values below 0.5 upward, so 0.5 to 1.0 always.
    p_value: Mann-Whitney U test of whether true and false lengths differ.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    rows = []
    for col in LENGTH_COLUMNS:
        auc = roc_auc_score(df["label"], df[col])
        p = mannwhitneyu(df.loc[df["label"] == 1, col], df.loc[df["label"] == 0, col]).pvalue
        best = max(auc, 1 - auc)
        rows.append({
            "measure": col,
            "mean_true": round(df.loc[df["label"] == 1, col].mean(), 2),
            "mean_false": round(df.loc[df["label"] == 0, col].mean(), 2),
            "auroc": round(auc, 3),
            "auroc_best_direction": round(best, 3),
            "mannwhitney_p": round(p, 4),
            "shortcut_risk": "high" if best >= 0.65 else "moderate" if best >= 0.58 else "low",
        })
    return pd.DataFrame(rows)


def length_extremes(df, n=5):
    """The n shortest and n longest statements by GPT-2 tokens, for a quick read."""
    shortest = df.nsmallest(n, "token_len").assign(position="shortest")
    longest = df.nlargest(n, "token_len").assign(position="longest")
    cols = ["position", "token_len", "word_len", "char_len", "label", "statement"]
    return pd.concat([shortest, longest])[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Formatting and the last token
# ---------------------------------------------------------------------------

FORMAT_CHECKS = {
    "ends_with_period": lambda s: s.endswith("."),
    "ends_with_other_punct": lambda s: s[-1:] in {"!", "?", ";", ":", ","},
    "ends_without_punct": lambda s: s[-1:].isalnum(),
    "has_quotes": lambda s: any(q in s for q in ['"', "\u201c", "\u201d"]),
    "has_non_ascii": lambda s: any(ord(c) > 127 for c in s),
    "has_double_space": lambda s: "  " in s,
    "starts_lowercase": lambda s: s[:1].islower(),
    "has_digits": lambda s: any(c.isdigit() for c in s),
}


def formatting_checks(df):
    """Count each formatting property, overall and per label.

    Returns (summary_table, rows_with_issues). 'Issues' are anything except
    ending in a period, having digits, or a normal capital first letter.
    """
    flags = pd.DataFrame({name: df["statement"].map(fn) for name, fn in FORMAT_CHECKS.items()})
    true, false = df["label"] == 1, df["label"] == 0

    summary = pd.DataFrame({
        "check": flags.columns,
        "count_all": flags.sum().values,
        "count_true": flags[true].sum().values,
        "count_false": flags[false].sum().values,
        "percent_all": (100 * flags.mean()).round(2).values,
    })

    issue_cols = [c for c in flags.columns if c not in {"ends_with_period", "has_digits"}]
    has_issue = flags[issue_cols].any(axis=1)
    issues = df[has_issue].copy()
    issues["issues"] = flags[has_issue][issue_cols].apply(
        lambda row: ", ".join(c for c in issue_cols if row[c]), axis=1)
    return summary, issues.reset_index(names="row")


def last_token_table(df, tokenizer):
    """Which GPT-2 token each statement ends on, with counts per label.

    We read activations at the LAST token, so ideally it is the same kind of
    token (a period) for every statement.
    """
    last = [tokenizer.decode(tokenizer.encode(s)[-1:]) for s in df["statement"]]
    tmp = pd.DataFrame({"last_token": last, "label": df["label"].values})
    table = (tmp.groupby("last_token")["label"]
             .agg(count="size", count_true="sum")
             .reset_index())
    table["count_false"] = table["count"] - table["count_true"]
    table["percent"] = (100 * table["count"] / len(tmp)).round(2)
    table["last_token"] = table["last_token"].map(repr)  # show spaces/newlines clearly
    return table.sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: Honest/untruthful pair construction checks
# ---------------------------------------------------------------------------

# RepE persona template, written as plain text for GPT-2 (no chat tags).
# Check the exact wording against examples/honesty/utils.py in the RepE repo.
PROMPT_TEMPLATE = "Pretend you're {persona} person making statements about the world. {statement}"
PERSONAS = {"honest": "an honest", "untruthful": "an untruthful"}


def build_pairs(df):
    """Wrap every TRUE statement in an honest and an untruthful prompt.

    One row per prompt; both prompts of a statement share a pair_id.
    label = 1 for honest, 0 for untruthful.
    """
    rows = []
    true_statements = df.loc[df["label"] == 1, "statement"].reset_index(drop=True)
    for pair_id, statement in enumerate(true_statements):
        for persona, label in [("honest", 1), ("untruthful", 0)]:
            rows.append({
                "pair_id": pair_id,
                "persona": persona,
                "label": label,
                "statement": statement,
                "prompt": PROMPT_TEMPLATE.format(persona=PERSONAS[persona], statement=statement),
            })
    return pd.DataFrame(rows)


def persona_tokenization(tokenizer):
    """How GPT-2 splits the two persona words into tokens."""
    rows = []
    for persona, text in PERSONAS.items():
        ids = tokenizer.encode(" " + text.split()[-1])
        rows.append({
            "persona": persona,
            "word": text.split()[-1],
            "n_tokens": len(ids),
            "tokens": " | ".join(repr(tokenizer.decode([i])) for i in ids),
        })
    return pd.DataFrame(rows)


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pair_token_check(pairs, tokenizer):
    """Per pair: token lengths, and proof the prompts differ ONLY in the persona.

    differing_tokens_* = tokens left after removing the shared start and end;
    statement_identical = the statement's tokens are the same in both prompts.
    """
    rows = []
    for pair_id, group in pairs.groupby("pair_id"):
        honest = tokenizer.encode(group.loc[group["persona"] == "honest", "prompt"].iloc[0])
        untruth = tokenizer.encode(group.loc[group["persona"] == "untruthful", "prompt"].iloc[0])
        statement_ids = tokenizer.encode(" " + group["statement"].iloc[0])

        prefix = _common_prefix(honest, untruth)
        suffix = _common_prefix(honest[::-1], untruth[::-1])
        suffix = min(suffix, min(len(honest), len(untruth)) - prefix)

        rows.append({
            "pair_id": pair_id,
            "honest_tokens": len(honest),
            "untruthful_tokens": len(untruth),
            "length_difference": len(untruth) - len(honest),
            "shared_prefix_tokens": prefix,
            "shared_suffix_tokens": suffix,
            "differing_tokens_honest": len(honest) - prefix - suffix,
            "differing_tokens_untruthful": len(untruth) - prefix - suffix,
            "statement_identical": honest[-len(statement_ids):] == untruth[-len(statement_ids):],
            "same_last_token": honest[-1] == untruth[-1],
        })
    return pd.DataFrame(rows)


def pair_summary(check):
    """One-row-per-fact summary of the pair check."""
    return pd.DataFrame([
        ("n_pairs", len(check)),
        ("n_prompts", 2 * len(check)),
        ("mean_prompt_tokens_honest", round(check["honest_tokens"].mean(), 2)),
        ("mean_prompt_tokens_untruthful", round(check["untruthful_tokens"].mean(), 2)),
        ("min_prompt_tokens", int(check[["honest_tokens", "untruthful_tokens"]].min().min())),
        ("max_prompt_tokens", int(check[["honest_tokens", "untruthful_tokens"]].max().max())),
        ("length_difference_values", ", ".join(map(str, sorted(check["length_difference"].unique())))),
        ("pairs_with_equal_length", int((check["length_difference"] == 0).sum())),
        ("pairs_statement_identical", int(check["statement_identical"].sum())),
        ("pairs_same_last_token", int(check["same_last_token"].sum())),
        ("max_differing_tokens", int(check[["differing_tokens_honest", "differing_tokens_untruthful"]].max().max())),
    ], columns=["check", "value"], dtype=object)


# ---------------------------------------------------------------------------
# Step 6: Word-level cues (possible shortcuts)
# ---------------------------------------------------------------------------

CUE_WORDS = {
    "negation": {"not", "no", "never", "none", "nothing", "neither", "nor", "cannot",
                 "isn't", "aren't", "doesn't", "don't", "can't", "won't", "wasn't"},
    "absolute_superlative": {"all", "always", "every", "only", "entirely", "completely",
                             "largest", "smallest", "biggest", "highest", "lowest",
                             "longest", "shortest", "fastest", "most", "least", "best", "worst"},
    "hedge": {"can", "may", "might", "often", "usually", "some", "sometimes", "typically", "generally"},
}


def words_of(text):
    """Lowercase words, keeping apostrophes (e.g. isn't)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


def distinctive_words(df, min_count=5, top_n=15, alpha=0.5):
    """Words most typical of true vs false statements (smoothed log-odds, z-score).

    Positive z = more typical of TRUE statements; negative = of FALSE ones.
    |z| > 1.96 is roughly 'unlikely to be chance'.
    """
    import numpy as np
    from collections import Counter

    true_counts = Counter(w for s in df.loc[df["label"] == 1, "statement"] for w in words_of(s))
    false_counts = Counter(w for s in df.loc[df["label"] == 0, "statement"] for w in words_of(s))
    vocab = [w for w in set(true_counts) | set(false_counts)
             if true_counts[w] + false_counts[w] >= min_count]
    n_true, n_false = sum(true_counts.values()), sum(false_counts.values())
    v = len(vocab)

    rows = []
    for w in vocab:
        ct, cf = true_counts[w] + alpha, false_counts[w] + alpha
        log_odds = np.log(ct / (n_true + alpha * v)) - np.log(cf / (n_false + alpha * v))
        z = log_odds / np.sqrt(1 / ct + 1 / cf)
        rows.append({"word": w, "count_true": true_counts[w], "count_false": false_counts[w],
                     "log_odds": round(log_odds, 3), "z": round(z, 2)})
    table = pd.DataFrame(rows).sort_values("z", ascending=False)
    top = pd.concat([table.head(top_n).assign(leans="true"),
                     table.tail(top_n).iloc[::-1].assign(leans="false")])
    return table.reset_index(drop=True), top.reset_index(drop=True)


def cue_categories(df):
    """How often negations, absolutes/superlatives, hedges and numbers appear, by label.

    fisher_p tests whether the category is more common in one label.
    """
    from scipy.stats import fisher_exact

    word_sets = df["statement"].map(lambda s: set(words_of(s)))
    flags = {name: word_sets.map(lambda ws, cue=cue: bool(ws & cue)) for name, cue in CUE_WORDS.items()}
    flags["number"] = df["statement"].str.contains(r"\d")

    true, false = df["label"] == 1, df["label"] == 0
    rows = []
    for name, flag in flags.items():
        a, b = int(flag[true].sum()), int(flag[false].sum())
        _, p = fisher_exact([[a, true.sum() - a], [b, false.sum() - b]])
        rows.append({
            "category": name,
            "count_true": a, "count_false": b,
            "percent_true": round(100 * a / true.sum(), 2),
            "percent_false": round(100 * b / false.sum(), 2),
            "fisher_p": round(p, 4),
            "leans": "true" if a / true.sum() > b / false.sum() else "false",
        })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Step 7: Topic coverage
# ---------------------------------------------------------------------------

# Simple keyword rules. A statement gets the FIRST topic whose keywords it contains,
# so more specific topics are listed first. Unmatched statements become "other".
TOPIC_KEYWORDS = {
    "space_astronomy": {"planet", "planets", "sun", "moon", "star", "stars", "galaxy", "solar",
                        "orbit", "orbits", "universe", "space", "mars", "jupiter", "venus",
                        "saturn", "mercury", "astronaut", "earth's"},
    "human_body_health": {"human", "humans", "body", "blood", "heart", "brain", "bones", "bone",
                          "skin", "lungs", "organ", "disease", "health", "vitamin", "muscles",
                          "teeth", "cells", "cell", "dna", "immune"},
    "animals": {"animal", "animals", "mammal", "mammals", "bird", "birds", "fish", "insect",
                "insects", "dog", "dogs", "cat", "cats", "species", "elephant", "elephants",
                "whale", "whales", "shark", "sharks", "lion", "lions", "penguins", "bees", "snake"},
    "plants_nature": {"plant", "plants", "tree", "trees", "flower", "flowers", "forest",
                      "photosynthesis", "leaves", "seeds", "rainforest"},
    "history_people": {"invented", "discovered", "war", "president", "king", "queen", "century",
                       "ancient", "empire", "wrote", "painted", "founded", "history", "born"},
    "technology": {"computer", "computers", "internet", "phone", "technology", "machine",
                   "engine", "software", "electric"},
    "food": {"food", "foods", "fruit", "fruits", "vegetable", "vegetables", "eat", "eating",
             "diet", "sugar", "milk", "coffee", "chocolate"},
    "geography": {"country", "countries", "city", "cities", "capital", "continent", "ocean",
                  "oceans", "river", "rivers", "mountain", "mountains", "desert", "located",
                  "island", "lake", "sea", "africa", "asia", "europe", "america", "australia"},
    "physics_chemistry": {"water", "temperature", "degrees", "celsius", "fahrenheit", "boils",
                          "freezes", "energy", "light", "sound", "gravity", "atom", "atoms",
                          "element", "elements", "chemical", "oxygen", "hydrogen", "carbon",
                          "gas", "metal", "speed", "electricity", "magnetic", "molecules"},
}


def assign_topic(statement):
    """First matching topic for a statement, or 'other'."""
    ws = set(words_of(statement))
    for topic, keywords in TOPIC_KEYWORDS.items():
        if ws & keywords:
            return topic
    return "other"


def topic_table(df):
    """Topic counts and label balance per topic, plus a chi-square test of topic vs label.

    Returns (per_topic_table, assignments, chi_square_result).
    """
    from scipy.stats import chi2_contingency

    assignments = df.copy()
    assignments["topic"] = assignments["statement"].map(assign_topic)

    table = (assignments.groupby("topic")["label"]
             .agg(count="size", count_true="sum").reset_index())
    table["count_false"] = table["count"] - table["count_true"]
    table["percent_of_data"] = (100 * table["count"] / len(assignments)).round(2)
    table["percent_true_in_topic"] = (100 * table["count_true"] / table["count"]).round(2)
    table = table.sort_values("count", ascending=False).reset_index(drop=True)

    chi2, p, dof, _ = chi2_contingency(table[["count_true", "count_false"]].values)
    chi = pd.DataFrame([("chi2", round(chi2, 3)), ("dof", int(dof)), ("p_value", round(p, 4))],
                       columns=["statistic", "value"], dtype=object)
    return table, assignments[["topic", "label", "statement"]], chi



# ---------------------------------------------------------------------------
# Step 8: Label quality
# ---------------------------------------------------------------------------

AMBIGUITY_WORDS = {
    "hedge": CUE_WORDS["hedge"],
    "opinion": {"best", "worst", "beautiful", "important", "good", "bad", "great", "better",
                "worse", "favorite", "popular", "interesting", "healthy", "dangerous"},
    "vague_quantity": {"many", "most", "few", "several", "some", "lots", "often", "rarely", "commonly"},
}


def review_sample(df, n_per_label=15, seed=0):
    """Random sample for MANUAL review: n true + n false, with empty columns to fill.

    Fill 'agree_with_label' with yes / no / unsure, and add a short 'note'.
    """
    sample = pd.concat([
        df[df["label"] == 1].sample(n=min(n_per_label, (df["label"] == 1).sum()), random_state=seed),
        df[df["label"] == 0].sample(n=min(n_per_label, (df["label"] == 0).sum()), random_state=seed),
    ]).sample(frac=1, random_state=seed)  # shuffle so true/false are mixed
    sample = sample.reset_index(names="row")
    sample["agree_with_label"] = ""
    sample["note"] = ""
    return sample[["row", "label", "statement", "agree_with_label", "note"]]


def summarize_review(review):
    """Summary of a filled review file. Returns None if nothing is filled in yet."""
    answers = review["agree_with_label"].fillna("").astype(str).str.strip().str.lower()
    filled = answers.isin(["yes", "no", "unsure"])
    if not filled.any():
        return None
    rows = []
    for group, mask in [("all", filled), ("true", filled & (review["label"] == 1)),
                        ("false", filled & (review["label"] == 0))]:
        n = int(mask.sum())
        for answer in ["yes", "no", "unsure"]:
            k = int((answers[mask] == answer).sum())
            rows.append({"group": group, "answer": answer, "count": k,
                         "percent": round(100 * k / n, 2) if n else None})
    return pd.DataFrame(rows)


def ambiguity_flags(df):
    """Statements containing hedges, opinion words or vague quantities.

    These are not necessarily wrong, but their true/false label is easier to dispute.
    """
    word_sets = df["statement"].map(lambda s: set(words_of(s)))
    out = df.copy()
    for name, words in AMBIGUITY_WORDS.items():
        out[name] = word_sets.map(lambda ws, w=words: ", ".join(sorted(ws & w)))
    flagged = out[(out[list(AMBIGUITY_WORDS)] != "").any(axis=1)]

    summary = pd.DataFrame([{
        "flag": name,
        "count_true": int(((out[name] != "") & (out["label"] == 1)).sum()),
        "count_false": int(((out[name] != "") & (out["label"] == 0)).sum()),
    } for name in AMBIGUITY_WORDS])
    summary["count_all"] = summary["count_true"] + summary["count_false"]
    summary["percent_all"] = (100 * summary["count_all"] / len(df)).round(2)
    return summary, flagged.reset_index(names="row")


# ---------------------------------------------------------------------------
# Step 9: Train/test split
# ---------------------------------------------------------------------------

def make_split(df, test_size=0.3, seed=0, min_group=6):
    """Split STATEMENTS 70/30, stratified by label and topic.

    Near-duplicates were already removed in step 1, so no statement can sit in
    both sets. Honesty pairs inherit their statement's split, which keeps both
    prompts of a pair together. Rare topics are pooled as 'other' for stratifying.
    """
    from sklearn.model_selection import train_test_split

    out = df.copy()
    out["topic"] = out["statement"].map(assign_topic)
    strat = out["label"].astype(str) + "_" + out["topic"]
    counts = strat.value_counts()
    rare = strat.map(counts) < min_group
    strat[rare] = out.loc[rare, "label"].astype(str) + "_pooled"

    train_idx, test_idx = train_test_split(out.index, test_size=test_size,
                                           random_state=seed, stratify=strat)
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def split_pairs(pairs, split_df):
    """Give every honesty prompt the split of its statement."""
    lookup = split_df.set_index("statement")["split"]
    out = pairs.copy()
    out["split"] = out["statement"].map(lookup)
    return out


def split_summary(split_df, pair_split):
    """Size and balance of each split, for statements and for honesty pairs."""
    rows = []
    for name in ["train", "test"]:
        part = split_df[split_df["split"] == name]
        pairs = pair_split[pair_split["split"] == name]
        rows.append({
            "split": name,
            "n_statements": len(part),
            "n_true": int((part["label"] == 1).sum()),
            "n_false": int((part["label"] == 0).sum()),
            "percent_true": round(100 * (part["label"] == 1).mean(), 2),
            "n_pairs": pairs["pair_id"].nunique(),
            "n_prompts": len(pairs),
            "mean_token_len": round(part["token_len"].mean(), 2),
        })
    return pd.DataFrame(rows)


def split_checks(split_df, pair_split):
    """Leakage and balance checks between train and test."""
    from scipy.stats import chi2_contingency, ks_2samp

    train, test = split_df[split_df["split"] == "train"], split_df[split_df["split"] == "test"]
    norm_train = set(train["statement"].map(normalize_statement))
    norm_test = set(test["statement"].map(normalize_statement))
    pairs_in_both = (pair_split.groupby("pair_id")["split"].nunique() > 1).sum()

    topic_counts = pd.crosstab(split_df["topic"], split_df["split"])
    _, topic_p, _, _ = chi2_contingency(topic_counts.values)
    length_p = ks_2samp(train["token_len"], test["token_len"]).pvalue

    return pd.DataFrame([
        ("statements_in_both_splits_exact", len(set(train["statement"]) & set(test["statement"]))),
        ("statements_in_both_splits_near", len(norm_train & norm_test)),
        ("pairs_split_across_train_and_test", int(pairs_in_both)),
        ("prompts_without_split", int(pair_split["split"].isna().sum())),
        ("topic_balance_chi2_p", round(topic_p, 4)),
        ("length_balance_ks_p", round(length_p, 4)),
    ], columns=["check", "value"], dtype=object)


def split_topics(split_df):
    """Percent of each split that falls in each topic."""
    table = pd.crosstab(split_df["topic"], split_df["split"], normalize="columns").mul(100).round(2)
    table = table.rename(columns={"train": "percent_train", "test": "percent_test"})
    counts = pd.crosstab(split_df["topic"], split_df["split"])
    table["count_train"], table["count_test"] = counts["train"], counts["test"]
    return table.reset_index().sort_values("count_train", ascending=False)


# ---------------------------------------------------------------------------
# Step 10: Does GPT-2 know these facts?
# ---------------------------------------------------------------------------

QUESTION = "Is the following statement true or false?\n"
ANSWER_WORDS = (" True", " False")


def load_gpt2_model():
    """GPT-2 Small with its language-model head (downloads ~500 MB the first time)."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model


def answer_token_ids(tokenizer):
    """Token ids for ' True' and ' False'; both must be single GPT-2 tokens."""
    ids = [tokenizer.encode(word) for word in ANSWER_WORDS]
    for word, tok in zip(ANSWER_WORDS, ids):
        if len(tok) != 1:
            raise ValueError(f"{word!r} is {len(tok)} tokens; pick single-token answer words.")
    table = pd.DataFrame({"answer": [repr(w) for w in ANSWER_WORDS],
                          "token_id": [t[0] for t in ids], "n_tokens": [len(t) for t in ids]})
    return ids[0][0], ids[1][0], table


def few_shot_examples(split_df, seed=0):
    """Two true and two false TRAIN statements, used as worked examples in the prompt."""
    train = split_df[split_df["split"] == "train"]
    t = train[train["label"] == 1].sample(2, random_state=seed)["statement"].tolist()
    f = train[train["label"] == 0].sample(2, random_state=seed)["statement"].tolist()
    return [(t[0], "True"), (f[0], "False"), (f[1], "False"), (t[1], "True")]


def build_prompt(statement, examples=None):
    """Question + optional worked examples + the statement, ending at 'Answer:'."""
    text = QUESTION
    for ex_statement, ex_answer in examples or []:
        text += f"Statement: {ex_statement}\nAnswer: {ex_answer}\n\n"
    return text + f"Statement: {statement}\nAnswer:"


def score_prompts(prompts, model, tokenizer, true_id, false_id):
    """log P(' True') - log P(' False') for the next token after each prompt.

    Positive score = GPT-2 leans 'True'; negative = leans 'False'.
    """
    import torch

    scores = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts, start=1):
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            logp = torch.log_softmax(model(ids).logits[0, -1], dim=-1)
            scores.append(float(logp[true_id] - logp[false_id]))
            if i % 100 == 0 or i == len(prompts):
                print(f"    scored {i}/{len(prompts)}")
    return scores


def _best_threshold(scores, labels):
    """Threshold on the score that gives the highest accuracy (used on TRAIN only)."""
    import numpy as np

    candidates = np.unique(scores)
    accs = [((scores > c).astype(int) == labels).mean() for c in candidates]
    return float(candidates[int(np.argmax(accs))])


def knowledge_summary(scored, length_col="token_len"):
    """Per prompt format and split: AUROC, accuracy, and the 'always says True' bias.

    threshold_from_train: the cut-off picked on TRAIN, then applied to both splits,
    because GPT-2 may simply prefer one answer word overall.
    A 'length_only' row gives the shortcut baseline from step 3 on the same data.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rows = []
    for fmt, part in scored.groupby("format"):
        train = part[part["split"] == "train"]
        thr = _best_threshold(train["score"].values, train["label"].values)
        for split_name, s in part.groupby("split"):
            y, x = s["label"].values, s["score"].values
            auc = roc_auc_score(y, x)
            rows.append({
                "format": fmt, "split": split_name, "n": len(s),
                "auroc": round(auc, 3),
                "accuracy_at_0": round(((x > 0).astype(int) == y).mean(), 3),
                "threshold_from_train": round(thr, 3),
                "accuracy_at_train_threshold": round(((x > thr).astype(int) == y).mean(), 3),
                "percent_answered_true_at_0": round(100 * (x > 0).mean(), 2),
                "mean_score_true": round(x[y == 1].mean(), 3),
                "mean_score_false": round(x[y == 0].mean(), 3),
            })

    base = scored.drop_duplicates("statement")
    for split_name, s in base.groupby("split"):
        rows.append({"format": "length_only", "split": split_name, "n": len(s),
                     "auroc": round(roc_auc_score(s["label"], s[length_col]), 3)})
    return pd.DataFrame(rows)


def knowledge_by_topic(scored, fmt="few_shot", split="test", min_n=10):
    """AUROC and accuracy per topic, for topics with enough statements of both labels."""
    from sklearn.metrics import roc_auc_score

    part = scored[(scored["format"] == fmt) & (scored["split"] == split)]
    rows = []
    for topic, s in part.groupby("topic"):
        both = s["label"].nunique() == 2
        rows.append({
            "topic": topic, "n": len(s), "n_true": int(s["label"].sum()),
            "auroc": round(roc_auc_score(s["label"], s["score"]), 3) if both and len(s) >= min_n else None,
            "accuracy_at_0": round(((s["score"] > 0).astype(int) == s["label"]).mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)