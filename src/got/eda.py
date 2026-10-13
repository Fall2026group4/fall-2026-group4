"""EDA functions for the Geometry of Truth datasets. One section per EDA step."""

import pandas as pd

from src.eda.explore import integrity_report, normalize_statement


# ---------------------------------------------------------------------------
# Step 1: Integrity
# ---------------------------------------------------------------------------

def column_overview(name, df):
    """Which columns a dataset has, beyond the shared statement and label."""
    extra = [c for c in df.columns if c not in ("statement", "label")]
    return {"dataset": name, "n_rows": len(df), "n_columns": len(df.columns),
            "columns": ", ".join(df.columns), "metadata_columns": ", ".join(extra) or "(none)"}


def clean_keep_columns(df):
    """Same cleaning rules as the RepE EDA, but KEEP the metadata columns (city, n1, ...).

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

    norm = out["statement"].map(normalize_statement)
    conflict = norm.map(out.groupby(norm)["label"].nunique()) > 1
    log.append(("conflicting_label_duplicates", int(conflict.sum())))
    out, norm = out[~conflict], norm[~conflict]

    dup = norm.duplicated(keep="first")
    log.append(("redundant_duplicates", int(dup.sum())))
    out = out[~dup].reset_index(drop=True)

    log.append(("rows_kept", len(out)))
    return out, dict(log)


def integrity_all(datasets, load):
    """Run the RepE integrity checks and cleaning on every dataset.

    Returns (columns_table, summary_wide, duplicates, cleaning_log, {name: clean_df}).
    """
    columns, summaries, dups, logs, clean = [], {}, [], {}, {}
    for name in datasets:
        raw = load(name)
        columns.append(column_overview(name, raw))
        summary, dup_rows = integrity_report(raw[["statement", "label"]])
        summaries[name] = summary.set_index("check")["value"]
        if len(dup_rows):
            dups.append(dup_rows.assign(dataset=name))
        clean[name], logs[name] = clean_keep_columns(raw)

    summary_wide = pd.DataFrame(summaries)
    summary_wide = summary_wide.drop(index=["columns", "dtypes"], errors="ignore").reset_index(names="check")
    log_wide = pd.DataFrame(logs).reset_index(names="step")
    duplicates = pd.concat(dups, ignore_index=True) if dups else pd.DataFrame(columns=["dataset"])
    return pd.DataFrame(columns), summary_wide, duplicates, log_wide, clean



# ---------------------------------------------------------------------------
# Step 2: Label balance
# ---------------------------------------------------------------------------

def label_balance_all(datasets, load, family):
    """True/false counts per dataset, with the topic family it belongs to."""
    rows = []
    for name in datasets:
        df = load(name)
        n, n_true = len(df), int((df["label"] == 1).sum())
        rows.append({
            "dataset": name, "family": family[name], "n_statements": n,
            "n_true": n_true, "n_false": n - n_true,
            "percent_true": round(100 * n_true / n, 2),
            "true_to_false_ratio": round(n_true / (n - n_true), 3) if n > n_true else None,
        })
    table = pd.DataFrame(rows)
    total = {"dataset": "ALL", "family": "-", "n_statements": table["n_statements"].sum(),
             "n_true": table["n_true"].sum(), "n_false": table["n_false"].sum()}
    total["percent_true"] = round(100 * total["n_true"] / total["n_statements"], 2)
    total["true_to_false_ratio"] = round(total["n_true"] / total["n_false"], 3)
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)



# ---------------------------------------------------------------------------
# Step 3: Length and the length shortcut
# ---------------------------------------------------------------------------

def lengths_all(datasets, load, tokenizer):
    """Length statistics and the length-only shortcut test (from the RepE EDA) for every dataset.

    Returns (stats, shortcut, {name: df with char_len, word_len, token_len}).
    """
    from src.eda.explore import add_length_columns, length_shortcut_test, length_stats

    stats, shortcut, frames = [], [], {}
    for name in datasets:
        df = add_length_columns(load(name), tokenizer=tokenizer)
        frames[name] = df
        stats.append(length_stats(df).assign(dataset=name))
        shortcut.append(length_shortcut_test(df).assign(dataset=name))
    stats = pd.concat(stats, ignore_index=True)
    shortcut = pd.concat(shortcut, ignore_index=True)
    first = ["dataset"]
    return (stats[first + [c for c in stats.columns if c not in first]],
            shortcut[first + [c for c in shortcut.columns if c not in first]], frames)



# ---------------------------------------------------------------------------
# Step 4: Formatting and the last token
# ---------------------------------------------------------------------------

def formatting_all(datasets, load, tokenizer):
    """Formatting checks and last-token counts (from the RepE EDA) for every dataset.

    Returns (formatting_wide, last_tokens_long).
    """
    from src.eda.explore import formatting_checks, last_token_table

    fmt, last = {}, []
    for name in datasets:
        df = load(name)
        summary, _ = formatting_checks(df)
        fmt[name] = summary.set_index("check")["percent_all"]
        last.append(last_token_table(df, tokenizer).assign(dataset=name))
    fmt = pd.DataFrame(fmt).reset_index(names="check")
    last = pd.concat(last, ignore_index=True)
    return fmt, last[["dataset"] + [c for c in last.columns if c != "dataset"]]



# ---------------------------------------------------------------------------
# Step 5: Subject structure
# ---------------------------------------------------------------------------

import re  # noqa: E402

_SPANISH_WORD = re.compile(r"The Spanish word '(.+?)'")


def subject_of(name, df):
    """The 'subject' each statement is about; true and false versions share it.

    cities      -> the city
    translation -> the Spanish word (parsed from the sentence; no metadata column)
    numbers     -> the unordered number pair, e.g. '51-52'
    """
    if "city" in df.columns:
        return df["city"].astype(str)
    if "n1" in df.columns and "n2" in df.columns:
        lo = df[["n1", "n2"]].min(axis=1).astype(int).astype(str)
        hi = df[["n1", "n2"]].max(axis=1).astype(int).astype(str)
        return lo + "-" + hi
    words = df["statement"].str.extract(_SPANISH_WORD)[0]
    if words.isna().any():
        bad = df.loc[words.isna(), "statement"].head(3).tolist()
        raise ValueError(f"{name}: could not find the Spanish word in e.g. {bad}")
    return words.str.lower()


def subject_summary(name, df):
    """How many subjects, statements per subject, and how many have both a true and a false version."""
    per = df.groupby("subject")["label"].agg(["size", "nunique", "sum"])
    return {
        "dataset": name,
        "n_statements": len(df),
        "n_subjects": len(per),
        "statements_per_subject_mean": round(per["size"].mean(), 2),
        "statements_per_subject_min": int(per["size"].min()),
        "statements_per_subject_max": int(per["size"].max()),
        "subjects_with_both_labels": int((per["nunique"] == 2).sum()),
        "percent_subjects_with_both_labels": round(100 * (per["nunique"] == 2).mean(), 2),
        "subjects_only_true": int((per["sum"] == per["size"]).sum()),
        "subjects_only_false": int((per["sum"] == 0).sum()),
    }


def twin_overlap(a_name, a, b_name, b):
    """Do a dataset and its twin use the same subjects (and the same statements, minus 'not')?"""
    sa, sb = set(a["subject"]), set(b["subject"])
    return {
        "dataset": a_name, "twin": b_name,
        "subjects": len(sa), "twin_subjects": len(sb),
        "shared_subjects": len(sa & sb),
        "percent_shared": round(100 * len(sa & sb) / max(len(sa | sb), 1), 2),
    }


def add_subjects(datasets, load):
    """{name: clean df with a 'subject' column}."""
    out = {}
    for name in datasets:
        df = load(name)
        df["subject"] = subject_of(name, df)
        out[name] = df
    return out



# ---------------------------------------------------------------------------
# Step 6: Shortcut cues
# ---------------------------------------------------------------------------

_ENGLISH_ANSWER = re.compile(r"mean[s]? '(.+?)'\.?$")


def answer_of(df):
    """The 'answer' part of each statement: the country, or the English meaning."""
    if "country" in df.columns:
        return df["country"].astype(str)
    if "statement" in df.columns and df["statement"].str.contains("Spanish word").all():
        return df["statement"].str.extract(_ENGLISH_ANSWER)[0].str.lower()
    return None


def _auroc_both(y, score):
    from sklearn.metrics import roc_auc_score

    if len(set(y)) < 2 or len(set(score)) < 2:
        return 0.5, 0.5
    auc = roc_auc_score(y, score)
    return round(auc, 3), round(max(auc, 1 - auc), 3)


def answer_prior_auroc(df, answers, n_folds=5, alpha=1.0):
    """How well 'how often was this answer true in OTHER subjects' predicts the label.

    Out-of-fold by subject, so a statement never scores itself. A high value means
    the answer word alone (e.g. a country) leaks the label.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    y = df["label"].values
    scores = np.zeros(len(df))
    n_folds = min(n_folds, df["subject"].nunique())
    for tr, te in GroupKFold(n_splits=n_folds).split(df, y, groups=df["subject"]):
        train = pd.DataFrame({"answer": answers.iloc[tr].values, "label": y[tr]})
        stats = train.groupby("answer")["label"].agg(["sum", "count"])
        rate = (stats["sum"] + alpha) / (stats["count"] + 2 * alpha)
        scores[te] = answers.iloc[te].map(rate).fillna(0.5).values
    return _auroc_both(y, scores)


def shortcut_table(datasets, load):
    """AUROC of each surface cue per dataset (0.5 = no shortcut)."""
    rows = []
    for name in datasets:
        df = load(name)
        y = df["label"].values
        has_not = df["statement"].str.contains(r"\bnot\b", case=False).astype(int).values
        auc, best = _auroc_both(y, has_not)
        rows.append({"dataset": name, "cue": "contains 'not'", "auroc": auc, "auroc_best_direction": best,
                     "percent_with_cue": round(100 * has_not.mean(), 2)})

        answers = answer_of(df)
        if answers is not None and answers.notna().all():
            auc, best = answer_prior_auroc(df, answers)
            rows.append({"dataset": name, "cue": "answer word only", "auroc": auc,
                         "auroc_best_direction": best, "percent_with_cue": None})

        for col in ["n1", "n2", "abs_diff"]:
            if col in df.columns:
                auc, best = _auroc_both(y, df[col].values)
                rows.append({"dataset": name, "cue": f"{col} only", "auroc": auc,
                             "auroc_best_direction": best, "percent_with_cue": None})
    table = pd.DataFrame(rows)
    table["shortcut_risk"] = pd.cut(table["auroc_best_direction"], [0, 0.58, 0.65, 1.01],
                                    labels=["low", "moderate", "high"], right=False).astype(str)
    return table


def answer_counts(df, answers, top_n=15):
    """Answers (countries / English meanings) most unevenly split between true and false."""
    t = pd.DataFrame({"answer": answers, "label": df["label"]}).groupby("answer")["label"].agg(
        n="size", n_true="sum")
    t["n_false"] = t["n"] - t["n_true"]
    t["percent_true"] = (100 * t["n_true"] / t["n"]).round(1)
    t = t[t["n"] >= 5]
    t["imbalance"] = (t["percent_true"] - 50).abs()
    return t.sort_values(["imbalance", "n"], ascending=False).head(top_n).reset_index()



# ---------------------------------------------------------------------------
# Step 7: Number properties
# ---------------------------------------------------------------------------

def grouped_auroc(labels, scores, groups, min_size=4):
    """AUROC computed INSIDE each group, then averaged (weighted by group size).

    Grouping by the first number (n1) removes the 'n1 alone' shortcut: within a group,
    n1 is constant, so only a real comparison of n1 and n2 can rank true above false.
    Reused in later stages to evaluate truth directions on the numbers datasets.
    """
    from sklearn.metrics import roc_auc_score

    t = pd.DataFrame({"y": labels, "s": scores, "g": groups})
    total, weight = 0.0, 0
    for _, part in t.groupby("g"):
        if len(part) >= min_size and part["y"].nunique() == 2:
            total += roc_auc_score(part["y"], part["s"]) * len(part)
            weight += len(part)
    return round(total / weight, 3) if weight else None


def number_summary(name, df):
    """Range of the numbers, gap sizes by label, and shortcut AUROCs with and without n1 held fixed."""
    from sklearn.metrics import roc_auc_score

    y = df["label"].values
    comparison = (df["n1"] - df["n2"]).values  # what a real comparison would use
    if name.startswith("smaller"):
        comparison = -comparison
    return {
        "dataset": name,
        "n1_range": f"{int(df['n1'].min())}-{int(df['n1'].max())}",
        "n2_range": f"{int(df['n2'].min())}-{int(df['n2'].max())}",
        "distinct_numbers": int(pd.concat([df["n1"], df["n2"]]).nunique()),
        "mean_gap_true": round(df.loc[df["label"] == 1, "abs_diff"].mean(), 2),
        "mean_gap_false": round(df.loc[df["label"] == 0, "abs_diff"].mean(), 2),
        "n1_only_auroc": round(max(roc_auc_score(y, df["n1"]), 1 - roc_auc_score(y, df["n1"])), 3),
        "n1_only_auroc_within_n1": grouped_auroc(y, df["n1"].values, df["n1"].values),
        "true_comparison_auroc_within_n1": grouped_auroc(y, comparison, df["n1"].values),
    }


def percent_true_by_n1(name, df):
    """For each first number, how many statements and what share are true."""
    t = df.groupby("n1")["label"].agg(n="size", n_true="sum").reset_index()
    t["percent_true"] = (100 * t["n_true"] / t["n"]).round(1)
    return t.assign(dataset=name)[["dataset", "n1", "n", "n_true", "percent_true"]]



# ---------------------------------------------------------------------------
# Step 8: Manual label review
# ---------------------------------------------------------------------------

def review_sample_all(datasets, load, per_dataset=5, seed=0):
    """A few random statements per dataset for MANUAL review (fill agree_with_label: yes/no/unsure)."""
    parts = []
    for name in datasets:
        df = load(name)
        parts.append(df.sample(n=min(per_dataset, len(df)), random_state=seed)
                     .reset_index(names="row").assign(dataset=name))
    sample = pd.concat(parts, ignore_index=True)
    sample["agree_with_label"] = ""
    sample["note"] = ""
    return sample[["dataset", "row", "label", "statement", "agree_with_label", "note"]]


def review_by_dataset(review):
    """yes / no / unsure counts per dataset, for a filled review sheet."""
    answers = review["agree_with_label"].fillna("").astype(str).str.strip().str.lower()
    filled = review.assign(answer=answers)[answers.isin(["yes", "no", "unsure"])]
    if filled.empty:
        return None
    table = pd.crosstab(filled["dataset"], filled["answer"]).reindex(columns=["yes", "no", "unsure"], fill_value=0)
    table["reviewed"] = table.sum(axis=1)
    table["percent_yes"] = (100 * table["yes"] / table["reviewed"]).round(1)
    total = table[["yes", "no", "unsure", "reviewed"]].sum()
    table.loc["ALL"] = list(total) + [round(100 * total["yes"] / total["reviewed"], 1)]
    table[["yes", "no", "unsure", "reviewed"]] = table[["yes", "no", "unsure", "reviewed"]].astype(int)
    return table.reset_index()


# ---------------------------------------------------------------------------
# Step 9: Train/test split by subject
# ---------------------------------------------------------------------------

def family_split(frames, family, test_size=0.3, seed=0):
    """70/30 split of SUBJECTS, shared by a dataset and its twin.

    A city / Spanish word / number pair goes to train or test for BOTH twins together,
    so the negated version of a training statement never appears in test.
    Subjects are stratified by their label pattern in the affirmative dataset.
    """
    from sklearn.model_selection import train_test_split

    out = {}
    for fam in dict.fromkeys(family.values()):
        names = [n for n in frames if family[n] == fam]
        first = frames[names[0]]
        pattern = first.groupby("subject")["label"].agg(lambda s: "both" if s.nunique() == 2 else str(s.iloc[0]))
        subjects = sorted(set().union(*[set(frames[n]["subject"]) for n in names]))
        strat = pd.Series(subjects).map(pattern).fillna("missing")
        strat = strat.where(strat.map(strat.value_counts()) >= 2, "rare")
        train_s, test_s = train_test_split(subjects, test_size=test_size, random_state=seed,
                                           stratify=strat if strat.nunique() > 1 else None)
        lookup = {s: "train" for s in train_s} | {s: "test" for s in test_s}
        for n in names:
            out[n] = frames[n].assign(split=frames[n]["subject"].map(lookup))
    return out


def split_checks(split_frames, family, tokenizer=None):
    """Sizes, label balance and leakage checks per dataset."""
    from scipy.stats import ks_2samp

    rows = []
    for name, df in split_frames.items():
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        twin = [n for n in split_frames if n != name and family[n] == family[name]]
        twin_mismatch = 0
        if twin:
            other = split_frames[twin[0]].groupby("subject")["split"].first()
            mine = df.groupby("subject")["split"].first()
            shared = mine.index.intersection(other.index)
            twin_mismatch = int((mine[shared] != other[shared]).sum())
        lengths = df["statement"].map(lambda s: len(tokenizer.encode(s))) if tokenizer else df["statement"].str.len()
        rows.append({
            "dataset": name, "train": len(tr), "test": len(te),
            "train_subjects": tr["subject"].nunique(), "test_subjects": te["subject"].nunique(),
            "percent_true_train": round(100 * tr["label"].mean(), 2),
            "percent_true_test": round(100 * te["label"].mean(), 2),
            "subjects_in_both_splits": len(set(tr["subject"]) & set(te["subject"])),
            "subjects_split_differently_from_twin": twin_mismatch,
            "unassigned_rows": int(df["split"].isna().sum()),
            "length_balance_ks_p": round(ks_2samp(lengths[df["split"] == "train"],
                                                  lengths[df["split"] == "test"]).pvalue, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 10: Does GPT-2 know these facts?
# ---------------------------------------------------------------------------

def sentence_logprob(model, tokenizer, text):
    """Total and per-token log-probability GPT-2 assigns to a sentence (after a BOS token)."""
    import torch

    ids = tokenizer(tokenizer.bos_token + text, return_tensors="pt").input_ids
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0], dim=-1)
    n = int(ids.shape[1])
    total = sum(float(logp[i, int(ids[0, i + 1])]) for i in range(n - 1))
    return total, total / (n - 1)


def score_statements(frames, model, tokenizer):
    """Log-probability of every statement in every dataset."""
    rows = []
    for name, df in frames.items():
        print(f"    scoring {name} ({len(df)} statements)")
        for _, r in df.iterrows():
            total, per_token = sentence_logprob(model, tokenizer, r["statement"])
            rows.append({"dataset": name, "subject": r["subject"], "label": r["label"], "split": r.get("split"),
                         "n1": r.get("n1"), "logprob": total, "logprob_per_token": per_token,
                         "statement": r["statement"]})
    return pd.DataFrame(rows)


def knowledge_summary(scored):
    """Does GPT-2 find true statements more likely than false ones?

    pair_accuracy : within a subject's true/false pair, share where the TRUE one is more likely
                    (cities and numbers only; translations are not paired)
    auroc_logprob : statement-level AUROC of the log-probability (0.5 = no preference)
    auroc_within_n1 : numbers only, with the first-number shortcut removed
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for name, d in scored.groupby("dataset", sort=False):
        per = d.groupby("subject")
        paired = per["label"].nunique().eq(2) & per.size().eq(2)
        pair_ids = paired[paired].index
        if len(pair_ids):
            p = d[d["subject"].isin(pair_ids)].pivot_table(index="subject", columns="label", values="logprob")
            pair_acc = round(float((p[1] > p[0]).mean()), 3)
            mean_gap = round(float((p[1] - p[0]).mean()), 3)
        else:
            pair_acc = mean_gap = None
        within = grouped_auroc(d["label"].values, d["logprob"].values, d["n1"].values) if d["n1"].notna().all() else None
        rows.append({
            "dataset": name, "n": len(d), "n_pairs": len(pair_ids),
            "pair_accuracy": pair_acc, "mean_logprob_gap_true_minus_false": mean_gap,
            "auroc_logprob": round(roc_auc_score(d["label"], d["logprob"]), 3),
            "auroc_logprob_per_token": round(roc_auc_score(d["label"], d["logprob_per_token"]), 3),
            "auroc_within_n1": within,
        })
    return pd.DataFrame(rows)