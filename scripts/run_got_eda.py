"""EDA for the Geometry of Truth datasets (cities, translations, numbers + negated twins).

Run from the project root:
    python scripts/run_got_eda.py --step 1      # run one step
    python scripts/run_got_eda.py               # run every step that exists so far

Outputs: results/tables/got/got_s0_<step>_*.csv, results/figures/got/got_s0_<step>_*.svg
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.got import eda, plots  # noqa: E402
from src.got.io import (CLEAN_DIR, DATASETS, FAMILY, REVIEW_PATH, SPLIT_DIR,  # noqa: E402
                        load_clean, load_raw, save_table)

def step_1_integrity():
    """Step 1: columns, missing values, labels, duplicates; write a clean copy of each dataset."""
    print("\n=== Step 1: Integrity ===")
    columns, summary, duplicates, log, clean = eda.integrity_all(DATASETS, load_raw)

    print("Columns per dataset:")
    print(columns[["dataset", "n_rows", "metadata_columns"]].to_string(index=False))
    save_table(columns, "got_s0_01_columns")

    print("\nIntegrity checks (one column per dataset):")
    print(summary.to_string(index=False))
    save_table(summary, "got_s0_01_integrity_summary")
    save_table(duplicates, "got_s0_01_duplicates")

    print("\nCleaning log:")
    print(log.to_string(index=False))
    save_table(log, "got_s0_01_cleaning_log")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in clean.items():
        df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
    print(f"  saved clean datasets in {CLEAN_DIR.relative_to(PROJECT_ROOT)}")

def step_2_label_balance():
    """Step 2: true/false counts per dataset."""
    print("\n=== Step 2: Label balance ===")
    table = eda.label_balance_all(DATASETS, load_clean, FAMILY)
    print(table.to_string(index=False))
    save_table(table, "got_s0_02_label_balance")
    plots.plot_label_balance(table)  

def step_3_lengths():
    """Step 3: characters, words, GPT-2 tokens; can length alone predict the label?"""
    from src.eda.explore import load_gpt2_tokenizer

    print("\n=== Step 3: Length and the length shortcut ===")
    stats, shortcut, frames = eda.lengths_all(DATASETS, load_clean, load_gpt2_tokenizer())

    tokens = stats[(stats["measure"] == "token_len")][["dataset", "group", "mean", "min", "median", "max"]]
    print("GPT-2 tokens per statement:")
    print(tokens.to_string(index=False))
    save_table(stats, "got_s0_03_length_stats")

    print("\nCan length alone predict the label? (0.5 = no)")
    view = shortcut.pivot(index="dataset", columns="measure", values="auroc_best_direction").reindex(DATASETS)
    view["token_len_p"] = shortcut[shortcut["measure"] == "token_len"].set_index("dataset")["mannwhitney_p"]
    print(view.to_string())
    save_table(shortcut, "got_s0_03_length_shortcut_test")

    plots.plot_token_lengths(frames, shortcut)

def step_4_formatting():
    """Step 4: punctuation, odd characters, and the token each statement ends on."""
    from src.eda.explore import load_gpt2_tokenizer

    print("\n=== Step 4: Formatting and the last token ===")
    fmt, last = eda.formatting_all(DATASETS, load_clean, load_gpt2_tokenizer())
    print("Formatting checks (% of statements):")
    print(fmt.to_string(index=False))
    save_table(fmt, "got_s0_04_formatting_checks")

    print("\nMost common last tokens per dataset:")
    top = last.sort_values(["dataset", "count"], ascending=[True, False]).groupby("dataset").head(3)
    print(top[["dataset", "last_token", "count", "count_true", "count_false", "percent"]].to_string(index=False))
    save_table(last, "got_s0_04_last_token_counts")

    plots.plot_last_tokens(last, DATASETS)

def step_5_subjects():
    """Step 5: who/what each statement is about; add a 'subject' column to the clean files."""
    import pandas as pd

    from src.got.io import FAMILY

    print("\n=== Step 5: Subject structure ===")
    frames = eda.add_subjects(DATASETS, load_clean)

    summary = pd.DataFrame([eda.subject_summary(n, df) for n, df in frames.items()])
    print(summary.drop(columns=["statements_per_subject_min", "statements_per_subject_max"]).to_string(index=False))
    save_table(summary, "got_s0_05_subject_summary")

    twins = [(a, b) for a in DATASETS for b in DATASETS if a < b and FAMILY[a] == FAMILY[b]]
    overlap = pd.DataFrame([eda.twin_overlap(a, frames[a], b, frames[b]) for a, b in twins])
    print("\nDo twins share subjects?")
    print(overlap.to_string(index=False))
    save_table(overlap, "got_s0_05_twin_overlap")

    for name, df in frames.items():
        df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
    print(f"  added a 'subject' column to the clean files in {CLEAN_DIR.relative_to(PROJECT_ROOT)}")
    plots.plot_subjects(summary)

def step_6_shortcuts():
    """Step 6: can 'not', the answer word, or one number alone predict the label?"""
    print("\n=== Step 6: Shortcut cues ===")
    table = eda.shortcut_table(DATASETS, load_clean)
    print(table.to_string(index=False))
    save_table(table, "got_s0_06_shortcut_aurocs")

    for name in ["cities", "sp_en_trans"]:
        df = load_clean(name)
        counts = eda.answer_counts(df, eda.answer_of(df))
        print(f"\nMost unevenly split answers in {name} (at least 5 statements):")
        print(counts.head(8).to_string(index=False))
        save_table(counts, f"got_s0_06_answer_counts_{name}")

    plots.plot_shortcuts(table, DATASETS)

def step_7_numbers():
    """Step 7: number ranges, gap sizes, and the n1 shortcut with n1 held fixed."""
    import pandas as pd

    print("\n=== Step 7: Number properties ===")
    frames = {name: load_clean(name) for name in ["larger_than", "smaller_than"]}

    summary = pd.DataFrame([eda.number_summary(n, df) for n, df in frames.items()])
    print(summary.to_string(index=False))
    save_table(summary, "got_s0_07_number_summary")

    by_n1 = pd.concat([eda.percent_true_by_n1(n, df) for n, df in frames.items()], ignore_index=True)
    save_table(by_n1, "got_s0_07_percent_true_by_n1")
    print("\n% true for a few first numbers:")
    print(by_n1.pivot(index="n1", columns="dataset", values="percent_true").iloc[::8].to_string())

    plots.plot_numbers(by_n1, frames)

def step_8_label_review():
    """Step 8: manual label review in two rounds (create the sheet, then summarise it)."""
    import pandas as pd

    print("\n=== Step 8: Manual label review ===")
    if not REVIEW_PATH.exists():
        sample = eda.review_sample_all(DATASETS, load_clean)
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(REVIEW_PATH, index=False)
        print(f"  created {REVIEW_PATH.relative_to(PROJECT_ROOT)} ({len(sample)} statements)")
        print("  ROUND 1: fill 'agree_with_label' with yes / no / unsure, add a note where useful,")
        print("  save, then run step 8 again.")
        return

    review = pd.read_csv(REVIEW_PATH)
    table = eda.review_by_dataset(review)
    if table is None:
        print(f"  {REVIEW_PATH.relative_to(PROJECT_ROOT)} exists but is not filled in yet.")
        return
    print(table.to_string(index=False))
    save_table(table, "got_s0_08_label_review_summary")
    answers = review["agree_with_label"].astype(str).str.strip().str.lower()
    save_table(review[answers.isin(["no", "unsure"])], "got_s0_08_label_review_disputed")
    plots.plot_review(table)


def step_9_split():
    """Step 9: 70/30 split by subject, shared by each dataset and its twin."""
    from src.eda.explore import load_gpt2_tokenizer

    print("\n=== Step 9: Train/test split by subject ===")
    frames = {name: load_clean(name) for name in DATASETS}
    split_frames = eda.family_split(frames, FAMILY)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in split_frames.items():
        df.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    print(f"  saved split datasets in {SPLIT_DIR.relative_to(PROJECT_ROOT)}")

    checks = eda.split_checks(split_frames, FAMILY, load_gpt2_tokenizer())
    print(checks.to_string(index=False))
    save_table(checks, "got_s0_09_split_checks")
    plots.plot_split(checks)


def step_10_gpt2_knowledge():
    """Step 10: does GPT-2 find true statements more likely than false ones?"""
    import pandas as pd

    from src.eda.explore import load_gpt2_model, load_gpt2_tokenizer

    print("\n=== Step 10: Does GPT-2 know these facts? ===")
    frames = {name: pd.read_csv(SPLIT_DIR / f"{name}.csv") for name in DATASETS}
    print("  loading GPT-2...")
    scored = eda.score_statements(frames, load_gpt2_model(), load_gpt2_tokenizer())
    save_table(scored, "got_s0_10_gpt2_scores")

    summary = eda.knowledge_summary(scored)
    print("\nPair accuracy / AUROC (0.5 = GPT-2 has no preference for the true version):")
    print(summary.to_string(index=False))
    save_table(summary, "got_s0_10_gpt2_summary")
    plots.plot_knowledge(summary)

STEPS = {
    1: step_1_integrity,
    2: step_2_label_balance,
    3: step_3_lengths,
    4: step_4_formatting,
    5: step_5_subjects,
    6: step_6_shortcuts,
    7: step_7_numbers,
    8: step_8_label_review,
    9: step_9_split,
    10: step_10_gpt2_knowledge,
}



def main():
    parser = argparse.ArgumentParser(description="Geometry of Truth EDA")
    parser.add_argument("--step", type=int, choices=sorted(STEPS), help="Run only this step")
    args = parser.parse_args()

    for number in ([args.step] if args.step else sorted(STEPS)):
        STEPS[number]()
    print("\nDone.")


if __name__ == "__main__":
    main()