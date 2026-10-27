"""EDA for the RepE honesty dataset (facts_true_false.csv).

Run from the project root:
    python scripts/run_eda.py --step 1      # run one step
    python scripts/run_eda.py               # run every step that exists so far

Outputs are named after their step, e.g.
    results/tables/eda/repe_01_integrity_summary.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import explore, plots  # noqa: E402
from src.eda.load_data import CLEAN_PATH, load_clean, load_raw, save_table  # noqa: E402

PAIRS_PATH = PROJECT_ROOT / "data" / "repe" / "honesty_pairs.csv"
REVIEW_PATH = PROJECT_ROOT / "results" / "tables" / "eda" / "repe_08_label_review_sample.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "repe" / "facts_split.csv"
PAIRS_SPLIT_PATH = PROJECT_ROOT / "data" / "repe" / "honesty_pairs_split.csv"

def step_1_integrity():
    """Step 1: structure, missing values, labels, duplicates; write the clean dataset."""
    print("\n=== Step 1: Data integrity ===")
    raw = load_raw()

    summary, duplicates = explore.integrity_report(raw)
    print(summary.to_string(index=False))
    save_table(summary, "repe_01_integrity_summary")
    save_table(duplicates, "repe_01_duplicates")

    clean, log = explore.clean_dataset(raw)
    print("\nCleaning log:")
    print(log.to_string(index=False))
    save_table(log, "repe_01_cleaning_log")

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"  saved {CLEAN_PATH.relative_to(PROJECT_ROOT)}  ({len(clean)} rows)")


def step_2_label_balance():
    """Step 2: true/false balance and how many honest/untruthful pairs we get."""
    print("\n=== Step 2: Label balance and usable pairs ===")
    n_raw = len(load_raw())
    df = load_clean()

    table = explore.label_balance(df)
    print(table.to_string(index=False))
    save_table(table, "repe_02_label_balance")

    n_true = int((df["label"] == 1).sum())
    n_false = int((df["label"] == 0).sum())
    plots.plot_label_balance(n_raw, len(df), n_true, n_false)

def step_3_lengths():
    """Step 3: length in characters, words and GPT-2 tokens; length-shortcut test."""
    print("\n=== Step 3: Length statistics ===")
    df = load_clean()
    tokenizer = explore.load_gpt2_tokenizer()
    df = explore.add_length_columns(df, tokenizer)

    stats = explore.length_stats(df)
    print(stats.to_string(index=False))
    save_table(stats, "repe_03_length_stats")

    shortcut = explore.length_shortcut_test(df)
    print("\nCan length alone predict the label?")
    print(shortcut.to_string(index=False))
    save_table(shortcut, "repe_03_length_shortcut_test")

    extremes = explore.length_extremes(df)
    save_table(extremes, "repe_03_length_extremes")

    plots.plot_length_distribution(df)

def step_4_formatting():
    """Step 4: punctuation, odd characters, and which token each statement ends on."""
    print("\n=== Step 4: Formatting and the last token ===")
    df = load_clean()

    summary, issues = explore.formatting_checks(df)
    print(summary.to_string(index=False))
    save_table(summary, "repe_04_formatting_checks")
    save_table(issues, "repe_04_formatting_issues")

    tokenizer = explore.load_gpt2_tokenizer()
    last = explore.last_token_table(df, tokenizer)
    print("\nMost common last tokens:")
    print(last.head(10).to_string(index=False))
    save_table(last, "repe_04_last_token_counts")

    plots.plot_last_tokens(last)

def step_5_pairs():
    """Step 5: build honest/untruthful pairs and check they differ only in the persona."""
    print("\n=== Step 5: Pair construction checks ===")
    df = load_clean()
    tokenizer = explore.load_gpt2_tokenizer()

    pairs = explore.build_pairs(df)
    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"  saved {PAIRS_PATH.relative_to(PROJECT_ROOT)}  ({len(pairs)} prompts)")
    print("\nExample pair:")
    for prompt in pairs["prompt"].head(2):
        print("  ", prompt)

    personas = explore.persona_tokenization(tokenizer)
    print("\nHow GPT-2 tokenizes the persona words:")
    print(personas.to_string(index=False))
    save_table(personas, "repe_05_persona_tokens")

    check = explore.pair_token_check(pairs, tokenizer)
    save_table(check, "repe_05_pair_token_check")

    summary = explore.pair_summary(check)
    print("\nPair summary:")
    print(summary.to_string(index=False))
    save_table(summary, "repe_05_pair_summary")

    plots.plot_pair_lengths(check)
def step_6_word_cues():
    """Step 6: words and cue types that lean towards true or false statements."""
    print("\n=== Step 6: Word-level cues ===")
    df = load_clean()

    all_words, top_words = explore.distinctive_words(df)
    print("Most distinctive words (positive z = true, negative z = false):")
    print(top_words.to_string(index=False))
    save_table(all_words, "repe_06_word_log_odds_all")
    save_table(top_words, "repe_06_distinctive_words")

    categories = explore.cue_categories(df)
    print("\nCue categories:")
    print(categories.to_string(index=False))
    save_table(categories, "repe_06_cue_categories")

    plots.plot_word_cues(top_words, categories)
def step_7_topics():
    """Step 7: rough topic breakdown and whether some topics are mostly true or false."""
    print("\n=== Step 7: Topic coverage ===")
    df = load_clean()

    table, assignments, chi = explore.topic_table(df)
    print(table.to_string(index=False))
    print("\nIs topic related to label? (chi-square test)")
    print(chi.to_string(index=False))
    save_table(table, "repe_07_topic_counts")
    save_table(assignments, "repe_07_topic_assignments")
    save_table(chi, "repe_07_topic_label_chi_square")

    plots.plot_topics(table)

def step_8_label_quality():
    """Step 8: automatic ambiguity flags + a manual review sample (two rounds)."""
    import pandas as pd

    print("\n=== Step 8: Label quality ===")
    df = load_clean()

    summary, flagged = explore.ambiguity_flags(df)
    print("Statements with words that make labels easier to dispute:")
    print(summary.to_string(index=False))
    save_table(summary, "repe_08_ambiguity_summary")
    save_table(flagged, "repe_08_ambiguity_flagged")

    review_path = REVIEW_PATH
    if not review_path.exists():
        sample = explore.review_sample(df)
        sample.to_csv(review_path, index=False)
        print(f"\n  created {review_path.relative_to(PROJECT_ROOT)}")
        print("  ROUND 1: open it, fill 'agree_with_label' with yes / no / unsure for each row,")
        print("  add a short note where useful, save, then run step 8 again.")
        return

    review = pd.read_csv(review_path)
    result = explore.summarize_review(review)
    if result is None:
        print(f"\n  {review_path.relative_to(PROJECT_ROOT)} exists but is not filled in yet.")
        return
    print("\nManual review results:")
    print(result.to_string(index=False))
    save_table(result, "repe_08_label_review_summary")
    disputed = review[review["agree_with_label"].astype(str).str.strip().str.lower().isin(["no", "unsure"])]
    save_table(disputed, "repe_08_label_review_disputed")
    plots.plot_label_review(result)

def step_9_split():
    """Step 9: create the official train/test split and check it is fair."""
    import pandas as pd

    print("\n=== Step 9: Train/test split ===")
    df = load_clean()
    tokenizer = explore.load_gpt2_tokenizer()
    df = explore.add_length_columns(df, tokenizer)

    split_df = explore.make_split(df)
    pair_split = explore.split_pairs(pd.read_csv(PAIRS_PATH), split_df)

    split_df.to_csv(SPLIT_PATH, index=False)
    pair_split.to_csv(PAIRS_SPLIT_PATH, index=False)
    print(f"  saved {SPLIT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  saved {PAIRS_SPLIT_PATH.relative_to(PROJECT_ROOT)}")

    summary = explore.split_summary(split_df, pair_split)
    print("\nSplit sizes:")
    print(summary.to_string(index=False))
    save_table(summary, "repe_09_split_summary")

    checks = explore.split_checks(split_df, pair_split)
    print("\nLeakage and balance checks:")
    print(checks.to_string(index=False))
    save_table(checks, "repe_09_split_checks")

    topics = explore.split_topics(split_df)
    save_table(topics, "repe_09_split_topics")

    plots.plot_split_balance(summary, topics, split_df)
    
def step_10_gpt2_knowledge():

    """Step 10: GPT-2's own True/False judgement, zero-shot and few-shot."""
    import pandas as pd

    print("\n=== Step 10: Does GPT-2 know these facts? ===")
    split_df = pd.read_csv(SPLIT_PATH)
    tokenizer = explore.load_gpt2_tokenizer()
    print("  loading GPT-2 (first time downloads ~500 MB)...")
    model = explore.load_gpt2_model()

    true_id, false_id, answer_table = explore.answer_token_ids(tokenizer)
    save_table(answer_table, "repe_10_answer_tokens")

    examples = explore.few_shot_examples(split_df)
    example_statements = {s for s, _ in examples}
    print("\nFew-shot prompt example:")
    print(explore.build_prompt("<test statement>", examples))

    scored = []
    for fmt, ex in [("zero_shot", None), ("few_shot", examples)]:
        part = split_df[~split_df["statement"].isin(example_statements)].copy()
        print(f"\n  scoring {len(part)} statements ({fmt})")
        prompts = [explore.build_prompt(s, ex) for s in part["statement"]]
        part["score"] = explore.score_prompts(prompts, model, tokenizer, true_id, false_id)
        part["format"] = fmt
        scored.append(part)
    scored = pd.concat(scored, ignore_index=True)
    save_table(scored[["format", "split", "topic", "label", "score", "token_len", "statement"]],
               "repe_10_gpt2_scores")

    summary = explore.knowledge_summary(scored)
    print("\nSummary (AUROC 0.5 = guessing, 1.0 = perfect):")
    print(summary.to_string(index=False))
    save_table(summary, "repe_10_gpt2_summary")

    by_topic = explore.knowledge_by_topic(scored)
    save_table(by_topic, "repe_10_gpt2_by_topic")

    plots.plot_gpt2_knowledge(scored, summary)
STEPS = {
    1: step_1_integrity,
    2: step_2_label_balance,
    3: step_3_lengths,
    4: step_4_formatting,
    5: step_5_pairs,
    6: step_6_word_cues,
    7: step_7_topics,
    8: step_8_label_quality,
    9: step_9_split,
    10: step_10_gpt2_knowledge,
}



def main():
    parser = argparse.ArgumentParser(description="RepE dataset EDA")
    parser.add_argument("--step", type=int, choices=sorted(STEPS), help="Run only this step")
    args = parser.parse_args()

    for number in ([args.step] if args.step else sorted(STEPS)):
        STEPS[number]()
    print("\nDone.")


if __name__ == "__main__":
    main()