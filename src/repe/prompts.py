"""Prompt sets for the honesty analysis.

original         : the RepE wording (honest vs untruthful), true statements
matched_<a>_<b>  : persona words with EQUAL GPT-2 token counts (position control)
false_statements : the RepE wording on the 300 FALSE statements (generalisation test)
"""

import pandas as pd

TEMPLATE = "Pretend you're {persona} person making statements about the world. {statement}"
ORIGINAL = ("an honest", "an untruthful")

# Candidate (honest, dishonest) wordings for the length-matched control.
# Only pairs whose prompts come out the SAME length in GPT-2 tokens are kept.
MATCHED_CANDIDATES = [
    ("an honest", "a dishonest"),
    ("a truthful", "a deceptive"),
    ("a sincere", "an insincere"),
    ("an honest", "a lying"),
    ("a truthful", "a lying"),
]


def persona_lengths(count_tokens, candidates=MATCHED_CANDIDATES, probe="The sky is blue."):
    """Token count of a full prompt for each persona in each candidate pair."""
    rows = []
    for honest, dishonest in candidates:
        n_h = count_tokens(TEMPLATE.format(persona=honest, statement=probe))
        n_d = count_tokens(TEMPLATE.format(persona=dishonest, statement=probe))
        rows.append({"honest_persona": honest, "dishonest_persona": dishonest,
                     "honest_tokens": n_h, "dishonest_tokens": n_d,
                     "length_matched": n_h == n_d})
    return pd.DataFrame(rows)


def set_name(honest, dishonest):
    """matched_honest_dishonest style name for a persona pair."""
    return f"matched_{honest.split()[-1]}_{dishonest.split()[-1]}"


def build_pair_set(statements, honest, dishonest, name):
    """One honest and one dishonest prompt per statement.

    statements: DataFrame with 'statement' and 'split' columns.
    label = 1 for the honest prompt, 0 for the dishonest one.
    """
    rows = []
    for pair_id, (statement, split) in enumerate(zip(statements["statement"], statements["split"])):
        for persona_word, label in [(honest, 1), (dishonest, 0)]:
            rows.append({
                "prompt_set": name, "pair_id": pair_id, "label": label,
                "persona": "honest" if label == 1 else "dishonest",
                "persona_words": persona_word, "split": split, "statement": statement,
                "prompt": TEMPLATE.format(persona=persona_word, statement=statement),
            })
    return pd.DataFrame(rows)


def build_all_sets(split_df, original_pairs, lengths_table):
    """Every prompt set used by the baseline, as {name: DataFrame}."""
    sets = {}

    original = original_pairs.copy()
    original["prompt_set"] = "original"
    original["persona"] = original["persona"].replace({"untruthful": "dishonest"})
    original["persona_words"] = original["label"].map({1: ORIGINAL[0], 0: ORIGINAL[1]})
    sets["original"] = original[["prompt_set", "pair_id", "label", "persona", "persona_words",
                                 "split", "statement", "prompt"]]

    true_statements = split_df[split_df["label"] == 1].reset_index(drop=True)
    for _, row in lengths_table[lengths_table["length_matched"]].iterrows():
        name = set_name(row["honest_persona"], row["dishonest_persona"])
        sets[name] = build_pair_set(true_statements, row["honest_persona"], row["dishonest_persona"], name)

    false_statements = split_df[split_df["label"] == 0].reset_index(drop=True)
    sets["false_statements"] = build_pair_set(false_statements, *ORIGINAL, "false_statements")
    return sets