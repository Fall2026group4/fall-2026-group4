"""Tests for the data-cleaning and label-balance steps in src/eda/explore.py.

Pure pandas, no model or network. Run with: python -m pytest tests/test_eda.py
"""

import pandas as pd

from src.eda.explore import clean_dataset, integrity_report, label_balance, normalize_statement


def _df():
    return pd.DataFrame({
        "statement": [
            "The sky is blue.",
            "the sky is blue",          # near-duplicate of row 0, same label
            "Paris is in Spain.",
            "Paris is in Spain!",       # near-duplicate of row 2, same label
            "Water is wet.",
            "Water is wet",             # near-duplicate with a CONFLICTING label (row 5 is 0)
            "   ",                      # empty
            "Cats are mammals.",
        ],
        "label": [1, 1, 0, 0, 1, 0, 1, 1],
    })


def test_normalize_statement_ignores_case_spacing_and_end_punctuation():
    assert normalize_statement("  The  Sky is Blue.  ") == "the sky is blue"
    assert normalize_statement("Really?!") == "really"


def test_integrity_report_counts_duplicates_empties_and_conflicts():
    summary, dups = integrity_report(_df())
    s = dict(zip(summary["check"], summary["value"]))
    assert s["n_rows"] == 8
    assert s["empty_statement"] == 1
    assert s["near_duplicate_rows"] == 6
    assert s["conflicting_label_groups"] == 1
    assert dups["conflicting_labels"].sum() == 2


def test_clean_dataset_drops_empties_conflicts_and_redundant_copies():
    clean, log = clean_dataset(_df())
    steps = dict(zip(log["step"], log["rows"]))
    assert steps["missing_or_invalid"] == 1
    assert steps["conflicting_label_duplicates"] == 2
    assert steps["redundant_duplicates"] == 2
    assert steps["rows_kept"] == 3
    assert len(clean) == len(set(clean["statement"].map(normalize_statement)))
    assert set(clean["label"]) <= {0, 1}


def test_label_balance_counts_pairs_from_true_statements_only():
    t = label_balance(pd.DataFrame({"label": [1, 1, 1, 0]})).set_index("metric")["value"]
    assert t["total_statements"] == 4
    assert t["true_statements"] == 3
    assert t["usable_pairs"] == 3
    assert t["total_prompts"] == 6
