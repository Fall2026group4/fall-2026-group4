"""Tests for src/evaluation/detection.py.

``build_steering_results`` is the function with the bug the instructor
review found (a ``mentions_paris`` column referenced but never created,
causing a KeyError on a fresh notebook run). These tests pin down the
fixed behavior so a future edit can't silently reintroduce it.
"""

import pandas as pd

from src.evaluation.detection import build_steering_results


def test_build_steering_results_creates_mentions_and_count_columns():
    outputs = {
        0: "The best place to spend my vacation is in the mountains.",
        20: "The best place to spend my vacation is at the beach.",
    }

    results = build_steering_results(outputs, concept_word="paris")

    assert list(results.columns) == [
        "steering_strength",
        "output",
        "mentions_concept",
        "concept_count",
    ]
    assert not results["mentions_concept"].any()
    assert (results["concept_count"] == 0).all()


def test_build_steering_results_detects_concept_mentions_case_insensitively():
    outputs = {
        0: "I went to London.",
        50: "I went to Paris and loved Paris.",
    }

    results = build_steering_results(outputs, concept_word="Paris")
    results = results.set_index("steering_strength")

    assert results.loc[0, "mentions_concept"] is False or results.loc[0, "mentions_concept"] == False
    assert results.loc[50, "mentions_concept"] == True
    assert results.loc[50, "concept_count"] == 2


def test_build_steering_results_sorts_by_strength_with_baseline_first():
    # Pass keys out of order to make sure output ordering doesn't depend on
    # dict insertion order.
    outputs = {50: "b", 0: "a", 20: "c"}

    results = build_steering_results(outputs, concept_word="x")

    assert list(results["steering_strength"]) == [0, 20, 50]
    assert list(results["output"]) == ["a", "c", "b"]
