"""Concept-detection and steering-result scoring helpers (Phases 3-5).

Pulled out of ``cookbooks/02_pretained_sae_baseline.ipynb`` so the scoring
logic can be unit tested (Instructor Review #2, "Nothing in the repo is
testable") instead of only being checkable by re-running the whole notebook
against a downloaded model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from src.models.gpt2_sae import SAEBundle, get_feature_activation


def score_examples(
    bundle: SAEBundle,
    positive_examples: Sequence[str],
    negative_examples: Sequence[str],
    feature_id: int,
) -> pd.DataFrame:
    """Score a feature's activation on labeled positive/negative examples.

    Returns one row per example with columns ``sentence``, ``label``
    (``"positive"``/``"negative"``), and ``feature_activation``.
    """
    rows = []
    for sentence in positive_examples:
        rows.append(
            {
                "sentence": sentence,
                "label": "positive",
                "feature_activation": get_feature_activation(bundle, sentence, feature_id),
            }
        )
    for sentence in negative_examples:
        rows.append(
            {
                "sentence": sentence,
                "label": "negative",
                "feature_activation": get_feature_activation(bundle, sentence, feature_id),
            }
        )
    return pd.DataFrame(rows)


def build_steering_results(
    steering_outputs: dict, concept_word: str
) -> pd.DataFrame:
    """Turn a ``{strength: generated_text}`` mapping into a results table.

    Adds ``mentions_concept`` (bool) and ``concept_count`` (int) columns by
    searching the lower-cased output for ``concept_word`` — this is the
    exact computation that was missing a source column in the original
    notebook (``mentions_paris`` was referenced but never assigned, causing
    a ``KeyError`` on a fresh run; see Review #2 finding).

    ``steering_outputs`` keys become the ``steering_strength`` column and
    are sorted ascending so strength 0 (baseline) is always first.
    """
    strengths = sorted(steering_outputs.keys())
    outputs = [steering_outputs[s] for s in strengths]

    results = pd.DataFrame({"steering_strength": strengths, "output": outputs})
    lowered = results["output"].str.lower()
    concept_lower = concept_word.lower()
    results["mentions_concept"] = lowered.str.contains(concept_lower)
    results["concept_count"] = lowered.str.count(concept_lower)
    return results
