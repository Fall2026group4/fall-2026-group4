"""Stage 4A utilities: detect truth-associated SAE features."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _to_numpy(x):
    """Convert torch tensor or array-like input to NumPy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def rank_features_on_train(
    train_features,
    train_labels,
    top_n=50,
):
    """Rank SAE features using the training split only.

    Features are ranked by their ability to separate TRUE/FALSE
    examples on the training data.

    Parameters
    ----------
    train_features : array-like
        Shape [n_examples, n_features].

    train_labels : array-like
        Binary labels, where 1=TRUE and 0=FALSE.

    top_n : int
        Number of candidate features to retain.

    Returns
    -------
    pandas.DataFrame
        Top training-selected SAE features.
    """

    features = _to_numpy(train_features)
    labels = _to_numpy(train_labels).reshape(-1)

    true_mask = labels == 1
    false_mask = labels == 0

    rows = []

    for feature_id in range(features.shape[1]):
        values = features[:, feature_id]

        true_mean = values[true_mask].mean()
        false_mean = values[false_mask].mean()

        mean_difference = true_mean - false_mean

        true_frequency = (
            values[true_mask] > 0
        ).mean()

        false_frequency = (
            values[false_mask] > 0
        ).mean()

        frequency_difference = (
            true_frequency - false_frequency
        )

        # Constant features cannot discriminate the labels.
        if np.all(values == values[0]):
            raw_auc = 0.5
        else:
            raw_auc = roc_auc_score(
                labels,
                values,
            )

        # Keep direction learned from TRAIN only.
        polarity = (
            1
            if raw_auc >= 0.5
            else -1
        )

        detection_auc = max(
            raw_auc,
            1.0 - raw_auc,
        )

        rows.append(
            {
                "feature_id": feature_id,
                "train_raw_auroc": raw_auc,
                "train_detection_auroc": detection_auc,
                "polarity": polarity,
                "train_true_mean": true_mean,
                "train_false_mean": false_mean,
                "train_mean_difference": mean_difference,
                "train_true_frequency": true_frequency,
                "train_false_frequency": false_frequency,
                "train_frequency_difference": frequency_difference,
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "train_detection_auroc",
        ascending=False,
    ).reset_index(drop=True)

    return results.head(top_n)


def evaluate_selected_features_on_test(
    selected_features,
    test_features,
    test_labels,
):
    """Evaluate TRAIN-selected SAE features on held-out test data."""

    features = _to_numpy(test_features)
    labels = _to_numpy(test_labels).reshape(-1)

    rows = []

    for _, row in selected_features.iterrows():
        feature_id = int(row["feature_id"])
        polarity = int(row["polarity"])

        values = (
            features[:, feature_id]
            * polarity
        )

        if np.all(values == values[0]):
            test_auc = 0.5
        else:
            test_auc = roc_auc_score(
                labels,
                values,
            )

        rows.append(
            {
                "feature_id": feature_id,
                "polarity": polarity,
                "train_detection_auroc": (
                    row["train_detection_auroc"]
                ),
                "test_auroc": test_auc,
                "train_mean_difference": (
                    row["train_mean_difference"]
                ),
                "train_frequency_difference": (
                    row["train_frequency_difference"]
                ),
            }
        )

    return pd.DataFrame(rows)