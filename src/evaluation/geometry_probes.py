"""Linear probe utilities for Geometry of Truth activations."""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _to_numpy(x):
    """Convert a torch tensor or array-like object to NumPy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def evaluate_linear_probes(
    train_activations,
    train_labels,
    test_activations,
    test_labels,
    seed=42,
):
    """Train and evaluate one linear probe per GPT-2 layer.

    Parameters
    ----------
    train_activations : array-like
        Shape [n_train, n_layers, hidden_dim].

    train_labels : array-like
        Binary labels for the training set.

    test_activations : array-like
        Shape [n_test, n_layers, hidden_dim].

    test_labels : array-like
        Binary labels for the test set.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Layer-wise Accuracy and AUROC results.

    dict
        Trained probe pipeline for each layer.
    """

    train_activations = _to_numpy(train_activations)
    test_activations = _to_numpy(test_activations)

    train_labels = _to_numpy(train_labels).reshape(-1)
    test_labels = _to_numpy(test_labels).reshape(-1)

    if train_activations.ndim != 3:
        raise ValueError(
            "train_activations must have shape "
            "[examples, layers, hidden_dim]"
        )

    if test_activations.ndim != 3:
        raise ValueError(
            "test_activations must have shape "
            "[examples, layers, hidden_dim]"
        )

    n_layers = train_activations.shape[1]

    results = []
    probes = {}

    for layer in range(n_layers):
        x_train = train_activations[:, layer, :]
        x_test = test_activations[:, layer, :]

        probe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )

        probe.fit(
            x_train,
            train_labels,
        )

        predictions = probe.predict(x_test)

        probabilities = probe.predict_proba(
            x_test
        )[:, 1]

        accuracy = accuracy_score(
            test_labels,
            predictions,
        )

        auroc = roc_auc_score(
            test_labels,
            probabilities,
        )

        results.append(
            {
                "layer": layer,
                "accuracy": accuracy,
                "auroc": auroc,
            }
        )

        probes[layer] = probe

        print(
            f"Layer {layer:02d} | "
            f"Accuracy: {accuracy:.4f} | "
            f"AUROC: {auroc:.4f}"
        )

    results_df = pd.DataFrame(results)

    return results_df, probes