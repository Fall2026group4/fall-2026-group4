"""Stage 4B utilities for SAE reconstruction faithfulness."""

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def reconstruct_from_sae_features(
    sae,
    features,
    batch_size=64,
):
    """Decode SAE features back into residual-stream activations.

    Parameters
    ----------
    sae
        Loaded SAE Lens SAE.

    features : torch.Tensor
        Shape [n_examples, d_sae].

    batch_size : int
        Number of examples decoded at once.

    Returns
    -------
    torch.Tensor
        Reconstructed residual activations with shape
        [n_examples, 768].
    """

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [examples, d_sae]"
        )

    reconstructed_batches = []

    with torch.inference_mode():
        for start in range(
            0,
            features.shape[0],
            batch_size,
        ):
            batch = features[
                start:start + batch_size
            ]

            reconstructed = sae.decode(batch)

            reconstructed_batches.append(
                reconstructed.detach().cpu()
            )

    return torch.cat(
        reconstructed_batches,
        dim=0,
    )


def evaluate_reconstructed_probe(
    train_reconstructed,
    train_labels,
    test_reconstructed,
    test_labels,
    seed=42,
):
    """Train a linear probe on SAE-reconstructed residuals.

    The probe setup matches the Stage 2 baseline:
    StandardScaler + LogisticRegression.
    """

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    x_train = to_numpy(train_reconstructed)
    y_train = to_numpy(train_labels).reshape(-1)

    x_test = to_numpy(test_reconstructed)
    y_test = to_numpy(test_labels).reshape(-1)

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
        y_train,
    )

    predictions = probe.predict(x_test)

    probabilities = probe.predict_proba(
        x_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auroc = roc_auc_score(
        y_test,
        probabilities,
    )

    return {
        "accuracy": accuracy,
        "auroc": auroc,
    }