"""Stage 5 utilities for causal steering of Geometry of Truth representations."""

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.sae.steering import make_steering_hook


def _to_numpy(x):
    """Convert torch tensor or array-like input to NumPy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def train_fixed_truth_probe(
    train_residuals,
    train_labels,
    seed=42,
):
    """Train the fixed Layer-10 truth probe used for steering evaluation."""

    x_train = _to_numpy(train_residuals)
    y_train = _to_numpy(train_labels).reshape(-1)

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

    return probe


def get_probe_scores(
    probe,
    residuals,
):
    """Return signed truth-probe scores.

    Positive = more TRUE-like.
    Negative = more FALSE-like.
    """

    residuals = _to_numpy(
        residuals
    )

    return probe.decision_function(
        residuals
    )


def feature_scale_from_train(
    train_features,
    feature_id,
):
    """Return training-set standard deviation for one SAE feature."""

    values = train_features[
        :,
        feature_id,
    ]

    if hasattr(values, "detach"):
        values = values.detach().cpu()

    scale = float(
        values.float().std().item()
    )

    if scale <= 0:
        raise ValueError(
            f"Feature {feature_id} has zero variance."
        )

    return scale


def get_sae_decoder_direction(
    sae,
    feature_id,
):
    """Return the decoder direction for one SAE feature."""

    return (
        sae.W_dec[feature_id]
        .detach()
        .cpu()
        .float()
    )


def apply_direction_steering(
    residuals,
    direction,
    steering_strength,
):
    """Apply the team's shared last-token steering hook.

    Cached residuals have shape [batch, d_model].
    We temporarily add a sequence dimension so that the same
    make_steering_hook() implementation used by the team can
    modify the final token only.
    """

    if residuals.ndim != 2:
        raise ValueError(
            "residuals must have shape [examples, hidden_dim]"
        )

    residuals = residuals.detach().cpu().float()

    if isinstance(direction, np.ndarray):
        direction = torch.tensor(
            direction,
            dtype=residuals.dtype,
        )
    else:
        direction = (
            direction
            .detach()
            .cpu()
            .to(dtype=residuals.dtype)
        )

    if direction.ndim != 1:
        raise ValueError(
            "direction must have shape [hidden_dim]"
        )

    if direction.shape[0] != residuals.shape[1]:
        raise ValueError(
            "direction and residual hidden dimensions do not match"
        )

    # Shape expected by make_steering_hook:
    # [batch, sequence, d_model]
    activation = (
        residuals
        .clone()
        .unsqueeze(1)
    )

    hook_fn = make_steering_hook(
        steering_direction=direction,
        steering_strength=steering_strength,
    )

    steered = hook_fn(
        activation,
        hook=None,
    )

    return steered[:, -1, :]


def apply_sae_feature_steering(
    sae,
    residuals,
    feature_id,
    steering_strength,
):
    """Steer along one SAE decoder direction."""

    direction = get_sae_decoder_direction(
        sae=sae,
        feature_id=feature_id,
    )

    return apply_direction_steering(
        residuals=residuals,
        direction=direction,
        steering_strength=steering_strength,
    )


def get_raw_probe_direction(
    probe,
):
    """Recover the truth-probe direction in raw residual coordinates."""

    scaler = probe.named_steps[
        "scaler"
    ]

    classifier = probe.named_steps[
        "classifier"
    ]

    standardized_direction = (
        classifier.coef_[0]
    )

    raw_direction = (
        standardized_direction
        / scaler.scale_
    )

    norm = np.linalg.norm(
        raw_direction
    )

    if norm == 0:
        raise ValueError(
            "Probe direction has zero norm."
        )

    return (
        raw_direction
        / norm
    )


def apply_probe_direction_control(
    residuals,
    probe,
    steering_strength,
    toward_true=True,
):
    """Positive-control intervention along the fixed truth-probe direction."""

    direction = get_raw_probe_direction(
        probe
    )

    if not toward_true:
        direction = -direction

    return apply_direction_steering(
        residuals=residuals,
        direction=direction,
        steering_strength=steering_strength,
    )