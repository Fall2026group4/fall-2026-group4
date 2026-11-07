"""Honesty directions fit on TRAINING pairs only.

repe_pca  : RepE's method - PCA on within-pair differences (random sign flips),
            first component, signed so honest prompts score higher.
mean_diff : mean(honest) - mean(dishonest).
logistic  : L2 logistic regression on standardised activations (C fixed at 0.1,
            strong regularisation because there are fewer prompts than dimensions).
"""

import numpy as np

METHODS = ["repe_pca", "mean_diff", "logistic"]


def pair_arrays(X, meta):
    """Honest and dishonest rows aligned by pair_id: (H, D, pair_ids)."""
    h = meta[meta["label"] == 1].set_index("pair_id")
    d = meta[meta["label"] == 0].set_index("pair_id")
    ids = sorted(set(h.index) & set(d.index))
    return X[h.loc[ids, "row"].values], X[d.loc[ids, "row"].values], np.array(ids)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fit_repe_pca(X, meta, seed=0):
    H, D, _ = pair_arrays(X, meta)
    diffs = H - D
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(diffs))[:, None]
    flipped = diffs * signs
    flipped = flipped - flipped.mean(axis=0)
    _, _, vt = np.linalg.svd(flipped, full_matrices=False)
    w = vt[0]
    if (diffs @ w).mean() < 0:  # honest should score higher
        w = -w
    return _unit(w)


def fit_mean_diff(X, meta):
    H, D, _ = pair_arrays(X, meta)
    return _unit(H.mean(axis=0) - D.mean(axis=0))


def fit_logistic(X, meta, C=0.1):
    """Returns the probe's direction in the ORIGINAL activation space, plus the fitted model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    model.fit(X, meta["label"].values)
    scaler, clf = model[0], model[1]
    return _unit(clf.coef_[0] / scaler.scale_), model


def fit(method, X, meta):
    """Fit one method. Returns (direction, scorer) where scorer(X) gives a score per row."""
    if method == "logistic":
        w, model = fit_logistic(X, meta)
        return w, model.decision_function
    w = fit_repe_pca(X, meta) if method == "repe_pca" else fit_mean_diff(X, meta)
    return w, lambda Z, w=w: Z @ w


def random_directions(n, dim=768, seed=0):
    """n random unit vectors (the 'no signal' reference)."""
    v = np.random.default_rng(seed).normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)