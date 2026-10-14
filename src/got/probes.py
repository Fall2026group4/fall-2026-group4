"""Truth directions and their evaluation for the Geometry of Truth datasets.

All directions are fitted on TRAIN statements only.
  mass_mean : mean(true) - mean(false)              (the Geometry of Truth paper's probe)
  logistic  : L2 logistic regression, standardised  (C = 0.1, strong regularisation)
  pca       : top principal component of the training activations, signed so true scores higher
              (unsupervised: does truth appear as the main axis of variation?)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

METHODS = ["mass_mean", "logistic", "pca"]


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fit(method, X, y):
    """Returns a scorer: scorer(X) -> one score per row (higher = more 'true')."""
    y = np.asarray(y)
    if method == "mass_mean":
        w = _unit(X[y == 1].mean(0) - X[y == 0].mean(0))
        return lambda Z: Z @ w
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X, y)
        return model.decision_function
    if method == "pca":
        centred = X - X.mean(0)
        w = np.linalg.svd(centred, full_matrices=False)[2][0]
        if roc_auc_score(y, X @ w) < 0.5:
            w = -w
        return lambda Z: Z @ w
    raise ValueError(method)


def direction(method, X, y):
    """The unit direction for mass_mean / pca, or the logistic weights mapped back to activation space."""
    y = np.asarray(y)
    if method == "mass_mean":
        return _unit(X[y == 1].mean(0) - X[y == 0].mean(0))
    if method == "pca":
        w = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
        return w if roc_auc_score(y, X @ w) >= 0.5 else -w
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(scaler.transform(X), y)
    return _unit(clf.coef_[0] / scaler.scale_)


def auroc(y, scores):
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")


def fast_auroc(y, S):
    """AUROC for many score columns at once (S: n x k) via ranks; equals roc_auc_score per column."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    S = np.asarray(S).reshape(len(y), -1)
    n1, n0 = y.sum(), (~y).sum()
    ranks = rankdata(S, axis=0)
    return (ranks[y].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, scores, subjects, n_boot=500, seed=0):
    """95% interval for AUROC, resampling whole SUBJECTS (a city's true and false versions stay together)."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    codes, _ = pd.factorize(np.asarray(subjects))
    n_groups = codes.max() + 1
    aucs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        idx = np.repeat(np.arange(len(y)), counts[codes])  # each statement repeated as often as its subject
        if len(set(y[idx])) == 2:
            aucs.append(fast_auroc(y[idx], scores[idx])[0])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def random_band(X_tr, y_tr, X_te, y_te, n=200, seed=0):
    """Test AUROC of random unit directions, each signed on TRAIN (the 'no signal' reference)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, X_tr.shape[1]))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    tr_auc = fast_auroc(y_tr, X_tr @ W.T)
    W[tr_auc < 0.5] *= -1
    return fast_auroc(y_te, X_te @ W.T)