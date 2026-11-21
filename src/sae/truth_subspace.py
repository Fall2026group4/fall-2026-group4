"""Truth-subspace utilities for the TSP (truth-subspace-preserving) study (2026-11-14).

A "truth subspace" S is a rank-r orthonormal basis P (d x r) estimated ONLY from GoT-train group A rows:
  * the per-polarity class-mean contrasts mu(T, pol) - mu(F, pol) for cities and neg_cities, and their pooled contrast;
  * the raw-space weight vectors of logistic probes fitted on grouped bootstrap resamples of group A
    (a multi-probe estimate of the directions a linear reader uses).
The span of these vectors is orthonormalised by SVD and the top-r left singular vectors are kept.

Controls with the same rank:
  * ``random_subspace``: Haar-random orthonormal basis (label-free, geometry-free);
  * ``pca_subspace``: top-r principal directions of the same group-A activations (label-free, high variance).

Nothing here touches group B, validation or test rows; callers are responsible for passing group-A rows only.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _orthonormal_top(vectors, r):
    """Top-r left singular vectors of the column-stacked (unit-normalised) vectors."""
    V = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vectors], axis=1)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    r = min(r, int((s > 1e-8 * s[0]).sum()))
    return U[:, :r].astype(np.float64), s


def logistic_raw_weight(X, y, C=0.1, seed=0):
    """Raw-space weight of a standardised logistic probe: w_raw = w_std / sigma."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))
    pipe.fit(X, y)
    sc, lr = pipe[0], pipe[1]
    return lr.coef_[0] / sc.scale_


def estimate_truth_subspace(X, y, polarity, groups, r=8, n_boot=24, C=0.1, seed=0):
    """Return (P, info). X: (n, d) group-A activations; y: truth labels; polarity: 0 cities / 1 neg_cities;
    groups: entity ids for grouped bootstrap."""
    X = np.asarray(X, dtype=np.float64)
    y, polarity, groups = np.asarray(y), np.asarray(polarity), np.asarray(groups)
    vecs = []
    for pol in np.unique(polarity):
        m = polarity == pol
        vecs.append(X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0))
    vecs.append(X[y == 1].mean(0) - X[y == 0].mean(0))
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    for b in range(n_boot):
        pick = rng.choice(u, len(u), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vecs.append(logistic_raw_weight(X[idx], y[idx], C=C, seed=b))
    P, s = _orthonormal_top(vecs, r)
    return P, dict(n_vectors=len(vecs), singular_values=s[:max(r, 1)].tolist(), rank=int(P.shape[1]))


def random_subspace(d, r, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, r)))
    return q.astype(np.float64)


def pca_subspace(X, r):
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, 0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:r].T.copy()


def subspace_rel_error(X, H, P, center=None):
    """Share of the centred signal energy inside span(P) that the reconstruction fails to reproduce:
    sum ||P^T (x - x_hat)||^2 / sum ||P^T (x - center)||^2  (0 = perfect inside S)."""
    X, H = np.asarray(X, np.float64), np.asarray(H, np.float64)
    c = X.mean(0) if center is None else center
    num = ((X - H) @ P) ** 2
    den = ((X - c) @ P) ** 2
    return float(num.sum() / den.sum())


def variance_share(cov, P):
    """Average variance per direction in span(P) relative to the average per-direction variance (1 = typical)."""
    per_dir = np.trace(P.T @ cov @ P) / P.shape[1]
    return float(per_dir / (np.trace(cov) / cov.shape[0]))


def subspace_loss(x, x_hat, P, denom):
    """Torch: E||P^T (x - x_hat)||^2 / denom, where denom = E||P^T (x - mean)||^2 on the training rows."""
    return ((x - x_hat) @ P).pow(2).sum(-1).mean() / denom
