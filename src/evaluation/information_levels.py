"""M2 information-faithfulness helpers: probe families, exact linear decomposition, noise and PCA references, grouped
paired bootstrap, and the pre-registered four-level cell labels.

Rules (results/shared/m2_four_level_faithfulness/preregistration.json):
  Preserved              frozen mass-mean gap CI and refit-logistic gap CI both within [-0.02, 0.02]
  Readout shifted        frozen logistic gap CI low > 0, but logistic refitted on x_hat recovers (refit gap CI high < 0.02)
  Reduced for F          refit-on-x_hat AUROC < refit-on-x AUROC for family F (gap CI low > 0)
  Not recovered by tested families
                         Reduced for EVERY family (logistic, sparse-z, MLP, kNN), every family >= 0.75 on x, the
                         learning curve does not still rise on x_hat while x is saturated, and the refit-logistic gap
                         exceeds that of the PCA reconstruction at matched OWT NMSE
  Displacement (flag)    the frozen original logistic direction applied to the error e alone has AUROC CI low > 0.5
"Irrecoverable" is never claimed: these labels describe the probe families tested, with finite samples.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

C_GRID = (0.01, 0.1, 1.0)
FAMILIES = ('logistic', 'sparse_z', 'mlp', 'knn')
TOL = 0.02


# ------------------------------------------------------------------------------------------------ AUROC

def auroc(y, s):
    """Mann-Whitney AUROC with tie-averaged ranks (equals sklearn roc_auc_score)."""
    y = np.asarray(y).astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_members(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]


def bootstrap_indices(groups, n_boot, rng):
    m = group_members(groups)
    return [np.concatenate([m[g] for g in rng.integers(len(m), size=len(m))]) for _ in range(n_boot)]


def boot_auroc(y, s, idx_list):
    return np.array([auroc(y[i], s[i]) for i in idx_list])


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def p_greater(a):
    """One-sided bootstrap p for 'mean > 0' (fraction of resamples <= 0, floored at 1/(n+1))."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float((1 + (a <= 0).sum()) / (1 + len(a)))


# ---------------------------------------------------------------------------------------------- probes

def logistic(C, seed=42):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=seed))


def select_C(X, y, groups, make, grid=C_GRID, folds=3):
    """Choose C by grouped CV on TRAIN rows only (ties -> smaller C)."""
    best = None
    for C in grid:
        sc = np.zeros(len(y))
        for a, b in GroupKFold(folds).split(X, y, groups):
            sc[b] = make(C).fit(X[a], y[a]).decision_function(X[b])
        v = auroc(y, sc)
        if best is None or v > best[0] + 1e-12:
            best = (v, C)
    return best[1]


def fit_logistic(Xtr, ytr, gtr):
    C = select_C(Xtr, ytr, gtr, logistic)
    return logistic(C).fit(Xtr, ytr), C


def fit_sparse_z(Ztr, ytr, gtr):
    make = lambda C: make_pipeline(MaxAbsScaler(), LogisticRegression(C=C, max_iter=3000, random_state=42))
    C = select_C(Ztr, ytr, gtr, make)
    return make(C).fit(Ztr, ytr), C


def mlp_scores(Xtr, ytr, Xte, seeds=(0, 1, 2)):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    out = [MLPClassifier((64,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=s).fit(a, ytr)
           .predict_proba(b)[:, 1] for s in seeds]
    return np.mean(out, 0)


def knn_scores(Xtr, ytr, Xte, k=15):
    sc = StandardScaler().fit(Xtr)
    return KNeighborsClassifier(k).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


def massmean_direction(X, y):
    X = np.asarray(X, np.float64)
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def linear_form(pipe):
    """For StandardScaler + LogisticRegression: decision_function(x) == x @ a + c exactly (up to float error)."""
    sc, lr = pipe[0], pipe[-1]
    a = lr.coef_[0] / sc.scale_
    c = lr.intercept_[0] - (sc.mean_ / sc.scale_) @ lr.coef_[0]
    return a, c


# ------------------------------------------------------------------------------------------ references

def norm_matched_noise(e, n_draws, rng):
    """Isotropic random vectors with the same per-row norm as e: array [n_draws, n, d]."""
    g = rng.standard_normal((n_draws,) + e.shape).astype(np.float32)
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    return g * np.linalg.norm(e, axis=-1, keepdims=True)[None]


def noise_projection(e, direction, n_draws, rng):
    """Exact distribution of direction . eta for eta uniform on the sphere of radius ||e_i||: [n_draws, n].

    Linear scores of x + eta equal score(x) + direction . eta, so only this projection is needed. Drawn by sampling
    the full vector (no approximation)."""
    out = np.empty((n_draws, len(e)), np.float64)
    for k in range(n_draws):
        out[k] = norm_matched_noise(e, 1, rng)[0] @ direction
    return out


def pca_basis(mean, cov):
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def pca_k_for_nmse(eigvals, target_nmse):
    """Smallest k whose residual variance fraction is <= target NMSE."""
    frac = 1 - np.cumsum(eigvals) / eigvals.sum()
    return int(np.argmax(frac <= target_nmse) + 1)


def pca_k_for_val_nmse(Xval, mean, V, target_nmse):
    """Smallest k whose NMSE on held-out rows Xval (normalised by Xval's own variance) is <= target.

    Exact: ||xc - P_k P_k^T xc||^2 = ||xc||^2 - sum_{j<=k} (xc . v_j)^2 with xc = x - mean (PCA mean)."""
    Xval = np.asarray(Xval, np.float64)
    xc = Xval - mean
    energy = ((xc @ V) ** 2).sum(0)
    denom = ((Xval - Xval.mean(0)) ** 2).sum()
    nm = ((xc ** 2).sum() - np.cumsum(energy)) / denom
    ok = np.flatnonzero(nm <= target_nmse)
    return int(ok[0] + 1) if len(ok) else V.shape[1]


def pca_reconstruct(X, mean, V, k):
    P = V[:, :k]
    return ((X - mean) @ P) @ P.T + mean


# -------------------------------------------------------------------------------------------- labels

def within(ci_pair, tol=TOL):
    lo, hi = ci_pair
    return bool(np.isfinite(lo) and lo >= -tol and hi <= tol)


def label_cell(s):
    """s: dict with
         informative (bool), frozen_massmean_gap_ci, frozen_logistic_gap_ci,
         refit_gap_ci = {family: (lo, hi)}, original_auroc = {family: auroc on x},
         lc_sample_inefficient (bool), refit_logistic_gap, pca_refit_logistic_gap (or None for PCA sources)
       Returns (label, reduced_families)."""
    if not s['informative']:
        return 'Uninformative (original refit logistic AUROC < 0.60)', []
    reduced = [f for f, c in s['refit_gap_ci'].items() if np.isfinite(c[0]) and c[0] > 0]
    if within(s['frozen_massmean_gap_ci']) and within(s['refit_gap_ci']['logistic']):
        return 'Preserved', reduced
    fams = [f for f in FAMILIES if f in s['refit_gap_ci']]
    all_reduced = len(fams) == len(FAMILIES) and set(fams) <= set(reduced)
    if all_reduced and all(s['original_auroc'][f] >= 0.75 for f in fams) and not s['lc_sample_inefficient'] \
            and s.get('pca_refit_logistic_gap') is not None and s['refit_logistic_gap'] > s['pca_refit_logistic_gap']:
        return 'Not recovered by tested families', reduced
    if s['frozen_logistic_gap_ci'][0] > 0 and s['refit_gap_ci']['logistic'][1] < TOL:
        return 'Readout shifted', reduced
    if reduced:
        return 'Reduced for ' + '+'.join(reduced), reduced
    return 'Inconclusive', reduced


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q
