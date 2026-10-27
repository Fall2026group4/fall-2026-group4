"""Scoring honesty directions on held-out pairs."""

import numpy as np
import pandas as pd


def pair_accuracy(scores, meta):
    """Share of pairs where the honest prompt scores above the dishonest one (0.5 = chance)."""
    s = pd.Series(scores, index=meta["row"].values)
    h = meta[meta["label"] == 1].set_index("pair_id")["row"]
    d = meta[meta["label"] == 0].set_index("pair_id")["row"]
    ids = sorted(set(h.index) & set(d.index))
    return float((s[h.loc[ids].values].values > s[d.loc[ids].values].values).mean())


def auroc(scores, meta):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(meta["label"].values, scores))


def bootstrap_ci(scores, meta, n_boot=1000, seed=0):
    """95% intervals for pair accuracy and AUROC, resampling whole PAIRS."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    s = pd.Series(scores, index=meta["row"].values)
    h = meta[meta["label"] == 1].set_index("pair_id")["row"]
    d = meta[meta["label"] == 0].set_index("pair_id")["row"]
    ids = np.array(sorted(set(h.index) & set(d.index)))
    sh, sd = s[h.loc[ids].values].values, s[d.loc[ids].values].values

    accs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        accs.append((sh[idx] > sd[idx]).mean())
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        aucs.append(roc_auc_score(y, np.r_[sh[idx], sd[idx]]))
    return {"pair_acc_low": np.percentile(accs, 2.5), "pair_acc_high": np.percentile(accs, 97.5),
            "auroc_low": np.percentile(aucs, 2.5), "auroc_high": np.percentile(aucs, 97.5)}


def random_band(X_train, meta_train, X_test, meta_test, directions):
    """Test pair accuracy and AUROC of random directions, each signed on TRAIN like a real method."""
    accs, aucs = [], []
    for w in directions:
        sign = 1.0 if pair_accuracy(X_train @ w, meta_train) >= 0.5 else -1.0
        scores = X_test @ (sign * w)
        accs.append(pair_accuracy(scores, meta_test))
        aucs.append(auroc(scores, meta_test))
    return np.array(accs), np.array(aucs)

def grouped_cv(method_fit, X, meta, n_folds=5, seed=0):
    """Pair accuracy across folds that never split a pair (robustness to the one fixed split)."""
    from sklearn.model_selection import GroupKFold

    accs = []
    for tr, te in GroupKFold(n_splits=n_folds).split(X, meta["label"], groups=meta["pair_id"]):
        m_tr = meta.iloc[tr].assign(row=np.arange(len(tr)))
        m_te = meta.iloc[te].assign(row=np.arange(len(te)))
        _, scorer = method_fit(X[tr], m_tr)
        accs.append(pair_accuracy(scorer(X[te]), m_te))
    return float(np.mean(accs)), float(np.std(accs))