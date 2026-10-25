"""Stage 2 (baseline per dataset and layer) and stage 3 (transfer, negation, joint training)."""

import numpy as np
import pandas as pd

from src.got import probes
from src.got.eda import grouped_auroc
from src.repe.activations import POSITIONS

SAE_POSITIONS = [f"L{layer}" for layer in range(1, 12)]  # where the SAEs attach; L0 is embeddings only


def load_all(directory, datasets, read="last"):
    """{name: (acts [n, 13, 768], meta)} for every dataset."""
    from src.got.activations import load_set

    out = {}
    for name in datasets:
        last, mean, meta = load_set(directory, name)
        out[name] = (last if read == "last" else mean, meta)
    return out


def _split(acts, meta, p, split):
    mask = (meta["split"] == split).values
    return acts[mask, p], meta[mask].reset_index(drop=True)


def _within_n1(meta, scores):
    return grouped_auroc(meta["label"].values, scores, meta["n1"].values) if "n1" in meta.columns else None


# ---------------------------------------------------------------------------
# Stage 2: baseline per dataset and layer
# ---------------------------------------------------------------------------

def baseline(data, read, n_boot=500, n_random=200):
    rows, band = [], []
    for name, (acts, meta) in data.items():
        for p, position in enumerate(POSITIONS):
            X_tr, m_tr = _split(acts, meta, p, "train")
            X_te, m_te = _split(acts, meta, p, "test")
            for method in probes.METHODS:
                scores = probes.fit(method, X_tr, m_tr["label"].values)(X_te)
                lo, hi = probes.bootstrap_ci(m_te["label"].values, scores, m_te["subject"].values, n_boot=n_boot)
                rows.append({"dataset": name, "read": read, "position": position, "position_index": p,
                             "method": method,
                             "auroc": round(probes.auroc(m_te["label"].values, scores), 4),
                             "auroc_low": round(lo, 4), "auroc_high": round(hi, 4),
                             "auroc_within_n1": _within_n1(m_te, scores)})
            aucs = probes.random_band(X_tr, m_tr["label"].values, X_te, m_te["label"].values, n=n_random)
            band.append({"dataset": name, "read": read, "position": position, "position_index": p,
                         "random_low": round(np.percentile(aucs, 2.5), 4),
                         "random_high": round(np.percentile(aucs, 97.5), 4)})
        print(f"  baseline: {name} ({read}) done")
    return pd.DataFrame(rows), pd.DataFrame(band)


def save_directions(data, read, directory):
    """Unit directions per method and layer, fitted on train, for the SAE and steering stages."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (acts, meta) in data.items():
        out = {}
        for method in probes.METHODS:
            out[method] = np.stack([
                probes.direction(method, *(_xy(acts, meta, p))) for p in range(len(POSITIONS))
            ]).astype(np.float32)
        np.savez(directory / f"{name}_{read}.npz", **out)


def _xy(acts, meta, p):
    X_tr, m_tr = _split(acts, meta, p, "train")
    return X_tr, m_tr["label"].values


# ---------------------------------------------------------------------------
# Stage 3: transfer, negation, joint training
# ---------------------------------------------------------------------------

def transfer(data, methods=("mass_mean", "logistic")):
    """Fit on each SOURCE dataset's train split, score every TARGET dataset's test split, same layer."""
    rows = []
    for source, (acts_s, meta_s) in data.items():
        for p, position in enumerate(POSITIONS):
            X_tr, m_tr = _split(acts_s, meta_s, p, "train")
            scorers = {m: probes.fit(m, X_tr, m_tr["label"].values) for m in methods}
            for target, (acts_t, meta_t) in data.items():
                X_te, m_te = _split(acts_t, meta_t, p, "test")
                for method, scorer in scorers.items():
                    s = scorer(X_te)
                    rows.append({"source": source, "target": target, "position": position,
                                 "position_index": p, "method": method,
                                 "auroc": round(probes.auroc(m_te["label"].values, s), 4),
                                 "auroc_within_n1": _within_n1(m_te, s)})
        print(f"  transfer: fitted on {source} done")
    return pd.DataFrame(rows)


def joint_training(data, family, method="mass_mean"):
    """Fit on a dataset AND its twin together; test on the other families' datasets."""
    rows = []
    for fam in dict.fromkeys(family.values()):
        names = [n for n in data if family[n] == fam]
        others = [n for n in data if family[n] != fam]
        for p, position in enumerate(POSITIONS):
            parts = [_split(*data[n], p, "train") for n in names]
            X = np.concatenate([x for x, _ in parts])
            y = np.concatenate([m["label"].values for _, m in parts])
            scorer = probes.fit(method, X, y)
            for target in others:
                X_te, m_te = _split(*data[target], p, "test")
                rows.append({"trained_on": "+".join(names), "target": target, "position": position,
                             "position_index": p, "method": method,
                             "auroc": round(probes.auroc(m_te["label"].values, scorer(X_te)), 4)})
    return pd.DataFrame(rows)


def best_layers(transfer_df, method="mass_mean", top=3):
    """Rank SAE layers by mean AUROC on a DIFFERENT dataset than the one fitted on.

    Raw AUROC is used, so a direction that FLIPS on the twin (AUROC < 0.5) is penalised.
    """
    off = transfer_df[(transfer_df["source"] != transfer_df["target"]) &
                      (transfer_df["method"] == method) & (transfer_df["position"].isin(SAE_POSITIONS))]
    t = (off.groupby("position")["auroc"].agg(["mean", "min"]).round(4)
         .rename(columns={"mean": "mean_transfer_auroc", "min": "worst_transfer_auroc"})
         .sort_values("mean_transfer_auroc", ascending=False).reset_index())
    t["rank"] = range(1, len(t) + 1)
    t["selected"] = t["rank"] <= top
    return t