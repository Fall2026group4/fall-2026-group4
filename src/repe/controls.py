"""Stage 3 controls: does the honesty signal survive when the shortcuts are removed?

within-set : fit and test inside each prompt set (length-matched sets have NO position gap)
position   : how well token position ALONE predicts the label
transfer   : fit on one wording, test on another (and on false statements)
"""

import numpy as np
import pandas as pd

from src.repe import activations, directions, evaluate

SAE_POSITIONS = [f"L{layer}" for layer in range(1, 12)]  # where SAEs attach; L0 is the position check


def load_sets(directory, read="last"):
    """{name: (acts [n, 13, 768], meta)} for every cached prompt set, 'original' first."""
    names = sorted(p.stem for p in directory.glob("*.npz"))
    names = ["original"] + [n for n in names if n != "original"]
    out = {}
    for name in names:
        last, mean, meta = activations.load_prompt_set(directory, name)
        out[name] = (last if read == "last" else mean, meta.assign(row=np.arange(len(meta))))
    return out


def split_rows(X, meta, split):
    mask = (meta["split"] == split).values
    return X[mask], meta[mask].reset_index(drop=True).assign(row=lambda m: np.arange(len(m)))


def within_set(sets, n_random=200, n_boot=500):
    """AUROC per prompt set, layer and method, with CIs and a random-direction band."""
    rand = directions.random_directions(n_random)
    rows, band = [], []
    for name, (acts, meta) in sets.items():
        for p, position in enumerate(activations.POSITIONS):
            X_tr, m_tr = split_rows(acts[:, p], meta, "train")
            X_te, m_te = split_rows(acts[:, p], meta, "test")
            for method in directions.METHODS:
                _, scorer = directions.fit(method, X_tr, m_tr)
                scores = scorer(X_te)
                ci = evaluate.bootstrap_ci(scores, m_te, n_boot=n_boot)
                rows.append({"prompt_set": name, "position": position, "position_index": p,
                             "method": method, "auroc": round(evaluate.auroc(scores, m_te), 4),
                             "auroc_low": round(ci["auroc_low"], 4), "auroc_high": round(ci["auroc_high"], 4)})
            _, aucs = evaluate.random_band(X_tr, m_tr, X_te, m_te, rand)
            band.append({"prompt_set": name, "position": position, "position_index": p,
                         "random_auroc_low": round(np.percentile(aucs, 2.5), 4),
                         "random_auroc_high": round(np.percentile(aucs, 97.5), 4)})
        print(f"  within-set: {name} done")
    return pd.DataFrame(rows), pd.DataFrame(band)


def position_only(sets):
    """AUROC of a logistic probe that sees ONLY the prompt's token count."""
    from sklearn.linear_model import LogisticRegression

    rows = []
    for name, (_, meta) in sets.items():
        tr, te = meta[meta["split"] == "train"], meta[meta["split"] == "test"]
        clf = LogisticRegression().fit(tr[["n_tokens"]], tr["label"])
        scores = clf.decision_function(te[["n_tokens"]])
        rows.append({"prompt_set": name,
                     "mean_tokens_honest": round(meta.loc[meta["label"] == 1, "n_tokens"].mean(), 2),
                     "mean_tokens_dishonest": round(meta.loc[meta["label"] == 0, "n_tokens"].mean(), 2),
                     "position_only_auroc": round(evaluate.auroc(scores, te.assign(row=range(len(te)))), 4)})
    return pd.DataFrame(rows)


def transfer(sets):
    """Fit on each source set's TRAIN split, score every target set's TEST split, same layer."""
    rows = []
    for source, (acts_s, meta_s) in sets.items():
        for p, position in enumerate(activations.POSITIONS):
            X_tr, m_tr = split_rows(acts_s[:, p], meta_s, "train")
            fitted = {m: directions.fit(m, X_tr, m_tr)[1] for m in directions.METHODS}
            for target, (acts_t, meta_t) in sets.items():
                X_te, m_te = split_rows(acts_t[:, p], meta_t, "test")
                for method, scorer in fitted.items():
                    rows.append({"source": source, "target": target, "position": position,
                                 "position_index": p, "method": method,
                                 "auroc": round(evaluate.auroc(scorer(X_te), m_te), 4)})
        print(f"  transfer: fitted on {source} done")
    return pd.DataFrame(rows)


def best_layers(transfer_df, method="mean_diff", top=3):
    """Rank SAE layers by mean AUROC when the TEST wording differs from the TRAIN wording.

    Off-diagonal transfer rewards a direction that is not tied to one persona word or position.
    """
    off = transfer_df[(transfer_df["source"] != transfer_df["target"]) &
                      (transfer_df["method"] == method) &
                      (transfer_df["position"].isin(SAE_POSITIONS))]
    table = (off.groupby("position")["auroc"].agg(["mean", "min"]).round(4)
             .rename(columns={"mean": "mean_transfer_auroc", "min": "worst_transfer_auroc"})
             .sort_values("mean_transfer_auroc", ascending=False).reset_index())
    table["rank"] = range(1, len(table) + 1)
    table["selected"] = table["rank"] <= top
    return table