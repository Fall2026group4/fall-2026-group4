"""Stage 5 (RepE): can a probe still read honesty after the SAE?

Main test = CROSS-WORDING TRANSFER (as in Stage 3): train on one length-matched set's TRAIN
pairs, test on a set with DIFFERENT persona words (TEST pairs only), so the probe cannot just
detect a word. Also reported: within-set (mostly word detection) and a wording control.
  inputs : original, reconstruction, error (original - reconstruction), features (all latents),
           top-k features (k = 1, 5, 20, 100), ranked on the training pairs
  probes : mass_mean (difference of class means) and logistic regression
  scores : AUROC across prompts (main) and pair accuracy (near ceiling, for reference)
Faithfulness gap = original minus reconstruction (logistic transfer AUROC).

Run from the repo root:
    python scripts/run_repe_s5.py --quick    # position 6 only
    python scripts/run_repe_s5.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_repe_s4 as s4  # reuse SAE loading and data code

TOP_K = [1, 5, 20, 100]
MATCHED = [s for s in s4.SETS if s.startswith("matched_")]
TRANSFER = [("matched_honest_lying", "matched_truthful_deceptive"),
            ("matched_truthful_deceptive", "matched_honest_lying"),
            ("matched_honest_dishonest", "matched_truthful_lying"),
            ("matched_truthful_lying", "matched_honest_dishonest")]
CTRL_A, CTRL_B = "matched_honest_lying", "matched_truthful_lying"
INPUT_ORDER = ["original", "reconstruction", "error", "features"] + [f"top{k}" for k in TOP_K]


# ---------- SAE ----------
@torch.no_grad()
def encode_decode(sae, x, batch=256):
    zs, xhs = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dtype=sae.W_enc.dtype)
        z = sae.encode(xb)
        xh = sae.decode(z)
        zs.append(sp.csr_matrix(z.float().numpy()))
        xhs.append(xh.float().numpy())
    return sp.vstack(zs).tocsr(), np.concatenate(xhs)


# ---------- tasks: rows index into the named sets stacked in order ----------
def split_of(d, idx):
    return d["meta"]["split"].astype(str).to_numpy()[idx]


def within_task(data, name):
    d = data[name]
    return dict(kind="within", sets=(name,), pos=d["h"], neg=d["u"],
                train=split_of(d, d["h"]) == "train")


def transfer_task(data, a, b):
    da, db = data[a], data[b]
    tr = split_of(da, da["h"]) == "train"
    te = split_of(db, db["h"]) == "test"
    off = len(da["x"])
    return dict(kind="transfer", sets=(a, b),
                pos=np.r_[da["h"][tr], off + db["h"][te]],
                neg=np.r_[da["u"][tr], off + db["u"][te]],
                train=np.r_[np.ones(tr.sum(), bool), np.zeros(te.sum(), bool)])


def wording_task(data):
    """Control: honest prompts worded 'honest' (set A) vs 'truthful' (set B), same statement."""
    ma, mb = data[CTRL_A]["meta"], data[CTRL_B]["meta"]
    a = ma[ma["label"] == 1].reset_index().set_index("statement")["index"]
    b = mb[mb["label"] == 1].reset_index().set_index("statement")["index"]
    common = a.index.intersection(b.index)
    ia, ib = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    off = len(data[CTRL_A]["x"])
    return dict(kind="control", sets=(CTRL_A, CTRL_B), pos=ia, neg=off + ib,
                train=split_of(data[CTRL_A], ia) == "train")


# ---------- probes ----------
def mass_mean_dir(X, y):
    if sp.issparse(X):
        return (np.asarray(X[np.where(y == 1)[0]].mean(0)).ravel()
                - np.asarray(X[np.where(y == 0)[0]].mean(0)).ravel())
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def logistic_scores(Xtr, ytr, Xte):
    if sp.issparse(Xtr):
        sc, solver = MaxAbsScaler().fit(Xtr), "liblinear"
    else:
        sc, solver = StandardScaler().fit(Xtr), "lbfgs"
    clf = LogisticRegression(C=1.0, max_iter=2000, solver=solver)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def top_features(z, pos_tr, neg_tr, k):
    diff = (z[pos_tr] - z[neg_tr]).toarray()
    score = diff.mean(0) / (diff.std(0) + 1e-6)
    return np.argsort(-np.abs(score))[:k]


def evaluate(X_all, z, task):
    pos, neg, tr = task["pos"], task["neg"], task["train"]
    p_tr, n_tr, p_te, n_te = pos[tr], neg[tr], pos[~tr], neg[~tr]
    rows_tr, rows_te = np.r_[p_tr, n_tr], np.r_[p_te, n_te]
    y_tr = np.r_[np.ones(len(p_tr)), np.zeros(len(n_tr))]
    y_te = np.r_[np.ones(len(p_te)), np.zeros(len(n_te))]
    m = len(p_te)

    inputs = dict(X_all)
    inputs["features"] = z
    for k in TOP_K:
        inputs[f"top{k}"] = z[:, top_features(z, p_tr, n_tr, k)].toarray()

    out = []
    for name in INPUT_ORDER:
        X = inputs[name]
        Xtr, Xte = X[rows_tr], X[rows_te]
        scores = {"mass_mean": np.asarray(Xte @ mass_mean_dir(Xtr, y_tr)).ravel()}
        if name != "top1":
            scores["logistic"] = logistic_scores(Xtr, y_tr, Xte)
        for method, s in scores.items():
            out.append(dict(input=name, method=method, auroc=float(roc_auc_score(y_te, s)),
                            pair_acc=float((s[:m] > s[m:]).mean()),
                            n_train_pairs=len(p_tr), n_test_pairs=m))
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="position 6 only")
    args = ap.parse_args()

    data = {n: s4.load_set(n) for n in s4.SETS}
    tasks = {n: within_task(data, n) for n in s4.SETS}
    for a, b in TRANSFER:
        tasks[f"{a}->{b}"] = transfer_task(data, a, b)
    tasks["control_wording"] = wording_task(data)
    for n, t in tasks.items():
        print(f"{t['kind']:8s} {n}: {t['train'].sum()} train pairs, {(~t['train']).sum()} test pairs")

    rows = []
    for st, release in s4.RELEASES.items():
        todo = s4.sae_ids(release)
        if args.quick:
            todo = [p for p in todo if p[0] == 6]
        print(f"\n{st}: {len(todo)} SAEs")
        for pos, sid in todo:
            t0 = time.time()
            sae = s4.load_sae(release, sid)
            enc = {}
            for n in data:
                x = data[n]["x"][:, pos, :]
                z, xh = encode_decode(sae, x)
                enc[n] = (x, xh, z)
            for tname, task in tasks.items():
                x = np.vstack([enc[n][0] for n in task["sets"]])
                xh = np.vstack([enc[n][1] for n in task["sets"]])
                z = sp.vstack([enc[n][2] for n in task["sets"]]).tocsr()
                X_all = {"original": x, "reconstruction": xh, "error": x - xh}
                for r in evaluate(X_all, z, task):
                    rows.append(dict(sae_type=st, position=pos, kind=task["kind"], task=tname, **r))
            df = pd.DataFrame(rows)
            cur = df[(df.sae_type == st) & (df.position == pos) & (df.method == "logistic")]
            tr = cur[cur.kind == "transfer"].groupby("input")["auroc"].mean()
            ct = cur[cur.kind == "control"].set_index("input")["auroc"]
            print(f"  P{pos:02d}  TRANSFER AUROC: orig {tr['original']:.2f} recon {tr['reconstruction']:.2f} "
                  f"error {tr['error']:.2f} features {tr['features']:.2f}  |  wording control: orig "
                  f"{ct['original']:.2f} recon {ct['reconstruction']:.2f}  ({time.time() - t0:.0f}s)")
            del sae

    df = pd.DataFrame(rows)
    if args.quick:
        lg = df[(df.method == "logistic") & (df.kind != "within")]
        print("\n" + lg.groupby(["sae_type", "kind", "input"])["auroc"].mean().round(3).to_string())
        print("\nQuick test done.")
        return

    s4.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(s4.TAB_DIR / "repe_s5_probes.csv", index=False)
    print("\nSaved results/tables/repe/repe_s5_probes.csv")

    lg = df[df.method == "logistic"]
    for kind, title in [("transfer", "HONESTY, cross-wording transfer (main test)"),
                        ("within", "Honesty, within-set (mostly word detection), matched sets"),
                        ("control", "Control: honest vs truthful wording")]:
        sub = lg[lg.kind == kind]
        if kind == "within":
            sub = sub[sub.task.isin(MATCHED)]
        t = sub.pivot_table(index=["sae_type", "position"], columns="input", values="auroc")
        t = t[[c for c in INPUT_ORDER if c in t.columns]]
        t["gap"] = t["original"] - t["reconstruction"]
        print(f"\n{title}: logistic AUROC")
        print(t.round(3).to_string())
    top1 = df[(df.input == "top1") & (df.kind == "transfer")].groupby(["sae_type", "position"])["auroc"].mean()
    print("\nBest single feature (top1), transfer AUROC:")
    print(top1.round(3).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()