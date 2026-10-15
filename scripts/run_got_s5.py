"""Stage 5 (Geometry of Truth): can a probe still read truth after the SAE?

For each SAE type x position x dataset, trains the truth probe on four inputs:
  original        the GPT-2 activation (Stage 2 baseline, recomputed here)
  reconstruction  what the SAE rebuilds
  error           original - reconstruction (what the SAE throws away)
  features        the SAE's sparse features
Probes: mass_mean (difference of class means, as in Stage 2) and logistic regression.
Extra: the original direction applied to the reconstruction (transfer), and a null
control comparing the truth difference with differences between random halves.

Run on EC2 from the repo root:
    python scripts/run_got_s5.py --quick     # cities, position 6
    python scripts/run_got_s5.py             # full run (resumable)
"""
import argparse
import re
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

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "data" / "got" / "activations"
PART_DIR = ROOT / "data" / "got" / "sae_latents" / "_s5_parts"
TAB_DIR = ROOT / "results" / "tables" / "got"
FIG_DIR = ROOT / "results" / "figures" / "got"

DATASETS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans",
            "larger_than", "smaller_than"]
RELEASES = {"relu": "gpt2-small-res-jb", "topk": "gpt2-small-resid-post-v5-32k"}
GROUP_COLS = ["n1", "first_number", "num1"]   # for the numbers datasets (within-n1 AUROC)
INPUTS = ["original", "reconstruction", "error", "features"]
N_NULL = 20
SEED = 0


# ---------- SAEs (same as Stage 4) ----------
def position_for(sae_id):
    m = re.fullmatch(r"blocks\.(\d+)\.hook_resid_(pre|post)", sae_id)
    if not m:
        return None
    layer, kind = int(m.group(1)), m.group(2)
    return layer if kind == "pre" else layer + 1


def sae_ids(release):
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    ids = list(get_pretrained_saes_directory()[release].saes_map)
    pairs = [(position_for(i), i) for i in ids]
    return sorted((p, i) for p, i in pairs if p is not None and p > 0)


def load_sae(release, sae_id, device):
    from sae_lens import SAE
    out = SAE.from_pretrained(release, sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out
    sae.eval()
    return sae


@torch.no_grad()
def run_sae(sae, x, device, batch=256):
    dtype = sae.W_enc.dtype
    zs, xhs = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device=device, dtype=dtype)
        z = sae.encode(xb)
        xh = sae.decode(z)
        zs.append(sp.csr_matrix(z.float().cpu().numpy()))
        xhs.append(xh.float().cpu().numpy())
    return sp.vstack(zs).tocsr(), np.concatenate(xhs)


# ---------- data ----------
def load_dataset(name):
    x = np.load(ACT_DIR / f"{name}.npz")["last"]
    meta = pd.read_csv(ACT_DIR / f"{name}.csv")
    y = meta["label"].astype(int).to_numpy()
    split = meta["split"].astype(str).to_numpy()
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    if len(tr) == 0 or len(te) == 0:
        sys.exit(f"{name}: split column needs 'train' and 'test' rows; found {sorted(set(split))}")
    gcol = next((c for c in GROUP_COLS if c in meta.columns), None)
    g = meta[gcol].to_numpy() if gcol else None
    return dict(x=x, y=y, tr=tr, te=te, g=g, gcol=gcol)


# ---------- probes ----------
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def auc_within(y, s, g):
    if g is None:
        return np.nan
    total, acc = 0, 0.0
    for v in np.unique(g):
        m = g == v
        if len(np.unique(y[m])) == 2:
            acc += roc_auc_score(y[m], s[m]) * m.sum()
            total += m.sum()
    return acc / total if total else np.nan


def mass_mean_dir(X, y):
    if sp.issparse(X):
        return (np.asarray(X[np.where(y == 1)[0]].mean(0)).ravel()
                - np.asarray(X[np.where(y == 0)[0]].mean(0)).ravel())
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def project(X, w):
    return np.asarray(X @ w).ravel()


def logistic_scores(Xtr, ytr, Xte):
    if sp.issparse(Xtr):
        sc, solver = MaxAbsScaler().fit(Xtr), "liblinear"
    else:
        sc, solver = StandardScaler().fit(Xtr), "lbfgs"
    clf = LogisticRegression(C=1.0, max_iter=2000, solver=solver)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def evaluate(inputs, d):
    y, tr, te, g = d["y"], d["tr"], d["te"], d["g"]
    gte = g[te] if g is not None else None
    rows = []
    for name in INPUTS:
        X = inputs[name]
        Xtr, Xte = X[tr], X[te]
        scores = {"mass_mean": project(Xte, mass_mean_dir(Xtr, y[tr])),
                  "logistic": logistic_scores(Xtr, y[tr], Xte)}
        for method, s in scores.items():
            rows.append(dict(input=name, method=method, auroc=auc(y[te], s),
                             auroc_within_n1=auc_within(y[te], s, gte)))
    # transfer: the ORIGINAL truth direction, applied to the reconstruction
    w = mass_mean_dir(inputs["original"][tr], y[tr])
    s = project(inputs["reconstruction"][te], w)
    rows.append(dict(input="reconstruction", method="original_direction",
                     auroc=auc(y[te], s), auroc_within_n1=auc_within(y[te], s, gte)))
    return rows


# ---------- null control ----------
def lost_fraction(x, xh, lab):
    dd = x[lab == 1].mean(0) - x[lab == 0].mean(0)
    dh = xh[lab == 1].mean(0) - xh[lab == 0].mean(0)
    return float(np.linalg.norm(dd - dh) / (np.linalg.norm(dd) + 1e-8))


def null_control(x, xh, y, rng):
    truth = lost_fraction(x, xh, y)
    nulls = np.array([lost_fraction(x, xh, rng.permutation(y)) for _ in range(N_NULL)])
    return dict(lost_truth=truth, lost_null_mean=float(nulls.mean()),
                lost_null_min=float(nulls.min()), lost_null_max=float(nulls.max()),
                truth_lost_less_than_all_null=bool(truth < nulls.min()),
                truth_lost_more_than_all_null=bool(truth > nulls.max()))


# ---------- outputs ----------
def plot(probe, null):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mm = probe[probe.method == "mass_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, t in zip(axes[:2], ["relu", "topk"]):
        g = mm[mm.sae_type == t]
        for inp in INPUTS:
            m = g[g.input == inp].groupby("position")["auroc"].mean()
            ax.plot(m.index, m.values, marker="o", label=inp)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(f"{t}: truth probe AUROC (mass-mean, mean of 6 datasets)")
        ax.set_xlabel("position (12 = final output)")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
    axes[0].legend(title="probe input")
    ax = axes[2]
    for t, g in null.groupby("sae_type"):
        m = g.groupby("position")[["lost_truth", "lost_null_mean"]].mean()
        line, = ax.plot(m.index, m.lost_truth, marker="o", label=f"{t}: truth")
        ax.plot(m.index, m.lost_null_mean, ls="--", color=line.get_color(),
                label=f"{t}: random halves")
    ax.set_title("Share of the difference lost in SAE error")
    ax.set_xlabel("position (12 = final output)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "got_s5_probe_fidelity.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path.relative_to(ROOT)}")


def combine():
    probe = pd.concat([pd.read_csv(p) for p in sorted(PART_DIR.glob("probe_*.csv"))],
                      ignore_index=True)
    null = pd.concat([pd.read_csv(p) for p in sorted(PART_DIR.glob("null_*.csv"))],
                     ignore_index=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    probe.to_csv(TAB_DIR / "got_s5_probe_by_layer.csv", index=False)
    null.to_csv(TAB_DIR / "got_s5_null_control.csv", index=False)
    print(f"\nSaved got_s5_probe_by_layer.csv ({len(probe)} rows) and "
          f"got_s5_null_control.csv ({len(null)} rows) in results/tables/got")

    mm = probe[probe.method == "mass_mean"]
    print("\nTest AUROC, mass-mean, mean over 6 datasets:")
    print(mm.pivot_table(index=["sae_type", "position"], columns="input",
                         values="auroc")[INPUTS].round(3).to_string())
    tr = probe[probe.method == "original_direction"]
    print("\nOriginal truth direction applied to the reconstruction (AUROC):")
    print(tr.pivot_table(index="position", columns="sae_type", values="auroc").round(3).to_string())
    print("\nShare of difference lost: truth vs random halves (mean over datasets):")
    print(null.groupby(["sae_type", "position"])[["lost_truth", "lost_null_mean"]]
          .mean().round(3).to_string())
    plot(probe, null)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="cities, position 6 only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["cities"] if args.quick else DATASETS
    data = {n: load_dataset(n) for n in names}
    for n, d in data.items():
        print(f"{n}: train {len(d['tr'])}, test {len(d['te'])}, "
              f"group column for within-AUROC: {d['gcol']}")

    PART_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for t, release in RELEASES.items():
        todo = sae_ids(release)
        if args.quick:
            todo = [p for p in todo if p[0] == 6]
        print(f"\n{t}: {len(todo)} SAEs")
        for pos, sid in todo:
            ppart = PART_DIR / f"probe_{t}_P{pos:02d}.csv"
            npart = PART_DIR / f"null_{t}_P{pos:02d}.csv"
            if ppart.exists() and npart.exists() and not args.quick:
                print(f"  P{pos:02d} already done, skipping")
                continue
            t0 = time.time()
            sae = load_sae(release, sid, device)
            prows, nrows = [], []
            for n, d in data.items():
                x = d["x"][:, pos, :]
                z, xh = run_sae(sae, x, device)
                inputs = {"original": x, "reconstruction": xh, "error": x - xh, "features": z}
                key = dict(sae_type=t, position=pos, dataset=n)
                prows += [{**key, **r} for r in evaluate(inputs, d)]
                nrows.append({**key, **null_control(x, xh, d["y"], rng)})
            pdf, ndf = pd.DataFrame(prows), pd.DataFrame(nrows)
            mm = pdf[pdf.method == "mass_mean"].groupby("input")["auroc"].mean()
            print(f"  P{pos:02d}  AUROC orig {mm['original']:.3f}  recon {mm['reconstruction']:.3f}  "
                  f"error {mm['error']:.3f}  features {mm['features']:.3f}  |  lost: truth "
                  f"{ndf.lost_truth.mean():.2f} vs random {ndf.lost_null_mean.mean():.2f}  "
                  f"({time.time() - t0:.0f}s)")
            if args.quick:
                print(pdf.round(3).to_string(index=False))
                print(ndf.round(3).to_string(index=False))
            else:
                pdf.to_csv(ppart, index=False)
                ndf.to_csv(npart, index=False)
            del sae

    if args.quick:
        print("\nQuick test done.")
    else:
        combine()
        print("\nDone.")


if __name__ == "__main__":
    main()