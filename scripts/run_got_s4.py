"""Stage 4 (Geometry of Truth): pass saved activations through pretrained GPT-2 Small SAEs.

For each SAE type x position x dataset it measures how much the SAE loses:
  fvu             share of variance (around the dataset mean) the reconstruction misses
  rel_err         ||x - x_hat||^2 / ||x||^2 (uncentered; less inflated when prompts are similar)
  cos_mean        cosine(x, x_hat), averaged over prompts
  l0_mean         active latents per prompt
  norm_ratio      median ||x_hat|| / ||x||
  diff_cos        cosine between the true-minus-false mean difference before and after the SAE
  diff_norm_ratio how much of that difference's length survives
  diff_err_frac   share of the difference that ends up in the SAE error
  dead_frac_all_sets  latents that never fire on any of the six datasets

Run on EC2 from the repo root:
    python scripts/run_got_s4.py --quick     # smoke test: cities, position 6, both SAE types
    python scripts/run_got_s4.py             # full run (resumable: finished SAEs are skipped)
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

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "data" / "got" / "activations"
LAT_DIR = ROOT / "data" / "got" / "sae_latents"
PART_DIR = LAT_DIR / "_parts"
TAB_DIR = ROOT / "results" / "tables" / "got"
FIG_DIR = ROOT / "results" / "figures" / "got"

DATASETS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans",
            "larger_than", "smaller_than"]
RELEASES = {"relu": "gpt2-small-res-jb", "topk": "gpt2-small-resid-post-v5-32k"}
LABEL_COLS = ["label", "labels", "is_true", "truth", "truth_value"]
SUMMARY_COLS = ["fvu", "rel_err", "cos_mean", "l0_mean", "diff_cos",
                "diff_err_frac", "dead_frac_all_sets"]


# ---------- mapping SAEs to your 13 positions ----------
def position_for(sae_id):
    """resid_pre K -> position K; resid_post K -> position K+1 (12 = final output)."""
    m = re.fullmatch(r"blocks\.(\d+)\.hook_resid_(pre|post)", sae_id)
    if not m:
        return None
    layer, kind = int(m.group(1)), m.group(2)
    return layer if kind == "pre" else layer + 1


def sae_ids(release):
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    ids = list(get_pretrained_saes_directory()[release].saes_map)
    pairs = [(position_for(i), i) for i in ids]
    return sorted((p, i) for p, i in pairs if p is not None)


# ---------- loading ----------
def load_labels(name):
    meta = pd.read_csv(ACT_DIR / f"{name}.csv")
    col = next((c for c in LABEL_COLS if c in meta.columns), None)
    if col is None:
        sys.exit(f"No label column found in {name}.csv. Columns are: {list(meta.columns)}")
    y = meta[col]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    return y.astype(int).to_numpy()


def load_all(names):
    data = {}
    for n in names:
        x = np.load(ACT_DIR / f"{n}.npz")["last"]
        y = load_labels(n)
        if len(x) != len(y):
            sys.exit(f"{n}: {len(x)} activations but {len(y)} labels")
        data[n] = (x, y)
    return data


def load_sae(release, sae_id, device):
    from sae_lens import SAE
    out = SAE.from_pretrained(release, sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out  # older versions return a tuple
    sae.eval()
    return sae


# ---------- encode / decode ----------
@torch.no_grad()
def run_sae(sae, x, device, batch=256):
    dtype = sae.W_enc.dtype
    zs, xhs = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device=device, dtype=dtype)
        z = sae.encode(xb)
        xh = sae.decode(z)  # decode right after encode: some SAEs normalise per batch
        zs.append(sp.csr_matrix(z.float().cpu().numpy()))
        xhs.append(xh.float().cpu().numpy())
    return sp.vstack(zs).tocsr(), np.concatenate(xhs)


def fidelity(x, xh, z, y):
    err = x - xh
    nx = np.linalg.norm(x, axis=1)
    nxh = np.linalg.norm(xh, axis=1)
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    dh = xh[y == 1].mean(0) - xh[y == 0].mean(0)
    nd = np.linalg.norm(d) + 1e-8
    return dict(
        n=len(x),
        fvu=float((err ** 2).sum() / ((x - x.mean(0)) ** 2).sum()),
        rel_err=float((err ** 2).sum() / (x ** 2).sum()),
        cos_mean=float(((x * xh).sum(1) / (nx * nxh + 1e-8)).mean()),
        l0_mean=float(np.diff(z.indptr).mean()),
        norm_ratio=float(np.median(nxh / (nx + 1e-8))),
        diff_cos=float(d @ dh / (nd * (np.linalg.norm(dh) + 1e-8))),
        diff_norm_ratio=float(np.linalg.norm(dh) / nd),
        diff_err_frac=float(np.linalg.norm(d - dh) / nd),
    )


# ---------- outputs ----------
def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panels = [("rel_err", "Reconstruction error (lower = better)"),
              ("diff_cos", "Truth difference kept: cosine (1 = perfect)"),
              ("diff_err_frac", "Truth difference lost to SAE error")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (col, title) in zip(axes, panels):
        for t, g in df.groupby("sae_type"):
            m = g.groupby("position")[col].mean()
            ax.plot(m.index, m.values, marker="o", label=t)
        ax.set_title(title)
        ax.set_xlabel("position (resid_pre layer; 12 = final output)")
        ax.grid(alpha=0.3)
    axes[0].legend(title="SAE type")
    fig.suptitle("Stage 4: SAE fidelity on Geometry of Truth (mean over 6 datasets)")
    fig.tight_layout()
    path = FIG_DIR / "got_s4_sae_fidelity.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path.relative_to(ROOT)}")


def combine(expected):
    parts = sorted(PART_DIR.glob("*.csv"))
    if len(parts) < expected:
        print(f"Warning: only {len(parts)} of {expected} SAEs finished. Rerun to finish the rest.")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / "got_s4_sae_fidelity.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved {path.relative_to(ROOT)}  ({len(df)} rows)\n")
    summ = df.groupby(["sae_type", "position"])[SUMMARY_COLS].mean().round(3)
    print("Mean over datasets, by SAE type and position:")
    print(summ.to_string())
    plot(df)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: cities, position 6 only")
    ap.add_argument("--types", nargs="+", default=list(RELEASES), choices=list(RELEASES))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["cities"] if args.quick else DATASETS
    data = load_all(names)
    for n, (x, y) in data.items():
        print(f"{n}: {len(y)} statements, {y.mean():.0%} true, activations {x.shape}")

    PART_DIR.mkdir(parents=True, exist_ok=True)
    expected = 0
    for t in args.types:
        release = RELEASES[t]
        todo = sae_ids(release)
        if args.quick:
            todo = [p for p in todo if p[0] == 6]
        expected += len(todo)
        print(f"\n{t}: {release}, {len(todo)} SAEs")

        for pos, sid in todo:
            part = PART_DIR / f"{t}_P{pos:02d}.csv"
            if part.exists() and not args.quick:
                print(f"  P{pos:02d} already done, skipping")
                continue
            t0 = time.time()
            sae = load_sae(release, sid, device)
            arch = type(sae).__name__
            norm = getattr(sae.cfg, "normalize_activations", "?")
            d_sae = sae.cfg.d_sae
            active = np.zeros(d_sae, dtype=bool)
            rows = []
            for n, (x, y) in data.items():
                xp = x[:, pos, :]
                z, xh = run_sae(sae, xp, device)
                active[np.unique(z.indices)] = True
                if not args.quick:
                    out = LAT_DIR / t / f"P{pos:02d}"
                    out.mkdir(parents=True, exist_ok=True)
                    sp.save_npz(out / f"{n}.npz", z)
                rows.append(dict(sae_type=t, arch=arch, sae_id=sid, position=pos,
                                 dataset=n, d_sae=d_sae, **fidelity(xp, xh, z, y)))
            df = pd.DataFrame(rows)
            df["dead_frac_all_sets"] = 1 - active.mean()
            print(f"  P{pos:02d} {sid} [{arch}, norm={norm}, d_sae={d_sae}]  "
                  f"rel_err {df.rel_err.mean():.3f}  FVU {df.fvu.mean():.3f}  "
                  f"L0 {df.l0_mean.mean():.0f}  diff_cos {df.diff_cos.mean():.2f}  "
                  f"({time.time() - t0:.0f}s)")
            if args.quick:
                print(df.drop(columns=["sae_id", "arch"]).round(3).to_string(index=False))
            else:
                df.to_csv(part, index=False)
            del sae

    if args.quick:
        print("\nQuick test done.")
    else:
        combine(expected)
        print("\nDone.")


if __name__ == "__main__":
    main()