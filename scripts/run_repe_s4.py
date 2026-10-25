"""Stage 4 (RepE): pass the cached RepE activations through pretrained GPT-2 Small SAEs.

Per SAE type x position x prompt set:
  rel_err, fvu, cos_mean, l0_mean, norm_ratio   general reconstruction quality
  diff_cos, diff_err_frac                       honest-minus-untruthful CLASS-mean difference kept/lost
  pair_err_frac                                 median share of each pair's honesty difference
                                                (same statement, honest vs untruthful persona) lost in SAE error
  null_err_frac                                 same for STATEMENT differences (different statements,
                                                same persona): the control
  dead_frac_all_sets                            latents that never fire on any prompt set

Run from the repo root:
    python scripts/run_repe_s4.py --quick     # original set, position 6
    python scripts/run_repe_s4.py             # full run
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "data" / "repe" / "activations"
TAB_DIR = ROOT / "results" / "tables" / "repe"
FIG_DIR = ROOT / "results" / "figures" / "repe"

SETS = ["original", "matched_honest_dishonest", "matched_honest_lying",
        "matched_truthful_deceptive", "matched_truthful_lying", "false_statements"]
RELEASES = {"relu": "gpt2-small-res-jb", "topk": "gpt2-small-resid-post-v5-32k"}


# ---------- SAEs ----------
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


def load_sae(release, sae_id):
    from sae_lens import SAE
    out = SAE.from_pretrained(release, sae_id, device="cpu")
    sae = out[0] if isinstance(out, tuple) else out
    sae.eval()
    return sae


@torch.no_grad()
def run_sae(sae, x, batch=256):
    dtype = sae.W_enc.dtype
    l0, active, xhs = [], np.zeros(sae.cfg.d_sae, dtype=bool), []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dtype=dtype)
        z = sae.encode(xb)
        xh = sae.decode(z)  # right after encode: some SAEs normalise per batch
        nz = (z > 0)
        l0.append(nz.sum(1).cpu().numpy())
        active |= nz.any(0).cpu().numpy()
        xhs.append(xh.float().cpu().numpy())
    return np.concatenate(xhs), np.concatenate(l0), active


# ---------- data ----------
def load_set(name):
    x = np.load(ACT_DIR / f"{name}.npz")["last"]
    meta = pd.read_csv(ACT_DIR / f"{name}.csv")
    honest = (meta["label"].astype(int) == 1).to_numpy()
    h_idx, u_idx = [], []
    for _, g in meta.groupby("pair_id"):
        if len(g) == 2 and honest[g.index].sum() == 1:
            h_idx.append(g.index[honest[g.index]][0])
            u_idx.append(g.index[~honest[g.index]][0])
    return dict(x=x, meta=meta, honest=honest, h=np.array(h_idx), u=np.array(u_idx))


def pair_lost(d, dh):
    """Median share of each difference vector that ends up in the SAE error."""
    return float(np.median(np.linalg.norm(d - dh, axis=1) / (np.linalg.norm(d, axis=1) + 1e-8)))


def fidelity(x, xh, l0, d):
    err = x - xh
    nx, nxh = np.linalg.norm(x, axis=1), np.linalg.norm(xh, axis=1)
    hm, um = x[d["honest"]].mean(0) - x[~d["honest"]].mean(0), None
    dd = x[d["honest"]].mean(0) - x[~d["honest"]].mean(0)
    ddh = xh[d["honest"]].mean(0) - xh[~d["honest"]].mean(0)
    nd = np.linalg.norm(dd) + 1e-8
    h, u = d["h"], d["u"]
    other = np.roll(h, 1)  # honest prompt of a DIFFERENT statement: same persona, different fact
    return dict(
        n=len(x), n_pairs=len(h),
        rel_err=float((err ** 2).sum() / (x ** 2).sum()),
        fvu=float((err ** 2).sum() / ((x - x.mean(0)) ** 2).sum()),
        cos_mean=float(((x * xh).sum(1) / (nx * nxh + 1e-8)).mean()),
        l0_mean=float(l0.mean()),
        norm_ratio=float(np.median(nxh / (nx + 1e-8))),
        diff_cos=float(dd @ ddh / (nd * (np.linalg.norm(ddh) + 1e-8))),
        diff_err_frac=float(np.linalg.norm(dd - ddh) / nd),
        pair_err_frac=pair_lost(x[h] - x[u], xh[h] - xh[u]),
        null_err_frac=pair_lost(x[h] - x[other], xh[h] - xh[other]),
        pair_norm=float(np.median(np.linalg.norm(x[h] - x[u], axis=1))),
        null_norm=float(np.median(np.linalg.norm(x[h] - x[other], axis=1))),
    )


# ---------- outputs ----------
def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for t, g in df.groupby("sae_type"):
        m = g.groupby("position")[["rel_err", "pair_err_frac", "null_err_frac"]].mean()
        axes[0].plot(m.index, m.rel_err, marker="o", label=t)
        line, = axes[1].plot(m.index, m.pair_err_frac, marker="o", label=f"{t}: honesty difference")
        axes[1].plot(m.index, m.null_err_frac, ls="--", color=line.get_color(),
                     label=f"{t}: statement difference (control)")
    axes[0].set_title("Reconstruction error (lower = better)")
    axes[1].set_title("Share of the difference lost in SAE error")
    for ax in axes:
        ax.set_xlabel("position (12 = final output)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("RepE Stage 4: SAE fidelity (mean over prompt sets)")
    fig.tight_layout()
    path = FIG_DIR / "repe_s4_sae_fidelity.svg"
    fig.savefig(path, dpi=150)
    print(f"Saved {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="original set, position 6 only")
    args = ap.parse_args()

    names = ["original"] if args.quick else SETS
    data = {n: load_set(n) for n in names}
    m0 = data[names[0]]["meta"]
    print("Check which label means 'honest' (label 1 should be the honest/truthful persona):")
    print(pd.crosstab(m0["label"], m0["persona"]).to_string(), "\n")
    for n, d in data.items():
        print(f"{n}: {len(d['x'])} prompts, {len(d['h'])} honest/untruthful pairs")

    rows = []
    for t, release in RELEASES.items():
        todo = sae_ids(release)
        if args.quick:
            todo = [p for p in todo if p[0] == 6]
        print(f"\n{t}: {len(todo)} SAEs")
        for pos, sid in todo:
            t0 = time.time()
            sae = load_sae(release, sid)
            active = np.zeros(sae.cfg.d_sae, dtype=bool)
            part = []
            for n, d in data.items():
                x = d["x"][:, pos, :]
                xh, l0, act = run_sae(sae, x)
                active |= act
                part.append(dict(sae_type=t, sae_id=sid, position=pos, prompt_set=n,
                                 d_sae=sae.cfg.d_sae, **fidelity(x, xh, l0, d)))
            dead = 1 - active.mean()
            for r in part:
                r["dead_frac_all_sets"] = dead
            rows += part
            p = pd.DataFrame(part)
            print(f"  P{pos:02d} {sid}: rel_err {p.rel_err.mean():.3f}  L0 {p.l0_mean.mean():.0f}  "
                  f"honesty lost {p.pair_err_frac.mean():.2f} vs statement lost {p.null_err_frac.mean():.2f}  "
                  f"({time.time() - t0:.0f}s)")
            del sae

    df = pd.DataFrame(rows)
    if args.quick:
        print("\n" + df.drop(columns=["sae_id"]).round(3).to_string(index=False))
        print("\nQuick test done.")
        return
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / "repe_s4_sae_fidelity.csv", index=False)
    print("\nSaved results/tables/repe/repe_s4_sae_fidelity.csv\n")
    summ = df.groupby(["sae_type", "position"])[["rel_err", "cos_mean", "l0_mean", "diff_cos",
                                                 "pair_err_frac", "null_err_frac",
                                                 "dead_frac_all_sets"]].mean().round(3)
    print("Mean over prompt sets:")
    print(summ.to_string())
    plot(df)
    print("\nDone.")


if __name__ == "__main__":
    main()