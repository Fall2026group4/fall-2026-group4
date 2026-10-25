"""Figures for RepE Stage 5, in the same layout as the Geometry of Truth Stage 5 figure.

Panels 1-2: honesty probe AUROC by layer (cross-wording transfer, mean of the 4 directions)
            for original / reconstruction / error / features, ReLU and TopK.
Panel 3   : share of the difference lost in SAE error: honesty vs the size-matched
            wording control (from repe_s4_size_control.csv).
Saves one figure with mass-mean probes (as in GoT) and one with logistic probes.

Run from the repo root:  python scripts/plot_repe_s5.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables" / "repe"
FIG = ROOT / "results" / "figures" / "repe"
INPUTS = ["original", "reconstruction", "error", "features"]

probes = pd.read_csv(TAB / "repe_s5_probes.csv")
control = pd.read_csv(TAB / "repe_s4_size_control.csv")
control["wording_lost"] = control[["honest_wording_lost", "dishonest_wording_lost"]].mean(axis=1)
FIG.mkdir(parents=True, exist_ok=True)

for method, label in [("mass_mean", "mass-mean"), ("logistic", "logistic")]:
    tr = probes[(probes.kind == "transfer") & (probes.method == method)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, t in zip(axes[:2], ["relu", "topk"]):
        g = tr[tr.sae_type == t]
        for inp in INPUTS:
            m = g[g.input == inp].groupby("position")["auroc"].mean()
            ax.plot(m.index, m.values, marker="o", label=inp)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(f"{t}: honesty probe AUROC ({label}, cross-wording transfer)")
        ax.set_xlabel("position (12 = final output)")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
    axes[0].legend(title="probe input")

    ax = axes[2]
    for t, g in control.groupby("sae_type"):
        g = g.sort_values("position")
        line, = ax.plot(g.position, g.honesty_lost, marker="o", label=f"{t}: honesty")
        ax.plot(g.position, g.wording_lost, ls="--", color=line.get_color(),
                label=f"{t}: wording swap (size-matched)")
    ax.set_title("Share of the difference lost in SAE error")
    ax.set_xlabel("position (12 = final output)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    name = "repe_s5_probe_fidelity.svg" if method == "mass_mean" else "repe_s5_probe_fidelity_logistic.svg"
    fig.savefig(FIG / name, dpi=150)
    print(f"Saved results/figures/repe/{name}")