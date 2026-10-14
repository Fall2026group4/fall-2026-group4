"""Geometry of Truth analysis: baseline (stages 1-3), then SAE stages.

Run from the project root:
    python scripts/run_got.py --stage 1

Inputs come from the EDA: data/got/split/<dataset>.csv.
Tables go to results/tables/got/, figures to results/figures/got/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.got import activations, baseline, plots, steering  # noqa: E402
from src.got.io import (ACTIVATIONS_DIR, DATASETS, DIRECTIONS_DIR, FAMILY, SPLIT_DIR,  # noqa: E402
                        save_table)


def stage_1_cache():
    """Stage 1: run every statement through GPT-2 once and save its activations."""
    print("\n=== Stage 1: Cache activations ===")
    print("  loading GPT-2 (TransformerLens)...")
    model = activations.load_model()

    summary = []
    for name in DATASETS:
        df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        print(f"\n  caching '{name}' ({len(df)} statements)")
        last, mean, meta = activations.cache_statements(model, df)
        activations.save_set(ACTIVATIONS_DIR, name, last, mean, meta)
        summary.append({
            "dataset": name, "n_statements": len(meta),
            "train": int((meta["split"] == "train").sum()), "test": int((meta["split"] == "test").sum()),
            "mean_tokens": round(meta["n_tokens"].mean(), 2),
            "activation_shape": str(last.shape),
        })

    summary = pd.DataFrame(summary)
    print("\nCached sets:")
    print(summary.to_string(index=False))
    save_table(summary, "got_s1_activation_sets")
    print(f"  activations saved in {ACTIVATIONS_DIR.relative_to(PROJECT_ROOT)}")


def stage_2_baseline():
    """Stage 2: truth directions per dataset and layer, no SAE."""
    print("\n=== Stage 2: Baseline (no SAE) ===")
    all_rows, all_bands = [], []
    for read in ["last", "mean"]:
        data = baseline.load_all(ACTIVATIONS_DIR, DATASETS, read)
        rows, band = baseline.baseline(data, read)
        all_rows.append(rows)
        all_bands.append(band)
        baseline.save_directions(data, read, DIRECTIONS_DIR)
    results, band = pd.concat(all_rows, ignore_index=True), pd.concat(all_bands, ignore_index=True)
    save_table(results, "got_s2_baseline_by_layer")
    save_table(band, "got_s2_random_band")
    print(f"  directions saved in {DIRECTIONS_DIR.relative_to(PROJECT_ROOT)}")

    view = results[(results["read"] == "last") & (results["method"] == "mass_mean")].pivot(
        index="position", columns="dataset", values="auroc").reindex(baseline.POSITIONS)[DATASETS]
    print("\nTest AUROC, mass-mean, last token (0.5 = chance):")
    print(view.to_string())
    nums = results[(results["read"] == "last") & (results["method"] == "mass_mean") &
                   results["dataset"].isin(["larger_than", "smaller_than"])].pivot(
        index="position", columns="dataset", values="auroc_within_n1").reindex(baseline.POSITIONS)
    print("\nNumbers, AUROC WITHIN the same first number (the fair test; shortcut removed):")
    print(nums.to_string())

    for read in ["last", "mean"]:
        plots.plot_baseline(results, band, DATASETS, read)


def stage_3_controls():
    """Stage 3: cross-dataset transfer, negation, joint training, best layers."""
    print("\n=== Stage 3: Transfer and controls ===")
    data = baseline.load_all(ACTIVATIONS_DIR, DATASETS, "last")

    tr = baseline.transfer(data)
    save_table(tr, "got_s3_transfer")
    view = tr[(tr["method"] == "mass_mean") & tr["position"].isin(baseline.SAE_POSITIONS)].pivot_table(
        index="source", columns="target", values="auroc", aggfunc="mean").loc[DATASETS, DATASETS].round(3)
    print("\nTransfer, mass-mean, mean AUROC over L1-L11 (rows = fitted on, columns = tested on):")
    print(view.to_string())

    joint = baseline.joint_training(data, FAMILY)
    save_table(joint, "got_s3_joint_training")
    jview = joint[joint["position"].isin(baseline.SAE_POSITIONS)].groupby(["trained_on", "target"])["auroc"].mean()
    print("\nJoint training (dataset + its twin), mean AUROC over L1-L11 on other datasets:")
    print(jview.round(3).to_string())

    best = baseline.best_layers(tr)
    print("\nLayers ranked by transfer to a DIFFERENT dataset (mass-mean):")
    print(best.to_string(index=False))
    save_table(best, "got_s3_best_layers")

    plots.plot_transfer_heatmap(tr, baseline.SAE_POSITIONS, DATASETS)
    plots.plot_negation(tr)


# Stage 6a settings: layers chosen from stage 2 (cities peaks L7-L8, numbers strong through L8, RepE best L6)
N_PAIRS = 60
SCAN_LAYERS = list(range(1, 12))
SCAN_ALPHAS = [-0.5, 0.5]
DOSE_LAYERS = [6, 7, 8]
DOSE_ALPHAS = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]


def stage_6a_steering():
    """Stage 6a: steer GPT-2 with the stage-2 mass-mean directions (no SAE)."""
    import numpy as np

    print("\n=== Stage 6a: Baseline steering (no SAE) ===")
    model = activations.load_model()
    pairs, directions, norms = {}, {}, {}
    for name in steering.STEER_DATASETS:
        last, _, meta = activations.load_set(ACTIVATIONS_DIR, name)
        pairs[name] = steering.test_pairs(meta, N_PAIRS)
        norms[name] = steering.layer_norms(last)
        directions[name] = np.load(DIRECTIONS_DIR / f"{name}_last.npz")["mass_mean"]
        print(f"  {name}: {len(pairs[name])} test pairs")

    print("\n  pass 1: scan, layers 1-11")
    scan = steering.steering_grid(model, pairs, directions, norms, SCAN_LAYERS, SCAN_ALPHAS,
                                  sources=("own", "twin", "random"))
    save_table(scan, "got_s6a_steering_scan")

    print("\n  pass 2: dose-response, layers " + ", ".join(f"L{l}" for l in DOSE_LAYERS))
    dose = steering.steering_grid(model, pairs, directions, norms, DOSE_LAYERS, DOSE_ALPHAS,
                                  sources=("own", "twin", "random"))
    save_table(dose, "got_s6a_steering_dose")

    base = scan[scan["source"] == "none"][["dataset", "mean_gap", "pair_accuracy"]]
    print("\nWithout steering (true minus false log-prob, and share of pairs preferring the true one):")
    print(base.to_string(index=False))
    view = scan[(scan["alpha"] > 0) & (scan["source"] != "none")].pivot_table(
        index="layer", columns=["dataset", "source"], values="effect")
    view = view.loc[sorted(view.index, key=lambda s: int(s[1:]))]
    print(f"\nScan: effect of steering +{max(SCAN_ALPHAS)} x norm (> 0 = pushed towards the true sentence):")
    print(view.xs("own", axis=1, level=1).round(3).to_string())
    print("\nSame, with the TWIN's direction (a flip predicts the opposite sign):")
    print(view.xs("twin", axis=1, level=1).round(3).to_string())

    plots.plot_steering_scan(scan)
    plots.plot_steering_dose(dose)


STAGES = {
    1: stage_1_cache,
    2: stage_2_baseline,
    3: stage_3_controls,
    6: stage_6a_steering,
}


def main():
    parser = argparse.ArgumentParser(description="Geometry of Truth analysis")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="Run only this stage")
    args = parser.parse_args()
    for number in ([args.stage] if args.stage else sorted(STAGES)):
        STAGES[number]()
    print("\nDone.")


if __name__ == "__main__":
    main()