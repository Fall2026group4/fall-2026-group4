#!/usr/bin/env python3
"""Final held-out GoT evaluation, GPT-2 Small Layer 6.

Uses fixed SAE checkpoints and five predetermined MLP seeds.
Trains probes on training rows only. Evaluates test rows once.
"""

import csv
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

SOURCE = Path("scripts/geometry_of_truth/08_multiseed_truth_probe.py")
OUTPUT = Path("results/geometry_of_truth/final_test_evaluation")
SEEDS = (42, 123, 456, 789, 2026)


def load_source():
    spec = importlib.util.spec_from_file_location("multiseed", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_domain(names):
    """Load all split assignments; never use test for training."""
    parts = []
    group_to_split = {}

    for name in names:
        path = Path("data/got/activations/gpt2_small") / f"{name}_final_token.pt"
        split_path = Path("data/got/splits") / f"{name}_splits.csv"

        data = torch.load(path, map_location="cpu", weights_only=False)
        splits = pd.read_csv(split_path)

        row_ids = np.asarray(data["row_ids"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)

        if len(row_ids) != len(splits):
            raise ValueError(f"{name}: row count mismatch")
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError(f"{name}: duplicate row IDs")
        if splits["row_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate split row IDs")

        aligned = splits.set_index("row_id").reindex(row_ids)

        if aligned.isna().any().any():
            raise ValueError(f"{name}: missing split assignments")
        if not np.array_equal(
            labels, aligned["label"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"{name}: label mismatch")

        assignment = aligned["split"].to_numpy()
        groups = aligned["group_id"].astype(str).to_numpy()

        if not set(assignment).issubset({"train", "val", "test"}):
            raise ValueError(f"{name}: invalid split assignments")

        for group, split in zip(groups, assignment):
            if group in group_to_split and group_to_split[group] != split:
                raise ValueError(f"Group leakage: {group}")
            group_to_split[group] = split

        activations = data["activations"]
        X = activations[6].float().numpy()

        if X.shape != (len(labels), 768):
            raise ValueError(f"{name}: invalid activation shape")
        if not np.isfinite(X).all():
            raise ValueError(f"{name}: non-finite activations")

        parts.append((X, labels, assignment))

    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def main():
    torch.set_num_threads(4)
    module = load_source()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = module.import_module(module.BASELINE, "final_sae_baseline")

    checkpoints = {}
    for arch in module.ARCHITECTURES:
        for variant in module.VARIANTS:
            path = module.checkpoint_path(arch, variant)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints[f"{arch}__{variant}"] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    rows = []

    print("FINAL HELD-OUT EVALUATION", flush=True)
    print("Device:", device, flush=True)
    print("Seeds:", SEEDS, flush=True)

    for domain, names in module.DOMAINS.items():
        X, y, split = load_domain(names)
        train = split == "train"
        test = split == "test"

        if not train.any() or not test.any():
            raise ValueError(f"{domain}: missing train/test rows")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError(f"{domain}: both labels required")

        print(
            f"\n{domain}: train={train.sum()}, "
            f"test={test.sum()}",
            flush=True,
        )

        X_test = X[test]
        y_test = y[test]

        # SAE reconstructions do not depend on MLP random seed.
        reconstructed = {}
        sparsity = {}

        for arch in module.ARCHITECTURES:
            for variant in module.VARIANTS:
                path = module.checkpoint_path(arch, variant)
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                state = checkpoint["model_state_dict"]
                d_sae, d_in = state["encoder.weight"].shape

                if (d_in, d_sae) != (768, 3072):
                    raise ValueError(f"Unexpected SAE dimensions: {path}")

                sae = baseline.SAE(
                    d_in, d_sae, arch, 128, temperature=0.1
                ).to(device)
                sae.load_state_dict(state, strict=True)
                sae.eval()

                rec, l0 = module.reconstruct(sae, X_test, device)
                reconstructed[(arch, variant)] = rec
                sparsity[(arch, variant)] = l0
                del sae

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            probe, scaler = module.train_probe(X[train], y[train], device)

            original_scores = module.probe_scores(
                probe, scaler, X_test, device
            )
            original_auc = float(roc_auc_score(y_test, original_scores))

            print(
                f"  Seed {seed}: original AUROC={original_auc:.6f}",
                flush=True,
            )

            for arch in module.ARCHITECTURES:
                for variant in module.VARIANTS:
                    key = (arch, variant)
                    scores = module.probe_scores(
                        probe, scaler, reconstructed[key], device
                    )
                    auc = float(roc_auc_score(y_test, scores))

                    rows.append({
                        "seed": seed,
                        "domain": domain,
                        "layer": 6,
                        "architecture": arch,
                        "variant": variant,
                        "original_auroc": original_auc,
                        "reconstructed_auroc": auc,
                        "auroc_gap": original_auc - auc,
                        "mean_l0": sparsity[key],
                        "test_examples": int(test.sum()),
                    })

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "test_results_all_seeds.csv", index=False)

    summary = (
        results.groupby(
            ["domain", "architecture", "variant"], as_index=False
        )
        .agg(
            mean_original_auroc=("original_auroc", "mean"),
            mean_auroc=("reconstructed_auroc", "mean"),
            std_auroc=("reconstructed_auroc", "std"),
            mean_gap=("auroc_gap", "mean"),
            std_gap=("auroc_gap", "std"),
            mean_l0=("mean_l0", "mean"),
            test_examples=("test_examples", "first"),
        )
    )

    summary.to_csv(OUTPUT / "test_summary.csv", index=False)

    manifest = {
        "seeds": list(SEEDS),
        "layer": 6,
        "probe_architecture": "768-64-ReLU-1",
        "epochs": module.EPOCHS,
        "batch_size": module.BATCH_SIZE,
        "learning_rate": module.LR,
        "probe_training_split": "train",
        "evaluation_split": "test",
        "checkpoint_hashes": checkpoints,
        "test_used_for_model_selection": False,
    }

    (OUTPUT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nFINAL TEST SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nResults saved to:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
