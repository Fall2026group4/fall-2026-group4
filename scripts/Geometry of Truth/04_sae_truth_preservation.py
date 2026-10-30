\
#!/usr/bin/env python3
"""Stage 4: SAE truth preservation on Geometry of Truth validation data."""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/got")
ACT_DIR = ROOT / "activations" / "gpt2_small"
SPLIT_DIR = ROOT / "splits"

CONVERGENCE = Path("results/openwebtext/convergence")
PROBE_RESULTS = Path("results/got/original_truth_probes")
OUTPUT = Path("results/got/sae_truth_preservation")

BASELINE = Path("scripts/openwebtext/03_training_sae.py")

DOMAINS = {
    "geographic": ("cities", "neg_cities"),
    "numerical": ("larger_than", "smaller_than"),
}

LAYERS = (2, 6, 10)
ARCHITECTURES = ("relu", "topk", "jumprelu")

STEP = 15000
TARGET_L0 = 128
BATCH_SIZE = 256
SEED = 42


def load_sae_module():
    spec = importlib.util.spec_from_file_location("baseline_sae", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(layer, arch):
    if layer == 6:
        folder = f"{arch}_l0_128_resume_5000_to_15000"
    else:
        folder = f"{arch}_l0_128_warmstart_1000_to_15000"

    return (
        CONVERGENCE
        / f"layer_{layer}"
        / folder
        / f"checkpoint_step_{STEP}.pt"
    )


def load_sae(module, layer, arch, device):
    path = checkpoint_path(layer, arch)

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    if int(checkpoint["step"]) != STEP:
        raise ValueError(f"Unexpected checkpoint step: {path}")

    state = checkpoint["model_state_dict"]
    d_sae, d_in = state["encoder.weight"].shape

    if (d_in, d_sae) != (768, 3072):
        raise ValueError(f"Unexpected SAE dimensions: {path}")

    model = module.SAE(
        d_in, d_sae, arch, TARGET_L0, temperature=0.1
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def load_domain(names, layer):
    pieces = []

    for name in names:
        data = torch.load(
            ACT_DIR / f"{name}_final_token.pt",
            map_location="cpu",
            weights_only=False,
        )

        splits = pd.read_csv(SPLIT_DIR / f"{name}_splits.csv")

        row_ids = np.asarray(data["row_ids"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)

        if len(row_ids) != len(splits):
            raise ValueError(f"{name}: row count mismatch")

        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError(f"{name}: duplicate row IDs")

        aligned = splits.set_index("row_id").reindex(row_ids)

        if aligned.isna().any().any():
            raise ValueError(f"{name}: missing split assignments")

        if not np.array_equal(
            labels, aligned["label"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"{name}: label mismatch")

        split = aligned["split"].to_numpy()

        if not set(np.unique(split)).issubset(
            {"train", "val", "test"}
        ):
            raise ValueError(f"{name}: invalid split")

        activations = data["activations"]
        x = activations[layer].float().numpy()

        if x.shape != (len(labels), 768):
            raise ValueError(f"{name}: invalid activation shape")

        # Select only train/validation rows.
        keep = np.isin(split, ["train", "val"])

        pieces.append({
            "name": name,
            "X": x[keep],
            "y": labels[keep],
            "split": split[keep],
        })

    X = np.concatenate([p["X"] for p in pieces])
    y = np.concatenate([p["y"] for p in pieces])
    split = np.concatenate([p["split"] for p in pieces])

    source = np.concatenate([
        np.repeat(p["name"], len(p["y"])) for p in pieces
    ])

    if not np.isfinite(X).all():
        raise ValueError("Non-finite activations")

    return X, y, split, source


def safe_auc(labels, scores):
    if np.unique(labels).size < 2:
        return float("nan")

    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def reconstruct(model, X, device):
    reconstructed = []
    total_active = 0
    total_rows = 0

    for start in range(0, len(X), BATCH_SIZE):
        batch = torch.from_numpy(
            X[start:start + BATCH_SIZE]
        ).to(device)

        rec, z, _ = model(batch)

        reconstructed.append(rec.cpu().numpy())

        total_active += (z != 0).sum().item()
        total_rows += z.shape[0]

    return (
        np.concatenate(reconstructed),
        total_active / total_rows,
    )


def main():
    torch.set_num_threads(4)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    module = load_sae_module()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        PROBE_RESULTS / "validation_best_C_by_layer.csv"
    )

    rows = []

    print("Device:", device)
    print("Validation-only SAE truth preservation")
    print("Test set is not evaluated.\n")

    for domain, names in DOMAINS.items():

        print(f"\nDOMAIN: {domain}", flush=True)

        for layer in LAYERS:

            X, y, split, source = load_domain(names, layer)

            train = split == "train"
            val = split == "val"

            choice = selected[
                (selected["domain"] == domain)
                & (selected["layer"] == layer)
            ]

            if len(choice) != 1:
                raise ValueError(
                    f"Missing selected C: {domain}, layer {layer}"
                )

            C = float(choice.iloc[0]["C"])

            # Fit scaler and classifier ONLY on original training data.
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    random_state=SEED,
                ),
            )

            probe.fit(X[train], y[train])

            X_val = X[val]
            y_val = y[val]
            source_val = source[val]

            # Use raw logits to prevent probability saturation.
            original_scores = probe.decision_function(X_val)
            original_auc = safe_auc(y_val, original_scores)

            expected_auc = float(
                choice.iloc[0]["validation_auroc"]
            )

            if abs(original_auc - expected_auc) > 1e-6:
                raise ValueError(
                    f"Probe reproduction mismatch: "
                    f"{domain}, layer {layer}: "
                    f"{original_auc} vs {expected_auc}"
                )

            print(
                f"\nLayer {layer} | C={C:g} "
                f"| Original AUROC={original_auc:.6f}",
                flush=True,
            )

            for arch in ARCHITECTURES:

                model = load_sae(module, layer, arch, device)

                X_rec, l0 = reconstruct(
                    model, X_val, device
                )

                # Evaluate using raw decision scores.
                reconstructed_scores = probe.decision_function(X_rec)

                reconstructed_auc = safe_auc(
                    y_val, reconstructed_scores
                )

                # Validation-only probe-score distribution diagnostics.
                score_diagnostics = {
                    "original_score_mean": float(np.mean(original_scores)),
                    "original_score_std": float(np.std(original_scores)),
                    "sae_score_mean": float(np.mean(reconstructed_scores)),
                    "sae_score_std": float(np.std(reconstructed_scores)),
                    "sae_score_min": float(np.min(reconstructed_scores)),
                    "sae_score_max": float(np.max(reconstructed_scores)),
                    "score_correlation": (
                        float(np.corrcoef(
                            original_scores, reconstructed_scores
                        )[0, 1])
                        if np.std(original_scores) > 1e-12
                        and np.std(reconstructed_scores) > 1e-12
                        else float("nan")
                    ),
                }

                mse = float(
                    np.mean((X_val - X_rec) ** 2)
                )

                row = {
                    "domain": domain,
                    "layer": layer,
                    "architecture": arch,
                    "C": C,
                    "validation_n": len(y_val),
                    "original_auroc": original_auc,
                    "sae_auroc": reconstructed_auc,
                    "auroc_gap": original_auc - reconstructed_auc,
                    "reconstruction_mse": mse,
                    "mean_l0": l0,
                    "checkpoint": str(
                        checkpoint_path(layer, arch)
                    ),
                    **score_diagnostics,
                }

                for name in names:
                    mask = source_val == name

                    row[f"{name}_original_auroc"] = safe_auc(
                        y_val[mask],
                        original_scores[mask],
                    )

                    row[f"{name}_sae_auroc"] = safe_auc(
                        y_val[mask],
                        reconstructed_scores[mask],
                    )

                    row[f"{name}_auroc_gap"] = (
                        row[f"{name}_original_auroc"]
                        - row[f"{name}_sae_auroc"]
                    )

                rows.append(row)

                print(
                    f"  {arch:8s} | "
                    f"SAE AUROC={reconstructed_auc:.6f} | "
                    f"Gap={row['auroc_gap']:+.6f} | "
                    f"MSE={mse:.6f} | "
                    f"L0={l0:.2f}",
                    flush=True,
                )

                del model

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT / "validation_truth_preservation.csv",
        index=False,
    )

    metadata = {
        "model": "gpt2-small",
        "layers": list(LAYERS),
        "architectures": list(ARCHITECTURES),
        "sae_training_step": STEP,
        "target_l0": TARGET_L0,
        "probe_training": "original_train_only",
        "probe_evaluation": "original_and_sae_validation",
        "auc_score_type": "decision_function_raw_logits",
        "test_evaluated": False,
        "seed": SEED,
    }

    with open(
        OUTPUT / "run_info.json", "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:", OUTPUT / "validation_truth_preservation.csv")
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
