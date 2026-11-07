#!/usr/bin/env python3
"""Check 3: Alternative MLP probe evaluation on GoT validation data.

Train MLP probes on original GPT-2 layer-6 training activations.
Evaluate frozen probes on original and nine SAE-reconstructed
validation activations. Never use test examples.
"""

import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

STAGE4 = Path("scripts/geometry_of_truth/04_sae_truth_preservation.py")
BASELINE = Path("scripts/openwebtext/03_training_sae.py")
CONVERGENCE = Path("results/openwebtext/convergence/layer_6")
FINETUNED = Path("results/geometry_of_truth/sae_finetuning")
OUTPUT = Path("results/geometry_of_truth/independent_truth_probe")

DOMAINS = {
    "geographic": ("cities", "neg_cities"),
    "numerical": ("larger_than", "smaller_than"),
}
ARCHITECTURES = ("relu", "topk", "jumprelu")
VARIANTS = ("pretrained", "reconstruction_only", "truth_aware")

SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TruthMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def checkpoint_path(arch, variant):
    if variant == "pretrained":
        return (
            CONVERGENCE
            / f"{arch}_l0_128_resume_5000_to_15000"
            / "checkpoint_step_15000.pt"
        )
    return (
        FINETUNED / f"layer_6_{arch}"
        / variant / "finetuned_checkpoint.pt"
    )


def train_probe(X, y, device):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    x = torch.from_numpy(X_scaled).to(device)
    labels = torch.from_numpy(y.astype(np.float32)).to(device)

    probe = TruthMLP().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    rng = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        probe.train()
        indices = torch.randperm(len(x), generator=rng)

        for batch_indices in indices.split(BATCH_SIZE):
            xb = x[batch_indices.to(device)]
            yb = labels[batch_indices.to(device)]

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe, scaler


@torch.inference_mode()
def probe_scores(probe, scaler, X, device):
    X_scaled = scaler.transform(X).astype(np.float32)
    x = torch.from_numpy(X_scaled).to(device)
    scores = []
    for batch in x.split(BATCH_SIZE):
        scores.append(probe(batch).cpu().numpy())
    return np.concatenate(scores)


@torch.inference_mode()
def reconstruct(sae, X, device):
    x = torch.from_numpy(X.astype(np.float32))
    reconstructed = []
    active = 0

    for batch in x.split(BATCH_SIZE):
        rec, z, _ = sae(batch.to(device))
        reconstructed.append(rec.cpu().numpy())
        active += (z != 0).sum().item()

    return np.concatenate(reconstructed), active / len(X)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage4 = import_module(STAGE4, "got_stage4_check3")
    baseline = import_module(BASELINE, "baseline_check3")

    for arch in ARCHITECTURES:
        for variant in VARIANTS:
            path = checkpoint_path(arch, variant)
            if not path.is_file():
                raise FileNotFoundError(path)

    rows = []
    print("Device:", device, flush=True)
    print("Check 3: GoT validation, alternative MLP probe", flush=True)

    for domain, names in DOMAINS.items():
        X, y, split, _ = stage4.load_domain(names, 6)
        train = split == "train"
        val = split == "val"

        if not train.any() or not val.any():
            raise ValueError(f"Missing train/val split: {domain}")

        print(f"\nDomain: {domain}", flush=True)
        print(f"Train={train.sum()} Validation={val.sum()}", flush=True)

        probe, scaler = train_probe(X[train], y[train], device)

        original_scores = probe_scores(probe, scaler, X[val], device)
        original_auc = float(roc_auc_score(y[val], original_scores))

        print(f"Original GPT-2 AUROC: {original_auc:.6f}", flush=True)

        for arch in ARCHITECTURES:
            for variant in VARIANTS:
                path = checkpoint_path(arch, variant)
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

                rec, l0 = reconstruct(sae, X[val], device)
                scores = probe_scores(probe, scaler, rec, device)
                auc = float(roc_auc_score(y[val], scores))

                row = {
                    "domain": domain,
                    "layer": 6,
                    "architecture": arch,
                    "variant": variant,
                    "original_auroc": original_auc,
                    "reconstructed_auroc": auc,
                    "auroc_gap": original_auc - auc,
                    "mean_l0": l0,
                    "validation_examples": int(val.sum()),
                }
                rows.append(row)

                print(
                    f"{arch:8s} {variant:20s} "
                    f"AUROC={auc:.6f} "
                    f"gap={row['auroc_gap']:.6f} "
                    f"L0={l0:.2f}",
                    flush=True,
                )
                del sae

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "validation_mlp_probe_results.csv"
    json_path = OUTPUT / "validation_mlp_probe_results.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2))

    print("\nSaved:", csv_path, flush=True)
    print("Saved:", json_path, flush=True)
    print("Test split was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
