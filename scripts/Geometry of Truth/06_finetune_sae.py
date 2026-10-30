
#!/usr/bin/env python3
"""Stage 6: Fine-tune Layer 6 SAEs, reconstruction-only vs truth-aware.

Supported architectures: relu, topk, jumprelu.

Run from project root. Uses only GoT train/validation splits;
test is not evaluated.

Frozen truth probes are refitted on original GoT training activations
using Stage 3's validation-selected C values, as in Stage 4.
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="Fine-tune a pretrained SAE on Geometry of Truth."
)

parser.add_argument(
    "--architecture",
    choices=["relu", "topk", "jumprelu"],
    required=True,
    help="SAE architecture to fine-tune",
)

args = parser.parse_args()

STAGE4 = Path(
    "scripts/Geometry of Truth/04_sae_truth_preservation.py"
)

PROBE_CSV = Path(
    "results/got/original_truth_probes/"
    "validation_best_C_by_layer.csv"
)

LAYER = 6
ARCH = args.architecture

OUTPUT = Path(
    f"results/got/sae_finetuning/layer_{LAYER}_{ARCH}"
)

STEPS = 500
BATCH_SIZE = 128
LR = 1e-5
SEED = 42
TRUTH_WEIGHT = 0.1
LOG_EVERY = 50


# ============================================================
# LOAD STAGE 4 UTILITIES
# ============================================================

def import_stage4():
    spec = importlib.util.spec_from_file_location(
        "got_stage4", STAGE4
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# LOAD DATA AND FROZEN TRUTH PROBES
# ============================================================

def load_data_and_teachers(s4, device):
    choices = pd.read_csv(PROBE_CSV)

    parts = []
    teachers = {}

    for domain, names in s4.DOMAINS.items():

        X, y, split, _ = s4.load_domain(names, LAYER)

        train = split == "train"
        val = split == "val"

        if (
            not train.any()
            or not val.any()
            or np.any(split == "test")
        ):
            raise ValueError(
                f"{domain}: expected train/validation only"
            )

        choice = choices[
            (choices["domain"] == domain)
            & (choices["layer"] == LAYER)
        ]

        if len(choice) != 1:
            raise ValueError(
                f"Missing selected probe C for "
                f"{domain} layer {LAYER}"
            )

        C = float(choice.iloc[0]["C"])

        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=3000,
                random_state=SEED,
            ),
        )

        probe.fit(X[train], y[train])

        observed = s4.safe_auc(
            y[val],
            probe.predict_proba(X[val])[:, 1],
        )

        expected = float(
            choice.iloc[0]["validation_auroc"]
        )

        if abs(observed - expected) > 1e-6:
            raise ValueError(
                f"{domain}: probe baseline mismatch: "
                f"{observed} != {expected}"
            )

        # Convert the frozen sklearn probe into an affine
        # function whose logits can be evaluated in PyTorch.
        scaler = probe.steps[0][1]
        classifier = probe.steps[1][1]

        weights = classifier.coef_[0] / scaler.scale_

        bias = float(
            classifier.intercept_[0]
            - np.dot(weights, scaler.mean_)
        )

        w = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        b = torch.tensor(
            bias,
            dtype=torch.float32,
            device=device,
        )

        original_train_logits = (
            X[train] @ weights + bias
        )

        logit_scale = max(
            float(np.std(original_train_logits)),
            1e-6,
        )

        teachers[domain] = (w, b, logit_scale)

        parts.append({
            "domain": domain,
            "train_X": torch.tensor(
                X[train],
                dtype=torch.float32,
                device=device,
            ),
            "train_y": y[train],
            "val_X": torch.tensor(
                X[val],
                dtype=torch.float32,
                device=device,
            ),
            "val_y": y[val],
            "original_auc": observed,
            "C": C,
        })

        print(
            f"{domain}: "
            f"train={train.sum()} "
            f"val={val.sum()} "
            f"original AUROC={observed:.6f} "
            f"C={C:g} "
            f"teacher logit std={logit_scale:.4g}",
            flush=True,
        )

    return parts, teachers


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, parts, teachers, s4):
    model.eval()
    rows = []

    for part in parts:

        w, b, scale = teachers[part["domain"]]
        x = part["val_X"]

        reconstructions = []
        active = 0

        for batch in x.split(BATCH_SIZE):

            rec, z, _ = model(batch)

            reconstructions.append(rec)

            active += int(
                (z > 0).sum().item()
            )

        rec = torch.cat(reconstructions)

        original_logits = (
            x @ w + b
        ).cpu().numpy()

        reconstructed_logits = (
            rec @ w + b
        ).cpu().numpy()

        auc = s4.safe_auc(
            part["val_y"],
            reconstructed_logits,
        )

        rows.append({
            "domain": part["domain"],
            "original_auroc": part["original_auc"],
            "frozen_probe_sae_auroc": auc,
            "auroc_gap": (
                part["original_auc"] - auc
            ),
            "mse": float(
                F.mse_loss(rec, x).item()
            ),
            "mean_l0": active / len(x),
            "normalized_logit_mse": float(
                np.mean(
                    (
                        (reconstructed_logits - original_logits)
                        / scale
                    ) ** 2
                )
            ),
        })

    return rows


# ============================================================
# FINE-TUNING EXPERIMENT
# ============================================================

def run_experiment(
    name,
    s4,
    sae_module,
    parts,
    teachers,
    device,
):
    # Every experiment starts from the SAME original
    # pretrained checkpoint, not another fine-tuned model.

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = s4.load_sae(
        sae_module,
        LAYER,
        ARCH,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    rng = torch.Generator(
        device="cpu"
    ).manual_seed(SEED)

    history = []

    before = evaluate(
        model,
        parts,
        teachers,
        s4,
    )

    print(
        f"\n{name}: baseline",
        before,
        flush=True,
    )

    for step in range(1, STEPS + 1):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction_losses = []
        truth_losses = []

        # Give geographic and numerical domains equal weight.

        for part in parts:

            indices = torch.randint(
                len(part["train_X"]),
                (BATCH_SIZE,),
                generator=rng,
            )

            x = part["train_X"][
                indices.to(device)
            ]

            rec, _, _ = model(x)

            reconstruction_losses.append(
                F.mse_loss(rec, x)
            )

            w, b, scale = teachers[
                part["domain"]
            ]

            teacher_logits = x @ w + b

            reconstructed_logits = rec @ w + b

            truth_losses.append(
                F.mse_loss(
                    reconstructed_logits / scale,
                    teacher_logits / scale,
                )
            )

        recon_loss = torch.stack(
            reconstruction_losses
        ).mean()

        truth_loss = torch.stack(
            truth_losses
        ).mean()

        if name == "truth_aware":
            loss = (
                recon_loss
                + TRUTH_WEIGHT * truth_loss
            )
        else:
            loss = recon_loss

        if not torch.isfinite(loss):
            raise ValueError(
                f"{name}: non-finite loss "
                f"at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            10.0,
        )

        optimizer.step()

        model.normalize_decoder()

        if (
            step == 1
            or step % LOG_EVERY == 0
            or step == STEPS
        ):

            entry = {
                "experiment": name,
                "step": step,
                "train_recon_mse": float(
                    recon_loss.item()
                ),
                "train_normalized_truth_loss": float(
                    truth_loss.item()
                ),
                "train_total_loss": float(
                    loss.item()
                ),
            }

            history.append(entry)

            print(
                f"{name} "
                f"step {step}/{STEPS} "
                f"recon="
                f"{entry['train_recon_mse']:.6f} "
                f"truth="
                f"{entry['train_normalized_truth_loss']:.6f}",
                flush=True,
            )

    after = evaluate(
        model,
        parts,
        teachers,
        s4,
    )

    for row in after:
        row.update({
            "experiment": name,
            "steps": STEPS,
            "lr": LR,
            "truth_weight": (
                TRUTH_WEIGHT
                if name == "truth_aware"
                else 0.0
            ),
        })

    folder = OUTPUT / name

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.cpu().state_dict()
            ),
            "source_checkpoint": str(
                s4.checkpoint_path(
                    LAYER,
                    ARCH,
                )
            ),
            "experiment": name,
            "steps": STEPS,
            "lr": LR,
            "truth_weight": (
                TRUTH_WEIGHT
                if name == "truth_aware"
                else 0.0
            ),
        },
        folder / "finetuned_checkpoint.pt",
    )

    pd.DataFrame(history).to_csv(
        folder / "training_log.csv",
        index=False,
    )

    print(
        f"{name}: validation",
        after,
        flush=True,
    )

    return before, after


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(4)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    s4 = import_stage4()

    sae_module = s4.load_sae_module()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Stage 6: Layer {LAYER} "
        f"{ARCH.upper()} fine-tuning "
        f"| device: {device}",
        flush=True,
    )

    print(
        "Train/validation only. "
        "Test untouched. "
        "Original checkpoint unchanged.",
        flush=True,
    )

    parts, teachers = load_data_and_teachers(
        s4,
        device,
    )

    before, reconstruction_rows = run_experiment(
        "reconstruction_only",
        s4,
        sae_module,
        parts,
        teachers,
        device,
    )

    _, truth_rows = run_experiment(
        "truth_aware",
        s4,
        sae_module,
        parts,
        teachers,
        device,
    )

    baseline_rows = [
        dict(
            row,
            experiment="pretrained",
            steps=0,
            lr=0.0,
            truth_weight=0.0,
        )
        for row in before
    ]

    results = pd.DataFrame(
        baseline_rows
        + reconstruction_rows
        + truth_rows
    )

    results.to_csv(
        OUTPUT / "validation_comparison.csv",
        index=False,
    )

    (OUTPUT / "run_info.json").write_text(
        json.dumps(
            {
                "layer": LAYER,
                "architecture": ARCH,
                "steps": STEPS,
                "batch_size_per_domain": BATCH_SIZE,
                "lr": LR,
                "truth_weight": TRUTH_WEIGHT,
                "seed": SEED,
                "teacher": (
                    "Original frozen logistic probes, "
                    "logit matching"
                ),
                "truth_loss": (
                    "Mean per-domain normalized "
                    "teacher logit MSE"
                ),
                "data": (
                    "Geometry of Truth "
                    "train/validation only"
                ),
                "test_evaluated": False,
                "source_checkpoint": str(
                    s4.checkpoint_path(
                        LAYER,
                        ARCH,
                    )
                ),
            },
            indent=2,
        )
    )

    print(
        "\nVALIDATION COMPARISON",
        flush=True,
    )

    print(
        results[
            [
                "experiment",
                "domain",
                "original_auroc",
                "frozen_probe_sae_auroc",
                "auroc_gap",
                "mse",
                "mean_l0",
            ]
        ].to_string(index=False),
        flush=True,
    )

    print(
        "\nSaved:",
        OUTPUT / "validation_comparison.csv",
        flush=True,
    )

    print(
        "Test set remains untouched.",
        flush=True,
    )


if __name__ == "__main__":
    main()
