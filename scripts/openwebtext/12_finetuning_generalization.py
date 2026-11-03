#!/usr/bin/env python3
"""Compare pretrained and fine-tuned layer-6 SAEs on OWT validation."""
import csv
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

BASELINE = Path("scripts/openwebtext/03_training_sae.py")
DATA = Path("data/activations/gpt2_small/validation")
CONVERGENCE = Path("results/openwebtext/convergence/layer_6")
FINETUNED = Path("results/got/sae_finetuning")
OUTPUT = Path(
    "results/openwebtext/reconstruction_evaluation/"
    "layer_6_finetuning_generalization"
)

LAYER = 6
ARCHITECTURES = ("relu", "topk", "jumprelu")
VARIANTS = ("pretrained", "reconstruction_only", "truth_aware")
BATCH_SIZE = 256


def load_sae_class():
    spec = importlib.util.spec_from_file_location("baseline_sae", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAE


def checkpoint_path(arch, variant):
    if variant == "pretrained":
        return (
            CONVERGENCE
            / f"{arch}_l0_128_resume_5000_to_15000"
            / "checkpoint_step_15000.pt"
        )
    return (
        FINETUNED
        / f"layer_6_{arch}"
        / variant
        / "finetuned_checkpoint.pt"
    )


def iter_validation_batches():
    files = sorted(DATA.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"No validation chunks in {DATA}")

    hook = "blocks.6.hook_resid_pre"
    for path in files:
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        activations = chunk["activations"][hook].float()
        tokens = chunk["tokens"]
        lengths = chunk["lengths"]

        selected = []
        for index, length in enumerate(lengths):
            length = int(length)
            if int(tokens[index, 0]) != 50256:
                raise ValueError(f"Unexpected BOS token in {path}")
            if length > 1:
                selected.append(activations[index, 1:length])

        if selected:
            x = torch.cat(selected, dim=0)
            for batch in x.split(BATCH_SIZE):
                yield batch


@torch.inference_mode()
def evaluate(model, device):
    squared_error = 0.0
    sum_cosine = 0.0
    sum_l0 = 0.0
    total_tokens = 0
    total_elements = 0
    sum_x = None
    sum_x_squared = 0.0

    for batch in iter_validation_batches():
        x = batch.to(device)
        reconstructed, z, _ = model(x)

        squared_error += (x - reconstructed).square().sum().item()
        sum_cosine += F.cosine_similarity(
            x, reconstructed, dim=-1
        ).sum().item()
        sum_l0 += (z != 0).sum().item()
        total_tokens += x.shape[0]
        total_elements += x.numel()

        batch_sum = x.double().sum(dim=0).cpu()
        sum_x = batch_sum if sum_x is None else sum_x + batch_sum
        sum_x_squared += x.double().square().sum().item()

    if total_tokens == 0:
        raise RuntimeError("No validation tokens found")

    total_variance = (
        sum_x_squared
        - sum_x.square().sum().item() / total_tokens
    )

    return {
        "validation_tokens": total_tokens,
        "mse": squared_error / total_elements,
        "mean_cosine_similarity": sum_cosine / total_tokens,
        "explained_variance": (
            1 - squared_error / total_variance
            if total_variance > 0 else float("nan")
        ),
        "actual_l0": sum_l0 / total_tokens,
    }


def main():
    torch.set_num_threads(4)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    SAE = load_sae_class()

    # Check all nine checkpoint paths before evaluating anything.
    for arch in ARCHITECTURES:
        for variant in VARIANTS:
            path = checkpoint_path(arch, variant)
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {path}")

    rows = []
    print("Device:", device, flush=True)
    print("Dataset: OpenWebText VALIDATION", flush=True)

    for arch in ARCHITECTURES:
        for variant in VARIANTS:
            path = checkpoint_path(arch, variant)
            ckpt = torch.load(
                path, map_location="cpu", weights_only=False
            )
            state = ckpt["model_state_dict"]
            d_sae, d_in = state["encoder.weight"].shape

            if (d_in, d_sae) != (768, 3072):
                raise ValueError(
                    f"Unexpected SAE dimensions in {path}"
                )

            model = SAE(
                d_in, d_sae, arch, 128, temperature=0.1
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()

            print(
                f"\nEvaluating {arch} / {variant}",
                flush=True
            )
            metrics = evaluate(model, device)

            rows.append({
                "layer": LAYER,
                "architecture": arch,
                "variant": variant,
                **metrics,
            })

            print(
                f"  tokens={metrics['validation_tokens']} "
                f"MSE={metrics['mse']:.6f} "
                f"EV={metrics['explained_variance']:.6f} "
                f"cosine={metrics['mean_cosine_similarity']:.6f} "
                f"L0={metrics['actual_l0']:.2f}",
                flush=True
            )
            del model

    OUTPUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT / "validation_generalization_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUTPUT / "validation_generalization_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print("\nSaved:", csv_path, flush=True)
    print("Saved:", json_path, flush=True)


if __name__ == "__main__":
    main()
