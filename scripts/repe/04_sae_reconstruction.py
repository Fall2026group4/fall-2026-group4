import importlib.util
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

SOURCE = Path("data/repe/activations/gpt2_small/repe_final_token.pt")
OUTPUT = Path("results/repe/sae_reconstruction")
OUTPUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "sae_training", "scripts/openwebtext/03_training_sae.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SAE = module.SAE

data = torch.load(SOURCE, map_location="cpu", weights_only=True)
x = data["activations"][6].float()
splits = data["splits"]

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []

for arch in ["relu", "topk", "jumprelu"]:
    paths = {
        "pretrained": Path(
            f"results/openwebtext/convergence/layer_6/"
            f"{arch}_l0_128_resume_5000_to_15000/checkpoint_step_15000.pt"
        ),
        "reconstruction_only": Path(
            f"results/geometry_of_truth/sae_finetuning/layer_6_{arch}/"
            f"reconstruction_only/finetuned_checkpoint.pt"
        ),
        "truth_aware": Path(
            f"results/geometry_of_truth/sae_finetuning/layer_6_{arch}/"
            f"truth_aware/finetuned_checkpoint.pt"
        ),
    }

    for variant, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]

        model = SAE(
            d_in=768,
            d_sae=3072,
            architecture=arch,
            k=128,
            temperature=0.1,
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        reconstructed_parts = []
        active_parts = []

        with torch.inference_mode():
            for batch in x.split(256):
                rec, z, _ = model(batch.to(device))
                reconstructed_parts.append(rec.cpu())
                active_parts.append((z > 0).sum(dim=-1).cpu())

        rec = torch.cat(reconstructed_parts)
        active = torch.cat(active_parts)

        assert rec.shape == x.shape
        assert torch.isfinite(rec).all()

        torch.save(
            {
                "architecture": arch,
                "variant": variant,
                "layer": 6,
                "reconstructed_activations": rec,
                "active_features": active,
            },
            OUTPUT / f"{arch}_{variant}.pt",
        )

        for split in ["train", "validation", "test"]:
            mask = torch.tensor([s == split for s in splits])
            original = x[mask]
            rebuilt = rec[mask]

            squared_error = (original - rebuilt).square().sum().item()
            total_variance = (
                (original - original.mean(dim=0)).square().sum().item()
            )

            rows.append({
                "architecture": arch,
                "variant": variant,
                "split": split,
                "mse": F.mse_loss(rebuilt, original).item(),
                "cosine_similarity": F.cosine_similarity(
                    rebuilt, original, dim=-1
                ).mean().item(),
                "explained_variance": (
                    1 - squared_error / max(total_variance, 1e-12)
                ),
                "mean_l0": active[mask].float().mean().item(),
            })

        print(f"Completed: {arch} / {variant}", flush=True)

result = pd.DataFrame(rows)
result.to_csv(OUTPUT / "reconstruction_metrics.csv", index=False)

print("\n=== VALIDATION RECONSTRUCTION METRICS ===")
print(
    result[result["split"] == "validation"].to_string(index=False)
)
print("\n=== REPE SAE RECONSTRUCTION COMPLETE ===")
