import importlib.util
from pathlib import Path

import torch

SOURCE = Path(
    "data/repe/activations/cross_wording/repe_final_token.pt"
)
OUTPUT = Path("results/repe/cross_wording/reconstructions")
OUTPUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "sae_training",
    "scripts/openwebtext/03_training_sae.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SAE = module.SAE

data = torch.load(SOURCE, map_location="cpu", weights_only=True)
x = data["activations"][6].float()

assert x.shape == (180, 768)
assert set(data["splits"]) == {"validation"}

device = "cuda" if torch.cuda.is_available() else "cpu"

for arch in ["relu", "topk", "jumprelu"]:
    paths = {
        "pretrained": Path(
            f"results/openwebtext/convergence/layer_6/"
            f"{arch}_l0_128_resume_5000_to_15000/"
            f"checkpoint_step_15000.pt"
        ),
        "reconstruction_only": Path(
            f"results/got/sae_finetuning/layer_6_{arch}/"
            f"reconstruction_only/finetuned_checkpoint.pt"
        ),
        "truth_aware": Path(
            f"results/got/sae_finetuning/layer_6_{arch}/"
            f"truth_aware/finetuned_checkpoint.pt"
        ),
    }

    for variant, path in paths.items():
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )

        model = SAE(
            d_in=768,
            d_sae=3072,
            architecture=arch,
            k=128,
            temperature=0.1,
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        model.eval()

        parts = []

        with torch.inference_mode():
            for batch in x.split(256):
                rec, _, _ = model(batch.to(device))
                parts.append(rec.cpu())

        reconstructed = torch.cat(parts)

        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed).all()

        name = f"{arch}_{variant}"

        torch.save(
            {
                "architecture": arch,
                "variant": variant,
                "layer": 6,
                "reconstructed_activations": reconstructed,
            },
            OUTPUT / f"{name}.pt",
        )

        print(f"Completed: {name}", flush=True)

print("\n=== CROSS-WORDING SAE RECONSTRUCTION COMPLETE ===")
