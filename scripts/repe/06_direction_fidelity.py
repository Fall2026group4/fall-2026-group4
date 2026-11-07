from pathlib import Path
import numpy as np
import pandas as pd
import torch

source = torch.load(
    "data/repe/activations/gpt2_small/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

x = source["activations"][6].numpy()
labels = source["persona_labels"].numpy()
splits = np.array(source["splits"])
train = splits == "train"

def mean_difference(values):
    return (
        values[train & (labels == 1)].mean(axis=0)
        - values[train & (labels == 0)].mean(axis=0)
    )

original = mean_difference(x)
original_norm = np.linalg.norm(original)

root = Path("results/repe/sae_reconstruction")
rows = []

for arch in ["relu", "topk", "jumprelu"]:
    for variant in ["pretrained", "reconstruction_only", "truth_aware"]:
        name = f"{arch}_{variant}"

        saved = torch.load(
            root / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        reconstructed = saved["reconstructed_activations"].numpy()

        direction = mean_difference(reconstructed)
        direction_norm = np.linalg.norm(direction)

        cosine = np.dot(original, direction) / (
            original_norm * direction_norm
        )

        rows.append({
            "model": name,
            "direction_cosine": cosine,
            "original_direction_norm": original_norm,
            "sae_direction_norm": direction_norm,
            "norm_ratio": direction_norm / original_norm,
        })

results = pd.DataFrame(rows)
output = Path("results/repe/direction_fidelity.csv")
results.to_csv(output, index=False)

print(results.round(4).to_string(index=False))
print("\nSaved:", output)
print("=== REPE DIRECTION FIDELITY COMPLETE ===")
