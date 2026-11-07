from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

data = torch.load(
    "data/repe/activations/gpt2_small/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

x = data["activations"][6].numpy()
labels = data["persona_labels"].numpy()
splits = np.array(data["splits"])
ids = np.array(data["statement_ids"])

train = splits == "train"
validation = splits == "validation"

direction = (
    x[train & (labels == 1)].mean(axis=0)
    - x[train & (labels == 0)].mean(axis=0)
)
direction /= np.linalg.norm(direction)

root = Path("results/repe/sae_reconstruction")
rows = []

def evaluate(name, activations):
    scores = activations[validation] @ direction
    y = labels[validation]

    auc = roc_auc_score(y, scores)

    pairs = pd.DataFrame({
        "id": ids[validation],
        "label": y,
        "score": scores,
    }).pivot(index="id", columns="label", values="score")

    margins = pairs[1] - pairs[0]

    return {
        "model": name,
        "validation_auroc": auc,
        "pair_accuracy": (margins > 0).mean(),
        "mean_pair_margin": margins.mean(),
        "median_pair_margin": margins.median(),
    }

rows.append(evaluate("original_gpt2", x))

for arch in ["relu", "topk", "jumprelu"]:
    for variant in ["pretrained", "reconstruction_only", "truth_aware"]:
        name = f"{arch}_{variant}"
        checkpoint = torch.load(
            root / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        reconstructed = checkpoint["reconstructed_activations"].numpy()
        assert reconstructed.shape == x.shape
        rows.append(evaluate(name, reconstructed))

results = pd.DataFrame(rows)

original_auc = results.loc[
    results["model"] == "original_gpt2", "validation_auroc"
].iloc[0]

results["auroc_gap"] = original_auc - results["validation_auroc"]

output = Path("results/repe/representation_preservation.csv")
results.to_csv(output, index=False)

print(results.round(4).to_string(index=False))
print("\nSaved:", output)
print("=== REPE REPRESENTATION PRESERVATION COMPLETE ===")
