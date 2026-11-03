from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

original = torch.load(
    "data/repe/activations/gpt2_small/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

alternative = torch.load(
    "data/repe/activations/cross_wording/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

x_train = original["activations"][6].numpy()
labels_train = original["persona_labels"].numpy()
train_mask = np.array(original["splits"]) == "train"

direction = (
    x_train[train_mask & (labels_train == 1)].mean(axis=0)
    - x_train[train_mask & (labels_train == 0)].mean(axis=0)
)
direction /= np.linalg.norm(direction)

labels = alternative["persona_labels"].numpy()
ids = np.array(alternative["statement_ids"])

def evaluate(name, activations):
    scores = activations @ direction

    pairs = pd.DataFrame({
        "id": ids,
        "label": labels,
        "score": scores,
    }).pivot(index="id", columns="label", values="score")

    margins = pairs[1] - pairs[0]

    return {
        "model": name,
        "auroc": roc_auc_score(labels, scores),
        "pair_accuracy": (margins > 0).mean(),
        "mean_margin": margins.mean(),
        "median_margin": margins.median(),
    }

baseline = evaluate(
    "original_gpt2",
    alternative["activations"][6].numpy(),
)

rows = [baseline]
root = Path("results/repe/cross_wording/reconstructions")

for arch in ["relu", "topk", "jumprelu"]:
    for variant in [
        "pretrained",
        "reconstruction_only",
        "truth_aware",
    ]:
        name = f"{arch}_{variant}"

        saved = torch.load(
            root / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )

        rows.append(
            evaluate(
                name,
                saved["reconstructed_activations"].numpy(),
            )
        )

results = pd.DataFrame(rows)
results["auroc_gap"] = baseline["auroc"] - results["auroc"]

output = Path("results/repe/cross_wording/evaluation.csv")
results.to_csv(output, index=False)

print("\n=== CROSS-WORDING SAE EVALUATION ===")
print(results.round(4).to_string(index=False))
print("\nSaved:", output)
print("=== CROSS-WORDING SAE EVALUATION COMPLETE ===")
