from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

source = Path("data/repe/activations/gpt2_small/repe_final_token.pt")
output = Path("results/repe/original_baseline.csv")
output.parent.mkdir(parents=True, exist_ok=True)

data = torch.load(source, map_location="cpu", weights_only=True)

splits = np.array(data["splits"])
labels = data["persona_labels"].numpy()
ids = np.array(data["statement_ids"])

train = splits == "train"
validation = splits == "validation"

results = []

for layer in range(12):
    X = data["activations"][layer].numpy()

    honest_mean = X[train & (labels == 1)].mean(axis=0)
    untruthful_mean = X[train & (labels == 0)].mean(axis=0)

    direction = honest_mean - untruthful_mean
    norm = np.linalg.norm(direction)

    if norm == 0:
        raise ValueError(f"Zero direction at layer {layer}")

    direction = direction / norm

    scores = X[validation] @ direction
    y = labels[validation]
    pair_ids = ids[validation]

    auc = roc_auc_score(y, scores)

    pairs = pd.DataFrame({
        "id": pair_ids,
        "label": y,
        "score": scores,
    })

    pivot = pairs.pivot(index="id", columns="label", values="score")
    pair_accuracy = (pivot[1] > pivot[0]).mean()

    results.append({
        "layer": layer,
        "validation_auroc": auc,
        "validation_pair_accuracy": pair_accuracy,
        "direction_norm": norm,
    })

    print(
        f"Layer {layer:2d} | "
        f"AUROC: {auc:.4f} | "
        f"Pair accuracy: {pair_accuracy:.4f}"
    )

pd.DataFrame(results).to_csv(output, index=False)

print(f"\nSaved: {output}")
print("=== REPE STAGE 3 BASELINE COMPLETE ===")
