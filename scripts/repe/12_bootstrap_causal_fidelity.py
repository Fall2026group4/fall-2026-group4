import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ns = runpy.run_path("scripts/repe/11_heldout_steering.py")

evaluate = ns["evaluate"]
directions = ns["directions"]
device = ns["device"]

baseline = np.array(evaluate(None, 0))

models = [
    name for name in directions
    if not name.startswith("random_")
    and name != "original_gpt2"
]

alphas = [-10, 10]
errors = {}

for alpha in alphas:
    original = (
        np.array(
            evaluate(
                directions["original_gpt2"].to(device),
                alpha
            )
        ) - baseline
    )

    for name in models:
        sae_effect = (
            np.array(
                evaluate(directions[name].to(device), alpha)
            ) - baseline
        )

        errors[(name, alpha)] = np.abs(
            original - sae_effect
        )

rng = np.random.default_rng(42)
n = len(baseline)
B = 10000

# Resample the same questions for every model and alpha.
indices = rng.integers(0, n, size=(B, n))

bootstrap_scores = {}
rows = []

for name in models:
    combined = np.mean(
        np.stack([
            errors[(name, alpha)]
            for alpha in alphas
        ]),
        axis=0
    )

    samples = combined[indices].mean(axis=1)
    bootstrap_scores[name] = samples

    rows.append({
        "model": name,
        "mean_mae": combined.mean(),
        "ci_lower": np.percentile(samples, 2.5),
        "ci_upper": np.percentile(samples, 97.5),
    })

results = pd.DataFrame(rows).sort_values("mean_mae")

output_dir = Path("results/repe/steering")
output_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output_dir / "bootstrap_fidelity_ci.csv",
    index=False
)

print("\n=== 95% BOOTSTRAP CONFIDENCE INTERVALS ===")
print(results.round(4).to_string(index=False))

# Paired comparison of leading configurations
best = "jumprelu_truth_aware"

for competitor in [
    "topk_truth_aware",
    "relu_reconstruction_only",
    "jumprelu_pretrained",
]:
    difference = (
        bootstrap_scores[competitor]
        - bootstrap_scores[best]
    )

    print(f"\n{competitor} minus {best}")
    print("Mean MAE difference:", round(difference.mean(), 4))
    print(
        "95% CI:",
        np.round(np.percentile(difference, [2.5, 97.5]), 4)
    )

print("\n=== STEP 82 COMPLETE ===")
