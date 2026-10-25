"""Pretrained SAE Baseline and Steering Experiment.

Evaluates a pretrained sparse autoencoder (SAE) on GPT-2 Small. Examines
SAE feature activations, tests Feature 974 ("mentions of Paris") on
positive and negative examples, and performs a causal steering intervention.

Model/SAE loading, the steering hook, and the results-table logic all live
in `src/` (`src/models/gpt2_sae.py`, `src/sae/steering.py`,
`src/evaluation/detection.py`) and are covered by tests in `src/tests/`.
This script only calls into them and renders figures/tables - it does not
reimplement any of that logic.

Run from anywhere:
    python cookbooks/02_pretained_sae_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "results" / "figures"
TABLES_DIR = REPO_ROOT / "results" / "tables"

FEATURE_ID = 974
STEERING_STRENGTHS = [10, 20, 30, 50, 75, 100]

POSITIVE_EXAMPLES = [
    "I traveled to Paris last summer.",
    "Paris is the capital of France.",
    "She moved to Paris for work.",
    "The conference will be held in Paris.",
    "We visited Paris during our vacation.",
]

NEGATIVE_EXAMPLES = [
    "I traveled to London last summer.",
    "Berlin is the capital of Germany.",
    "She moved to Chicago for work.",
    "The conference will be held in Tokyo.",
    "We visited Rome during our vacation.",
]


def main(device: str = "cpu") -> None:
    from src.evaluation.detection import build_steering_results, score_examples
    from src.models.gpt2_sae import get_feature_activation, load_sae_bundle
    from src.sae.steering import run_steering_sweep

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Loads GPT-2 Small and the gpt2-small-res-jb SAE at blocks.8.hook_resid_pre.
    # See src/models/gpt2_sae.py for the defaults and src/tests/ for coverage
    # of the pure-logic pieces (this call needs network + model weights, so
    # it isn't unit tested directly).
    bundle = load_sae_bundle(device=device)
    print("Loaded model + SAE. Hook point:", bundle.hook_name)

    text = "The Eiffel Tower is located in Paris."
    activation = get_feature_activation(bundle, text, FEATURE_ID)
    print(f"Feature {FEATURE_ID} activation on example sentence: {activation:.4f}")

    print("Positive examples:", len(POSITIVE_EXAMPLES))
    print("Negative examples:", len(NEGATIVE_EXAMPLES))

    results_df = score_examples(bundle, POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, FEATURE_ID)
    print(results_df)

    positive_mean = results_df.loc[results_df["label"] == "positive", "feature_activation"].mean()
    negative_mean = results_df.loc[results_df["label"] == "negative", "feature_activation"].mean()
    print(f"Average positive activation: {positive_mean:.2f}")
    print(f"Average negative activation: {negative_mean:.2f}")

    labels = ["Paris examples", "Non-Paris examples"]
    means = [positive_mean, negative_mean]
    plt.bar(labels, means)
    plt.ylabel(f"Mean Feature {FEATURE_ID} Activation")
    plt.title(f"Feature {FEATURE_ID}: Paris vs Non-Paris Examples")
    # Vector output only -- PNG is not accepted for results figures (Review #2).
    plt.savefig(FIGURES_DIR / "feature_974_paris_vs_nonparis.svg", bbox_inches="tight")
    plt.close()

    results_df.to_csv(TABLES_DIR / "feature_974_detection_results.csv", index=False)
    print("Detection results saved.")

    # Steering intervention: one sweep, using run_steering_sweep from
    # src/sae/steering.py (fixed-seed, single code path for baseline vs
    # steered generations - see src/tests/test_steering.py).
    prompt = "The best place to spend my vacation is"
    steering_direction = bundle.sae.W_dec[FEATURE_ID]

    steering_outputs = run_steering_sweep(
        bundle.sae_model,
        prompt=prompt,
        hook_name=bundle.hook_name,
        steering_direction=steering_direction,
        strengths=STEERING_STRENGTHS,
        seed=0,
    )

    for strength in sorted(steering_outputs):
        print(f"Strength {strength}: {steering_outputs[strength]}")

    steering_results = build_steering_results(steering_outputs, concept_word="paris")
    print(steering_results)

    steering_results.to_csv(TABLES_DIR / "feature_974_steering_results.csv", index=False)
    print("Steering results saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the model/SAE on (e.g. 'cpu' or 'cuda').",
    )
    args = parser.parse_args()
    main(device=args.device)
