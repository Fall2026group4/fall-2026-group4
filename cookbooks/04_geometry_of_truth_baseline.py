"""Geometry of Truth - GPT-2 + Sparse Autoencoder Baseline.

For each labeled true/false statement in the Geometry of Truth "cities"
dataset:

1. Tokenize with GPT-2 Small.
2. Run GPT-2 and capture the last-token residual stream.
3. Encode that representation with the pretrained SAE used elsewhere in
   this project (`src/sae/geometry_sae.py`, `blocks.10.hook_resid_pre`,
   matching the `scripts/run_geometry_sae.py` pipeline).
4. Compare SAE feature activation statistics between true and false
   statements and surface the strongest candidate features.

All model/SAE loading and feature-statistics logic lives in `src/`
(`src/models/gpt2_activations.py`, `src/sae/geometry_sae.py`,
`src/evaluation/geometry_truth.py`) - this script only calls into it and
renders the walkthrough output. The full batch pipeline (train/test split,
caching, detection, faithfulness) lives in `scripts/` as separate stages;
this notebook-turned-script is the narrative, single-pass version of the
same pipeline for exploration, kept consistent with it rather than
re-implementing the loading/encoding code with different defaults.

Requires: torch, transformer-lens, sae-lens, pandas, numpy, matplotlib
    pip install torch transformer-lens sae-lens pandas numpy matplotlib

Run from anywhere:
    python cookbooks/04_geometry_of_truth_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "results" / "figures"
RESULTS_DIR = REPO_ROOT / "results" / "geometry_of_truth"

DATASET_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv"
SEED = 42
TOP_N_CANDIDATES = 20


def main(device: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import torch

    from src.evaluation.geometry_truth import (
        compute_feature_statistics,
        load_geometry_dataset,
        rank_candidate_features,
        split_true_false,
    )
    from src.models.gpt2_activations import extract_last_token_residuals, load_gpt2_small
    from src.sae.geometry_sae import (
        SAE_ID,
        SAE_LAYER,
        SAE_RELEASE,
        encode_residual_activations,
        load_geometry_sae,
        select_sae_layer,
    )

    torch.manual_seed(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the dataset once (single source of truth for this walkthrough -
    # the original notebook fetched the same CSV a second time in a later
    # cell with a different column-selection step, which could silently
    # diverge from the df used for the earlier true/false split).
    df = load_geometry_dataset(DATASET_URL)
    df = df[["statement", "label"]].dropna().reset_index(drop=True)
    df["statement"] = df["statement"].astype(str)
    df["label"] = df["label"].astype(int)
    print("Rows:", len(df))
    print(df["label"].value_counts())

    true_df, false_df = split_true_false(df)
    print("True rows:", true_df.shape)
    print("False rows:", false_df.shape)

    # 2. Load GPT-2 Small via the shared loader (src/models/gpt2_activations.py)
    # instead of calling TransformerLens directly, so this stays consistent
    # with scripts/cache_geometry_activations.py.
    print("\nLoading GPT-2 Small...")
    model = load_gpt2_small(device=device)
    print("Model:", model.cfg.model_name)
    print("d_model:", model.cfg.d_model)

    # 3. Load the pretrained SAE via the shared loader (src/sae/geometry_sae.py).
    # Uses SAE_LAYER (10) consistently with scripts/run_geometry_sae.py -
    # the original notebook hardcoded layer 8 here, which would have
    # produced results inconsistent with the rest of the Geometry of Truth
    # pipeline.
    print("\nLoading pretrained SAE...")
    sae = load_geometry_sae(device=device)
    hook_name = f"blocks.{SAE_LAYER}.hook_resid_pre"
    print("SAE release:", SAE_RELEASE)
    print("SAE ID:", SAE_ID)
    print("SAE input dimension:", sae.cfg.d_in)
    print("SAE feature dimension:", sae.cfg.d_sae)
    assert sae.cfg.d_in == model.cfg.d_model, (
        f"SAE expects {sae.cfg.d_in} dimensions but GPT-2 has {model.cfg.d_model}."
    )

    # 4. Sanity check with one statement.
    example_statement = df.iloc[0]["statement"]
    example_label = int(df.iloc[0]["label"])
    example_residuals = extract_last_token_residuals(
        model, [example_statement], batch_size=1, device=device
    )
    example_layer_residual = select_sae_layer(example_residuals, layer=SAE_LAYER)
    example_features = encode_residual_activations(sae, example_layer_residual)
    print("\nStatement:", example_statement)
    print("Label:", example_label)
    print("Hook:", hook_name)
    print("Selected representation:", tuple(example_layer_residual.shape))
    print("SAE feature shape:", tuple(example_features.shape))
    print("Active SAE features:", int((example_features[0] > 0).sum().item()))

    # 5. Extract SAE features for the full dataset (batched, via the shared
    # extraction + encoding helpers - no inline reimplementation here).
    print("\nExtracting GPT-2 residuals + SAE features for all statements...")
    texts = df["statement"].tolist()
    residuals = extract_last_token_residuals(model, texts, batch_size=16, device=device)
    layer_residuals = select_sae_layer(residuals, layer=SAE_LAYER)
    sae_features = encode_residual_activations(sae, layer_residuals, batch_size=64)
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    print("Feature matrix:", tuple(sae_features.shape))
    print("Labels:", tuple(labels.shape))
    assert sae_features.shape[0] == len(df)

    # 6. Separate true and false statements.
    true_features = sae_features[labels == 1]
    false_features = sae_features[labels == 0]
    print("\nTrue feature matrix :", tuple(true_features.shape))
    print("False feature matrix:", tuple(false_features.shape))

    # 7. Baseline feature statistics + top candidate features.
    feature_stats = compute_feature_statistics(true_features, false_features)
    print(feature_stats.head())

    top_true, top_false = rank_candidate_features(feature_stats, top_n=TOP_N_CANDIDATES)
    print("\nFeatures with larger mean activation for TRUE statements:")
    print(top_true.to_string(index=False))
    print("\nFeatures with larger mean activation for FALSE statements:")
    print(top_false.to_string(index=False))

    # 8. Inspect the statements that most activate the top candidate feature.
    candidate_feature = int(top_true.iloc[0]["feature_id"])
    values = sae_features[:, candidate_feature].numpy()
    inspection = df[["statement", "label"]].copy()
    inspection["activation"] = values
    print(f"\nTop statements activating candidate feature {candidate_feature}:")
    print(inspection.sort_values("activation", ascending=False).head(10).to_string(index=False))

    # 9. Plot the largest absolute mean-activation differences.
    plot_df = (
        feature_stats.assign(abs_difference=lambda x: x["mean_activation_difference"].abs())
        .nlargest(15, "abs_difference")
        .sort_values("mean_activation_difference")
    )
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature_id"].astype(str), plot_df["mean_activation_difference"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean activation difference (True - False)")
    plt.ylabel("SAE Feature ID")
    plt.title("Geometry of Truth: Largest SAE Feature Differences")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "geometry_of_truth_sae_feature_differences.svg", bbox_inches="tight"
    )
    plt.close()

    # 10. Save baseline outputs so later stages (probes, detection,
    # faithfulness - see scripts/) can reuse them without rerunning
    # GPT-2 + SAE extraction.
    feature_stats.to_csv(RESULTS_DIR / "geometry_of_truth_feature_stats.csv", index=False)
    torch.save(
        {
            "features": sae_features,
            "labels": labels,
            "statements": df["statement"].tolist(),
            "hook_name": hook_name,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
        },
        RESULTS_DIR / "geometry_of_truth_sae_features.pt",
    )
    print("\nSaved to:", RESULTS_DIR.resolve())
    print("Figure saved to:", FIGURES_DIR.resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=None,
        help="Device to load the model/SAE on (defaults to cuda if available, else cpu).",
    )
    args = parser.parse_args()
    main(device=args.device)
