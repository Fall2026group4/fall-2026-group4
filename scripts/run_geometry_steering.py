"""Stage 5: Causal steering evaluation for Geometry of Truth."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.geometry_steering import (
    apply_direction_steering,
    apply_probe_direction_control,
    feature_scale_from_train,
    get_probe_scores,
    get_raw_probe_direction,
    get_sae_decoder_direction,
    train_fixed_truth_probe,
)
from src.sae.geometry_sae import (
    SAE_LAYER,
    load_geometry_sae,
    select_sae_layer,
)


SEED = 42

ACTIVATION_DIR = Path(
    "data/processed/geometry_of_truth/activations"
)

FEATURE_DIR = Path(
    "data/processed/geometry_of_truth/sae_features"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

TRAIN_FEATURE_RESULTS = (
    RESULTS_DIR / "sae_top_features_train.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR / "sae_steering_results.csv"
)

# Positive multiplier = intervention toward TRUE.
# Negative multiplier = intervention toward FALSE.
STRENGTH_MULTIPLIERS = [
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
]

# We already established this as the top TRAIN-selected feature.
EXPECTED_CANDIDATE_FEATURE = 19579


def summarize_condition(
    condition,
    feature_id,
    multiplier,
    steering_strength,
    perturbation_l2,
    baseline_scores,
    steered_scores,
    labels,
):
    """Create one result row for one intervention condition."""

    shifts = (
        steered_scores
        - baseline_scores
    )

    labels = np.asarray(labels)

    true_mask = labels == 1
    false_mask = labels == 0

    return {
        "condition": condition,
        "feature_id": feature_id,
        "multiplier": multiplier,
        "steering_strength": steering_strength,
        "perturbation_l2": perturbation_l2,
        "mean_baseline_probe_score": (
            float(np.mean(baseline_scores))
        ),
        "mean_steered_probe_score": (
            float(np.mean(steered_scores))
        ),
        "mean_probe_score_shift": (
            float(np.mean(shifts))
        ),
        "median_probe_score_shift": (
            float(np.median(shifts))
        ),
        "mean_true_example_shift": (
            float(np.mean(shifts[true_mask]))
        ),
        "mean_false_example_shift": (
            float(np.mean(shifts[false_mask]))
        ),
        "fraction_shifted_toward_true": (
            float(np.mean(shifts > 0))
        ),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Stage 5 — Geometry SAE Steering")

    
    # Load raw GPT-2 Layer-10 residuals
    

    train_activation_cache = torch.load(
        ACTIVATION_DIR / "train_activations.pt",
        map_location="cpu",
    )

    test_activation_cache = torch.load(
        ACTIVATION_DIR / "test_activations.pt",
        map_location="cpu",
    )

    train_residuals = select_sae_layer(
        train_activation_cache["activations"],
        layer=SAE_LAYER,
    )

    test_residuals = select_sae_layer(
        test_activation_cache["activations"],
        layer=SAE_LAYER,
    )

    train_labels = (
        train_activation_cache["labels"]
    )

    test_labels = (
        test_activation_cache["labels"]
    )

    
    # Load SAE feature activations
    

    train_feature_cache = torch.load(
        FEATURE_DIR / "train_sae_features.pt",
        map_location="cpu",
    )

    train_features = (
        train_feature_cache["features"]
    )


    # Lock candidate using TRAIN ranking only
    

    ranking = pd.read_csv(
        TRAIN_FEATURE_RESULTS
    )

    candidate_feature = int(
        ranking.iloc[0]["feature_id"]
    )

    candidate_polarity = int(
        ranking.iloc[0]["polarity"]
    )

    if candidate_feature != EXPECTED_CANDIDATE_FEATURE:
        raise ValueError(
            "Unexpected top TRAIN feature: "
            f"{candidate_feature}"
        )

    print(
        f"Locked TRAIN-selected feature: "
        f"{candidate_feature}"
    )

    print(
        f"TRAIN polarity: "
        f"{candidate_polarity}"
    )


    # Train fixed Layer-10 truth probe


    probe = train_fixed_truth_probe(
        train_residuals=train_residuals,
        train_labels=train_labels,
        seed=SEED,
    )

    baseline_scores = get_probe_scores(
        probe,
        test_residuals,
    )

    # Load SAE and candidate decoder direction


    sae = load_geometry_sae(
        device="cpu"
    )

    candidate_direction = (
        get_sae_decoder_direction(
            sae,
            candidate_feature,
        )
    )

    candidate_feature_scale = (
        feature_scale_from_train(
            train_features,
            candidate_feature,
        )
    )

    candidate_direction_norm = float(
        torch.linalg.vector_norm(
            candidate_direction
        ).item()
    )

    print(
        "Candidate TRAIN feature SD:",
        round(candidate_feature_scale, 6),
    )

    print(
        "Candidate decoder norm:",
        round(candidate_direction_norm, 6),
    )

    # Deterministic random-feature control
    # Exclude all top-50 TRAIN-selected candidates.
    

    excluded_features = set(
        ranking["feature_id"]
        .astype(int)
        .tolist()
    )

    feature_std = (
        train_features
        .float()
        .std(dim=0)
    )

    eligible = [
        feature_id
        for feature_id in range(
            train_features.shape[1]
        )
        if (
            feature_id not in excluded_features
            and feature_std[feature_id].item() > 0
        )
    ]

    rng = np.random.default_rng(
        SEED
    )

    random_feature = int(
        rng.choice(eligible)
    )

    random_direction = (
        get_sae_decoder_direction(
            sae,
            random_feature,
        )
    )

    random_direction_norm = float(
        torch.linalg.vector_norm(
            random_direction
        ).item()
    )

    print(
        f"Random control feature: "
        f"{random_feature}"
    )

    # Steering sweep

    rows = []

    labels_np = (
        test_labels
        .detach()
        .cpu()
        .numpy()
    )

    for multiplier in STRENGTH_MULTIPLIERS:

        # candidate_polarity = -1 means decreasing
        # feature activation corresponds to TRUE direction.
        candidate_strength = (
            multiplier
            * candidate_polarity
            * candidate_feature_scale
        )

        perturbation_l2 = (
            abs(candidate_strength)
            * candidate_direction_norm
        )

        candidate_steered = (
            apply_direction_steering(
                residuals=test_residuals,
                direction=candidate_direction,
                steering_strength=candidate_strength,
            )
        )

        candidate_scores = (
            get_probe_scores(
                probe,
                candidate_steered,
            )
        )

        rows.append(
            summarize_condition(
                condition="sae_candidate",
                feature_id=candidate_feature,
                multiplier=multiplier,
                steering_strength=(
                    candidate_strength
                ),
                perturbation_l2=(
                    perturbation_l2
                ),
                baseline_scores=(
                    baseline_scores
                ),
                steered_scores=(
                    candidate_scores
                ),
                labels=labels_np,
            )
        )

    
        # Random SAE feature control
        # Match residual-space L2 perturbation magnitude.

        if random_direction_norm == 0:
            raise ValueError(
                "Random decoder direction has zero norm."
            )

        random_strength = (
            np.sign(multiplier)
            * perturbation_l2
            / random_direction_norm
            if multiplier != 0
            else 0.0
        )

        random_steered = (
            apply_direction_steering(
                residuals=test_residuals,
                direction=random_direction,
                steering_strength=random_strength,
            )
        )

        random_scores = (
            get_probe_scores(
                probe,
                random_steered,
            )
        )

        rows.append(
            summarize_condition(
                condition="random_sae_feature",
                feature_id=random_feature,
                multiplier=multiplier,
                steering_strength=(
                    random_strength
                ),
                perturbation_l2=(
                    perturbation_l2
                ),
                baseline_scores=(
                    baseline_scores
                ),
                steered_scores=(
                    random_scores
                ),
                labels=labels_np,
            )
        )


        # Probe-direction positive control
        # Positive multiplier = toward TRUE.
        # Match the same residual L2 magnitude.
    

        probe_steered = (
            apply_probe_direction_control(
                residuals=test_residuals,
                probe=probe,
                steering_strength=(
                    multiplier
                    * candidate_feature_scale
                    * candidate_direction_norm
                ),
                toward_true=True,
            )
        )

        probe_scores = (
            get_probe_scores(
                probe,
                probe_steered,
            )
        )

        rows.append(
            summarize_condition(
                condition="probe_direction",
                feature_id=-1,
                multiplier=multiplier,
                steering_strength=(
                    multiplier
                    * candidate_feature_scale
                    * candidate_direction_norm
                ),
                perturbation_l2=(
                    perturbation_l2
                ),
                baseline_scores=(
                    baseline_scores
                ),
                steered_scores=(
                    probe_scores
                ),
                labels=labels_np,
            )
        )


    # Save results
    
    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nStage 5 complete.")
    print(
        f"Results saved: "
        f"{OUTPUT_PATH}"
    )

    print("\nMean truth-probe score shifts:")
    print(
        results[
            [
                "condition",
                "multiplier",
                "mean_probe_score_shift",
                "perturbation_l2",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()