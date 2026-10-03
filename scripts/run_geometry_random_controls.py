"""Stage 5B: Multi-random SAE feature controls for Geometry steering."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from src.evaluation.geometry_steering import (
    apply_direction_steering,
    feature_scale_from_train,
    get_probe_scores,
    get_sae_decoder_direction,
    train_fixed_truth_probe,
)
from src.sae.geometry_sae import (
    SAE_LAYER,
    load_geometry_sae,
    select_sae_layer,
)


SEED = 42
N_RANDOM = 100

CANDIDATE_FEATURE = 19579

ACTIVATION_DIR = Path(
    "data/processed/geometry_of_truth/activations"
)

FEATURE_DIR = Path(
    "data/processed/geometry_of_truth/sae_features"
)

RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

TRAIN_RANKING_PATH = (
    RESULTS_DIR / "sae_top_features_train.csv"
)

RANDOM_RESULTS_PATH = (
    RESULTS_DIR / "sae_random_control_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "sae_random_control_summary.csv"
)


def get_train_polarity(
    values,
    labels,
):
    """Determine TRUE/FALSE direction using TRAIN data only."""

    values = np.asarray(values)
    labels = np.asarray(labels)

    raw_auc = roc_auc_score(
        labels,
        values,
    )

    polarity = (
        1
        if raw_auc >= 0.5
        else -1
    )

    detection_auc = max(
        raw_auc,
        1.0 - raw_auc,
    )

    return (
        raw_auc,
        detection_auc,
        polarity,
    )


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(
        "Stage 5B — 100 Random SAE Feature Controls"
    )

    # ---------------------------------------------------------
    # Load Stage 1 residual activations
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Load TRAIN SAE features
    # ---------------------------------------------------------

    train_feature_cache = torch.load(
        FEATURE_DIR / "train_sae_features.pt",
        map_location="cpu",
    )

    train_features = (
        train_feature_cache["features"]
    )

    train_labels_np = (
        train_labels
        .detach()
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # Lock candidate and TRAIN-derived polarity
    # ---------------------------------------------------------

    ranking = pd.read_csv(
        TRAIN_RANKING_PATH
    )

    candidate_row = ranking.loc[
        ranking["feature_id"]
        == CANDIDATE_FEATURE
    ].iloc[0]

    candidate_polarity = int(
        candidate_row["polarity"]
    )

    print(
        f"Candidate feature: "
        f"{CANDIDATE_FEATURE}"
    )

    print(
        f"Candidate TRAIN polarity: "
        f"{candidate_polarity}"
    )

    # ---------------------------------------------------------
    # Train fixed truth probe
    # ---------------------------------------------------------

    probe = train_fixed_truth_probe(
        train_residuals=train_residuals,
        train_labels=train_labels,
        seed=SEED,
    )

    baseline_scores = get_probe_scores(
        probe,
        test_residuals,
    )

    # ---------------------------------------------------------
    # Load SAE
    # ---------------------------------------------------------

    sae = load_geometry_sae(
        device="cpu"
    )

    candidate_direction = (
        get_sae_decoder_direction(
            sae,
            CANDIDATE_FEATURE,
        )
    )

    candidate_direction_norm = float(
        torch.linalg.vector_norm(
            candidate_direction
        ).item()
    )

    candidate_feature_scale = (
        feature_scale_from_train(
            train_features,
            CANDIDATE_FEATURE,
        )
    )

    # +1 truth-directed candidate intervention
    candidate_strength = (
        candidate_polarity
        * candidate_feature_scale
    )

    target_l2 = (
        abs(candidate_strength)
        * candidate_direction_norm
    )

    candidate_steered = (
        apply_direction_steering(
            residuals=test_residuals,
            direction=candidate_direction,
            steering_strength=(
                candidate_strength
            ),
        )
    )

    candidate_scores = get_probe_scores(
        probe,
        candidate_steered,
    )

    candidate_shift = float(
        np.mean(
            candidate_scores
            - baseline_scores
        )
    )

    print(
        "Candidate +1 truth-directed shift:",
        round(candidate_shift, 6),
    )

    print(
        "Target perturbation L2:",
        round(target_l2, 6),
    )

    # ---------------------------------------------------------
    # Choose random SAE features
    # ---------------------------------------------------------

    # Exclude the top-50 TRAIN-selected features.
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

    eligible_features = np.array(
        [
            feature_id
            for feature_id in range(
                train_features.shape[1]
            )
            if (
                feature_id not in excluded_features
                and feature_std[
                    feature_id
                ].item() > 0
            )
        ],
        dtype=int,
    )

    rng = np.random.default_rng(
        SEED
    )

    random_features = rng.choice(
        eligible_features,
        size=N_RANDOM,
        replace=False,
    )

    # ---------------------------------------------------------
    # Evaluate random controls
    # ---------------------------------------------------------

    rows = []

    for i, feature_id in enumerate(
        random_features,
        start=1,
    ):
        feature_id = int(
            feature_id
        )

        values = (
            train_features[
                :,
                feature_id,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        (
            raw_auc,
            detection_auc,
            polarity,
        ) = get_train_polarity(
            values,
            train_labels_np,
        )

        direction = (
            get_sae_decoder_direction(
                sae,
                feature_id,
            )
        )

        direction_norm = float(
            torch.linalg.vector_norm(
                direction
            ).item()
        )

        if direction_norm == 0:
            continue

        # Match candidate residual-space L2 magnitude.
        steering_strength = (
            polarity
            * target_l2
            / direction_norm
        )

        steered = apply_direction_steering(
            residuals=test_residuals,
            direction=direction,
            steering_strength=(
                steering_strength
            ),
        )

        steered_scores = get_probe_scores(
            probe,
            steered,
        )

        mean_shift = float(
            np.mean(
                steered_scores
                - baseline_scores
            )
        )

        rows.append(
            {
                "feature_id": feature_id,
                "train_raw_auroc": raw_auc,
                "train_detection_auroc": (
                    detection_auc
                ),
                "train_polarity": polarity,
                "decoder_norm": direction_norm,
                "steering_strength": (
                    steering_strength
                ),
                "perturbation_l2": target_l2,
                "mean_probe_score_shift": (
                    mean_shift
                ),
                "abs_mean_probe_score_shift": (
                    abs(mean_shift)
                ),
            }
        )

        if i % 20 == 0:
            print(
                f"Processed "
                f"{i}/{N_RANDOM} "
                f"random controls..."
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        RANDOM_RESULTS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Candidate vs null distribution
    # ---------------------------------------------------------

    random_shifts = results[
        "mean_probe_score_shift"
    ].to_numpy()

    random_abs_shifts = np.abs(
        random_shifts
    )

    candidate_percentile = (
        100.0
        * np.mean(
            random_shifts
            <= candidate_shift
        )
    )

    signed_empirical_p = (
        1
        + np.sum(
            random_shifts
            >= candidate_shift
        )
    ) / (
        len(random_shifts)
        + 1
    )

    absolute_empirical_p = (
        1
        + np.sum(
            random_abs_shifts
            >= abs(candidate_shift)
        )
    ) / (
        len(random_shifts)
        + 1
    )

    summary = pd.DataFrame(
        [
            {
                "candidate_feature": (
                    CANDIDATE_FEATURE
                ),
                "candidate_train_polarity": (
                    candidate_polarity
                ),
                "candidate_probe_shift": (
                    candidate_shift
                ),
                "perturbation_l2": (
                    target_l2
                ),
                "n_random_controls": (
                    len(random_shifts)
                ),
                "random_mean_shift": float(
                    np.mean(
                        random_shifts
                    )
                ),
                "random_median_shift": float(
                    np.median(
                        random_shifts
                    )
                ),
                "random_std_shift": float(
                    np.std(
                        random_shifts
                    )
                ),
                "random_95th_percentile": float(
                    np.percentile(
                        random_shifts,
                        95,
                    )
                ),
                "candidate_percentile": (
                    candidate_percentile
                ),
                "signed_empirical_p": (
                    signed_empirical_p
                ),
                "absolute_empirical_p": (
                    absolute_empirical_p
                ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\nStage 5B complete."
    )

    print(
        f"Random controls evaluated: "
        f"{len(random_shifts)}"
    )

    print(
        f"Candidate shift: "
        f"{candidate_shift:.6f}"
    )

    print(
        f"Random mean shift: "
        f"{np.mean(random_shifts):.6f}"
    )

    print(
        f"Random 95th percentile: "
        f"{np.percentile(random_shifts, 95):.6f}"
    )

    print(
        f"Candidate percentile: "
        f"{candidate_percentile:.1f}%"
    )

    print(
        f"Signed empirical p: "
        f"{signed_empirical_p:.4f}"
    )

    print(
        f"Absolute empirical p: "
        f"{absolute_empirical_p:.4f}"
    )

    print(
        f"\nSaved: "
        f"{RANDOM_RESULTS_PATH}"
    )

    print(
        f"Saved: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()