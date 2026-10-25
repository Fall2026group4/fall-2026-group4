"""Stage 6: Generate publication-ready Geometry of Truth figures."""

from pathlib import Path

from src.evaluation.geometry_figures import (
    generate_all_geometry_figures,
)


RESULTS_DIR = Path(
    "results/geometry_of_truth"
)

FIGURE_DIR = Path(
    "results/figures"
)


def main():
    print(
        "Stage 6 — Generating Geometry of Truth figures..."
    )

    generate_all_geometry_figures(
        results_dir=RESULTS_DIR,
        figure_dir=FIGURE_DIR,
    )

    expected_files = [
        "geometry_baseline_probe_performance.svg",
        "geometry_sae_faithfulness.svg",
        "geometry_steering_dose_response.svg",
        "geometry_random_control_null.svg",
    ]

    print("\nGenerated figures:")

    for filename in expected_files:
        path = FIGURE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Expected figure not created: {path}"
            )

        print(
            f"  {path}"
        )

    print(
        "\nStage 6 figure generation complete."
    )


if __name__ == "__main__":
    main()