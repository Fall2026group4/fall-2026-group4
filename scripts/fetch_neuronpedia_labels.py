"""Fetch Neuronpedia auto-interpretation labels for sampled Geometry features."""

from pathlib import Path
import time

import pandas as pd
import requests


MODEL_ID = "gpt2-small"
SAE_LAYER = "10-res-jb"

INPUT_PATH = Path(
    "results/geometry_of_truth/feature_interpretation_sample.csv"
)

OUTPUT_PATH = Path(
    "results/geometry_of_truth/feature_interpretation_with_labels.csv"
)

TIMEOUT = 30


def fetch_feature_metadata(session, feature_id):
    """Fetch one Neuronpedia feature record."""

    url = (
        f"https://www.neuronpedia.org/api/feature/"
        f"{MODEL_ID}/{SAE_LAYER}/{feature_id}"
    )

    response = session.get(
        url,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    explanations = data.get(
        "explanations",
        []
    )

    if explanations:
        first = explanations[0]

        description = first.get(
            "description"
        )

        explanation_model = first.get(
            "explanationModelName"
        )

        explanation_type = first.get(
            "typeName"
        )
    else:
        description = None
        explanation_model = None
        explanation_type = None

    return {
        "neuronpedia_label": description,
        "explanation_model": explanation_model,
        "explanation_type": explanation_type,
        "n_explanations": len(explanations),
        "neuronpedia_url": (
            f"https://www.neuronpedia.org/"
            f"{MODEL_ID}/{SAE_LAYER}/{feature_id}"
        ),
    }


def main():
    sample = pd.read_csv(
        INPUT_PATH
    )

    print(
        "Fetching Neuronpedia labels "
        f"for {len(sample)} features..."
    )

    session = requests.Session()

    rows = []

    for i, row in sample.iterrows():
        feature_id = int(
            row["feature_id"]
        )

        try:
            metadata = fetch_feature_metadata(
                session,
                feature_id,
            )

            status = (
                metadata["neuronpedia_label"]
                if metadata["neuronpedia_label"]
                else "NO EXPLANATION"
            )

            print(
                f"{i + 1:02d}/{len(sample)} "
                f"Feature {feature_id}: {status}"
            )

        except Exception as exc:
            print(
                f"{i + 1:02d}/{len(sample)} "
                f"Feature {feature_id}: ERROR — {exc}"
            )

            metadata = {
                "neuronpedia_label": None,
                "explanation_model": None,
                "explanation_type": None,
                "n_explanations": 0,
                "neuronpedia_url": (
                    f"https://www.neuronpedia.org/"
                    f"{MODEL_ID}/{SAE_LAYER}/{feature_id}"
                ),
            }

        combined = row.to_dict()
        combined.update(
            metadata
        )

        rows.append(
            combined
        )

        # Be polite to the public API.
        time.sleep(
            0.15
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    labelled = output[
        "neuronpedia_label"
    ].notna().sum()

    print(
        "\nNeuronpedia label retrieval complete."
    )

    print(
        f"Features with explanations: "
        f"{labelled}/{len(output)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()