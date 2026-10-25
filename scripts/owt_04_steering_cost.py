"""Stage 4: On the 200 neutral snippets, steer with 10 random latent
decoder directions per SAE type/layer (plus any shared truth/honesty
directions in config), at alpha in {0.5, 1, 2, 4}. Records KL divergence
and CE loss change - the side-effect table the other two project parts
(Geometry of Truth, RepE) cite.

Run: python scripts/owt_04_steering_cost.py --config configs/owt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config_path: str) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from src.owt.data import load_config, load_neutral_snippets, set_all_seeds
    from src.owt.saes import enabled_sae_types, hook_name_for_layer, load_all_layer_saes, load_model
    from src.owt.steering_cost import steering_cost_for_direction

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    snippets = load_neutral_snippets(data_dir / "neutral_200.jsonl")
    n_snippets = config["steering"].get("n_snippets")  # optional: use fewer than all 200 for a quick pass
    if n_snippets:
        snippets = snippets[:n_snippets]
    tokens = torch.tensor([s["token_ids"] for s in snippets], dtype=torch.long)
    print(f"Neutral snippets: {tokens.shape}")

    print("Loading GPT-2 Small...")
    model = load_model(config)

    alphas = config["steering"]["alphas"]
    n_random = config["steering"]["n_random_directions"]
    rng = np.random.default_rng(config["seed"])

    extra_directions = []
    for entry in config["steering"].get("extra_directions", []):
        direction = torch.load(REPO_ROOT / entry["path"], map_location="cpu")
        extra_directions.append((entry["name"], direction))
    if not extra_directions:
        print("No shared truth/honesty directions configured yet "
              "(configs/owt.yaml steering.extra_directions is empty) - "
              "random-direction sweep only. Add them once Simba/Aditi share theirs.")

    sae_types = enabled_sae_types(config)
    all_rows = []

    for sae_type in sae_types:
        print(f"\n=== SAE type: {sae_type} ===")
        layer_saes = load_all_layer_saes(config, sae_type, device=config["model"]["device"])

        for layer in config["layers"]:
            sae = layer_saes[layer]
            hook_name = hook_name_for_layer(config, sae_type, layer)

            random_feature_ids = rng.choice(sae.cfg.d_sae, size=n_random, replace=False)

            directions = [
                (f"random_feature_{fid}", sae.W_dec[fid].detach().clone())
                for fid in random_feature_ids
            ]
            directions += extra_directions

            for direction_name, direction in directions:
                for start in range(0, tokens.shape[0], config["batch_size"]):
                    batch = tokens[start:start + config["batch_size"]]
                    rows = steering_cost_for_direction(
                        model, batch, hook_name, direction, alphas
                    )
                    for row in rows:
                        row.update(
                            {
                                "sae_type": sae_type,
                                "layer": layer,
                                "direction": direction_name,
                                "batch_start": start,
                                "batch_size": batch.shape[0],
                            }
                        )
                        all_rows.append(row)

            print(f"Layer {layer:2d} | {len(directions)} directions x {len(alphas)} alphas done")

        del layer_saes

    df = pd.DataFrame(all_rows)
    # Average over the per-batch rows so the saved table is one row per
    # (sae_type, layer, direction, alpha), weighted by batch size.
    df["weighted_kl"] = df["kl_divergence"] * df["batch_size"]
    df["weighted_ce"] = df["ce_loss_change"] * df["batch_size"]
    grouped = (
        df.groupby(["sae_type", "layer", "direction", "alpha"], as_index=False)
        .agg(
            n_snippets=("batch_size", "sum"),
            kl_divergence=("weighted_kl", "sum"),
            ce_loss_change=("weighted_ce", "sum"),
        )
    )
    grouped["kl_divergence"] /= grouped["n_snippets"]
    grouped["ce_loss_change"] /= grouped["n_snippets"]

    output_path = results_dir / "tables" / "steering_cost.csv"
    grouped.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)