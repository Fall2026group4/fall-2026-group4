"""Stage 1: Baseline CE loss (no SAE) and per-layer zero-ablation loss on
the loss subset. Needed as the reference points for Stage 2's loss
recovered.

Run: python scripts/owt_01_baseline_loss.py --config configs/owt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config_path: str) -> None:
    import pandas as pd
    import torch

    from src.owt.data import load_config, load_token_array, set_all_seeds
    from src.owt.fidelity import mean_ce_loss, zero_ablation_logits
    from src.owt.saes import load_model

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    tokens = load_token_array(data_dir / "tokens" / "owt_20k.npy")
    n_loss = config["sample"]["loss_documents"]
    loss_tokens = torch.from_numpy(tokens[:n_loss])
    print(f"Loss subset: {loss_tokens.shape}")

    print("Loading GPT-2 Small...")
    model = load_model(config)

    batch_size = config["batch_size"]
    rows = []

    clean_losses = []
    for start in range(0, loss_tokens.shape[0], batch_size):
        batch = loss_tokens[start:start + batch_size]
        with torch.inference_mode():
            logits = model(batch, return_type="logits")
        loss, _ = mean_ce_loss(logits, batch)
        clean_losses.append(loss)
    clean_loss = sum(clean_losses) / len(clean_losses)
    print(f"Clean CE loss: {clean_loss:.4f}")

    for layer in config["layers"]:
        hook_name = f"blocks.{layer}.hook_resid_pre"
        layer_losses = []
        for start in range(0, loss_tokens.shape[0], batch_size):
            batch = loss_tokens[start:start + batch_size]
            logits = zero_ablation_logits(model, batch, hook_name)
            loss, _ = mean_ce_loss(logits, batch)
            layer_losses.append(loss)
        zero_loss = sum(layer_losses) / len(layer_losses)
        print(f"Layer {layer:2d} | zero-ablation CE loss: {zero_loss:.4f}")
        rows.append({"layer": layer, "hook_name": hook_name, "zero_ablation_loss": zero_loss})

    df = pd.DataFrame(rows)
    df["clean_loss"] = clean_loss
    output_path = results_dir / "tables" / "baseline_loss.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)
