"""Stage 2: Per SAE type and layer - FVU, L0, cosine similarity on the
fidelity sample; spliced CE loss and loss recovered on the loss subset.

Loads one SAE type at a time (~2 GB for 12 layers), per the spec's
compute-saving note, rather than holding every type in memory together.
Start with ReLU only (the only type enabled in configs/owt.yaml until
TopK/JumpReLU release names are confirmed - see that file's TODOs).

Run: python scripts/owt_02_fidelity.py --config configs/owt.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config_path: str) -> None:
    import pandas as pd
    import torch

    from src.owt.data import load_config, load_token_array, set_all_seeds
    from src.owt.fidelity import (
        FidelityAccumulator,
        get_activations,
        loss_recovered,
        mean_ce_loss,
        spliced_sae_logits,
    )
    from src.owt.saes import enabled_sae_types, hook_name_for_layer, load_all_layer_saes, load_model

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    tokens = load_token_array(data_dir / "tokens" / "owt_20k.npy")
    n_fid = config["sample"]["fidelity_documents"]
    n_loss = config["sample"]["loss_documents"]
    fidelity_tokens = torch.from_numpy(tokens[:n_fid])
    loss_tokens = torch.from_numpy(tokens[:n_loss])

    baseline_path = results_dir / "tables" / "baseline_loss.csv"
    baseline_df = pd.read_csv(baseline_path)
    clean_loss = float(baseline_df["clean_loss"].iloc[0])
    zero_loss_by_layer = dict(zip(baseline_df["layer"], baseline_df["zero_ablation_loss"]))

    print("Loading GPT-2 Small...")
    model = load_model(config)
    batch_size = config["batch_size"]

    sae_types = enabled_sae_types(config)
    if not sae_types:
        raise RuntimeError(
            "No SAE types enabled in configs/owt.yaml. Enable at least 'relu' "
            "(release gpt2-small-res-jb) to run Stage 2."
        )

    output_path = results_dir / "tables" / "fidelity.csv"
    all_rows = []
    done = set()
    if output_path.exists():
        prev = pd.read_csv(output_path)
        all_rows = prev.to_dict("records")
        done = {(r["sae_type"], int(r["layer"])) for r in all_rows}
        print(f"Resuming: {len(done)} (sae_type, layer) pairs already in {output_path.name}")

    t0 = time.time()
    for sae_type in sae_types:
        print(f"\n=== SAE type: {sae_type} ===")
        layer_saes = load_all_layer_saes(config, sae_type, device=config["model"]["device"])

        for layer in config["layers"]:
            if (sae_type, layer) in done:
                print(f"Layer {layer:2d} | already done, skipping")
                continue
            sae = layer_saes[layer]
            hook_name = hook_name_for_layer(config, sae_type, layer)
            d_model = model.cfg.d_model

            # FVU / L0 / cosine on the fidelity sample.
            acc = FidelityAccumulator(d_model=d_model)
            for start in range(0, fidelity_tokens.shape[0], batch_size):
                batch = fidelity_tokens[start:start + batch_size]
                activations = get_activations(model, batch, hook_name)[:, 1:]  # exclude BOS
                flat = activations.reshape(-1, d_model)
                with torch.inference_mode():
                    features = sae.encode(flat)
                    reconstruction = sae.decode(features)
                acc.update(flat, reconstruction, features)
                if (start // batch_size) % 50 == 0:
                    print(f"  layer {layer} fidelity batch {start // batch_size + 1}/"
                          f"{-(-fidelity_tokens.shape[0] // batch_size)} "
                          f"({time.time() - t0:.0f}s elapsed)", flush=True)
            fidelity_metrics = acc.finalize()

            # Spliced CE loss on the loss subset.
            spliced_losses = []
            for start in range(0, loss_tokens.shape[0], batch_size):
                batch = loss_tokens[start:start + batch_size]
                logits = spliced_sae_logits(model, sae, batch, hook_name)
                loss, _ = mean_ce_loss(logits, batch)
                spliced_losses.append(loss)
                if (start // batch_size) % 25 == 0:
                    print(f"  layer {layer} spliced batch {start // batch_size + 1}/"
                          f"{-(-loss_tokens.shape[0] // batch_size)} "
                          f"({time.time() - t0:.0f}s elapsed)", flush=True)
            spliced_loss = sum(spliced_losses) / len(spliced_losses)

            zero_loss = float(zero_loss_by_layer[layer])
            recovered = loss_recovered(clean_loss, zero_loss, spliced_loss)

            print(
                f"Layer {layer:2d} | FVU: {fidelity_metrics['fvu']:.4f} | "
                f"L0: {fidelity_metrics['l0']:.1f} | cos: {fidelity_metrics['cosine']:.4f} | "
                f"loss recovered: {recovered:.1f}%",
                flush=True,
            )

            all_rows.append(
                {
                    "sae_type": sae_type,
                    "layer": layer,
                    "hook_name": hook_name,
                    "width": sae.cfg.d_sae,
                    "fvu": fidelity_metrics["fvu"],
                    "l0": fidelity_metrics["l0"],
                    "cosine": fidelity_metrics["cosine"],
                    "ce_clean": clean_loss,
                    "ce_spliced": spliced_loss,
                    "ce_zero_ablation": zero_loss,
                    "loss_recovered_pct": recovered,
                }
            )
            # Save after every layer so a crash never loses finished work.
            pd.DataFrame(all_rows).to_csv(output_path, index=False)

        del layer_saes  # free this SAE type's memory before loading the next

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)