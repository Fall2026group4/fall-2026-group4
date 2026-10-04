"""Stage 5: Repeat Stages 1-2 (baseline + zero-ablation loss, then
fidelity/spliced loss/loss recovered) on 2,000 WikiText-103 test
snippets - a corpus the res-jb SAEs were NOT trained on, so this shows how
much of the OpenWebText fidelity numbers hold out of distribution.

Run: python scripts/owt_05_wikitext_check.py --config configs/owt.yaml
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

    from src.owt.data import load_config, set_all_seeds, stream_and_tokenize
    from src.owt.fidelity import (
        FidelityAccumulator,
        get_activations,
        loss_recovered,
        mean_ce_loss,
        spliced_sae_logits,
        zero_ablation_logits,
    )
    from src.owt.saes import enabled_sae_types, hook_name_for_layer, load_all_layer_saes, load_model

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Streaming {config['sample']['wikitext_snippets']} WikiText-103 test snippets...")
    wikitext_tokens_np = stream_and_tokenize(
        dataset_name=config["corpus"]["wikitext_dataset"],
        dataset_config=config["corpus"]["wikitext_config"],
        split=config["corpus"]["wikitext_split"],
        tokenizer=tokenizer,
        n_documents=config["sample"]["wikitext_snippets"],
        context_length=config["sample"]["context_length"],
        shuffle_buffer_size=config["sample"]["shuffle_buffer_size"],
        seed=config["seed"],
    )
    wikitext_tokens = torch.from_numpy(wikitext_tokens_np)
    print(f"WikiText tokens: {wikitext_tokens.shape}")

    print("Loading GPT-2 Small...")
    model = load_model(config)
    batch_size = config["batch_size"]

    # Baseline + zero-ablation on WikiText.
    clean_losses = []
    for start in range(0, wikitext_tokens.shape[0], batch_size):
        batch = wikitext_tokens[start:start + batch_size]
        with torch.inference_mode():
            logits = model(batch, return_type="logits")
        loss, _ = mean_ce_loss(logits, batch)
        clean_losses.append(loss)
    clean_loss = sum(clean_losses) / len(clean_losses)
    print(f"WikiText clean CE loss: {clean_loss:.4f}")

    zero_loss_by_layer = {}
    for layer in config["layers"]:
        hook_name = f"blocks.{layer}.hook_resid_pre"
        layer_losses = []
        for start in range(0, wikitext_tokens.shape[0], batch_size):
            batch = wikitext_tokens[start:start + batch_size]
            logits = zero_ablation_logits(model, batch, hook_name)
            loss, _ = mean_ce_loss(logits, batch)
            layer_losses.append(loss)
        zero_loss_by_layer[layer] = sum(layer_losses) / len(layer_losses)

    sae_types = enabled_sae_types(config)
    all_rows = []
    for sae_type in sae_types:
        print(f"\n=== SAE type: {sae_type} ===")
        layer_saes = load_all_layer_saes(config, sae_type, device=config["model"]["device"])

        for layer in config["layers"]:
            sae = layer_saes[layer]
            hook_name = hook_name_for_layer(config, sae_type, layer)
            d_model = model.cfg.d_model

            acc = FidelityAccumulator(d_model=d_model)
            for start in range(0, wikitext_tokens.shape[0], batch_size):
                batch = wikitext_tokens[start:start + batch_size]
                activations = get_activations(model, batch, hook_name)[:, 1:]
                flat = activations.reshape(-1, d_model)
                with torch.inference_mode():
                    features = sae.encode(flat)
                    reconstruction = sae.decode(features)
                acc.update(flat, reconstruction, features)
            fidelity_metrics = acc.finalize()

            spliced_losses = []
            for start in range(0, wikitext_tokens.shape[0], batch_size):
                batch = wikitext_tokens[start:start + batch_size]
                logits = spliced_sae_logits(model, sae, batch, hook_name)
                loss, _ = mean_ce_loss(logits, batch)
                spliced_losses.append(loss)
            spliced_loss = sum(spliced_losses) / len(spliced_losses)

            zero_loss = zero_loss_by_layer[layer]
            recovered = loss_recovered(clean_loss, zero_loss, spliced_loss)

            print(f"Layer {layer:2d} | FVU: {fidelity_metrics['fvu']:.4f} | "
                  f"loss recovered: {recovered:.1f}%")

            all_rows.append(
                {
                    "sae_type": sae_type,
                    "layer": layer,
                    "hook_name": hook_name,
                    "fvu": fidelity_metrics["fvu"],
                    "l0": fidelity_metrics["l0"],
                    "cosine": fidelity_metrics["cosine"],
                    "ce_clean": clean_loss,
                    "ce_spliced": spliced_loss,
                    "ce_zero_ablation": zero_loss,
                    "loss_recovered_pct": recovered,
                }
            )

        del layer_saes

    df = pd.DataFrame(all_rows)
    output_path = results_dir / "tables" / "fidelity_wikitext.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    owt_fidelity_path = results_dir / "tables" / "fidelity.csv"
    if owt_fidelity_path.exists():
        owt_df = pd.read_csv(owt_fidelity_path)
        merged = df.merge(
            owt_df[["sae_type", "layer", "loss_recovered_pct"]],
            on=["sae_type", "layer"],
            suffixes=("_wikitext", "_owt"),
        )
        merged["loss_recovered_gap"] = (
            merged["loss_recovered_pct_owt"] - merged["loss_recovered_pct_wikitext"]
        )
        print("\nLoss recovered, OWT vs WikiText (positive gap = worse out of distribution):")
        print(
            merged[["sae_type", "layer", "loss_recovered_pct_owt", "loss_recovered_pct_wikitext", "loss_recovered_gap"]]
            .to_string(index=False)
        )
    else:
        print("\n(Run owt_02_fidelity.py first to compare against OWT loss recovered.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)
