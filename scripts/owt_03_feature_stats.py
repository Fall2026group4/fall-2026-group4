"""Stage 3: Firing frequency of every latent, dead/dense counts, and the
top 10 activating snippets for the 20 most frequent latents per layer.

Uses the same fidelity sample as Stage 2 (so the two can be combined into
one script later if runtime matters - "Combine Stages 2 and 3 in one
script so feature frequencies are counted from the same batches"); kept
separate here to match the repo structure in the spec, at the cost of a
second forward pass over the same batches.

Run: python scripts/owt_03_feature_stats.py --config configs/owt.yaml
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

    from src.owt.data import load_config, load_token_array, set_all_seeds
    from src.owt.features import (
        FeatureFrequencyAccumulator,
        TopActivatingTracker,
        dead_and_dense_counts,
        top_frequent_feature_ids,
    )
    from src.owt.fidelity import get_activations
    from src.owt.saes import enabled_sae_types, hook_name_for_layer, load_all_layer_saes, load_model

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables" / "feature_freq").mkdir(parents=True, exist_ok=True)

    tokens = load_token_array(data_dir / "tokens" / "owt_20k.npy")
    n_fid = config["sample"]["fidelity_documents"]
    fidelity_tokens = torch.from_numpy(tokens[:n_fid])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Loading GPT-2 Small...")
    model = load_model(config)
    batch_size = config["batch_size"]
    dense_threshold = config["feature_health"]["dense_threshold"]

    sae_types = enabled_sae_types(config)
    summary_rows = []
    snippet_rows = []

    for sae_type in sae_types:
        print(f"\n=== SAE type: {sae_type} ===")
        layer_saes = load_all_layer_saes(config, sae_type, device=config["model"]["device"])

        for layer in config["layers"]:
            sae = layer_saes[layer]
            hook_name = hook_name_for_layer(config, sae_type, layer)

            # Pass 1: firing frequency.
            freq_acc = FeatureFrequencyAccumulator(d_sae=sae.cfg.d_sae)
            for start in range(0, fidelity_tokens.shape[0], batch_size):
                batch = fidelity_tokens[start:start + batch_size]
                activations = get_activations(model, batch, hook_name)[:, 1:]
                with torch.inference_mode():
                    features = sae.encode(activations.reshape(-1, activations.shape[-1]))
                freq_acc.update(features)
            frequency = freq_acc.finalize()

            dead, dense = dead_and_dense_counts(frequency, dense_threshold=dense_threshold)
            top_ids = top_frequent_feature_ids(frequency, n=20)

            print(f"Layer {layer:2d} | dead: {dead} | dense (>{dense_threshold:.0%}): {dense} "
                  f"| top feature: {top_ids[0]} (freq {frequency[top_ids[0]]:.4f})")

            freq_path = results_dir / "tables" / "feature_freq" / f"{sae_type}_L{layer}.npy"
            np.save(freq_path, frequency.numpy())

            summary_rows.append(
                {
                    "sae_type": sae_type,
                    "layer": layer,
                    "d_sae": sae.cfg.d_sae,
                    "dead_latents": dead,
                    "dense_latents": dense,
                    "dense_threshold": dense_threshold,
                    "mean_frequency": frequency.mean().item(),
                }
            )

            # Pass 2: top-10 activating snippets for the 20 most frequent latents.
            trackers = {fid: TopActivatingTracker(k=10) for fid in top_ids}
            context_length = fidelity_tokens.shape[1]
            for start in range(0, fidelity_tokens.shape[0], batch_size):
                batch = fidelity_tokens[start:start + batch_size]
                activations = get_activations(model, batch, hook_name)[:, 1:]  # exclude BOS
                with torch.inference_mode():
                    features = sae.encode(activations)  # [batch, seq-1, d_sae]

                doc_ids = list(range(start, start + batch.shape[0]))
                positions = list(range(1, context_length))  # position in the original (BOS-included) sequence

                for fid in top_ids:
                    feature_acts = features[:, :, fid]  # [batch, seq-1]
                    flat_acts = feature_acts.reshape(-1)
                    flat_doc_ids = [d for d in doc_ids for _ in positions]
                    flat_positions = positions * len(doc_ids)
                    trackers[fid].update(flat_acts, flat_doc_ids, flat_positions)

            for fid, tracker in trackers.items():
                for activation, doc_id, position in tracker.top():
                    token_ids = tokens[doc_id].tolist()
                    snippet = tokenizer.decode(token_ids[max(1, position - 10):position + 1])
                    snippet_rows.append(
                        {
                            "sae_type": sae_type,
                            "layer": layer,
                            "feature_id": fid,
                            "frequency": frequency[fid].item(),
                            "doc_id": doc_id,
                            "position": position,
                            "activation": activation,
                            "snippet": snippet,
                        }
                    )

        del layer_saes

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "tables" / "feature_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    snippets_df = pd.DataFrame(snippet_rows)
    snippets_path = results_dir / "tables" / "feature_top_snippets.csv"
    snippets_df.to_csv(snippets_path, index=False)
    print(f"Saved: {snippets_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)
