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
import time
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

    layers = config["layers"]
    n_batches = -(-fidelity_tokens.shape[0] // batch_size)
    context_length = fidelity_tokens.shape[1]
    positions = torch.arange(1, context_length)  # positions in the original (BOS-included) sequence
    t0 = time.time()

    def all_layer_activations(batch, hook_names):
        """ONE GPT-2 forward pass that returns the residual stream at every
        layer's hook point (stop_at_layer skips the final norm/unembed).
        Previously every layer re-ran the whole model twice."""
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                batch, names_filter=list(hook_names.values()), stop_at_layer=max(layers) + 1
            )
        return {layer: cache[name][:, 1:] for layer, name in hook_names.items()}  # exclude BOS

    for sae_type in sae_types:
        print(f"\n=== SAE type: {sae_type} ===")
        layer_saes = load_all_layer_saes(config, sae_type, device=config["model"]["device"])
        hook_names = {layer: hook_name_for_layer(config, sae_type, layer) for layer in layers}

        # Pass 1: firing frequency, all layers from the same forward pass.
        freq_accs = {layer: FeatureFrequencyAccumulator(d_sae=layer_saes[layer].cfg.d_sae) for layer in layers}
        for b, start in enumerate(range(0, fidelity_tokens.shape[0], batch_size)):
            batch = fidelity_tokens[start:start + batch_size]
            acts = all_layer_activations(batch, hook_names)
            for layer in layers:
                with torch.inference_mode():
                    features = layer_saes[layer].encode(acts[layer].reshape(-1, acts[layer].shape[-1]))
                freq_accs[layer].update(features)
            if b % 10 == 0:
                print(f"  pass 1 batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s elapsed)", flush=True)

        frequency = {layer: freq_accs[layer].finalize() for layer in layers}
        top_ids = {layer: top_frequent_feature_ids(frequency[layer], n=20) for layer in layers}
        trackers = {layer: {fid: TopActivatingTracker(k=10) for fid in top_ids[layer]} for layer in layers}

        for layer in layers:
            sae = layer_saes[layer]
            dead, dense = dead_and_dense_counts(frequency[layer], dense_threshold=dense_threshold)
            first = top_ids[layer][0]
            print(f"Layer {layer:2d} | dead: {dead} | dense (>{dense_threshold:.0%}): {dense} "
                  f"| top feature: {first} (freq {frequency[layer][first]:.4f})", flush=True)

            freq_path = results_dir / "tables" / "feature_freq" / f"{sae_type}_L{layer}.npy"
            np.save(freq_path, frequency[layer].numpy())

            summary_rows.append(
                {
                    "sae_type": sae_type,
                    "layer": layer,
                    "d_sae": sae.cfg.d_sae,
                    "dead_latents": dead,
                    "dense_latents": dense,
                    "dense_threshold": dense_threshold,
                    "mean_frequency": frequency[layer].mean().item(),
                }
            )

        # Pass 2: top-10 activating snippets for the 20 most frequent latents,
        # again all layers per forward pass. Only each feature's own top-10 per
        # batch is handed to the tracker (same result, far less Python work).
        n_pos = positions.shape[0]
        for b, start in enumerate(range(0, fidelity_tokens.shape[0], batch_size)):
            batch = fidelity_tokens[start:start + batch_size]
            acts = all_layer_activations(batch, hook_names)
            for layer in layers:
                with torch.inference_mode():
                    features = layer_saes[layer].encode(acts[layer])  # [batch, seq-1, d_sae]
                for fid in top_ids[layer]:
                    flat_acts = features[:, :, fid].reshape(-1)
                    k = min(10, flat_acts.shape[0])
                    vals, idx = torch.topk(flat_acts, k=k)
                    doc_ids = (start + torch.div(idx, n_pos, rounding_mode="floor")).tolist()
                    pos_list = positions[idx % n_pos].tolist()
                    trackers[layer][fid].update(vals, doc_ids, pos_list)
            if b % 10 == 0:
                print(f"  pass 2 batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s elapsed)", flush=True)

        for layer in layers:
            for fid, tracker in trackers[layer].items():
                for activation, doc_id, position in tracker.top():
                    token_ids = tokens[doc_id].tolist()
                    snippet = tokenizer.decode(token_ids[max(1, position - 10):position + 1])
                    snippet_rows.append(
                        {
                            "sae_type": sae_type,
                            "layer": layer,
                            "feature_id": fid,
                            "frequency": frequency[layer][fid].item(),
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