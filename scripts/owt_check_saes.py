"""Sanity check before trusting a new SAE type (run once for topk).

For each layer it prints: width, normalisation mode, mean L0 and whether the
reconstruction of a row changes when it is encoded alone vs inside a batch.
Some SAELens SAEs (e.g. OpenAI TopK with layer-norm input scaling) cache
statistics at encode time and use them in decode, so encode and decode must be
called back to back on the same batch. A large batch-dependence here means the
fidelity numbers for that type cannot be trusted.

Run: python scripts/owt_check_saes.py --only topk --layers 1 6 11
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(config_path, only, layers):
    import torch

    from src.owt.data import load_config, load_token_array
    from src.owt.fidelity import get_activations
    from src.owt.saes import (enabled_sae_types, hook_name_for_layer, layers_for_type,
                              load_model, load_sae, restrict_sae_types)

    config = load_config(REPO_ROOT / config_path)
    restrict_sae_types(config, only)
    tokens = torch.from_numpy(load_token_array(REPO_ROOT / config["paths"]["data_dir"] / "tokens" / "owt_20k.npy")[:16])
    model = load_model(config)
    for sae_type in enabled_sae_types(config):
        for layer in [l for l in layers_for_type(config, sae_type) if not layers or l in layers]:
            sae = load_sae(config, sae_type, layer)
            acts = get_activations(model, tokens, hook_name_for_layer(config, sae_type, layer))[:, 1:]
            flat = acts.reshape(-1, acts.shape[-1])[:512]
            with torch.inference_mode():
                full = sae.decode(sae.encode(flat))
                half = torch.cat([sae.decode(sae.encode(flat[i:i + 64])) for i in range(0, 512, 64)])
                l0 = (sae.encode(flat) > 0).float().sum(-1).mean().item()
            dep = (full - half).norm() / full.norm()
            print(f"{sae_type} L{layer}: width={sae.cfg.d_sae} norm={getattr(sae.cfg, 'normalize_activations', '?')} "
                  f"L0={l0:.1f} batch-dependence={dep:.2e}  {'OK' if dep < 1e-3 else 'CHECK: results depend on batch'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/owt.yaml")
    ap.add_argument("--only", nargs="+", default=["topk"])
    ap.add_argument("--layers", nargs="*", type=int, default=[1, 6, 11])
    a = ap.parse_args()
    main(a.config, a.only, a.layers)
