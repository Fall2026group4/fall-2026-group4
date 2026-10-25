"""Load GPT-2 Small and each pretrained SAE type, with consistent hook
points across SAE types (Fixed decisions: a resid_post SAE at layer L is
filed as resid_pre L+1, so every SAE type lines up by position).

Reuses ``src/models/gpt2_activations.py::load_gpt2_small`` for the model
instead of reimplementing TransformerLens loading a third time in this
project (see Geometry of Truth's ``src/sae/geometry_sae.py`` for the same
pattern).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_model",
    "enabled_sae_types",
    "hook_name_for_layer",
    "layers_for_type",
    "restrict_sae_types",
    "save_merged",
    "sae_id_for_layer",
    "normalize_hook_point",
    "load_sae",
    "load_all_layer_saes",
]


def normalize_hook_point(point_type: str, layer: int) -> tuple[int, str]:
    """Map a (point_type, layer) pair to the canonical (effective_layer,
    hook_name) expressed as ``hook_resid_pre``, so every SAE type lines up
    by position regardless of which point type its release was trained on
    (Fixed decisions: "a resid_post SAE at layer L is filed as resid_pre
    L+1").

    >>> normalize_hook_point("resid_pre", 5)
    (5, 'blocks.5.hook_resid_pre')
    >>> normalize_hook_point("resid_post", 5)
    (6, 'blocks.6.hook_resid_pre')
    """
    if point_type == "resid_pre":
        effective_layer = layer
    elif point_type == "resid_post":
        effective_layer = layer + 1
    else:
        raise ValueError(f"Unknown hook point type: {point_type!r}")
    return effective_layer, f"blocks.{effective_layer}.hook_resid_pre"


def load_model(config: dict[str, Any]):
    """Load GPT-2 Small per the Fixed Decisions table (TransformerLens,
    float32, device from config).

    Imports transformer_lens lazily (inside the function, not at module
    top level) so this module - and anything pure-logic that imports from
    it, like ``normalize_hook_point`` - stays importable without
    transformer_lens installed. Same pattern as
    ``src/models/gpt2_sae.py::load_sae_bundle``.
    """
    from src.models.gpt2_activations import load_gpt2_small

    device = config["model"]["device"]
    model = load_gpt2_small(device=device)
    return model


def enabled_sae_types(config: dict[str, Any]) -> list[str]:
    """SAE types with a confirmed release (see configs/owt.yaml - TopK and
    JumpReLU are disabled until their release names are confirmed, per the
    spec's "Start with ReLU only" guidance)."""
    return [name for name, cfg in config["sae_types"].items() if cfg.get("enabled")]


def layers_for_type(config: dict[str, Any], sae_type: str) -> list[int]:
    """Positions (hook_resid_pre layers) an SAE type covers.

    Defaults to ``config["layers"]``. A type can narrow it with its own
    ``layers`` list: the TopK release is trained on resid_post, so it has
    no SAE for position 0 (the embeddings) and covers positions 1-11 here.
    """
    return list(config["sae_types"][sae_type].get("layers", config["layers"]))


def sae_id_for_layer(config: dict[str, Any], sae_type: str, layer: int) -> str:
    """Release-specific SAE id for a position. ``{layer}`` is the position
    (resid_pre index); ``{source_layer}`` is the block the SAE was trained
    on, which is ``layer - 1`` for a resid_post release (position L is the
    output of block L - 1)."""
    type_cfg = config["sae_types"][sae_type]
    point_type = type_cfg.get("point_type", "resid_pre")
    source_layer = layer - 1 if point_type == "resid_post" else layer
    return type_cfg["sae_id_template"].format(layer=layer, source_layer=source_layer)


def hook_name_for_layer(config: dict[str, Any], sae_type: str, layer: int) -> str:
    """The GPT-2 hook point a given SAE type/layer reads from."""
    template = config["sae_types"][sae_type]["hook_template"]
    return template.format(layer=layer)


def load_sae(config: dict[str, Any], sae_type: str, layer: int, device: str | None = None):
    """Load one SAE (one type, one layer)."""
    from sae_lens import SAE

    type_cfg = config["sae_types"][sae_type]
    if not type_cfg.get("enabled"):
        raise ValueError(
            f"SAE type '{sae_type}' is not enabled in configs/owt.yaml "
            "(release not confirmed yet - see the TODO comments there)."
        )

    release = type_cfg["release"]
    sae_id_template = type_cfg["sae_id_template"]
    if release is None or sae_id_template is None:
        raise ValueError(
            f"SAE type '{sae_type}' has no release/sae_id_template configured."
        )

    sae_id = sae_id_for_layer(config, sae_type, layer)
    device = device or config["model"]["device"]
    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out  # older SAELens returns a tuple
    sae.eval()
    return sae


def load_all_layer_saes(
    config: dict[str, Any], sae_type: str, device: str | None = None
) -> dict[int, Any]:
    """Load one SAE per layer for a given type, keyed by layer index.

    Per the spec's compute-saving note: "load one SAE type at a time
    (about 2 GB for 12 layers)" - callers should load one type, run every
    layer, discard, then move to the next type rather than holding
    multiple types in memory at once.
    """
    return {layer: load_sae(config, sae_type, layer, device) for layer in layers_for_type(config, sae_type)}


def restrict_sae_types(config: dict[str, Any], only: list[str] | None) -> None:
    """Disable every SAE type not in ``only`` (CLI ``--only topk``), so a new
    type can be run without redoing the others."""
    if not only:
        return
    unknown = [t for t in only if t not in config["sae_types"]]
    if unknown:
        raise ValueError(f"Unknown SAE types in --only: {unknown}")
    for name, cfg in config["sae_types"].items():
        cfg["enabled"] = name in only


def save_merged(df, path, key_cols=("sae_type",)) -> None:
    """Write ``df`` to ``path`` without losing rows of SAE types that were not
    part of this run: existing rows whose key matches a key in ``df`` are
    replaced, all other existing rows are kept."""
    import pandas as pd
    from pathlib import Path

    path = Path(path)
    if path.exists() and len(df):
        old = pd.read_csv(path)
        key_cols = list(key_cols)
        new_keys = set(map(tuple, df[key_cols].astype(str).values))
        keep = ~old[key_cols].astype(str).apply(tuple, axis=1).isin(new_keys)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(path, index=False)
