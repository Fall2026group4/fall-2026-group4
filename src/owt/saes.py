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

    sae_id = sae_id_template.format(layer=layer)
    device = device or config["model"]["device"]
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
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
    return {layer: load_sae(config, sae_type, layer, device) for layer in config["layers"]}
