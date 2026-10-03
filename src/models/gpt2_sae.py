"""Model and SAE loading helpers for the SAE-Faithful project.

This module centralizes everything that talks to TransformerLens / SAELens so
notebooks under ``cookbooks/`` can stay narrative-only: they import from here
instead of re-implementing model/SAE loading in notebook cells.

Currently targets GPT-2 Small with the ``gpt2-small-res-jb`` pretrained SAE
release (block 8, ``hook_resid_pre``), matching Phase 1/2 of the project
proposal. Extending to Gemma-2-2B + Gemma Scope (Phase 6 cross-model
comparison) should follow the same pattern: add a second set of loader
functions here rather than branching inside the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"
DEFAULT_SAE_ID = "blocks.8.hook_resid_pre"
DEFAULT_HOOK_NAME = "blocks.8.hook_resid_pre"


@dataclass
class SAEBundle:
    """A loaded model + SAE pair, plus the hook point they were paired at."""

    sae_model: object
    sae: object
    hook_name: str = DEFAULT_HOOK_NAME


def load_sae_bundle(
    model_name: str = DEFAULT_MODEL_NAME,
    sae_release: str = DEFAULT_SAE_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    hook_name: str = DEFAULT_HOOK_NAME,
    device: str = "cpu",
) -> SAEBundle:
    """Load a HookedSAETransformer and a matching pretrained SAE.

    Kept as a single entry point so every notebook/script loads the model and
    SAE the same way, with the same hook point, instead of each copying
    slightly different loading code (the drift that caused the Review #1
    finding that "everything is in two notebooks").
    """
    from sae_lens import SAE, HookedSAETransformer

    sae_model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
    )
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    return SAEBundle(sae_model=sae_model, sae=sae, hook_name=hook_name)


def get_feature_activation(bundle: SAEBundle, text: str, feature_id: int) -> float:
    """Return the max activation of ``feature_id`` over a text's tokens.

    Skips position 0 (the BOS/``<|endoftext|>`` token), matching the original
    exploratory notebook's behavior, since BOS activations are not
    meaningful token-level signal for this feature-detection use case.
    """
    tokens = bundle.sae_model.to_tokens(text)
    _, cache = bundle.sae_model.run_with_cache(tokens)
    layer_activations = cache[bundle.hook_name]
    features = bundle.sae.encode(layer_activations)
    return features[0, 1:, feature_id].max().item()
