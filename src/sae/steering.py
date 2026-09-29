"""Causal steering interventions for a single SAE feature (Phase 4).

Fixes two problems flagged in the instructor review of the original
notebook implementation:

1. The steering hook used to mutate a global ``steering_strength`` variable
   captured by closure, which was fragile and made two competing, partially
   duplicated sweeps end up in the notebook. Here the strength is an
   explicit argument.
2. No seeding / fixed sampling parameters: steering effects were compared
   across generations without pinning a seed, so an observed difference
   could be sampling noise rather than a real steering effect. ``run_steering_sweep``
   now seeds torch before every generation call.
"""

from __future__ import annotations

from typing import Iterable

DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_SEED = 0


def make_steering_hook(steering_direction, steering_strength: float):
    """Build a forward hook that pushes the last-token residual stream
    along ``steering_direction`` by ``steering_strength``.

    ``steering_direction`` is typically ``sae.W_dec[feature_id]`` — the
    SAE decoder direction for the feature being steered.
    """

    def hook_fn(activation, hook):
        activation[:, -1, :] += steering_strength * steering_direction
        return activation

    return hook_fn


def generate_with_steering(
    sae_model,
    prompt: str,
    hook_name: str,
    steering_direction=None,
    steering_strength: float = 0.0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = DEFAULT_SEED,
) -> str:
    """Generate from ``prompt``, optionally steering along a feature direction.

    Pass ``steering_strength=0`` (or ``steering_direction=None``) for an
    unsteered baseline generation using the exact same code path as a
    steered one, so baseline and steered outputs are only ever different
    because of the intervention, not because of different call sites.
    """
    import torch

    torch.manual_seed(seed)

    if steering_direction is None or steering_strength == 0.0:
        return sae_model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)

    hook_fn = make_steering_hook(steering_direction, steering_strength)
    with sae_model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        return sae_model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)


def run_steering_sweep(
    sae_model,
    prompt: str,
    hook_name: str,
    steering_direction,
    strengths: Iterable[float],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run one baseline (strength 0) plus one generation per strength in
    ``strengths``. Returns ``{strength: output_text}`` including ``0``.

    A single sweep function replaces the two separate, partially-broken
    sweeps that had accumulated in the notebook (see Instructor Review #2,
    "Blockers" -> commit attribution / duplicated steering code finding).
    """
    outputs = {
        0: generate_with_steering(
            sae_model, prompt, hook_name, max_new_tokens=max_new_tokens, seed=seed
        )
    }
    for strength in strengths:
        outputs[strength] = generate_with_steering(
            sae_model,
            prompt,
            hook_name,
            steering_direction=steering_direction,
            steering_strength=strength,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    return outputs
