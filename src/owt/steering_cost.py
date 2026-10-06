"""Steering cost: how much adding a feature direction disrupts GPT-2's
predictions on neutral text (KL divergence and CE loss change).

Per the Fixed decisions table, the direction is added "at all positions"
(unlike the BOS-exclusion used for fidelity/feature-health averages - this
is a deliberate difference, not an inconsistency: the intervention itself
applies everywhere, only the *measurement* of normal activation statistics
excludes BOS elsewhere in the pipeline).
"""

from __future__ import annotations

import torch

from src.owt.fidelity import mean_ce_loss


def mean_residual_norm(activations) -> float:
    """Mean L2 norm of the residual stream at a hook, over all tokens and
    positions in a batch. Used to scale a unit steering direction to the
    layer's natural activation magnitude."""
    return activations.norm(dim=-1).mean().item()


def unit_direction(direction) -> torch.Tensor:
    """Normalize a steering direction (e.g. an SAE decoder vector) to unit
    norm, so ``alpha`` alone controls steering strength."""
    norm = direction.norm()
    if norm == 0:
        raise ValueError("Cannot normalize a zero steering direction.")
    return direction / norm


def make_steering_hook(unit_dir: torch.Tensor, alpha: float, residual_norm: float):
    """Build a forward hook that adds ``alpha * unit_dir * residual_norm``
    to the residual stream at every position (Fixed decisions: "add
    alpha x unit direction x mean residual norm at that layer at all
    positions")."""

    def hook_fn(activation, hook):
        return activation + alpha * residual_norm * unit_dir

    return hook_fn


def steered_logits(model, tokens, hook_name: str, unit_dir: torch.Tensor, alpha: float, residual_norm: float):
    if alpha == 0.0:
        with torch.inference_mode():
            return model(tokens, return_type="logits")

    hook_fn = make_steering_hook(unit_dir, alpha, residual_norm)
    with torch.inference_mode():
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            return model(tokens, return_type="logits")


def kl_divergence(clean_logits_t, steered_logits_t) -> torch.Tensor:
    """Per-position KL(clean || steered) next-token-distribution
    divergence - how much the steered model's predictions move away from
    the clean model's, at every position that has a next-token target
    (same positions as CE loss; see fidelity.next_token_logits_and_targets).
    Returns a [batch, seq-1] tensor; callers take ``.mean()`` for a scalar.
    """
    import torch.nn.functional as F

    # Same position range as CE loss (fidelity.next_token_logits_and_targets):
    # every position has a next-token target except the last.
    clean_shifted = clean_logits_t[:, :-1]
    steered_shifted = steered_logits_t[:, :-1]

    clean_log_probs = F.log_softmax(clean_shifted.float(), dim=-1)
    steered_log_probs = F.log_softmax(steered_shifted.float(), dim=-1)
    clean_probs = clean_log_probs.exp()

    kl = (clean_probs * (clean_log_probs - steered_log_probs)).sum(dim=-1)
    return kl


def steering_cost_for_direction(
    model,
    tokens,
    hook_name: str,
    direction: torch.Tensor,
    alphas: list[float],
) -> list[dict]:
    """Run one steering-cost sweep (all alphas) for a single direction on
    a batch of neutral snippets. Returns one row per alpha with mean KL
    divergence and CE loss change relative to the clean (alpha=0) run.
    """
    unit_dir = unit_direction(direction)

    # One forward pass gives both the clean logits and the residual norm
    # (previously two separate passes per direction per batch).
    with torch.inference_mode():
        clean_logits_t, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    clean_loss, _ = mean_ce_loss(clean_logits_t, tokens)
    residual_norm = mean_residual_norm(cache[hook_name])

    rows = []
    for alpha in alphas:
        logits_t = steered_logits(model, tokens, hook_name, unit_dir, alpha=alpha, residual_norm=residual_norm)
        loss, _ = mean_ce_loss(logits_t, tokens)
        kl = kl_divergence(clean_logits_t, logits_t).mean().item()
        rows.append(
            {
                "alpha": alpha,
                "kl_divergence": kl,
                "ce_loss": loss,
                "ce_loss_change": loss - clean_loss,
            }
        )
    return rows