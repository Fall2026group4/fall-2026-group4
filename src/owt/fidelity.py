"""Fidelity metrics: FVU, L0, cosine similarity, spliced CE loss, loss
recovered.

Everything here is batch-accumulated (never holds the full activation
tensor for the whole sample in memory - see "Disk discipline matters most
for this part" in the project spec) and excludes the BOS token position
from activation-based statistics (Fixed decisions: "Token positions: all
except BOS - BOS activations are huge outliers that distort averages").
CE loss is the standard next-token loss, which already only scores real
(non-BOS) targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def next_token_logits_and_targets(logits, tokens):
    """Shift logits/tokens for next-token prediction.

    ``logits``: [batch, seq, vocab]. ``tokens``: [batch, seq].
    Returns ``(logits[:, :-1], targets)`` where ``targets = tokens[:, 1:]``.
    Position 0's logits (predicted from BOS alone) are included - this is
    a normal next-token prediction, not a BOS *activation* average.
    """
    return logits[:, :-1], tokens[:, 1:]


def mean_ce_loss(logits, tokens) -> tuple[float, "torch.Tensor"]:
    """Mean next-token cross-entropy loss, plus the per-position loss
    tensor (``[batch, seq-1]``) for callers that need it (e.g. steering
    cost's CE-loss-change metric)."""
    import torch.nn.functional as F

    shifted_logits, targets = next_token_logits_and_targets(logits, tokens)
    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_position_loss = -token_log_probs
    return per_position_loss.mean().item(), per_position_loss


def get_activations(model, tokens, hook_name):
    """Run GPT-2 with cache and return the residual stream at ``hook_name``."""
    import torch

    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    return cache[hook_name]


def run_with_replacement(model, tokens, hook_name, replacement_fn):
    """Run GPT-2, replacing the activation at ``hook_name`` with
    ``replacement_fn(activation)`` on every forward pass. Used for
    zero-ablation and for splicing in an SAE reconstruction."""
    import torch

    def hook_fn(activation, hook):
        return replacement_fn(activation)

    with torch.inference_mode():
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            logits = model(tokens, return_type="logits")
    return logits


def clean_logits(model, tokens):
    import torch

    with torch.inference_mode():
        return model(tokens, return_type="logits")


def zero_ablation_logits(model, tokens, hook_name):
    import torch

    return run_with_replacement(model, tokens, hook_name, lambda act: torch.zeros_like(act))


def spliced_sae_logits(model, sae, tokens, hook_name):
    """Splice the SAE's reconstruction (encode then decode) into the
    forward pass at ``hook_name``."""

    def reconstruct(activation):
        features = sae.encode(activation)
        return sae.decode(features)

    return run_with_replacement(model, tokens, hook_name, reconstruct)


def loss_recovered(clean_loss: float, zero_ablation_loss: float, spliced_loss: float) -> float:
    """Percentage of the zero-ablation damage that the SAE reconstruction
    recovers. 100% = as good as the clean model; 0% = as bad as zeroing
    the whole layer out."""
    denom = zero_ablation_loss - clean_loss
    if denom == 0:
        return float("nan")
    return 100.0 * (zero_ablation_loss - spliced_loss) / denom


@dataclass
class FidelityAccumulator:
    """Batch-by-batch accumulator for FVU, L0 and cosine similarity.

    FVU's denominator (``sum ||x - mean(x)||^2``) needs the dataset mean,
    which this computes exactly in a single pass via
    ``sum ||x||^2 - n * ||mean||^2`` (parallel-axis identity) instead of a
    second pass over the data.
    """

    d_model: int
    n_tokens: int = 0
    sum_sq_error: float = 0.0
    sum_l0: float = 0.0
    sum_cosine: float = 0.0
    sum_x: "torch.Tensor | None" = field(default=None, repr=False)
    sum_x_sq_norm: float = 0.0

    def __post_init__(self):
        import torch

        if self.sum_x is None:
            self.sum_x = torch.zeros(self.d_model)

    def update(self, original, reconstruction, features) -> None:
        """``original``/``reconstruction``: [n_tokens, d_model] (BOS already
        excluded, batch dims already flattened). ``features``: [n_tokens, d_sae]."""
        import torch.nn.functional as F

        n = original.shape[0]
        self.n_tokens += n
        self.sum_x += original.sum(dim=0).to(self.sum_x.dtype)
        self.sum_x_sq_norm += original.pow(2).sum().item()
        self.sum_sq_error += (original - reconstruction).pow(2).sum().item()
        self.sum_l0 += (features > 0).float().sum().item()
        self.sum_cosine += F.cosine_similarity(original, reconstruction, dim=-1).sum().item()

    def finalize(self) -> dict[str, float]:
        if self.n_tokens == 0:
            return {"fvu": float("nan"), "l0": float("nan"), "cosine": float("nan")}

        mean = self.sum_x / self.n_tokens
        total_variance = self.sum_x_sq_norm - self.n_tokens * mean.pow(2).sum().item()

        fvu = self.sum_sq_error / total_variance if total_variance > 0 else float("nan")
        mean_l0 = self.sum_l0 / (self.n_tokens * 1.0) if self.n_tokens else float("nan")
        mean_cosine = self.sum_cosine / self.n_tokens if self.n_tokens else float("nan")

        return {"fvu": fvu, "l0": mean_l0, "cosine": mean_cosine}
