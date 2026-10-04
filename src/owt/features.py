"""Feature health: firing frequency, dead/dense latent counts, and top
activating snippets for a quick sanity read.

Firing frequency is accumulated batch by batch (never holds a
[n_tokens, d_sae] matrix for the whole sample - same disk/memory
discipline as fidelity.py) and, like fidelity, excludes the BOS position.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class FeatureFrequencyAccumulator:
    """Accumulates how often each latent fires (activation > 0) across
    all non-BOS tokens seen so far."""

    d_sae: int
    n_tokens: int = 0
    fire_counts: "torch.Tensor | None" = field(default=None, repr=False)

    def __post_init__(self):
        import torch

        if self.fire_counts is None:
            self.fire_counts = torch.zeros(self.d_sae)

    def update(self, features) -> None:
        """``features``: [n_tokens, d_sae] (BOS already excluded)."""
        self.n_tokens += features.shape[0]
        self.fire_counts += (features > 0).float().sum(dim=0).cpu()

    def finalize(self) -> "torch.Tensor":
        """Return the firing-frequency vector, shape [d_sae]."""
        if self.n_tokens == 0:
            return self.fire_counts
        return self.fire_counts / self.n_tokens


def dead_and_dense_counts(frequency, dense_threshold: float = 0.10) -> tuple[int, int]:
    """Count latents that never fire (dead) and latents that fire on more
    than ``dense_threshold`` of tokens (dense)."""
    dead = int((frequency == 0).sum().item())
    dense = int((frequency > dense_threshold).sum().item())
    return dead, dense


def top_frequent_feature_ids(frequency, n: int = 20) -> list[int]:
    """The ``n`` most frequently firing latent ids, for the quick
    sanity-read snippet dump."""
    import torch

    top = torch.topk(frequency, k=min(n, frequency.shape[0]))
    return top.indices.tolist()


class TopActivatingTracker:
    """Streaming top-k tracker for one feature's top activating
    (document, position, snippet) triples - a min-heap so the full
    activation matrix never has to be held in memory."""

    def __init__(self, k: int = 10):
        self.k = k
        self._heap: list[tuple[float, int, int]] = []  # (activation, doc_id, position)

    def update(self, activations, doc_ids: list[int], positions: list[int]) -> None:
        for act, doc_id, pos in zip(activations.tolist(), doc_ids, positions):
            entry = (act, doc_id, pos)
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, entry)
            elif act > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)

    def top(self) -> list[tuple[float, int, int]]:
        """Return (activation, doc_id, position) sorted strongest first."""
        return sorted(self._heap, key=lambda x: -x[0])
