"""Tests for the OpenWebText x Pretrained SAEs pipeline (src/owt/).

Per the project spec: "checks every chunk is exactly 128 tokens with BOS
first; hook-point mapping (resid_post L = resid_pre L+1); splicing in the
original activation (no SAE) gives exactly the clean loss; zero-strength
steering gives zero KL." Run with `python -m pytest tests/test_owt.py`
before every push.

These use a tiny fake hooked model (no GPT-2 download, no network) - same
philosophy as src/tests/test_steering.py's fake torch stand-ins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.owt.data import chunk_tokens
from src.owt.fidelity import clean_logits, mean_ce_loss, run_with_replacement
from src.owt.saes import normalize_hook_point
from src.owt.steering_cost import kl_divergence, steered_logits, unit_direction


# ---------------------------------------------------------------------------
# Stage 0: chunking
# ---------------------------------------------------------------------------


def test_chunk_is_exactly_context_length_with_bos_first():
    bos = 50256
    token_ids = list(range(1000, 1300))  # plenty of real tokens
    chunk = chunk_tokens(token_ids, bos_token_id=bos, context_length=128)

    assert chunk is not None
    assert len(chunk) == 128
    assert chunk[0] == bos
    assert chunk[1:] == token_ids[:127]


def test_chunk_skips_documents_shorter_than_context_length():
    bos = 50256
    token_ids = list(range(10))  # far fewer than 127 real tokens needed
    chunk = chunk_tokens(token_ids, bos_token_id=bos, context_length=128)
    assert chunk is None


# ---------------------------------------------------------------------------
# Hook-point mapping
# ---------------------------------------------------------------------------


def test_resid_post_layer_maps_to_resid_pre_next_layer():
    effective_layer, hook_name = normalize_hook_point("resid_post", layer=5)
    assert effective_layer == 6
    assert hook_name == "blocks.6.hook_resid_pre"


def test_resid_pre_layer_maps_to_itself():
    effective_layer, hook_name = normalize_hook_point("resid_pre", layer=5)
    assert effective_layer == 5
    assert hook_name == "blocks.5.hook_resid_pre"


def test_unknown_hook_point_type_raises():
    import pytest

    with pytest.raises(ValueError):
        normalize_hook_point("not_a_real_point", layer=0)


# ---------------------------------------------------------------------------
# Fake hooked model, standing in for GPT-2/TransformerLens
# ---------------------------------------------------------------------------


class FakeHookedModel:
    """Minimal stand-in for a TransformerLens HookedTransformer: supports
    ``.hooks(fwd_hooks=...)`` as a context manager and
    ``__call__(tokens, return_type="logits")``. The "activation" at the
    (single, implicit) hook point is just a deterministic embedding
    lookup, so a hook that returns its input unchanged is exactly
    equivalent to no hook at all - enough to test splicing/steering
    mechanics without a real model.
    """

    def __init__(self, d_model: int = 4, vocab_size: int = 6, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(vocab_size, d_model, generator=g)
        self.unembed = torch.randn(d_model, vocab_size, generator=g)
        self._fwd_hooks: list[tuple[str, object]] = []

    class _HooksContext:
        def __init__(self, model: "FakeHookedModel", fwd_hooks):
            self._model = model
            self._fwd_hooks = fwd_hooks

        def __enter__(self):
            self._model._fwd_hooks = self._fwd_hooks
            return self._model

        def __exit__(self, *exc_info):
            self._model._fwd_hooks = []
            return False

    def hooks(self, fwd_hooks):
        return FakeHookedModel._HooksContext(self, fwd_hooks)

    def __call__(self, tokens, return_type: str = "logits"):
        activation = self.embed[tokens]  # [batch, seq, d_model]
        for _hook_name, hook_fn in self._fwd_hooks:
            activation = hook_fn(activation, hook=None)
        return activation @ self.unembed


# ---------------------------------------------------------------------------
# Splicing the original activation back in must be a no-op
# ---------------------------------------------------------------------------


def test_splicing_original_activation_gives_exactly_clean_loss():
    model = FakeHookedModel()
    tokens = torch.randint(0, 6, (2, 7))

    clean = clean_logits(model, tokens)
    spliced = run_with_replacement(model, tokens, hook_name="fake_hook", replacement_fn=lambda act: act)

    assert torch.allclose(clean, spliced)

    clean_loss, _ = mean_ce_loss(clean, tokens)
    spliced_loss, _ = mean_ce_loss(spliced, tokens)
    assert abs(clean_loss - spliced_loss) < 1e-6


# ---------------------------------------------------------------------------
# Zero-strength steering must give zero KL divergence
# ---------------------------------------------------------------------------


def test_zero_strength_steering_gives_zero_kl():
    model = FakeHookedModel()
    tokens = torch.randint(0, 6, (2, 7))
    direction = torch.randn(4)
    unit_dir = unit_direction(direction)

    clean = clean_logits(model, tokens)
    steered = steered_logits(model, tokens, hook_name="fake_hook", unit_dir=unit_dir, alpha=0.0, residual_norm=1.0)

    assert torch.allclose(clean, steered)

    kl = kl_divergence(clean, steered)
    assert kl.abs().max().item() < 1e-6


def test_nonzero_steering_changes_logits_and_gives_positive_kl():
    model = FakeHookedModel()
    tokens = torch.randint(0, 6, (2, 7))
    direction = torch.randn(4)
    unit_dir = unit_direction(direction)

    clean = clean_logits(model, tokens)
    steered = steered_logits(model, tokens, hook_name="fake_hook", unit_dir=unit_dir, alpha=4.0, residual_norm=1.0)

    assert not torch.allclose(clean, steered)

    kl = kl_divergence(clean, steered)
    assert kl.mean().item() > 0
