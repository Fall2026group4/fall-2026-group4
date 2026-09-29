"""Tests for src/sae/steering.py.

These avoid downloading GPT-2/SAE weights by faking the tiny slice of the
TransformerLens API the steering code actually uses: ``.hooks(fwd_hooks=...)``
as a context manager and ``.generate(...)``. This is enough to check the
steering math and the sweep bookkeeping without needing model weights or a
GPU/CPU-heavy forward pass in CI.
"""

import numpy as np
import pytest

from src.sae.steering import (
    generate_with_steering,
    make_steering_hook,
    run_steering_sweep,
)


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    """Stub out torch for every test in this file.

    The steering module only needs ``torch.manual_seed`` at call time, so
    tests don't need a real torch install (which can be large/slow to set
    up in CI) — a minimal fake is enough to prove seeding happens.
    """

    class _FakeTorchModule:
        seeds_used = []

        @staticmethod
        def manual_seed(seed):
            _FakeTorchModule.seeds_used.append(seed)

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorchModule())
    yield _FakeTorchModule
    _FakeTorchModule.seeds_used = []


class _FakeHookContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeSAEModel:
    """Minimal stand-in for HookedSAETransformer for steering tests."""

    def __init__(self):
        self.generate_calls = []

    def hooks(self, fwd_hooks):
        # Record that a hook was registered, then just no-op: the test
        # exercises make_steering_hook directly for the actual math.
        self.last_fwd_hooks = fwd_hooks
        return _FakeHookContext()

    def generate(self, prompt, max_new_tokens, temperature):
        call = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "steered": getattr(self, "last_fwd_hooks", None) is not None,
        }
        self.generate_calls.append(call)
        return f"generated:{prompt}:{len(self.generate_calls)}"


def test_make_steering_hook_adds_scaled_direction_to_last_token_only():
    direction = np.array([1.0, 2.0, 3.0])
    hook_fn = make_steering_hook(direction, steering_strength=10.0)

    # shape: (batch=1, seq_len=2, d_model=3)
    activation = np.zeros((1, 2, 3))
    result = hook_fn(activation, hook=None)

    # Only the last position should be modified.
    assert np.allclose(result[:, 0, :], [0.0, 0.0, 0.0])
    assert np.allclose(result[:, -1, :], [10.0, 20.0, 30.0])


def test_generate_with_steering_baseline_uses_no_hook():
    model = _FakeSAEModel()
    output = generate_with_steering(
        model, "hello", hook_name="blocks.8.hook_resid_pre", steering_strength=0.0
    )
    assert model.generate_calls[-1]["steered"] is False
    assert "hello" in output


def test_run_steering_sweep_includes_baseline_and_all_strengths():
    model = _FakeSAEModel()
    direction = np.array([1.0, 0.0])

    outputs = run_steering_sweep(
        model,
        prompt="The best place to spend my vacation is",
        hook_name="blocks.8.hook_resid_pre",
        steering_direction=direction,
        strengths=[10, 20, 30],
    )

    assert set(outputs.keys()) == {0, 10, 20, 30}
    # baseline (0) plus 3 strengths = 4 generate() calls
    assert len(model.generate_calls) == 4


def test_run_steering_sweep_is_deterministic_given_fixed_seed(_fake_torch):
    """Two sweeps with the same seed should request the same torch seed
    before every generation — this is the fix for "No seeding anywhere"
    from the instructor review."""
    model = _FakeSAEModel()
    run_steering_sweep(
        model,
        prompt="p",
        hook_name="h",
        steering_direction=np.array([1.0]),
        strengths=[5, 10],
        seed=42,
    )

    assert _fake_torch.seeds_used == [42, 42, 42]  # baseline + 2 strengths, same seed
