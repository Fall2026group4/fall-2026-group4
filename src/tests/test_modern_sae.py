"""Tests for src/sae/modern_sae.py (the R2 data-scaled SAEs) and the R2 trainer's frozen configuration.

CPU only, no checkpoints or activation caches. Run from the repository root:
    python -m pytest -q src/tests/test_modern_sae.py
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from src.sae import modern_sae as ms

ROOT = Path(__file__).resolve().parents[2]
D, W, K = 12, 48, 4


def _sae(arch, seed=0, **kw):
    torch.manual_seed(seed)
    return ms.ModernSAE(D, W, arch, k=K, **kw)


# ------------------------------------------------------------------ architectures
@pytest.mark.parametrize('arch', ['topk', 'topk_frozen_decoder'])
def test_topk_keeps_at_most_k_nonnegative_latents(arch):
    sae = _sae(arch).eval()
    _, z, pre = sae(torch.randn(64, D))
    assert ((z > 0).sum(-1) <= K).all()
    assert (z >= 0).all()
    # the kept values are the largest positive pre-activations
    kept = torch.where(z > 0, z, torch.full_like(z, float('inf'))).min(-1).values
    dropped = torch.where(z > 0, torch.full_like(z, -float('inf')), torch.relu(pre)).max(-1).values
    has = (z > 0).any(-1)
    assert (kept[has] >= dropped[has]).all()


def test_frozen_decoder_has_no_decoder_gradient_and_is_not_renormalised():
    sae = _sae('topk_frozen_decoder')
    assert not sae.decoder.weight.requires_grad
    w0 = sae.decoder.weight.clone()
    with torch.no_grad():
        sae.decoder.weight.mul_(2.0)
    sae.normalize_decoder()
    assert torch.allclose(sae.decoder.weight, 2 * w0)


def test_normalize_decoder_gives_unit_columns():
    sae = _sae('topk')
    with torch.no_grad():
        sae.decoder.weight.mul_(3.0)
    sae.normalize_decoder()
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(W), atol=1e-6)


def test_batchtopk_training_keeps_k_times_batch_and_inference_needs_threshold():
    sae = _sae('batchtopk')
    x = torch.randn(32, D)
    sae.train()
    _, z, _ = sae(x)
    assert int((z > 0).sum()) <= K * 32
    sae.eval()
    with pytest.raises(RuntimeError, match='threshold not set'):
        sae(x)
    sae.bt_threshold.fill_(0.3)
    _, z, pre = sae(x)
    pos = torch.relu(pre)
    assert torch.equal(z > 0, pos > 0.3)
    assert torch.allclose(z[z > 0], pos[z > 0])


def test_relu_is_plain_relu():
    sae = _sae('relu').eval()
    _, z, pre = sae(torch.randn(16, D))
    assert torch.equal(z, torch.relu(pre))


def test_jumprelu_eval_gate_and_unknown_architecture():
    sae = _sae('jumprelu', init_threshold=0.2).eval()
    _, z, pre = sae(torch.randn(16, D))
    assert torch.equal(z, pre * (pre > 0.2))
    with pytest.raises(AssertionError):
        ms.ModernSAE(D, W, 'gated')


# ------------------------------------------------------------------ straight-through estimators
def test_jumprelu_ste_gradients():
    """d/dx = 1[x > theta]; d/dtheta = -(theta/eps) * rect((x - theta)/eps), summed over the batch."""
    eps = 0.1
    x = torch.tensor([[0.05, 0.50, 0.32, -1.0]], requires_grad=True)
    theta = torch.tensor([0.30, 0.30, 0.30, 0.30], requires_grad=True)
    ms._JumpReLU.apply(x, theta, eps).sum().backward()
    assert x.grad.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    # only x=0.32 lies within eps/2 of theta
    assert torch.allclose(theta.grad, torch.tensor([0.0, 0.0, -0.30 / eps, 0.0]))


def test_step_ste_gradients_flow_to_threshold_only():
    eps = 0.1
    x = torch.tensor([[0.28, 0.9], [0.31, 0.0]], requires_grad=True)
    theta = torch.tensor([0.3, 0.3], requires_grad=True)
    out = ms._Step.apply(x, theta, eps)
    assert out.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    out.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.allclose(theta.grad, torch.tensor([-2.0 / eps, 0.0]))


def test_l0_surrogate_counts_active_latents():
    sae = _sae('jumprelu', init_threshold=0.1)
    pre = sae.pre(torch.randn(20, D))
    expected = (pre > 0.1).float().sum(-1).mean()
    assert torch.isclose(sae.l0_surrogate(pre), expected)


# ------------------------------------------------------------------ fold_scale
@pytest.mark.parametrize('arch', ['topk', 'batchtopk', 'relu', 'jumprelu', 'topk_frozen_decoder'])
def test_fold_scale_round_trip(arch):
    """An SAE trained on s*x, after fold_scale(s), maps raw x to (its output on s*x) / s with identical z."""
    sae = _sae(arch, seed=3)
    with torch.no_grad():
        sae.output_bias.normal_()
        sae.encoder.bias.normal_(0, 0.1)
    if arch == 'batchtopk':
        sae.bt_threshold.fill_(0.2)
    sae.eval()
    s = 7.5
    x = torch.randn(40, D, dtype=torch.float64)
    sae = sae.double()
    rec_s, z_s, pre_s = sae(s * x)
    sae.fold_scale(s)
    rec, z, pre = sae(x)
    assert torch.allclose(pre, pre_s, atol=1e-10)
    assert torch.equal(z > 0, z_s > 0) and torch.allclose(z, z_s, atol=1e-10)
    assert torch.allclose(rec, rec_s / s, atol=1e-10)


def test_build_from_checkpoint_round_trip():
    sae = _sae('batchtopk', seed=5)
    sae.bt_threshold.fill_(0.25)
    ckpt = dict(model_state_dict=sae.state_dict(), config=sae.config())
    m = ms.build_from_checkpoint(ckpt).eval()
    x = torch.randn(10, D)
    assert torch.equal(m(x)[0], sae.eval()(x)[0])
    assert m.config() == sae.config() and m.config()['format'] == ms.FORMAT


# ------------------------------------------------------------------ R2 trainer configuration (frozen in the R2 README)
def _trainer():
    spec = importlib.util.spec_from_file_location('r2t', ROOT / 'scripts/openwebtext/17_streaming_sae_training.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r2_trainer_configuration():
    t = _trainer()
    assert t.TARGET_L0 == 128 and t.WIDTH == 12288 and t.LAYERS == (1, 8, 11) and t.SEEDS == (42, 1042, 2042)
    names = {c[0]: c for c in t.CONFIGS}
    assert set(names) == {'topk', 'batchtopk', 'relu', 'topk_w3072', 'topk_frozen_decoder'}
    assert names['topk_w3072'][2] == 3072 and names['topk_frozen_decoder'][3] == (42,)
    assert 'jumprelu' not in names        # abandoned in the pilot (R2 README 1.2)
    assert t.SAVE_AT == (10_000_000, 25_000_000, 50_000_000)


def test_r2_lr_schedule_warmup_constant_decay():
    t = _trainer()
    r = t.Run.__new__(t.Run)
    r.total = 10_000
    r.step = 0
    assert r.lr() == pytest.approx(t.LR / t.WARMUP)
    r.step = t.WARMUP + 10
    assert r.lr() == t.LR
    r.step = r.total
    assert r.lr() == 0.0
    r.step = int(r.total * (1 - t.DECAY_FRAC / 2))
    assert 0 < r.lr() < t.LR


def _cpu_run(t, arch, width):
    """A Run built around a small CPU SAE, bypassing __init__ (which builds a 768-d model on a device)."""
    r = t.Run.__new__(t.Run)
    torch.manual_seed(0)
    r.sae = ms.ModernSAE(D, width, arch, k=K)
    r.arch, r.total, r.step, r.ema_l0, r.bt_ema = arch, 1000, t.WARMUP, None, None
    r.opt = torch.optim.Adam([dict(params=[p for p in r.sae.parameters() if p.requires_grad], lr_mult=1.0)], lr=t.LR)
    r.lam = {'relu': 0.1, 'jumprelu': 0.01}.get(arch, 0.0)
    r.last_fired = torch.zeros(width, dtype=torch.long)
    return r


@pytest.mark.filterwarnings('ignore::UserWarning')   # autocast('cuda') is disabled on CPU-only machines
def test_r2_relu_sparsity_controller_lowers_lambda_when_l0_below_target():
    """The real train_step: a width-48 ReLU SAE has L0 < 48 < TARGET_L0 = 128, so lambda must fall by exp(-eta·c)."""
    t = _trainer()
    r = _cpu_run(t, 'relu', 48)
    out = r.train_step(torch.randn(64, D))
    assert out['l0'] < t.TARGET_L0
    import math
    assert r.lam == pytest.approx(0.1 * math.exp(-t.CTRL_ETA))       # log-ratio clipped at -1
    # decoder columns are renormalised after the step
    assert torch.allclose(r.sae.decoder.weight.norm(dim=0), torch.ones(48), atol=1e-5)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_r2_batchtopk_step_sets_inference_threshold_to_min_kept_activation():
    t = _trainer()
    r = _cpu_run(t, 'batchtopk', W)
    x = torch.randn(64, D)
    r.sae.train()
    with torch.no_grad():
        _, z, _ = r.sae(x)
    r.train_step(x)
    assert r.bt_ema == pytest.approx(float(z[z > 0].min()), rel=1e-5)
    assert float(r.sae.bt_threshold) == pytest.approx(r.bt_ema)
