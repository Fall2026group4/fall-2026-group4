"""Modern SAE implementations for the R2 data-scaled retraining (2026-11-10).

Same interface as the in-house SAE class (scripts/openwebtext/03_training_sae.py), so every evaluation script works
unchanged: ``forward(x) -> (reconstruction, z, pre)``, ``encoder`` (Linear), ``decoder`` (Linear without bias,
weight [d_in, d_sae]) and ``output_bias`` (subtracted before encoding, added after decoding).

Architectures (published recipes):
  * ``topk``      TopK with AuxK dead-latent loss (Gao et al. 2024, arXiv 2406.04093)
  * ``batchtopk`` BatchTopK: top k*B over the batch in training, a global threshold at inference
                  (Bussmann, Leask & Nanda 2024, arXiv 2412.06410)
  * ``jumprelu``  JumpReLU with straight-through estimators for the threshold and an L0 penalty
                  (Rajamanoharan et al. 2024, arXiv 2407.14435); rectangle kernel
  * ``relu``      ReLU with an L1 penalty on decoder-norm-weighted activations (continuity with our earlier SAEs)
  * ``topk_frozen_decoder``  TopK whose decoder stays at its random initialisation (random baseline,
                  Korznikov et al. 2026, arXiv 2602.14111)

Training happens on activations scaled by ``scale`` (so that E||x||^2 = d_in). ``fold_scale`` rewrites the weights so
the saved SAE acts on raw activations; the activation pattern z is unchanged by folding.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'modern_v1'


def _rect(u):
    return ((u > -0.5) & (u < 0.5)).to(u.dtype)


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, eps):
        ctx.save_for_backward(x, theta)
        ctx.eps = eps
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        eps = ctx.eps
        gx = g * (x > theta).to(x.dtype)
        gtheta = (-(theta / eps) * _rect((x - theta) / eps) * g).sum(0)
        return gx, gtheta, None


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, eps):
        ctx.save_for_backward(x, theta)
        ctx.eps = eps
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        eps = ctx.eps
        gtheta = (-(1.0 / eps) * _rect((x - theta) / eps) * g).sum(0)
        return torch.zeros_like(x), gtheta, None


class ModernSAE(nn.Module):
    def __init__(self, d_in, d_sae, architecture, k=128, jump_eps=1e-3, init_threshold=1e-3):
        super().__init__()
        assert architecture in ('topk', 'batchtopk', 'jumprelu', 'relu', 'topk_frozen_decoder')
        self.architecture, self.k, self.jump_eps = architecture, int(k), float(jump_eps)
        self.d_in, self.d_sae = d_in, d_sae
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            w = F.normalize(torch.randn(d_in, d_sae), dim=0)
            self.decoder.weight.copy_(w)
            self.encoder.weight.copy_(w.T)               # encoder initialised to the decoder transpose (Gao et al.)
            self.encoder.bias.zero_()
        if architecture == 'jumprelu':
            self.log_threshold = nn.Parameter(torch.full((d_sae,), math.log(init_threshold)))
        # BatchTopK inference threshold (estimated during training); -1 = not set
        self.register_buffer('bt_threshold', torch.tensor(-1.0))
        if architecture == 'topk_frozen_decoder':
            self.decoder.weight.requires_grad_(False)

    # ------------------------------------------------------------------ forward
    def pre(self, x):
        return self.encoder(x - self.output_bias)

    def activate(self, pre):
        a = self.architecture
        if a == 'relu':
            return F.relu(pre)
        if a in ('topk', 'topk_frozen_decoder'):
            pos = F.relu(pre)
            v, i = pos.topk(self.k, dim=-1)
            return torch.zeros_like(pos).scatter(-1, i, v)
        if a == 'batchtopk':
            pos = F.relu(pre)
            if self.training:
                flat = pos.flatten()
                n = min(self.k * pos.shape[0], flat.numel())
                v, i = flat.topk(n)
                return torch.zeros_like(flat).scatter(0, i, v).view_as(pos)
            thr = self.bt_threshold.item()
            if thr < 0:
                raise RuntimeError('BatchTopK inference threshold not set')
            return pos * (pos > thr).to(pos.dtype)
        theta = self.log_threshold.exp()
        if self.training:
            return _JumpReLU.apply(pre, theta, self.jump_eps)
        return pre * (pre > theta).to(pre.dtype)

    def decode(self, z):
        return F.linear(z, self.decoder.weight) + self.output_bias

    def forward(self, x):
        pre = self.pre(x)
        z = self.activate(pre)
        return self.decode(z), z, pre

    # ------------------------------------------------------------------ training helpers
    def l0_surrogate(self, pre):
        """JumpReLU: STE estimate of the number of active latents per row (gradient flows to the threshold only)."""
        return _Step.apply(pre, self.log_threshold.exp(), self.jump_eps).sum(-1).mean()

    @torch.no_grad()
    def normalize_decoder(self):
        if self.architecture != 'topk_frozen_decoder':
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    @torch.no_grad()
    def fold_scale(self, scale):
        """Trained on s*x; afterwards act on raw x. z is unchanged; reconstruction is divided by s."""
        self.encoder.weight.mul_(scale)
        self.decoder.weight.div_(scale)
        self.output_bias.div_(scale)

    def config(self):
        return dict(format=FORMAT, architecture=self.architecture, d_in=self.d_in, d_sae=self.d_sae, k=self.k,
                    jump_eps=self.jump_eps)


def build_from_checkpoint(ckpt):
    cfg = ckpt['config']
    m = ModernSAE(cfg['d_in'], cfg['d_sae'], cfg['architecture'], k=cfg.get('k', 128),
                  jump_eps=cfg.get('jump_eps', 1e-3))
    m.load_state_dict(ckpt['model_state_dict'])
    return m
