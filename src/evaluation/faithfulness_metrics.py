"""Shared reconstruction and faithfulness metrics (single source of truth since 2026-11-07).

Definitions match the OWT evaluation in ``scripts/openwebtext/03_training_sae.py:evaluate`` exactly, so numbers
computed here on OWT reproduce ``test_metrics_all_runs.csv``:

* ``mse``                 = sum ||x - x_hat||^2 / (n * d)                  (raw, scale-dependent; compare within a layer only)
* ``explained_variance``  = 1 - sum ||x - x_hat||^2 / sum ||x - mean(x)||^2  (mean taken over the evaluated set)
* ``nmse``                = 1 - explained_variance
* ``relative_sq_error``   = sum ||x - x_hat||^2 / sum ||x||^2               (does not depend on the set's spread)
* ``cosine``              = mean over rows of cos(x, x_hat)
* ``l0``                  = mean over rows of #(z > 0)

Truth / honesty metrics use *frozen* probes: a probe (or direction) is fit on ORIGINAL train activations, its
hyper-parameters are chosen on ORIGINAL validation activations, and it is then applied unchanged to reconstructions.
``auroc_gap = auroc(original) - auroc(reconstructed)`` on the same rows (positive = information lost).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
_TRAINING_SCRIPT = REPO / 'scripts' / 'openwebtext' / '03_training_sae.py'
_SAE_CLASS = None


def sae_class():
    """The in-house SAE class, imported from the training script (not duplicated)."""
    global _SAE_CLASS
    if _SAE_CLASS is None:
        spec = importlib.util.spec_from_file_location('owt03_training_sae', _TRAINING_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SAE_CLASS = module.SAE
    return _SAE_CLASS


def load_sae(path, device='cpu', architecture=None, k=None):
    """Load an in-house SAE checkpoint. Returns (model, checkpoint dict).

    ``architecture`` / ``k`` are only needed for checkpoints without a config (GoT fine-tuned SAEs); otherwise they
    are read from ``config`` (or ``config['original_config']`` for resumed runs).
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config') or {}
    orig = cfg.get('original_config') or {}
    state = ckpt['model_state_dict']
    d_sae, d_in = state['encoder.weight'].shape
    arch = architecture or cfg.get('architecture') or orig.get('architecture')
    k = k or cfg.get('target_l0') or orig.get('target_l0') or 128
    if arch is None:
        raise ValueError(f'{path}: architecture unknown; pass architecture=')
    if (arch == 'jumprelu') != ('log_threshold' in state):
        raise ValueError(f'{path}: architecture {arch!r} inconsistent with state dict')
    model = sae_class()(d_in, d_sae, arch, int(k), temperature=cfg.get('temperature', 0.1))
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


@torch.no_grad()
def reconstruct(model, x, device='cpu', batch_size=4096):
    """Return (x_hat, l0_per_row, active_per_latent) for a float tensor/array x of shape (n, d)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    outs, l0s = [], []
    active = None
    for batch in x.split(batch_size):
        rec, z, _ = model(batch.to(device))
        outs.append(rec.float().cpu())
        nz = z > 0
        l0s.append(nz.sum(-1).float().cpu())
        a = nz.any(0).cpu()
        active = a if active is None else active | a
    return torch.cat(outs), torch.cat(l0s), active


def reconstruction_metrics(x, x_hat, l0_per_row=None):
    """Reconstruction metrics of x_hat against x (both (n, d)); definitions in the module docstring."""
    x = torch.as_tensor(x, dtype=torch.float64)
    x_hat = torch.as_tensor(x_hat, dtype=torch.float64)
    if x.shape != x_hat.shape or x.ndim != 2:
        raise ValueError(f'shape mismatch {tuple(x.shape)} vs {tuple(x_hat.shape)}')
    sse = (x - x_hat).square().sum().item()
    total = (x - x.mean(0)).square().sum().item()
    energy = x.square().sum().item()
    ev = 1 - sse / max(total, 1e-12)
    out = {
        'n': int(x.shape[0]),
        'mse': sse / x.numel(),
        'explained_variance': ev,
        'nmse': 1 - ev,
        'relative_sq_error': sse / max(energy, 1e-12),
        'cosine': torch.nn.functional.cosine_similarity(x, x_hat, dim=-1, eps=1e-8).mean().item(),
    }
    if l0_per_row is not None:
        out['l0'] = float(torch.as_tensor(l0_per_row, dtype=torch.float64).mean())
    return out


def dead_fraction(active_per_latent):
    """Fraction of latents that never fire on the evaluated set."""
    a = torch.as_tensor(active_per_latent, dtype=torch.bool)
    return 1 - a.float().mean().item()


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    if np.unique(labels).size < 2 or not np.isfinite(np.asarray(scores, dtype=float)).all():
        return float('nan')
    return float(roc_auc_score(labels, np.asarray(scores)))


def diff_of_means_direction(X, labels, mask=None, normalize=True):
    """Mean(X | label 1) - mean(X | label 0) over ``mask`` rows (the RepE / GoT mass-mean direction)."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    mask = np.ones(len(labels), bool) if mask is None else np.asarray(mask, bool)
    d = X[mask & (labels == 1)].mean(0) - X[mask & (labels == 0)].mean(0)
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError('zero-norm direction')
    return d / n if normalize else d


def direction_fidelity(original_direction, reconstructed_direction):
    """Cosine and norm ratio between a direction re-estimated on reconstructions and the original one."""
    a = np.asarray(original_direction, dtype=np.float64)
    b = np.asarray(reconstructed_direction, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {'direction_cosine': float(a @ b / (na * nb)), 'direction_norm_ratio': float(nb / na)}


def _fast_auroc(labels, scores):
    """Mann-Whitney AUROC with tie-averaged ranks; equals sklearn's roc_auc_score."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = len(labels) - n1
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def grouped_bootstrap_gap(labels, scores_original, scores_reconstructed, groups, n_boot=1000, seed=0):
    """Paired 95 % CI of auroc(original) - auroc(reconstructed), resampling whole groups with replacement.

    Both score vectors are evaluated on the same resampled rows, so the CI reflects the paired gap.
    """
    labels = np.asarray(labels)
    so, sr = np.asarray(scores_original), np.asarray(scores_reconstructed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))])
        if np.unique(labels[idx]).size < 2:
            continue
        gaps.append(_fast_auroc(labels[idx], so[idx]) - _fast_auroc(labels[idx], sr[idx]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def pair_accuracy(pair_ids, labels, scores):
    """Fraction of (honest, untruthful) pairs where the label-1 member scores higher (RepE)."""
    import pandas as pd
    t = pd.DataFrame({'id': pair_ids, 'label': labels, 'score': scores}).pivot(index='id', columns='label', values='score')
    return float((t[1] > t[0]).mean())
