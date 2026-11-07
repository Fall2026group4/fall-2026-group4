"""Checkpoint registry: one row per SAE checkpoint in ``results/``, with provenance and status.

Build / refresh:  python scripts/shared/build_checkpoint_registry.py   (writes docs/checkpoint_registry.csv)
Look up in code:  from src.sae.checkpoint_registry import load_registry, uninterrupted_checkpoint

The registry is read-only with respect to checkpoints: it never moves, rewrites or deletes a file.
Statuses follow ``docs/results_registry.md``.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
REGISTRY_CSV = REPO / 'docs' / 'checkpoint_registry.csv'
UNINTERRUPTED_METRICS = (REPO / 'results' / 'openwebtext' / 'reconstruction_evaluation' / 'uninterrupted_protocol'
                         / 'all_layers_per_run.csv')

# (regex on repo-relative path, family, status, note). First match wins.
FAMILIES = [
    (r'^results/professor_review_corrections/smoke/', 'smoke_original_copy', 'SMOKE', 'original location of smoke runs'),
    (r'^results/professor_review_corrections/', 'uninterrupted_original_copy', 'HISTORICAL_COPY',
     'pre-migration copy; byte-identical to results/openwebtext/convergence/*_uninterrupted_seed*'),
    (r'/smoke/', 'smoke', 'SMOKE', 'pipeline check, never a result'),
    (r'_uninterrupted_seed\d+/', 'owt_uninterrupted', 'AUTHORITATIVE', 'single uninterrupted 15k-step run'),
    (r'_p1_seed\d+/', 'owt_p1_pilot', 'PILOT', 'P1 pilot (controller anneal/freeze, JumpReLU threshold init, val selection); seed 42 only'),
    (r'_resume_5000_to_15000/', 'owt_resume_layer6', 'VALID_SCOPE',
     'layer-6 base SAEs of every GoT/RepE/fine-tuning/causal experiment; within-layer comparisons only'),
    (r'_warmstart_1000_to_15000/', 'owt_warmstart', 'SUPERSEDED', 'replaced by *_uninterrupted_seed*'),
    (r'jumprelu_kernel_ste', 'owt_kernel_ste', 'EXPLORATORY', 'kernel-STE JumpReLU; different estimator, script missing'),
    (r'^results/openwebtext/convergence/layer_\d+/\w+_l0_\d+/', 'owt_convergence_initial', 'VALID_SCOPE',
     'first 1k/5k steps from which the layer-6 resume runs continued'),
    (r'^results/openwebtext/sae_sparsity_sweep/', 'owt_sweep_1000', 'VALID_SCOPE', '1000 steps; convergence speed only'),
    (r'^results/openwebtext/sae_pilot/', 'owt_pilot', 'HISTORICAL', 'first 1000-step pilot'),
    (r'sae_finetuning/truth_aware_pilot_', 'got_truth_aware_pilot', 'PILOT',
     'corrected truth-aware pilot (A/B groups, independent probes); validation only'),
    (r'^results/geometry_of_truth/sae_finetuning/', 'got_finetuned', 'VALID_SCOPE',
     'fine-tuned on GoT; not sparsity-matched to its base SAE'),
]


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _classify(rel):
    for pattern, family, status, note in FAMILIES:
        if re.search(pattern, rel):
            return family, status, note
    return 'unknown', 'UNREVIEWED', ''


def _num(pattern, text, cast=int):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def describe(path):
    """Registry row for one checkpoint file, or None if the file holds no SAE weights."""
    path = Path(path)
    rel = path.resolve().relative_to(REPO).as_posix()
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict') if isinstance(ckpt, dict) else None
    if not state or 'encoder.weight' not in state:
        return None
    cfg = ckpt.get('config') or {}
    orig = cfg.get('original_config') or {}
    family, status, note = _classify(rel)
    d_sae, d_in = state['encoder.weight'].shape
    arch = cfg.get('architecture') or orig.get('architecture')
    if arch is None:
        m = re.search(r'(jumprelu|topk|relu)', rel)
        arch = m.group(1) if m else ('jumprelu' if 'log_threshold' in state else None)
    source = cfg.get('source_checkpoint') or ckpt.get('source_checkpoint') or ''
    if family == 'got_finetuned' and source:
        # Fine-tuned SAEs inherit layer / target / seed from their base checkpoint.
        base = torch.load(REPO / source, map_location='cpu', weights_only=False).get('config') or {}
        base_orig = base.get('original_config') or {}
        cfg = {'target_l0': base.get('target_l0', base_orig.get('target_l0')),
               'seed': base.get('seed', base_orig.get('seed')), 'layer': base.get('layer', base_orig.get('layer'))}
    val = ckpt.get('validation_metrics') or {}
    test = ckpt.get('test_metrics') or {}
    return {
        'path': rel,
        'family': family,
        'status': status,
        'layer': cfg.get('layer', orig.get('layer', _num(r'layer_(\d+)', rel))),
        'architecture': arch,
        'target_l0': cfg.get('target_l0', orig.get('target_l0', _num(r'l0_(\d+)', rel))),
        'd_in': int(d_in),
        'd_sae': int(d_sae),
        'seed': cfg.get('seed', orig.get('seed', _num(r'seed(\d+)', rel))),
        'step': ckpt.get('step', _num(r'step_(\d+)', path.name)),
        'procedure': cfg.get('procedure') or cfg.get('training_method') or ckpt.get('experiment') or '',
        'source_checkpoint': source,
        'val_l0': val.get('l0', val.get('actual_l0')),
        'val_explained_variance': val.get('explained_variance'),
        'owt_test_l0': test.get('actual_l0'),
        'owt_test_explained_variance': test.get('explained_variance'),
        'loadable_with_shared_sae_class': family != 'owt_kernel_ste',
        'size_bytes': path.stat().st_size,
        'sha256': _sha256(path),
        'note': note,
    }


def build(root=REPO / 'results', out=REGISTRY_CSV):
    rows = [r for p in sorted(Path(root).rglob('*.pt')) if (r := describe(p)) is not None]
    df = pd.DataFrame(rows)
    if UNINTERRUPTED_METRICS.exists():
        flags = pd.read_csv(UNINTERRUPTED_METRICS)[['checkpoint_path', 'sparsity_flag', 'run_ok']]
        df = df.merge(flags.rename(columns={'checkpoint_path': 'path'}), on='path', how='left')
    df.to_csv(out, index=False)
    return df


def load_registry(path=REGISTRY_CSV):
    return pd.read_csv(path)


def uninterrupted_checkpoint(layer, architecture, seed, target_l0=128):
    """Path of the authoritative OWT checkpoint for (layer, architecture, seed)."""
    p = (REPO / 'results' / 'openwebtext' / 'convergence' / f'layer_{layer}'
         / f'{architecture}_l0_{target_l0}_uninterrupted_seed{seed}' / 'checkpoint_step_15000.pt')
    if not p.exists():
        raise FileNotFoundError(p)
    return p
