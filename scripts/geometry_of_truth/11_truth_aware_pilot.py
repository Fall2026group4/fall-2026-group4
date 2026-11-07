#!/usr/bin/env python3
"""Corrected truth-aware fine-tuning pilot: layer 6, TopK, seeds 42/1042/2042, 3 variants (9 fine-tunes).

Protocol: docs/2026-11-07_truth_aware_finetuning_audit.md and the pre-registration written by --prepare.

Run from repository root, in order:
    python scripts/geometry_of_truth/11_truth_aware_pilot.py --prepare   # A/B group split + checks + pre-registration
    python scripts/geometry_of_truth/11_truth_aware_pilot.py --dry-run   # verify inputs/paths, no training
    python scripts/geometry_of_truth/11_truth_aware_pilot.py --train     # 9 fine-tunes

Design (fixed before any fine-tune):
* GoT TRAIN groups are split 50/50 per domain into A and B (seeded, group level; every group has 2 true + 2 false
  rows, so both halves are class-balanced). A: SAE fine-tuning batches and the training teacher. B: never seen by
  fine-tuning; used only to fit the independent EVALUATION probes (script 12). Validation: evaluation only.
  Test: never read.
* Base SAEs: results/openwebtext/convergence/layer_6/topk_l0_128_uninterrupted_seed{S}/checkpoint_step_15000.pt
  (TopK: L0 fixed at 128, so no sparsity drift). Base checkpoints are read only.
* Variants (same seed, same batches, same steps): reconstruction_only (lambda = 0), truth_aware (teacher = logistic
  probe fitted on ORIGINAL A activations, C chosen on ORIGINAL validation activations), random_teacher (same loss with
  a random unit direction per domain; bias and logit scale set like the real teacher's on A).
* Loss = 0.5 * MSE(GoT-A batch) + 0.5 * MSE(OWT-train batch) + lambda * mean_domain MSE(normalised teacher logits),
  lambda = 0.1 for truth_aware and random_teacher. Batch: 64 geographic-A + 64 numerical-A + 128 OWT-train rows.
  Adam lr 1e-5, 500 steps, grad clip 10, decoder renormalised every step, no checkpoint selection.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

LAYER, ARCH, SEEDS = 6, 'topk', (42, 1042, 2042)
VARIANTS = ('reconstruction_only', 'truth_aware', 'random_teacher')
LAMBDA, LR, STEPS, CLIP = 0.1, 1e-5, 500, 10.0
GOT_PER_DOMAIN, OWT_BATCH = 64, 128
SPLIT_SEED = 20261107
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
DOMAINS = {'geographic': ('cities', 'neg_cities'), 'numerical': ('larger_than', 'smaller_than')}
SPLIT_CSV = REPO / 'data/got/splits/train_AB_groups_truth_aware_pilot.csv'
OUT = REPO / 'results/geometry_of_truth/truth_aware_pilot'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/truth_aware_pilot_layer6_topk'
MIN_INFORMATIVE = 0.70


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_domain(domain):
    """Original layer-6 activations with split, group and A/B half (if the split file exists)."""
    rows, Xs = [], []
    for name in DOMAINS[domain]:
        d = torch.load(REPO / f'data/got/activations/gpt2_small/{name}_final_token.pt', map_location='cpu', weights_only=False)
        s = pd.read_csv(REPO / f'data/got/splits/{name}_splits.csv').set_index('row_id').reindex(np.asarray(d['row_ids']))
        assert (s.label.to_numpy() == np.asarray(d['labels'])).all()
        rows.append(s.assign(dataset=name).reset_index())
        Xs.append(d['activations'][LAYER].numpy().astype(np.float32))
    meta = pd.concat(rows, ignore_index=True)
    meta['group_id'] = meta.group_id.astype(str)
    if SPLIT_CSV.exists():
        ab = pd.read_csv(SPLIT_CSV, dtype={'group_id': str})
        ab = ab[ab.domain == domain].set_index('group_id')['half']
        meta['half'] = meta.group_id.map(ab).where(meta.split == 'train', '')
    return np.concatenate(Xs), meta


def select_C(X, y, fit_mask, val_mask):
    best = None
    for C in C_GRID:
        p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=42)).fit(X[fit_mask], y[fit_mask])
        auc = fm.safe_auroc(y[val_mask], p.decision_function(X[val_mask]))
        if best is None or auc > best[0] + 1e-15:
            best = (auc, C, p)
    return best


def prepare():
    if SPLIT_CSV.exists():
        raise SystemExit(f'{SPLIT_CSV} exists; refusing to overwrite.')
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)
    rows, checks = [], {}
    for domain in DOMAINS:
        _, meta = load_domain(domain)
        groups = np.array(sorted(meta[meta.split == 'train'].group_id.unique()))
        perm = rng.permutation(len(groups))
        half = np.where(np.isin(np.arange(len(groups)), perm[:len(groups) // 2]), 'A', 'B')
        rows += [{'domain': domain, 'group_id': g, 'half': h} for g, h in zip(groups, half)]
    pd.DataFrame(rows).to_csv(SPLIT_CSV, index=False)
    ok = True
    for domain in DOMAINS:
        X, m = load_domain(domain)
        y = m.label.to_numpy()
        A, B = (m.half == 'A').to_numpy(), (m.half == 'B').to_numpy()
        va = (m.split == 'val').to_numpy()
        gA, gB = set(m.group_id[A]), set(m.group_id[B])
        other = set(m.group_id[m.split != 'train'])
        c = {'rows_A': int(A.sum()), 'rows_B': int(B.sum()), 'groups_A': len(gA), 'groups_B': len(gB),
             'A_B_group_overlap': len(gA & gB), 'AB_vs_val_test_group_overlap': len((gA | gB) & other),
             'all_train_rows_assigned': bool(((m.split == 'train') == (A | B)).all()),
             'label_mean_A': float(y[A].mean()), 'label_mean_B': float(y[B].mean())}
        for name in DOMAINS[domain]:
            sub = (m.dataset == name).to_numpy()
            c[f'label_mean_A_{name}'] = float(y[A & sub].mean())
            c[f'label_mean_B_{name}'] = float(y[B & sub].mean())
        # sufficiency: probes fitted on ORIGINAL activations of each half must be informative on ORIGINAL validation
        for h, mask in (('A', A), ('B', B)):
            auc, C, _ = select_C(X, y, mask, va)
            c[f'probe_{h}_val_auroc'], c[f'probe_{h}_C'] = auc, C
        aucs_full = select_C(X, y, A | B, va)[0]
        c['probe_full_train_val_auroc'] = aucs_full
        c['passes'] = bool(c['A_B_group_overlap'] == 0 and c['AB_vs_val_test_group_overlap'] == 0
                           and c['all_train_rows_assigned'] and abs(c['label_mean_A'] - 0.5) < 0.01
                           and abs(c['label_mean_B'] - 0.5) < 0.01 and c['rows_A'] >= 500 and c['rows_B'] >= 500
                           and c['probe_A_val_auroc'] >= MIN_INFORMATIVE and c['probe_B_val_auroc'] >= MIN_INFORMATIVE)
        ok &= c['passes']
        checks[domain] = c
    (OUT / 'split_checks.json').write_text(json.dumps(checks, indent=2))
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'script_sha256': sha(__file__),
        'split_file': SPLIT_CSV.relative_to(REPO).as_posix(), 'split_file_sha256': sha(SPLIT_CSV), 'split_seed': SPLIT_SEED,
        'design': {'layer': LAYER, 'architecture': ARCH, 'seeds': SEEDS, 'variants': VARIANTS, 'lambda': LAMBDA,
                   'lr': LR, 'steps': STEPS, 'batch': f'{GOT_PER_DOMAIN}+{GOT_PER_DOMAIN} GoT-A + {OWT_BATCH} OWT train',
                   'teacher': 'logistic on ORIGINAL A activations, C chosen on ORIGINAL validation',
                   'random_teacher': 'unit Gaussian direction per domain (seed 7000+seed), bias/scale from A logits',
                   'checkpoint_selection': 'none (step 500)'},
        'evaluation': {
            'data': 'validation only (GoT val, RepE validation, OWT validation); test never read',
            'independent_probes': 'fitted on ORIGINAL B activations: logistic (C chosen on original validation), '
                                  'mass-mean direction, MLP 768-64-1 x 5 seeds (scripts/geometry_of_truth/07 code)',
            'teacher_probe': 'reported separately as objective attainment only',
            'diagnostic': 'logistic re-fitted on RECONSTRUCTED B, scored on reconstructed validation (decodability only)',
            'behaviour': 'activation splicing (not steering): OWT delta CE/KL, GoT statement KL and cities truth '
                         'preference, RepE final KL and persona-shift cosine',
            'repe': 'frozen logistic + mass-mean probes, direction fidelity (all-layer evaluation code)'},
        'success_criteria': {
            'primary': 'GEOGRAPHIC domain, for BOTH the B-logistic gap and the B-MLP gap (mean over 5 probe seeds): '
                       'truth_aware gap < reconstruction_only gap AND < random_teacher gap on all 3 seeds, with mean '
                       'paired difference > 2 x sd of the paired differences',
            'secondary': 'same test for the NUMERICAL domain (reported, not required)',
            'guards': ['OWT validation delta CE of truth_aware <= 1.10 x its paired base SAE',
                       'GoT cities truth-preference change of truth_aware >= reconstruction_only - 0.018 (2 pairs), mean over seeds',
                       'RepE validation mass-mean gap of truth_aware <= reconstruction_only + 0.02, mean over seeds'],
            'success': 'primary AND all guards'},
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    print(json.dumps(checks, indent=2))
    print('ALL SPLIT CHECKS PASS' if ok else 'SPLIT CHECKS FAILED')
    return ok


def teachers(seed, device):
    """Real and random teacher per domain: affine logit w.x + b and logit scale s (from ORIGINAL A activations)."""
    out = {}
    for domain in DOMAINS:
        X, m = load_domain(domain)
        y = m.label.to_numpy()
        A, va = (m.half == 'A').to_numpy(), (m.split == 'val').to_numpy()
        auc, C, p = select_C(X, y, A, va)
        w = p[1].coef_[0] / p[0].scale_
        b = float(p[1].intercept_[0] - w @ p[0].mean_)
        g = torch.Generator().manual_seed(7000 + seed + (0 if domain == 'geographic' else 1))
        wr = torch.randn(768, generator=g).numpy().astype(np.float64)
        wr /= np.linalg.norm(wr)
        la_r = X[A] @ wr
        br = -float(la_r.mean())
        out[domain] = {'real': (w, b, float(np.std(X[A] @ w + b)), C, auc),
                       'random': (wr, br, float(np.std(la_r + br)), None, None),
                       'X_A': torch.tensor(X[A], device=device)}
    return out


def finetune(variant, seed, T, owt_train, device):
    sae, _ = fm.load_sae(uninterrupted_checkpoint(LAYER, ARCH, seed), device)
    sae.train()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    lam = 0.0 if variant == 'reconstruction_only' else LAMBDA
    kind = 'random' if variant == 'random_teacher' else 'real'
    tt = {d: tuple(torch.tensor(v, dtype=torch.float32, device=device) if i < 2 else v
                   for i, v in enumerate(T[d][kind][:3])) for d in DOMAINS}
    log = []
    for step in range(1, STEPS + 1):
        # identical batch indices for every variant of the same seed
        idx = {d: torch.randint(len(T[d]['X_A']), (GOT_PER_DOMAIN,), generator=gen) for d in DOMAINS}
        oidx = torch.randint(len(owt_train), (OWT_BATCH,), generator=gen)
        xg = torch.cat([T[d]['X_A'][idx[d].to(device)] for d in DOMAINS])
        xo = owt_train[oidx].to(device)
        rg, _, _ = sae(xg)
        ro, _, _ = sae(xo)
        recon = 0.5 * F.mse_loss(rg, xg) + 0.5 * F.mse_loss(ro, xo)
        truth = []
        for i, d in enumerate(DOMAINS):
            w, b, s = tt[d]
            sl = slice(i * GOT_PER_DOMAIN, (i + 1) * GOT_PER_DOMAIN)
            truth.append(F.mse_loss((rg[sl] @ w + b) / s, (xg[sl] @ w + b) / s))
        truth = torch.stack(truth).mean()
        loss = recon + lam * truth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CLIP)
        opt.step()
        sae.normalize_decoder()
        if step in (1, 100, 200, 300, 400, 500):
            log.append({'variant': variant, 'seed': seed, 'step': step, 'recon': recon.item(), 'teacher_loss': truth.item(),
                        'total': loss.item()})
    return sae, log


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--train', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        sys.exit(0 if prepare() else 1)
    checks = json.loads((OUT / 'split_checks.json').read_text())
    prereg = json.loads((OUT / 'preregistration.json').read_text())
    assert all(c['passes'] for c in checks.values()), 'split checks failed'
    assert sha(SPLIT_CSV) == prereg['split_file_sha256'], 'split file changed since pre-registration'
    plan = {(v, s): CKPT / f'{v}_seed{s}' / 'finetuned_checkpoint.pt' for v in VARIANTS for s in SEEDS}
    clash = [str(p) for p in plan.values() if p.exists()] + [str(OUT / 'training_log.csv')] * (OUT / 'training_log.csv').exists()
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    bases = {s: uninterrupted_checkpoint(LAYER, ARCH, s) for s in SEEDS}
    base_sha = {s: sha(p) for s, p in bases.items()}
    if a.dry_run:
        for k, p in plan.items():
            print(k, '->', p.relative_to(REPO))
        T = teachers(42, 'cpu')
        for d in DOMAINS:
            print(d, 'teacher C', T[d]['real'][3], 'teacher val AUROC (original)', round(T[d]['real'][4], 4),
                  'A rows', len(T[d]['X_A']))
        print('base checkpoints:', {s: str(p.relative_to(REPO)) for s, p in bases.items()})
        print(f'{len(plan)} fine-tunes planned; nothing trained.')
        return
    device = 'cuda'
    sys.path.insert(0, str(REPO / 'scripts/openwebtext'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('o3', REPO / 'scripts/openwebtext/03_training_sae.py')
    o3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o3)
    owt_train = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    logs, info = [], []
    t0 = time.time()
    for seed in SEEDS:
        T = teachers(seed, device)
        for v in VARIANTS:
            sae, log = finetune(v, seed, T, owt_train, device)
            logs += log
            p = plan[(v, seed)]
            p.parent.mkdir(parents=True, exist_ok=False)
            torch.save({'model_state_dict': {k: t.detach().cpu() for k, t in sae.state_dict().items()},
                        'config': {'architecture': ARCH, 'target_l0': 128, 'layer': LAYER, 'seed': seed, 'variant': v,
                                   'lambda': 0.0 if v == 'reconstruction_only' else LAMBDA, 'lr': LR, 'steps': STEPS,
                                   'procedure': 'truth_aware_pilot_2026-11-07'},
                        'source_checkpoint': bases[seed].relative_to(REPO).as_posix(),
                        'source_checkpoint_sha256': base_sha[seed]}, p)
            info.append({'variant': v, 'seed': seed, 'checkpoint': p.relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(p)})
            print(f'DONE {v} seed {seed} ({time.time() - t0:.0f}s)', flush=True)
    assert all(sha(p) == base_sha[s] for s, p in bases.items()), 'a base checkpoint changed!'
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    (OUT / 'training_info.json').write_text(json.dumps({'script_sha256': sha(__file__), 'runs': info,
                                                        'base_checkpoints_unchanged': True,
                                                        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))


if __name__ == '__main__':
    main()
