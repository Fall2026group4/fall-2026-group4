#!/usr/bin/env python3
"""E2E-INSPIRED SAE fine-tuning at GPT-2 layer 6 (NOT an exact reproduction of Braun et al. 2024, arXiv:2405.12241).

Differences from the paper (by design, approved): fine-tunes existing local SAEs for 500 steps (Method A's budget)
instead of training from scratch on ~410M tokens; KEEPS the local reconstruction term; keeps each architecture's own
sparsity penalty/controller instead of phi*L1 with phi = lambda/d; KL is KL(p_original || p_spliced); BOS is never
replaced; downstream terms are variance-normalised per layer.

Methods (nested; identical local batches and RNG to Method A):
  A (existing)  L_A = 0.5*MSE(GoT-A rows) + 0.5*MSE(OWT rows) + c*S(z)        [script 14 / pilot TopK]
  B             L_B = L_A + alpha * 0.5*(KL_OWT + KL_GoT)
  C             L_C = L_B + beta  * 0.5*(DS_OWT + DS_GoT),  DS = mean_{k=7..11} NMSE(resid_pre_k spliced vs original)
KL_x = mean over spliced positions of sum_v p_orig(v) [log p_orig(v) - log p_spliced(v)], with GPT-2 FROZEN and the
layer-6 SAE output replacing blocks.6.hook_resid_pre at every non-BOS position. Per step: 8 OWT TRAIN sequences
(128 tokens) + the full statements of the step's 128 GoT-A rows (no labels used). alpha, beta: train-only
scale-matching at step 0 (see --prepare). Adam lr 1e-5, 500 steps, clip 10, decoder renorm.

Run from repository root, in order:
    python scripts/geometry_of_truth/17_e2e_inspired_finetune.py --prepare   # checks, alpha/beta, pre-registration
    python scripts/geometry_of_truth/17_e2e_inspired_finetune.py --dry-run
    python scripts/geometry_of_truth/17_e2e_inspired_finetune.py --train     # repro check first, then 18 runs
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S14 = load('scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py', 's14')
P11 = S14.P11
LAYER, SEEDS, ARCHS, METHODS = 6, (42, 1042, 2042), ('relu', 'topk', 'jumprelu'), ('B', 'C')
DS_LAYERS = tuple(range(7, 12))
HOOK = f'blocks.{LAYER}.hook_resid_pre'
DS_HOOKS = [f'blocks.{k}.hook_resid_pre' for k in DS_LAYERS]
N_OWT_SEQ, SCALE_BATCHES, MONITOR_EVERY = 8, 20, 50
OUT = REPO / 'results/geometry_of_truth/e2e_inspired_layer6'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/e2e_inspired_layer6'
METHOD_A = {('topk', s): Path(str(S14.PILOT_TOPK).format(s=s)) for s in SEEDS}
METHOD_A.update({(a, s): S14.CKPT / f'{a}_got_replay_seed{s}' / 'finetuned_checkpoint.pt' for a in ('relu', 'jumprelu') for s in SEEDS})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_fingerprint(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ data
def load_data(model, device):
    """GoT-A activations + tokens (aligned with script 14's X_A order) and OWT TRAIN token sequences."""
    X_A, T_A = {}, {}
    for d in P11.DOMAINS:
        X, m = P11.load_domain(d)
        A = (m.half == 'A').to_numpy()
        X_A[d] = torch.tensor(X[A], device=device)
        stmts = []
        csv = {n: pd.read_csv(REPO / f'data/got/{n}.csv') for n in P11.DOMAINS[d]}
        for r in m[A].itertuples():
            stmts.append(csv[r.dataset].statement.iloc[int(r.row_id)])
        T_A[d] = [model.to_tokens(s, prepend_bos=True, move_to_device=False)[0] for s in stmts]
    seqs = []
    for p in sorted((REPO / 'data/activations/gpt2_small/train').glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        for i, n in enumerate(c['lengths']):
            seqs.append(c['tokens'][i, :int(n)])
    return X_A, T_A, seqs


def pad(seqs):
    lens = torch.tensor([len(s) for s in seqs])
    toks = torch.full((len(seqs), int(lens.max())), 50256, dtype=torch.long)
    for i, s in enumerate(seqs):
        toks[i, :len(s)] = s
    return toks, lens


def e2e_terms(model, sae, seqs, device, need_ds, grad=True):
    """KL(p_orig || p_spliced) mean over spliced positions, and mean downstream NMSE over blocks 7..11 resid_pre."""
    toks, lens = pad(seqs)
    toks = toks.to(device)
    ar = torch.arange(toks.shape[1], device=device)[None]
    mask = (ar >= 1) & (ar < lens.to(device)[:, None])
    with torch.no_grad():
        lo, cache = model.run_with_cache(toks, names_filter=lambda n: n in DS_HOOKS)
        lo = lo.float().log_softmax(-1)
    store = {}

    def h6(r, hook):
        r = r.clone()
        r[mask] = sae(r[mask])[0]
        return r

    def keep(r, hook):
        store[hook.name] = r
        return r
    hooks = [(HOOK, h6)] + ([(n, keep) for n in DS_HOOKS] if need_ds else [])
    with torch.set_grad_enabled(grad):
        ls = model.run_with_hooks(toks, fwd_hooks=hooks).float().log_softmax(-1)
        kl = (lo.exp() * (lo - ls)).sum(-1)[mask].mean()
        ds = None
        if need_ds:
            terms = []
            for n in DS_HOOKS:
                a, ah = cache[n][mask].float(), store[n][mask].float()
                terms.append((ah - a).square().sum() / (a - a.mean(0)).square().sum())
            ds = torch.stack(terms).mean()
    return kl, ds


def local_batch(X_A, owt_rows, gen, device):
    idx = {d: torch.randint(len(X_A[d]), (S14.GOT_PER_DOMAIN,), generator=gen) for d in P11.DOMAINS}
    oidx = torch.randint(len(owt_rows), (S14.OWT_BATCH,), generator=gen)
    return idx, oidx


# ------------------------------------------------------------------ prepare: checks, alpha/beta, pre-registration
def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'preregistration.json').exists():
        raise SystemExit('preregistration.json exists; refusing to overwrite.')
    device = 'cuda'
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    checks = {'split': S14.verify_split()}
    X_A, T_A, seqs = load_data(model, device)
    # hook check 1: GoT-A statements -> fresh layer-6 final-token residual == cached X_A
    errs = []
    for d in P11.DOMAINS:
        toks, lens = pad(T_A[d][:64])
        _, c = model.run_with_cache(toks.to(device), names_filter=lambda n: n == HOOK)
        fresh = c[HOOK][torch.arange(len(lens)), lens - 1].float()
        errs.append(float((fresh - X_A[d][:64]).abs().max()))
    checks['hook_got_A_fresh_vs_cached_max_abs'] = max(errs)
    # hook check 2: OWT train tokens -> fresh layer-6 residual == cached fp16 train activations (relative)
    c0 = torch.load(sorted((REPO / 'data/activations/gpt2_small/train').glob('chunk_*.pt'))[0], map_location='cpu', weights_only=False)
    n0 = int(c0['lengths'][0])
    _, c = model.run_with_cache(c0['tokens'][:1, :n0].to(device), names_filter=lambda n: n == HOOK)
    cached = c0['activations'][HOOK][0, :n0].float()
    checks['hook_owt_train_fresh_vs_cached_rel'] = float((c[HOOK][0].cpu().float() - cached).abs().max() / cached.abs().max())
    # KL direction / identity / downstream checks with an identity "SAE"
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x.clone(), None, None
    kl_id, ds_id = e2e_terms(model, Identity(), seqs[:4], device, True, grad=False)
    checks['identity_splice_kl'], checks['identity_splice_ds'] = float(kl_id), float(ds_id)
    p = torch.tensor([[0.7, 0.2, 0.1]]).log()
    q = torch.tensor([[0.2, 0.5, 0.3]]).log()
    manual = float((0.7 * math.log(0.7 / 0.2) + 0.2 * math.log(0.2 / 0.5) + 0.1 * math.log(0.1 / 0.3)))
    checks['kl_formula_is_KL(p_orig||p_spliced)'] = abs(float((p.exp() * (p - q)).sum()) - manual) < 1e-6
    # gradient check: SAE gets finite non-zero gradients, GPT-2 gets none; GPT-2 unchanged
    fp0 = model_fingerprint(model)
    sae, _ = fm.load_sae(uninterrupted_checkpoint(LAYER, 'topk', 42), device)
    sae.train()
    kl, ds = e2e_terms(model, sae, seqs[:2], device, True)
    (kl + ds).backward()
    gn = sum(float(q.grad.norm()) for q in sae.parameters() if q.grad is not None)
    checks['sae_grad_norm'] = gn
    checks['gpt2_params_with_grad'] = sum(int(q.grad is not None) for q in model.parameters())
    checks['gpt2_unchanged_after_backward'] = model_fingerprint(model) == fp0
    # downstream NMSE check: independent numpy recomputation over all 5 layers vs e2e_terms
    with torch.no_grad():
        toks, lens = pad(seqs[:2])
        toks = toks.to(device)
        ar = torch.arange(toks.shape[1], device=device)[None]
        mask = (ar >= 1) & (ar < lens.to(device)[:, None])
        _, co = model.run_with_cache(toks, names_filter=lambda n: n in DS_HOOKS)
        st = {}

        def h6(r, hook):
            r = r.clone()
            r[mask] = sae(r[mask])[0]
            return r

        def keep(r, hook):
            st[hook.name] = r.clone()
            return r
        model.run_with_hooks(toks, fwd_hooks=[(HOOK, h6)] + [(n, keep) for n in DS_HOOKS])
        per = []
        for n in DS_HOOKS:
            a_, ah_ = co[n][mask].double().cpu().numpy(), st[n][mask].double().cpu().numpy()
            per.append(((ah_ - a_) ** 2).sum() / ((a_ - a_.mean(0)) ** 2).sum())
        _, ds_fn = e2e_terms(model, sae, seqs[:2], device, True, grad=False)
        checks['ds_numpy_mean_over_layers'] = float(np.mean(per))
        checks['ds_e2e_terms'] = float(ds_fn)
        checks['ds_rel_diff'] = abs(float(ds_fn) - float(np.mean(per))) / float(np.mean(per))
    # sparsity controllers: starting coefficients from the bases (same controller as Method A)
    checks['controller'] = {'target': S14.TARGET, 'adjust_every': S14.ADJUST_EVERY, 'adjust_rate': S14.ADJUST_RATE,
                            'ema_on': 'OWT local rows', 'penalty_on': 'local batch rows (as Method A)'}
    checks['start_coefficients'] = {f'{a}_seed{s}': (None if a == 'topk' else float(torch.load(
        uninterrupted_checkpoint(LAYER, a, s), weights_only=False)['validation_metrics']['sparsity_coefficient']))
        for a in ARCHS for s in SEEDS}
    ok = (checks['split']['passes'] and checks['hook_got_A_fresh_vs_cached_max_abs'] < 1e-3
          and checks['hook_owt_train_fresh_vs_cached_rel'] < 1e-2 and checks['identity_splice_kl'] == 0.0
          and checks['identity_splice_ds'] == 0.0 and checks['kl_formula_is_KL(p_orig||p_spliced)']
          and np.isfinite(gn) and gn > 0 and checks['gpt2_params_with_grad'] == 0 and checks['gpt2_unchanged_after_backward']
          and checks['ds_rel_diff'] < 1e-4)
    checks['passes'] = bool(ok)
    # alpha / beta: train-only scale matching at step 0, per base SAE
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_rows = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    coefs = {}
    for a in ARCHS:
        for s in SEEDS:
            sae, _ = fm.load_sae(uninterrupted_checkpoint(LAYER, a, s), device)
            gen = torch.Generator().manual_seed(s + 9000)
            rec, kls, dss = [], [], []
            for _ in range(SCALE_BATCHES):
                idx, oidx = local_batch(X_A, owt_rows, gen, device)
                xg = torch.cat([X_A[d][idx[d].to(device)] for d in P11.DOMAINS])
                xo = owt_rows[oidx].to(device)
                with torch.no_grad():
                    rec.append(float(0.5 * F.mse_loss(sae(xg)[0], xg) + 0.5 * F.mse_loss(sae(xo)[0], xo)))
                    sidx = torch.randint(len(seqs), (N_OWT_SEQ,), generator=gen)
                    k1, d1 = e2e_terms(model, sae, [seqs[i] for i in sidx], device, True, grad=False)
                    gs = [T_A[d][i] for d in P11.DOMAINS for i in idx[d].tolist()]
                    k2, d2 = e2e_terms(model, sae, gs, device, True, grad=False)
                kls.append(0.5 * float(k1 + k2))
                dss.append(0.5 * float(d1 + d2))
            coefs[f'{a}_seed{s}'] = {'median_local_recon': float(np.median(rec)), 'median_kl': float(np.median(kls)),
                                     'median_ds': float(np.median(dss)), 'alpha': float(np.median(rec) / np.median(kls)),
                                     'beta': float(np.median(rec) / np.median(dss))}
            print(a, s, coefs[f'{a}_seed{s}'], flush=True)
    (OUT / 'pretraining_checks.json').write_text(json.dumps(checks, indent=2))
    (OUT / 'alpha_beta.json').write_text(json.dumps(coefs, indent=2))
    if not ok:
        raise SystemExit(f'PRE-TRAINING CHECKS FAILED: {json.dumps(checks, indent=2)}')
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'label': 'E2E-INSPIRED fine-tuning of local SAEs; NOT an exact reproduction of Braun et al. 2024 (arXiv:2405.12241)',
        'split_file_sha256': checks['split']['sha256'],
        'budget': {'steps': S14.STEPS, 'lr': S14.LR, 'optimizer': 'Adam', 'grad_clip': S14.CLIP,
                   'local_batch': '64+64 GoT-A rows + 128 OWT train rows (identical to Method A)',
                   'e2e_batch': f'{N_OWT_SEQ} OWT train sequences (128 tokens) + full statements of the 128 GoT-A rows'},
        'losses': {'B': 'L_A + alpha*0.5*(KL_OWT+KL_GoT)', 'C': 'L_B + beta*0.5*(DS_OWT+DS_GoT)',
                   'KL': 'mean over spliced (non-BOS) positions of KL(p_orig || p_spliced), GPT-2 frozen',
                   'DS': 'mean over blocks.7..11.hook_resid_pre of sum||a_spliced-a||^2 / sum||a-mean(a)||^2'},
        'alpha_beta_rule': f'per base SAE: alpha = median(local recon)/median(KL), beta = median(local recon)/median(DS) '
                           f'over {SCALE_BATCHES} TRAIN batches at step 0 (values in alpha_beta.json)',
        'matrix': {'existing': 'base x 9, Method A x 9', 'new': 'B x 9, C x 9 (relu/topk/jumprelu x seeds 42/1042/2042)',
                   'repro_check': 'alpha=beta=0 TopK seed 42 must be bit-identical to Method A TopK seed 42; stop otherwise',
                   'new_runs': 18, 'saes_evaluated': 36},
        'evaluation': 'validation only (scripts/shared/layer6_faithfulness_eval.py): OWT EV/MSE/cosine/dead, L0 on OWT, '
                      'GoT and RepE separately, downstream NMSE layers 7-11, OWT splice delta CE + KL, GoT statement KL '
                      '+ cities preference, RepE final KL + persona-shift cosine (splicing, not steering), GoT B-group '
                      'logistic/MLP/mass-mean gaps, decodability diagnostic, RepE logistic/mass-mean gaps + direction '
                      'fidelity. No validation/test data in training or selection; no checkpoint selection.',
        'criteria': {
            'unit': 'per architecture, for B and C separately, paired by seed vs Method A; "consistent" = better on all '
                    '3 seeds AND mean paired difference > 2 x its sd',
            'P1': 'OWT validation splice KL lower than Method A (consistent)',
            'P2': 'GoT validation statement KL lower than Method A (consistent)',
            'H_truth': 'GoT geographic B-MLP gap lower than Method A (consistent)',
            'guards': ['OWT-validation L0 within +-5 % of the base SAE (every seed)',
                       'OWT validation EV >= Method A - 0.02 (every seed)',
                       'cities preference change >= Method A - 0.018 (mean)',
                       'RepE mass-mean gap <= Method A + 0.02 (mean)',
                       'GoT geographic B-logistic gap <= Method A + 0.02 (mean)'],
            'behavioural_success': 'P1 AND P2 AND all guards', 'truth_success': 'behavioural_success AND H_truth',
            'C_vs_B': 'descriptive (downstream NMSE expected lower for C)'},
        'longer_budget_rule': {
            'note': 'flag only; a longer run is a separate experiment needing approval; never run automatically',
            'flag_if': 'for an architecture x method: behavioural_success is NOT met AND in >= 2 of 3 seeds the TRAIN '
                       'monitor KL (fixed 8 OWT train sequences, every 50 steps) is (i) >= 5 % below its step-0 value at '
                       'step 500 and (ii) still falling: KL(500) <= 0.98 * KL(400)'},
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    files = ['scripts/geometry_of_truth/17_e2e_inspired_finetune.py', 'scripts/geometry_of_truth/18_e2e_inspired_evaluation.py',
             'scripts/shared/layer6_faithfulness_eval.py', P11.SPLIT_CSV.relative_to(REPO).as_posix(),
             'results/geometry_of_truth/e2e_inspired_layer6/preregistration.json',
             'results/geometry_of_truth/e2e_inspired_layer6/alpha_beta.json']
    (OUT / 'pretraining_hashes.json').write_text(json.dumps({'recorded': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             'sha256': {f: sha(REPO / f) for f in files}}, indent=2))
    print(json.dumps(checks, indent=2))


# ------------------------------------------------------------------ training
def train_one(arch, seed, method, alpha, beta, model, X_A, T_A, seqs, owt_rows, monitor, device):
    """Method A local loop of script 14 (identical op order and RNG) + E2E terms when alpha/beta > 0."""
    sae, ck = fm.load_sae(uninterrupted_checkpoint(LAYER, arch, seed), device)
    sae.train()
    coef = float(ck['validation_metrics'].get('sparsity_coefficient', 0.0)) if arch != 'topk' else 0.0
    coef0 = coef
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    gen_e2e = torch.Generator().manual_seed(seed + 5000)
    opt = torch.optim.Adam(sae.parameters(), lr=S14.LR)
    ema, log = None, []
    use_e2e = alpha > 0 or beta > 0
    for step in range(0, S14.STEPS + 1):
        if step % MONITOR_EVERY == 0 and use_e2e:
            kmo, dmo = e2e_terms(model, sae, monitor, device, True, grad=False)
            log.append({'step': step, 'monitor_kl_owt_train': float(kmo), 'monitor_ds_owt_train': float(dmo)})
        if step == 0:
            continue
        idx = {d: torch.randint(len(X_A[d]), (S14.GOT_PER_DOMAIN,), generator=gen) for d in P11.DOMAINS}
        oidx = torch.randint(len(owt_rows), (S14.OWT_BATCH,), generator=gen)
        xg = torch.cat([X_A[d][idx[d].to(device)] for d in P11.DOMAINS])
        xo = owt_rows[oidx].to(device)
        rg, zg, pg = sae(xg)
        ro, zo, po = sae(xo)
        loss = 0.5 * F.mse_loss(rg, xg) + 0.5 * F.mse_loss(ro, xo)
        z_all, p_all = torch.cat([zg, zo]), torch.cat([pg, po])
        if arch == 'relu':
            loss = loss + coef * z_all.sum(-1).mean()
        elif arch == 'jumprelu':
            loss = loss + coef * torch.sigmoid((p_all - sae.log_threshold.exp()) / S14.TEMPERATURE).sum(-1).mean()
        if use_e2e:
            sidx = torch.randint(len(seqs), (N_OWT_SEQ,), generator=gen_e2e)
            k1, d1 = e2e_terms(model, sae, [seqs[i] for i in sidx], device, method == 'C')
            k2, d2 = e2e_terms(model, sae, [T_A[d][i] for d in P11.DOMAINS for i in idx[d].tolist()], device, method == 'C')
            loss = loss + alpha * 0.5 * (k1 + k2)
            if method == 'C':
                loss = loss + beta * 0.5 * (d1 + d2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), S14.CLIP)
        opt.step()
        sae.normalize_decoder()
        if arch != 'topk':
            with torch.no_grad():
                l0 = (zo > 0).sum(-1).float().mean().item()
            ema = l0 if ema is None else 0.9 * ema + 0.1 * l0
            if step % S14.ADJUST_EVERY == 0:
                coef *= math.exp(S14.ADJUST_RATE * max(-2.0, min(2.0, math.log(max(ema, 1.0) / S14.TARGET))))
                coef = max(S14.COEF_MIN, min(S14.COEF_MAX, coef))
        if use_e2e and step % MONITOR_EVERY == 0:
            log[-1].update({'loss': float(loss), 'batch_kl': float(0.5 * (k1 + k2)),
                            'batch_ds': float(0.5 * (d1 + d2)) if method == 'C' else float('nan'), 'coef': coef})
    return sae, coef0, coef, log


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--train', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
    bad = [p for p, h in hashes.items() if sha(REPO / p) != h]
    if bad:
        raise SystemExit(f'files changed since pre-registration: {bad}')
    if not json.loads((OUT / 'pretraining_checks.json').read_text())['passes'] or not S14.verify_split()['passes']:
        raise SystemExit('pre-training checks failed')
    coefs = json.loads((OUT / 'alpha_beta.json').read_text())
    runs = [(m, ar, s) for m in METHODS for ar in ARCHS for s in SEEDS]
    paths = {k: CKPT / f'{k[1]}_method{k[0]}_seed{k[2]}' / 'finetuned_checkpoint.pt' for k in runs}
    repro_path = CKPT / 'topk_alpha0_beta0_seed42_REPRO_CHECK' / 'finetuned_checkpoint.pt'
    clash = [str(p) for p in list(paths.values()) + [repro_path] if p.exists()]
    missing = [str(p) for p in METHOD_A.values() if not p.exists()]
    if clash or missing:
        raise SystemExit(f'refusing: clash={clash} missing={missing}')
    if a.dry_run:
        for k, p in paths.items():
            c = coefs[f'{k[1]}_seed{k[2]}']
            print(f'{str(k):26s} alpha={c["alpha"]:.3f} beta={c["beta"] if k[0] == "C" else 0:.3f} -> {p.relative_to(REPO)}')
        print(f'repro check -> {repro_path.relative_to(REPO)}; {len(runs)} runs planned; nothing trained.')
        return
    device = 'cuda'
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fp0 = model_fingerprint(model)
    X_A, T_A, seqs = load_data(model, device)
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_rows = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    gm = torch.Generator().manual_seed(99)
    monitor = [seqs[i] for i in torch.randint(len(seqs), (N_OWT_SEQ,), generator=gm)]
    protected = {str(p): sha(p) for p in list(METHOD_A.values()) + [uninterrupted_checkpoint(LAYER, ar, s) for ar in ARCHS for s in SEEDS]}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    # 1) reproduction check (alpha = beta = 0) must equal Method A TopK seed 42 bit-for-bit
    sae, _, _, _ = train_one('topk', 42, 'B', 0.0, 0.0, model, X_A, T_A, seqs, owt_rows, monitor, device)
    repro_path.parent.mkdir(parents=True, exist_ok=False)
    torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in sae.state_dict().items()}, 'note': 'repro check only'}, repro_path)
    ref = torch.load(METHOD_A[('topk', 42)], weights_only=False)['model_state_dict']
    diff = max(float((sae.state_dict()[k].cpu().float() - ref[k].float()).abs().max()) for k in ref)
    print('REPRO CHECK max |param diff| vs Method A TopK seed 42:', diff, flush=True)
    if diff != 0.0:
        (OUT / 'training_info.json').write_text(json.dumps({'repro_max_abs_diff': diff, 'stopped': True}, indent=2))
        raise SystemExit('Reproduction check FAILED; stopping before any E2E run.')
    # 2) the 18 runs
    info, logs = [], []
    for k in runs:
        m, ar, s = k
        c = coefs[f'{ar}_seed{s}']
        alpha, beta = c['alpha'], (c['beta'] if m == 'C' else 0.0)
        t1 = time.time()
        sae, c0, c1, log = train_one(ar, s, m, alpha, beta, model, X_A, T_A, seqs, owt_rows, monitor, device)
        paths[k].parent.mkdir(parents=True, exist_ok=False)
        torch.save({'model_state_dict': {kk: v.detach().cpu() for kk, v in sae.state_dict().items()},
                    'config': {'architecture': ar, 'target_l0': 128, 'layer': LAYER, 'seed': s, 'method': m, 'alpha': alpha,
                               'beta': beta, 'lr': S14.LR, 'steps': S14.STEPS, 'procedure': 'e2e_inspired_layer6_2026-11-07'},
                    'validation_metrics': {'sparsity_coefficient': c1}, 'sparsity_coefficient_start': c0,
                    'source_checkpoint': uninterrupted_checkpoint(LAYER, ar, s).relative_to(REPO).as_posix()}, paths[k])
        logs += [{'method': m, 'architecture': ar, 'seed': s, **r} for r in log]
        info.append({'method': m, 'architecture': ar, 'seed': s, 'alpha': alpha, 'beta': beta, 'coef_start': c0, 'coef_end': c1,
                     'seconds': round(time.time() - t1, 1), 'checkpoint': paths[k].relative_to(REPO).as_posix()})
        print(f'DONE method {m} {ar} seed {s} ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)', flush=True)
        pd.DataFrame(logs).to_csv(OUT / 'training_monitor.csv', index=False)
    gpt2_unchanged = model_fingerprint(model) == fp0
    protected_ok = all(sha(p) == h for p, h in protected.items())
    (OUT / 'training_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'repro_max_abs_diff': diff, 'repro_bit_identical': True, 'runs': info,
        'gpt2_parameters_unchanged': gpt2_unchanged, 'base_and_method_a_checkpoints_unchanged': protected_ok,
        'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    assert gpt2_unchanged and protected_ok, 'GPT-2 or a protected checkpoint changed!'


if __name__ == '__main__':
    main()
