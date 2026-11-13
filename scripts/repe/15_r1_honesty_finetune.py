#!/usr/bin/env python3
"""R1 at GPT-2 Small layer 6: RepE honesty-preservation fine-tuning of the 9 uninterrupted OWT SAEs.

Run from repository root, in order:
    python scripts/repe/15_r1_honesty_finetune.py --prepare   # verify split, write preregistration.json + hashes
    python scripts/repe/15_r1_honesty_finetune.py --dry-run   # paths, start coefficients, lambda values
    python scripts/repe/15_r1_honesty_finetune.py --train     # 27 runs + 1 reproduction check

Pre-registration: results/repe/r1_honesty_preservation_layer6/README.md (§2, and the pre-training addendum §10).
GPT-2 is not loaded (its activations are cached); only SAE parameters are trained.

Arms (all start from the base SAE; identical batch RNG; loss on 32 A statement pairs = 64 RepE rows + 128 OWT rows):
  r_recon        0.5 * MSE(RepE rows) + 0.5 * MSE(OWT rows)                                  (control)
  r_honesty      r_recon + lambda * L_pair,  L_pair = mean_i ||(xh_H,i - xh_U,i) - (x_H,i - x_U,i)||^2 / D_H
                 (same statement, honest vs untruthful persona); D_H = mean ||x_H - x_U||^2 over all A pairs
  r_samepersona  r_recon + lambda * L_same, same form on SAME-persona, DIFFERENT-statement pairs of the same batch:
                 pair i = (row of statement i, row of statement (i+1) mod 32), honest rows for even i, untruthful rows
                 for odd i; D_S = mean ||x_a - x_b||^2 over the analogous full-A pairing (statement j with j+1 mod n_A,
                 persona by parity of j)
lambda: per SAE and arm, at initialisation on all A rows: lambda * L(all A) = 0.5 * MSE_RepE(all A). No sweep.
Protocol = Method A (scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py): Adam lr 1e-5, 500 steps, clip
10, decoder renormalisation, no checkpoint selection; ReLU/JumpReLU keep their penalty on all rows with the
multiplicative controller continued from the base coefficient (target 128, every 25 steps, rate 0.5, EMA L0 on OWT
rows only); TopK: L0 fixed by k. Torch seed and batch generator = SAE seed.
Only RepE train-A rows are read for training (asserted). B, validation, cross-wording and test rows are never used.
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


S14 = load('scripts/repe/14_prepare_r1_r2_splits_and_wording.py', 's14')
LAYER, SEEDS, ARCHS = 6, (42, 1042, 2042), ('relu', 'topk', 'jumprelu')
ARMS = ('r_recon', 'r_honesty', 'r_samepersona')
LR, STEPS, CLIP, PAIR_BATCH, OWT_BATCH = 1e-5, 500, 10.0, 32, 128
TARGET, ADJUST_EVERY, ADJUST_RATE, TEMPERATURE = 128, 25, 0.5, 0.1
COEF_MIN, COEF_MAX = 1e-7, 100.0
OUT = S14.OUT
CKPT = REPO / 'results/repe/sae_finetuning/r1_honesty_preservation_layer6'
CACHE = REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt'
VERIFIED = ('scripts/repe/14_prepare_r1_r2_splits_and_wording.py', 'scripts/repe/15_r1_honesty_finetune.py',
            'scripts/repe/16_r1_evaluation.py', 'data/repe/splits/r1_train_AB_groups.csv',
            'data/repe/persona_pairs.csv', 'results/repe/r1_honesty_preservation_layer6/preregistration.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    runs = [dict(name=f'{arm}_{a}_seed{s}', arm=arm, arch=a, seed=s, base=uninterrupted_checkpoint(LAYER, a, s))
            for arm in ARMS for a in ARCHS for s in SEEDS]
    runs.append(dict(name='r_recon_topk_seed42_REPRO_CHECK', arm='r_recon', arch='topk', seed=42,
                     base=uninterrupted_checkpoint(LAYER, 'topk', 42)))
    for r in runs:
        r['path'] = CKPT / r['name'] / 'finetuned_checkpoint.pt'
    return runs


def load_A():
    """Honest / untruthful layer-6 activations of the train-A statements, row-aligned by statement (sorted id)."""
    c = torch.load(CACHE, map_location='cpu', weights_only=True)
    sid, split, persona = np.array(c['statement_ids']), np.array(c['splits']), np.array(c['personas'])
    ab = pd.read_csv(S14.SPLIT_CSV)
    a_ids = np.sort(ab.statement_id[ab.half == 'A'].to_numpy())
    X = c['activations'][LAYER].float()
    idx_H, idx_U = [], []
    for i in a_ids:
        h = np.flatnonzero((sid == i) & (persona == 'honest'))
        u = np.flatnonzero((sid == i) & (persona == 'untruthful'))
        assert len(h) == 1 and len(u) == 1 and split[h[0]] == 'train' and split[u[0]] == 'train'
        idx_H.append(h[0])
        idx_U.append(u[0])
    used = set(sid[idx_H + idx_U])
    assert used == set(a_ids) and not (used & set(ab.statement_id[ab.half == 'B'])), 'non-A row reached training'
    return X[idx_H], X[idx_U], a_ids


def same_persona_pairs(xh, xu):
    """Pair i = (statement i, statement i+1 mod n); honest rows for even i, untruthful for odd i."""
    n = len(xh)
    j = (torch.arange(n, device=xh.device) + 1) % n
    even = (torch.arange(n, device=xh.device) % 2 == 0)[:, None]
    return even, j


def pair_term(arm, rh, ru, xh, xu, D):
    if arm == 'r_honesty':
        return ((rh - ru) - (xh - xu)).square().sum(-1).mean() / D
    even, j = same_persona_pairs(xh, xu)
    ra, rb = torch.where(even, rh, ru), torch.where(even, rh[j], ru[j])
    xa, xb = torch.where(even, xh, xu), torch.where(even, xh[j], xu[j])
    return ((ra - rb) - (xa - xb)).square().sum(-1).mean() / D


def denominators(XH, XU):
    even, j = same_persona_pairs(XH, XU)
    xa, xb = torch.where(even, XH, XU), torch.where(even, XH[j], XU[j])
    return {'r_honesty': float((XH - XU).square().sum(-1).mean()), 'r_samepersona': float((xa - xb).square().sum(-1).mean())}


@torch.no_grad()
def init_lambda(sae, arm, XH, XU, D):
    if arm == 'r_recon':
        return 0.0, float('nan'), float('nan')
    rh, ru = sae(XH)[0], sae(XU)[0]
    mse = 0.5 * (F.mse_loss(rh, XH) + F.mse_loss(ru, XU))
    lp = pair_term(arm, rh, ru, XH, XU, D[arm])
    return float(0.5 * mse / lp), float(mse), float(lp)


def finetune(run, XH, XU, D, owt_train, device, steps=STEPS):
    sae, ck = fm.load_sae(run['base'], device)
    arch, seed, arm = run['arch'], run['seed'], run['arm']
    lam, mse0, lp0 = init_lambda(sae.eval(), arm, XH, XU, D)
    sae.train()
    coef = float(ck['validation_metrics'].get('sparsity_coefficient', 0.0)) if arch != 'topk' else 0.0
    coef0 = coef
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    ema, trace = None, []
    for step in range(1, steps + 1):
        idx = torch.randint(len(XH), (PAIR_BATCH,), generator=gen)      # identical draw order in every arm
        oidx = torch.randint(len(owt_train), (OWT_BATCH,), generator=gen)
        xh, xu = XH[idx.to(device)], XU[idx.to(device)]
        xr = torch.cat([xh, xu])
        xo = owt_train[oidx].to(device)
        rr, zr, pr = sae(xr)
        ro, zo, po = sae(xo)
        rec_repe, rec_owt = F.mse_loss(rr, xr), F.mse_loss(ro, xo)
        loss = 0.5 * rec_repe + 0.5 * rec_owt
        lp = torch.zeros((), device=device)
        if arm != 'r_recon':
            lp = pair_term(arm, rr[:PAIR_BATCH], rr[PAIR_BATCH:], xh, xu, D[arm])
            loss = loss + lam * lp
        z_all, p_all = torch.cat([zr, zo]), torch.cat([pr, po])
        if arch == 'relu':
            loss = loss + coef * z_all.sum(-1).mean()
        elif arch == 'jumprelu':
            loss = loss + coef * torch.sigmoid((p_all - sae.log_threshold.exp()) / TEMPERATURE).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CLIP)
        opt.step()
        sae.normalize_decoder()
        with torch.no_grad():
            l0_owt = (zo > 0).sum(-1).float().mean().item()
            l0_repe = (zr > 0).sum(-1).float().mean().item()
        if arch != 'topk':
            ema = l0_owt if ema is None else 0.9 * ema + 0.1 * l0_owt
            if step % ADJUST_EVERY == 0:
                coef *= math.exp(ADJUST_RATE * max(-2.0, min(2.0, math.log(max(ema, 1.0) / TARGET))))
                coef = max(COEF_MIN, min(COEF_MAX, coef))
        if step in (1, 50, 100, 250, 500) or (steps < STEPS and step == steps):
            trace.append({'run': run['name'], 'step': step, 'loss': loss.item(), 'mse_repe': rec_repe.item(),
                          'mse_owt': rec_owt.item(), 'pair_term': lp.item(), 'lambda_x_pair': lam * lp.item(),
                          'coef': coef, 'batch_l0_owt': l0_owt, 'batch_l0_repe': l0_repe,
                          'ema_owt_l0': ema if ema is not None else float('nan')})
    return sae, coef0, coef, lam, mse0, lp0, trace


def prepare():
    for p in (OUT / 'preregistration.json', OUT / 'pretraining_hashes.json'):
        if p.exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
    ver = json.loads((OUT / 'split_verification.json').read_text())
    if not ver['passes'] or ver['split_file_sha256'] != sha(S14.SPLIT_CSV):
        raise SystemExit('split verification missing, failed, or split file changed')
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': 'R1 RepE honesty representation preservation, GPT-2 Small layer 6',
        'documents': ['results/repe/r1_honesty_preservation_layer6/README.md (as drafted, commit d17bb00)',
                      'same README §10 pre-training addendum (clarifications; criteria unchanged)'],
        'split_file': S14.SPLIT_CSV.relative_to(REPO).as_posix(), 'split_file_sha256': sha(S14.SPLIT_CSV),
        'excluded_validation_statements': list(S14.EXCLUDED_VALIDATION),
        'bases': {f'{a}/{s}': Path(uninterrupted_checkpoint(LAYER, a, s)).relative_to(REPO).as_posix() for a in ARCHS for s in SEEDS},
        'base_sha256': {f'{a}/{s}': sha(uninterrupted_checkpoint(LAYER, a, s)) for a in ARCHS for s in SEEDS},
        'arms': {'r_recon': '0.5 MSE(RepE-A rows) + 0.5 MSE(OWT train)',
                 'r_honesty': 'r_recon + lambda * L_pair (honest-vs-untruthful, same statement)',
                 'r_samepersona': 'r_recon + lambda * L_same (same persona, different statements; control)',
                 'reference_only': 'base (9), Method A (relu/jumprelu got_replay, topk = pilot reconstruction-only)'},
        'lambda_rule': 'per SAE and arm, at initialisation on all A rows: lambda * L(all A) = 0.5 * MSE_RepE(all A)',
        'protocol': {'lr': LR, 'steps': STEPS, 'clip': CLIP, 'pair_batch': PAIR_BATCH, 'owt_batch': OWT_BATCH,
                     'optimizer': 'Adam', 'decoder_renormalisation': True, 'checkpoint_selection': None,
                     'sparsity': 'relu/jumprelu: penalty on all rows, controller continued from base coef (target 128, '
                                 'every 25, rate 0.5, EMA L0 on OWT rows); topk: fixed k',
                     'seeds': list(SEEDS), 'torch_and_batch_seed': 'SAE seed'},
        'runs': {'new': 27, 'repro_check': 'r_recon topk seed 42 trained twice; must be bit-identical; excluded'},
        'evaluation': 'scripts/repe/16_r1_evaluation.py: validation (89 statements) + cross-wording validation (89); '
                      'test NOT scored (deferred, user instruction 2026-11-07)',
        'primary_metric': 'B-MLP AUROC gap (768-64-1, 5 inits 42/123/456/789/2026, fitted on ORIGINAL train-B rows, '
                          'applied frozen to reconstructions), validation',
        'criteria': {
            'consistent': 'better on all 3 seeds AND mean paired difference > 2 x sd of paired differences',
            'R1a': 'r_recon B-MLP gap consistently lower than base (validation)',
            'R1b_primary': 'r_honesty B-MLP gap consistently lower than r_recon (validation) AND lower than '
                           'r_samepersona on all 3 seeds',
            'R1_transfer': 'cross-wording: mean paired diff (r_honesty - r_recon) of B-MLP gap < 0 and < 0 on >= 2/3 seeds',
            'guards_vs_own_base': ['OWT validation L0 within +-5 % (every seed)',
                                   'RepE validation-prompt L0 within +-5 % (every seed)',
                                   'OWT validation EV >= base - 0.02 (every seed)',
                                   'OWT dead fraction <= base + 0.05 (every seed)',
                                   'OWT spliced CE increase <= base + 0.01 nats (mean over seeds)',
                                   'GoT geographic B-MLP gap <= base + 0.02 (mean over seeds)'],
            'verdict': 'SUPPORTED = R1b AND R1_transfer AND all guards for r_honesty and r_recon; '
                       'NOT SUPPORTED = R1b fails; INCONCLUSIVE otherwise. r_samepersona guard failures are flagged.'},
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    hashes = {p: sha(REPO / p) for p in VERIFIED}
    hashes['results/repe/r1_honesty_preservation_layer6/README.md'] = sha(OUT / 'README.md')
    (OUT / 'pretraining_hashes.json').write_text(json.dumps(
        {'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'note': 'recorded before any R1 run; README hash is the '
         'pre-training version (it is updated with results later and is not re-verified)', 'sha256': hashes}, indent=2))
    print(json.dumps(hashes, indent=2))


def smoke():
    device = 'cuda'
    XH, XU, a_ids = load_A()
    XH, XU = XH.to(device), XU.to(device)
    D = denominators(XH, XU)
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    for arch in ('relu', 'topk', 'jumprelu'):
        for arm in ARMS:
            r = dict(name=f'SMOKE_{arm}_{arch}', arm=arm, arch=arch, seed=42, base=uninterrupted_checkpoint(LAYER, arch, 42))
            out = finetune(r, XH, XU, D, owt, device, steps=20)
            for row in out[-1]:
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    print('A statements', len(a_ids), 'D', D, 'smoke OK; nothing saved')


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--train', action='store_true')
    g.add_argument('--smoke', action='store_true', help='20 steps, 2 SAEs x 3 arms, nothing saved')
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    if a.smoke:
        return smoke()
    hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
    bad = [p for p in VERIFIED if sha(REPO / p) != hashes[p]]
    if bad:
        raise SystemExit(f'files changed since pre-registration: {bad}')
    runs = plan()
    clash = [str(r['path']) for r in runs if r['path'].exists()] + [str(OUT / 'training_info.json')] * (OUT / 'training_info.json').exists()
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    device = 'cuda'
    XH, XU, a_ids = load_A()
    XH, XU = XH.to(device), XU.to(device)
    D = denominators(XH, XU)
    if a.dry_run:
        for r in runs:
            sae, ck = fm.load_sae(r['base'], device)
            lam, mse0, lp0 = init_lambda(sae.eval(), r['arm'], XH, XU, D)
            print(f"{r['name']:38s} start_coef={ck['validation_metrics'].get('sparsity_coefficient') if r['arch'] != 'topk' else None} "
                  f"lambda={lam:.4g} mse0={mse0:.4g} L0pair={lp0:.4g}")
        print(f'{len(runs)} runs planned; A statements={len(a_ids)}; D={D}; nothing trained.')
        return
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_train = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    base_sha = {str(r['base']): sha(r['base']) for r in runs}
    info, traces = [], []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for r in runs:
        t1 = time.time()
        sae, c0, c1, lam, mse0, lp0, tr = finetune(r, XH, XU, D, owt_train, device)
        traces += tr
        r['path'].parent.mkdir(parents=True, exist_ok=False)
        torch.save({'model_state_dict': {k: t.detach().cpu() for k, t in sae.state_dict().items()},
                    'config': {'architecture': r['arch'], 'target_l0': TARGET, 'layer': LAYER, 'seed': r['seed'],
                               'variant': r['arm'], 'lr': LR, 'steps': STEPS, 'lambda': lam,
                               'procedure': 'r1_honesty_preservation_layer6_2026-11-07'},
                    'validation_metrics': {'sparsity_coefficient': c1},
                    'sparsity_coefficient_start': c0,
                    'source_checkpoint': Path(r['base']).relative_to(REPO).as_posix(),
                    'source_checkpoint_sha256': base_sha[str(r['base'])]}, r['path'])
        info.append({'run': r['name'], 'arm': r['arm'], 'arch': r['arch'], 'seed': r['seed'],
                     'checkpoint': r['path'].relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(r['path']),
                     'coef_start': c0, 'coef_end': c1, 'lambda': lam, 'init_mse_repe_A': mse0, 'init_pair_term_A': lp0,
                     'seconds': round(time.time() - t1, 2)})
        print(f"DONE {r['name']} ({time.time() - t0:.0f}s) lambda={lam:.4g} coef {c0:.4g} -> {c1:.4g}", flush=True)
    new = torch.load(next(r['path'] for r in runs if r['name'].endswith('REPRO_CHECK')), weights_only=False)['model_state_dict']
    old = torch.load(CKPT / 'r_recon_topk_seed42' / 'finetuned_checkpoint.pt', weights_only=False)['model_state_dict']
    repro = max(float((new[k].float() - old[k].float()).abs().max()) for k in old)
    assert all(sha(p) == h for p, h in base_sha.items()), 'a base checkpoint changed!'
    pd.DataFrame(traces).to_csv(OUT / 'training_trace.csv', index=False)
    (OUT / 'training_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'runs': info, 'pair_denominators': D, 'A_statements': len(a_ids),
        'base_checkpoints_unchanged': True, 'repro_max_abs_param_diff': repro, 'repro_bit_identical': repro == 0.0,
        'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print('reproduction check max |param diff|:', repro)


if __name__ == '__main__':
    main()
