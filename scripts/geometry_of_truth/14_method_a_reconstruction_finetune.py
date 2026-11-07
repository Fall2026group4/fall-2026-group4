#!/usr/bin/env python3
"""Method A at layer 6: reconstruction-only fine-tuning on GoT train groups A with OWT replay, for ReLU, TopK and
JumpReLU (seeds 42/1042/2042), plus an OWT-only continued-training control and the P1 JumpReLU base (seed 42).

Run from repository root, in order:
    python scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py --prepare   # verify split, pre-register
    python scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py --dry-run   # paths, coefficients
    python scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py --train     # 17 runs

Reference protocol = the reconstruction-only arm of the truth-aware pilot (script 11): same A/B split, batches
(64 geographic-A + 64 numerical-A + 128 OWT train), loss 0.5*MSE_GoT + 0.5*MSE_OWT, Adam lr 1e-5, 500 steps, clip 10,
decoder renormalisation, same batch RNG, no checkpoint selection. Group B, validation and test are never read here.

Sparsity control (approved): ReLU / JumpReLU keep their training penalty (ReLU: L1 of z; JumpReLU: sigmoid count) on
all rows, with the original multiplicative controller (every 25 steps, rate 0.5, target 128) CONTINUED from the base
checkpoint's saved final coefficient; the controller's EMA L0 is measured on the OWT rows only (the target is defined
on OWT). TopK: no penalty (L0 fixed). Control 'owt_only': identical, but all 256 rows from OWT train.

Runs (17): got_replay {relu, jumprelu} x 3 seeds (6) + got_replay from the P1 JumpReLU base seed 42 (1) + TopK seed-42
reproduction check (1; must reproduce the pilot checkpoint, excluded from results) + owt_only {relu, topk, jumprelu}
x 3 seeds (9). Method A TopK = the existing pilot reconstruction-only checkpoints (reused, not retrained).
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


P11 = load('scripts/geometry_of_truth/11_truth_aware_pilot.py', 'p11')
LAYER, SEEDS, ARCHS = 6, (42, 1042, 2042), ('relu', 'topk', 'jumprelu')
LR, STEPS, CLIP = P11.LR, P11.STEPS, P11.CLIP
GOT_PER_DOMAIN, OWT_BATCH = P11.GOT_PER_DOMAIN, P11.OWT_BATCH
TARGET, ADJUST_EVERY, ADJUST_RATE, TEMPERATURE = 128, 25, 0.5, 0.1
COEF_MIN, COEF_MAX = 1e-7, 100.0
OUT = REPO / 'results/geometry_of_truth/method_a_layer6'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/method_a_layer6'
PILOT_TOPK = P11.CKPT / 'reconstruction_only_seed{s}' / 'finetuned_checkpoint.pt'
P1_JUMPRELU = REPO / 'results/openwebtext/convergence/layer_6/jumprelu_l0_128_p1_seed42/checkpoint_selected.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    runs = []
    for a in ('relu', 'jumprelu'):
        runs += [dict(name=f'{a}_got_replay_seed{s}', arch=a, seed=s, variant='got_replay', base=uninterrupted_checkpoint(LAYER, a, s))
                 for s in SEEDS]
    runs.append(dict(name='jumprelu_p1base_got_replay_seed42', arch='jumprelu', seed=42, variant='got_replay', base=P1_JUMPRELU))
    runs.append(dict(name='topk_got_replay_seed42_REPRO_CHECK', arch='topk', seed=42, variant='got_replay',
                     base=uninterrupted_checkpoint(LAYER, 'topk', 42)))
    for a in ARCHS:
        runs += [dict(name=f'{a}_owt_only_seed{s}', arch=a, seed=s, variant='owt_only', base=uninterrupted_checkpoint(LAYER, a, s))
                 for s in SEEDS]
    for r in runs:
        r['path'] = CKPT / r['name'] / 'finetuned_checkpoint.pt'
    return runs


def verify_split():
    """Re-check the committed A/B split: same file as the pilot, disjoint, balanced, sufficient, B never trained on."""
    pre = json.loads((P11.OUT / 'preregistration.json').read_text())
    res = {'split_file': P11.SPLIT_CSV.relative_to(REPO).as_posix(), 'sha256': sha(P11.SPLIT_CSV),
           'matches_pilot_preregistration': sha(P11.SPLIT_CSV) == pre['split_file_sha256']}
    for domain in P11.DOMAINS:
        _, m = P11.load_domain(domain)
        y = m.label.to_numpy()
        A, B = (m.half == 'A').to_numpy(), (m.half == 'B').to_numpy()
        gA, gB, other = set(m.group_id[A]), set(m.group_id[B]), set(m.group_id[m.split != 'train'])
        res[domain] = {'rows_A': int(A.sum()), 'rows_B': int(B.sum()), 'A_B_overlap': len(gA & gB),
                       'train_vs_val_test_overlap': len((gA | gB) & other), 'label_mean_A': float(y[A].mean()),
                       'label_mean_B': float(y[B].mean()), 'test_rows_in_A_or_B': int(((m.split == 'test') & (A | B)).sum())}
    res['passes'] = bool(res['matches_pilot_preregistration'] and all(
        res[d]['A_B_overlap'] == 0 and res[d]['train_vs_val_test_overlap'] == 0 and res[d]['test_rows_in_A_or_B'] == 0
        and abs(res[d]['label_mean_A'] - .5) < .01 and abs(res[d]['label_mean_B'] - .5) < .01 and min(res[d]['rows_A'], res[d]['rows_B']) >= 500
        for d in P11.DOMAINS))
    return res


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'preregistration.json').exists():
        raise SystemExit('preregistration.json exists; refusing to overwrite.')
    checks = verify_split()
    (OUT / 'split_verification.json').write_text(json.dumps(checks, indent=2))
    if not checks['passes']:
        raise SystemExit('split verification failed')
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reference_protocol': 'reconstruction_only arm of scripts/geometry_of_truth/11_truth_aware_pilot.py',
        'split_file_sha256': checks['sha256'],
        'matrix': {'baselines_evaluated': '9 uninterrupted (relu/topk/jumprelu x 3 seeds) + P1 JumpReLU seed 42',
                   'method_a': 'relu x3, jumprelu x3 (new), topk x3 (existing pilot checkpoints, reused), P1-JumpReLU base x1 (new, descriptive)',
                   'control_owt_only': 'relu/topk/jumprelu x 3 seeds (new)',
                   'repro_check': 'topk seed 42 retrained with this script; must equal the pilot checkpoint; excluded from results',
                   'new_training_runs': 17, 'saes_evaluated': 29},
        'sparsity_control': 'ReLU/JumpReLU: training penalty kept, controller continued from the base coefficient '
                            '(target 128, every 25 steps, rate 0.5, EMA L0 on OWT rows); TopK: none',
        'evaluation': 'validation only; independent probes fitted on ORIGINAL activations of train groups B (logistic '
                      'with C chosen on original validation, MLP x5 seeds, mass-mean); decodability diagnostic; OWT, GoT, '
                      'RepE reconstruction and L0 reported separately; splicing (OWT CE/KL, GoT statement KL, cities '
                      'preference, RepE final KL, persona-shift cosine); RepE frozen probes and direction fidelity. '
                      'No probe, validation or test result is used for training or selection.',
        'success_criteria': {
            'primary_per_architecture': 'GoT geographic B-MLP gap (mean of 5 probe seeds): method_a < paired base AND '
                                        '< paired owt_only on all 3 seeds, mean paired difference > 2 x its sd',
            'secondary': 'same test for the B-logistic gap and for the numerical domain (reported, not required)',
            'guards_per_architecture': [
                'L0 matched: OWT-validation L0 of every method_a seed within +-5 % of its base',
                'OWT splice delta CE <= 1.10 x base on every seed',
                'OWT validation EV >= base - 0.005 on every seed',
                'GoT cities truth-preference change >= base - 0.018 (mean over seeds)',
                'RepE validation mass-mean gap <= base + 0.02 (mean over seeds)'],
            'success': 'primary AND all guards (per architecture); P1 JumpReLU base is descriptive only',
            'cross_architecture_comparison': 'only among L0-matched cells'},
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    hashes = {p: sha(REPO / p) for p in ('scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py',
                                        'scripts/geometry_of_truth/15_method_a_evaluation.py',
                                        P11.SPLIT_CSV.relative_to(REPO).as_posix(),
                                        'results/geometry_of_truth/method_a_layer6/preregistration.json')}
    (OUT / 'pretraining_hashes.json').write_text(json.dumps({'recorded': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             'note': 'recorded before any Method A run', 'sha256': hashes}, indent=2))
    print(json.dumps(checks, indent=2))


def finetune(run, X_A, owt_train, device):
    sae, ck = fm.load_sae(run['base'], device)
    sae.train()
    arch, seed = run['arch'], run['seed']
    coef = float(ck['validation_metrics'].get('sparsity_coefficient', 0.0)) if arch != 'topk' else 0.0
    coef0 = coef
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    ema, trace = None, []
    for step in range(1, STEPS + 1):
        if run['variant'] == 'got_replay':     # identical draw order to the pilot
            idx = {d: torch.randint(len(X_A[d]), (GOT_PER_DOMAIN,), generator=gen) for d in P11.DOMAINS}
            oidx = torch.randint(len(owt_train), (OWT_BATCH,), generator=gen)
            xg = torch.cat([X_A[d][idx[d].to(device)] for d in P11.DOMAINS])
            xo = owt_train[oidx].to(device)
            rg, zg, pg = sae(xg)
            ro, zo, po = sae(xo)
            loss = 0.5 * F.mse_loss(rg, xg) + 0.5 * F.mse_loss(ro, xo)
            z_all, p_all = torch.cat([zg, zo]), torch.cat([pg, po])
        else:
            oidx = torch.randint(len(owt_train), (2 * OWT_BATCH,), generator=gen)
            xo = owt_train[oidx].to(device)
            ro, zo, po = sae(xo)
            loss = F.mse_loss(ro, xo)
            z_all, p_all = zo, po
        if arch == 'relu':
            loss = loss + coef * z_all.sum(-1).mean()
        elif arch == 'jumprelu':
            loss = loss + coef * torch.sigmoid((p_all - sae.log_threshold.exp()) / TEMPERATURE).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CLIP)
        opt.step()
        sae.normalize_decoder()
        if arch != 'topk':
            with torch.no_grad():
                l0 = (zo > 0).sum(-1).float().mean().item()
            ema = l0 if ema is None else 0.9 * ema + 0.1 * l0
            if step % ADJUST_EVERY == 0:
                coef *= math.exp(ADJUST_RATE * max(-2.0, min(2.0, math.log(max(ema, 1.0) / TARGET))))
                coef = max(COEF_MIN, min(COEF_MAX, coef))
        if step in (1, 100, 250, 500):
            trace.append({'run': run['name'], 'step': step, 'loss': loss.item(), 'coef': coef,
                          'ema_owt_l0': ema if ema is not None else float('nan')})
    return sae, coef0, coef, trace


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
    if not verify_split()['passes']:
        raise SystemExit('split verification failed')
    runs = plan()
    clash = [str(r['path']) for r in runs if r['path'].exists()] + [str(OUT / 'training_info.json')] * (OUT / 'training_info.json').exists()
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    missing = [str(r['base']) for r in runs if not Path(r['base']).exists()]
    missing += [str(PILOT_TOPK).format(s=s) for s in SEEDS if not Path(str(PILOT_TOPK).format(s=s)).exists()]
    if missing:
        raise SystemExit(f'missing inputs: {missing}')
    if a.dry_run:
        for r in runs:
            ck = torch.load(r['base'], map_location='cpu', weights_only=False)
            c = ck['validation_metrics'].get('sparsity_coefficient') if r['arch'] != 'topk' else None
            print(f"{r['name']:40s} base={Path(r['base']).relative_to(REPO)} start_coef={c}")
        print(f'{len(runs)} training runs planned (incl. 1 reproduction check); nothing trained.')
        return
    device = 'cuda'
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_train = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    X_A = {}
    for d in P11.DOMAINS:
        X, m = P11.load_domain(d)
        X_A[d] = torch.tensor(X[(m.half == 'A').to_numpy()], device=device)
    base_sha = {str(r['base']): sha(r['base']) for r in runs}
    pilot_sha = {s: sha(str(PILOT_TOPK).format(s=s)) for s in SEEDS}
    info, traces = [], []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for r in runs:
        sae, c0, c1, tr = finetune(r, X_A, owt_train, device)
        traces += tr
        r['path'].parent.mkdir(parents=True, exist_ok=False)
        torch.save({'model_state_dict': {k: t.detach().cpu() for k, t in sae.state_dict().items()},
                    'config': {'architecture': r['arch'], 'target_l0': TARGET, 'layer': LAYER, 'seed': r['seed'],
                               'variant': r['variant'], 'lr': LR, 'steps': STEPS, 'procedure': 'method_a_layer6_2026-11-07'},
                    'validation_metrics': {'sparsity_coefficient': c1},
                    'sparsity_coefficient_start': c0,
                    'source_checkpoint': Path(r['base']).relative_to(REPO).as_posix(),
                    'source_checkpoint_sha256': base_sha[str(r['base'])]}, r['path'])
        info.append({'run': r['name'], 'arch': r['arch'], 'seed': r['seed'], 'variant': r['variant'],
                     'checkpoint': r['path'].relative_to(REPO).as_posix(), 'coef_start': c0, 'coef_end': c1})
        print(f"DONE {r['name']} ({time.time() - t0:.0f}s) coef {c0:.4g} -> {c1:.4g}", flush=True)
    # reproduction check: new TopK seed-42 run vs the pilot reconstruction-only checkpoint
    new = torch.load(next(r['path'] for r in runs if r['name'].endswith('REPRO_CHECK')), weights_only=False)['model_state_dict']
    old = torch.load(str(PILOT_TOPK).format(s=42), weights_only=False)['model_state_dict']
    repro = {k: float((new[k].float() - old[k].float()).abs().max()) for k in old}
    assert all(sha(p) == h for p, h in base_sha.items()), 'a base checkpoint changed!'
    assert all(sha(str(PILOT_TOPK).format(s=s)) == h for s, h in pilot_sha.items()), 'a pilot checkpoint changed!'
    pd.DataFrame(traces).to_csv(OUT / 'training_trace.csv', index=False)
    (OUT / 'training_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'runs': info, 'base_and_pilot_checkpoints_unchanged': True,
        'topk_repro_max_abs_param_diff': max(repro.values()), 'topk_repro_bit_identical': max(repro.values()) == 0.0,
        'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print('TopK reproduction max |param diff| vs pilot:', max(repro.values()))


if __name__ == '__main__':
    main()
