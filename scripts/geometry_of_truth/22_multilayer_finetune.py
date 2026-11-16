#!/usr/bin/env python3
"""IMP-L: multi-layer improvement experiment (Instructor Review #8 blocker; 11-03 session note "fine-tune other layers").

The layer-6 fine-tuning protocols are repeated, unchanged, at layers 3, 8 and 9:
  owt_only  continued training on OWT train only (control for extra steps / drift)      [script 14, variant owt_only]
  method_a  0.5*MSE(GoT train groups A) + 0.5*MSE(OWT train) + sparsity penalty           [script 14, variant got_replay]
  kl        method_a + alpha * 0.5*(KL_OWT + KL_GoT), KL(p_orig || p_spliced), GPT-2 frozen [script 17, method B]
for ReLU / TopK / JumpReLU x seeds 42 / 1042 / 2042, each starting from the uninterrupted 15k-step checkpoint of that
layer. Adam lr 1e-5, 500 steps, grad clip 10, decoder renormalisation; ReLU/JumpReLU keep their penalty with the
multiplicative L0 controller continued from the base coefficient (target 128, EMA on OWT rows); TopK k = 128.
alpha is set per base SAE by the layer-6 rule (train-only scale matching at step 0), recomputed for each layer.

The training functions are imported from scripts 14 and 17 and called unchanged; only their layer constants are set
per layer. --prepare verifies that this reuse reproduces the existing layer-6 checkpoints bit for bit
(Method A TopK seed 42 and Method B TopK seed 42) before anything is pre-registered.

No validation or test row is read for training. GoT groups B, validation and test are never used here.

Run from the repository root, in order:
    python scripts/geometry_of_truth/22_multilayer_finetune.py --prepare    # checks, reproduction, alpha, pre-registration
    python scripts/geometry_of_truth/22_multilayer_finetune.py --dry-run    # list runs, inputs and coefficients
    python scripts/geometry_of_truth/22_multilayer_finetune.py --train [--layers 8]
Outputs: results/geometry_of_truth/multilayer_finetune/{preregistration.json, pretraining_checks.json, alpha.json,
pretraining_hashes.json, training_info_layer{L}.json, training_monitor_layer{L}.csv}; checkpoints (gitignored) in
results/geometry_of_truth/sae_finetuning/multilayer_finetune/layer_{L}/{arch}_{arm}_seed{S}/finetuned_checkpoint.pt.
Refuses to overwrite anything.
"""
import argparse
import hashlib
import importlib.util
import json
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

LAYERS, ARCHS, SEEDS, ARMS = (3, 8, 9), ('relu', 'topk', 'jumprelu'), (42, 1042, 2042), ('owt_only', 'method_a', 'kl')
OUT = REPO / 'results/geometry_of_truth/multilayer_finetune'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/multilayer_finetune'
EVAL_SCRIPT = 'scripts/geometry_of_truth/23_multilayer_evaluation.py'


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S17 = load('scripts/geometry_of_truth/17_e2e_inspired_finetune.py', 's17')
S14, P11 = S17.S14, S17.P11          # one shared module instance each, so layer settings apply everywhere


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_layer(layer):
    """Point the imported layer-6 modules at `layer` (data loading, splice hook, base checkpoint)."""
    P11.LAYER = layer
    S14.LAYER = layer
    S17.LAYER = layer
    S17.HOOK = f'blocks.{layer}.hook_resid_pre'
    S17.DS_HOOKS = [f'blocks.{k}.hook_resid_pre' for k in range(layer + 1, 12)]   # monitor only (method B has no DS term)


def ckpt_path(layer, arch, arm, seed):
    return CKPT / f'layer_{layer}' / f'{arch}_{arm}_seed{seed}' / 'finetuned_checkpoint.pt'


def plan(layers):
    return [dict(layer=L, arch=a, arm=r, seed=s, base=uninterrupted_checkpoint(L, a, s), path=ckpt_path(L, a, r, s))
            for L in layers for r in ARMS for a in ARCHS for s in SEEDS]


def load_gpt2(device):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def layer_data(model, layer, device):
    set_layer(layer)
    X_A, T_A, seqs = S17.load_data(model, device)
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_rows = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', layer)
    gm = torch.Generator().manual_seed(99)
    monitor = [seqs[i] for i in torch.randint(len(seqs), (S17.N_OWT_SEQ,), generator=gm)]
    return X_A, T_A, seqs, owt_rows, monitor


def run_one(r, model, data, alpha, device):
    """Train one SAE with the unchanged layer-6 functions. Returns (sae, coef_start, coef_end, monitor_log)."""
    X_A, T_A, seqs, owt_rows, monitor = data
    set_layer(r['layer'])
    if r['arm'] == 'kl':
        return S17.train_one(r['arch'], r['seed'], 'B', alpha, 0.0, model, X_A, T_A, seqs, owt_rows, monitor, device)
    run = dict(name=f"{r['arch']}_{r['arm']}_seed{r['seed']}", arch=r['arch'], seed=r['seed'], base=r['base'],
               variant='got_replay' if r['arm'] == 'method_a' else 'owt_only')
    sae, c0, c1, trace = S14.finetune(run, X_A, owt_rows, device)
    return sae, c0, c1, trace


def max_param_diff(sae, ref_path):
    ref = torch.load(ref_path, map_location='cpu', weights_only=False)['model_state_dict']
    return max(float((sae.state_dict()[k].detach().cpu().float() - ref[k].float()).abs().max()) for k in ref)


# ------------------------------------------------------------------------------------------------ prepare
def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ('preregistration.json', 'pretraining_checks.json', 'alpha.json'):
        if (OUT / f).exists():
            raise SystemExit(f'{f} exists; refusing to overwrite.')
    if not (REPO / EVAL_SCRIPT).exists():
        raise SystemExit(f'{EVAL_SCRIPT} must exist before pre-registration (its hash is frozen).')
    device = 'cuda'
    t0 = time.time()
    model = load_gpt2(device)
    checks = {'split': S14.verify_split()}
    missing = [str(r['base']) for r in plan(LAYERS) if not Path(r['base']).exists()]
    checks['missing_base_checkpoints'] = missing

    # 1) reproduction at layer 6 through this script's reuse path (must be bit-identical to existing checkpoints)
    data6 = layer_data(model, 6, device)
    a6 = json.loads((S17.OUT / 'alpha_beta.json').read_text())['topk_seed42']['alpha']
    r6 = dict(layer=6, arch='topk', seed=42, base=uninterrupted_checkpoint(6, 'topk', 42))
    sae, *_ = run_one({**r6, 'arm': 'method_a'}, model, data6, None, device)
    checks['repro_method_a_topk42_vs_pilot'] = max_param_diff(sae, str(S14.PILOT_TOPK).format(s=42))
    sae, *_ = run_one({**r6, 'arm': 'kl'}, model, data6, a6, device)
    checks['repro_kl_topk42_vs_method_B'] = max_param_diff(
        sae, S17.CKPT / 'topk_methodB_seed42' / 'finetuned_checkpoint.pt')
    del data6

    # 2) per-layer checks and alpha (train-only scale matching at step 0, the layer-6 rule)
    alpha, per_layer = {}, {}
    for L in LAYERS:
        X_A, T_A, seqs, owt_rows, monitor = layer_data(model, L, device)
        errs = []
        for d in P11.DOMAINS:                                  # cached GoT-A final-token activations == fresh forward
            toks, lens = S17.pad(T_A[d][:64])
            _, c = model.run_with_cache(toks.to(device), names_filter=lambda n: n == S17.HOOK)
            errs.append(float((c[S17.HOOK][torch.arange(len(lens)), lens - 1].float() - X_A[d][:64]).abs().max()))

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x.clone(), None, None
        kl_id, _ = S17.e2e_terms(model, Identity(), seqs[:4], device, False, grad=False)
        base_l0 = {}
        for a in ARCHS:
            for s in SEEDS:
                ck = torch.load(uninterrupted_checkpoint(L, a, s), map_location='cpu', weights_only=False)
                base_l0[f'{a}_seed{s}'] = {'val_l0': float(ck['validation_metrics'].get('l0', float('nan'))),
                                          'sparsity_coefficient': ck['validation_metrics'].get('sparsity_coefficient')}
                sae, _ = fm.load_sae(uninterrupted_checkpoint(L, a, s), device)
                gen = torch.Generator().manual_seed(s + 9000)
                rec, kls = [], []
                for _ in range(S17.SCALE_BATCHES):
                    idx, oidx = S17.local_batch(X_A, owt_rows, gen, device)
                    xg = torch.cat([X_A[d][idx[d].to(device)] for d in P11.DOMAINS])
                    xo = owt_rows[oidx].to(device)
                    with torch.no_grad():
                        rec.append(float(0.5 * F.mse_loss(sae(xg)[0], xg) + 0.5 * F.mse_loss(sae(xo)[0], xo)))
                        sidx = torch.randint(len(seqs), (S17.N_OWT_SEQ,), generator=gen)
                        k1, _ = S17.e2e_terms(model, sae, [seqs[i] for i in sidx], device, False, grad=False)
                        k2, _ = S17.e2e_terms(model, sae, [T_A[d][i] for d in P11.DOMAINS for i in idx[d].tolist()],
                                              device, False, grad=False)
                    kls.append(0.5 * float(k1 + k2))
                alpha[f'layer{L}_{a}_seed{s}'] = {'median_local_recon': float(np.median(rec)),
                                                  'median_kl': float(np.median(kls)),
                                                  'alpha': float(np.median(rec) / np.median(kls))}
        per_layer[L] = {'hook_got_A_fresh_vs_cached_max_abs': max(errs), 'identity_splice_kl': float(kl_id),
                        'rows_A': {d: int(len(X_A[d])) for d in P11.DOMAINS}, 'owt_train_rows': int(len(owt_rows)),
                        'base': base_l0}
        print(f'layer {L} checks done ({time.time() - t0:.0f}s)', flush=True)
    checks['per_layer'] = per_layer
    ok = (checks['split']['passes'] and not missing and checks['repro_method_a_topk42_vs_pilot'] == 0.0
          and checks['repro_kl_topk42_vs_method_B'] == 0.0
          and all(v['hook_got_A_fresh_vs_cached_max_abs'] < 1e-3 and v['identity_splice_kl'] == 0.0 for v in per_layer.values()))
    checks['passes'] = bool(ok)
    checks['runtime_seconds'] = round(time.time() - t0, 1)
    (OUT / 'pretraining_checks.json').write_text(json.dumps(checks, indent=2))
    (OUT / 'alpha.json').write_text(json.dumps(alpha, indent=2))
    if not ok:
        raise SystemExit('PRE-TRAINING CHECKS FAILED; nothing pre-registered. See pretraining_checks.json')
    (OUT / 'preregistration.json').write_text(json.dumps(PREREG, indent=2))
    files = ['scripts/geometry_of_truth/22_multilayer_finetune.py', EVAL_SCRIPT,
             'scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py',
             'scripts/geometry_of_truth/17_e2e_inspired_finetune.py', P11.SPLIT_CSV.relative_to(REPO).as_posix(),
             'results/geometry_of_truth/multilayer_finetune/preregistration.json',
             'results/geometry_of_truth/multilayer_finetune/alpha.json']
    (OUT / 'pretraining_hashes.json').write_text(json.dumps(
        {'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'note': 'recorded before any IMP-L training run',
         'sha256': {f: sha(REPO / f) for f in files}}, indent=2))
    print(json.dumps({k: v for k, v in checks.items() if k != 'per_layer'}, indent=2))


PREREG = {
    'experiment': 'IMP-L multi-layer improvement (Instructor Review #8 blocker)',
    'evidence_label': 'VALIDATION (no test row read by training or evaluation)',
    'question': 'Do the layer-6 fine-tuning methods (Method A; output-KL) improve faithfulness at layers 3, 8 and 9 '
                'relative to the original SAEs and to an OWT-only continued-training control, at matched sparsity?',
    'layers': list(LAYERS), 'architectures': list(ARCHS), 'seeds': list(SEEDS),
    'arms': {'base': 'uninterrupted 15k-step SAE (no training)', 'owt_only': 'control: 500 steps OWT-only',
             'method_a': '500 steps 0.5*MSE_GoT-A + 0.5*MSE_OWT + penalty',
             'kl': 'method_a + alpha*0.5*(KL_OWT+KL_GoT) (layer-6 Method B), alpha per base SAE in alpha.json'},
    'budget': {'steps': 500, 'lr': 1e-5, 'optimizer': 'Adam', 'grad_clip': 10.0,
               'batch': '64 geographic-A + 64 numerical-A + 128 OWT train rows (owt_only: 256 OWT rows); kl adds 8 OWT '
                        'train sequences + the 128 GoT-A statements per step'},
    'new_training_runs': len(LAYERS) * len(ARMS) * len(ARCHS) * len(SEEDS),
    'layer_3_caveat': 'the original GoT logistic probe is below the 0.70 informative threshold at layer 3 (val 0.648, '
                      'all-layer frozen-probe table); layer-3 truth results are reported but are descriptive if the '
                      'original B-MLP validation AUROC is < 0.70',
    'evaluation': f'{EVAL_SCRIPT}, validation only: OWT validation EV/MSE/cosine/L0/dead fraction; GoT geographic and '
                  'numerical relative error and L0; RepE-LM persona-prompt L0; GoT B-group probes fitted on ORIGINAL '
                  'activations (MLP x 5 seeds, logistic with C chosen on original validation, mass-mean); M2-style '
                  're-trained logistic gap on GoT geographic and RepE-LM persona; OWT splice delta CE and KL, GoT '
                  'statement KL, cities truth preference. Layer 8 additionally: causal preservation T_rec/T_full at the '
                  'final token with the M4 protocol (E3, separate pre-registered step after training).',
    'statistics': 'per layer x architecture, paired by seed: arm - base and arm - owt_only (and kl - method_a); '
                  'mean, sd, per-seed values, 95 % t-interval over the 3 seeds (t = 4.303). No pooling across layers.',
    'criteria': {
        'consistent': 'better on all 3 seeds AND |mean paired difference| > 2 x its sd (the layer-6 rule)',
        'T_method_a': 'GoT geographic B-MLP gap: method_a < base AND method_a < owt_only, both consistent',
        'B_kl': 'OWT splice KL AND GoT statement KL: kl < method_a, both consistent',
        'T_kl': 'GoT geographic B-MLP gap: kl < method_a, consistent',
        'guards': ['OWT-validation L0 within +-5 % of the base SAE on every seed',
                   'OWT validation EV >= base - 0.005 on every seed (kl: >= method_a - 0.02, the layer-6 rule)',
                   'OWT splice delta CE <= 1.10 x base on every seed',
                   'cities truth-preference change >= base - 0.018 (mean over seeds)'],
        'classification': 'CONFIRMED is not used (validation only). PRELIMINARY IMPROVEMENT = criterion AND all guards '
                          'met; NO IMPROVEMENT = criterion not met with guards met; NOT CREDITABLE = any guard failed '
                          '(reported, excluded from rankings); REGRESSION = consistent change in the wrong direction.'},
    'unchanged_from_layer_6': 'all training code is imported from scripts 14 and 17; --prepare reproduced Method A TopK '
                              'seed 42 and Method B TopK seed 42 bit for bit through this script before registration',
}


# ------------------------------------------------------------------------------------------------ train / dry run
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--train', action='store_true')
    ap.add_argument('--layers', type=int, nargs='+', default=list(LAYERS))
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    if any(L not in LAYERS for L in a.layers):
        raise SystemExit(f'layers must be a subset of {LAYERS}')
    hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
    bad = [p for p, h in hashes.items() if sha(REPO / p) != h]
    if bad:
        raise SystemExit(f'files changed since pre-registration: {bad}')
    if not json.loads((OUT / 'pretraining_checks.json').read_text())['passes']:
        raise SystemExit('pre-training checks did not pass')
    alpha = json.loads((OUT / 'alpha.json').read_text())
    runs = plan(a.layers)
    clash = [str(r['path']) for r in runs if r['path'].exists()]
    clash += [str(OUT / f'training_info_layer{L}.json') for L in a.layers if (OUT / f'training_info_layer{L}.json').exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    if a.dry_run:
        for r in runs:
            al = alpha[f"layer{r['layer']}_{r['arch']}_seed{r['seed']}"]['alpha'] if r['arm'] == 'kl' else 0.0
            print(f"L{r['layer']} {r['arm']:9s} {r['arch']:8s} seed {r['seed']:4d} alpha={al:.3f} "
                  f"base={Path(r['base']).relative_to(REPO)}")
        print(f'{len(runs)} training runs planned; nothing trained.')
        return
    device = 'cuda'
    model = load_gpt2(device)
    fp0 = S17.model_fingerprint(model)
    for L in a.layers:
        data = layer_data(model, L, device)
        protected = {str(r['base']): sha(r['base']) for r in runs if r['layer'] == L}
        info, logs = [], []
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for r in [r for r in runs if r['layer'] == L]:
            t1 = time.time()
            al = alpha[f"layer{L}_{r['arch']}_seed{r['seed']}"]['alpha'] if r['arm'] == 'kl' else 0.0
            sae, c0, c1, log = run_one(r, model, data, al, device)
            r['path'].parent.mkdir(parents=True, exist_ok=False)
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                        'config': {'architecture': r['arch'], 'target_l0': 128, 'layer': L, 'seed': r['seed'],
                                   'arm': r['arm'], 'alpha': al, 'lr': S14.LR, 'steps': S14.STEPS,
                                   'procedure': 'imp_l_multilayer_2026-11-09'},
                        'validation_metrics': {'sparsity_coefficient': c1}, 'sparsity_coefficient_start': c0,
                        'source_checkpoint': Path(r['base']).relative_to(REPO).as_posix(),
                        'source_checkpoint_sha256': protected[str(r['base'])]}, r['path'])
            logs += [{'layer': L, 'arm': r['arm'], 'architecture': r['arch'], 'seed': r['seed'], **x} for x in log]
            info.append({'arm': r['arm'], 'architecture': r['arch'], 'seed': r['seed'], 'alpha': al, 'coef_start': c0,
                         'coef_end': c1, 'seconds': round(time.time() - t1, 1),
                         'checkpoint': r['path'].relative_to(REPO).as_posix()})
            print(f"DONE L{L} {r['arm']} {r['arch']} seed {r['seed']} ({time.time() - t1:.0f}s, layer total "
                  f"{time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(logs).to_csv(OUT / f'training_monitor_layer{L}.csv', index=False)
        ok_bases = all(sha(p) == h for p, h in protected.items())
        (OUT / f'training_info_layer{L}.json').write_text(json.dumps({
            'script_sha256': sha(__file__), 'layer': L, 'runs': info, 'base_checkpoints_unchanged': ok_bases,
            'gpt2_parameters_unchanged': S17.model_fingerprint(model) == fp0,
            'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
            'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
        assert ok_bases and S17.model_fingerprint(model) == fp0, 'a base checkpoint or GPT-2 changed!'


if __name__ == '__main__':
    main()
