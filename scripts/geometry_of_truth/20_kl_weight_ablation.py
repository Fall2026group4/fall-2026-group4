#!/usr/bin/env python3
"""KL-weight ablation, GPT-2 Small layer 6, TopK (L0 = 128): does an intermediate output-KL weight keep Method A's
independent truth-probe performance while retaining part of E2E-inspired Method B's behavioural gain?

Run from repository root, in order (each step refuses to overwrite):
    python scripts/geometry_of_truth/20_kl_weight_ablation.py --verify    # provenance + reproduce A/B results (no training)
    python scripts/geometry_of_truth/20_kl_weight_ablation.py --prepare   # pre-registration + hashes
    python scripts/geometry_of_truth/20_kl_weight_ablation.py --dry-run
    python scripts/geometry_of_truth/20_kl_weight_ablation.py --train     # repro check (fraction 1.0, seed 42) + 6 runs

Protocol = E2E-inspired Method B (scripts/geometry_of_truth/17_e2e_inspired_finetune.py, imported UNCHANGED; its
`train_one` is called directly): init from the ORIGINAL uninterrupted TopK checkpoint of each seed; local loss
0.5*MSE(GoT-A rows) + 0.5*MSE(OWT rows) with Method A's batches/RNG; + w * alpha_B(seed) * 0.5*(KL_OWT + KL_GoT),
KL(p_orig || p_spliced), GPT-2 frozen; Adam lr 1e-5, 500 steps, clip 10; KL batches 8 OWT train sequences + the
step's 128 GoT-A statements, generator seed+5000 (identical data order for every w).
alpha_B(seed) is Method B's coefficient (train-only scale matching, results/geometry_of_truth/e2e_inspired_layer6/
alpha_beta.json). w in {0.25, 0.50} (new); w = 0 is Method A and w = 1 is Method B (existing checkpoints, reused).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S17 = load('scripts/geometry_of_truth/17_e2e_inspired_finetune.py', 's17')
LAYER, ARCH, SEEDS, FRACTIONS = 6, 'topk', (42, 1042, 2042), (0.25, 0.50)
E2E = REPO / 'results/geometry_of_truth/e2e_inspired_layer6'
OUT = REPO / 'results/geometry_of_truth/kl_weight_ablation_layer6_topk'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/kl_weight_ablation_layer6_topk'
SPLIT_SHA_EXPECTED = 'd43e83833e2d7427bf579c1358cab7068e7b1a34d42ff9bf0cdf769196dcaf41'
REPRO_TOL = 1e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paths():
    base = {s: uninterrupted_checkpoint(LAYER, ARCH, s) for s in SEEDS}
    A = {s: S17.METHOD_A[(ARCH, s)] for s in SEEDS}
    B = {s: S17.CKPT / f'{ARCH}_methodB_seed{s}' / 'finetuned_checkpoint.pt' for s in SEEDS}
    new = {(f, s): CKPT / f'{ARCH}_kl{int(f * 100):03d}pct_seed{s}' / 'finetuned_checkpoint.pt' for f in FRACTIONS for s in SEEDS}
    return base, A, B, new


def alphas():
    ab = json.loads((E2E / 'alpha_beta.json').read_text())
    return {s: ab[f'{ARCH}_seed{s}']['alpha'] for s in SEEDS}


# ------------------------------------------------------------------ step 1: verification (no training)
def verify():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'verification.json').exists():
        raise SystemExit('verification.json exists; refusing to overwrite.')
    base, A, B, new = paths()
    al = alphas()
    rep = {'split': S17.S14.verify_split()}
    rep['split_sha_matches_expected'] = rep['split']['sha256'] == SPLIT_SHA_EXPECTED
    e2e_hashes = json.loads((E2E / 'pretraining_hashes.json').read_text())['sha256']
    rep['e2e_pretraining_hashes_still_match'] = {p: sha(REPO / p) == h for p, h in e2e_hashes.items()}
    per = pd.read_csv(E2E / 'per_sae.csv')
    per = per[per.architecture == ARCH]
    ck = {}
    for s in SEEDS:
        bA = torch.load(A[s], map_location='cpu', weights_only=False)
        bB = torch.load(B[s], map_location='cpu', weights_only=False)
        row = {
            'base_path': base[s].relative_to(REPO).as_posix(), 'base_sha256': sha(base[s]),
            'base_sha_matches_e2e_eval': sha(base[s]) == per[(per.method == 'base') & (per.seed == s)].checkpoint_sha256.iloc[0],
            'A_path': A[s].relative_to(REPO).as_posix(), 'A_sha256': sha(A[s]),
            'A_sha_matches_e2e_eval': sha(A[s]) == per[(per.method == 'A') & (per.seed == s)].checkpoint_sha256.iloc[0],
            'A_source': bA.get('source_checkpoint'), 'A_source_sha_matches_base': bA.get('source_checkpoint_sha256') == sha(base[s]),
            'A_lambda': bA.get('config', {}).get('lambda'),
            'B_path': B[s].relative_to(REPO).as_posix(), 'B_sha256': sha(B[s]),
            'B_sha_matches_e2e_eval': sha(B[s]) == per[(per.method == 'B') & (per.seed == s)].checkpoint_sha256.iloc[0],
            'B_source': bB.get('source_checkpoint'), 'B_method': bB['config']['method'], 'B_beta': bB['config']['beta'],
            'B_alpha_in_checkpoint': bB['config']['alpha'], 'B_alpha_in_alpha_beta_json': al[s],
            'B_lr': bB['config']['lr'], 'B_steps': bB['config']['steps'],
            'new_run_init': base[s].relative_to(REPO).as_posix() + ' (S17.train_one loads uninterrupted_checkpoint)',
            'new_paths_free': all(not new[(f, s)].exists() for f in FRACTIONS)}
        row['provenance_ok'] = bool(row['base_sha_matches_e2e_eval'] and row['A_sha_matches_e2e_eval'] and row['A_source_sha_matches_base']
                                    and row['A_lambda'] == 0.0 and row['B_sha_matches_e2e_eval'] and row['B_method'] == 'B'
                                    and row['B_beta'] == 0.0 and row['B_alpha_in_checkpoint'] == al[s]
                                    and row['B_source'] == base[s].relative_to(REPO).as_posix()
                                    and row['B_lr'] == S17.S14.LR and row['B_steps'] == S17.S14.STEPS and row['new_paths_free'])
        ck[s] = row
    rep['checkpoints'] = ck
    # reproduce saved A / B / base validation results from the checkpoints (no retraining)
    ev6 = load('scripts/shared/layer6_faithfulness_eval.py', 'ev6')
    ctx = ev6.build_context('cuda')
    rows = []
    for m, d in (('base', base), ('A', A), ('B', B)):
        for s in SEEDS:
            sae, _ = fm.load_sae(d[s], 'cuda', architecture=ARCH)
            sae.eval()
            rows.append({'method': m, 'seed': s, **ev6.evaluate_sae(sae, ctx)})
    re = pd.DataFrame(rows)
    re.to_csv(OUT / 'reproduction_of_saved_results.csv', index=False)
    metrics = [c for c in re.columns if c not in ('method', 'seed')]
    old = per.set_index(['method', 'seed'])[metrics]
    diff = (re.set_index(['method', 'seed'])[metrics] - old.loc[re.set_index(['method', 'seed']).index]).abs()
    rep['reproduction_max_abs_diff_by_metric'] = {k: float(v) for k, v in diff.max().items()}
    rep['reproduction_max_abs_diff'] = float(diff.max().max())
    rep['reproduction_ok'] = rep['reproduction_max_abs_diff'] <= REPRO_TOL
    rep['passes'] = bool(rep['split']['passes'] and rep['split_sha_matches_expected']
                         and all(rep['e2e_pretraining_hashes_still_match'].values())
                         and all(r['provenance_ok'] for r in ck.values()) and rep['reproduction_ok'])
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('split', 'reproduction_max_abs_diff_by_metric')}, indent=2, default=str))
    if not rep['passes']:
        raise SystemExit('VERIFICATION FAILED - stopping.')


# ------------------------------------------------------------------ step 2: pre-registration
def prepare():
    if (OUT / 'preregistration.json').exists():
        raise SystemExit('preregistration.json exists; refusing to overwrite.')
    v = json.loads((OUT / 'verification.json').read_text())
    if not v['passes']:
        raise SystemExit('verification did not pass')
    al = alphas()
    weights = {f'{int(f * 100)}pct': {f'seed{s}': {'alpha': f * al[s], 'per_KL_term_weight': 0.5 * f * al[s]} for s in SEEDS}
               for f in (0.0, 0.25, 0.50, 1.0)}
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'question': 'Can an intermediate output-KL weight keep Method A\'s independent truth-probe performance while retaining '
                    'part of Method B\'s behavioural (output-KL / CE) gain? GPT-2 Small layer 6, TopK L0=128.',
        'protocol': 'E2E-inspired Method B protocol (script 17 train_one, unchanged) with alpha scaled by w; init from the '
                    'original uninterrupted TopK checkpoint per seed; identical data order for all w; 500 steps, Adam lr 1e-5',
        'coefficient_semantics': 'Method B loss = L_A + alpha*0.5*(KL_OWT+KL_GoT); alpha is the only KL coefficient (raw '
                                 'multiplier = the train-only scale-matched value). Fractions scale alpha; per-term weight = 0.5*alpha.',
        'weights': weights, 'fractions_new': FRACTIONS, 'fractions_fixed': 'not changed after seeing any validation result',
        'seeds': SEEDS, 'split_sha256': SPLIT_SHA_EXPECTED,
        'reuse': {'w=0 (Method A)': 'results/geometry_of_truth/sae_finetuning/truth_aware_pilot_layer6_topk/reconstruction_only_seed*/',
                  'w=1 (Method B)': 'results/geometry_of_truth/sae_finetuning/e2e_inspired_layer6/topk_methodB_seed*/'},
        'repro_check': f'w=1.0 seed 42 retrained with this script must reproduce the Method B seed-42 checkpoint: bit-identical, or '
                       f'max|param diff| <= {REPRO_TOL} (GPU non-determinism) -> pass with note; otherwise STOP. Excluded from results.',
        'evaluation': 'validation only, scripts/shared/layer6_faithfulness_eval.py (independent GoT B-group probes fitted on '
                      'original activations; never the teacher; no fine-tuning teacher exists in this experiment); no test rows',
        'primary_outcomes': {'truth': 'GoT geographic independent B-MLP AUROC gap (lower better)',
                             'behaviour': ['OWT validation output KL (lower better)', 'OWT validation spliced CE increase (lower better)'],
                             'sparsity': 'TopK L0 = 128 on OWT, GoT and RepE (must hold exactly)'},
        'secondary_outcomes': ['GoT numerical B-MLP gap', 'GoT B-logistic gaps', 'RepE logistic/mass-mean gaps, direction cosine, '
                               'persona-shift cosine', 'OWT MSE, EV, cosine, dead fraction', 'downstream NMSE L7-11',
                               'GoT statement KL', 'cities preference change'],
        'criteria': {
            'unit': 'per w in {0.25, 0.50}, paired by seed; n = 3 seeds -> descriptive rules, no significance claims',
            'consistent': 'better than the reference on all 3 seeds AND mean paired difference > 2 x sd of the paired differences',
            'BEHAV': 'OWT output KL AND OWT spliced CE increase both consistently lower than Method A (w=0)',
            'TRUTH_NONINFERIOR': 'mean paired (geo MLP gap(w) - gap(A)) <= +0.02 (margin = the guard margin used before)',
            'TRADE': 'on every seed, (KL_w, gap_w) lies strictly below the seed\'s A-B chord: gap_w < gap_A + (gap_B - gap_A) * '
                     '(KL_A - KL_w) / (KL_A - KL_B), using OWT output KL',
            'guards': ['TopK L0 == 128 on OWT, GoT, RepE validation', 'OWT EV >= Method A - 0.02 (every seed)',
                       'cities preference change >= Method A - 0.018 (mean)', 'RepE mass-mean gap <= Method A + 0.02 (mean)',
                       'GoT geographic B-logistic gap <= Method A + 0.02 (mean)'],
            'verdict': 'IMPROVES TRADE-OFF only if BEHAV AND TRUTH_NONINFERIOR AND TRADE AND all guards. If the endpoints do not '
                       'differ in truth (mean gap_B - gap_A <= 0.02, i.e. no measurable trade-off), TRADE is uninformative and '
                       'the trade-off conclusion is INCONCLUSIVE regardless of TRADE. One metric improving while another '
                       'worsens is not success.',
            'also_reported': 'fraction of Method B\'s KL and CE gain retained = (A - w)/(A - B), per seed and mean'},
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    files = ['scripts/geometry_of_truth/20_kl_weight_ablation.py', 'scripts/geometry_of_truth/21_kl_weight_ablation_evaluation.py',
             'scripts/geometry_of_truth/17_e2e_inspired_finetune.py', 'scripts/shared/layer6_faithfulness_eval.py',
             'data/got/splits/train_AB_groups_truth_aware_pilot.csv', 'results/geometry_of_truth/e2e_inspired_layer6/alpha_beta.json',
             'results/geometry_of_truth/kl_weight_ablation_layer6_topk/preregistration.json',
             'results/geometry_of_truth/kl_weight_ablation_layer6_topk/verification.json']
    (OUT / 'pretraining_hashes.json').write_text(json.dumps({'recorded': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             'sha256': {f: sha(REPO / f) for f in files}}, indent=2))
    print(json.dumps(weights, indent=2))


# ------------------------------------------------------------------ step 3: training
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ('verify', 'prepare', 'dry-run', 'train'):
        g.add_argument(f'--{f}', action='store_true')
    a = ap.parse_args()
    if a.verify:
        return verify()
    if a.prepare:
        return prepare()
    hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
    bad = [p for p, h in hashes.items() if sha(REPO / p) != h]
    if bad:
        raise SystemExit(f'files changed since pre-registration: {bad}')
    base, A, B, new = paths()
    repro = CKPT / f'{ARCH}_kl100pct_seed42_REPRO_CHECK' / 'finetuned_checkpoint.pt'
    clash = [str(p) for p in list(new.values()) + [repro] if p.exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    al = alphas()
    if a.dry_run:
        for (f, s), p in new.items():
            print(f'w={f:.2f} seed {s}: alpha={f * al[s]:.4f} (per-term {0.5 * f * al[s]:.4f}) init={base[s].relative_to(REPO)} -> {p.relative_to(REPO)}')
        print(f'repro check (w=1.0, seed 42) -> {repro.relative_to(REPO)}; 6 runs planned; nothing trained.')
        return
    device = 'cuda'
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fp0 = S17.model_fingerprint(model)
    X_A, T_A, seqs = S17.load_data(model, device)
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_rows = o3.load_split(REPO / 'data/activations/gpt2_small', 'train', LAYER)
    gm = torch.Generator().manual_seed(99)
    monitor = [seqs[i] for i in torch.randint(len(seqs), (S17.N_OWT_SEQ,), generator=gm)]
    protected = {str(p): sha(p) for d in (base, A, B) for p in d.values()}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    sae, _, _, _ = S17.train_one(ARCH, 42, 'B', 1.0 * al[42], 0.0, model, X_A, T_A, seqs, owt_rows, monitor, device)
    repro.parent.mkdir(parents=True, exist_ok=False)
    torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in sae.state_dict().items()}, 'note': 'repro check only'}, repro)
    ref = torch.load(B[42], weights_only=False)['model_state_dict']
    diff = max(float((sae.state_dict()[k].cpu().float() - ref[k].float()).abs().max()) for k in ref)
    print('REPRO CHECK (w=1.0 seed 42 vs Method B) max |param diff|:', diff, flush=True)
    if diff > REPRO_TOL:
        (OUT / 'training_info.json').write_text(json.dumps({'repro_max_abs_diff': diff, 'stopped': True}, indent=2))
        raise SystemExit('Reproduction check FAILED; stopping before the ablation runs.')
    info, logs = [], []
    for (f, s), p in new.items():
        t1 = time.time()
        sae, c0, c1, log = S17.train_one(ARCH, s, 'B', f * al[s], 0.0, model, X_A, T_A, seqs, owt_rows, monitor, device)
        p.parent.mkdir(parents=True, exist_ok=False)
        torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                    'config': {'architecture': ARCH, 'target_l0': 128, 'layer': LAYER, 'seed': s, 'method': 'B_scaled_kl',
                               'kl_fraction_of_method_B': f, 'alpha': f * al[s], 'alpha_method_B': al[s], 'beta': 0.0,
                               'lr': S17.S14.LR, 'steps': S17.S14.STEPS, 'procedure': 'kl_weight_ablation_layer6_topk_2026-11-07'},
                    'validation_metrics': {'sparsity_coefficient': c1},
                    'source_checkpoint': base[s].relative_to(REPO).as_posix(), 'source_checkpoint_sha256': protected[str(base[s])]}, p)
        logs += [{'kl_fraction': f, 'seed': s, **r} for r in log]
        info.append({'kl_fraction': f, 'seed': s, 'alpha': f * al[s], 'seconds': round(time.time() - t1, 1),
                     'checkpoint': p.relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(p)})
        print(f'DONE w={f} seed {s} ({time.time() - t1:.0f}s)', flush=True)
        pd.DataFrame(logs).to_csv(OUT / 'training_monitor.csv', index=False)
    ok_gpt2 = S17.model_fingerprint(model) == fp0
    ok_prot = all(sha(p) == h for p, h in protected.items())
    (OUT / 'training_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'repro_max_abs_param_diff': diff, 'repro_bit_identical': diff == 0.0, 'runs': info,
        'gpt2_parameters_unchanged': ok_gpt2, 'base_A_B_checkpoints_unchanged': ok_prot,
        'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    assert ok_gpt2 and ok_prot, 'GPT-2 or a protected checkpoint changed!'


if __name__ == '__main__':
    main()
