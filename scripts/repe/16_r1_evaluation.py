#!/usr/bin/env python3
"""R1 evaluation (validation + cross-wording validation only) against the pre-registration.

Run from repository root after `15_r1_honesty_finetune.py --train`:
    python scripts/repe/16_r1_evaluation.py

SAEs: 9 bases, 9 Method A (reference), 27 R1 fine-tunes; EXPLORATORY: TopK bases re-evaluated at k = 164 / 180.
Independent probes are fitted ONCE on ORIGINAL train-B activations (layer 6, final token) and applied frozen:
  B-MLP (768-64-1, inits 42/123/456/789/2026; primary), B-logistic (C by statement-grouped 5-fold CV within B),
  B-mass-mean (unit direction from B means).
Scored sets: RepE validation (89 statements; 326 excluded) and the cross-wording validation set (89; 326 excluded).
No test row is read (test pass deferred). The legacy all-train probe metrics from the shared layer-6 evaluator
(`repe_*`) are DIAGNOSTIC (their train rows include A). OWT, GoT and splicing metrics come from
scripts/shared/layer6_faithfulness_eval.py (unchanged), with RepE validation statement 326 removed from its inputs.
Outputs (results/repe/r1_honesty_preservation_layer6/): per_sae.csv, original_baseline.json, summary.csv,
paired_differences.csv, criteria.json, eval_info.json.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

MLP_SEEDS = (42, 123, 456, 789, 2026)
N_BOOT = 1000


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


S15 = load('scripts/repe/15_r1_honesty_finetune.py', 's15')
S14, OUT, L = S15.S14, S15.OUT, S15.LAYER
METHOD_A = REPO / 'results/geometry_of_truth/sae_finetuning/method_a_layer6'
PILOT = REPO / 'results/geometry_of_truth/sae_finetuning/truth_aware_pilot_layer6_topk'


def sae_list():
    out = []
    for a in S15.ARCHS:
        for s in S15.SEEDS:
            out.append(('base', a, s, Path(uninterrupted_checkpoint(L, a, s)), None))
            out.append(('method_a', a, s, (PILOT / f'reconstruction_only_seed{s}' if a == 'topk'
                                           else METHOD_A / f'{a}_got_replay_seed{s}') / 'finetuned_checkpoint.pt', None))
            for arm in S15.ARMS:
                out.append((arm, a, s, S15.CKPT / f'{arm}_{a}_seed{s}' / 'finetuned_checkpoint.pt', None))
    for k in (164, 180):
        for s in S15.SEEDS:
            out.append((f'EXPLORATORY_topk_base_k{k}', 'topk', s, Path(uninterrupted_checkpoint(L, 'topk', s)), k))
    return out


def fit_B_probes(X, y, g, B, device, g07, ev):
    best = None
    for C in sorted(ev.C_GRID):
        aucs = []
        for tr, te in GroupKFold(5).split(X[B], y[B], g[B]):
            p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=ev.PROBE_SEED))
            p.fit(X[B][tr], y[B][tr])
            aucs.append(fm.safe_auroc(y[B][te], p.decision_function(X[B][te])))
        if best is None or np.mean(aucs) > best[0] + 1e-15:
            best = (float(np.mean(aucs)), C)
    log = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000, random_state=ev.PROBE_SEED)).fit(X[B], y[B])
    d = X[B & (y == 1)].astype(np.float64).mean(0) - X[B & (y == 0)].astype(np.float64).mean(0)
    mlps = []
    for s in MLP_SEEDS:
        g07.SEED = s
        torch.manual_seed(s)
        mlps.append(g07.train_probe(X[B], y[B], device))
    return dict(log=log, C=best[1], C_cv_auroc=best[0], dir=d / np.linalg.norm(d), mlps=mlps)


def probe_scores(P, X, device, g07):
    mlp = [g07.probe_scores(p, s_, X, device) for p, s_ in P['mlps']]
    return {'log': P['log'].decision_function(X), 'mm': X @ P['dir'], 'mlp': mlp, 'mlp_mean': np.mean(mlp, 0)}


def margin(ids, y, s):
    t = pd.DataFrame({'id': ids, 'y': y, 's': s}).pivot(index='id', columns='y', values='s')
    return float((t[1] - t[0]).mean())


def pair_dir(ids, y, X):
    o = pd.Series(np.arange(len(ids)))
    h = o[(y == 1)].to_numpy()[np.argsort(ids[y == 1])]
    u = o[(y == 0)].to_numpy()[np.argsort(ids[y == 0])]
    assert (ids[h] == ids[u]).all()
    return (X[h].astype(np.float64) - X[u].astype(np.float64)).mean(0)


def set_metrics(pre, X, Xh, l0, m, y, ids, P, so, device, g07):
    yv, iv = y[m], ids[m]
    sr = probe_scores(P, Xh[m], device, g07)
    gaps = [fm.safe_auroc(yv, o) - fm.safe_auroc(yv, r) for o, r in zip(so['mlp'], sr['mlp'])]
    lo, hi = fm.grouped_bootstrap_gap(yv, so['mlp_mean'], sr['mlp_mean'], iv, n_boot=N_BOOT, seed=0)
    rec = fm.reconstruction_metrics(X[m], Xh[m], l0[torch.as_tensor(m)])
    d_o, d_r = pair_dir(iv, yv, X[m]), pair_dir(iv, yv, Xh[m])
    out = {
        'B_mlp_gap': float(np.mean(gaps)), 'B_mlp_gap_probe_sd': float(np.std(gaps, ddof=1)),
        'B_mlp_meanscore_gap': fm.safe_auroc(yv, so['mlp_mean']) - fm.safe_auroc(yv, sr['mlp_mean']),
        'B_mlp_meanscore_gap_ci_lo': lo, 'B_mlp_meanscore_gap_ci_hi': hi,
        'B_logistic_gap': fm.safe_auroc(yv, so['log']) - fm.safe_auroc(yv, sr['log']),
        'B_massmean_gap': fm.safe_auroc(yv, so['mm']) - fm.safe_auroc(yv, sr['mm']),
        'B_mlp_auroc_recon': float(np.mean([fm.safe_auroc(yv, r) for r in sr['mlp']])),
        'B_logistic_auroc_recon': fm.safe_auroc(yv, sr['log']), 'B_massmean_auroc_recon': fm.safe_auroc(yv, sr['mm']),
        'pair_acc_logistic': fm.pair_accuracy(iv, yv, sr['log']), 'pair_acc_massmean': fm.pair_accuracy(iv, yv, sr['mm']),
        'pair_acc_mlp': fm.pair_accuracy(iv, yv, sr['mlp_mean']),
        'margin_retention_logistic': margin(iv, yv, sr['log']) / margin(iv, yv, so['log']),
        'margin_retention_massmean': margin(iv, yv, sr['mm']) / margin(iv, yv, so['mm']),
        'pair_direction_cosine': float(d_o @ d_r / (np.linalg.norm(d_o) * np.linalg.norm(d_r))),
        'recon_mse': rec['mse'], 'recon_ev': rec['explained_variance'], 'recon_cosine': rec['cosine'], 'l0': rec['l0']}
    return {f'{pre}{k}': v for k, v in out.items()}


def consistent_lower(d):
    return bool((d < 0).all() and -d.mean() > 2 * d.std())


def main():
    smoke = '--smoke' in sys.argv[1:]   # 2 base SAEs + 1 exploratory, written to smoke/<time>/; no criteria
    global OUT
    if smoke:
        OUT = OUT / 'smoke' / time.strftime('%Y%m%d_%H%M%S')
        OUT.mkdir(parents=True)
    for p in ('per_sae.csv', 'criteria.json'):
        if (OUT / p).exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
    if not smoke:
        hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
        bad = [p for p in S15.VERIFIED if sha(REPO / p) != hashes[p]]
        if bad:
            raise SystemExit(f'files changed since pre-registration: {bad}')
    todo = ([x for x in sae_list() if x[0] == 'base' and x[2] == 42 and x[1] != 'jumprelu']
            + [x for x in sae_list() if x[0].startswith('EXPLORATORY')][:1]) if smoke else sae_list()
    missing = [str(p) for *_, p, _ in todo if not p.exists()]
    if missing:
        raise SystemExit(f'missing checkpoints: {missing}')
    device = 'cuda'
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    L6 = load('scripts/shared/layer6_faithfulness_eval.py', 'l6')
    ctx = L6.build_context(device)
    ev, g07 = ctx['ev'], ctx['g07']
    # ---- RepE inputs: drop validation statement 326 everywhere (main + cross-wording rows, splicing set)
    rd = ctx['rd']
    excl = np.isin(rd['groups'], [str(i) for i in S14.EXCLUDED_VALIDATION] + [f'cw{i}' for i in S14.EXCLUDED_VALIDATION])
    assert set(rd['split'][excl]) <= {'validation', 'cross_wording'} and excl.sum() == 4
    rd['split'] = np.where(excl, 'excluded', rd['split'])
    ctx['repe_probes'] = ev.fit_original_probes(rd['X'][L], rd['y'], rd['split'])     # legacy, DIAGNOSTIC only
    v = ctx['repe_v']
    ctx['repe_v'] = v[~v.statement_id.isin(S14.EXCLUDED_VALIDATION)].reset_index(drop=True)
    toks, lens = ctx['sp'].pad(list(ctx['repe_v'].tokens), 50256)
    ctx['repe_orig'] = ctx['S'].logprobs(toks, lens)[torch.arange(len(ctx['repe_v']), device=device), (lens - 1).to(device)]

    X, y, split = rd['X'][L], rd['y'], rd['split']
    ids = np.array([int(g[2:]) if g.startswith('cw') else int(g) for g in rd['groups']])
    ab = pd.read_csv(S14.SPLIT_CSV)
    is_main = np.array([not g.startswith('cw') for g in rd['groups']])
    B = is_main & np.isin(ids, ab.statement_id[ab.half == 'B']) & (split == 'train')
    A = is_main & np.isin(ids, ab.statement_id[ab.half == 'A']) & (split == 'train')
    sets = {'val_': split == 'validation', 'cw_': split == 'cross_wording'}
    assert B.sum() == 418 and A.sum() == 416 and sets['val_'].sum() == 178 and sets['cw_'].sum() == 178
    assert not (split == 'test')[B | A | sets['val_'] | sets['cw_']].any()
    P = fit_B_probes(X, y, ids, B, device, g07, ev)
    so = {k: probe_scores(P, X[m], device, g07) for k, m in sets.items()}
    original = {'B_logistic_C': P['C'], 'B_logistic_C_cv_auroc': P['C_cv_auroc'],
                'B_probe_train_rows': int(B.sum()), 'excluded_validation_statements': list(S14.EXCLUDED_VALIDATION)}
    for k, m in sets.items():
        yv, iv, s = y[m], ids[m], so[k]
        original.update({f'{k}B_mlp_auroc': float(np.mean([fm.safe_auroc(yv, o) for o in s['mlp']])),
                         f'{k}B_mlp_auroc_probe_sd': float(np.std([fm.safe_auroc(yv, o) for o in s['mlp']], ddof=1)),
                         f'{k}B_logistic_auroc': fm.safe_auroc(yv, s['log']), f'{k}B_massmean_auroc': fm.safe_auroc(yv, s['mm']),
                         f'{k}pair_acc_logistic': fm.pair_accuracy(iv, yv, s['log']),
                         f'{k}pair_acc_massmean': fm.pair_accuracy(iv, yv, s['mm']),
                         f'{k}pair_acc_mlp': fm.pair_accuracy(iv, yv, s['mlp_mean']),
                         f'{k}margin_logistic': margin(iv, yv, s['log']), f'{k}margin_massmean': margin(iv, yv, s['mm'])})
    (OUT / 'original_baseline.json').write_text(json.dumps(original, indent=2))
    print(json.dumps(original, indent=2), flush=True)

    rows = []
    for variant, arch, seed, path, k in todo:
        sae, _ = fm.load_sae(path, device, architecture=arch, k=k)
        Xh, l0, _ = fm.reconstruct(sae, X, device)
        Xh = Xh.numpy()
        r = {'variant': variant, 'architecture': arch, 'seed': seed, 'topk_k': k,
             'checkpoint': path.relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(path)}
        for pre, m in sets.items():
            r.update(set_metrics(pre, X, Xh, l0, m, y, ids, P, so[pre], device, g07))
        for nm, m in (('trainA_', A), ('trainB_', B)):
            rec = fm.reconstruction_metrics(X[m], Xh[m], l0[torch.as_tensor(m)])
            r.update({f'{nm}recon_mse': rec['mse'], f'{nm}recon_ev': rec['explained_variance'], f'{nm}l0': rec['l0']})
        r.update({('DIAG_legacy_' + c if c.startswith('repe_') else c): val for c, val in L6.evaluate_sae(sae, ctx).items()})
        rows.append(r)
        print(f'{variant} {arch} seed {seed} done ({time.time() - t0:.0f}s) val B-MLP gap {r["val_B_mlp_gap"]:.4f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_sae.csv', index=False)
    if smoke:
        print(df.T.to_string())
        return

    keys = ('variant', 'architecture', 'seed', 'topk_k', 'checkpoint', 'checkpoint_sha256')
    metrics = [c for c in df.columns if c not in keys]
    summ = df.groupby(['architecture', 'variant'])[metrics].agg(['mean', 'std'])
    summ.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False)

    comps, crit = [], {}
    pairs = [('r_recon', 'base'), ('r_honesty', 'base'), ('r_samepersona', 'base'), ('r_honesty', 'r_recon'),
             ('r_honesty', 'r_samepersona'), ('r_samepersona', 'r_recon'), ('method_a', 'base'), ('r_recon', 'method_a')]
    for arch in S15.ARCHS:
        piv = {v_: df[(df.architecture == arch) & (df.variant == v_)].set_index('seed')[metrics]
               for v_ in ('base', 'method_a') + S15.ARMS}
        for a_, b_ in pairs:
            diff = piv[a_] - piv[b_]
            for mtr in metrics:
                d = diff[mtr]
                comps.append({'architecture': arch, 'comparison': f'{a_} - {b_}', 'metric': mtr, 'mean': d.mean(),
                              'sd': d.std(), **{f'seed{s}': d.get(s) for s in S15.SEEDS}})
        g = 'val_B_mlp_gap'
        d_rb = piv['r_recon'][g] - piv['base'][g]
        d_hr = piv['r_honesty'][g] - piv['r_recon'][g]
        d_hs = piv['r_honesty'][g] - piv['r_samepersona'][g]
        d_cw = piv['r_honesty']['cw_B_mlp_gap'] - piv['r_recon']['cw_B_mlp_gap']
        tests = {'R1a_r_recon_lt_base': {'pass': consistent_lower(d_rb), 'diffs': d_rb.round(5).tolist()},
                 'R1b_r_honesty_lt_r_recon_consistent': {'pass': consistent_lower(d_hr), 'diffs': d_hr.round(5).tolist()},
                 'R1b_r_honesty_lt_r_samepersona_all_seeds': {'pass': bool((d_hs < 0).all()), 'diffs': d_hs.round(5).tolist()},
                 'R1_transfer_cw': {'pass': bool(d_cw.mean() < 0 and (d_cw < 0).sum() >= 2), 'diffs': d_cw.round(5).tolist()}}
        b = piv['base']
        guards = {}
        for arm in S15.ARMS:
            f = piv[arm]
            guards[arm] = {
                'owt_l0_within_5pct': bool(((f['owt_val_l0'] - b['owt_val_l0']).abs() / b['owt_val_l0'] <= 0.05).all()),
                'repe_val_l0_within_5pct': bool(((f['val_l0'] - b['val_l0']).abs() / b['val_l0'] <= 0.05).all()),
                'owt_ev_ge_base_minus_0.02': bool((f['owt_val_ev'] >= b['owt_val_ev'] - 0.02).all()),
                'dead_le_base_plus_0.05': bool((f['owt_val_dead_fraction'] <= b['owt_val_dead_fraction'] + 0.05).all()),
                'owt_splice_ce_le_base_plus_0.01_mean': bool(f['splice_owt_delta_ce'].mean() <= b['splice_owt_delta_ce'].mean() + 0.01),
                'got_geo_B_mlp_gap_le_base_plus_0.02_mean': bool(f['got_geographic_indep_mlp_gap_mean'].mean()
                                                                 <= b['got_geographic_indep_mlp_gap_mean'].mean() + 0.02)}
        r1b = tests['R1b_r_honesty_lt_r_recon_consistent']['pass'] and tests['R1b_r_honesty_lt_r_samepersona_all_seeds']['pass']
        g_ok = all(guards['r_honesty'].values()) and all(guards['r_recon'].values())
        verdict = ('SUPPORTED' if (r1b and tests['R1_transfer_cw']['pass'] and g_ok)
                   else 'NOT SUPPORTED' if not r1b else 'INCONCLUSIVE')
        crit[arch] = {'tests': tests, 'R1a_pass': tests['R1a_r_recon_lt_base']['pass'], 'R1b_pass': r1b,
                      'guards': guards, 'guards_pass_r_honesty_and_r_recon': g_ok,
                      'r_samepersona_guard_flag': not all(guards['r_samepersona'].values()), 'VERDICT': verdict}
    main_v = df[~df.variant.str.startswith('EXPLORATORY')]
    rho = spearmanr(main_v['val_l0'], main_v['val_B_mlp_gap'])
    crit['EXPLORATORY_spearman_val_B_mlp_gap_vs_repe_val_l0_all_main_saes'] = {'rho': float(rho.statistic), 'p': float(rho.pvalue),
                                                                              'n': int(len(main_v))}
    pd.DataFrame(comps).to_csv(OUT / 'paired_differences.csv', index=False)
    (OUT / 'criteria.json').write_text(json.dumps(crit, indent=2))
    (OUT / 'eval_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'data': 'validation + cross-wording validation; statement 326 excluded',
        'test_rows_used': False, 'mlp_probe_seeds': MLP_SEEDS, 'bootstrap_resamples': N_BOOT, 'n_saes': len(df),
        'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print(json.dumps({a: {'R1a': c['R1a_pass'], 'R1b': c['R1b_pass'], 'transfer': c['tests']['R1_transfer_cw']['pass'],
                          'guards': c['guards_pass_r_honesty_and_r_recon'], 'VERDICT': c['VERDICT']}
                      for a, c in crit.items() if a in S15.ARCHS}, indent=2))


if __name__ == '__main__':
    main()
