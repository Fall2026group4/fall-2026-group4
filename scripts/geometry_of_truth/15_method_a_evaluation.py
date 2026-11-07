#!/usr/bin/env python3
"""Validation-only evaluation of Method A at layer 6 (script 14) against its pre-registration.

Run from repository root after `14_method_a_reconstruction_finetune.py --train`:
    python scripts/geometry_of_truth/15_method_a_evaluation.py

29 SAEs: 9 uninterrupted bases + P1 JumpReLU base, Method A (relu/jumprelu new, topk = pilot reconstruction-only
checkpoints, P1-base fine-tune), OWT-only controls. The TopK reproduction-check run is not evaluated.
All probes are fitted on ORIGINAL activations (GoT: train groups B; RepE: train). GoT/RepE test rows are never scored.
Outputs (results/geometry_of_truth/method_a_layer6/): per_sae.csv, summary.csv, comparisons.csv, criteria.json,
eval_info.json; figure results/geometry_of_truth/figures/method_a_layer6.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

MLP_SEEDS = (42, 123, 456, 789, 2026)


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sae_list(s14):
    L, out = s14.LAYER, []
    for a in s14.ARCHS:
        for s in s14.SEEDS:
            out.append(('base', a, s, uninterrupted_checkpoint(L, a, s)))
            out.append(('owt_only', a, s, s14.CKPT / f'{a}_owt_only_seed{s}' / 'finetuned_checkpoint.pt'))
            ma = (Path(str(s14.PILOT_TOPK).format(s=s)) if a == 'topk'
                  else s14.CKPT / f'{a}_got_replay_seed{s}' / 'finetuned_checkpoint.pt')
            out.append(('method_a', a, s, ma))
    out.append(('p1_base', 'jumprelu', 42, s14.P1_JUMPRELU))
    out.append(('p1_method_a', 'jumprelu', 42, s14.CKPT / 'jumprelu_p1base_got_replay_seed42' / 'finetuned_checkpoint.pt'))
    return out


def main():
    s14 = load('scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py', 's14')
    P11, OUT, L = s14.P11, s14.OUT, s14.LAYER
    if (OUT / 'per_sae.csv').exists():
        raise SystemExit('per_sae.csv exists; refusing to overwrite.')
    g07 = load('scripts/geometry_of_truth/07_independent_truth_probe.py', 'g07')
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    ev = load('scripts/shared/all_layer_frozen_probe_evaluation.py', 'ev')
    device = 'cuda'
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    got = {}
    for domain in P11.DOMAINS:
        X, m = P11.load_domain(domain)
        y = m.label.to_numpy()
        B, va = (m.half == 'B').to_numpy(), (m.split == 'val').to_numpy()
        _, CB, logB = P11.select_C(X, y, B, va)
        dB = X[B & (y == 1)].astype(np.float64).mean(0) - X[B & (y == 0)].astype(np.float64).mean(0)
        mlps = []
        for s in MLP_SEEDS:
            g07.SEED = s
            torch.manual_seed(s)
            mlps.append(g07.train_probe(X[B], y[B], device))
        got[domain] = dict(X=X, y=y, B=B, va=va, logB=logB, CB=CB, dB=dB, mlps=mlps,
                           o_log=logB.decision_function(X[va]), o_dm=X[va] @ (dB / np.linalg.norm(dB)),
                           o_mlp=[g07.probe_scores(p, s_, X[va], device) for p, s_ in mlps])
    rd = ev.load_repe()[0]['honesty']
    repe_probes = ev.fit_original_probes(rd['X'][L], rd['y'], rd['split'])

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    S = sp.Splicer(model, device)
    owt_seqs, _ = sp.load_owt_validation()
    got_v = sp.load_got_validation(model)
    repe_v = sp.load_repe_validation(model)
    owt_orig = sp.owt_original(S, owt_seqs)
    got_orig = sp.got_original(S, got_v)
    toks, lens = sp.pad(list(repe_v.tokens), 50256)
    repe_orig = S.logprobs(toks, lens)[torch.arange(len(repe_v), device=device), (lens - 1).to(device)]
    cities_orig = sp.got_eval(S, got_v, 0, lambda x: x.clone(), got_orig)['cities_truth_pair_accuracy']
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_val = o3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L).to(device)

    rows = []
    for variant, arch, seed, path in sae_list(s14):
        sae, _ = fm.load_sae(path, device, architecture=arch)
        r = {'variant': variant, 'architecture': arch, 'seed': seed, 'checkpoint': Path(path).relative_to(REPO).as_posix(),
             'checkpoint_sha256': sha(path)}
        for domain, d in got.items():
            X, y, va, B = d['X'], d['y'], d['va'], d['B']
            Xh, l0_all, _ = fm.reconstruct(sae, X, device)
            Xh = Xh.numpy()
            rec = fm.reconstruction_metrics(X[va], Xh[va], l0_all[torch.as_tensor(va)])
            yv = y[va]
            s_log = d['logB'].decision_function(Xh[va])
            mlp_gaps = [fm.safe_auroc(yv, o) - fm.safe_auroc(yv, g07.probe_scores(p, s_, Xh[va], device))
                        for (p, s_), o in zip(d['mlps'], d['o_mlp'])]
            refit = make_pipeline(StandardScaler(), LogisticRegression(C=d['CB'], max_iter=3000, random_state=42)).fit(Xh[B], y[B])
            dBh = Xh[B & (y == 1)].astype(np.float64).mean(0) - Xh[B & (y == 0)].astype(np.float64).mean(0)
            pre = f'got_{domain}_'
            r.update({pre + 'indep_logistic_gap': fm.safe_auroc(yv, d['o_log']) - fm.safe_auroc(yv, s_log),
                      pre + 'indep_mlp_gap_mean': float(np.mean(mlp_gaps)), pre + 'indep_mlp_gap_sd': float(np.std(mlp_gaps, ddof=1)),
                      pre + 'indep_diffmean_gap': fm.safe_auroc(yv, d['o_dm']) - fm.safe_auroc(yv, Xh[va] @ (d['dB'] / np.linalg.norm(d['dB']))),
                      pre + 'indep_logistic_score_corr': float(np.corrcoef(d['o_log'], s_log)[0, 1]),
                      pre + 'DIAG_refit_on_recon_val_auroc': fm.safe_auroc(yv, refit.decision_function(Xh[va])),
                      pre + 'direction_cosine_B': fm.direction_fidelity(d['dB'], dBh)['direction_cosine'],
                      pre + 'val_relative_sq_error': rec['relative_sq_error'], pre + 'val_cosine': rec['cosine'],
                      pre + 'val_l0': rec['l0']})
        Xr = rd['X'][L]
        Xrh, l0r, _ = fm.reconstruct(sae, Xr, device)
        Xrh = Xrh.numpy()
        rv, rtr, yr = rd['split'] == 'validation', rd['split'] == 'train', rd['y']
        so, sr = ev.scores(repe_probes, Xr), ev.scores(repe_probes, Xrh)
        dr = Xrh[rtr & (yr == 1)].astype(np.float64).mean(0) - Xrh[rtr & (yr == 0)].astype(np.float64).mean(0)
        rrec = fm.reconstruction_metrics(Xr[rv], Xrh[rv], l0r[torch.as_tensor(rv)])
        r.update(repe_logistic_gap=fm.safe_auroc(yr[rv], so['logistic'][rv]) - fm.safe_auroc(yr[rv], sr['logistic'][rv]),
                 repe_diffmean_gap=fm.safe_auroc(yr[rv], so['diffmean'][rv]) - fm.safe_auroc(yr[rv], sr['diffmean'][rv]),
                 repe_direction_cosine=fm.direction_fidelity(repe_probes['raw_direction'], dr)['direction_cosine'],
                 repe_val_relative_sq_error=rrec['relative_sq_error'], repe_val_l0=rrec['l0'])
        xh, l0, act = fm.reconstruct(sae, owt_val, device)
        rec = fm.reconstruction_metrics(owt_val.cpu(), xh, l0)
        r.update(owt_val_ev=rec['explained_variance'], owt_val_mse=rec['mse'], owt_val_cosine=rec['cosine'],
                 owt_val_l0=rec['l0'], owt_val_dead_fraction=fm.dead_fraction(act))
        fn = sp.sae_fn(sae)
        o = sp.owt_eval(S, owt_seqs, L, fn, owt_orig)
        gg = sp.got_eval(S, got_v, L, fn, got_orig)
        rp = sp.repe_eval(S, repe_v, L, fn, repe_orig, 'final')
        r.update(splice_owt_delta_ce=o['delta_ce'], splice_owt_kl=o['kl_mean'], splice_got_statement_kl=gg['statement_kl_mean'],
                 splice_got_cities_pref_change=gg['cities_truth_pair_accuracy'] - cities_orig,
                 splice_repe_final_kl=rp['final_kl_mean'], splice_repe_persona_shift_cosine=rp['persona_shift_cosine'])
        rows.append(r)
        print(f'{variant} {arch} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_sae.csv', index=False)

    metrics = [c for c in df.columns if c not in ('variant', 'architecture', 'seed', 'checkpoint', 'checkpoint_sha256')]
    summ = df.groupby(['architecture', 'variant'])[metrics].agg(['mean', 'std'])
    summ.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False)
    comps, crit = [], {}
    for arch in s14.ARCHS:
        piv = {v: df[(df.architecture == arch) & (df.variant == v)].set_index('seed')[metrics] for v in ('base', 'owt_only', 'method_a')}
        for a_, b_ in [('method_a', 'base'), ('method_a', 'owt_only'), ('owt_only', 'base')]:
            diff = piv[a_] - piv[b_]
            for mtr in metrics:
                d = diff[mtr]
                comps.append({'architecture': arch, 'comparison': f'{a_} - {b_}', 'metric': mtr, 'mean': d.mean(), 'sd': d.std(),
                              **{f'seed{s}': d.get(s) for s in s14.SEEDS}})
        tests = {}
        for dom in ('geographic', 'numerical'):
            for pr in ('indep_mlp_gap_mean', 'indep_logistic_gap'):
                for other in ('base', 'owt_only'):
                    d = piv['method_a'][f'got_{dom}_{pr}'] - piv[other][f'got_{dom}_{pr}']
                    tests[f'{dom}:{pr}:method_a<{other}'] = {
                        'pass': bool((d < 0).all() and -d.mean() > 2 * d.std()), 'paired_diffs_by_seed': d.round(4).tolist()}
        ma, b = piv['method_a'], piv['base']
        guards = {
            'l0_matched_owt_val_within_5pct': bool(((ma['owt_val_l0'] - b['owt_val_l0']).abs() / b['owt_val_l0'] <= 0.05).all()),
            'owt_splice_delta_ce_le_1.10x_base': bool((ma['splice_owt_delta_ce'] <= 1.10 * b['splice_owt_delta_ce']).all()),
            'owt_val_ev_ge_base_minus_0.005': bool((ma['owt_val_ev'] >= b['owt_val_ev'] - 0.005).all()),
            'cities_pref_ge_base_minus_0.018': bool(ma['splice_got_cities_pref_change'].mean() >= b['splice_got_cities_pref_change'].mean() - 0.018),
            'repe_diffmean_gap_le_base_plus_0.02': bool(ma['repe_diffmean_gap'].mean() <= b['repe_diffmean_gap'].mean() + 0.02)}
        primary = tests['geographic:indep_mlp_gap_mean:method_a<base']['pass'] and tests['geographic:indep_mlp_gap_mean:method_a<owt_only']['pass']
        crit[arch] = {'tests': tests, 'primary_pass': primary, 'guards': guards, 'SUCCESS': bool(primary and all(guards.values()))}
    pd.DataFrame(comps).to_csv(OUT / 'comparisons.csv', index=False)
    (OUT / 'criteria.json').write_text(json.dumps(crit, indent=2))
    (OUT / 'eval_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'data': 'validation only', 'test_rows_used': False, 'mlp_probe_seeds': MLP_SEEDS,
        'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print(json.dumps({k: {'primary': v['primary_pass'], 'guards': v['guards'], 'SUCCESS': v['SUCCESS']} for k, v in crit.items()}, indent=2))


if __name__ == '__main__':
    main()
