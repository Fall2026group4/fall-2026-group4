#!/usr/bin/env python3
"""Validation-only evaluation of the truth-aware pilot (script 11) against the pre-registration.

Run from repository root after `11_truth_aware_pilot.py --train`:
    python scripts/geometry_of_truth/12_truth_aware_pilot_evaluation.py

SAEs: 3 base (uninterrupted layer-6 TopK seeds 42/1042/2042) + 9 fine-tunes (3 variants x 3 seeds).
All probes are fitted on ORIGINAL activations; nothing is fitted on reconstructions except the labelled
decodability diagnostic. GoT test and RepE test rows are never scored.
Outputs: results/geometry_of_truth/truth_aware_pilot/{per_sae.csv, comparisons.csv, summary.csv, criteria.json,
eval_info.json}, figure results/geometry_of_truth/figures/truth_aware_pilot.png
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/geometry_of_truth/truth_aware_pilot'
MLP_SEEDS = (42, 123, 456, 789, 2026)


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (OUT / 'per_sae.csv').exists():
        raise SystemExit('per_sae.csv exists; refusing to overwrite.')
    p11 = load('scripts/geometry_of_truth/11_truth_aware_pilot.py', 'p11')
    g07 = load('scripts/geometry_of_truth/07_independent_truth_probe.py', 'g07')
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    ev = load('scripts/shared/all_layer_frozen_probe_evaluation.py', 'ev')
    prereg = json.loads((OUT / 'preregistration.json').read_text())
    assert sha(p11.SPLIT_CSV) == prereg['split_file_sha256']
    device = 'cuda'
    L = p11.LAYER
    t0 = time.time()

    # ---------------- GoT probes on ORIGINAL activations
    got = {}
    for domain in p11.DOMAINS:
        X, m = p11.load_domain(domain)
        y = m.label.to_numpy()
        A, B = (m.half == 'A').to_numpy(), (m.half == 'B').to_numpy()
        va = (m.split == 'val').to_numpy()
        assert not (m.split == 'test').to_numpy()[A | B | va].any()
        aucB, CB, logB = p11.select_C(X, y, B, va)
        _, CA, teacher = p11.select_C(X, y, A, va)
        dB = X[B & (y == 1)].astype(np.float64).mean(0) - X[B & (y == 0)].astype(np.float64).mean(0)
        mlps = []
        for s in MLP_SEEDS:
            g07.SEED = s
            torch.manual_seed(s)
            mlps.append(g07.train_probe(X[B], y[B], device))
        got[domain] = dict(X=X, m=m, y=y, A=A, B=B, va=va, logB=logB, CB=CB, teacher=teacher, CA=CA, dB=dB, mlps=mlps,
                           orig={'logB': logB.decision_function(X[va]), 'teacher': teacher.decision_function(X[va]),
                                 'diffmean': X[va] @ (dB / np.linalg.norm(dB)),
                                 'mlp': [g07.probe_scores(p, s_, X[va], device) for p, s_ in mlps]})
    repe_data, _ = ev.load_repe()
    rd = repe_data['honesty']
    repe_probes = ev.fit_original_probes(rd['X'][L], rd['y'], rd['split'])

    # ---------------- splicing data (validation only)
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

    saes = [('base', s, uninterrupted_checkpoint(L, 'topk', s)) for s in p11.SEEDS]
    saes += [(v, s, p11.CKPT / f'{v}_seed{s}' / 'finetuned_checkpoint.pt') for v in p11.VARIANTS for s in p11.SEEDS]
    rows = []
    for variant, seed, path in saes:
        sae, _ = fm.load_sae(path, device)
        r = {'variant': variant, 'seed': seed, 'checkpoint': Path(path).relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(path)}
        for domain, d in got.items():
            X, y, va, B = d['X'], d['y'], d['va'], d['B']
            Xh = fm.reconstruct(sae, X, device)[0].numpy()
            rec = fm.reconstruction_metrics(X[va], Xh[va], fm.reconstruct(sae, X[va], device)[1])
            yv = y[va]
            s_log = d['logB'].decision_function(Xh[va])
            s_t = d['teacher'].decision_function(Xh[va])
            s_dm = Xh[va] @ (d['dB'] / np.linalg.norm(d['dB']))
            mlp_gaps = [fm.safe_auroc(yv, o) - fm.safe_auroc(yv, g07.probe_scores(p, s_, Xh[va], device))
                        for (p, s_), o in zip(d['mlps'], d['orig']['mlp'])]
            # decodability diagnostic: C fixed to the B-probe's C (no selection on reconstructions)
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            refit = make_pipeline(StandardScaler(), LogisticRegression(C=d['CB'], max_iter=3000, random_state=42)).fit(Xh[B], y[B])
            dBh = Xh[B & (y == 1)].astype(np.float64).mean(0) - Xh[B & (y == 0)].astype(np.float64).mean(0)
            pre = f'got_{domain}_'
            r.update({pre + 'indep_logistic_gap': fm.safe_auroc(yv, d['orig']['logB']) - fm.safe_auroc(yv, s_log),
                      pre + 'indep_mlp_gap_mean': float(np.mean(mlp_gaps)), pre + 'indep_mlp_gap_sd': float(np.std(mlp_gaps, ddof=1)),
                      pre + 'indep_diffmean_gap': fm.safe_auroc(yv, d['orig']['diffmean']) - fm.safe_auroc(yv, s_dm),
                      pre + 'indep_logistic_score_corr': float(np.corrcoef(d['orig']['logB'], s_log)[0, 1]),
                      pre + 'TEACHER_gap_objective_only': fm.safe_auroc(yv, d['orig']['teacher']) - fm.safe_auroc(yv, s_t),
                      pre + 'DIAG_refit_on_recon_val_auroc': fm.safe_auroc(yv, refit.decision_function(Xh[va])),
                      pre + 'direction_cosine_B': fm.direction_fidelity(d['dB'], dBh)['direction_cosine'],
                      pre + 'val_relative_sq_error': rec['relative_sq_error'], pre + 'val_cosine': rec['cosine'],
                      pre + 'val_l0': rec['l0']})
        Xr = rd['X'][L]
        Xrh = fm.reconstruct(sae, Xr, device)[0].numpy()
        rv, rtr, yr = rd['split'] == 'validation', rd['split'] == 'train', rd['y']
        so, sr = ev.scores(repe_probes, Xr), ev.scores(repe_probes, Xrh)
        dr = Xrh[rtr & (yr == 1)].astype(np.float64).mean(0) - Xrh[rtr & (yr == 0)].astype(np.float64).mean(0)
        r.update(repe_logistic_gap=fm.safe_auroc(yr[rv], so['logistic'][rv]) - fm.safe_auroc(yr[rv], sr['logistic'][rv]),
                 repe_diffmean_gap=fm.safe_auroc(yr[rv], so['diffmean'][rv]) - fm.safe_auroc(yr[rv], sr['diffmean'][rv]),
                 repe_direction_cosine=fm.direction_fidelity(repe_probes['raw_direction'], dr)['direction_cosine'])
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
        print(f'{variant} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_sae.csv', index=False)

    # ---------------- paired comparisons (by seed)
    metrics = [c for c in df.columns if c not in ('variant', 'seed', 'checkpoint', 'checkpoint_sha256')]
    piv = {v: df[df.variant == v].set_index('seed')[metrics] for v in df.variant.unique()}
    comps = []
    for a_, b_ in [('truth_aware', 'base'), ('reconstruction_only', 'base'), ('random_teacher', 'base'),
                   ('truth_aware', 'reconstruction_only'), ('truth_aware', 'random_teacher'),
                   ('random_teacher', 'reconstruction_only')]:
        diff = piv[a_] - piv[b_]
        for mtr in metrics:
            d = diff[mtr]
            comps.append({'comparison': f'{a_} - {b_}', 'metric': mtr, 'mean': d.mean(), 'sd': d.std(),
                          'seed42': d.get(42), 'seed1042': d.get(1042), 'seed2042': d.get(2042)})
    comps = pd.DataFrame(comps)
    comps.to_csv(OUT / 'comparisons.csv', index=False)
    summ = df.groupby('variant')[metrics].agg(['mean', 'std'])
    summ.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False)

    # ---------------- pre-registered criteria
    def consistent_lower(mtr, other):
        d = (piv['truth_aware'][mtr] - piv[other][mtr])
        return bool((d < 0).all() and -d.mean() > 2 * d.std()), d.round(4).tolist()

    crit = {}
    for dom in ('geographic', 'numerical'):
        for pr in ('indep_logistic_gap', 'indep_mlp_gap_mean'):
            for other in ('reconstruction_only', 'random_teacher'):
                ok, d = consistent_lower(f'got_{dom}_{pr}', other)
                crit[f'{dom}:{pr}:truth_aware<{other}'] = {'pass': ok, 'paired_diffs_by_seed': d}
    primary = all(v['pass'] for k, v in crit.items() if k.startswith('geographic'))
    ta, ro, base = piv['truth_aware'], piv['reconstruction_only'], piv['base']
    g1 = bool((ta['splice_owt_delta_ce'] <= 1.10 * base['splice_owt_delta_ce']).all())
    g2 = bool(ta['splice_got_cities_pref_change'].mean() >= ro['splice_got_cities_pref_change'].mean() - 0.018)
    g3 = bool(ta['repe_diffmean_gap'].mean() <= ro['repe_diffmean_gap'].mean() + 0.02)
    result = {'tests': crit, 'primary_geographic_pass': primary,
              'secondary_numerical_pass': all(v['pass'] for k, v in crit.items() if k.startswith('numerical')),
              'guard_owt_delta_ce_le_1.10x_base_all_seeds': g1,
              'guard_cities_pref_not_worse_than_recon_only_by_2_pairs': g2,
              'guard_repe_diffmean_gap_not_worse_than_recon_only_by_0.02': g3,
              'SUCCESS': bool(primary and g1 and g2 and g3)}
    (OUT / 'criteria.json').write_text(json.dumps(result, indent=2))
    (OUT / 'eval_info.json').write_text(json.dumps({'script_sha256': sha(__file__), 'data': 'validation only',
                                                    'test_rows_used': False, 'mlp_probe_seeds': MLP_SEEDS,
                                                    'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
