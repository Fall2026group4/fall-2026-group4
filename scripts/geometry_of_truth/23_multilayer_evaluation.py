#!/usr/bin/env python3
"""IMP-L evaluation (validation only) for scripts/geometry_of_truth/22_multilayer_finetune.py.

For each layer in (3, 8, 9) and each of 36 SAEs (base / owt_only / method_a / kl x ReLU, TopK, JumpReLU x 3 seeds):
  reconstruction  OWT validation EV, MSE, cosine, L0, dead fraction; GoT geographic + numerical validation relative
                  squared error and L0; RepE-LM persona validation L0
  information     GoT B-group probes fitted on ORIGINAL activations of train groups B and applied frozen (MLP x 5
                  seeds, logistic with C chosen on original validation, mass-mean), the layer-6 protocol of script 15;
                  M2-style re-trained logistic gap (grouped C selection on train, src/evaluation/information_levels.py)
                  on GoT geographic and on RepE-LM persona (T1-T4, H vs U)
  behaviour       splice into GPT-2 at blocks.L.hook_resid_pre (non-BOS): OWT delta CE and KL, GoT statement KL,
                  cities truth preference (scripts/shared/spliced_reconstruction_evaluation.py)
Then paired comparisons and the pre-registered criteria (results/geometry_of_truth/multilayer_finetune/preregistration.json).
Test rows are never loaded: GoT rows with split == 'test' are dropped before scoring; RepE-LM has no test rows.

Run from the repository root after training:
    python scripts/geometry_of_truth/23_multilayer_evaluation.py [--layers 8]
Code check before training (base SAEs only, one layer, written to a separate folder):
    python scripts/geometry_of_truth/23_multilayer_evaluation.py --smoke --smoke-out DIR
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import information_levels as il  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

MLP_SEEDS = (42, 123, 456, 789, 2026)
T_975_DF2 = 4.302652729911275          # two-sided 95 % t quantile, 2 degrees of freedom (3 seeds)
ARMS_ALL = ('base', 'owt_only', 'method_a', 'kl')


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


S22 = load('scripts/geometry_of_truth/22_multilayer_finetune.py', 's22')
P11 = S22.P11


def sae_list(layer, smoke):
    out = []
    for a in S22.ARCHS:
        for s in S22.SEEDS:
            out.append(('base', a, s, uninterrupted_checkpoint(layer, a, s)))
            if not smoke:
                out += [(arm, a, s, S22.ckpt_path(layer, a, arm, s)) for arm in S22.ARMS]
    return out


def repe_lm_persona(layer):
    d = pd.read_csv(REPO / 'data/repe/repe_lm/prompts.csv')
    assert set(d.split) <= {'train', 'validation'}, 'RepE-LM must not contain test rows'
    m = (d.template.isin(['T1', 'T2', 'T3', 'T4']) & d.variant.isin(['H', 'U'])).to_numpy()
    X = np.load(REPO / f'data/repe/activations/repe_lm/final_layer{layer}.npy')[m].astype(np.float32)
    return X, d.label.to_numpy()[m], d.split.to_numpy()[m], d.norm_group.to_numpy()[m]


def evaluate_layer(L, saes, model, device):
    g07 = load('scripts/geometry_of_truth/07_independent_truth_probe.py', 'g07')
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    S22.set_layer(L)
    got = {}
    for domain in P11.DOMAINS:
        X, m = P11.load_domain(domain)
        keep = (m.split != 'test').to_numpy()                 # drop test rows before anything is scored
        X, m = X[keep], m[keep].reset_index(drop=True)
        y = m.label.to_numpy()
        B, va, tr = (m.half == 'B').to_numpy(), (m.split == 'val').to_numpy(), (m.split == 'train').to_numpy()
        _, CB, logB = P11.select_C(X, y, B, va)
        dB = X[B & (y == 1)].astype(np.float64).mean(0) - X[B & (y == 0)].astype(np.float64).mean(0)
        mlps = []
        for s in MLP_SEEDS:
            g07.SEED = s
            torch.manual_seed(s)
            mlps.append(g07.train_probe(X[B], y[B], device))
        g = m.group_id.to_numpy()
        refit_orig = il.fit_logistic(X[tr], y[tr], g[tr])[0] if domain == 'geographic' else None
        got[domain] = dict(X=X, y=y, B=B, va=va, tr=tr, g=g, logB=logB, CB=CB, dB=dB, mlps=mlps,
                           o_log=logB.decision_function(X[va]), o_dm=X[va] @ (dB / np.linalg.norm(dB)),
                           o_mlp=[g07.probe_scores(p, s_, X[va], device) for p, s_ in mlps],
                           o_refit=(il.auroc(y[va], refit_orig.decision_function(X[va])) if refit_orig is not None else None))
    Xp, yp, spl, gp = repe_lm_persona(L)
    ptr, pva = spl == 'train', spl == 'validation'
    persona_orig = il.auroc(yp[pva], il.fit_logistic(Xp[ptr], yp[ptr], gp[ptr])[0].decision_function(Xp[pva]))

    S = sp.Splicer(model, device)
    owt_seqs, _ = sp.load_owt_validation()
    got_v = sp.load_got_validation(model)
    owt_orig = sp.owt_original(S, owt_seqs)
    got_orig = sp.got_original(S, got_v)
    cities_orig = sp.got_eval(S, got_v, 0, lambda x: x.clone(), got_orig)['cities_truth_pair_accuracy']
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    owt_val = o3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L).to(device)

    rows = []
    t0 = time.time()
    for arm, arch, seed, path in saes:
        sae, _ = fm.load_sae(path, device, architecture=arch)
        sae.eval()
        r = {'layer': L, 'arm': arm, 'architecture': arch, 'seed': seed,
             'checkpoint': Path(path).relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(path)}
        for domain, d in got.items():
            X, y, va, B = d['X'], d['y'], d['va'], d['B']
            Xh, l0_all, _ = fm.reconstruct(sae, X, device)
            Xh = Xh.numpy()
            rec = fm.reconstruction_metrics(X[va], Xh[va], l0_all[torch.as_tensor(va)])
            yv = y[va]
            s_log = d['logB'].decision_function(Xh[va])
            mlp_gaps = [fm.safe_auroc(yv, o) - fm.safe_auroc(yv, g07.probe_scores(p, s_, Xh[va], device))
                        for (p, s_), o in zip(d['mlps'], d['o_mlp'])]
            pre = f'got_{domain}_'
            r.update({pre + 'indep_mlp_gap_mean': float(np.mean(mlp_gaps)), pre + 'indep_mlp_gap_sd': float(np.std(mlp_gaps, ddof=1)),
                      pre + 'orig_mlp_auroc_mean': float(np.mean([fm.safe_auroc(yv, o) for o in d['o_mlp']])),
                      pre + 'indep_logistic_gap': fm.safe_auroc(yv, d['o_log']) - fm.safe_auroc(yv, s_log),
                      pre + 'orig_logistic_auroc': fm.safe_auroc(yv, d['o_log']),
                      pre + 'indep_diffmean_gap': fm.safe_auroc(yv, d['o_dm']) - fm.safe_auroc(yv, Xh[va] @ (d['dB'] / np.linalg.norm(d['dB']))),
                      pre + 'val_relative_sq_error': rec['relative_sq_error'], pre + 'val_cosine': rec['cosine'],
                      pre + 'val_l0': rec['l0']})
            if d['o_refit'] is not None:
                tr, g = d['tr'], d['g']
                refit = il.fit_logistic(Xh[tr], y[tr], g[tr])[0]
                r[pre + 'refit_logistic_gap'] = d['o_refit'] - il.auroc(yv, refit.decision_function(Xh[va]))
        Xph, l0p, _ = fm.reconstruct(sae, Xp, device)
        Xph = Xph.numpy()
        r['repe_lm_persona_val_l0'] = float(l0p[torch.as_tensor(pva)].float().mean())
        r['repe_lm_persona_refit_logistic_gap'] = persona_orig - il.auroc(
            yp[pva], il.fit_logistic(Xph[ptr], yp[ptr], gp[ptr])[0].decision_function(Xph[pva]))
        xh, l0, act = fm.reconstruct(sae, owt_val, device)
        rec = fm.reconstruction_metrics(owt_val.cpu(), xh, l0)
        r.update(owt_val_ev=rec['explained_variance'], owt_val_mse=rec['mse'], owt_val_cosine=rec['cosine'],
                 owt_val_l0=rec['l0'], owt_val_dead_fraction=fm.dead_fraction(act))
        fn = sp.sae_fn(sae)
        o = sp.owt_eval(S, owt_seqs, L, fn, owt_orig)
        gg = sp.got_eval(S, got_v, L, fn, got_orig)
        r.update(splice_owt_delta_ce=o['delta_ce'], splice_owt_kl=o['kl_mean'], splice_got_statement_kl=gg['statement_kl_mean'],
                 splice_got_cities_pref_change=gg['cities_truth_pair_accuracy'] - cities_orig)
        rows.append(r)
        print(f'L{L} {arm} {arch} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
    return pd.DataFrame(rows)


LOWER_BETTER = ('gap', 'error', 'kl', 'delta_ce', 'mse', 'dead')


def compare(df):
    metrics = [c for c in df.columns if c not in ('layer', 'arm', 'architecture', 'seed', 'checkpoint', 'checkpoint_sha256')
               and not c.endswith('_sd') and df[c].dtype != object]
    comps, crit = [], {}
    for (L, arch), sub in df.groupby(['layer', 'architecture']):
        piv = {v: sub[sub.arm == v].set_index('seed')[metrics] for v in ARMS_ALL}
        for a_, b_ in [('owt_only', 'base'), ('method_a', 'base'), ('method_a', 'owt_only'), ('kl', 'base'),
                       ('kl', 'owt_only'), ('kl', 'method_a')]:
            diff = piv[a_] - piv[b_]
            for mtr in metrics:
                dd = diff[mtr]
                comps.append({'layer': L, 'architecture': arch, 'comparison': f'{a_} - {b_}', 'metric': mtr,
                              'base_mean': piv[b_][mtr].mean(), 'arm_mean': piv[a_][mtr].mean(),
                              'mean_diff': dd.mean(), 'sd_diff': dd.std(), 'ci95_low': dd.mean() - T_975_DF2 * dd.std() / np.sqrt(len(dd)),
                              'ci95_high': dd.mean() + T_975_DF2 * dd.std() / np.sqrt(len(dd)),
                              'pct_change': 100 * dd.mean() / piv[b_][mtr].mean() if piv[b_][mtr].mean() != 0 else np.nan,
                              **{f'seed{s}': dd.get(s) for s in S22.SEEDS}})

        def consistent(a_, b_, mtr):                    # lower is better for every criterion metric
            dd = piv[a_][mtr] - piv[b_][mtr]
            return {'pass': bool((dd < 0).all() and -dd.mean() > 2 * dd.std()), 'paired_diffs_by_seed': dd.round(5).tolist()}

        def wrong_way(a_, b_, mtr):
            dd = piv[a_][mtr] - piv[b_][mtr]
            return bool((dd > 0).all() and dd.mean() > 2 * dd.std())

        b, ma, kl = piv['base'], piv['method_a'], piv['kl']
        mlp = 'got_geographic_indep_mlp_gap_mean'

        def guards(arm, ev_ref, ev_tol):
            x = piv[arm]
            return {'l0_within_5pct_of_base': bool(((x.owt_val_l0 - b.owt_val_l0).abs() / b.owt_val_l0 <= 0.05).all()),
                    f'owt_ev_ge_ref_minus_{ev_tol}': bool((x.owt_val_ev >= ev_ref.owt_val_ev - ev_tol).all()),
                    'owt_delta_ce_le_1.10x_base': bool((x.splice_owt_delta_ce <= 1.10 * b.splice_owt_delta_ce).all()),
                    'cities_pref_ge_base_minus_0.018': bool(x.splice_got_cities_pref_change.mean()
                                                            >= b.splice_got_cities_pref_change.mean() - 0.018)}
        g_ma, g_kl = guards('method_a', b, 0.005), guards('kl', ma, 0.02)
        t_ma = {'vs_base': consistent('method_a', 'base', mlp), 'vs_owt_only': consistent('method_a', 'owt_only', mlp)}
        b_kl = {'owt_kl': consistent('kl', 'method_a', 'splice_owt_kl'),
                'got_statement_kl': consistent('kl', 'method_a', 'splice_got_statement_kl')}
        t_kl = consistent('kl', 'method_a', mlp)
        informative = bool(b['got_geographic_orig_mlp_auroc_mean'].mean() >= 0.70)

        def classify(primary, g, wrong):
            if not all(g.values()):
                return 'NOT CREDITABLE (guard failed)'
            if wrong:
                return 'REGRESSION'
            return 'PRELIMINARY IMPROVEMENT' if primary else 'NO IMPROVEMENT'
        crit[f'layer{L}_{arch}'] = {
            'truth_informative_at_layer': informative,
            'method_a': {'T_method_a': t_ma, 'guards': g_ma,
                         'class_truth': classify(t_ma['vs_base']['pass'] and t_ma['vs_owt_only']['pass'], g_ma,
                                                 wrong_way('method_a', 'base', mlp))},
            'kl': {'B_kl': b_kl, 'T_kl': t_kl, 'guards': g_kl,
                   'class_behaviour': classify(b_kl['owt_kl']['pass'] and b_kl['got_statement_kl']['pass'], g_kl,
                                               wrong_way('kl', 'method_a', 'splice_owt_kl')),
                   'class_truth': classify(t_kl['pass'], g_kl, wrong_way('kl', 'method_a', mlp))}}
    return pd.DataFrame(comps), crit


TABLE_METRICS = [('got_geographic_indep_mlp_gap_mean', 'GoT truth gap, B-MLP'), ('got_geographic_indep_logistic_gap', 'GoT truth gap, B-logistic'),
                 ('got_geographic_refit_logistic_gap', 'GoT re-trained logistic gap'), ('repe_lm_persona_refit_logistic_gap', 'Persona re-trained logistic gap'),
                 ('got_geographic_val_relative_sq_error', 'GoT relative error'), ('owt_val_ev', 'OWT EV'),
                 ('splice_owt_delta_ce', 'OWT spliced ΔCE'), ('splice_owt_kl', 'OWT spliced KL'),
                 ('splice_got_statement_kl', 'GoT statement KL'), ('owt_val_l0', 'OWT L0'),
                 ('got_geographic_val_l0', 'GoT L0'), ('repe_lm_persona_val_l0', 'RepE-LM L0')]


def improvement_table(comps):
    keep = comps[comps.comparison.isin(['owt_only - base', 'method_a - base', 'kl - base', 'method_a - owt_only', 'kl - method_a'])
                 & comps.metric.isin([m for m, _ in TABLE_METRICS])].copy()
    keep['metric_label'] = keep.metric.map(dict(TABLE_METRICS))
    return keep[['layer', 'architecture', 'comparison', 'metric', 'metric_label', 'base_mean', 'arm_mean', 'mean_diff',
                 'ci95_low', 'ci95_high', 'pct_change', 'seed42', 'seed1042', 'seed2042']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, nargs='+', default=list(S22.LAYERS))
    ap.add_argument('--smoke', action='store_true', help='base SAEs only, first requested layer, separate output dir')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    out = S22.OUT
    if a.smoke:
        if a.smoke_out is None:
            raise SystemExit('--smoke needs --smoke-out')
        out, a.layers = a.smoke_out, a.layers[:1]
        out.mkdir(parents=True, exist_ok=True)
    else:
        hashes = json.loads((out / 'pretraining_hashes.json').read_text())['sha256']
        bad = [p for p, h in hashes.items() if sha(REPO / p) != h]
        if bad:
            raise SystemExit(f'files changed since pre-registration: {bad}')
    clash = [str(out / f'per_sae_layer{L}.csv') for L in a.layers if (out / f'per_sae_layer{L}.csv').exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite: {clash}')
    missing = [str(p) for L in a.layers for *_, p in sae_list(L, a.smoke) if not Path(p).exists()]
    if missing:
        raise SystemExit(f'missing checkpoints (train first): {missing[:5]} ... ({len(missing)})')
    device = 'cuda'
    model = S22.load_gpt2(device)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for L in a.layers:
        df = evaluate_layer(L, sae_list(L, a.smoke), model, device)
        df.to_csv(out / f'per_sae_layer{L}.csv', index=False)
    info = {'script_sha256': sha(__file__), 'layers': a.layers, 'smoke': a.smoke, 'data': 'validation only',
            'test_rows_used': 0, 'mlp_probe_seeds': MLP_SEEDS,
            'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
            'runtime_seconds': round(time.time() - t0, 1)}
    (out / ('smoke_info.json' if a.smoke else f'eval_info_layers{"_".join(map(str, a.layers))}.json')).write_text(json.dumps(info, indent=2))
    if a.smoke:
        print(json.dumps(info, indent=2))
        return
    done = [L for L in S22.LAYERS if (out / f'per_sae_layer{L}.csv').exists()]
    df = pd.concat([pd.read_csv(out / f'per_sae_layer{L}.csv') for L in done], ignore_index=True)
    comps, crit = compare(df)
    comps.to_csv(out / 'comparisons.csv', index=False)
    improvement_table(comps).to_csv(out / 'improvement_table.csv', index=False)
    metrics = [c for c in df.columns if c not in ('layer', 'arm', 'architecture', 'seed', 'checkpoint', 'checkpoint_sha256')]
    summ = df.groupby(['layer', 'architecture', 'arm'])[metrics].agg(['mean', 'std'])
    summ.columns = [f'{x}_{"sd" if y == "std" else y}' for x, y in summ.columns]
    summ.reset_index().to_csv(out / 'summary.csv', index=False)
    (out / 'criteria.json').write_text(json.dumps({'layers_evaluated': done, **crit}, indent=2))
    print(json.dumps({k: {'method_a_truth': v['method_a']['class_truth'], 'kl_behaviour': v['kl']['class_behaviour'],
                          'kl_truth': v['kl']['class_truth']} for k, v in crit.items()}, indent=2))


if __name__ == '__main__':
    main()
