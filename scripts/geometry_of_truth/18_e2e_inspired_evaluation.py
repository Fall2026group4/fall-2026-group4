#!/usr/bin/env python3
"""Validation-only evaluation of the E2E-inspired layer-6 fine-tunes (script 17) against the pre-registration.

Run from repository root:  python scripts/geometry_of_truth/18_e2e_inspired_evaluation.py

36 SAEs = {base, Method A, Method B, Method C} x {relu, topk, jumprelu} x seeds {42, 1042, 2042}.
Metrics: scripts/shared/layer6_faithfulness_eval.py (no test rows; probes on original activations).
Outputs (results/geometry_of_truth/e2e_inspired_layer6/): per_sae.csv, summary.csv, comparisons.csv, criteria.json,
eval_info.json.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def consistent(d, lower=True):
    d = d if lower else -d          # d = method - reference; better means negative after this flip
    return bool((d < 0).all() and -d.mean() > 2 * d.std())


def main():
    s17 = load('scripts/geometry_of_truth/17_e2e_inspired_finetune.py', 's17')
    ev6 = load('scripts/shared/layer6_faithfulness_eval.py', 'ev6')
    OUT = s17.OUT
    if (OUT / 'per_sae.csv').exists():
        raise SystemExit('per_sae.csv exists; refusing to overwrite.')
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    ctx = ev6.build_context('cuda')
    rows = []
    for ar in s17.ARCHS:
        for s in s17.SEEDS:
            items = [('base', uninterrupted_checkpoint(s17.LAYER, ar, s)), ('A', s17.METHOD_A[(ar, s)])]
            items += [(m, s17.CKPT / f'{ar}_method{m}_seed{s}' / 'finetuned_checkpoint.pt') for m in s17.METHODS]
            for m, path in items:
                sae, _ = fm.load_sae(path, 'cuda', architecture=ar)
                sae.eval()
                r = {'method': m, 'architecture': ar, 'seed': s, 'checkpoint': Path(path).relative_to(REPO).as_posix(),
                     'checkpoint_sha256': sha(path)}
                r.update(ev6.evaluate_sae(sae, ctx))
                rows.append(r)
                print(f'{m} {ar} seed {s} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_sae.csv', index=False)
    metrics = [c for c in df.columns if c not in ('method', 'architecture', 'seed', 'checkpoint', 'checkpoint_sha256')]
    summ = df.groupby(['architecture', 'method'])[metrics].agg(['mean', 'std'])
    summ.columns = [f'{a}_{"sd" if b == "std" else b}' for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False)

    comps, crit = [], {}
    mon = pd.read_csv(OUT / 'training_monitor.csv')
    for ar in s17.ARCHS:
        piv = {m: df[(df.architecture == ar) & (df.method == m)].set_index('seed')[metrics] for m in ('base', 'A', 'B', 'C')}
        for a_, b_ in [('B', 'A'), ('C', 'A'), ('C', 'B'), ('A', 'base'), ('B', 'base'), ('C', 'base')]:
            diff = piv[a_] - piv[b_]
            for mtr in metrics:
                d = diff[mtr]
                comps.append({'architecture': ar, 'comparison': f'{a_} - {b_}', 'metric': mtr, 'mean': d.mean(), 'sd': d.std(),
                              **{f'seed{s}': d.get(s) for s in s17.SEEDS}})
        for m in s17.METHODS:
            M, A, base = piv[m], piv['A'], piv['base']
            tests = {'P1_owt_splice_kl': consistent(M['splice_owt_kl'] - A['splice_owt_kl']),
                     'P2_got_statement_kl': consistent(M['splice_got_statement_kl'] - A['splice_got_statement_kl']),
                     'H_truth_geographic_mlp_gap': consistent(M['got_geographic_indep_mlp_gap_mean'] - A['got_geographic_indep_mlp_gap_mean'])}
            guards = {'l0_owt_within_5pct_of_base': bool(((M['owt_val_l0'] - base['owt_val_l0']).abs() / base['owt_val_l0'] <= 0.05).all()),
                      'owt_ev_ge_A_minus_0.02': bool((M['owt_val_ev'] >= A['owt_val_ev'] - 0.02).all()),
                      'cities_pref_ge_A_minus_0.018': bool(M['splice_got_cities_pref_change'].mean() >= A['splice_got_cities_pref_change'].mean() - 0.018),
                      'repe_diffmean_gap_le_A_plus_0.02': bool(M['repe_diffmean_gap'].mean() <= A['repe_diffmean_gap'].mean() + 0.02),
                      'got_geo_logistic_gap_le_A_plus_0.02': bool(M['got_geographic_indep_logistic_gap'].mean() <= A['got_geographic_indep_logistic_gap'].mean() + 0.02)}
            behav = tests['P1_owt_splice_kl'] and tests['P2_got_statement_kl'] and all(guards.values())
            # pre-registered longer-budget flag (training monitor, TRAIN data only)
            still = 0
            for s in s17.SEEDS:
                g = mon[(mon.method == m) & (mon.architecture == ar) & (mon.seed == s)].set_index('step')['monitor_kl_owt_train']
                if g[500] <= 0.95 * g[0] and g[500] <= 0.98 * g[400]:
                    still += 1
            crit[f'{ar}:{m}'] = {'tests': tests, 'paired_diffs': {
                'owt_splice_kl': (M['splice_owt_kl'] - A['splice_owt_kl']).round(5).tolist(),
                'got_statement_kl': (M['splice_got_statement_kl'] - A['splice_got_statement_kl']).round(5).tolist(),
                'geographic_mlp_gap': (M['got_geographic_indep_mlp_gap_mean'] - A['got_geographic_indep_mlp_gap_mean']).round(4).tolist()},
                'guards': guards, 'behavioural_success': bool(behav), 'truth_success': bool(behav and tests['H_truth_geographic_mlp_gap']),
                'longer_budget_flag': bool(not behav and still >= 2), 'seeds_still_improving_on_train_monitor': still}
    pd.DataFrame(comps).to_csv(OUT / 'comparisons.csv', index=False)
    (OUT / 'criteria.json').write_text(json.dumps(crit, indent=2))
    (OUT / 'eval_info.json').write_text(json.dumps({'script_sha256': sha(__file__), 'data': 'validation only', 'test_rows_used': False,
                                                    'gpu_peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                    'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print(json.dumps({k: {'tests': v['tests'], 'guards_all': all(v['guards'].values()), 'behavioural': v['behavioural_success'],
                          'truth': v['truth_success'], 'longer_budget_flag': v['longer_budget_flag']} for k, v in crit.items()}, indent=2))


if __name__ == '__main__':
    main()
