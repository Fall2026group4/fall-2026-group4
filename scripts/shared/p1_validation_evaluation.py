#!/usr/bin/env python3
"""Validation-only evaluation of the P1 pilot SAEs vs their uninterrupted seed-42 baselines.

Run from repository root after scripts/openwebtext/16_p1_sparsity_control_pilot.py:
    python scripts/shared/p1_validation_evaluation.py

For each (layer in 0/2/6/10, architecture) it evaluates, with IDENTICAL code:
  * baseline  = results/openwebtext/convergence/layer_L/{arch}_l0_128_uninterrupted_seed42/checkpoint_step_15000.pt
  * p1_selected (pre-defined validation selection rule) and p1_final (step 15 000, for transparency)
Metrics (validation splits only; no test row is read):
  1. OWT validation reconstruction: MSE, NMSE, EV, cosine, L0, dead-latent fraction.
  2. Spliced-in OWT next-token loss increase and KL (Phase-0 code).
  3. GoT val: frozen logistic / mass-mean probe AUROC gaps (probes fit on original train, C chosen on original val,
     same as the all-layer evaluation) and spliced cities truth preference (pairwise accuracy change).
  4. RepE validation: frozen-probe AUROC gaps, direction fidelity, spliced persona-shift cosine (activation splicing,
     NOT steering).
Decision rules fixed before the P1 results were seen:
  * L0-matched pair: |L0_p1 - L0_base| / L0_base <= 0.02 (OWT validation L0).
  * Genuine change: difference in the better direction AND |difference| > 2 x the baseline cell's seed sd
    (sd over seeds 42/1042/2042 of the same metric, same code). Single-seed P1 => descriptive pilot evidence only.
Outputs: results/openwebtext/p1_sparsity_control_pilot/{validation_evaluation_long.csv, baseline_vs_p1.csv,
         seed_sd_reference.csv, eval_info.json}; figure results/openwebtext/figures/p1_vs_baseline.png
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

OUT = REPO / 'results/openwebtext/p1_sparsity_control_pilot'
LAYERS, ARCHS = (0, 2, 6, 10), ('relu', 'topk', 'jumprelu')
L0_MATCH = 0.02
# metric -> True if lower is better
METRICS = {
    'owt_val_mse': True, 'owt_val_nmse': True, 'owt_val_ev': False, 'owt_val_cosine': False, 'owt_val_l0': None,
    'owt_val_dead_fraction': True, 'owt_delta_ce': True, 'owt_kl': True,
    'got_geographic_logistic_gap': True, 'got_numerical_logistic_gap': True, 'got_geographic_diffmean_gap': True,
    'got_cities_pair_accuracy_change': False, 'got_statement_kl': True,
    'repe_logistic_gap': True, 'repe_diffmean_gap': True, 'repe_direction_cosine': False,
    'repe_persona_shift_cosine_final': False, 'repe_final_kl': True,
}


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out_long = OUT / 'validation_evaluation_long.csv'
    if out_long.exists():
        raise SystemExit(f'{out_long} exists; refusing to overwrite.')
    runs = pd.read_csv(OUT / 'runs.csv')
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    ev = load('scripts/shared/all_layer_frozen_probe_evaluation.py', 'ev')
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    device = 'cuda'
    t0 = time.time()
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
    got_id = sp.got_eval(S, got_v, 0, lambda x: x.clone(), got_orig)   # identity -> original truth preference
    got_data, _ = ev.load_got()
    repe_data, _ = ev.load_repe()

    rows = []
    for layer in LAYERS:
        x_val = o3.load_split(REPO / 'data/activations/gpt2_small', 'validation', layer).to(device)
        probes = {}
        for dom, d in list(got_data.items()) + list(repe_data.items()):
            probes[dom] = (d, ev.fit_original_probes(d['X'][layer], d['y'], d['split']))
        for arch in ARCHS:
            r = runs[(runs.layer == layer) & (runs.architecture == arch)].iloc[0]
            cands = [('baseline', s, uninterrupted_checkpoint(layer, arch, s)) for s in (42, 1042, 2042)]
            cands += [('p1_selected', 42, REPO / r.checkpoint_dir / 'checkpoint_selected.pt'),
                      ('p1_final', 42, REPO / r.checkpoint_dir / 'checkpoint_step_15000.pt')]
            for variant, seed, path in cands:
                sae, _ = fm.load_sae(path, device)
                x_hat, l0, active = fm.reconstruct(sae, x_val, device)
                rec = fm.reconstruction_metrics(x_val.cpu(), x_hat, l0)
                row = {'layer': layer, 'architecture': arch, 'variant': variant, 'seed': seed,
                       'checkpoint_path': Path(path).relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(path),
                       'owt_val_mse': rec['mse'], 'owt_val_nmse': rec['nmse'], 'owt_val_ev': rec['explained_variance'],
                       'owt_val_cosine': rec['cosine'], 'owt_val_l0': rec['l0'], 'owt_val_dead_fraction': fm.dead_fraction(active)}
                fn = sp.sae_fn(sae)
                o = sp.owt_eval(S, owt_seqs, layer, fn, owt_orig)
                g = sp.got_eval(S, got_v, layer, fn, got_orig)
                rp = sp.repe_eval(S, repe_v, layer, fn, repe_orig, 'final')
                row.update(owt_delta_ce=o['delta_ce'], owt_kl=o['kl_mean'], got_statement_kl=g['statement_kl_mean'],
                           got_cities_pair_accuracy=g['cities_truth_pair_accuracy'],
                           got_cities_pair_accuracy_change=g['cities_truth_pair_accuracy'] - got_id['cities_truth_pair_accuracy'],
                           repe_persona_shift_cosine_final=rp['persona_shift_cosine'], repe_final_kl=rp['final_kl_mean'])
                for dom, (d, pr) in probes.items():
                    X, y, split = d['X'][layer], d['y'], d['split']
                    va, tr = split == 'validation', split == 'train'
                    Xh = fm.reconstruct(sae, X, device)[0].numpy()
                    so, sr = ev.scores(pr, X), ev.scores(pr, Xh)
                    key = 'repe' if dom == 'honesty' else f'got_{dom}'
                    for p in ('logistic', 'diffmean'):
                        row[f'{key}_{p}_gap'] = fm.safe_auroc(y[va], so[p][va]) - fm.safe_auroc(y[va], sr[p][va])
                    if dom == 'honesty':
                        rd = Xh[tr & (y == 1)].astype(np.float64).mean(0) - Xh[tr & (y == 0)].astype(np.float64).mean(0)
                        row['repe_direction_cosine'] = fm.direction_fidelity(pr['raw_direction'], rd)['direction_cosine']
                rows.append(row)
            print(f'layer {layer} {arch} done ({time.time() - t0:.0f}s)', flush=True)
    long = pd.DataFrame(rows)
    long.to_csv(out_long, index=False)

    # consistency check: baseline numbers must reproduce Phase 0 (same code, same data)
    ph0 = pd.read_csv(REPO / 'results/openwebtext/spliced_evaluation/per_run.csv')
    ph0 = ph0[ph0.intervention == 'sae_reconstruction'].set_index(['layer', 'architecture', 'seed'])
    b = long[long.variant == 'baseline'].set_index(['layer', 'architecture', 'seed'])
    phase0_max_abs_diff = float((b['owt_delta_ce'] - ph0.loc[b.index, 'owt__delta_ce']).abs().max())

    sd = (long[long.variant == 'baseline'].groupby(['layer', 'architecture'])[list(METRICS)].std())
    sd.reset_index().to_csv(OUT / 'seed_sd_reference.csv', index=False)
    comp = []
    for (layer, arch), g in long.groupby(['layer', 'architecture']):
        base = g[(g.variant == 'baseline') & (g.seed == 42)].iloc[0]
        base_mean = g[g.variant == 'baseline'][list(METRICS)].mean()
        for variant in ('p1_selected', 'p1_final'):
            p = g[g.variant == variant].iloc[0]
            l0_rel = abs(p.owt_val_l0 - base.owt_val_l0) / base.owt_val_l0
            for m, lower in METRICS.items():
                diff = p[m] - base[m]
                s = sd.loc[(layer, arch), m]
                if lower is None:
                    verdict = ''
                else:
                    better = diff < 0 if lower else diff > 0
                    big = abs(diff) > 2 * s if np.isfinite(s) and s > 0 else False
                    verdict = ('improved' if better else 'regressed') if big else 'no clear change'
                comp.append({'layer': layer, 'architecture': arch, 'p1_variant': variant, 'metric': m,
                             'lower_is_better': lower, 'baseline_seed42': base[m], 'baseline_3seed_mean': base_mean[m],
                             'p1': p[m], 'difference_p1_minus_seed42': diff, 'baseline_seed_sd': s,
                             'difference_in_seed_sd': diff / s if s and np.isfinite(s) and s > 0 else np.nan,
                             'p1_l0': p.owt_val_l0, 'baseline_l0': base.owt_val_l0, 'l0_rel_diff': l0_rel,
                             'l0_matched_2pct': l0_rel <= L0_MATCH,
                             'p1_l0_within_5pct_of_128': abs(p.owt_val_l0 - 128) / 128 <= 0.05,
                             'verdict': verdict if l0_rel <= L0_MATCH or lower is None else f'{verdict} (L0 NOT matched)'})
    pd.DataFrame(comp).to_csv(OUT / 'baseline_vs_p1.csv', index=False)
    (OUT / 'eval_info.json').write_text(json.dumps({
        'script': 'scripts/shared/p1_validation_evaluation.py', 'script_sha256': sha(__file__),
        'data': 'validation only', 'test_rows_used': False, 'l0_match_rule': L0_MATCH,
        'genuine_change_rule': '|diff| > 2 x baseline seed sd, in the better direction',
        'phase0_reproduction_max_abs_diff_delta_ce': phase0_max_abs_diff,
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print('phase-0 reproduction max |diff| delta_ce:', phase0_max_abs_diff)


if __name__ == '__main__':
    main()
