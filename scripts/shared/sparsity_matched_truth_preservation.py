#!/usr/bin/env python3
"""Truth / honesty preservation compared at EQUAL sparsity on the truth and honesty prompts (roadmap S3).

Run from repository root (CPU is enough; a GPU is faster):
    python scripts/shared/sparsity_matched_truth_preservation.py --dataset got
    python scripts/shared/sparsity_matched_truth_preservation.py --dataset repe
    (add --smoke for layer 6 and seed 42 only, written to .../smoke/)

WHY. The 108 uninterrupted SAEs are L0-matched on OpenWebText (target 128), but not on the prompts the truth and
honesty probes read: on GoT final tokens ReLU uses ~66 latents, TopK exactly 128 and JumpReLU ~46 (RepE: 163 / 128 /
179). Every architecture comparison of frozen-probe gaps is therefore confounded with sparsity on those prompts.

METHOD (inference-time top-m truncation; nothing is trained or fitted on reconstructions).
  For every SAE and every m in M_GRID: encode the activation, keep only its m largest latent values, decode
  (decoder + output bias). For TopK this is exactly the same SAE with k = m, because its latents already are its
  top-k positives. For ReLU / JumpReLU it removes their weakest active latents, so a row with fewer than m active
  latents is unchanged. The SAE was not trained for truncation: m below its natural L0 is a post-hoc sparsity
  control, stated as such.
  The frozen ORIGINAL probes of scripts/shared/all_layer_frozen_probe_evaluation.py are applied to every
  reconstruction (same code, imported): logistic (StandardScaler + LogisticRegression, C chosen on original
  validation) and diffmean (unit mass-mean direction). auroc_gap = auroc(original) - auroc(reconstructed), same rows.
  Lower gap is better. Category: representation preservation.

MATCHING RULE (fixed before running). A cell (domain, layer, m) is MATCHED when every architecture and every seed
  reaches a mean prompt L0 within +-10 % of m on the evaluated rows (the CLAUDE.md baseline tolerance). Only matched
  cells enter matched_comparison.csv; the rest stay in per_run.csv. Rankings use paired differences across the three
  SAE seeds; a difference counts only if all three seeds agree in sign.

ROWS. VALIDATION rows only (test rows untouched). Evidential status: validation finding.
  The natural (untruncated) rows must reproduce the saved validation AUROCs of the all-layer evaluation within 1e-6
  (asserted unless --skip-reproduction-check, needed for checkpoints retrained on another machine).

Existing outputs are never overwritten.
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

M_GRID = (8, 16, 24, 32, 48, 64, 96, 128)
TOLERANCE = 0.10
OUT_ROOT = REPO / 'results/shared/sparsity_matched_truth_preservation'
PAIRS = (('topk', 'relu'), ('topk', 'jumprelu'), ('jumprelu', 'relu'))


# ---------------------------------------------------------------- pure functions (unit-tested)
def truncate_topm(z, m):
    """Keep the m largest entries of every row of z (others set to 0). Rows with <= m positives are unchanged."""
    if m >= z.shape[-1]:
        return z.clone()
    vals, idx = z.topk(m, dim=-1)
    return torch.zeros_like(z).scatter(-1, idx, vals)


def decode(sae, z):
    """In-house SAE decoder: z @ D^T + output_bias (scripts/openwebtext/03_training_sae.py:SAE.forward)."""
    return torch.nn.functional.linear(z, sae.decoder.weight) + sae.output_bias


def is_matched(l0_values, m, tolerance=TOLERANCE):
    """True if every mean prompt L0 is within +-tolerance of m."""
    l0 = np.asarray(l0_values, dtype=float)
    return bool(len(l0) and np.all(np.abs(l0 - m) <= tolerance * m))


def paired_comparison(df, metric, a, b):
    """Paired a - b over seeds (rows: one per architecture x seed). Returns mean, sd, n and sign agreement."""
    pa = df[df['architecture'] == a].set_index('seed')[metric]
    pb = df[df['architecture'] == b].set_index('seed')[metric]
    seeds = sorted(set(pa.index) & set(pb.index))
    if not seeds:
        return {'n_seeds': 0}
    d = np.array([pa[s] - pb[s] for s in seeds], dtype=float)
    if len(d) < 2:
        sign = 'single_seed'
    else:
        sign = 'a_lower' if (d < 0).all() else ('a_higher' if (d > 0).all() else 'mixed')
    return {'n_seeds': len(seeds), 'mean_diff': float(d.mean()),
            'sd_diff': float(d.std(ddof=1)) if len(d) > 1 else float('nan'),
            'per_seed': ';'.join(f'{x:+.4f}' for x in d), 'all_seeds_agree': sign}


def matched_table(per_run, metrics=('logistic_auroc_gap', 'diffmean_auroc_gap', 'relative_sq_error')):
    """One row per matched (domain, layer, m) cell and architecture pair."""
    rows = []
    trunc = per_run[per_run['truncation'] != 'natural']
    for (domain, layer, m), cell in trunc.groupby(['domain', 'layer', 'm']):
        if not is_matched(cell['l0'], m):
            continue
        base = {'domain': domain, 'layer': layer, 'm': m, 'informative_layer': bool(cell['informative_layer'].iloc[0]),
                'l0_min': float(cell['l0'].min()), 'l0_max': float(cell['l0'].max())}
        for arch, d in cell.groupby('architecture'):
            for met in metrics:
                base[f'{arch}_{met}_mean'] = float(d[met].mean())
                base[f'{arch}_{met}_sd'] = float(d[met].std(ddof=1)) if len(d) > 1 else float('nan')
        for a, b in PAIRS:
            for met in metrics[:2]:
                r = paired_comparison(cell, met, a, b)
                rows.append({**base, 'comparison': f'{a} - {b}', 'metric': met, **r})
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('got', 'repe'), required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--skip-reproduction-check', action='store_true')
    a = ap.parse_args()
    from src.evaluation import faithfulness_metrics as fm
    from src.sae.checkpoint_registry import uninterrupted_checkpoint
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    spec = importlib.util.spec_from_file_location('frozen_eval', REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    spec2 = importlib.util.spec_from_file_location('fped', REPO / 'scripts/shared/frozen_probe_error_decomposition.py')
    fped = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fped)

    out = OUT_ROOT / a.dataset / ('smoke' if a.smoke else '')
    if (out / 'per_run.csv').exists():
        raise SystemExit(f'{out}/per_run.csv exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    layers = [6] if a.smoke else list(ev.LAYERS)
    seeds = ev.SEEDS[:1] if a.smoke else ev.SEEDS

    if a.dataset == 'got':
        data, _ = ev.load_got()
    elif (REPO / 'data/repe/activations/cross_wording/repe_final_token.pt').exists():
        data, _ = ev.load_repe()
    else:
        data = fped.load_repe_main_only()

    saved = None
    saved_path = ev.OUT[a.dataset] / 'per_run.csv'
    if saved_path.exists():
        saved = pd.read_csv(saved_path)
        saved = saved[saved['eval_set'] == 'validation'].set_index(['domain', 'layer', 'architecture', 'seed'])

    ckpts = {(l, ar, s): uninterrupted_checkpoint(l, ar, s) for l in layers for ar in ev.ARCHS for s in seeds}
    (out / 'protocol.json').write_text(json.dumps({
        'dataset': a.dataset, 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'scripts/shared/sparsity_matched_truth_preservation.py', 'script_sha256': sha(__file__),
        'probe_code_sha256': sha(REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py'),
        'checkpoint_sha256': {f'{l}/{ar}/{s}': sha(p) for (l, ar, s), p in ckpts.items()},
        'm_grid': list(M_GRID), 'matching_tolerance': TOLERANCE,
        'matching_rule': 'cell matched if every architecture and seed has mean prompt L0 within +-10% of m',
        'ranking_rule': 'paired differences over seeds; a difference counts only if all seeds agree in sign',
        'rows': 'validation only', 'test_rows_scored': False, 'status': 'validation finding',
        'reproduction_check_enforced': not a.skip_reproduction_check, 'smoke': a.smoke}, indent=2))

    rows, probes_cache, max_repro = [], {}, 0.0
    t0 = time.time()
    for layer in layers:
        for arch in ev.ARCHS:
            for seed in seeds:
                sae, _ = fm.load_sae(ckpts[(layer, arch, seed)], device)
                for domain, d in data.items():
                    y, split = d['y'], d['split']
                    X = d['X'][layer]
                    m_val = split == 'validation'
                    if (domain, layer) not in probes_cache:
                        probes = ev.fit_original_probes(X, y, split)
                        s_orig = ev.scores(probes, X)
                        val_auc = fm.safe_auroc(y[m_val], s_orig['logistic'][m_val])
                        probes_cache[(domain, layer)] = (probes, s_orig, val_auc)
                    probes, s_orig, val_auc = probes_cache[(domain, layer)]
                    Xv = torch.as_tensor(X[m_val], dtype=torch.float32, device=device)
                    with torch.inference_mode():
                        _, z, _ = sae(Xv)
                    natural_l0 = float((z > 0).sum(-1).float().mean())
                    for trunc in ['natural'] + list(M_GRID):
                        with torch.inference_mode():
                            zt = z if trunc == 'natural' else truncate_topm(z, trunc)
                            X_hat = decode(sae, zt).float().cpu().numpy()
                            l0 = (zt > 0).sum(-1).float().cpu().numpy()
                        s_rec = ev.scores(probes, X_hat)
                        r = {'domain': domain, 'layer': layer, 'architecture': arch, 'seed': seed,
                             'truncation': 'natural' if trunc == 'natural' else 'top_m',
                             'm': np.nan if trunc == 'natural' else trunc, 'natural_l0': natural_l0,
                             'informative_layer': bool(val_auc >= ev.INFORMATIVE_VAL_AUROC)}
                        rm = fm.reconstruction_metrics(X[m_val], X_hat, l0)
                        r.update({k: rm[k] for k in ('n', 'relative_sq_error', 'explained_variance', 'cosine', 'l0')})
                        for pr in ('logistic', 'diffmean'):
                            so, sr = s_orig[pr][m_val], s_rec[pr]          # X_hat holds validation rows only
                            ao, ar_ = fm.safe_auroc(y[m_val], so), fm.safe_auroc(y[m_val], sr)
                            r[f'{pr}_auroc_original'], r[f'{pr}_auroc_reconstructed'] = ao, ar_
                            r[f'{pr}_auroc_gap'] = ao - ar_
                            if trunc == 'natural' and saved is not None and (domain, layer, arch, seed) in saved.index \
                                    and np.isfinite(ar_):
                                diff = abs(float(saved.loc[(domain, layer, arch, seed), f'{pr}_auroc_reconstructed']) - ar_)
                                max_repro = max(max_repro, diff)
                                if diff > 1e-6 and not a.skip_reproduction_check:
                                    raise SystemExit(f'reproduction check failed: {domain} L{layer} {arch} s{seed} {pr}: '
                                                     f'diff {diff:.2e}. Retrained checkpoints? use --skip-reproduction-check')
                        rows.append(r)
                print(f'layer {layer} {arch} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / 'per_run.csv', index=False)
    mt = matched_table(per_run)
    mt.to_csv(out / 'matched_comparison.csv', index=False)
    nat = per_run[per_run['truncation'] == 'natural'].groupby(['domain', 'layer', 'architecture'])[
        ['l0', 'logistic_auroc_gap', 'diffmean_auroc_gap', 'relative_sq_error']].agg(['mean', 'std'])
    nat.columns = [f'{c}_{s}' for c, s in nat.columns]
    nat.reset_index().to_csv(out / 'natural_l0_summary.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'n_rows': len(per_run), 'matched_cells': int(mt[['domain', 'layer', 'm']].drop_duplicates().shape[0]) if len(mt) else 0,
        'max_abs_reproduction_difference': max_repro if saved is not None else None,
        'runtime_seconds': round(time.time() - t0, 1), 'device': device}, indent=2))
    if len(mt):
        show = mt[(mt['metric'] == 'logistic_auroc_gap') & mt['informative_layer']]
        print(show[['domain', 'layer', 'm', 'comparison', 'mean_diff', 'sd_diff', 'all_seeds_agree']].round(4).to_string(index=False))
    else:
        print('no matched cells')
    if saved is not None:
        print(f'reproduction check: max |difference| = {max_repro:.2e}'
              + (' (not enforced)' if a.skip_reproduction_check else ' (passed)'))


if __name__ == '__main__':
    main()
