#!/usr/bin/env python3
"""Why does the frozen truth / honesty readout degrade on SAE reconstructions? Error decomposition + noise control.

Run from repository root (CPU is enough; needs the cached activations and the SAE checkpoints):
    python scripts/shared/frozen_probe_error_decomposition.py --dataset got
    python scripts/shared/frozen_probe_error_decomposition.py --dataset repe
    python scripts/shared/frozen_probe_error_decomposition.py --dataset got --saes public_relu    # G9 baseline
    python scripts/shared/frozen_probe_error_decomposition.py --dataset got --saes public_topk
    (add --smoke for layers 6 only, 1 seed, 5 noise draws, written to .../smoke/)

BACKGROUND. scripts/shared/all_layer_frozen_probe_evaluation.py found large frozen-probe AUROC gaps although
reconstruction cosine is ~0.98. A gap can come from two different failures, and AUROC alone cannot tell them apart:
  (a) the SAE error is simply large, and ANY error of that size would break the readout; or
  (b) the SAE error is aimed at the truth direction: it removes the class separation (signal loss) or adds
      scatter along the probe direction (noise) far beyond what its size explains.
This script separates them. It fits nothing new: the probes are the frozen ORIGINAL probes of that evaluation
(same code, imported), and the in-house numbers must reproduce its saved validation rows exactly (asserted).

ROWS. VALIDATION rows only. No test row is scored, so the test splits stay untouched. Status: EXPLORATORY diagnostic.

PER (domain, layer, SAE), for each frozen probe (logistic, diffmean), with s = probe score, e = x_hat - x:
  auroc_original, auroc_reconstructed, auroc_gap                      (gap > 0: information lost; lower is better)
  noise control      AUROC of the frozen probe on x + r, where r has the SAME per-row norm as e but a random
                     direction (N_NOISE draws). noise_gap_mean = gap that structureless error of this size causes.
                     excess_gap = auroc_gap - noise_gap_mean (> 0: the SAE error hurts more than random error).
                     p_sae_worse_than_noise = (1 + #{draws with AUROC <= reconstructed AUROC}) / (1 + N_NOISE).
  signal_retention   (mean_1 - mean_0 of s on reconstructions) / (same on originals). 1 = class separation kept.
  noise_inflation    pooled within-class sd of s on reconstructions / on originals. 1 = no added scatter.
  auroc_signal_only  AUROC if ONLY the class means moved (original within-class scatter, reconstructed class means).
  auroc_noise_only   AUROC if ONLY the within-class scatter changed (reconstructed scatter, original class means).
  error_anisotropy   mean (e . w)^2 / (mean ||e||^2 / d) for the unit probe direction w. 1 = isotropic error;
                     >> 1 = the error is concentrated on the probe direction.
Reconstruction on the same rows: relative_sq_error, cosine, l0 (src/evaluation/faithfulness_metrics.py).

SAEs. inhouse: the 108 uninterrupted OWT SAEs (d_sae 3072, target L0 128, 3 seeds).
  public_relu: SAELens gpt2-small-res-jb, blocks.{L}.hook_resid_pre, layers 0-11 (d_sae 24 576).
  public_topk: SAELens gpt2-small-resid-post-v5-32k, blocks.{L-1}.hook_resid_post, read at resid_pre L, layers 1-11
               (resid_post of block L-1 is the same tensor as resid_pre of block L; d_sae 32 768, k = 32).
  Public and in-house SAEs differ in width, training data and L0. They are NOT sparsity-matched: report them side
  by side as an external reference, never as a ranking.

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

N_NOISE = 50
NOISE_SEED = 20261110
OUT_ROOT = REPO / 'results/shared/frozen_probe_error_decomposition'
PUBLIC = {
    'public_relu': dict(release='gpt2-small-res-jb', sae_id='blocks.{src}.hook_resid_pre', offset=0, layers=range(12)),
    'public_topk': dict(release='gpt2-small-resid-post-v5-32k', sae_id='blocks.{src}.hook_resid_post', offset=1,
                        layers=range(1, 12)),
}


# ---------------------------------------------------------------- pure functions (unit-tested)
def auroc(labels, scores):
    """Rank AUROC with ties averaged (no sklearn dependency so the tests need only numpy)."""
    labels, scores = np.asarray(labels).astype(bool), np.asarray(scores, dtype=float)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0 or not np.isfinite(scores).all():
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def class_stats(labels, scores):
    """(mean_1 - mean_0, pooled within-class sd) of the scores."""
    labels, scores = np.asarray(labels).astype(bool), np.asarray(scores, dtype=float)
    a, b = scores[labels], scores[~labels]
    pooled = np.sqrt(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / np.sqrt(max(len(scores) - 2, 1))
    return float(a.mean() - b.mean()), float(pooled)


def decompose_scores(labels, s_orig, s_rec):
    """Signal / noise decomposition of a frozen probe's scores. See the module docstring."""
    labels = np.asarray(labels).astype(bool)
    s_orig, s_rec = np.asarray(s_orig, dtype=float), np.asarray(s_rec, dtype=float)
    dmu_o, sd_o = class_stats(labels, s_orig)
    dmu_r, sd_r = class_stats(labels, s_rec)
    mean_o = np.where(labels, s_orig[labels].mean(), s_orig[~labels].mean())
    mean_r = np.where(labels, s_rec[labels].mean(), s_rec[~labels].mean())
    return {
        'signal_retention': dmu_r / dmu_o if dmu_o != 0 else float('nan'),
        'noise_inflation': sd_r / sd_o if sd_o != 0 else float('nan'),
        'dprime_original': dmu_o / sd_o if sd_o != 0 else float('nan'),
        'dprime_reconstructed': dmu_r / sd_r if sd_r != 0 else float('nan'),
        'auroc_signal_only': auroc(labels, s_orig - mean_o + mean_r),
        'auroc_noise_only': auroc(labels, s_rec - mean_r + mean_o),
    }


def norm_matched_noise(error, rng):
    """Random-direction rows with the same per-row L2 norm as ``error`` ([n, d])."""
    r = rng.standard_normal(error.shape)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return r * np.linalg.norm(error, axis=1, keepdims=True)


def error_anisotropy(error, w):
    """mean (e . w_unit)^2 / (mean ||e||^2 / d). 1 for isotropic error; d if the error lies along w."""
    error, w = np.asarray(error, dtype=np.float64), np.asarray(w, dtype=np.float64)
    nw = np.linalg.norm(w)
    if nw == 0:
        return float('nan')
    proj = error @ (w / nw)
    denom = (error ** 2).sum(1).mean() / error.shape[1]
    return float((proj ** 2).mean() / denom) if denom > 0 else float('nan')


def empirical_p_le(target, draws):
    draws = np.asarray(draws, dtype=float)
    return (1 + int(np.sum(draws <= target))) / (1 + len(draws))


def probe_direction_raw(pipeline):
    """Direction in raw activation space of a StandardScaler + LogisticRegression pipeline (coef / scale)."""
    scaler, clf = pipeline.steps[0][1], pipeline.steps[-1][1]
    return clf.coef_[0] / scaler.scale_


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- SAE providers
def inhouse_provider(layers, archs, seeds, device):
    from src.evaluation import faithfulness_metrics as fm
    from src.sae.checkpoint_registry import uninterrupted_checkpoint

    def runs():
        for layer in layers:
            for arch in archs:
                for seed in seeds:
                    path = uninterrupted_checkpoint(layer, arch, seed)
                    model, _ = fm.load_sae(path, device)

                    def recon(X, _m=model):
                        x_hat, l0, _ = fm.reconstruct(_m, X, device)
                        return x_hat.numpy(), l0.numpy()
                    yield layer, arch, seed, path.relative_to(REPO).as_posix(), recon
    return runs


def public_provider(kind, layers, device, batch_size=512):
    import torch
    from sae_lens import SAE
    spec = PUBLIC[kind]

    def runs():
        for layer in layers:
            if layer not in spec['layers']:
                continue
            sae_id = spec['sae_id'].format(src=layer - spec['offset'])
            loaded = SAE.from_pretrained(release=spec['release'], sae_id=sae_id, device=device)
            sae = loaded[0] if isinstance(loaded, tuple) else loaded      # older SAELens returns a tuple
            sae.eval()

            def recon(X, _sae=sae):
                outs, l0s = [], []
                with torch.inference_mode():
                    for batch in torch.as_tensor(X, dtype=torch.float32).split(batch_size):
                        z = _sae.encode(batch.to(device))
                        outs.append(_sae.decode(z).float().cpu())
                        l0s.append((z > 0).sum(-1).float().cpu())
                return torch.cat(outs).numpy(), torch.cat(l0s).numpy()
            yield layer, kind, 0, f"{spec['release']}/{sae_id}", recon
    return runs


def load_repe_main_only():
    """RepE loader without the cross-wording set (only validation rows are scored here, and they are all in the
    main cache). Same row order, labels, splits and leakage check as all_layer_frozen_probe_evaluation.load_repe."""
    import torch
    main = torch.load(REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt', map_location='cpu', weights_only=True)
    ids, split = np.array(main['statement_ids']), np.array(main['splits'])
    if any(len(set(split[ids == i])) > 1 for i in np.unique(ids)):
        raise ValueError('RepE: a statement crosses splits')
    return {'honesty': {'X': {l: main['activations'][l].numpy().astype(np.float32) for l in range(12)},
                        'y': main['persona_labels'].numpy(), 'split': split, 'groups': ids.astype(str)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('got', 'repe'), required=True)
    ap.add_argument('--saes', choices=('inhouse', 'public_relu', 'public_topk'), default='inhouse')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--skip-reproduction-check', action='store_true',
                    help='use only if the SAE checkpoints were retrained on this machine (their weights then differ '
                         'slightly from the ones behind the saved per_run.csv); the difference is still reported')
    a = ap.parse_args()
    import torch
    from src.evaluation import faithfulness_metrics as fm
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    spec = importlib.util.spec_from_file_location('frozen_eval', REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    out = OUT_ROOT / f'{a.dataset}_{a.saes}'
    if a.smoke:
        out = out / 'smoke'
    if (out / 'per_run.csv').exists():
        raise SystemExit(f'{out}/per_run.csv exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    layers = [6] if a.smoke else list(ev.LAYERS)
    seeds = ev.SEEDS[:1] if a.smoke else ev.SEEDS
    n_noise = 5 if a.smoke else N_NOISE
    runs = (inhouse_provider(layers, ev.ARCHS, seeds, device) if a.saes == 'inhouse'
            else public_provider(a.saes, layers, device))
    cross_wording = REPO / 'data/repe/activations/cross_wording/repe_final_token.pt'
    if a.dataset == 'got':
        data, _ = ev.load_got()
    elif cross_wording.exists():
        data, _ = ev.load_repe()
    else:   # the cross-wording cache is only needed for rows this script never scores
        data = load_repe_main_only()

    saved = None
    saved_path = ev.OUT[a.dataset] / 'per_run.csv'
    if a.saes == 'inhouse' and saved_path.exists():
        saved = pd.read_csv(saved_path)
        saved = saved[saved['eval_set'] == 'validation'].set_index(['domain', 'layer', 'architecture', 'seed'])

    protocol = {
        'dataset': a.dataset, 'saes': a.saes, 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'scripts/shared/frozen_probe_error_decomposition.py', 'script_sha256': sha(__file__),
        'probe_code': 'scripts/shared/all_layer_frozen_probe_evaluation.py (imported: load_*, fit_original_probes, scores)',
        'probe_code_sha256': sha(REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py'),
        'rows': 'validation only; probes fit on original train, C selected on original validation (as in the source evaluation)',
        'test_rows_scored': False, 'status': 'EXPLORATORY diagnostic',
        'noise_control': f'{n_noise} draws, per-row norm matched to the SAE error, random direction, seed {NOISE_SEED}',
        'reproduction_check': 'in-house validation AUROCs must equal the saved per_run.csv within 1e-6' if saved is not None
                              else 'not applicable (no saved rows for these SAEs)',
        'layers': layers, 'smoke': a.smoke, 'reproduction_check_enforced': not a.skip_reproduction_check,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    probes_cache, rows, max_repro = {}, [], 0.0
    t0 = time.time()
    for layer, arch, seed, source, recon in runs():
        for domain, d in data.items():
            y, split = d['y'], d['split']
            X = d['X'][layer]
            m = split == 'validation'
            if (domain, layer) not in probes_cache:
                probes = ev.fit_original_probes(X, y, split)
                s_orig = ev.scores(probes, X)
                probes_cache[(domain, layer)] = (probes, s_orig, fm.safe_auroc(y[m], s_orig['logistic'][m]))
            probes, s_orig, val_logistic = probes_cache[(domain, layer)]
            X_hat, l0 = recon(X)
            s_rec = ev.scores(probes, X_hat)
            err = (X_hat[m] - X[m]).astype(np.float64)
            r = {'domain': domain, 'layer': layer, 'architecture': arch, 'seed': seed, 'source': source,
                 'eval_set': 'validation', 'informative_layer': bool(val_logistic >= ev.INFORMATIVE_VAL_AUROC)}
            rm = fm.reconstruction_metrics(X[m], X_hat[m], l0[m])
            r.update({k: rm[k] for k in ('n', 'relative_sq_error', 'explained_variance', 'cosine', 'l0')})
            r['mean_error_norm'] = float(np.linalg.norm(err, axis=1).mean())
            r['mean_activation_norm'] = float(np.linalg.norm(X[m], axis=1).mean())
            directions = {'logistic': probe_direction_raw(probes['logistic']), 'diffmean': probes['direction']}
            for pr in ('logistic', 'diffmean'):
                so, sr = s_orig[pr][m], s_rec[pr][m]
                if not np.isfinite(so).all():
                    continue
                a_o, a_r = auroc(y[m], so), auroc(y[m], sr)
                r[f'{pr}_auroc_original'], r[f'{pr}_auroc_reconstructed'], r[f'{pr}_auroc_gap'] = a_o, a_r, a_o - a_r
                if saved is not None and (domain, layer, arch, seed) in saved.index:
                    ref = saved.loc[(domain, layer, arch, seed)]
                    for name, val in ((f'{pr}_auroc_original', a_o), (f'{pr}_auroc_reconstructed', a_r)):
                        diff = abs(float(ref[name]) - val)
                        max_repro = max(max_repro, diff)
                        if diff > 1e-6 and not a.skip_reproduction_check:
                            raise SystemExit(f'reproduction check failed: {domain} L{layer} {arch} s{seed} {name}: '
                                             f'{val} vs saved {float(ref[name])}. If the checkpoints were '
                                             'retrained on this machine, re-run with --skip-reproduction-check.')
                r.update({f'{pr}_{k}': v for k, v in decompose_scores(y[m], so, sr).items()})
                r[f'{pr}_error_anisotropy'] = error_anisotropy(err, directions[pr])
                rng = np.random.default_rng([NOISE_SEED, layer, seed, int(pr == 'logistic')])
                draws = []
                for _ in range(n_noise):
                    Xn = (X[m] + norm_matched_noise(err, rng)).astype(np.float32)
                    draws.append(auroc(y[m], ev.scores(probes, Xn)[pr]))
                draws = np.array(draws)
                r[f'{pr}_noise_auroc_mean'], r[f'{pr}_noise_auroc_sd'] = float(draws.mean()), float(draws.std(ddof=1))
                r[f'{pr}_noise_auroc_min'] = float(draws.min())
                r[f'{pr}_noise_gap_mean'] = a_o - float(draws.mean())
                r[f'{pr}_excess_gap'] = (a_o - a_r) - (a_o - float(draws.mean()))
                r[f'{pr}_p_sae_worse_than_noise'] = empirical_p_le(a_r, draws)
            rows.append(r)
        print(f'layer {layer} {arch} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_run.csv', index=False)
    num = [c for c in df.columns if df[c].dtype.kind == 'f' and c not in ('seed',)]
    summary = df.groupby(['domain', 'layer', 'architecture', 'informative_layer'])[num].agg(['mean', 'std'])
    summary.columns = [f'{c}_{s}' for c, s in summary.columns]
    summary.reset_index().to_csv(out / 'summary.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'n_rows': len(df), 'max_abs_reproduction_difference': max_repro if saved is not None else None,
        'runtime_seconds': round(time.time() - t0, 1), 'device': device}, indent=2))
    inf = df[df['informative_layer']]
    cols = ['logistic_auroc_original', 'logistic_auroc_reconstructed', 'logistic_noise_auroc_mean',
            'logistic_excess_gap', 'logistic_signal_retention', 'logistic_noise_inflation', 'logistic_error_anisotropy',
            'relative_sq_error', 'l0']
    print(inf.groupby(['domain', 'architecture'])[cols].mean().round(3).to_string())
    if saved is not None:
        print(f'reproduction check: max |difference| to the saved validation AUROCs = {max_repro:.2e}'
              + (' (not enforced)' if a.skip_reproduction_check else ' (passed)'))


if __name__ == '__main__':
    main()
