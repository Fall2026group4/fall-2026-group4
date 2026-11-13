#!/usr/bin/env python3
"""PCA-CM: capacity-matched PCA references for the GPT-2 SAEs (2026-11-13). VALIDATION ONLY.

Why: every PCA reference used so far (M2, the ladder, M5) is matched on OWT reconstruction error
(`information_levels.pca_k_for_val_nmse`). For the trained SAEs that needs k = 298-519 dense components, but only
k = 143-164 for the frozen random decoder (M5 `got_cities__pca_k`). "SAE vs PCA at equal error" therefore compares a
128-active-latent SAE with a 300-500-dimensional subspace. This script adds references matched on the per-token
active budget instead, and leaves the error-matched analysis untouched.

Protocol frozen in results/shared/pca_capacity_matched/README.md and preregistration.json (hashes checked at start).
For each layer in (1, 8, 11):
  * PCA fitted on OWT TRAIN activations (BOS excluded), exactly as M2.owt_pca;
  * PCA references at k in K_SWEEP (32, 64, 128, 256, 512): OWT VALIDATION reconstruction (NMSE, EV, cosine) and
    spliced OWT VALIDATION behaviour (ΔCE, KL, top-1), with the unchanged splice code;
  * every SAE in the M5 source list ('old', 'r2', 'r4', 'resjb') through the identical code on the same rows;
  * layers 8 and 11 only: GoT cities + neg_cities, readers fitted on TRAIN rows and scored on VALIDATION rows (M2's
    original_side / evaluate_source, unchanged), for every SAE and for PCA at
      - k = 128 (capacity-matched: the SAEs' OWT L0 target),
      - k = round(SAE's GoT task L0) (task-capacity-matched),
      - k = the SAE's error-matched k (pca_k_for_val_nmse; replicates the existing reference on validation);
    SAE - PCA gap differences use one shared grouped bootstrap per layer (paired).
No TEST row is read: OWT 'validation' cache, GoT split 'val'. Nothing is trained.

Run: python scripts/shared/pca_capacity_matched.py              (full; once)
     python scripts/shared/pca_capacity_matched.py --smoke DIR  (few sources, few chunks; pipeline check)
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
from src.evaluation import information_levels as il  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M5 = _load('scripts/shared/m5_confirmatory_test.py', 'm5')
LAD, M2, SP, O3 = M5.LAD, M5.M2, M5.SP, M5.O3
OUT = REPO / 'results/shared/pca_capacity_matched'
LAYERS = (1, 8, 11)
GOT_LAYERS = (8, 11)
SOURCES = ('old', 'r2', 'r4', 'resjb')
K_SWEEP = (32, 64, 128, 256, 512)
K_CAP = 128
SEED = 20261113
TOL = il.TOL          # 0.02 AUROC, as in M2
GAP_METRICS = ('refit_logistic_gap', 'frozen_logistic_gap')
SMOKE_SOURCES = {('old', 'topk', 42), ('r2', 'topk', 42), ('r2', 'topk_frozen_decoder', 42)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pca_torch_fn(mu, V, k, device):
    """x [m, 768] -> PCA-k reconstruction, float32 on device (P = top-k eigenvectors of the OWT TRAIN covariance)."""
    P = torch.as_tensor(V[:, :k], dtype=torch.float32, device=device)
    m = torch.as_tensor(mu, dtype=torch.float32, device=device)
    return lambda x: ((x - m) @ P) @ P.T + m


def classify(lo, hi):
    """Per-family classification over seeds, from per-seed 95 % CIs of (SAE gap - PCA gap); lower gap is better."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if (hi < 0).all():
        return 'SAE BETTER'
    if (lo > 0).all():
        return 'SAE WORSE'
    if ((lo >= -TOL) & (hi <= TOL)).all():
        return 'EQUIVALENT (±0.02)'
    return 'INCONCLUSIVE'


def owt_classify(sae_vals, pca_vals):
    """Deterministic OWT metrics (lower is better): SAE better/worse only if so for every seed."""
    d = np.asarray(sae_vals, float) - np.asarray(pca_vals, float)
    return 'SAE BETTER' if (d < 0).all() else 'SAE WORSE' if (d > 0).all() else 'MIXED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    out = a.smoke or OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs from the frozen protocol; refusing to run.')
    for f in ('rows.csv', 'got_paired.csv', 'decisions.csv', 'checks.json'):
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    SP.BATCH = 8
    spl = SP.Splicer(model, device)
    seqs = M5.load_owt('validation')
    if a.smoke:
        seqs = seqs[:16]
    orig_lp = LAD._OnDevice([t.cpu() for t in SP.owt_original(spl, seqs)], device)
    got, n_got_eval = M5.got_cities('val')
    assert set(got['split']) == {'train', 'validation'}
    pca_stats, n_pca = M2.owt_pca(bool(a.smoke))
    checks = dict(split='validation', got_split='val', n_owt_sequences=len(seqs), n_got_eval_rows=n_got_eval,
                  n_pca_train_tokens=int(n_pca), smoke=bool(a.smoke), test_rows_used=0)
    rows, paired = [], []
    for L in LAYERS:
        x_val = O3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L)
        checks[f'L{L}_owt_val_tokens'] = int(x_val.shape[0])
        mu, cov = pca_stats[L]
        _, V = il.pca_basis(mu, cov)
        xv_np = x_val.numpy()
        got_on = L in GOT_LAYERS
        if got_on:
            orig = M2.original_side(got, L, np.random.default_rng(M2.SEED + L))
            bidx = il.bootstrap_indices(got['groups'][got['split'] == 'validation'], M2.N_BOOT,
                                        np.random.default_rng(M2.SEED + 100 + L))
            Xg = got['X'][L]
        pca_got = {}                                    # k -> (row dict, bootstrap arrays)

        def pca_got_eval(k):
            if k not in pca_got:
                Hp = il.pca_reconstruct(Xg.astype(np.float64), mu, V, k).astype(np.float32)
                pca_got[k] = M2.evaluate_source(dict(), Xg, Hp, None, got['y'], got['split'], got['groups'], orig,
                                                bidx, SEED + 1000 * L + 7 * k)
            return pca_got[k]

        # ---- PCA references (sweep)
        for k in K_SWEEP:
            fn = pca_torch_fn(mu, V, k, device)
            xh = fn(x_val.to(device)).cpu()
            r = dict(layer=L, family='pca', name=f'pca_k{k}', seed=-1, k=k)
            r.update({f'owt_{m}': v for m, v in fm.reconstruction_metrics(x_val, xh).items()})
            r['owt_l0'] = float(k)
            r.update({f'splice_{m}': v for m, v in SP.owt_eval(spl, seqs, L, fn, orig_lp).items()})
            if got_on:
                rp, _ = pca_got_eval(k)
                r.update({f'got_{c}': rp[c] for c in ('task_nmse',) + GAP_METRICS})
                r['got_original_auroc'] = orig['auroc']['logistic']
            rows.append(r)
            print(f'L{L} PCA k={k}: nmse {r["owt_nmse"]:.4f} dCE {r["splice_delta_ce"]:.4f} ({time.time()-t0:.0f}s)',
                  flush=True)
        # ---- SAEs through the identical code
        srcs = LAD.sources(L, list(SOURCES))
        if a.smoke:
            srcs = [s for s in srcs if (s['family'], s['name'], s['seed']) in SMOKE_SOURCES]
        for i, src in enumerate(srcs):
            sae = LAD.load(src, L, device)
            r = dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'], k=np.nan,
                     checkpoint_sha256=sha(src['path']))
            xh, l0, _ = fm.reconstruct(sae, x_val, device)
            r.update({f'owt_{m}': v for m, v in fm.reconstruction_metrics(x_val, xh, l0).items()})
            r.update({f'splice_{m}': v for m, v in SP.owt_eval(spl, seqs, L, SP.sae_fn(sae), orig_lp).items()})
            if got_on:
                H, Z = LAD.encode_np(sae, Xg, device)
                rs, bs = M2.evaluate_source(dict(), Xg, H, Z, got['y'], got['split'], got['groups'], orig, bidx,
                                            SEED + 1000 * L + i)
                r.update({f'got_{c}': rs[c] for c in ('task_nmse', 'task_l0') + GAP_METRICS})
                r['got_original_auroc'] = orig['auroc']['logistic']
                refs = {'capacity_k128': K_CAP,
                        'task_l0_matched': int(max(1, round(rs['task_l0']))),
                        'error_matched': il.pca_k_for_val_nmse(xv_np, mu, V, r['owt_nmse'])}
                for ref, k in refs.items():
                    rp, bp = pca_got_eval(k)
                    p = dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'], reference=ref, pca_k=k,
                             sae_task_nmse=rs['task_nmse'], pca_task_nmse=rp['task_nmse'])
                    for m in GAP_METRICS:
                        d = bs[m] - bp[m]
                        lo, hi = il.ci(d)
                        p.update({f'sae_{m}': rs[m], f'pca_{m}': rp[m], f'diff_{m}': rs[m] - rp[m],
                                  f'diff_{m}_ci_low': lo, f'diff_{m}_ci_high': hi})
                    paired.append(p)
            rows.append(r)
            print(f"L{L} {src['family']} {src['name']} {src['seed']}: nmse {r['owt_nmse']:.4f} l0 {r['owt_l0']:.1f} "
                  f"dCE {r['splice_delta_ce']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del sae
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out / 'rows.partial.csv', index=False)
    df, pdf = pd.DataFrame(rows), pd.DataFrame(paired)
    df.to_csv(out / 'rows.csv', index=False)
    pdf.to_csv(out / 'got_paired.csv', index=False)
    (out / 'rows.partial.csv').unlink()
    # ---- decisions (pre-registered rules; see README)
    dec = []
    pca128 = df[(df.family == 'pca') & (df.k == K_CAP)].set_index('layer')
    for (L, fam, name), g in df[df.family != 'pca'].groupby(['layer', 'family', 'name']):
        for m in ('owt_nmse', 'splice_delta_ce', 'splice_kl_mean'):
            ref = pca128.loc[L, m]
            dec.append(dict(layer=L, family=fam, name=name, reference='capacity_k128', metric=m, n_seeds=len(g),
                            sae_mean=g[m].mean(), sae_sd=g[m].std(ddof=1), pca_value=ref,
                            diff_mean=(g[m] - ref).mean(), pct_vs_pca=100 * (ref - g[m].mean()) / ref,
                            cls=owt_classify(g[m], [ref] * len(g))))
    if len(pdf):
        for (L, fam, name, ref), g in pdf.groupby(['layer', 'family', 'name', 'reference']):
            for m in GAP_METRICS:
                dec.append(dict(layer=L, family=fam, name=name, reference=ref, metric=f'got_{m}', n_seeds=len(g),
                                pca_k=';'.join(map(str, sorted(set(g.pca_k)))), sae_mean=g[f'sae_{m}'].mean(),
                                sae_sd=g[f'sae_{m}'].std(ddof=1), pca_value=g[f'pca_{m}'].mean(),
                                diff_mean=g[f'diff_{m}'].mean(), ci_low_min=g[f'diff_{m}_ci_low'].min(),
                                ci_high_max=g[f'diff_{m}_ci_high'].max(),
                                cls=classify(g[f'diff_{m}_ci_low'], g[f'diff_{m}_ci_high'])))
    pd.DataFrame(dec).to_csv(out / 'decisions.csv', index=False)
    checks.update(runtime_s=time.time() - t0, script_sha256=sha(__file__), n_rows=len(df), n_paired=len(pdf))
    (out / 'checks.json').write_text(json.dumps(checks, indent=2, default=float))


if __name__ == '__main__':
    main()
