#!/usr/bin/env python3
"""M5: the single pre-registered confirmatory TEST pass of the GPT-2 improvement phase (2026-11-10).

Protocol frozen in results/shared/m5_confirmatory_test/README.md and preregistration.json (hashes checked at start).
For every SAE in the frozen list (old base SAEs, R2 SAEs, R4 fine-tunes, res-jb) at layers 1, 8, 11:
  * reconstruction on the OWT TEST cache (EV, NMSE, cosine, L0, dead fraction), BOS excluded;
  * behaviour: spliced OWT TEST CE increase and KL, with the same-norm noise reference
    (scripts/shared/spliced_reconstruction_evaluation.py, unchanged functions);
  * information (layers 8 and 11): GoT cities, readers fitted on TRAIN rows, scored on TEST rows (M2's original_side
    and evaluate_source, unchanged): frozen and re-fit logistic gaps; PCA re-fit gap with PCA at the SAE's OWT
    VALIDATION NMSE (as in M2 and the ladder).
The pre-registered comparisons (C1-C3, A1, A2 with the R2 rule; K1, K4 with the IMP-L rule) are applied once to the
TEST rows. RepE-LM has no test split, so persona, RLM-RP and M4 results stay VALIDATION-only.

--smoke DIR runs the identical pipeline on VALIDATION rows (OWT validation, GoT cities validation) for a few SAEs, to
check that it reproduces the ladder; it never reads a test row.

Run: python scripts/shared/m5_confirmatory_test.py            (TEST; once)
     python scripts/shared/m5_confirmatory_test.py --smoke DIR (VALIDATION check)
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


LAD = _load('scripts/shared/sae_ladder_evaluation.py', 'lad')
R2C = _load('scripts/shared/r2_comparisons.py', 'r2c')
M2, SP, O3 = LAD.M2, LAD.SP, LAD.O3
OUT = REPO / 'results/shared/m5_confirmatory_test'
LAYERS = (1, 8, 11)
GOT_LAYERS = (8, 11)
SOURCES = ('old', 'r2', 'r4', 'resjb')
SEED = 20261110
K_COMPARISONS = [('K1', ('r4_kl', 'topk'), ('r4_mse', 'topk'), 'R4 KL fine-tune vs MSE-only control'),
                 ('K4', ('r4_kl', 'topk'), ('r2', 'topk'), 'R4 KL fine-tune vs its R2 base')]
METRICS = [('owt_nmse', False), ('owt_explained_variance', True), ('owt_cosine', True),
           ('splice_delta_ce', False), ('splice_kl_mean', False), ('splice_sae_vs_noise_delta_ce_ratio', False),
           ('got_cities__frozen_logistic_gap', False), ('got_cities__refit_logistic_gap', False),
           ('got_cities__sae_minus_pca_refit_gap', False)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_owt(split):
    root = REPO / f'data/activations/gpt2_small/{split}'
    seqs = []
    for p in sorted(root.glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        seqs += [c['tokens'][i, :int(n)] for i, n in enumerate(c['lengths'])]
    return seqs


def got_cities(eval_split):
    """GoT cities + neg_cities exactly as M2.load_tasks, with eval rows = eval_split ('val' or 'test') renamed
    'validation' so that M2's reader functions run unchanged. The other held-out split is dropped."""
    spec = importlib.util.spec_from_file_location('got03', REPO / 'scripts/geometry_of_truth/03_original_truth_probes.py')
    got03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(got03)
    got03.ACT_DIR, got03.SPLIT_DIR = REPO / got03.ACT_DIR, REPO / got03.SPLIT_DIR
    parts = got03.load_domain(got03.DOMAINS['geographic'])
    split = np.concatenate([p['splits'] for p in parts])
    keep = np.isin(split, ['train', eval_split])
    t = dict(X={L: np.concatenate([p['layers'][L].numpy() for p in parts]).astype(np.float32)[keep] for L in GOT_LAYERS},
             y=np.concatenate([p['labels'] for p in parts])[keep],
             split=np.where(split[keep] == eval_split, 'validation', 'train'),
             groups=np.concatenate([p['groups'] for p in parts])[keep])
    tr, ev = t['split'] == 'train', t['split'] == 'validation'
    assert not (set(t['groups'][tr]) & set(t['groups'][ev])), 'group crosses train/eval'
    return t, int(ev.sum())


def paired_rule(new, ref):
    """IMP-L rule on seed-paired values (lower is better): better on every seed and |mean d| > 2 sd(d)."""
    d = np.asarray(new, float) - np.asarray(ref, float)
    if len(d) < 2 or np.isnan(d).any():
        return dict(cls='NOT EVALUATED')
    m, s = d.mean(), d.std(ddof=1)
    if (d < 0).all() and abs(m) > 2 * s:
        cls = 'IMPROVEMENT'
    elif (d > 0).all() and abs(m) > 2 * s:
        cls = 'REGRESSION'
    else:
        cls = 'NO CLEAR CHANGE'
    return dict(cls=cls, new_mean=float(np.mean(new)), ref_mean=float(np.mean(ref)), diff=float(m), diff_sd=float(s),
                improvement_pct=float(-100 * m / abs(np.mean(ref))) if np.mean(ref) else np.nan,
                n_seeds_better=int((d < 0).sum()), new_values=list(map(float, new)), ref_values=list(map(float, ref)))


def compare(df):
    df = df.copy()
    df['matched_l0'] = df.owt_l0.between(R2C.T_LOW, R2C.T_HIGH)
    rows = []
    for L in sorted(df.layer.unique()):
        d = df[(df.layer == L) & df.matched_l0]
        for cid, new, ref, desc in R2C.COMPARISONS:
            nv = d[(d.family == new[0]) & (d.name == new[1])].sort_values('seed')
            rv = d[(d.family == ref[0]) & (d.name == ref[1])].sort_values('seed')
            for m, hib in METRICS:
                if m in d and nv[m].notna().all() and len(nv):
                    rows.append(dict(layer=L, comparison=cid, description=desc, metric=m, rule='R2 (all pairs + Welch)',
                                     **R2C.rule(nv[m], rv[m], hib)))
        for cid, new, ref, desc in K_COMPARISONS:
            nv = d[(d.family == new[0]) & (d.name == new[1])].sort_values('seed')
            rv = d[(d.family == ref[0]) & (d.name == ref[1])].sort_values('seed')
            if len(nv) == 0:
                continue
            pr = nv.merge(rv, on='seed', suffixes=('_new', '_ref'))          # paired by seed
            for m in ('splice_delta_ce', 'splice_kl_mean', 'owt_nmse', 'got_cities__refit_logistic_gap'):
                if f'{m}_new' in pr and pr[f'{m}_new'].notna().all():
                    rows.append(dict(layer=L, comparison=cid, description=desc, metric=m, rule='IMP-L (paired by seed)',
                                     n_pairs=len(pr), **paired_rule(pr[f'{m}_new'], pr[f'{m}_ref'])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    split, got_split = ('validation', 'val') if a.smoke else ('test', 'test')
    out = a.smoke or OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs from the frozen protocol; refusing to run.')
    for f in ('m5_rows.csv', 'm5_rows.partial.csv', 'comparisons.csv', 'checks.json'):
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite (M5 runs once).')
    out.mkdir(parents=True, exist_ok=True)
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    SP.BATCH = 8
    spl = SP.Splicer(model, device)
    seqs = load_owt(split)
    orig_lp = LAD._OnDevice([t.cpu() for t in SP.owt_original(spl, seqs)], device)
    got, n_got_eval = got_cities(got_split)
    pca_stats, _ = M2.owt_pca(False)
    checks = dict(split=split, n_owt_sequences=len(seqs), n_got_eval_rows=n_got_eval, smoke=bool(a.smoke))
    rows = []
    for L in LAYERS:
        x = O3.load_split(REPO / 'data/activations/gpt2_small', split, L)
        x_val = O3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L)
        checks[f'L{L}_owt_tokens'] = int(x.shape[0])
        if L in GOT_LAYERS:
            rng = np.random.default_rng(M2.SEED + L)
            orig = M2.original_side(got, L, rng)
            bidx = il.bootstrap_indices(got['groups'][got['split'] == 'validation'], M2.N_BOOT,
                                        np.random.default_rng(M2.SEED + 100 + L))
            mu, cov = pca_stats[L]
            _, V = il.pca_basis(mu, cov)
            xv_np = x_val.numpy()
        srcs = LAD.sources(L, list(SOURCES))
        if a.smoke:
            srcs = [s for s in srcs if (s['family'], s['name'], s['seed']) in
                    {('old', 'topk', 42), ('r2', 'topk', 42), ('r4_kl', 'topk', 42), ('r4_mse', 'topk', 42),
                     ('resjb', 'resjb_relu_24576', -1)}]
        for i, src in enumerate(srcs):
            sae = LAD.load(src, L, device)
            r = dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'], split=split,
                     checkpoint_sha256=sha(src['path']))
            xh, l0, act = fm.reconstruct(sae, x, device)
            rec = fm.reconstruction_metrics(x, xh, l0)
            r.update({f'owt_{k}': v for k, v in rec.items() if not isinstance(v, (list, dict))})
            r['owt_dead_fraction'] = float(1 - act.float().mean())
            xhv, l0v, _ = fm.reconstruct(sae, x_val, device)
            r['owt_validation_nmse'] = fm.reconstruction_metrics(x_val, xhv, l0v)['nmse']
            o = SP.owt_eval(spl, seqs, L, SP.sae_fn(sae), orig_lp)
            r.update({f'splice_{k}': v for k, v in o.items()})
            gen = torch.Generator(device=device).manual_seed(SP.NOISE_SEED)
            on = SP.owt_eval(spl, seqs, L, SP.noise_like_error(sae, gen), orig_lp)
            r['splice_noise_delta_ce'] = on['delta_ce']
            r['splice_sae_vs_noise_delta_ce_ratio'] = o['delta_ce'] / on['delta_ce'] if on['delta_ce'] else np.nan
            if L in GOT_LAYERS:
                H, Z = LAD.encode_np(sae, got['X'][L], device)
                rs, bs = M2.evaluate_source(dict(), got['X'][L], H, Z, got['y'], got['split'], got['groups'], orig, bidx,
                                            SEED + 1000 * L + i)
                k = il.pca_k_for_val_nmse(xv_np, mu, V, r['owt_validation_nmse'])
                Hp = il.pca_reconstruct(got['X'][L].astype(np.float64), mu, V, k).astype(np.float32)
                rp, _ = M2.evaluate_source(dict(), got['X'][L], Hp, None, got['y'], got['split'], got['groups'], orig,
                                           bidx, SEED + 1000 * L + i)
                for c in ('task_l0', 'frozen_logistic_gap', 'frozen_logistic_gap_ci_low', 'frozen_logistic_gap_ci_high',
                          'noise_frozen_logistic_gap', 'refit_logistic_gap', 'refit_logistic_gap_ci_low',
                          'refit_logistic_gap_ci_high'):
                    if c in rs:
                        r[f'got_cities__{c}'] = rs[c]
                r['got_cities__pca_k'] = int(k)
                r['got_cities__pca_refit_logistic_gap'] = rp['refit_logistic_gap']
                r['got_cities__sae_minus_pca_refit_gap'] = rs['refit_logistic_gap'] - rp['refit_logistic_gap']
                r['got_cities__original_auroc'] = orig['auroc']['logistic']
            rows.append(r)
            pd.DataFrame(rows).to_csv(out / 'm5_rows.partial.csv', index=False)
            print(f"L{L} {src['family']} {src['name']} {src['seed']} [{split}]: ev {r['owt_explained_variance']:.4f} "
                  f"l0 {r['owt_l0']:.1f} dCE {r['splice_delta_ce']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'm5_rows.csv', index=False)
    comp = compare(df)
    checks.update(runtime_s=time.time() - t0, script_sha256=sha(__file__), n_saes=len(df),
                  test_rows_used=0 if a.smoke else int(len(seqs) + n_got_eval))
    comp.to_csv(out / 'comparisons.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2, default=float))
    (out / 'm5_rows.partial.csv').unlink()
    if len(comp):
        print(comp.pivot_table(index=['comparison', 'metric'], columns='layer', values='cls', aggfunc='first').to_string())


if __name__ == '__main__':
    main()
