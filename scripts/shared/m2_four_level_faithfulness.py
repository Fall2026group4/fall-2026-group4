#!/usr/bin/env python3
"""M2: four-level information faithfulness of the 108 OWT SAEs, all 12 layers, on GoT cities/neg_cities and the
M1 RepE-LM task (persona T1-T4 AND neutral N1-N2, always side by side). VALIDATION scoring; TRAIN fitting/selection.

Pre-registered in results/shared/m2_four_level_faithfulness/preregistration.json.

Sources compared with the original activation x (blocks.L.hook_resid_pre, final token):
  * 108 SAE reconstructions x_hat (+ codes z for the sparse-z probe),
  * 36 PCA reconstructions (PCA fitted on OWT TRAIN activations, BOS excluded; one per layer x architecture, with k
    the smallest whose OWT VALIDATION NMSE is <= that architecture's mean OWT validation NMSE),
  * norm-matched isotropic noise x + eta, ||eta_i|| = ||e_i|| (200 draws), for the frozen linear probes.
Probe ladder per source: frozen original logistic and mass-mean; logistic, MLP, kNN refitted on the source;
sparse-z (SAE codes); error-term probe (frozen original logistic direction on e = x - x_hat alone, exact identity
w.x = w.x_hat + w.e); learning curve (logistic at 50 % vs 100 % of train groups).
Labels: src/evaluation/information_levels.py:label_cell.

Run from repository root:  python scripts/shared/m2_four_level_faithfulness.py [--smoke]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

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
from joblib import Parallel, delayed
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import information_levels as il  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/shared/m2_four_level_faithfulness'
M0 = REPO / 'results/shared/m0_within_layer_reanalysis'
OWT_TRAIN = REPO / 'data/activations/gpt2_small/train'
OWT_VAL = REPO / 'data/activations/gpt2_small/validation'
LAYERS, ARCHS, SEEDS = list(range(12)), ('relu', 'topk', 'jumprelu'), (42, 1042, 2042)
N_BOOT, N_NOISE, N_NOISE_BOOT, SEED = 1000, 200, 20, 20261108
INFORMATIVE = 0.60
OUTPUT_FILES = ['per_cell.csv', 'original_by_layer.csv', 'pca_reference.csv', 'h2a_h2b.csv', 'summary_by_layer_arch.csv',
                'persona_vs_neutral.csv', 'sparsity_width_within_layer.csv', 'checks.json', 'bootstrap_arrays.npz']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data

def load_tasks():
    spec = importlib.util.spec_from_file_location('got03', REPO / 'scripts/geometry_of_truth/03_original_truth_probes.py')
    got03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(got03)
    got03.ACT_DIR, got03.SPLIT_DIR = REPO / got03.ACT_DIR, REPO / got03.SPLIT_DIR
    parts = got03.load_domain(got03.DOMAINS['geographic'])       # combined cache file; test rows dropped at once
    split = np.concatenate([p['splits'] for p in parts])
    keep = np.isin(split, ['train', 'val'])
    tasks = {'got_cities': dict(
        X={L: np.concatenate([p['layers'][L].numpy() for p in parts]).astype(np.float32)[keep] for L in LAYERS},
        y=np.concatenate([p['labels'] for p in parts])[keep],
        split=np.where(split[keep] == 'val', 'validation', split[keep]),
        groups=np.concatenate([p['groups'] for p in parts])[keep])}
    del parts
    assert 'test' not in set(tasks['got_cities']['split'])
    d = pd.read_csv(REPO / 'data/repe/repe_lm/prompts.csv')
    assert set(d.split) <= {'train', 'validation'}
    acts = {L: np.load(REPO / f'data/repe/activations/repe_lm/final_layer{L}.npy') for L in LAYERS}
    for name, m in [('repe_persona', d.template.isin(['T1', 'T2', 'T3', 'T4']) & d.variant.isin(['H', 'U'])),
                    ('repe_neutral', d.template.isin(['N1', 'N2']))]:
        idx = np.flatnonzero(m.to_numpy())
        tasks[name] = dict(X={L: acts[L][idx] for L in LAYERS}, y=d.label.to_numpy()[idx],
                           split=d.split.to_numpy()[idx], groups=d.norm_group.to_numpy()[idx])
    for t in tasks.values():
        assert set(t['split']) == {'train', 'validation'}
        tr, va = t['split'] == 'train', t['split'] == 'validation'
        assert not (set(t['groups'][tr]) & set(t['groups'][va])), 'group crosses train/validation'
    return tasks


def owt_pca(smoke=False):
    """Mean and covariance (float64) of OWT TRAIN activations per layer, BOS excluded; one pass over the chunks."""
    paths = sorted(OWT_TRAIN.glob('chunk_*.pt'))[: (5 if smoke else None)]
    s = {L: np.zeros(768) for L in LAYERS}
    ss = {L: np.zeros((768, 768)) for L in LAYERS}
    n = 0
    for p in paths:
        ch = torch.load(p, map_location='cpu', weights_only=False)
        rows = [(i, int(l)) for i, l in enumerate(ch['lengths']) if int(l) > 1]
        for L in LAYERS:
            a = ch['activations'][f'blocks.{L}.hook_resid_pre']
            x = torch.cat([a[i, 1:l].double() for i, l in rows]).numpy()
            s[L] += x.sum(0)
            ss[L] += x.T @ x
        n += sum(l - 1 for _, l in rows)
    out = {}
    for L in LAYERS:
        mu = s[L] / n
        out[L] = (mu, ss[L] / n - np.outer(mu, mu))
    return out, n


def owt_val_layer(L, smoke=False):
    paths = sorted(OWT_VAL.glob('chunk_*.pt'))[: (3 if smoke else None)]
    xs = []
    for p in paths:
        ch = torch.load(p, map_location='cpu', weights_only=False)
        a = ch['activations'][f'blocks.{L}.hook_resid_pre']
        xs += [a[i, 1:int(l)].float() for i, l in enumerate(ch['lengths']) if int(l) > 1]
    return torch.cat(xs).numpy()


def nmse(x, xh):
    return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


# -------------------------------------------------------------------------------------- original side

def original_side(t, L, rng):
    X, y, g = t['X'][L], t['y'], t['groups']
    tr, va = t['split'] == 'train', t['split'] == 'validation'
    pipe, C = il.fit_logistic(X[tr], y[tr], g[tr])
    direction = il.massmean_direction(X[tr], y[tr])
    a, c = il.linear_form(pipe)
    s = dict(logistic=pipe.decision_function(X[va]), massmean=X[va] @ direction,
             mlp=il.mlp_scores(X[tr], y[tr], X[va]), knn=il.knn_scores(X[tr], y[tr], X[va]))
    s['sparse_z'] = s['logistic']                     # sparse-z is compared with the original logistic readout
    half = half_groups(g[tr], rng)
    lc50 = il.logistic(C).fit(X[tr][half], y[tr][half]).decision_function(X[va])
    auc = {k: il.auroc(y[va], v) for k, v in s.items()}
    return dict(pipe=pipe, C=C, direction=direction, a=a, c=c, scores=s, auroc=auc,
                lc50=il.auroc(y[va], lc50), identity_err=float(np.abs(X[va] @ a + c - s['logistic']).max()))


def half_groups(gtr, rng):
    u = np.unique(gtr)
    chosen = set(rng.choice(u, len(u) // 2, replace=False))
    return np.array([x in chosen for x in gtr])


# --------------------------------------------------------------------------------------- one source

def evaluate_source(meta, X, H, Z, y, split, groups, orig, boot_idx, seed):
    rng = np.random.default_rng(seed)
    tr, va = split == 'train', split == 'validation'
    yv = y[va]
    Hva, Xva = H[va], X[va]
    E = Xva - Hva
    r = dict(meta)
    r['task_rel_sq_error'] = float(((E ** 2).sum(1) / (Xva ** 2).sum(1)).mean())
    r['task_nmse'] = nmse(Xva, Hva)
    r['task_cosine'] = float((np.sum(Xva * Hva, 1) / (np.linalg.norm(Xva, axis=1) * np.linalg.norm(Hva, axis=1))).mean())
    if Z is not None:
        r['task_l0'] = float((Z[va] > 0).sum(1).mean())
    # ---- frozen original probes on x_hat
    fro = dict(logistic=orig['pipe'].decision_function(Hva), massmean=Hva @ orig['direction'])
    for k, v in fro.items():
        r[f'frozen_{k}_auroc'] = il.auroc(yv, v)
        r[f'frozen_{k}_gap'] = orig['auroc'][k] - r[f'frozen_{k}_auroc']
    # ---- norm-matched noise reference (frozen probes are linear: score(x + eta) = score(x) + w.eta)
    proj_l = np.empty((N_NOISE, len(E)))
    proj_m = np.empty((N_NOISE, len(E)))
    for k in range(N_NOISE):
        eta = il.norm_matched_noise(E, 1, rng)[0].astype(np.float64)
        proj_l[k], proj_m[k] = eta @ orig['a'], eta @ orig['direction']
    nl = orig['scores']['logistic'][None] + proj_l
    nm = orig['scores']['massmean'][None] + proj_m
    r['noise_frozen_logistic_gap'] = orig['auroc']['logistic'] - np.mean([il.auroc(yv, s) for s in nl])
    r['noise_frozen_massmean_gap'] = orig['auroc']['massmean'] - np.mean([il.auroc(yv, s) for s in nm])
    # ---- error-term probe and exact identity
    se = E @ orig['a']
    r['error_term_auroc'] = il.auroc(yv, se)
    r['identity_max_abs_err'] = float(np.abs(Hva @ orig['a'] + se - (Xva @ orig['a'])).max())
    # ---- refitted families on the source
    Htr = H[tr]
    pipe, C = il.fit_logistic(Htr, y[tr], groups[tr])
    ref = dict(logistic=pipe.decision_function(Hva), mlp=il.mlp_scores(Htr, y[tr], Hva),
               knn=il.knn_scores(Htr, y[tr], Hva))
    if Z is not None:
        Zs = sparse.csr_matrix(Z)
        zp, zC = il.fit_sparse_z(Zs[np.flatnonzero(tr)], y[tr], groups[tr])
        ref['sparse_z'] = zp.decision_function(Zs[np.flatnonzero(va)])
        r['sparse_z_C'] = zC
    r['refit_logistic_C'] = C
    for k, v in ref.items():
        r[f'refit_{k}_auroc'] = il.auroc(yv, v)
        r[f'refit_{k}_gap'] = orig['auroc'][k] - r[f'refit_{k}_auroc']
    half = half_groups(groups[tr], np.random.default_rng(seed + 1))
    r['lc50_refit_logistic_auroc'] = il.auroc(yv, il.logistic(C).fit(Htr[half], y[tr][half]).decision_function(Hva))
    r['lc_gain_source'] = r['refit_logistic_auroc'] - r['lc50_refit_logistic_auroc']
    r['lc_gain_original'] = orig['auroc']['logistic'] - orig['lc50']
    r['lc_sample_inefficient'] = bool(r['lc_gain_source'] >= 0.02 and r['lc_gain_original'] < 0.01)
    # ---- paired grouped bootstrap (same resamples for every source of this task and layer)
    yb = [yv[i] for i in boot_idx]
    def bs(s):
        return np.array([il.auroc(yy, s[i]) for yy, i in zip(yb, boot_idx)])
    ob = {k: bs(orig['scores'][k]) for k in ('logistic', 'massmean', 'mlp', 'knn')}
    ob['sparse_z'] = ob['logistic']
    arr = {}
    for k, v in fro.items():
        arr[f'frozen_{k}_gap'] = ob[k] - bs(v)
    for k, v in ref.items():
        arr[f'refit_{k}_gap'] = ob[k] - bs(v)
    arr['noise_frozen_logistic_gap'] = ob['logistic'] - np.mean([bs(s) for s in nl[:N_NOISE_BOOT]], 0)
    arr['noise_frozen_massmean_gap'] = ob['massmean'] - np.mean([bs(s) for s in nm[:N_NOISE_BOOT]], 0)
    arr['error_term_auroc'] = bs(se)
    for k, v in arr.items():
        r[f'{k}_ci_low'], r[f'{k}_ci_high'] = il.ci(v)
    r['displacement_flag'] = bool(r['error_term_auroc_ci_low'] > 0.5)
    return r, {k: arr[k].astype(np.float32) for k in ('frozen_logistic_gap', 'refit_logistic_gap',
                                                       'noise_frozen_logistic_gap', 'frozen_massmean_gap',
                                                       'noise_frozen_massmean_gap')}


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='layers 0 and 6, seed 42 only, into a scratch directory')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    out = a.smoke_out if a.smoke else OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        rel = str(Path(__file__).resolve().relative_to(REPO))
        if pr['hashes'].get(rel) != sha(__file__) or pr['hashes'].get('src/evaluation/information_levels.py') \
                != sha(REPO / 'src/evaluation/information_levels.py'):
            raise SystemExit('script/module hash differs from the pre-registered hash; refusing to run.')
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    layers = [0, 6] if a.smoke else LAYERS
    seeds = (42,) if a.smoke else SEEDS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    tasks = load_tasks()
    m0 = pd.read_csv(M0 / 'merged_validation_table.csv')
    live = m0.set_index(['layer', 'architecture', 'seed']).live_latents
    matched = m0.set_index(['layer', 'architecture', 'seed']).included
    arch_nmse = m0[m0.included].groupby(['layer', 'architecture']).owt_val_nmse.mean()
    print('PCA pass over OWT train ...', flush=True)
    pca_stats, n_owt = owt_pca(a.smoke)
    rows, pca_rows, orig_rows, boots, checks = [], [], [], {}, dict(identity_errors=[], n_owt_train_tokens=n_owt)
    for L in layers:
        rng = np.random.default_rng(SEED + L)
        origs = {}
        for tn, t in tasks.items():
            origs[tn] = original_side(t, L, rng)
            o = origs[tn]
            orig_rows.append(dict(layer=L, task=tn, C=o['C'], lc50_logistic=o['lc50'], identity_err=o['identity_err'],
                                  **{f'original_{k}_auroc': v for k, v in o['auroc'].items() if k != 'sparse_z'}))
        # sources: SAEs (GPU reconstruction here) and PCA
        mu, cov = pca_stats[L]
        w, V = il.pca_basis(mu, cov)
        xv_owt = owt_val_layer(L, a.smoke)
        jobs = []
        for arch in ARCHS:
            k = il.pca_k_for_val_nmse(xv_owt, mu, V, arch_nmse.loc[(L, arch)])
            pca_rows.append(dict(layer=L, matched_to=arch, target_owt_val_nmse=arch_nmse.loc[(L, arch)], k=k,
                                 pca_owt_train_nmse=float(w[k:].sum() / w.sum()),
                                 pca_owt_val_nmse=nmse(xv_owt, il.pca_reconstruct(xv_owt, mu, V, k))))
            for tn, t in tasks.items():
                H = il.pca_reconstruct(t['X'][L].astype(np.float64), mu, V, k).astype(np.float32)
                meta = dict(layer=L, task=tn, source='pca', architecture=arch, seed=-1, pca_k=k, matched=True,
                            live_latents=np.nan)
                jobs.append((meta, t, H, None))
            for s in seeds:
                model, _ = fm.load_sae(uninterrupted_checkpoint(L, arch, s), device)
                for tn, t in tasks.items():
                    with torch.no_grad():
                        rec, z, _ = model(torch.as_tensor(t['X'][L]).to(device))
                    meta = dict(layer=L, task=tn, source='sae', architecture=arch, seed=s, pca_k=np.nan,
                                matched=bool(matched.loc[(L, arch, s)]), live_latents=float(live.loc[(L, arch, s)]))
                    jobs.append((meta, t, rec.float().cpu().numpy(), z.float().cpu().numpy()))
                del model
        bidx = {tn: il.bootstrap_indices(t['groups'][t['split'] == 'validation'], N_BOOT,
                                         np.random.default_rng(SEED + 100 + L)) for tn, t in tasks.items()}
        res = Parallel(n_jobs=4, backend='loky', max_nbytes='1M')(
            delayed(evaluate_source)(meta, t['X'][L], H, Z, t['y'], t['split'], t['groups'], strip(origs[meta['task']]),
                                     bidx[meta['task']], SEED + 1000 * L + i)
            for i, (meta, t, H, Z) in enumerate(jobs))
        for (r, b) in res:
            rows.append(r)
            key = f"{r['layer']}/{r['task']}/{r['source']}/{r['architecture']}/{r['seed']}"
            for k, v in b.items():
                boots[f'{key}/{k}'] = v
            checks['identity_errors'].append(r['identity_max_abs_err'])
        print(f'layer {L} done: {len(jobs)} cells, {time.time() - t0:.0f}s', flush=True)
        del jobs, res

    df = pd.DataFrame(rows)
    og = pd.DataFrame(orig_rows)
    pc = pd.DataFrame(pca_rows)
    df = label_all(df, og)
    h2 = hypotheses(df, boots)
    df.to_csv(out / 'per_cell.csv', index=False)
    og.to_csv(out / 'original_by_layer.csv', index=False)
    pc.to_csv(out / 'pca_reference.csv', index=False)
    h2.to_csv(out / 'h2a_h2b.csv', index=False)
    summaries(df, out)
    np.savez_compressed(out / 'bootstrap_arrays.npz', **{k.replace('/', '__'): v for k, v in boots.items()})
    checks.update(max_identity_error=float(np.max(checks.pop('identity_errors'))),
                  max_original_identity_error=float(og.identity_err.max()),
                  n_cells=int(len(df)), n_sae_cells=int((df.source == 'sae').sum()),
                  pca_val_vs_target_max_abs=float((pc.pca_owt_val_nmse - pc.target_owt_val_nmse).abs().max()),
                  test_rows_used=0, runtime_s=time.time() - t0, script_sha256=sha(__file__),
                  module_sha256=sha(REPO / 'src/evaluation/information_levels.py'))
    checks['identity_check_pass'] = checks['max_identity_error'] < 1e-3 and checks['max_original_identity_error'] < 1e-3
    (out / 'checks.json').write_text(json.dumps(checks, indent=2, default=float))
    print(json.dumps(checks, indent=1, default=float))


def strip(o):
    return {k: v for k, v in o.items()}


def label_all(df, og):
    o = og.set_index(['layer', 'task'])
    pca = df[df.source == 'pca'].set_index(['layer', 'task', 'architecture']).refit_logistic_gap
    labels, red = [], []
    for _, r in df.iterrows():
        oo = o.loc[(r.layer, r.task)]
        fams = ['logistic', 'mlp', 'knn'] + (['sparse_z'] if r.source == 'sae' else [])
        s = dict(informative=bool(oo.original_logistic_auroc >= INFORMATIVE),
                 frozen_massmean_gap_ci=(r.frozen_massmean_gap_ci_low, r.frozen_massmean_gap_ci_high),
                 frozen_logistic_gap_ci=(r.frozen_logistic_gap_ci_low, r.frozen_logistic_gap_ci_high),
                 refit_gap_ci={f: (r[f'refit_{f}_gap_ci_low'], r[f'refit_{f}_gap_ci_high']) for f in fams},
                 original_auroc={f: oo[f'original_{"logistic" if f == "sparse_z" else f}_auroc'] for f in fams},
                 lc_sample_inefficient=bool(r.lc_sample_inefficient), refit_logistic_gap=r.refit_logistic_gap,
                 pca_refit_logistic_gap=(pca.loc[(r.layer, r.task, r.architecture)] if r.source == 'sae' else None))
        lab, rf = il.label_cell(s)
        labels.append(lab)
        red.append('+'.join(rf))
    df['label'] = labels
    df['reduced_families'] = red
    df['informative'] = df.apply(lambda r: bool(o.loc[(r.layer, r.task)].original_logistic_auroc >= INFORMATIVE), 1)
    return df


def hypotheses(df, boots):
    """H2a: SAE frozen gap > noise frozen gap (logistic and mass-mean). H2b: SAE refit-logistic gap vs PCA (matched NMSE).
    One-sided paired bootstrap p; BH over the 108 SAEs within each task, separately per hypothesis."""
    out = []
    for _, r in df[df.source == 'sae'].iterrows():
        key = f'{r.layer}/{r.task}/sae/{r.architecture}/{r.seed}'
        pk = f'{r.layer}/{r.task}/pca/{r.architecture}/-1'
        d = dict(layer=r.layer, task=r.task, architecture=r.architecture, seed=r.seed, matched=r.matched,
                 informative=r.informative)
        for fam in ('logistic', 'massmean'):
            diff = boots[f'{key}/frozen_{fam}_gap'] - boots[f'{key}/noise_frozen_{fam}_gap']
            d[f'h2a_{fam}_sae_minus_noise'] = r[f'frozen_{fam}_gap'] - r[f'noise_frozen_{fam}_gap']
            d[f'h2a_{fam}_ci_low'], d[f'h2a_{fam}_ci_high'] = il.ci(diff)
            d[f'h2a_{fam}_p'] = il.p_greater(diff)
        diff = boots[f'{key}/refit_logistic_gap'] - boots[f'{pk}/refit_logistic_gap']
        d['h2b_sae_minus_pca_refit_gap'] = r.refit_logistic_gap - df[(df.source == 'pca') & (df.layer == r.layer) & (
            df.task == r.task) & (df.architecture == r.architecture)].refit_logistic_gap.iloc[0]
        d['h2b_ci_low'], d['h2b_ci_high'] = il.ci(diff)
        d['h2b_p_sae_better'] = il.p_greater(-diff)
        d['h2b_p_sae_worse'] = il.p_greater(diff)
        out.append(d)
    h = pd.DataFrame(out)
    for col in ('h2a_logistic_p', 'h2a_massmean_p', 'h2b_p_sae_better', 'h2b_p_sae_worse'):
        h[col + '_bh'] = np.nan
        for t in h.task.unique():
            m = h.task == t
            h.loc[m, col + '_bh'] = il.bh(h.loc[m, col].to_numpy())
    return h


def summaries(df, out):
    sae = df[(df.source == 'sae') & df.matched]
    cols = ['frozen_logistic_gap', 'frozen_massmean_gap', 'noise_frozen_logistic_gap', 'refit_logistic_gap',
            'refit_mlp_gap', 'refit_knn_gap', 'refit_sparse_z_gap', 'error_term_auroc', 'task_rel_sq_error',
            'task_l0', 'live_latents']
    g = sae.groupby(['task', 'layer', 'architecture'])
    s = g[cols].agg(['mean', 'std'])
    s.columns = [f'{a}_{b}' for a, b in s.columns]
    s['labels'] = g.label.agg(lambda v: '; '.join(f'{k} x{n}' for k, n in v.value_counts().items()))
    s['n_displacement'] = g.displacement_flag.sum()
    s = s.reset_index()
    p = df[df.source == 'pca'].set_index(['task', 'layer', 'architecture'])
    for c in ('frozen_logistic_gap', 'refit_logistic_gap', 'refit_mlp_gap', 'pca_k'):
        s[f'pca_{c}'] = [p.loc[(r.task, r.layer, r.architecture), c] for r in s.itertuples()]
    s.to_csv(out / 'summary_by_layer_arch.csv', index=False)
    pv = s[s.task == 'repe_persona'].merge(s[s.task == 'repe_neutral'], on=['layer', 'architecture'],
                                           suffixes=('_persona', '_neutral'))
    keep = ['layer', 'architecture'] + [f'{c}_mean_{x}' for c in ('frozen_logistic_gap', 'refit_logistic_gap',
                                                                   'refit_mlp_gap', 'error_term_auroc')
                                        for x in ('persona', 'neutral')] + ['labels_persona', 'labels_neutral']
    pv[keep].to_csv(out / 'persona_vs_neutral.csv', index=False)
    from scipy.stats import spearmanr
    sw = []
    for (t, L), gg in sae[sae.informative].groupby(['task', 'layer']):
        for y in ('frozen_logistic_gap', 'refit_logistic_gap', 'refit_mlp_gap'):
            for x in ('task_l0', 'live_latents', 'task_rel_sq_error'):
                rho = spearmanr(gg[x], gg[y])[0] if gg[x].nunique() > 1 else np.nan
                sw.append(dict(task=t, layer=L, x=x, y=y, n=len(gg), spearman=rho))
    pd.DataFrame(sw).to_csv(out / 'sparsity_width_within_layer.csv', index=False)


if __name__ == '__main__':
    main()
