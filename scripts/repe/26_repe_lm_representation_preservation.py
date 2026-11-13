#!/usr/bin/env python3
"""RLM-RP: RepE-LM representation preservation and robustness for 216 SAEs (layers 0-11).

Pre-registered in results/repe/repe_lm_representation_preservation/README.md (section 1) and preregistration.json.
Corrected equal-length RepE-LM prompts only (data/repe/repe_lm/prompts.csv); the historical length-confounded RepE
results are not used. Final-token activations from the cache data/repe/activations/repe_lm/final_layer{L}.npy.
Probe definitions follow M2 (scripts/shared/m2_four_level_faithfulness.py) through src/evaluation/information_levels.py.

SAEs: 108 base (uninterrupted 15k-step OWT SAEs, layers 0-11), 81 IMP-L fine-tunes (layers 3, 8, 9) and 27 layer-6
fine-tunes (owt_only, Method A, output-KL = E2E Method B). No training; no GPT-2 forward pass.

Run from the repository root:
    python scripts/repe/26_repe_lm_representation_preservation.py --prepare
    python scripts/repe/26_repe_lm_representation_preservation.py --smoke DIR      (layers 1 and 8, 2 SAEs each)
    python scripts/repe/26_repe_lm_representation_preservation.py --run
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import information_levels as il  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/repe/repe_lm_representation_preservation'
DATA = REPO / 'data/repe/repe_lm'
CACHE = REPO / 'data/repe/activations/repe_lm'
FT = REPO / 'results/geometry_of_truth/sae_finetuning'
IMPL = REPO / 'results/geometry_of_truth/multilayer_finetune'
M2 = REPO / 'results/shared/m2_four_level_faithfulness'
LAYERS = list(range(12))
ARCHS, SEEDS = ('relu', 'topk', 'jumprelu'), (42, 1042, 2042)
ARMS = ('base', 'owt_only', 'method_a', 'kl')
FT_LAYERS = (3, 6, 8, 9)
N_BOOT, N_NOISE, SEED, K_TRUNC, WORKERS = 1000, 200, 20261108, 128, 3
T_975_DF2 = 4.302652729911275
OUTPUT_FILES = ['per_sae.csv', 'original_by_layer.csv', 'summary_by_layer_arch.csv', 'comparisons.csv',
                'criteria.json', 'checks.json']
HASHED = ['scripts/repe/26_repe_lm_representation_preservation.py', 'src/evaluation/information_levels.py',
          'src/evaluation/faithfulness_metrics.py', 'data/repe/repe_lm/prompts.csv',
          'data/repe/repe_lm/split_verification.json', 'results/repe/repe_lm_representation_preservation/README.md',
          'results/geometry_of_truth/multilayer_finetune/criteria.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


# ------------------------------------------------------------------------------------------------ SAEs

def ckpt(L, arm, arch, seed):
    if arm == 'base':
        return Path(uninterrupted_checkpoint(L, arch, seed))
    if L in (3, 8, 9):
        return FT / f'multilayer_finetune/layer_{L}/{arch}_{arm}_seed{seed}/finetuned_checkpoint.pt'
    if arm == 'owt_only':
        return FT / f'method_a_layer6/{arch}_owt_only_seed{seed}/finetuned_checkpoint.pt'
    if arm == 'method_a':
        if arch == 'topk':
            return FT / f'truth_aware_pilot_layer6_topk/reconstruction_only_seed{seed}/finetuned_checkpoint.pt'
        return FT / f'method_a_layer6/{arch}_got_replay_seed{seed}/finetuned_checkpoint.pt'
    return FT / f'e2e_inspired_layer6/{arch}_methodB_seed{seed}/finetuned_checkpoint.pt'


def sae_list(layers):
    out = []
    for L in layers:
        for arm in ARMS:
            if arm != 'base' and L not in FT_LAYERS:
                continue
            for arch in ARCHS:
                for seed in SEEDS:
                    out.append((L, arm, arch, seed, ckpt(L, arm, arch, seed)))
    return out


# ------------------------------------------------------------------------------------------------ data

def load_rows():
    d = pd.read_csv(DATA / 'prompts.csv')
    assert set(d.split) <= {'train', 'validation'}, 'RepE-LM must not contain test rows'
    sv = json.loads((DATA / 'split_verification.json').read_text())
    neardup = {int(x['validation_id']) for x in sv['K6_duplicates']['near_duplicates_jaccard_gt_0_8']}
    tr, va = d.split == 'train', d.split == 'validation'
    persona = d.template.isin(['T1', 'T2', 'T3', 'T4'])
    idx = dict(
        p_tr=np.flatnonzero(persona & d.variant.isin(['H', 'U']) & tr),
        p_va=np.flatnonzero(persona & d.variant.isin(['H', 'U']) & va),
        syn_va=np.flatnonzero(persona & d.variant.isin(['Hp', 'U']) & va),
        hhp_va=np.flatnonzero(persona & d.variant.isin(['H', 'Hp']) & va),
        ho_va=np.flatnonzero(d.template.isin(['T5', 'T6']) & va),
        n_tr=np.flatnonzero(d.template.isin(['N1', 'N2']) & tr),
        n_va=np.flatnonzero(d.template.isin(['N1', 'N2']) & va),
        all_va=np.flatnonzero(va))
    # paired H/U validation rows for direction metrics
    pv = d[persona & va & d.variant.isin(['H', 'U'])]
    piv = pv.reset_index().pivot(index='pair_id', columns='variant', values='index')
    idx['pair_H'], idx['pair_U'] = piv['H'].to_numpy(), piv['U'].to_numpy()
    y = d.label.to_numpy()
    # H vs Hp: label H as 1 and Hp as 0 (wording only; both honest)
    y_hhp = (d.variant == 'H').astype(int).to_numpy()
    checks = dict(
        test_rows=int((d.split == 'test').sum()),
        groups_cross_split=len(set(d.norm_group[tr]) & set(d.norm_group[va])),
        statements_cross_split=len(set(d.statement_id[tr]) & set(d.statement_id[va])),
        pairs_equal_length=bool(d.groupby(['statement_id', 'template']).n_tokens.nunique().max() == 1),
        n_rows=dict({k: int(len(v)) for k, v in idx.items()}),
        n_validation_statements=int(d.statement_id[va].nunique()), n_train_statements=int(d.statement_id[tr].nunique()),
        near_duplicate_validation_statements=sorted(neardup))
    assert checks['test_rows'] == 0 and checks['groups_cross_split'] == 0 and checks['statements_cross_split'] == 0
    assert checks['pairs_equal_length']
    return d, idx, y, y_hhp, d.norm_group.to_numpy(), d.statement_id.to_numpy(), neardup, checks


# ----------------------------------------------------------------------------------- original side

G = {}          # per-layer state; copied into each worker by _init (start method independent)


def _init(state):
    G.clear()
    G.update(state)


def original_side(X):
    idx, y, g = G['idx'], G['y'], G['g']
    pipe, C = il.fit_logistic(X[idx['p_tr']], y[idx['p_tr']], g[idx['p_tr']])
    direction = il.massmean_direction(X[idx['p_tr']], y[idx['p_tr']])
    a, c = il.linear_form(pipe)
    npipe, nC = il.fit_logistic(X[idx['n_tr']], y[idx['n_tr']], g[idx['n_tr']])
    o = dict(pipe=pipe, C=C, direction=direction, a=a, c=c, npipe=npipe, nC=nC)
    sc = lambda rows: pipe.decision_function(X[rows])
    o['s_p'] = sc(idx['p_va'])
    o['auc'] = dict(logistic=il.auroc(y[idx['p_va']], o['s_p']),
                    massmean=il.auroc(y[idx['p_va']], X[idx['p_va']] @ direction),
                    syn=il.auroc(y[idx['syn_va']], sc(idx['syn_va'])),
                    hhp=il.auroc(G['y_hhp'][idx['hhp_va']], sc(idx['hhp_va'])),
                    heldout=il.auroc(y[idx['ho_va']], sc(idx['ho_va'])),
                    neutral=il.auroc(y[idx['n_va']], npipe.decision_function(X[idx['n_va']])))
    keep = ~np.isin(G['sid'][idx['p_va']], list(G['neardup']))
    o['keep'] = keep
    o['auc']['logistic_no_neardup'] = il.auroc(y[idx['p_va']][keep], o['s_p'][keep])
    return o


# ------------------------------------------------------------------------------------- per SAE (worker)

def evaluate(meta, H, Htrunc, l0_rows, dead):
    X, idx, y, g, o = G['X'], G['idx'], G['y'], G['g'], G['orig']
    rng = np.random.default_rng(SEED)
    r = dict(meta)
    pv = idx['p_va']
    yv, Xv, Hv = y[pv], X[pv], H[pv]
    E = Xv - Hv
    # ---- reconstruction (persona validation rows, M2 definitions)
    r['task_rel_sq_error'] = float(((E ** 2).sum(1) / (Xv ** 2).sum(1)).mean())
    r['task_nmse'] = float((E ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum())
    r['task_cosine'] = float((np.sum(Xv * Hv, 1) / (np.linalg.norm(Xv, axis=1) * np.linalg.norm(Hv, axis=1))).mean())
    r['task_l0'] = float(l0_rows[pv].mean())
    # ---- reconstruction (all RepE-LM validation rows)
    av = idx['all_va']
    Xa, Ha = X[av], H[av]
    Ea = ((Xa - Ha) ** 2).sum(1)
    r['repe_val_mse'] = float(Ea.mean() / X.shape[1])
    r['repe_val_ev'] = float(1 - Ea.sum() / ((Xa - Xa.mean(0)) ** 2).sum())
    r['repe_val_cosine'] = float((np.sum(Xa * Ha, 1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Ha, axis=1) + 1e-12)).mean())
    r['repe_val_l0'] = float(l0_rows[av].mean())
    r['repe_val_dead_fraction'] = dead
    # ---- persona: frozen readers
    r['orig_persona_logistic_auroc'] = o['auc']['logistic']
    r['orig_persona_massmean_auroc'] = o['auc']['massmean']
    sh = o['pipe'].decision_function(Hv)
    r['frozen_logistic_auroc'] = il.auroc(yv, sh)
    r['frozen_logistic_gap'] = o['auc']['logistic'] - r['frozen_logistic_auroc']
    r['frozen_massmean_auroc'] = il.auroc(yv, Hv @ o['direction'])
    r['frozen_massmean_gap'] = o['auc']['massmean'] - r['frozen_massmean_auroc']
    proj_l = np.empty((N_NOISE, len(E)))
    proj_m = np.empty((N_NOISE, len(E)))
    for k in range(N_NOISE):
        eta = il.norm_matched_noise(E, 1, rng)[0].astype(np.float64)
        proj_l[k], proj_m[k] = eta @ o['a'], eta @ o['direction']
    r['noise_frozen_logistic_gap'] = o['auc']['logistic'] - np.mean([il.auroc(yv, s) for s in o['s_p'][None] + proj_l])
    r['noise_frozen_massmean_gap'] = o['auc']['massmean'] - np.mean(
        [il.auroc(yv, s) for s in (Xv @ o['direction'])[None] + proj_m])
    r['error_term_auroc'] = il.auroc(yv, E @ o['a'])
    # ---- persona: re-trained reader
    ptr = idx['p_tr']
    rpipe, rC = il.fit_logistic(H[ptr], y[ptr], g[ptr])
    srh = rpipe.decision_function(Hv)
    r['refit_logistic_C'] = rC
    r['refit_logistic_auroc'] = il.auroc(yv, srh)
    r['refit_logistic_gap'] = o['auc']['logistic'] - r['refit_logistic_auroc']
    # ---- bootstrap CIs (statements)
    bi = il.bootstrap_indices(g[pv], N_BOOT, rng)
    bo = il.boot_auroc(yv, o['s_p'], bi)
    r['frozen_logistic_gap_ci_low'], r['frozen_logistic_gap_ci_high'] = il.ci(bo - il.boot_auroc(yv, sh, bi))
    r['refit_logistic_gap_ci_low'], r['refit_logistic_gap_ci_high'] = il.ci(bo - il.boot_auroc(yv, srh, bi))
    # ---- near-duplicate sensitivity
    kp = o['keep']
    r['frozen_logistic_gap_no_neardup'] = o['auc']['logistic_no_neardup'] - il.auroc(yv[kp], sh[kp])
    r['refit_logistic_gap_no_neardup'] = o['auc']['logistic_no_neardup'] - il.auroc(yv[kp], srh[kp])
    # ---- synonym and wording sensitivity (frozen persona probe)
    sy, hh = idx['syn_va'], idx['hhp_va']
    r['orig_syn_auroc'], r['orig_hhp_auroc'] = o['auc']['syn'], o['auc']['hhp']
    r['frozen_syn_auroc'] = il.auroc(y[sy], o['pipe'].decision_function(H[sy]))
    r['frozen_syn_gap'] = o['auc']['syn'] - r['frozen_syn_auroc']
    r['frozen_hhp_auroc'] = il.auroc(G['y_hhp'][hh], o['pipe'].decision_function(H[hh]))
    # ---- held-out wording T5/T6 (never fitted)
    ho = idx['ho_va']
    r['orig_heldout_auroc'] = o['auc']['heldout']
    r['frozen_heldout_auroc'] = il.auroc(y[ho], o['pipe'].decision_function(H[ho]))
    r['frozen_heldout_gap'] = o['auc']['heldout'] - r['frozen_heldout_auroc']
    r['refit_heldout_auroc'] = il.auroc(y[ho], rpipe.decision_function(H[ho]))
    r['refit_heldout_gap'] = o['auc']['heldout'] - r['refit_heldout_auroc']
    # ---- neutral control
    nt, nv = idx['n_tr'], idx['n_va']
    r['orig_neutral_auroc'] = o['auc']['neutral']
    r['frozen_neutral_gap'] = o['auc']['neutral'] - il.auroc(y[nv], o['npipe'].decision_function(H[nv]))
    npp, _ = il.fit_logistic(H[nt], y[nt], g[nt])
    r['refit_neutral_gap'] = o['auc']['neutral'] - il.auroc(y[nv], npp.decision_function(H[nv]))
    # ---- direction geometry (paired H/U validation rows)
    D = (X[idx['pair_H']] - X[idx['pair_U']]).astype(np.float64)
    Dh = (H[idx['pair_H']] - H[idx['pair_U']]).astype(np.float64)
    nd = np.linalg.norm(D, axis=1)
    ok = nd > 1e-4
    r['direction_valid_pairs'] = int(ok.sum())
    if ok.sum() >= 10:
        nh = np.linalg.norm(Dh, axis=1)
        r['pair_dir_cosine'] = float((np.sum(D * Dh, 1)[ok] / (nd[ok] * nh[ok] + 1e-12)).mean())
        r['pair_norm_retention'] = float((nh[ok] / nd[ok]).mean())
        r['pair_error_share'] = float((np.linalg.norm(D - Dh, axis=1)[ok] / nd[ok]).mean())
        m, mh = D[ok].mean(0), Dh[ok].mean(0)
        r['massmean_dir_cosine'] = float(m @ mh / (np.linalg.norm(m) * np.linalg.norm(mh) + 1e-12))
        r['massmean_norm_retention'] = float(np.linalg.norm(mh) / np.linalg.norm(m))
    else:
        for k in ('pair_dir_cosine', 'pair_norm_retention', 'pair_error_share', 'massmean_dir_cosine',
                  'massmean_norm_retention'):
            r[k] = np.nan
    # ---- exploratory L0-truncation control (ReLU / JumpReLU)
    if Htrunc is not None:
        Ht = Htrunc[pv]
        r['trunc_task_nmse'] = float(((Xv - Ht) ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum())
        r['trunc_frozen_logistic_gap'] = o['auc']['logistic'] - il.auroc(yv, o['pipe'].decision_function(Ht))
        tp, _ = il.fit_logistic(Htrunc[ptr], y[ptr], g[ptr])
        r['trunc_refit_logistic_gap'] = o['auc']['logistic'] - il.auroc(yv, tp.decision_function(Ht))
    return r


# ------------------------------------------------------------------------------------------- GPU side

@torch.no_grad()
def encode(sae, X, arch, device):
    Hs, l0s, act, Ts = [], [], None, []
    trunc_err = 0.0
    for xb in torch.as_tensor(X).split(4096):
        rec, z, _ = sae(xb.to(device))
        z = z.float()
        Hs.append(rec.float().cpu())
        l0s.append((z > 0).sum(1).cpu())
        a = (z > 0).any(0)
        act = a if act is None else (act | a)
        if arch != 'topk':
            dec = F.linear(z, sae.decoder.weight) + sae.output_bias
            trunc_err = max(trunc_err, float((dec.float() - rec.float()).abs().max()))
            v, i = z.topk(K_TRUNC, dim=1)
            zt = torch.zeros_like(z).scatter_(1, i, v)
            Ts.append((F.linear(zt, sae.decoder.weight) + sae.output_bias).float().cpu())
    return (torch.cat(Hs).numpy(), torch.cat(l0s).numpy(), act,
            torch.cat(Ts).numpy() if Ts else None, trunc_err)


# ------------------------------------------------------------------------------------------ analysis

def summarise(df):
    metrics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and c not in ('layer', 'seed', 'refit_logistic_C')]
    rows = []
    for (L, arm, arch), g in df.groupby(['layer', 'arm', 'architecture']):
        r = dict(layer=L, arm=arm, architecture=arch, n_seeds=len(g))
        for c in metrics:
            v = g.sort_values('seed')[c].astype(float)
            r[f'{c}_mean'], r[f'{c}_sd'] = v.mean(), v.std()
            h = T_975_DF2 * v.std() / np.sqrt(len(v))
            r[f'{c}_ci95_low'], r[f'{c}_ci95_high'] = v.mean() - h, v.mean() + h
            for s, x in zip(g.sort_values('seed').seed, v):
                r[f'{c}_seed{s}'] = x
        rows.append(r)
    return pd.DataFrame(rows)


LOWER = ('gap', 'error', 'mse', 'nmse', 'dead', 'share')


def compare(df):
    metrics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and c not in ('layer', 'seed', 'refit_logistic_C') and not c.startswith('orig_')]
    impl = json.loads((IMPL / 'criteria.json').read_text())
    comps, crit = [], {}
    for (L, arch), sub in df[df.layer.isin(FT_LAYERS)].groupby(['layer', 'architecture']):
        piv = {a: sub[sub.arm == a].set_index('seed').sort_index() for a in ARMS}
        for a_, b_ in [('owt_only', 'base'), ('method_a', 'base'), ('method_a', 'owt_only'), ('kl', 'base'),
                       ('kl', 'owt_only'), ('kl', 'method_a')]:
            for m in metrics:
                dd = piv[a_][m].astype(float) - piv[b_][m].astype(float)
                if dd.isna().all():
                    continue
                mu, sd = dd.mean(), dd.std()
                comps.append(dict(layer=L, architecture=arch, comparison=f'{a_} - {b_}', metric=m,
                                  lower_is_better=any(t in m for t in LOWER), base_mean=piv[b_][m].mean(),
                                  arm_mean=piv[a_][m].mean(), mean_diff=mu, sd_diff=sd,
                                  ci95_low=mu - T_975_DF2 * sd / np.sqrt(3), ci95_high=mu + T_975_DF2 * sd / np.sqrt(3),
                                  pct_change=100 * mu / piv[b_][m].mean() if piv[b_][m].mean() else np.nan,
                                  **{f'seed{s}': dd.get(s) for s in SEEDS}))

        def lower(a_, b_, m):
            dd = piv[a_][m] - piv[b_][m]
            return {'pass': bool((dd < 0).all() and -dd.mean() > 2 * dd.std()), 'paired_diffs_by_seed': dd.round(5).tolist()}

        def higher(a_, b_, m):
            dd = piv[a_][m] - piv[b_][m]
            return bool((dd > 0).all() and dd.mean() > 2 * dd.std())

        if L in (3, 8, 9):
            gma, gkl = impl[f'layer{L}_{arch}']['method_a']['guards'], impl[f'layer{L}_{arch}']['kl']['guards']
            guard_note = 'IMP-L guards'
        else:
            gma = gkl = {}
            guard_note = 'layer 6: IMP-L-format guards not available; class reported without guard checks'

        def classify(primary, g, wrong):
            if g and not all(g.values()):
                return 'NOT CREDITABLE (guard failed)'
            if wrong:
                return 'REGRESSION'
            return 'PRELIMINARY IMPROVEMENT' if primary else 'NO IMPROVEMENT'

        def l0flag(arm):
            x, b = piv[arm].repe_val_l0, piv['base'].repe_val_l0
            return {'repe_l0_within_5pct_of_base': bool(((x - b).abs() / b <= 0.05).all()),
                    'repe_l0_pct_change_by_seed': (100 * (x / b - 1)).round(2).tolist()}

        cell = {'guard_source': guard_note, 'method_a_guards': gma, 'kl_guards': gkl,
                'method_a_repe_l0': l0flag('method_a'), 'kl_repe_l0': l0flag('kl')}
        for name, m in (('frozen', 'frozen_logistic_gap'), ('refit', 'refit_logistic_gap')):
            va, vo = lower('method_a', 'base', m), lower('method_a', 'owt_only', m)
            cell[f'RP_A_{name}'] = {'vs_base': va, 'vs_owt_only': vo,
                                    'class': classify(va['pass'] and vo['pass'], gma, higher('method_a', 'base', m))}
            vk = lower('kl', 'method_a', m)
            cell[f'RP_KL_{name}'] = {'vs_method_a': vk, 'class': classify(vk['pass'], gkl, higher('kl', 'method_a', m))}
        crit[f'layer{L}_{arch}'] = cell
    return pd.DataFrame(comps), crit


def reproduction(df):
    out = {}
    m2 = pd.read_csv(M2 / 'per_cell.csv')
    m2 = m2[(m2.task == 'repe_persona') & (m2.source == 'sae')].set_index(['layer', 'architecture', 'seed'])
    b = df[df.arm == 'base'].set_index(['layer', 'architecture', 'seed'])
    common = b.index.intersection(m2.index)
    for c in ('frozen_logistic_gap', 'frozen_massmean_gap', 'refit_logistic_gap', 'task_cosine', 'task_l0', 'task_nmse'):
        out[f'm2_{c}_max_abs_diff'] = float((b.loc[common, c] - m2.loc[common, c]).abs().max())
    out['m2_rows_compared'] = int(len(common))
    imp = pd.concat([pd.read_csv(IMPL / f'per_sae_layer{L}.csv') for L in (3, 8, 9)]).set_index(
        ['layer', 'arm', 'architecture', 'seed'])
    mine = df[df.layer.isin([3, 8, 9])].set_index(['layer', 'arm', 'architecture', 'seed'])
    common = mine.index.intersection(imp.index)
    out['impl_refit_gap_max_abs_diff'] = float((mine.loc[common, 'refit_logistic_gap']
                                                - imp.loc[common, 'repe_lm_persona_refit_logistic_gap']).abs().max())
    out['impl_persona_l0_max_abs_diff'] = float((mine.loc[common, 'task_l0']
                                                 - imp.loc[common, 'repe_lm_persona_val_l0']).abs().max())
    out['impl_rows_compared'] = int(len(common))
    tol = {'impl_refit_gap_max_abs_diff': 1e-3, 'impl_persona_l0_max_abs_diff': 1e-4}
    out['tolerances'] = dict(tol, m2=1e-6)
    out['pass'] = bool(all(v < tol.get(k, 1e-6) for k, v in out.items() if k.endswith('max_abs_diff')))
    return out


# ------------------------------------------------------------------------------------------------ main

def prepare():
    p = OUT / 'preregistration.json'
    if p.exists():
        raise SystemExit(f'{p} exists; refusing to overwrite.')
    ck = {}
    for L, arm, arch, seed, path in sae_list(LAYERS):
        if not path.exists():
            raise SystemExit(f'missing {path}')
        if arm != 'base':
            c = torch.load(path, map_location='cpu', weights_only=False)
            src = c.get('source_checkpoint')
            assert src == rel(uninterrupted_checkpoint(L, arch, seed)), f'{path}: source {src} is not the base'
        ck[rel(path)] = sha(path)
    pr = dict(experiment='RLM-RP RepE-LM representation preservation and robustness (216 SAEs)',
              date=time.strftime('%Y-%m-%d %H:%M:%S'),
              approved_by_user='2026-11-10 ("Complete RepE representation preservation ... Complete RepE '
                               'generalization and robustness")',
              design='results/repe/repe_lm_representation_preservation/README.md section 1',
              evidence_label='VALIDATION (TRAIN for probe fitting and C; no test rows exist in RepE-LM)',
              hashes={r_: sha(REPO / r_) for r_ in HASHED}, checkpoint_hashes=ck, n_saes=len(ck))
    p.write_text(json.dumps(pr, indent=2))
    print(f'wrote {p} ({len(ck)} checkpoints)')


def run(out, smoke):
    if not smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if not r_.endswith('README.md') and sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs from preregistration.json; refusing to run.')
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = 'cuda'
    d, idx, y, y_hhp, g, sid, neardup, checks = load_rows()
    G.update(idx=idx, y=y, y_hhp=y_hhp, g=g, sid=sid, neardup=neardup)
    layers = [1, 8] if smoke else LAYERS
    saes = sae_list(layers)
    if smoke:
        saes = [s for s in saes if s[3] == 42 and s[1] in ('base', 'kl') and s[2] in ('relu', 'topk')]
    else:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for *_, path in saes:
            if sha(path) != pr['checkpoint_hashes'][rel(path)]:
                raise SystemExit(f'{path}: checkpoint hash differs; refusing to run.')
    rows, orig_rows, trunc_err = [], [], 0.0
    for L in layers:
        X = np.load(CACHE / f'final_layer{L}.npy').astype(np.float32)
        G['X'] = X
        G['orig'] = original_side(X)
        o = G['orig']
        orig_rows.append(dict(layer=L, persona_logistic_C=o['C'], neutral_logistic_C=o['nC'],
                              **{f'orig_{k}_auroc': v for k, v in o['auc'].items()}))
        jobs = []
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(dict(G),)) as ex:
            for (L_, arm, arch, seed, path) in [s for s in saes if s[0] == L]:
                sae, _ = fm.load_sae(path, device, architecture=arch)
                H, l0r, act, Ht, te = encode(sae, X, arch, device)
                trunc_err = max(trunc_err, te)
                va_act = None
                with torch.no_grad():                 # dead fraction on all RepE-LM validation rows
                    zs = []
                    for xb in torch.as_tensor(X[idx['all_va']]).split(4096):
                        zs.append((sae(xb.to(device))[1] > 0).any(0))
                    va_act = torch.stack(zs).any(0)
                dead = float((~va_act).float().mean())
                meta = dict(layer=L, arm=arm, architecture=arch, seed=seed, checkpoint=rel(path), checkpoint_sha256=sha(path))
                jobs.append(ex.submit(evaluate, meta, H, Ht, l0r, dead))
                del sae
            for j in jobs:
                rows.append(j.result())
        print(f'layer {L}: {len(jobs)} SAEs ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    checks['truncation_decode_max_abs_err'] = trunc_err
    rep = reproduction(df) if not smoke else {}
    checks['reproduction'] = rep
    summ = summarise(df)
    comps, crit = compare(df) if not smoke else (pd.DataFrame(), {})
    checks.update(n_saes=len(df), runtime_s=time.time() - t0, test_rows_used=0, script_sha256=sha(__file__),
                  smoke=smoke, layer0_note='persona AUROC 0.5 by construction at layer 0; direction metrics NaN')
    checks['pass'] = bool(trunc_err < 1e-3 and (smoke or rep['pass']))
    df.to_csv(out / 'per_sae.csv', index=False)
    pd.DataFrame(orig_rows).to_csv(out / 'original_by_layer.csv', index=False)
    summ.to_csv(out / 'summary_by_layer_arch.csv', index=False)
    comps.to_csv(out / 'comparisons.csv', index=False)
    (out / 'criteria.json').write_text(json.dumps(crit, indent=2, default=float))
    (out / 'checks.json').write_text(json.dumps(checks, indent=2, default=float))
    print(json.dumps(checks, indent=1, default=float))
    if not smoke:
        print(json.dumps({k: {h: v[h]['class'] for h in v if h.startswith('RP_')} for k, v in crit.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument('--prepare', action='store_true')
    gx.add_argument('--smoke', type=Path)
    gx.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.smoke:
        run(a.smoke, True)
    else:
        run(OUT, False)


if __name__ == '__main__':
    main()
