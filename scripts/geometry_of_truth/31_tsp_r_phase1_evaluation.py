#!/usr/bin/env python3
"""TSP-R Phase 1 evaluation and analysis (frozen pre-registration: replication/preregistration.json, D42).

Stages (run from the repository root, in order; every stage is resumable or refuses to overwrite):
  cities    per SAE x fold: probe scores on V_f (cities train+val only; cities TEST never loaded). Resumable: one
            npz per evaluation under replication/eval/cities/. Can run while fine-tunes are still training.
  owt       OWT validation cache: per-sequence NMSE and L0 for every SAE; spliced CE / KL per sequence (fold 0).
  ladder    the unchanged shared ladder (scripts/shared/sae_ladder_evaluation.py) on the 48 fold-0 SAEs:
            RepE persona re-fit gap, M4 T_rec/T_full, RepE L0 (cost guards).
  analyze   primary (cities) statistics, gatekeeping, labels, sensitivity, cost guards -> eval/analysis/.
            Commit its outputs before the next stage.
  test      LOCKED fresh-family TEST (companies secondary, common_claim distortion, counterfact exploratory). Refuses
            unless eval/analysis/primary.json is committed and unchanged.
  final     combines the committed cities analysis with the TEST results into the final verdict.
Metric definitions, uncertainty, multiplicity and labels follow preregistration.json exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import tspr_stats as T  # noqa: E402

ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
OUT = ROOT / 'replication'
EV = OUT / 'eval'
AN = EV / 'analysis'
LAYER = 8
C_GRID = (0.01, 0.1, 1.0)
N_BOOT, N_TWOWAY, BOOT_SEED = 2000, 4000, 20261115
READERS = ('frozen_logistic', 'frozen_massmean', 'refit_logistic', 'refit_mlp10', 'refit_mlp256')


def mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S30 = mod('scripts/geometry_of_truth/30_tsp_r_phase1_training.py', 's30')


def sha(p):
    return S30.sha(p)


# ------------------------------------------------------------------------------------------------ SAEs
class TopK:
    """TopK SAE from a modern_v1 state dict (scale folded): x_hat = W_d topk(relu(W_e (x - b) + b_e)) + b."""

    def __init__(self, path, device):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        s, cfg = ck['model_state_dict'], ck['config']
        assert cfg['architecture'] == 'topk' and cfg['format'] == 'modern_v1'
        self.k = int(cfg['k'])
        self.We, self.be = s['encoder.weight'].to(device), s['encoder.bias'].to(device)
        self.Wd, self.b = s['decoder.weight'].to(device), s['output_bias'].to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        pre = torch.relu((x - self.b) @ self.We.T + self.be)
        v, i = pre.topk(self.k, -1)
        z = torch.zeros_like(pre).scatter(-1, i, v)
        return z @ self.Wd.T + self.b, (z > 0).sum(-1).float()

    def np(self, X, bs=4096):
        H, L0 = [], []
        for b in range(0, len(X), bs):
            h, l0 = self(torch.as_tensor(X[b:b + bs], device=self.device, dtype=torch.float32))
            H.append(h.cpu().numpy())
            L0.append(l0.cpu().numpy())
        return np.concatenate(H), np.concatenate(L0)


def sae_list():
    """Every SAE of the design with its verified SHA-256: 8 bases and the ledgered fine-tunes."""
    M = pd.read_csv(OUT / 'plan_matrix.csv')
    led = pd.read_csv(S30.LEDGER) if S30.LEDGER.exists() else pd.DataFrame(columns=['run_id', 'sha256'])
    qc = json.loads((OUT / 'base_qc.json').read_text())
    plan = json.loads((OUT / 'plan_info.json').read_text())
    base_sha = {**{int(k): v for k, v in plan['existing_base_sha256'].items()},
                **{int(k): v for k, v in qc['new_sha256'].items()}}
    out = [dict(sae_id=f'base_s{s}', arm='base', seed=s, fold=None, path=S30.base_path(s), sha256=base_sha[s])
           for s in S30.SEEDS]
    done = dict(zip(led.run_id, led.sha256))
    for r in M.itertuples():
        if r.run_id in done:
            out.append(dict(sae_id=r.run_id, arm=r.arm, seed=int(r.seed), fold=int(r.fold), path=REPO / r.checkpoint,
                            sha256=done[r.run_id]))
    return out


def verified(src):
    if sha(src['path']) != src['sha256']:
        sys.exit(f"{src['sae_id']}: checkpoint hash mismatch; stopping.")
    return TopK(src['path'], 'cuda')


# ------------------------------------------------------------------------------------------------ probes
def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=0))


def pick_C(X, y, g):
    best = (-1.0, C_GRID[0])
    for C in C_GRID:
        sc = np.zeros(len(y))
        for a, b in GroupKFold(3).split(X, y, g):
            sc[b] = lr(C).fit(X[a], y[a]).decision_function(X[b])
        v = roc_auc_score(y, sc)
        if v > best[0] + 1e-12:
            best = (v, C)
    return best[1]


def mlp(Xtr, ytr, Xte, n, hidden):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    return np.mean([MLPClassifier((hidden,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=s)
                    .fit(a, ytr).predict_proba(b)[:, 1] for s in range(n)], 0)


def readers_orig(Xtr, ytr, gtr, Xte):
    C = pick_C(Xtr, ytr, gtr)
    pipe = lr(C).fit(Xtr, ytr)
    a = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    sc = dict(logistic=pipe.decision_function(Xte), massmean=Xte @ a, mlp10=mlp(Xtr, ytr, Xte, 10, 64),
              mlp256=mlp(Xtr, ytr, Xte, 5, 256))
    return sc, pipe, a, C


def readers_sae(Htr, ytr, gtr, Hte, pipe, a):
    C = pick_C(Htr, ytr, gtr)
    sc = dict(frozen_logistic=pipe.decision_function(Hte), frozen_massmean=Hte @ a,
              refit_logistic=lr(C).fit(Htr, ytr).decision_function(Hte), refit_mlp10=mlp(Htr, ytr, Hte, 10, 64),
              refit_mlp256=mlp(Htr, ytr, Hte, 5, 256))
    return sc, C


ORIG_OF = dict(frozen_logistic='logistic', frozen_massmean='massmean', refit_logistic='logistic',
               refit_mlp10='mlp10', refit_mlp256='mlp256')


# ------------------------------------------------------------------------------------------------ stage: cities
def stage_cities():
    d = EV / 'cities'
    d.mkdir(parents=True, exist_ok=True)
    t = S30.fold_rows()
    X, y, g = t['X'][LAYER], t['y'], t['groups'].astype(str)
    srcs = sae_list()
    for f in S30.FOLDS:
        r = S30.roles(f)
        B, V = t['fold'] == r['B'], t['fold'] == r['V']
        op = d / f'orig_fold{f}.npz'
        if not op.exists():
            sc, pipe, a, C = readers_orig(X[B], y[B], g[B], X[V])
            np.savez(op, y=y[V], groups=g[V], C=C, **sc)
        o = np.load(op)
        pipe = lr(float(o['C'])).fit(X[B], y[B])                  # deterministic refit of the original probe
        a = X[B][y[B] == 1].mean(0) - X[B][y[B] == 0].mean(0)
        assert np.allclose(pipe.decision_function(X[V]), o['logistic'])
        for s in srcs:
            if s['fold'] not in (None, f):
                continue
            p = d / f"fold{f}__{s['sae_id']}.npz"
            if p.exists():
                continue
            t0 = time.time()
            sae = verified(s)
            HB, l0B = sae.np(X[B])
            HV, l0V = sae.np(X[V])
            sc, C = readers_sae(HB, y[B], g[B], HV, pipe, a)
            nmse = float(((X[V] - HV) ** 2).sum() / ((X[V] - X[V].mean(0)) ** 2).sum())
            tmp = p.with_name(p.name + '.tmp.npz')
            np.savez(tmp, refit_C=C, task_nmse=nmse, l0_mean=float(np.r_[l0B, l0V].mean()),
                     l0_min=float(np.r_[l0B, l0V].min()), l0_max=float(np.r_[l0B, l0V].max()), sha256=s['sha256'], **sc)
            tmp.rename(p)
            print(f"[cities] fold{f} {s['sae_id']} refit-gap "
                  f"{roc_auc_score(o['y'], o['logistic']) - roc_auc_score(o['y'], sc['refit_logistic']):+.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ stage: owt
def owt_val():
    seqs, acts = [], []
    for p in sorted((REPO / 'data/activations/gpt2_small/validation').glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        A = c['activations'][f'blocks.{LAYER}.hook_resid_pre']
        for i, n in enumerate(c['lengths']):
            n = int(n)
            seqs.append(c['tokens'][i, :n])
            acts.append(A[i, 1:n].float())                                   # BOS excluded
    return seqs, acts


@torch.no_grad()
def splice_per_seq(model, seqs, sae, orig_lp, bs=16):
    """Per-sequence sums: CE original, CE spliced, KL(orig || spliced), number of predictions (positions >= 1)."""
    hook = f'blocks.{LAYER}.hook_resid_pre'
    out = []
    for b in range(0, len(seqs), bs):
        ss = seqs[b:b + bs]
        Tm = max(len(s) for s in ss)
        toks = torch.full((len(ss), Tm), 50256, dtype=torch.long)
        lens = torch.tensor([len(s) for s in ss])
        for i, s in enumerate(ss):
            toks[i, :len(s)] = s
        toks, lens = toks.cuda(), lens.cuda()
        ar = torch.arange(Tm, device='cuda')[None]
        rep = (ar >= 1) & (ar < lens[:, None])

        def fn(resid, hook):
            resid[rep] = sae(resid[rep].float())[0].to(resid.dtype)
            return resid
        lo = orig_lp[b // bs].cuda()
        ls = model.run_with_hooks(toks, fwd_hooks=[(hook, fn)]).float().log_softmax(-1)
        m = ((ar >= 1) & (ar <= lens[:, None] - 2))[:, :-1]
        tgt = toks[:, 1:, None]
        ce_o = (-lo[:, :-1].gather(-1, tgt)[..., 0] * m).sum(1)
        ce_s = (-ls[:, :-1].gather(-1, tgt)[..., 0] * m).sum(1)
        kl = ((lo[:, :-1].exp() * (lo[:, :-1] - ls[:, :-1])).sum(-1) * m).sum(1)
        out.append(torch.stack([ce_o, ce_s, kl, m.sum(1).float()], 1).cpu().numpy())
    return np.concatenate(out)


def stage_owt():
    from transformer_lens import HookedTransformer
    d = EV / 'owt'
    d.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    seqs, acts = owt_val()
    Xall = torch.cat(acts)
    mu = Xall.mean(0)
    den = np.array([float(((a - mu) ** 2).sum()) for a in acts])
    np.save(d / 'denominators.npy', den)
    model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
    model.eval()
    orig_lp = None
    for s in sae_list():
        p = d / f"{s['sae_id']}.npz"
        if p.exists():
            continue
        sae = verified(s)
        err, l0s = [], []
        for a in acts:
            h, l0 = sae(a.cuda())
            err.append(float(((h.cpu() - a) ** 2).sum()))
            l0s.append(l0.cpu().numpy())
        l0s = np.concatenate(l0s)
        extra = {}
        if s['fold'] in (None, 0):                                       # splicing on fold 0 + bases
            if orig_lp is None:
                orig_lp = []
                with torch.no_grad():
                    for b in range(0, len(seqs), 16):
                        ss = seqs[b:b + 16]
                        Tm = max(len(x) for x in ss)
                        toks = torch.full((len(ss), Tm), 50256, dtype=torch.long)
                        for i, x in enumerate(ss):
                            toks[i, :len(x)] = x
                        orig_lp.append(model(toks.cuda()).float().log_softmax(-1).cpu())
            extra['splice'] = splice_per_seq(model, seqs, sae, orig_lp)
        np.savez(p, err=np.array(err), l0_mean=float(l0s.mean()), l0_min=float(l0s.min()), l0_max=float(l0s.max()),
                 **extra)
        print(f"[owt] {s['sae_id']} nmse {np.sum(err) / den.sum():.4f}", flush=True)


# ------------------------------------------------------------------------------------------------ stage: ladder
def stage_ladder():
    out = EV / 'ladder_fold0.csv'
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite.')
    srcs = [s for s in sae_list() if s['fold'] in (None, 0)]
    n_exp = len(S30.SEEDS) * (1 + len(S30.ARMS))                          # 48 in the full design
    assert len(srcs) == n_exp, f'expected {n_exp} fold-0 SAEs, got {len(srcs)}'
    for s in srcs:
        if sha(s['path']) != s['sha256']:
            sys.exit(f"{s['sae_id']}: hash mismatch")
    LD = mod('scripts/shared/sae_ladder_evaluation.py', 'ladder')
    LD.sources = lambda L, which: [dict(family=s['arm'], name='topk', seed=s['seed'], path=s['path'], arch=None)
                                   for s in srcs]
    sys.argv = ['ladder', '--layer', str(LAYER), '--sources', 'custom', '--out', str(out)]
    LD.main()


# ------------------------------------------------------------------------------------------------ stage: analyze
def boot_weights(groups_by_fold, n=N_BOOT, seed=BOOT_SEED):
    return {f: T.group_bootstrap_counts(g, n, np.random.default_rng(seed + f)) for f, g in groups_by_fold.items()}


def cities_table(subset=None, n_boot=N_BOOT):
    """gap[(sae_key, seed, fold, reader)] = (point, boot (B,)), with sae_key = arm. subset: 'prev_train' keeps the
    523 previously-train-only cities."""
    d = EV / 'cities'
    folds = pd.read_csv(S30.FOLDS_CSV)
    prev = dict(zip(folds.group_id.astype(str), folds.previous_split))
    orig, masks, W = {}, {}, {}
    for f in S30.FOLDS:
        o = dict(np.load(d / f'orig_fold{f}.npz'))
        m = np.ones(len(o['y']), bool) if subset is None else np.array([prev[x] == 'train' for x in o['groups']])
        masks[f] = m
        orig[f] = o
    W = boot_weights({f: orig[f]['groups'][masks[f]] for f in S30.FOLDS}, n_boot,
                     BOOT_SEED + (0 if subset is None else 100))
    oauc = {(f, k): (T.weighted_auroc(orig[f]['y'][masks[f]], orig[f][k][masks[f]]),
                     T.weighted_auroc(orig[f]['y'][masks[f]], orig[f][k][masks[f]], W[f]))
            for f in S30.FOLDS for k in ('logistic', 'massmean', 'mlp10', 'mlp256')}
    gap, meta = {}, []
    for s in sae_list():
        for f in S30.FOLDS:
            if s['fold'] not in (None, f):
                continue
            z = dict(np.load(d / f"fold{f}__{s['sae_id']}.npz"))
            assert str(z['sha256']) == s['sha256']
            y, m = orig[f]['y'][masks[f]], masks[f]
            for rd in READERS:
                pt, bt = oauc[(f, ORIG_OF[rd])]
                gap[(s['arm'], s['seed'], f, rd)] = (pt - T.weighted_auroc(y, z[rd][m]),
                                                     bt - T.weighted_auroc(y, z[rd][m], W[f]))
            meta.append(dict(sae_id=s['sae_id'], arm=s['arm'], seed=s['seed'], fold=f, task_nmse=float(z['task_nmse']),
                             got_l0_mean=float(z['l0_mean']), got_l0_min=float(z['l0_min']),
                             got_l0_max=float(z['l0_max']), refit_C=float(z['refit_C']),
                             **{f'{rd}_gap': gap[(s['arm'], s['seed'], f, rd)][0] for rd in READERS}))
    wts = {f: len(set(orig[f]['groups'][masks[f]])) for f in S30.FOLDS}
    tot = sum(wts.values())
    return gap, {f: w / tot for f, w in wts.items()}, pd.DataFrame(meta)


def pooled(gap, w, arm, rd, seeds, folds=S30.FOLDS, ref=None):
    """Per-seed fold-weighted (arm - ref) gap: points (S,) and boots (S, B)."""
    wf = np.array([w[f] for f in folds])
    wf = wf / wf.sum()
    P, Bt = [], []
    for s in seeds:
        pt = sum(wi * (gap[(arm, s, f, rd)][0] - (gap[(ref, s, f, rd)][0] if ref else 0.0)) for wi, f in zip(wf, folds))
        bt = sum(wi * (gap[(arm, s, f, rd)][1] - (gap[(ref, s, f, rd)][1] if ref else 0.0)) for wi, f in zip(wf, folds))
        P.append(pt)
        Bt.append(bt)
    return np.array(P), np.array(Bt)


def hier_from(P, Bt):
    return T.hierarchical(P, Bt.mean(0))


def owt_tables():
    d = EV / 'owt'
    den = np.load(d / 'denominators.npy')
    W = T.group_bootstrap_counts(np.arange(len(den)), N_BOOT, np.random.default_rng(BOOT_SEED + 7))
    vals, rows = {}, []
    for s in sae_list():
        z = dict(np.load(d / f"{s['sae_id']}.npz"))
        e = z['err']
        nm = (e.sum() / den.sum(), (W @ e) / (W @ den))
        keys = [(s['arm'], s['seed'], f) for f in (S30.FOLDS if s['fold'] is None else [s['fold']])]
        for k in keys:
            vals[k + ('owt_nmse',)] = nm
        r = dict(sae_id=s['sae_id'], arm=s['arm'], seed=s['seed'], fold=s['fold'], owt_nmse=nm[0],
                 owt_l0_mean=float(z['l0_mean']), owt_l0_min=float(z['l0_min']), owt_l0_max=float(z['l0_max']))
        if 'splice' in z:
            sp = z['splice']
            n = sp[:, 3]
            dce = ((sp[:, 1] - sp[:, 0]).sum() / n.sum(), (W @ (sp[:, 1] - sp[:, 0])) / (W @ n))
            kl = (sp[:, 2].sum() / n.sum(), (W @ sp[:, 2]) / (W @ n))
            for k in [(s['arm'], s['seed'], 0)]:
                vals[k + ('splice_delta_ce',)] = dce
                vals[k + ('splice_kl',)] = kl
            r.update(splice_delta_ce=dce[0], splice_kl=kl[0])
        rows.append(r)
    return vals, pd.DataFrame(rows)


def relative(vals, arm, ref, metric, seeds, folds):
    """Per-seed relative change (arm / ref - 1), averaged over folds, with sequence-bootstrap replicates."""
    P, Bt = [], []
    for s in seeds:
        p = np.mean([vals[(arm, s, f, metric)][0] / vals[(ref, s, f, metric)][0] - 1 for f in folds])
        b = np.mean([vals[(arm, s, f, metric)][1] / vals[(ref, s, f, metric)][1] - 1 for f in folds], 0)
        P.append(p)
        Bt.append(b)
    return T.hierarchical(np.array(P), np.array(Bt).mean(0))


def ladder_guard(lad, arm, ref, col, seeds):
    """Per-seed paired difference of a ladder metric (fold 0). Statement uncertainty from the ladder CIs, treated
    conservatively (perfectly correlated across seeds, independent between arms)."""
    a = lad[lad.family == arm].set_index('seed').loc[list(seeds)]
    r = lad[lad.family == ref].set_index('seed').loc[list(seeds)]
    P = (a[col] - r[col]).to_numpy()
    if f'{col}_ci_low' in lad.columns:
        se_a = (a[f'{col}_ci_high'] - a[f'{col}_ci_low']) / (2 * 1.959964)
        se_r = (r[f'{col}_ci_high'] - r[f'{col}_ci_low']) / (2 * 1.959964)
        v = float((se_a ** 2).mean() + (se_r ** 2).mean())
        boot = np.random.default_rng(BOOT_SEED + 9).normal(P.mean(), np.sqrt(v), N_BOOT)
    else:
        boot = None
    return T.hierarchical(P, boot)


def stage_analyze():
    if (AN / 'primary.json').exists():
        sys.exit('analysis exists; refusing to overwrite.')
    AN.mkdir(parents=True, exist_ok=True)
    pr = json.loads((OUT / 'preregistration.json').read_text())
    seeds = list(S30.SEEDS)
    gap, w, per_sae = cities_table()
    n_exp = len(S30.SEEDS) * len(S30.FOLDS) * (1 + len(S30.ARMS))          # 144 in the full design
    assert len(per_sae) == n_exp, f'expected {n_exp} cities evaluations, got {len(per_sae)}'
    per_sae.to_csv(AN / 'per_sae_cities.csv', index=False)
    comps, store = [], {}
    pairs = [('tsp', 'ctrl'), ('tsp', 'shuf'), ('tsp', 'rand'), ('tsp', 'pca'), ('ctrl', 'base'), ('tsp', 'base'),
             ('shuf', 'base'), ('rand', 'base'), ('pca', 'base'), ('shuf', 'ctrl'), ('rand', 'ctrl'), ('pca', 'ctrl')]
    for a, r in pairs:
        for rd in READERS:
            h = hier_from(*pooled(gap, w, a, rd, seeds, ref=r))
            store[(a, r, rd)] = h
            comps.append(dict(arm=a, ref=r, metric=f'{rd}_gap', **{k: v for k, v in h.items() if k != 'per_seed'},
                              per_seed=json.dumps(h['per_seed'])))
    absg = {a: hier_from(*pooled(gap, w, a, 'refit_logistic', seeds)) for a in ('base', 'ctrl', 'tsp', 'shuf', 'rand', 'pca')}
    for a, h in absg.items():
        comps.append(dict(arm=a, ref='(absolute)', metric='refit_logistic_gap', **{k: v for k, v in h.items()
                                                                                  if k != 'per_seed'},
                          per_seed=json.dumps(h['per_seed'])))
    pd.DataFrame(comps).to_csv(AN / 'comparisons_cities.csv', index=False)
    H1, H2, H3 = store[('tsp', 'ctrl', 'refit_logistic')], store[('tsp', 'shuf', 'refit_logistic')], \
        store[('tsp', 'rand', 'refit_logistic')]
    g = T.gatekeeping(H1, H2, H3)
    label = T.h1_label(H1)
    spec = T.specificity_label(g)
    mlp_h = store[('tsp', 'ctrl', 'refit_mlp10')]
    frozen_h = store[('tsp', 'ctrl', 'frozen_logistic')]
    quals = dict(specificity=spec,
                 information=('linear and nonlinear' if mlp_h['ci_high'] < 0 else 'linear readout only'),
                 readout_only=bool(not g['H1']['rejected'] and frozen_h['ci_high'] < 0),
                 distortion_cities=bool(absg['tsp']['ci_high'] < -T.SESOI))
    # sensitivity
    sens = []
    P, Bt = pooled(gap, w, 'tsp', 'refit_logistic', seeds, ref='ctrl')
    lo, hi = T.two_way_bootstrap(Bt, N_TWOWAY, np.random.default_rng(BOOT_SEED + 3))
    sens.append(dict(analysis='two-way bootstrap (seeds x cities)', estimate=P.mean(), ci_low=lo, ci_high=hi))
    sens.append(dict(analysis='pilot rule t(7) on combined se', estimate=H1['estimate'], ci_low=H1['ci_low_pilot_rule'],
                     ci_high=H1['ci_high_pilot_rule'], p=H1['p_pilot_rule_t']))
    for name, ss in (('existing seeds 42/1042/2042', list(S30.OLD_SEEDS)), ('new seeds 3042-7042', list(S30.NEW_SEEDS))):
        h = hier_from(*pooled(gap, w, 'tsp', 'refit_logistic', ss, ref='ctrl'))
        sens.append(dict(analysis=name, estimate=h['estimate'], ci_low=h['ci_low'], ci_high=h['ci_high'], p=h['p']))
    for f in S30.FOLDS:
        h = hier_from(*pooled(gap, w, 'tsp', 'refit_logistic', seeds, folds=[f], ref='ctrl'))
        sens.append(dict(analysis=f'fold {f} only', estimate=h['estimate'], ci_low=h['ci_low'], ci_high=h['ci_high'],
                         p=h['p']))
    gs, ws, _ = cities_table(subset='prev_train')
    for a, r in (('tsp', 'ctrl'), ('tsp', 'shuf'), ('tsp', 'rand')):
        h = hier_from(*pooled(gs, ws, a, 'refit_logistic', seeds, ref=r))
        sens.append(dict(analysis=f'523 previously train-only cities: {a} - {r}', estimate=h['estimate'],
                         ci_low=h['ci_low'], ci_high=h['ci_high'], p=h['p']))
    pd.DataFrame(sens).to_csv(AN / 'sensitivity_cities.csv', index=False)
    # cost guards (tsp vs ctrl) and absolute costs vs base
    vals, owt = owt_tables()
    owt.to_csv(AN / 'per_sae_owt.csv', index=False)
    lad = pd.read_csv(EV / 'ladder_fold0.csv')
    guards, costs = [], []
    for metric, folds, margin in (('owt_nmse', S30.FOLDS, 0.02), ('splice_delta_ce', [0], 0.05), ('splice_kl', [0], 0.05)):
        h = relative(vals, 'tsp', 'ctrl', metric, seeds, folds)
        guards.append(dict(guard=f'{metric} relative (tsp/ctrl - 1)', **T.non_inferiority(h, margin), estimate=h['estimate'],
                           ci_low=h['ci_low'], ci_high=h['ci_high']))
        for a in ('ctrl', 'tsp', 'shuf', 'rand', 'pca'):
            hb = relative(vals, a, 'base', metric, seeds, folds)
            costs.append(dict(arm=a, ref='base', metric=f'{metric} relative', estimate=hb['estimate'], ci_low=hb['ci_low'],
                              ci_high=hb['ci_high'], p=hb['p']))
    hp = ladder_guard(lad, 'tsp', 'ctrl', 'repe_persona__refit_logistic_gap', seeds)
    guards.append(dict(guard='RepE persona re-fit gap (tsp - ctrl)', **T.non_inferiority(hp, 0.02), estimate=hp['estimate'],
                       ci_low=hp['ci_low'], ci_high=hp['ci_high']))
    hm = ladder_guard(lad, 'tsp', 'ctrl', 'm4__preservation_rec', seeds)
    guards.append(dict(guard='M4 T_rec/T_full (tsp - ctrl)', **T.non_inferiority(hm, 0.02, higher_is_worse=False),
                       estimate=hm['estimate'], ci_low=hm['ci_low'], ci_high=hm['ci_high']))
    l0_cols = [c for c in ('repe_persona__task_l0', 'repe_neutral__task_l0', 'rlm__repe_val_l0') if c in lad.columns]
    l0_ok = bool((owt.owt_l0_min == 128).all() and (owt.owt_l0_max == 128).all() and (per_sae.got_l0_min == 128).all()
                 and (per_sae.got_l0_max == 128).all() and all((lad[c] == 128).all() for c in l0_cols))
    guards.append(dict(guard='L0 = 128 exactly on OWT, GoT, RepE', passed=l0_ok, bound=None, margin=None))
    pd.DataFrame(guards).to_csv(AN / 'cost_guards.csv', index=False)
    pd.DataFrame(costs).to_csv(AN / 'absolute_costs_vs_base.csv', index=False)
    trade_off = not all(gd['passed'] for gd in guards)
    primary = dict(
        evidence_level='VALIDATION, confirmatory (pre-registered); cities train+val groups, cross-fitted',
        H1_tsp_vs_ctrl={k: v for k, v in H1.items()}, H2_tsp_vs_shuf=H2, H3_tsp_vs_rand=H3, gatekeeping=g,
        H1_label=label, qualifiers=quals, trade_off=trade_off,
        secondary_cities={'tsp - ctrl refit_mlp10': mlp_h, 'tsp - ctrl refit_mlp256': store[('tsp', 'ctrl', 'refit_mlp256')],
                          'tsp - ctrl frozen_logistic (readout)': frozen_h,
                          'tsp - ctrl frozen_massmean (readout)': store[('tsp', 'ctrl', 'frozen_massmean')],
                          'tsp - pca refit_logistic': store[('tsp', 'pca', 'refit_logistic')],
                          'tsp absolute refit gap': absg['tsp'], 'ctrl absolute refit gap': absg['ctrl'],
                          'base absolute refit gap': absg['base']},
        fold_weights=w, n_cities_evaluations=len(per_sae), test_rows_used=0,
        prereg_sha256=sha(OUT / 'preregistration.json'), script_sha256=sha(__file__),
        stats_sha256=sha(REPO / 'src/evaluation/tspr_stats.py'))
    (AN / 'primary.json').write_text(json.dumps(primary, indent=2, default=float))
    print(json.dumps(dict(H1=H1['estimate'], ci=(H1['ci_low'], H1['ci_high']), p=H1['p'], label=label, gate=g,
                          quals=quals, trade_off=trade_off), indent=1, default=float))


# ------------------------------------------------------------------------------------------------ stage: test
FRESH = {'companies_true_false': dict(role='secondary', C='grouped', group='entity'),
         'common_claim_true_false': dict(role='distortion', C='grouped', group='group_id'),
         'counterfact_true_false': dict(role='exploratory', C=0.1, group='entity')}


def committed(path):
    rel = str(Path(path).relative_to(REPO))
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
    return tracked and clean


def stage_test():
    if not committed(AN / 'primary.json'):
        sys.exit('the cities analysis (eval/analysis/primary.json) must be committed before TEST is read.')
    from transformer_lens import HookedTransformer
    P0 = mod('scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py', 'p0')
    d = EV / 'test'
    d.mkdir(parents=True, exist_ok=True)
    model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
    model.eval()
    srcs = sae_list()
    for fam, spec in FRESH.items():
        raw = pd.read_csv(P0.RAW / P0.FAMILIES[fam][0])
        sp = pd.read_csv(P0.SPLITS / f'{fam}_splits.csv', keep_default_na=False)
        sp['p2_primary'] = sp.p2_primary.astype(str) == 'True'
        dev_all = sp[(sp.split == 'DEV') & (sp.excluded == '')].reset_index(drop=True)
        dacts = torch.load(P0.ACTS / f'{fam}_DEV_final_token.pt', map_location='cpu', weights_only=False)
        assert np.array_equal(dacts['row_ids'].numpy(), dev_all.row_id.to_numpy())
        keep = dev_all.p2_primary.to_numpy()
        dev = dev_all[keep]
        Xd = dacts['activations'][LAYER].numpy()[keep]
        yd = raw.label.to_numpy()[dev.row_id]
        gd = dev[spec['group']].astype(str).to_numpy()
        te = sp[(sp.split == 'TEST') & (sp.excluded == '') & sp.p2_primary].copy()
        te['statement'] = raw.statement.to_numpy()[te.row_id]
        ap = d / f'{fam}_TEST_acts.npz'
        if not ap.exists():
            Xt = extract(model, te.statement.tolist())
            np.savez(ap, X=Xt, row_ids=te.row_id.to_numpy())
        Xt = np.load(ap)['X']
        yt = raw.label.to_numpy()[te.row_id]                              # TEST labels read here only
        gt = te[spec['group']].astype(str).to_numpy()
        C = pick_C(Xd, yd, gd) if spec['C'] == 'grouped' else spec['C']
        pipe = lr(C).fit(Xd, yd)
        o_auc = roc_auc_score(yt, pipe.decision_function(Xt))
        np.savez(d / f'{fam}__orig.npz', y=yt, groups=gt, logistic=pipe.decision_function(Xt), C=C, auroc=o_auc)
        for s in srcs:
            p = d / f"{fam}__{s['sae_id']}.npz"
            if p.exists():
                continue
            sae = verified(s)
            Hd, _ = sae.np(Xd)
            Ht, l0 = sae.np(Xt)
            Cr = pick_C(Hd, yd, gd) if spec['C'] == 'grouped' else spec['C']
            np.savez(p, refit_logistic=lr(Cr).fit(Hd, yd).decision_function(Ht), frozen_logistic=pipe.decision_function(Ht),
                     refit_C=Cr, l0_mean=float(l0.mean()), sha256=s['sha256'])
        print(f'[test] {fam}: TEST rows {len(yt)}, original AUROC {o_auc:.4f}', flush=True)
    (d / 'test_info.json').write_text(json.dumps(dict(read_after_commit_of=str((AN / 'primary.json').relative_to(REPO)),
                                                      head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                                                          capture_output=True, text=True).stdout.strip()),
                                                 indent=2))


@torch.no_grad()
def extract(model, statements, bs=32):
    hook = f'blocks.{LAYER}.hook_resid_pre'
    out = []
    for b in range(0, len(statements), bs):
        seqs = [model.to_tokens(s, prepend_bos=True, move_to_device=False)[0] for s in statements[b:b + bs]]
        lens = torch.tensor([len(s) for s in seqs])
        pad = torch.full((len(seqs), int(lens.max())), model.tokenizer.eos_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :len(s)] = s
        _, cache = model.run_with_cache(pad.cuda(), names_filter=lambda n: n == hook, return_type=None)
        out.append(cache[hook][torch.arange(len(seqs)), (lens - 1).cuda()].float().cpu().numpy())
    return np.concatenate(out)


def test_estimates(fam):
    d = EV / 'test'
    o = dict(np.load(d / f'{fam}__orig.npz'))
    W = T.group_bootstrap_counts(o['groups'], N_BOOT, np.random.default_rng(BOOT_SEED + 50 + list(FRESH).index(fam)))
    ob = T.weighted_auroc(o['y'], o['logistic'], W)
    oa = float(o['auroc'])
    gap = {}
    for s in sae_list():
        z = dict(np.load(d / f"{fam}__{s['sae_id']}.npz"))
        assert str(z['sha256']) == s['sha256']
        for rd in ('refit_logistic', 'frozen_logistic'):
            pt = oa - T.weighted_auroc(o['y'], z[rd])
            bt = ob - T.weighted_auroc(o['y'], z[rd], W)
            for f in (S30.FOLDS if s['fold'] is None else [s['fold']]):
                gap[(s['arm'], s['seed'], f, rd)] = (pt, bt)
    w = {f: 1 / 3 for f in S30.FOLDS}                                     # equal fold weights (same TEST rows)
    return gap, w


def stage_final():
    if not committed(AN / 'primary.json'):
        sys.exit('cities analysis not committed.')
    if (AN / 'final_verdict.json').exists():
        sys.exit('final verdict exists; refusing to overwrite.')
    seeds = list(S30.SEEDS)
    primary = json.loads((AN / 'primary.json').read_text())
    rows, fam_out = [], {}
    for fam, spec in FRESH.items():
        gap, w = test_estimates(fam)
        res = {}
        for a, r in (('tsp', 'ctrl'), ('tsp', 'shuf'), ('tsp', 'rand'), ('tsp', 'pca'), ('ctrl', 'base'), ('tsp', 'base')):
            for rd in ('refit_logistic', 'frozen_logistic'):
                h = hier_from(*pooled(gap, w, a, rd, seeds, ref=r))
                res[f'{a} - {r} {rd}_gap'] = h
                rows.append(dict(family=fam, role=spec['role'], arm=a, ref=r, metric=f'{rd}_gap',
                                 **{k: v for k, v in h.items() if k != 'per_seed'}, per_seed=json.dumps(h['per_seed'])))
        for a in ('base', 'ctrl', 'tsp'):
            h = hier_from(*pooled(gap, w, a, 'refit_logistic', seeds))
            res[f'{a} absolute refit_logistic_gap'] = h
            rows.append(dict(family=fam, role=spec['role'], arm=a, ref='(absolute)', metric='refit_logistic_gap',
                             **{k: v for k, v in h.items() if k != 'per_seed'}, per_seed=json.dumps(h['per_seed'])))
        fam_out[fam] = res
    pd.DataFrame(rows).to_csv(AN / 'test_families.csv', index=False)
    cc = fam_out['common_claim_true_false']
    distortion_cc = bool(cc['tsp - ctrl refit_logistic_gap']['ci_high'] < -T.SESOI
                         or cc['tsp absolute refit_logistic_gap']['ci_high'] < -T.SESOI)
    q = primary['qualifiers']
    distortion = q['distortion_cities'] or distortion_cc
    verdict = dict(
        primary_label=primary['H1_label'], H1=primary['gatekeeping']['H1'], H2=primary['gatekeeping']['H2'],
        H3=primary['gatekeeping']['H3'], specificity=q['specificity'], information=q['information'],
        readout_only=q['readout_only'], trade_off=primary['trade_off'], distortion_cities=q['distortion_cities'],
        distortion_common_claim=distortion_cc, distortion=distortion,
        companies_secondary=fam_out['companies_true_false']['tsp - ctrl refit_logistic_gap'],
        counterfact_exploratory=fam_out['counterfact_true_false']['tsp - ctrl refit_logistic_gap'],
        note='Labels follow preregistration.json; distortion replaces preservation wording if true.')
    (AN / 'final_verdict.json').write_text(json.dumps(verdict, indent=2, default=float))
    print(json.dumps(verdict, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['cities', 'owt', 'ladder', 'analyze', 'test', 'final'])
    {'cities': stage_cities, 'owt': stage_owt, 'ladder': stage_ladder, 'analyze': stage_analyze, 'test': stage_test,
     'final': stage_final}[ap.parse_args().stage]()
