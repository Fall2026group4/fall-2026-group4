#!/usr/bin/env python3
"""TI-2: multi-family truth-subspace-preserving fine-tune (TSP-MF), GPT-2 Small R2 TopK L8 (2026-11-21; VALIDATION).

Protocol: results/geometry_of_truth/tsp_multifamily/{README.md, preregistration.json} (frozen before any training).
Row roles (DEV / TRAIN only; every fresh-family TEST row stays locked; GoT cities TEST is never loaded):
  cities + neg_cities  fold-0 A groups -> S and replay; fold-1 B readers; fold-0 V evaluation (MF-H4)
  larger/smaller_than  GoT TRAIN -> S and replay (eligibility: fit on TRAIN, AUROC on GoT VAL >= 0.70)
  common_claim DEV     60 % of groups -> S and replay; the other 40 % -> MF-H5 guard (eligibility on the 40 %)
  counterfact DEV      60 % of relation groups -> S and replay (eligibility on the other 40 %)
  companies DEV        NEVER in S or replay: primary held-out evaluation (grouped 5-fold CV by entity)
Replay is capped at <= 1,000 rows per family (group-wise sample, fixed seed), so families weigh equally.
Arms (identical tokens / steps / lr / replay rows per seed): ctrl, tsp_cities (S = TSP-R fold-0 cities subspace),
tsp_mf (S = all eligible families, equal family weight, rank 8), rand (TSP-R fold-0 Haar subspace). 8 seeds (TSP-R).
Recipe: scripts/geometry_of_truth/25 run_one, unchanged. Readers: 31 readers_orig / readers_sae, unchanged.
Stages: gate | train | eval | analyze      (outputs refuse to overwrite; ledger-resumable training)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import tspr_stats as T  # noqa: E402
from src.sae import truth_subspace as ts  # noqa: E402

OUT = REPO / 'results/geometry_of_truth/tsp_multifamily'
CK = Path('/mnt/scratch/checkpoints/ti2')
LAYER, NAME, FOLD = 8, 'topk', 0
SEEDS = (42, 1042, 2042, 3042, 4042, 5042, 6042, 7042)
ARMS = ('ctrl', 'tsp_cities', 'tsp_mf', 'rand')
RANK, CAP, SPLIT_SEED, MF_SEED, GATE_AUROC = 8, 1000, 20261122, 20261123, 0.70
R0_TOL = 1e-6          # cross-process reproduction of a TSP-R checkpoint (fixed at freeze; exact value reported)
NFOLD = 5
# Amendment A-TF32 (user-approved 2026-11-22): match TSP-R's numerical configuration exactly. TSP-R trained with
# torch.backends.cuda.matmul.allow_tf32 = True (30_tsp_r_phase1_training.py:216, 25:126) and evaluated with it False
# (31:250). Training paths (gate R0/R1, train) set it True; the eval stage sets it False. Nothing else changes.
TRAIN_TF32, EVAL_TF32 = True, False


def mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S25 = mod('scripts/geometry_of_truth/25_truth_subspace_finetune.py', 's25')
S30 = mod('scripts/geometry_of_truth/30_tsp_r_phase1_training.py', 's30')
S31 = mod('scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py', 's31')
P0 = mod('scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py', 'p0')
FRESH_GROUP = {'companies_true_false': 'entity', 'common_claim_true_false': 'group_id', 'counterfact_true_false': 'entity'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_freeze():
    pr = json.loads((OUT / 'preregistration.json').read_text())
    for rel, h in pr['hashes'].items():
        if sha(REPO / rel) != h:
            raise SystemExit(f'{rel}: hash differs from the frozen pre-registration; refusing to run.')


def refuse(*ps):
    for p in ps:
        if Path(p).exists():
            raise SystemExit(f'{p} exists; refusing to overwrite')


# ------------------------------------------------------------------------------------------------ data / roles
def fresh_dev(fam):
    """DEV p2_primary rows exactly as TSP-R assembled them (31 stage_test), at LAYER. Labels from the raw file."""
    raw = P0.load_family(fam)
    sp = pd.read_csv(P0.SPLITS / f'{fam}_splits.csv', keep_default_na=False)
    sp['p2_primary'] = sp.p2_primary.astype(str) == 'True'
    dev_all = sp[(sp.split == 'DEV') & (sp.excluded == '')].reset_index(drop=True)
    acts = torch.load(P0.ACTS / f'{fam}_DEV_final_token.pt', map_location='cpu', weights_only=False)
    assert np.array_equal(acts['row_ids'].numpy(), dev_all.row_id.to_numpy())
    keep = dev_all.p2_primary.to_numpy()
    dev = dev_all[keep].reset_index(drop=True)
    assert set(dev.split) == {'DEV'}
    return dict(X=acts['activations'][LAYER].numpy()[keep].astype(np.float32), y=raw.label.to_numpy()[dev.row_id],
                groups=dev[FRESH_GROUP[fam]].astype(str).to_numpy(), split_groups=dev.group_id.astype(str).to_numpy(),
                statement=raw.statement.to_numpy()[dev.row_id], row_id=dev.row_id.to_numpy(),
                pol=np.zeros(len(dev), int))


def numeric():
    out = {}
    for pol, name in enumerate(('larger_than', 'smaller_than')):
        c = torch.load(REPO / f'data/got/activations/gpt2_small/{name}_final_token.pt', map_location='cpu', weights_only=False)
        sp = pd.read_csv(REPO / f'data/got/splits/{name}_splits.csv')
        assert np.array_equal(c['row_ids'].numpy(), sp.row_id.to_numpy())
        keep = sp.split.isin(['train', 'val']).to_numpy()            # GoT TEST rows dropped at load, never held
        X = c['activations'][LAYER].numpy()[keep].astype(np.float32)
        del c
        sp = sp[keep]
        out[name] = dict(X=X, y=sp.label.to_numpy().astype(int), groups=(name + ':' + sp.group_id.astype(str)).to_numpy(),
                         split=sp.split.to_numpy(), pol=np.full(len(sp), pol))
        assert 'test' not in set(out[name]['split'])
    cat = {k: np.concatenate([out[n][k] for n in out]) for k in ('X', 'y', 'groups', 'split', 'pol')}
    return cat


def group_split(groups, frac, seed):
    u = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    pick = set(rng.permutation(u)[:int(round(frac * len(u)))])
    return np.array([g in pick for g in groups])


def cap_rows(groups, cap, seed):
    """Group-wise subsample to <= cap rows (whole groups, fixed seed); all rows if already <= cap."""
    if len(groups) <= cap:
        return np.ones(len(groups), bool)
    rng = np.random.default_rng(seed)
    keep, n = np.zeros(len(groups), bool), 0
    for g in rng.permutation(np.array(sorted(set(groups)))):
        idx = np.flatnonzero(groups == g)
        if n + len(idx) > cap:
            continue
        keep[idx], n = True, n + len(idx)
    return keep


def entity_filter(statements, entities):
    """True for rows whose statement mentions a companies entity (whole word, >= 4 chars, case-insensitive)."""
    ents = sorted({e for e in entities if len(e) >= 4}, key=len, reverse=True)
    if not ents:
        return np.zeros(len(statements), bool)
    pat = re.compile(r'\b(' + '|'.join(re.escape(e) for e in ents) + r')\b', re.I)
    return np.array([bool(pat.search(str(s))) for s in statements])


def assemble():
    S30.LAYER, S30.NAME = LAYER, NAME
    t = S30.fold_rows()
    r = S30.roles(FOLD)
    cities = dict(t=t, A=t['fold'] == r['A'], B=t['fold'] == r['B'], V=t['fold'] == r['V'])
    comp = fresh_dev('companies_true_false')
    cc = fresh_dev('common_claim_true_false')
    cf = fresh_dev('counterfact_true_false')
    num = numeric()
    cc_S = group_split(cc['split_groups'], 0.6, SPLIT_SEED)
    cf_S = group_split(cf['split_groups'], 0.6, SPLIT_SEED + 1)
    ents = set(comp['groups'])
    leak = {'common_claim': entity_filter(cc['statement'], ents), 'counterfact': entity_filter(cf['statement'], ents)}
    return dict(cities=cities, companies=comp, common_claim=cc, counterfact=cf, numeric=num, cc_S=cc_S, cf_S=cf_S,
                leak=leak)


def s_families(D, elig):
    """Per eligible family: (X, y, pol, groups) S rows, leakage-filtered and capped."""
    t, c = D['cities']['t'], D['cities']
    fams = {'cities': (t['X'][LAYER][c['A']], t['y'][c['A']], t['pol'][c['A']], t['groups'][c['A']].astype(str))}
    n = D['numeric']
    m = n['split'] == 'train'
    fams['numeric'] = (n['X'][m], n['y'][m], n['pol'][m], n['groups'][m])
    for k, d, S in (('common_claim', D['common_claim'], D['cc_S']), ('counterfact', D['counterfact'], D['cf_S'])):
        m = S & ~D['leak'][k]
        fams[k] = (d['X'][m], d['y'][m], d['pol'][m], d['groups'][m])
    out = {}
    for i, (k, (X, y, pol, g)) in enumerate(fams.items()):
        if k != 'cities' and not elig[k]['eligible']:
            continue
        keep = cap_rows(g, CAP, SPLIT_SEED + 10 + i)
        out[k] = (X[keep], y[keep], pol[keep], g[keep])
    return out


def family_vectors(X, y, pol, groups, seed, n_boot=24, C=0.1):
    """The exact vector set of ts.estimate_truth_subspace (class-mean contrasts per polarity, pooled contrast,
    grouped-bootstrap logistic raw weights); returned instead of orthonormalised."""
    X = np.asarray(X, np.float64)
    vecs = [X[(pol == p) & (y == 1)].mean(0) - X[(pol == p) & (y == 0)].mean(0) for p in np.unique(pol)]
    vecs.append(X[y == 1].mean(0) - X[y == 0].mean(0))
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    for b in range(n_boot):
        pick = rng.choice(u, len(u), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vecs.append(ts.logistic_raw_weight(X[idx], y[idx], C=C, seed=b))
    return [v / (np.linalg.norm(v) + 1e-12) for v in vecs]


def mf_subspace(S):
    """Equal family weight: each family's unit vectors scaled by 1/sqrt(n_vectors) (block Frobenius norm 1)."""
    blocks = []
    for i, (k, (X, y, pol, g)) in enumerate(S.items()):
        v = family_vectors(X, y, pol, g, MF_SEED + i)
        blocks += [x / np.sqrt(len(v)) for x in v]
    V = np.stack(blocks, 1)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, :RANK].astype(np.float64), s[:RANK].tolist()


def eligibility(D):
    """Frozen >= 0.70 rule on rows never used for primary evaluation (original activations, refit logistic)."""
    from sklearn.metrics import roc_auc_score
    res = {}
    n = D['numeric']
    tr, va = n['split'] == 'train', n['split'] == 'val'
    sc, _, _, _ = S31.readers_orig(n['X'][tr], n['y'][tr], n['groups'][tr], n['X'][va])
    res['numeric'] = roc_auc_score(n['y'][va], sc['logistic'])
    for k, d, S in (('common_claim', D['common_claim'], D['cc_S']), ('counterfact', D['counterfact'], D['cf_S'])):
        m = S & ~D['leak'][k]
        sc, _, _, _ = S31.readers_orig(d['X'][m], d['y'][m], d['groups'][m], d['X'][~S])
        res[k] = roc_auc_score(d['y'][~S], sc['logistic'])
    c = D['companies']
    res['companies_cv'] = cv_auroc(c['X'], c['y'], c['groups'])
    return {k: dict(auroc=float(v), eligible=bool(v >= GATE_AUROC)) for k, v in res.items()}


def cv_scores_orig(X, y, g):
    sc = {k: np.zeros(len(y)) for k in ('logistic', 'massmean', 'mlp10', 'mlp256')}
    fits = []
    for a, b in GroupKFold(NFOLD).split(X, y, g):
        s, pipe, av, _ = S31.readers_orig(X[a], y[a], g[a], X[b])
        for k in sc:
            sc[k][b] = s[k]
        fits.append((a, b, pipe, av))
    return sc, fits


def cv_auroc(X, y, g):
    from sklearn.metrics import roc_auc_score
    sc, _ = cv_scores_orig(X, y, g)
    return roc_auc_score(y, sc['logistic'])


def cv_scores_sae(H, y, g, fits):
    sc = {k: np.zeros(len(y)) for k in S31.READERS}
    for a, b, pipe, av in fits:
        s, _ = S31.readers_sae(H[a], y[a], g[a], H[b], pipe, av)
        for k in sc:
            sc[k][b] = s[k]
    return sc


class SAEnp:
    """numpy reconstruction helper (identical to TI-1's 32_tsp_generalisation.SAEnp)."""
    def __init__(self, path):
        from src.evaluation import faithfulness_metrics as fm
        self.m, _ = fm.load_sae(path, 'cuda')
        self.m.eval()

    @torch.no_grad()
    def np(self, X, bs=4096):
        H, L0 = [], []
        for b in range(0, len(X), bs):
            h, z, _ = self.m(torch.as_tensor(X[b:b + bs], device='cuda', dtype=torch.float32))
            H.append(h.float().cpu().numpy())
            L0.append((z > 0).sum(-1).float().cpu().numpy())
        return np.concatenate(H), np.concatenate(L0)


# ------------------------------------------------------------------------------------------------ stages
def gpt2():
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def set_base(seed):
    S25.LAYER, S25.NAME = LAYER, NAME
    S25.R2 = S30.R2 if seed in S30.OLD_SEEDS else S30.NEWBASE / 'checkpoints'


def stage_gate():
    check_freeze()
    refuse(OUT / 'gate.json')
    torch.backends.cuda.matmul.allow_tf32 = TRAIN_TF32
    D = assemble()
    elig = eligibility(D)
    S = s_families(D, elig)
    checks = dict(
        test_rows_loaded=0,
        companies_rows=int(len(D['companies']['y'])), companies_groups=int(len(set(D['companies']['groups']))),
        leakage_dropped={k: int(v.sum()) for k, v in D['leak'].items()},
        companies_in_S_or_replay=0,      # by construction: companies is never passed to s_families
        S_rows={k: int(len(v[1])) for k, v in S.items()}, S_families=list(S),
        cc_guard_rows=int((~D['cc_S']).sum()))
    ok = elig['companies_cv']['eligible'] and len([k for k in S if k != 'cities']) >= 2
    # R0: S25.run_one with the TSP-R fold-0 cities subspace and cities-only replay reproduces TSP-R f0_tsp_s42
    model = gpt2()
    set_base(42)
    P, XA, _ = S30.subspaces_for_fold(D['cities']['t'], FOLD)
    a, _, _ = S25.run_one(model, 42, 'tsp', P['tsp'], XA, S30_chunks(), 'cuda', S25.TOKENS // (S25.SEQ_BATCH * 127))
    ref = torch.load(S30.ckpt_path(0, 'tsp', 42), map_location='cpu', weights_only=False)['model_state_dict']
    diff = max(float((a.state_dict()[k].float() - ref[k].float()).abs().max()) for k in ref)
    # family_vectors equals the frozen estimator's vector set (orthonormalised result identical)
    t, c = D['cities']['t'], D['cities']
    XA_, yA, pA, gA = t['X'][LAYER][c['A']], t['y'][c['A']], t['pol'][c['A']], t['groups'][c['A']]
    s0 = S30.SUB_SEED + LAYER + 1000 * (FOLD + 1)
    Pref, _ = ts.estimate_truth_subspace(XA_, yA, pA, gA, r=RANK, seed=s0)
    Pmine, _ = ts._orthonormal_top(family_vectors(XA_, yA, pA, gA, s0), RANK)
    vec_diff = float(np.abs(np.abs(Pref.T @ Pmine).diagonal() - 1).max())
    res = dict(eligibility=elig, checks=checks, R0_max_abs_param_diff=diff, R0_passed=diff <= R0_TOL,
               family_vectors_max_dev=vec_diff, family_vectors_passed=vec_diff < 1e-8,
               all_passed=bool(ok and diff <= R0_TOL and vec_diff < 1e-8))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if not res['all_passed']:
        raise SystemExit('TI-2 GATE FAILED: stopping (no counted run)')


def S30_chunks():
    return np.load(S25.TOK, mmap_mode='r')[S25.START_CHUNK:]


def subspaces(D):
    g = json.loads((OUT / 'gate.json').read_text())
    S = s_families(D, g['eligibility'])
    Pc, XAc, _ = S30.subspaces_for_fold(D['cities']['t'], FOLD)
    Pmf, sv = mf_subspace(S)
    XA = np.concatenate([v[0] for v in S.values()]).astype(np.float32)
    return dict(tsp_cities=Pc['tsp'], tsp_mf=Pmf, rand=Pc['rand']), XA, dict(mf_singular_values=sv,
                                                                            replay_rows=int(len(XA)))


def stage_train():
    check_freeze()
    torch.backends.cuda.matmul.allow_tf32 = TRAIN_TF32
    if not json.loads((OUT / 'gate.json').read_text())['all_passed']:
        raise SystemExit('gate not passed')
    D = assemble()
    P, XA, meta = subspaces(D)
    sp = OUT / 'subspaces.npz'
    if not sp.exists():
        np.savez(sp, **P)
        (OUT / 'subspace_meta.json').write_text(json.dumps(meta, indent=1))
    model, ch = gpt2(), S30_chunks()
    steps = S25.TOKENS // (S25.SEQ_BATCH * 127)
    led_p = OUT / 'ledger.csv'
    led = pd.read_csv(led_p) if led_p.exists() else pd.DataFrame(columns=['run_id'])
    t0 = time.time()
    for seed in SEEDS:                    # seed-major: an early stop leaves complete paired seed blocks
        set_base(seed)
        for arm in ARMS:
            rid = f'{arm}_s{seed}'
            if rid in set(led.run_id):
                continue
            p = CK / f'{arm}_seed{seed}/checkpoint.pt'
            refuse(p)
            sae, cfg, log = S25.run_one(model, seed, arm, None if arm == 'ctrl' else P[arm], XA, ch, 'cuda', steps)
            cfg.update(procedure='ti2_tsp_multifamily_2026-11-21', replay_rows=int(len(XA)))
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
            row = dict(run_id=rid, arm=arm, seed=seed, checkpoint=str(p), sha256=sha(p),
                       **{f'final_{k}': v for k, v in log[-1].items() if k not in ('seed', 'arm')})
            led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
            led.to_csv(led_p, index=False)
            print(f"{rid}: l0 {log[-1]['l0_owt']:.1f} nmse_owt {log[-1]['nmse_owt']:.4f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)


def srcs():
    out = []
    for s in SEEDS:
        set_base(s)
        out.append(dict(sae_id=f'base_s{s}', arm='base', seed=s, path=S25.R2 / f'layer_{LAYER}/{NAME}_seed{s}/checkpoint_final.pt'))
    for r in pd.read_csv(OUT / 'ledger.csv').itertuples():
        out.append(dict(sae_id=r.run_id, arm=r.arm, seed=int(r.seed), path=Path(r.checkpoint), sha256=r.sha256))
    return out


def stage_eval():
    check_freeze()
    torch.backends.cuda.matmul.allow_tf32 = EVAL_TF32
    E = OUT / 'eval'
    E.mkdir(exist_ok=True)
    D = assemble()
    comp, cc = D['companies'], D['common_claim']
    ccg = ~D['cc_S']
    t, c = D['cities']['t'], D['cities']
    X, y, g = t['X'][LAYER], t['y'], t['groups'].astype(str)
    B, V = c['B'], c['V']
    if not (E / 'orig.npz').exists():
        so, fits_c = cv_scores_orig(comp['X'], comp['y'], comp['groups'])
        sg, fits_g = cv_scores_orig(cc['X'][ccg], cc['y'][ccg], cc['groups'][ccg])
        sv, pipe, a, C = S31.readers_orig(X[B], y[B], g[B], X[V])
        np.savez(E / 'orig.npz', comp_y=comp['y'], comp_g=comp['groups'], cc_y=cc['y'][ccg], cc_g=cc['groups'][ccg],
                 cit_y=y[V], cit_g=g[V], cit_C=C, **{f'comp_{k}': v for k, v in so.items()},
                 **{f'cc_{k}': v for k, v in sg.items()}, **{f'cit_{k}': v for k, v in sv.items()})
    _, fits_c = cv_scores_orig(comp['X'], comp['y'], comp['groups'])
    _, fits_g = cv_scores_orig(cc['X'][ccg], cc['y'][ccg], cc['groups'][ccg])
    o = np.load(E / 'orig.npz')
    pipe = S31.lr(float(o['cit_C'])).fit(X[B], y[B])
    a = X[B][y[B] == 1].mean(0) - X[B][y[B] == 0].mean(0)
    for s in srcs():
        p = E / f"{s['sae_id']}.npz"
        if p.exists():
            continue
        if 'sha256' in s and sha(s['path']) != s['sha256']:
            raise SystemExit(f"{s['sae_id']}: checkpoint hash mismatch")
        sae = SAEnp(s['path'])
        Hc, l0c = sae.np(comp['X'])
        Hg, _ = sae.np(cc['X'][ccg])
        HB, _ = sae.np(X[B])
        HV, _ = sae.np(X[V])
        sc_c = cv_scores_sae(Hc, comp['y'], comp['groups'], fits_c)
        sc_g = cv_scores_sae(Hg, cc['y'][ccg], cc['groups'][ccg], fits_g)
        sc_v, _ = S31.readers_sae(HB, y[B], g[B], HV, pipe, a)
        np.savez(p, comp_l0=float(l0c.mean()), **{f'comp_{k}': v for k, v in sc_c.items()},
                 **{f'cc_{k}': v for k, v in sc_g.items()}, **{f'cit_{k}': v for k, v in sc_v.items()})
        print('eval', s['sae_id'], flush=True)
    # OWT validation cost (identical to TI-1 stage_eval)
    S31.LAYER = LAYER
    seqs, acts = S31.owt_val()
    Xall = torch.cat(acts)
    den = float(((Xall - Xall.mean(0)) ** 2).sum())
    model = gpt2()
    orig_lp = []
    with torch.no_grad():
        for b in range(0, len(seqs), 4):
            ss = seqs[b:b + 4]
            toks = torch.full((len(ss), max(len(q) for q in ss)), 50256, dtype=torch.long)
            for i, q in enumerate(ss):
                toks[i, :len(q)] = q
            orig_lp.append(model(toks.cuda()).float().log_softmax(-1).cpu())
    rows = []
    for s in srcs():
        p = E / f"owt__{s['sae_id']}.json"
        if not p.exists():
            sae = SAEnp(s['path'])
            H, l0 = sae.np(Xall.numpy())
            sp = S31.splice_per_seq(model, seqs, lambda x: (sae.m(x)[0],), orig_lp, bs=4)
            p.write_text(json.dumps(dict(sae_id=s['sae_id'], arm=s['arm'], seed=s['seed'],
                                         owt_nmse=float(((Xall.numpy() - H) ** 2).sum() / den), owt_l0=float(l0.mean()),
                                         owt_dce=float((sp[:, 1] - sp[:, 0]).sum() / sp[:, 3].sum()),
                                         owt_kl=float(sp[:, 2].sum() / sp[:, 3].sum()))))
        rows.append(json.loads(p.read_text()))
    pd.DataFrame(rows).to_csv(E / 'owt_summary.csv', index=False)


def gaps(prefix, n_boot=2000, seed=20261124):
    E = OUT / 'eval'
    o = dict(np.load(E / 'orig.npz'))
    y, g = o[f'{prefix}_y'], o[f'{prefix}_g']
    W = T.group_bootstrap_counts(g, n_boot, np.random.default_rng(seed))
    out = {}
    for s in srcs():
        z = dict(np.load(E / f"{s['sae_id']}.npz"))
        for rd in S31.READERS:
            orig = o[f"{prefix}_{S31.ORIG_OF[rd]}"]
            out[(s['arm'], s['seed'], rd)] = (T.weighted_auroc(y, orig) - T.weighted_auroc(y, z[f'{prefix}_{rd}']),
                                              T.weighted_auroc(y, orig, W) - T.weighted_auroc(y, z[f'{prefix}_{rd}'], W))
    return out


def contrast(G, arm, ref, rd):
    P = np.array([G[(arm, s, rd)][0] - G[(ref, s, rd)][0] for s in SEEDS])
    Bt = np.array([G[(arm, s, rd)][1] - G[(ref, s, rd)][1] for s in SEEDS])
    return T.hierarchical(P, Bt.mean(0))


def stage_analyze():
    check_freeze()
    refuse(OUT / 'analysis')
    Gc, Gv, Gg = gaps('comp'), gaps('cit'), gaps('cc')
    rows = []
    for name, G in (('companies', Gc), ('cities_V', Gv), ('common_claim_guard', Gg)):
        for rd in S31.READERS:
            for arm, ref in (('tsp_mf', 'ctrl'), ('tsp_mf', 'tsp_cities'), ('tsp_mf', 'rand'), ('tsp_cities', 'ctrl'),
                             ('ctrl', 'base'), ('tsp_mf', 'base')):
                h = contrast(G, arm, ref, rd)
                rows.append(dict(family=name, reader=rd, contrast=f'{arm}-{ref}', label=T.h1_label(h), **h))
    R = pd.DataFrame(rows)
    pick = lambda f, c: R[(R.family == f) & (R.reader == 'refit_logistic') & (R.contrast == c)].iloc[0]
    h1 = pick('companies', 'tsp_mf-ctrl')
    mf_h1 = bool(h1.label.startswith('SUPPORTED') and h1.p < 0.05)
    h2, h3 = pick('companies', 'tsp_mf-tsp_cities'), pick('companies', 'tsp_mf-rand')
    holm = T.holm([h2.p, h3.p]) if mf_h1 else [np.nan, np.nan]
    mf_h2 = bool(mf_h1 and h2.ci_high < 0 and holm[0] < 0.05)
    mf_h3 = bool(mf_h1 and h3.ci_high < 0 and holm[1] < 0.05)
    h4 = pick('cities_V', 'tsp_mf-ctrl')
    h5 = pick('common_claim_guard', 'tsp_mf-ctrl')
    mf_h4, mf_h5 = bool(h4.ci_high < 0), bool(h5.ci90_low > -0.010 and h5.ci90_high < 0.010)
    o = pd.read_csv(OUT / 'eval/owt_summary.csv')
    base = o[o.arm == 'base'].set_index('seed')
    t = o[o.arm == 'tsp_mf'].set_index('seed')
    cost = dict(nmse_rel=float(((t.owt_nmse - base.owt_nmse) / base.owt_nmse).mean()),
                dce_rel=float(((t.owt_dce - base.owt_dce) / base.owt_dce).mean()),
                l0_all_128=bool((o[o.arm != 'base'].owt_l0.round(3) == 128.0).all()))
    cost_ok = bool(cost['nmse_rel'] <= 0.03 and cost['dce_rel'] <= 0.05 and cost['l0_all_128'])
    if mf_h1 and mf_h3 and mf_h4 and cost_ok and mf_h5:
        verdict = 'TRANSFERS'
    elif mf_h1 and not mf_h3:
        verdict = 'TRANSFERS (NOT SPECIFIC)'
    elif mf_h1:
        verdict = 'TRADE-OFF'
    else:
        verdict = 'NO TRANSFER DETECTED (MDE about 0.03)'
    summ = dict(MF_H1=mf_h1, MF_H1_estimate=float(h1.estimate), MF_H1_ci=[float(h1.ci_low), float(h1.ci_high)],
                MF_H1_p=float(h1.p), MF_H1_label=h1.label, MF_H2=mf_h2, MF_H3=mf_h3, holm_H2_H3=list(map(float, holm)),
                MF_H4=mf_h4, MF_H4_estimate=float(h4.estimate), MF_H5=mf_h5, MF_H5_ci90=[float(h5.ci90_low), float(h5.ci90_high)],
                cost=cost, cost_ok=cost_ok, verdict=verdict)
    out = OUT / 'analysis'
    out.mkdir()
    R.to_csv(out / 'contrasts.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gate', 'train', 'eval', 'analyze'])
    a = ap.parse_args()
    {'gate': stage_gate, 'train': stage_train, 'eval': stage_eval, 'analyze': stage_analyze}[a.stage]()


if __name__ == '__main__':
    main()
