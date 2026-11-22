#!/usr/bin/env python3
"""TI-4: TSP x KL 2x2 factorial at fixed L0 (GPT-2 Small, R2 TopK L8; 2026-11-21; VALIDATION).

Protocol: results/geometry_of_truth/tsp_kl_factorial/{README.md, preregistration.json} (frozen before any training).
Arms (per seed 42/1042/2042 x cities fold 0/1/2; identical OWT tokens, GoT-A replay rows, steps, lr):
  ctrl   NMSE_OWT + NMSE_GoT-A                        -> the TSP-R checkpoint (reused after SHA-256 + gate R0)
  tsp    ctrl + LAM * truth-subspace term             -> the TSP-R checkpoint (reused after SHA-256 + gate R0)
  kl     ctrl + alpha * KL(p_orig || p_spliced)       -> trained here (R4's KL term and alpha rule)
  tsp_kl ctrl + both                                  -> trained here
Trainer `run_tsp_kl` = scripts/geometry_of_truth/25 run_one with R4's KL term (scripts/openwebtext/18 run_one) added:
  with use_kl=False it executes exactly S25.run_one's computation (gate R0 checks the reused TSP-R checkpoint);
  in pure-R4 mode (mu=0, no subspace, use_kl=True) 20 steps must equal R4's own run_one (gate R1: alpha + params).
alpha = NMSE_OWT / KL on the first batch (R4 rule). TopK keeps L0 = 128 exactly.
Outcomes: cities refit gap (folds, TI-1 readers), OWT VAL spliced dCE / KL / NMSE, M4 T_rec/T_full at the L8 final
site, RepE-LM persona refit gap at L8 (M2 readers).
Stages: gate | train | eval | analyze
"""
from __future__ import annotations

import argparse
import copy
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
from src.evaluation import tspr_stats as T  # noqa: E402
from src.sae import truth_subspace as ts  # noqa: E402

OUT = REPO / 'results/geometry_of_truth/tsp_kl_factorial'
CK = Path('/mnt/scratch/checkpoints/ti4')
LAYER, NAME = 8, 'topk'
SEEDS, FOLDS = (42, 1042, 2042), (0, 1, 2)
ARMS, NEW_ARMS = ('ctrl', 'tsp', 'kl', 'tsp_kl'), ('kl', 'tsp_kl')
TOL, SESOI = 1e-6, 0.010
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
S25.LAYER, S25.NAME, S30.LAYER, S30.NAME, S31.LAYER = LAYER, NAME, LAYER, NAME, LAYER


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


def gpt2():
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def chunks():
    return np.load(S25.TOK, mmap_mode='r')[S25.START_CHUNK:]


def steps_full():
    return S25.TOKENS // (S25.SEQ_BATCH * 127)


def run_tsp_kl(model, seed, P, XA, ch, steps, use_kl, mu=None, seed_torch=True, device='cuda'):
    """S25.run_one + optional R4 KL term. use_kl=False is S25.run_one's computation line for line."""
    mu = S25.MU if mu is None else mu
    src = S25.R2 / f'layer_{LAYER}/{NAME}_seed{seed}/checkpoint_final.pt'
    sae, ck = fm.load_sae(src, device)
    s = ck['config']['input_scale_folded']
    sae.fold_scale(1.0 / s)
    sae.train()
    if seed_torch:
        torch.manual_seed(seed)
    params = [p for p in sae.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=S25.LR)
    hook = f'blocks.{LAYER}.hook_resid_pre'
    xa = None if XA is None else torch.as_tensor(XA, device=device) * s
    if xa is not None:
        xa_c = xa - xa.mean(0)
        got_den = xa_c.pow(2).sum()
    Pt = None if P is None else torch.as_tensor(P, dtype=torch.float32, device=device)
    sub_den = None if P is None else (xa_c @ Pt).pow(2).sum(-1).mean()
    alpha, log = None, []
    for step in range(steps):
        toks = torch.as_tensor(ch[step * S25.SEQ_BATCH:(step + 1) * S25.SEQ_BATCH].astype(np.int64), device=device)
        with torch.no_grad():
            if use_kl:
                logits_o, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook)
                lp_o = logits_o[:, 1:].float().log_softmax(-1)
            else:
                _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook, stop_at_layer=LAYER + 1)
            x = cache[hook][:, 1:].float().reshape(-1, sae.decoder.weight.shape[0]) * s
        rec, z, _ = sae(x)
        nmse_owt = (rec - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        loss = nmse_owt
        nmse_got = torch.zeros((), device=device)
        za = torch.zeros(1, 1, device=device)
        if xa is not None:
            reca, za, _ = sae(xa)
            nmse_got = (reca - xa).pow(2).sum() / got_den
            loss = loss + mu * nmse_got
        sub = torch.zeros((), device=device)
        if Pt is not None:
            sub = ts.subspace_loss(xa, reca, Pt, sub_den)
            loss = loss + S25.LAM * sub
        kl = torch.zeros((), device=device)
        if use_kl:
            resid = cache[hook].clone()
            xr = cache[hook][:, 1:]
            resid = torch.cat([resid[:, :1], (rec / s).view_as(xr).to(resid.dtype)], 1)
            lp_s = model(resid, start_at_layer=LAYER)[:, 1:].float().log_softmax(-1)
            kl = (lp_o.exp() * (lp_o - lp_s)).sum(-1).mean()
            if alpha is None:
                alpha = float(nmse_owt.detach() / kl.detach().clamp_min(1e-8))
            loss = loss + alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sae.normalize_decoder()
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step} (seed {seed}); stopping.')
        if step % 100 == 0 or step == steps - 1:
            log.append(dict(seed=seed, step=step, nmse_owt=float(nmse_owt), nmse_got=float(nmse_got), sub=float(sub),
                            kl=float(kl), alpha=alpha, l0_owt=float((z > 0).float().sum(-1).mean()),
                            l0_got=float((za > 0).float().sum(-1).mean())))
    sae.eval()
    out = copy.deepcopy(sae).cpu()
    out.fold_scale(s)
    cfg = dict(ck['config'], finetune_tokens=steps * S25.SEQ_BATCH * 127, finetune_lr=S25.LR, mu=mu,
               lam=(0.0 if P is None else S25.LAM), subspace_rank=(0 if P is None else int(P.shape[1])),
               kl_alpha=alpha, use_kl=use_kl, source_checkpoint=str(src.relative_to(REPO)), source_sha256=sha(src),
               procedure='ti4_tsp_kl_factorial_2026-11-21')
    return out, cfg, log


def maxdiff(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in b)


def tspr_reuse():
    """(fold, arm, seed) -> TSP-R checkpoint path for ctrl / tsp, SHA-256-verified against the TSP-R ledger."""
    led = pd.read_csv(S30.LEDGER)
    out = {}
    for f in FOLDS:
        for arm in ('ctrl', 'tsp'):
            for s in SEEDS:
                r = led[(led.fold == f) & (led.arm == arm) & (led.seed == s)]
                assert len(r) == 1, (f, arm, s)
                p = REPO / r.checkpoint.iloc[0]
                if sha(p) != r.sha256.iloc[0]:
                    raise SystemExit(f'{p}: TSP-R checkpoint hash mismatch')
                out[(f, arm, s)] = p
    return out


def stage_gate():
    check_freeze()
    refuse(OUT / 'gate.json')
    torch.backends.cuda.matmul.allow_tf32 = TRAIN_TF32
    model, ch = gpt2(), chunks()
    t = S30.fold_rows()
    P, XA, _ = S30.subspaces_for_fold(t, 0)
    res = {}
    # R0: use_kl=False reproduces the reused TSP-R tsp checkpoint (full run) -> ctrl/tsp reuse is valid
    a, _, _ = run_tsp_kl(model, 42, P['tsp'], XA, ch, steps_full(), use_kl=False)
    ref = torch.load(S30.ckpt_path(0, 'tsp', 42), map_location='cpu', weights_only=False)['model_state_dict']
    d0 = maxdiff(a.state_dict(), ref)
    res['R0'] = dict(max_abs_param_diff=d0, passed=d0 <= TOL)
    # R1: pure-R4 mode equals R4's own run_one for 20 steps (alpha and parameters)
    R4 = mod('scripts/openwebtext/18_kl_finetune_fixed_l0.py', 'r4')
    b, cb, _ = R4.run_one(model, LAYER, NAME, 42, 'kl', ch, 'cuda', True)
    c, cc, _ = run_tsp_kl(model, 42, None, None, ch, 20, use_kl=True, mu=0.0, seed_torch=False)
    d1 = maxdiff(c.state_dict(), b.state_dict())
    da = abs(cc['kl_alpha'] - cb['finetune_alpha'])
    res['R1'] = dict(max_abs_param_diff=d1, alpha_abs_diff=da, passed=bool(d1 <= TOL and da <= 1e-5 * abs(cb['finetune_alpha'])))
    try:
        tspr_reuse()
        res['tspr_hashes'] = dict(passed=True)
    except (SystemExit, AssertionError) as e:
        res['tspr_hashes'] = dict(passed=False, error=str(e))
    res['all_passed'] = all(v['passed'] for v in res.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if not res['all_passed']:
        raise SystemExit('TI-4 GATE FAILED: stopping (no counted run)')


def stage_train():
    check_freeze()
    torch.backends.cuda.matmul.allow_tf32 = TRAIN_TF32
    if not json.loads((OUT / 'gate.json').read_text())['all_passed']:
        raise SystemExit('gate not passed')
    model, ch, t = gpt2(), chunks(), S30.fold_rows()
    led_p = OUT / 'ledger.csv'
    led = pd.read_csv(led_p) if led_p.exists() else pd.DataFrame(columns=['run_id'])
    t0 = time.time()
    for s in SEEDS:
        for f in FOLDS:
            P, XA, _ = S30.subspaces_for_fold(t, f)
            for arm in NEW_ARMS:
                rid = f'f{f}_{arm}_s{s}'
                if rid in set(led.run_id):
                    continue
                p = CK / f'fold{f}/{arm}_seed{s}/checkpoint.pt'
                refuse(p)
                sae, cfg, log = run_tsp_kl(model, s, P['tsp'] if arm == 'tsp_kl' else None, XA, ch, steps_full(), True)
                cfg.update(fold=f, arm=arm)
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
                row = dict(run_id=rid, fold=f, arm=arm, seed=s, checkpoint=str(p), sha256=sha(p),
                           **{f'final_{k}': v for k, v in log[-1].items() if k != 'seed'})
                led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
                led.to_csv(led_p, index=False)
                print(f"{rid}: l0 {log[-1]['l0_owt']:.1f} kl {log[-1]['kl']:.4f} alpha {log[-1]['alpha']:.3g} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)


def sources():
    out = [dict(sae_id=f'base_s{s}', arm='base', seed=s, fold=None,
                path=S25.R2 / f'layer_{LAYER}/{NAME}_seed{s}/checkpoint_final.pt') for s in SEEDS]
    for (f, arm, s), p in tspr_reuse().items():
        out.append(dict(sae_id=f'f{f}_{arm}_s{s}', arm=arm, seed=s, fold=f, path=p))
    for r in pd.read_csv(OUT / 'ledger.csv').itertuples():
        out.append(dict(sae_id=r.run_id, arm=r.arm, seed=int(r.seed), fold=int(r.fold), path=Path(r.checkpoint),
                        sha256=r.sha256))
    return out


class SAEnp:
    def __init__(self, path):
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


def stage_eval():
    check_freeze()
    torch.backends.cuda.matmul.allow_tf32 = EVAL_TF32
    E = OUT / 'eval'
    E.mkdir(exist_ok=True)
    t = S30.fold_rows()
    X, y, g = t['X'][LAYER], t['y'], t['groups'].astype(str)
    srcs = sources()
    for s_ in srcs:
        if 'sha256' in s_ and sha(s_['path']) != s_['sha256']:
            raise SystemExit(f"{s_['sae_id']}: checkpoint hash mismatch")
    # cities (TI-1 / TSP-R readers: B -> V per fold)
    for f in FOLDS:
        r = S30.roles(f)
        B, V = t['fold'] == r['B'], t['fold'] == r['V']
        op = E / f'orig_fold{f}.npz'
        if not op.exists():
            sc, _, _, C = S31.readers_orig(X[B], y[B], g[B], X[V])
            np.savez(op, y=y[V], groups=g[V], C=C, **sc)
        o = np.load(op)
        pipe = S31.lr(float(o['C'])).fit(X[B], y[B])
        a = X[B][y[B] == 1].mean(0) - X[B][y[B] == 0].mean(0)
        for s_ in srcs:
            if s_['fold'] not in (None, f):
                continue
            p = E / f"fold{f}__{s_['sae_id']}.npz"
            if p.exists():
                continue
            sae = SAEnp(s_['path'])
            HB, _ = sae.np(X[B])
            HV, _ = sae.np(X[V])
            sc, C = S31.readers_sae(HB, y[B], g[B], HV, pipe, a)
            np.savez(p, refit_C=C, **sc)
    # RepE-LM persona refit gap at L8 (M2 readers, train -> validation) and M4 T_rec / T_full at the L8 final site
    M2 = mod('scripts/shared/m2_four_level_faithfulness.py', 'm2')
    E1 = mod('scripts/shared/e1_task_l0_matched_topk.py', 'e1')
    tasks = {'repe_persona': M2.load_tasks()['repe_persona']}
    origs, bidx = E1.m2_layer_context(M2, tasks, LAYER)
    m4 = E1.M4Ctx('cuda')
    rows = []
    for j, s_ in enumerate(srcs):
        p = E / f"persona__{s_['sae_id']}.json"
        if not p.exists():
            sae, _ = fm.load_sae(s_['path'], 'cuda')
            rr = E1.m2_eval(M2, tasks, LAYER, origs, bidx, sae, 'repe_persona', 700 + j, 'cuda',
                            dict(layer=LAYER, task='repe_persona', source='sae', architecture=s_['arm'], seed=s_['seed'],
                                 pca_k=np.nan, matched=True, live_latents=np.nan))
            pres = m4.preservation(sae, LAYER)
            p.write_text(json.dumps(dict(sae_id=s_['sae_id'], arm=s_['arm'], seed=s_['seed'], fold=s_['fold'],
                                         persona_refit_gap=float(rr['refit_logistic_gap']),
                                         m4_preservation_rec=float(pres['preservation_rec']))))
        rows.append(json.loads(p.read_text()))
    pd.DataFrame(rows).to_csv(E / 'persona_m4_summary.csv', index=False)
    # OWT validation: per-sequence spliced CE / KL (paired over sequences), NMSE, L0
    seqs, acts = S31.owt_val()
    Xall = torch.cat(acts)
    den = float(((Xall - Xall.mean(0)) ** 2).sum())
    model = m4.model
    orig_lp = []
    with torch.no_grad():
        for b in range(0, len(seqs), 4):
            ss = seqs[b:b + 4]
            toks = torch.full((len(ss), max(len(q) for q in ss)), 50256, dtype=torch.long)
            for i, q in enumerate(ss):
                toks[i, :len(q)] = q
            orig_lp.append(model(toks.cuda()).float().log_softmax(-1).cpu())
    orows = []
    for s_ in srcs:
        p = E / f"owt__{s_['sae_id']}.npz"
        if not p.exists():
            sae = SAEnp(s_['path'])
            H, l0 = sae.np(Xall.numpy())
            sp = S31.splice_per_seq(model, seqs, lambda x: (sae.m(x)[0],), orig_lp, bs=4)
            np.savez(p, per_seq=sp, owt_nmse=float(((Xall.numpy() - H) ** 2).sum() / den), owt_l0=float(l0.mean()))
        z = np.load(p)
        sp = z['per_seq']
        orows.append(dict(sae_id=s_['sae_id'], arm=s_['arm'], seed=s_['seed'], fold=s_['fold'],
                          owt_nmse=float(z['owt_nmse']), owt_l0=float(z['owt_l0']),
                          owt_dce=float((sp[:, 1] - sp[:, 0]).sum() / sp[:, 3].sum()),
                          owt_kl=float(sp[:, 2].sum() / sp[:, 3].sum())))
    pd.DataFrame(orows).to_csv(E / 'owt_summary.csv', index=False)


def stage_analyze(n_boot=2000, seed=20261125):
    check_freeze()
    refuse(OUT / 'analysis')
    E = OUT / 'eval'
    srcs = sources()
    orig = {f: dict(np.load(E / f'orig_fold{f}.npz')) for f in FOLDS}
    W = {f: T.group_bootstrap_counts(orig[f]['groups'], n_boot, np.random.default_rng(seed + f)) for f in FOLDS}
    gap = {}
    for s_ in srcs:
        for f in FOLDS:
            if s_['fold'] not in (None, f):
                continue
            z = dict(np.load(E / f"fold{f}__{s_['sae_id']}.npz"))
            for rd in S31.READERS:
                o = orig[f][S31.ORIG_OF[rd]]
                yv = orig[f]['y']
                arm = s_['arm']
                gap[(arm, s_['seed'], f, rd)] = (T.weighted_auroc(yv, o) - T.weighted_auroc(yv, z[rd]),
                                                 T.weighted_auroc(yv, o, W[f]) - T.weighted_auroc(yv, z[rd], W[f]))
    tot = sum(len(set(orig[f]['groups'])) for f in FOLDS)
    w = {f: len(set(orig[f]['groups'])) / tot for f in FOLDS}

    def cities(arm, ref, rd='refit_logistic', ref2=None, arm2=None):
        P, Bt = [], []
        for s in SEEDS:
            pt = sum(w[f] * (gap[(arm, s, f, rd)][0] - gap[(ref, s, f, rd)][0]) for f in FOLDS)
            bt = sum(w[f] * (gap[(arm, s, f, rd)][1] - gap[(ref, s, f, rd)][1]) for f in FOLDS)
            if arm2:
                pt -= sum(w[f] * (gap[(arm2, s, f, rd)][0] - gap[(ref2, s, f, rd)][0]) for f in FOLDS)
                bt -= sum(w[f] * (gap[(arm2, s, f, rd)][1] - gap[(ref2, s, f, rd)][1]) for f in FOLDS)
            P.append(pt)
            Bt.append(bt)
        return T.hierarchical(np.array(P), np.array(Bt).mean(0))
    rows = []
    for rd in S31.READERS:
        for arm, ref in (('tsp_kl', 'kl'), ('tsp', 'ctrl'), ('kl', 'ctrl'), ('tsp_kl', 'ctrl'), ('tsp_kl', 'base')):
            h = cities(arm, ref, rd)
            rows.append(dict(outcome='cities_gap', reader=rd, contrast=f'{arm}-{ref}', label=T.h1_label(h), **h))
    f1 = cities('tsp_kl', 'kl')
    inter = cities('tsp_kl', 'kl', arm2='tsp', ref2='ctrl')
    # behaviour: per-sequence paired dCE (tsp_kl - tsp) over the 9 (seed, fold) pairs
    seq = {(s_['arm'], s_['seed'], s_['fold']): np.load(E / f"owt__{s_['sae_id']}.npz")['per_seq'] for s_ in srcs}
    pairs = [(s, f) for s in SEEDS for f in FOLDS]
    dce = lambda sp, idx=None: ((sp[:, 1] - sp[:, 0]) if idx is None else (sp[idx, 1] - sp[idx, 0])).sum() / \
        (sp[:, 3].sum() if idx is None else sp[idx, 3].sum())
    d_pairs = np.array([dce(seq[('tsp_kl', s, f)]) - dce(seq[('tsp', s, f)]) for s, f in pairs])
    n_seq = seq[('tsp', SEEDS[0], 0)].shape[0]
    rng = np.random.default_rng(seed + 9)
    boots = np.array([np.mean([dce(seq[('tsp_kl', s, f)], i) - dce(seq[('tsp', s, f)], i) for s, f in pairs])
                      for i in rng.integers(0, n_seq, size=(n_boot, n_seq))])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_f2 = min(1.0, 2 * min((boots >= 0).mean(), (boots <= 0).mean()) + 1 / n_boot)
    implrule = bool((d_pairs < 0).all() and abs(d_pairs.mean()) > 2 * d_pairs.std(ddof=1))
    f2 = dict(mean=float(d_pairs.mean()), sd=float(d_pairs.std(ddof=1)), per_pair=d_pairs.round(6).tolist(), ci=[float(lo), float(hi)],
              p=p_f2, imp_l_rule=implrule)
    holm = T.holm([f1['p'], p_f2])
    F_H1 = bool(T.h1_label(f1).startswith('SUPPORTED') and holm[0] < 0.05)
    F_H2 = bool(implrule and hi < 0 and holm[1] < 0.05)
    if inter['ci90_low'] > -SESOI and inter['ci90_high'] < SESOI:
        f3 = 'ADDITIVE'
    elif inter['ci_low'] > SESOI:
        f3 = 'ANTAGONISTIC'
    elif inter['ci_high'] < -SESOI:
        f3 = 'SYNERGISTIC'
    else:
        f3 = 'INCONCLUSIVE'
    o = pd.read_csv(E / 'owt_summary.csv')
    pm = pd.read_csv(E / 'persona_m4_summary.csv')
    base = o[o.arm == 'base'].set_index('seed')
    tk = o[o.arm == 'tsp_kl']
    nmse_rel = float(np.mean([(r.owt_nmse - base.loc[r.seed, 'owt_nmse']) / base.loc[r.seed, 'owt_nmse'] for r in tk.itertuples()]))
    dce_vs_base = float(np.mean([r.owt_dce - base.loc[r.seed, 'owt_dce'] for r in tk.itertuples()]))
    persona = float(pm[pm.arm == 'tsp_kl'].persona_refit_gap.mean() - pm[pm.arm == 'ctrl'].persona_refit_gap.mean())
    l0_ok = bool((o[o.arm != 'base'].owt_l0.round(3) == 128.0).all())
    cost_ok = bool(nmse_rel <= 0.05 and l0_ok and persona <= 0.02)
    # F-H4: tsp_kl better than base on both (cities gap: hierarchical CI high < 0; dCE: lower than base on every run)
    h4c = cities('tsp_kl', 'base')
    dce_lower_all = bool(all(r.owt_dce < base.loc[r.seed, 'owt_dce'] for r in tk.itertuples()))
    F_H4 = bool(h4c['ci_high'] < 0 and dce_lower_all)
    verdict = ('COMBINES' if F_H1 and F_H2 and cost_ok else 'TRUTH-ONLY' if F_H1 and not F_H2 else
               'BEHAVIOUR-ONLY' if F_H2 and not F_H1 else 'CONFLICT')
    if f3 == 'ANTAGONISTIC' and verdict == 'COMBINES':
        verdict = 'COMBINES (ANTAGONISTIC interaction)'
    summ = dict(F_H1=F_H1, F_H1_estimate=f1['estimate'], F_H1_ci=[f1['ci_low'], f1['ci_high']], F_H1_label=T.h1_label(f1),
                F_H2=F_H2, F_H2_detail=f2, holm=list(map(float, holm)), F_H3=f3, interaction=dict(
                    estimate=inter['estimate'], ci=[inter['ci_low'], inter['ci_high']], ci90=[inter['ci90_low'], inter['ci90_high']]),
                F_H4=F_H4, F_H4_cities=dict(estimate=h4c['estimate'], ci_high=h4c['ci_high']), F_H4_dce_lower_all_seeds=dce_lower_all,
                cost=dict(nmse_rel_vs_base=nmse_rel, dce_minus_base=dce_vs_base, persona_gap_vs_ctrl=persona, l0_all_128=l0_ok),
                cost_ok=cost_ok, m4_preservation_by_arm=pm.groupby('arm').m4_preservation_rec.mean().to_dict(),
                verdict=verdict)
    out = OUT / 'analysis'
    out.mkdir()
    pd.DataFrame(rows).to_csv(out / 'cities_contrasts.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gate', 'train', 'eval', 'analyze'])
    a = ap.parse_args()
    {'gate': stage_gate, 'train': stage_train, 'eval': stage_eval, 'analyze': stage_analyze}[a.stage]()


if __name__ == '__main__':
    main()
