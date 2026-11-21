#!/usr/bin/env python3
"""TI-3 (EXPLORATORY pilot): minimal-pair difference preservation, GPT-2 Small R2 TopK L8 (2026-11-21).

Protocol: results/geometry_of_truth/minimal_pair_preservation/{README.md, preregistration.json}. No confirmatory claim.
Minimal pairs: every (city, polarity) cell of GoT cities + neg_cities has exactly one true and one false statement with
the same template, differing only in the country (verified at freeze: 1,270 cells x 2 rows). Pairs come only from the
fold's A groups. No probe direction is fitted, so no evaluation reader can be the training teacher.
Arms (seeds 42/1042/2042 x cities folds 0/1/2; identical tokens, steps, lr, GoT-A replay):
  ctrl     NMSE_OWT + NMSE_GoT-A  -> the TSP-R ctrl checkpoint (reused only if gate R0 reproduces f0_ctrl_s42)
  mp       ctrl + beta * mean_pairs ||(x_t - x_f) - (xh_t - xh_f)||^2 / mean_pairs ||x_t - x_f||^2
  mp_shuf  the same term on mismatched pairs (true of city i, false of city j != i, same polarity)
beta = NMSE_GoT-A / pair term on the first batch (the TSP scale-matching rule), fixed.
Outcomes: cities refit gap (TI-1 readers), companies DEV refit gap (TI-2 CV readers; exploratory transfer),
OWT cost, held-out V pair-difference error. Stages: gate | train | eval | analyze
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

OUT = REPO / 'results/geometry_of_truth/minimal_pair_preservation'
CK = Path('/mnt/scratch/checkpoints/ti3')
LAYER, NAME = 8, 'topk'
SEEDS, FOLDS = (42, 1042, 2042), (0, 1, 2)
NEW_ARMS = ('mp', 'mp_shuf')
TOL, SHUF_SEED = 1e-6, 20261126


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


def pairs(groups, pol, y, shuffled=False, seed=SHUF_SEED):
    """(true_idx, false_idx) per (group, polarity) cell; shuffled: false partner taken from another group, same pol."""
    d = pd.DataFrame(dict(g=np.asarray(groups).astype(str), pol=np.asarray(pol), y=np.asarray(y), i=np.arange(len(y))))
    t_idx, f_idx, pl = [], [], []
    for (g, p), c in d.groupby(['g', 'pol'], sort=True):
        assert len(c) == 2 and set(c.y) == {0, 1}, (g, p)
        t_idx.append(int(c[c.y == 1].i.iloc[0]))
        f_idx.append(int(c[c.y == 0].i.iloc[0]))
        pl.append(p)
    t_idx, f_idx, pl = np.array(t_idx), np.array(f_idx), np.array(pl)
    if shuffled:
        rng = np.random.default_rng(seed)
        f_new = f_idx.copy()
        for p in np.unique(pl):
            m = np.flatnonzero(pl == p)
            perm = m.copy()
            while True:                                  # derangement within polarity: never the own city
                perm = rng.permutation(m)
                if (perm != m).all():
                    break
            f_new[m] = f_idx[perm]
        f_idx = f_new
    return t_idx, f_idx


def run_mp(model, seed, XA, pr, ch, steps, device='cuda'):
    """S25.run_one (ctrl loss) + beta * pair-difference term; pr=None is S25.run_one's ctrl computation."""
    src = S25.R2 / f'layer_{LAYER}/{NAME}_seed{seed}/checkpoint_final.pt'
    sae, ck = fm.load_sae(src, device)
    s = ck['config']['input_scale_folded']
    sae.fold_scale(1.0 / s)
    sae.train()
    torch.manual_seed(seed)
    params = [p for p in sae.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=S25.LR)
    hook = f'blocks.{LAYER}.hook_resid_pre'
    xa = torch.as_tensor(XA, device=device) * s
    xa_c = xa - xa.mean(0)
    got_den = xa_c.pow(2).sum()
    if pr is not None:
        ti, fi = (torch.as_tensor(v, device=device) for v in pr)
        dx = xa[ti] - xa[fi]
        pair_den = dx.pow(2).sum(-1).mean()
    beta, log = None, []
    for step in range(steps):
        toks = torch.as_tensor(ch[step * S25.SEQ_BATCH:(step + 1) * S25.SEQ_BATCH].astype(np.int64), device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook, stop_at_layer=LAYER + 1)
            x = cache[hook][:, 1:].float().reshape(-1, xa.shape[1]) * s
        rec, z, _ = sae(x)
        nmse_owt = (rec - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        reca, za, _ = sae(xa)
        nmse_got = (reca - xa).pow(2).sum() / got_den
        loss = nmse_owt + S25.MU * nmse_got
        pt = torch.zeros((), device=device)
        if pr is not None:
            pt = (dx - (reca[ti] - reca[fi])).pow(2).sum(-1).mean() / pair_den
            if beta is None:
                beta = float(nmse_got.detach() / pt.detach().clamp_min(1e-12))
            loss = loss + beta * pt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sae.normalize_decoder()
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step} (seed {seed}); stopping.')
        if step % 100 == 0 or step == steps - 1:
            log.append(dict(seed=seed, step=step, nmse_owt=float(nmse_owt), nmse_got=float(nmse_got), pair=float(pt),
                            beta=beta, l0_owt=float((z > 0).float().sum(-1).mean()),
                            l0_got=float((za > 0).float().sum(-1).mean())))
    sae.eval()
    out = copy.deepcopy(sae).cpu()
    out.fold_scale(s)
    cfg = dict(ck['config'], finetune_tokens=steps * S25.SEQ_BATCH * 127, finetune_lr=S25.LR, mu=S25.MU, beta=beta,
               n_pairs=(0 if pr is None else int(len(pr[0]))), source_checkpoint=str(src.relative_to(REPO)),
               source_sha256=sha(src), procedure='ti3_minimal_pair_2026-11-21')
    return out, cfg, log


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


def fold_A(t, f):
    A = t['fold'] == S30.roles(f)['A']
    return A, t['X'][LAYER][A], t['groups'][A], t['pol'][A], t['y'][A]


def tspr_ctrl():
    led = pd.read_csv(S30.LEDGER)
    out = {}
    for f in FOLDS:
        for s in SEEDS:
            r = led[(led.fold == f) & (led.arm == 'ctrl') & (led.seed == s)]
            assert len(r) == 1
            p = REPO / r.checkpoint.iloc[0]
            if sha(p) != r.sha256.iloc[0]:
                raise SystemExit(f'{p}: TSP-R ctrl checkpoint hash mismatch')
            out[(f, s)] = p
    return out


def stage_gate():
    check_freeze()
    refuse(OUT / 'gate.json')
    model, ch, t = gpt2(), chunks(), S30.fold_rows()
    _, XA, gA, pA, yA = fold_A(t, 0)
    a, _, _ = run_mp(model, 42, XA, None, ch, steps_full())
    ref = torch.load(S30.ckpt_path(0, 'ctrl', 42), map_location='cpu', weights_only=False)['model_state_dict']
    d0 = max(float((a.state_dict()[k].float() - ref[k].float()).abs().max()) for k in ref)
    ok_pairs = True
    for f in FOLDS:
        _, _, g, p, y = fold_A(t, f)
        ti, fi = pairs(g, p, y)
        ts_, fs_ = pairs(g, p, y, shuffled=True)
        ok_pairs &= bool((np.asarray(g).astype(str)[ti] == np.asarray(g).astype(str)[fi]).all()
                         and (np.asarray(g).astype(str)[ts_] != np.asarray(g).astype(str)[fs_]).all()
                         and (y[ti] == 1).all() and (y[fi] == 0).all())
    try:
        tspr_ctrl()
        hashes_ok = True
    except (SystemExit, AssertionError):
        hashes_ok = False
    res = dict(R0_max_abs_param_diff=d0, R0_passed=d0 <= TOL, pairs_valid=ok_pairs, tspr_ctrl_hashes=hashes_ok,
               all_passed=bool(d0 <= TOL and ok_pairs and hashes_ok))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if not res['all_passed']:
        raise SystemExit('TI-3 GATE FAILED: stopping (no counted run)')


def stage_train():
    check_freeze()
    if not json.loads((OUT / 'gate.json').read_text())['all_passed']:
        raise SystemExit('gate not passed')
    model, ch, t = gpt2(), chunks(), S30.fold_rows()
    led_p = OUT / 'ledger.csv'
    led = pd.read_csv(led_p) if led_p.exists() else pd.DataFrame(columns=['run_id'])
    t0 = time.time()
    for s in SEEDS:
        for f in FOLDS:
            _, XA, g, p, y = fold_A(t, f)
            for arm in NEW_ARMS:
                rid = f'f{f}_{arm}_s{s}'
                if rid in set(led.run_id):
                    continue
                pth = CK / f'fold{f}/{arm}_seed{s}/checkpoint.pt'
                refuse(pth)
                sae, cfg, log = run_mp(model, s, XA, pairs(g, p, y, shuffled=(arm == 'mp_shuf')), ch, steps_full())
                cfg.update(fold=f, arm=arm)
                pth.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), pth)
                pd.DataFrame(log).to_csv(pth.parent / 'monitor.csv', index=False)
                row = dict(run_id=rid, fold=f, arm=arm, seed=s, checkpoint=str(pth), sha256=sha(pth),
                           pair_first=log[0]['pair'], pair_last=log[-1]['pair'],
                           **{f'final_{k}': v for k, v in log[-1].items() if k != 'seed'})
                led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
                led.to_csv(led_p, index=False)
                print(f"{rid}: pair {log[0]['pair']:.4f} -> {log[-1]['pair']:.4f} l0 {log[-1]['l0_owt']:.1f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)


def sources():
    out = [dict(sae_id=f'base_s{s}', arm='base', seed=s, fold=None,
                path=S25.R2 / f'layer_{LAYER}/{NAME}_seed{s}/checkpoint_final.pt') for s in SEEDS]
    for (f, s), p in tspr_ctrl().items():
        out.append(dict(sae_id=f'f{f}_ctrl_s{s}', arm='ctrl', seed=s, fold=f, path=p))
    for r in pd.read_csv(OUT / 'ledger.csv').itertuples():
        out.append(dict(sae_id=r.run_id, arm=r.arm, seed=int(r.seed), fold=int(r.fold), path=Path(r.checkpoint),
                        sha256=r.sha256))
    return out


def stage_eval():
    check_freeze()
    E = OUT / 'eval'
    E.mkdir(exist_ok=True)
    TI4 = mod('scripts/geometry_of_truth/35_tsp_kl_factorial.py', 'ti4')
    TI2 = mod('scripts/geometry_of_truth/34_tsp_multifamily.py', 'ti2')
    t = S30.fold_rows()
    X, y, g = t['X'][LAYER], t['y'], t['groups'].astype(str)
    srcs = sources()
    comp = TI2.fresh_dev('companies_true_false')
    _, fits_c = TI2.cv_scores_orig(comp['X'], comp['y'], comp['groups'])
    if not (E / 'orig_companies.npz').exists():
        so, _ = TI2.cv_scores_orig(comp['X'], comp['y'], comp['groups'])
        np.savez(E / 'orig_companies.npz', y=comp['y'], groups=comp['groups'], **so)
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
        vt, vf = pairs(g[V], t['pol'][V], y[V])
        XV = X[V]
        for s_ in srcs:
            if s_['fold'] not in (None, f):
                continue
            p = E / f"fold{f}__{s_['sae_id']}.npz"
            if p.exists():
                continue
            if 'sha256' in s_ and sha(s_['path']) != s_['sha256']:
                raise SystemExit(f"{s_['sae_id']}: checkpoint hash mismatch")
            sae = TI4.SAEnp(s_['path'])
            HB, _ = sae.np(X[B])
            HV, _ = sae.np(XV)
            sc, C = S31.readers_sae(HB, y[B], g[B], HV, pipe, a)
            dx = XV[vt] - XV[vf]
            perr = float(((dx - (HV[vt] - HV[vf])) ** 2).sum(-1).mean() / (dx ** 2).sum(-1).mean())
            np.savez(p, refit_C=C, pair_rel_err_V=perr, **sc)
    for s_ in srcs:                                         # companies DEV (exploratory transfer)
        p = E / f"companies__{s_['sae_id']}.npz"
        if p.exists():
            continue
        sae = TI4.SAEnp(s_['path'])
        Hc, _ = sae.np(comp['X'])
        np.savez(p, **TI2.cv_scores_sae(Hc, comp['y'], comp['groups'], fits_c))
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
    for s_ in srcs:
        p = E / f"owt__{s_['sae_id']}.json"
        if not p.exists():
            sae = TI4.SAEnp(s_['path'])
            H, l0 = sae.np(Xall.numpy())
            sp = S31.splice_per_seq(model, seqs, lambda x: (sae.m(x)[0],), orig_lp, bs=4)
            p.write_text(json.dumps(dict(sae_id=s_['sae_id'], arm=s_['arm'], seed=s_['seed'], fold=s_['fold'],
                                         owt_nmse=float(((Xall.numpy() - H) ** 2).sum() / den), owt_l0=float(l0.mean()),
                                         owt_dce=float((sp[:, 1] - sp[:, 0]).sum() / sp[:, 3].sum()))))
        rows.append(json.loads(p.read_text()))
    pd.DataFrame(rows).to_csv(E / 'owt_summary.csv', index=False)


def stage_analyze(n_boot=2000, seed=20261127):
    check_freeze()
    refuse(OUT / 'analysis')
    E = OUT / 'eval'
    srcs = sources()
    orig = {f: dict(np.load(E / f'orig_fold{f}.npz')) for f in FOLDS}
    W = {f: T.group_bootstrap_counts(orig[f]['groups'], n_boot, np.random.default_rng(seed + f)) for f in FOLDS}
    gap, perr = {}, {}
    for s_ in srcs:
        for f in FOLDS:
            if s_['fold'] not in (None, f):
                continue
            z = dict(np.load(E / f"fold{f}__{s_['sae_id']}.npz"))
            o, yv = orig[f]['logistic'], orig[f]['y']
            gap[(s_['arm'], s_['seed'], f)] = (T.weighted_auroc(yv, o) - T.weighted_auroc(yv, z['refit_logistic']),
                                               T.weighted_auroc(yv, o, W[f]) - T.weighted_auroc(yv, z['refit_logistic'], W[f]))
            perr[(s_['arm'], s_['seed'], f)] = float(z['pair_rel_err_V'])
    tot = sum(len(set(orig[f]['groups'])) for f in FOLDS)
    w = {f: len(set(orig[f]['groups'])) / tot for f in FOLDS}

    def contrast(arm, ref, folds=FOLDS):
        ww = {f: w[f] / sum(w[q] for q in folds) for f in folds}
        P = [sum(ww[f] * (gap[(arm, s, f)][0] - gap[(ref, s, f)][0]) for f in folds) for s in SEEDS]
        B = [sum(ww[f] * (gap[(arm, s, f)][1] - gap[(ref, s, f)][1]) for f in folds) for s in SEEDS]
        return T.hierarchical(np.array(P), np.array(B).mean(0))
    h_ctrl, h_shuf = contrast('mp', 'ctrl'), contrast('mp', 'mp_shuf')
    per_fold_shuf = {f: contrast('mp', 'mp_shuf', (f,))['estimate'] for f in FOLDS}
    oc = dict(np.load(E / 'orig_companies.npz'))
    Wc = T.group_bootstrap_counts(oc['groups'], n_boot, np.random.default_rng(seed + 9))
    cg = {}
    for s_ in srcs:
        z = dict(np.load(E / f"companies__{s_['sae_id']}.npz"))
        cg.setdefault((s_['arm'], s_['seed']), []).append(
            (T.weighted_auroc(oc['y'], oc['logistic']) - T.weighted_auroc(oc['y'], z['refit_logistic']),
             T.weighted_auroc(oc['y'], oc['logistic'], Wc) - T.weighted_auroc(oc['y'], z['refit_logistic'], Wc)))
    comp = lambda arm, ref: T.hierarchical(
        np.array([np.mean([x[0] for x in cg[(arm, s)]]) - np.mean([x[0] for x in cg[(ref, s)]]) for s in SEEDS]),
        np.array([np.mean([x[1] for x in cg[(arm, s)]], 0) - np.mean([x[1] for x in cg[(ref, s)]], 0) for s in SEEDS]).mean(0))
    o = pd.read_csv(E / 'owt_summary.csv')
    base = o[o.arm == 'base'].set_index('seed')
    mp = o[o.arm == 'mp']
    nmse_rel = float(np.mean([(r.owt_nmse - base.loc[r.seed, 'owt_nmse']) / base.loc[r.seed, 'owt_nmse'] for r in mp.itertuples()]))
    dce_rel = float(np.mean([(r.owt_dce - base.loc[r.seed, 'owt_dce']) / base.loc[r.seed, 'owt_dce'] for r in mp.itertuples()]))
    l0_ok = bool((o[o.arm != 'base'].owt_l0.round(3) == 128.0).all())
    cost_ok = bool(nmse_rel <= 0.03 and dce_rel <= 0.05 and l0_ok)
    led = pd.read_csv(OUT / 'ledger.csv')
    learned = bool((led[led.arm == 'mp'].pair_last < led[led.arm == 'mp'].pair_first).all())
    promising = bool(h_ctrl['ci_high'] < 0 and sum(v < 0 for v in per_fold_shuf.values()) >= 2 and cost_ok and learned)
    summ = dict(evidence='EXPLORATORY pilot (no confirmatory claim)', mp_minus_ctrl=h_ctrl, mp_minus_mp_shuf=h_shuf,
                mp_minus_mp_shuf_per_fold=per_fold_shuf, companies_mp_minus_ctrl=comp('mp', 'ctrl'),
                pair_rel_err_V_by_arm={a: float(np.mean([v for k, v in perr.items() if k[0] == a])) for a in ('base', 'ctrl', 'mp', 'mp_shuf')},
                cost=dict(nmse_rel_vs_base=nmse_rel, dce_rel_vs_base=dce_rel, l0_all_128=l0_ok), cost_ok=cost_ok,
                pair_term_decreased_all_mp_runs=learned,
                pilot_reading='promising' if promising else 'not promising')
    out = OUT / 'analysis'
    out.mkdir()
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gate', 'train', 'eval', 'analyze'])
    a = ap.parse_args()
    {'gate': stage_gate, 'train': stage_train, 'eval': stage_eval, 'analyze': stage_analyze}[a.stage]()


if __name__ == '__main__':
    main()
