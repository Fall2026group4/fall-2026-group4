#!/usr/bin/env python3
"""TI-1: does the TSP-R truth-subspace fine-tune generalise beyond R2 TopK at layer 8? (2026-11-21, VALIDATION)

Pre-registration: results/geometry_of_truth/tsp_generalisation/preregistration.json (frozen before training).

Cells: L1_topk, L11_topk, L8_batchtopk, L8_relu (R2 bases, seeds 42 / 1042 / 2042). Arms: ctrl (GoT-A replay =
reconstruction-only baseline, same compute), tsp (truth subspace from A), rand (Haar-random subspace, same rank).
The pretrained base is evaluated too. Cities cross-fitting exactly as TSP-R (data/got/fresh/splits/
cities_crossfit_folds.csv): fold f -> V = f (evaluation), B = f+1 (independent readers), A = f+2 (training rows and the
subspace). No cities TEST row is loaded.

Recipe reuse:
  * TopK: scripts/geometry_of_truth/25_truth_subspace_finetune.py `run_one` itself, with its module constants LAYER /
    NAME patched (the TSP-R pattern).
  * BatchTopK / ReLU: `run_one_general` = the same code plus the R2 trainer's own sparsity control
    (scripts/openwebtext/17_streaming_sae_training.py): BatchTopK batch-level top-k in training and the EMA inference
    threshold, started from the base threshold; ReLU decoder-norm-weighted L1 with lambda started at the base's
    lambda_final and the same multiplicative L0 controller (target 128). Gate R0 checks that run_one_general equals
    run_one bit-for-bit for TopK.
  * Subspaces: scripts/geometry_of_truth/30_tsp_r_phase1_training.py `subspaces_for_fold` with LAYER patched.
  * Readers: scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py `readers_orig` / `readers_sae`.
Stages: gate | repro | train --cell C | eval --cell C | analyze
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
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

OUT = REPO / 'results/geometry_of_truth/tsp_generalisation'
CK = OUT / 'checkpoints'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
CELLS = {'L1_topk': (1, 'topk'), 'L11_topk': (11, 'topk'), 'L8_batchtopk': (8, 'batchtopk'), 'L8_relu': (8, 'relu')}
SEEDS, ARMS, FOLDS = (42, 1042, 2042), ('ctrl', 'tsp', 'rand'), (0, 1, 2)
TARGET_L0, CTRL_ETA, LAM_MIN, LAM_MAX = 128, 0.01, 1e-3, 1e3
GATE_AUROC = 0.70


def mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S25 = mod('scripts/geometry_of_truth/25_truth_subspace_finetune.py', 's25')
S30 = mod('scripts/geometry_of_truth/30_tsp_r_phase1_training.py', 's30')
S31 = mod('scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py', 's31')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_layer(layer, name):
    S25.LAYER, S25.NAME, S30.LAYER, S30.NAME = layer, name, layer, name


def gpt2():
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_one_general(model, layer, name, seed, arm, P, XA, chunks, device, steps):
    """S25.run_one generalised to BatchTopK / ReLU with the R2 trainer's sparsity control (identical for TopK)."""
    src = R2 / f'layer_{layer}/{name}_seed{seed}/checkpoint_final.pt'
    sae, ck = fm.load_sae(src, device)
    s = ck['config']['input_scale_folded']
    sae.fold_scale(1.0 / s)
    sae.train()
    torch.manual_seed(seed)
    params = [p for p in sae.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=S25.LR)
    hook = f'blocks.{layer}.hook_resid_pre'
    xa = torch.as_tensor(XA, device=device) * s
    xa_c = xa - xa.mean(0)
    got_den = xa_c.pow(2).sum()
    Pt = None if P is None else torch.as_tensor(P, dtype=torch.float32, device=device)
    sub_den = None if P is None else (xa_c @ Pt).pow(2).sum(-1).mean()
    lam = float(ck['config'].get('lambda_final', 0.0)) if name == 'relu' else 0.0
    bt_ema = float(sae.bt_threshold) if name == 'batchtopk' else None
    log = []
    for step in range(steps):
        toks = torch.as_tensor(chunks[step * S25.SEQ_BATCH:(step + 1) * S25.SEQ_BATCH].astype(np.int64), device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook, stop_at_layer=layer + 1)
            x = cache[hook][:, 1:].float().reshape(-1, xa.shape[1]) * s
        rec, z, _ = sae(x)
        nmse_owt = (rec - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        reca, za, _ = sae(xa)
        nmse_got = (reca - xa).pow(2).sum() / got_den
        loss = nmse_owt + S25.MU * nmse_got
        sub = torch.zeros((), device=device)
        if Pt is not None:
            sub = ts.subspace_loss(xa, reca, Pt, sub_den)
            loss = loss + S25.LAM * sub
        spars = torch.zeros((), device=device)
        if name == 'relu':
            # the R2 trainer's penalty is on the summed squared error scale; here the loss is NMSE, so the penalty is
            # put on the same relative scale: lam * sum_j z_j ||d_j|| / sum ||x - mean||^2 per row
            spars = lam * (z * sae.decoder.weight.norm(dim=0)).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
            loss = loss + spars
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sae.normalize_decoder()
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step} ({arm}, seed {seed}); stopping.')
        with torch.no_grad():
            l0v = float((z > 0).float().sum(-1).mean())
            if name == 'relu':
                lam = min(LAM_MAX, max(LAM_MIN, lam * math.exp(CTRL_ETA * max(-1.0, min(1.0, math.log(max(l0v, 1e-3) / TARGET_L0))))))
            if name == 'batchtopk':
                kept = z[z > 0]
                t = float(kept.min()) if kept.numel() else 0.0
                bt_ema = 0.99 * bt_ema + 0.01 * t
                sae.bt_threshold.fill_(bt_ema)
        if step % 100 == 0 or step == steps - 1:
            log.append(dict(seed=seed, arm=arm, step=step, nmse_owt=float(nmse_owt), nmse_got=float(nmse_got),
                            sub=float(sub), spars=float(spars), lam=lam, l0_owt=l0v,
                            l0_got=float((za > 0).float().sum(-1).mean())))
    sae.eval()
    out = copy.deepcopy(sae).cpu()
    out.fold_scale(s)
    cfg = dict(ck['config'], finetune_arm=arm, finetune_tokens=steps * S25.SEQ_BATCH * 127, finetune_lr=S25.LR,
               mu=S25.MU, lam=(0.0 if P is None else S25.LAM), subspace_rank=(0 if P is None else int(P.shape[1])),
               source_checkpoint=str(src.relative_to(REPO)), source_sha256=sha(src), relu_lambda_end=lam,
               procedure='ti1_tsp_generalisation_2026-11-21')
    return out, cfg, log


def fine_tune(model, layer, name, seed, arm, P, XA, chunks, steps):
    if name == 'topk':
        set_layer(layer, name)
        return S25.run_one(model, seed, arm, P, XA, chunks, 'cuda', steps)
    return run_one_general(model, layer, name, seed, arm, P, XA, chunks, 'cuda', steps)


def chunks():
    return np.load(S25.TOK, mmap_mode='r')[S25.START_CHUNK:]


def steps_full():
    return S25.TOKENS // (S25.SEQ_BATCH * 127)


# ------------------------------------------------------------------------------------------------ stages
def stage_gate():
    """Eligibility (pre-registered): original refit-logistic AUROC on V >= 0.70 in every fold, per layer."""
    out = OUT / 'gate_eligibility.json'
    if out.exists():
        raise SystemExit('gate exists')
    res = {}
    for layer in sorted({l for l, _ in CELLS.values()}):
        set_layer(layer, 'topk')
        t = S30.fold_rows()
        X, y, g = t['X'][layer], t['y'], t['groups'].astype(str)
        aucs = []
        for f in FOLDS:
            r = S30.roles(f)
            B, V = t['fold'] == r['B'], t['fold'] == r['V']
            sc, _, _, C = S31.readers_orig(X[B], y[B], g[B], X[V])
            aucs.append(float(T.weighted_auroc(y[V], sc['logistic'])))
        res[layer] = dict(auroc_by_fold=aucs, eligible=bool(min(aucs) >= GATE_AUROC))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_repro():
    """R0: run_one_general == S25.run_one (TopK, L8, seed 42, fold 0, tsp, 30 steps)."""
    model = gpt2()
    set_layer(8, 'topk')
    t = S30.fold_rows()
    P, XA, _ = S30.subspaces_for_fold(t, 0)
    ch = chunks()
    a, _, _ = S25.run_one(model, 42, 'tsp', P['tsp'], XA, ch, 'cuda', 30)
    b, _, _ = run_one_general(model, 8, 'topk', 42, 'tsp', P['tsp'], XA, ch, 'cuda', 30)
    diff = max(float((a.state_dict()[k].float() - b.state_dict()[k].float()).abs().max()) for k in a.state_dict())
    res = dict(max_abs_param_diff=diff, passed=diff == 0.0)
    (OUT / 'gate_r0_repro.json').write_text(json.dumps(res, indent=1))
    print(res)


def ledger_path(cell):
    return OUT / f'ledger_{cell}.csv'


def stage_train(cell):
    layer, name = CELLS[cell]
    gate = json.loads((OUT / 'gate_eligibility.json').read_text())
    if not gate[str(layer)]['eligible']:
        raise SystemExit(f'{cell}: layer {layer} not eligible by the pre-registered gate')
    model = gpt2()
    set_layer(layer, name)
    t = S30.fold_rows()
    ch = chunks()
    led = pd.read_csv(ledger_path(cell)) if ledger_path(cell).exists() else pd.DataFrame(columns=['run_id'])
    t0 = time.time()
    for f in FOLDS:
        P, XA, meta = S30.subspaces_for_fold(t, f)
        sp = CK / cell / f'subspaces_fold{f}.npz'
        sp.parent.mkdir(parents=True, exist_ok=True)
        if not sp.exists():
            np.savez(sp, **{k: P[k] for k in ('tsp', 'rand')})
        for seed in SEEDS:
            for arm in ARMS:
                rid = f'{cell}_f{f}_{arm}_s{seed}'
                if rid in set(led.run_id):
                    continue
                p = CK / cell / f'fold{f}_{arm}_seed{seed}/checkpoint.pt'
                if p.exists():
                    raise SystemExit(f'{p} exists but is not in the ledger; investigate')
                sae, cfg, log = fine_tune(model, layer, name, seed, arm, None if arm == 'ctrl' else P[arm], XA, ch,
                                          steps_full())
                p.parent.mkdir(parents=True, exist_ok=True)
                cfg.update(fold=f, cell=cell, subspace_meta=meta)
                torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
                row = dict(run_id=rid, cell=cell, fold=f, arm=arm, seed=seed, checkpoint=str(p.relative_to(REPO)),
                           sha256=sha(p), **{f'final_{k}': v for k, v in log[-1].items() if k not in ('seed', 'arm')})
                led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
                led.to_csv(ledger_path(cell), index=False)
                pd.DataFrame(log).to_csv(p.parent / 'monitor.csv', index=False)
                print(f'{rid}: l0 {log[-1]["l0_owt"]:.1f}/{log[-1]["l0_got"]:.1f} nmse_owt {log[-1]["nmse_owt"]:.4f} '
                      f'({(time.time() - t0) / 60:.1f} min)', flush=True)


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


def sources(cell):
    layer, name = CELLS[cell]
    out = [dict(sae_id=f'base_s{s}', arm='base', seed=s, fold=None,
                path=R2 / f'layer_{layer}/{name}_seed{s}/checkpoint_final.pt') for s in SEEDS]
    led = pd.read_csv(ledger_path(cell))
    for r in led.itertuples():
        out.append(dict(sae_id=r.run_id, arm=r.arm, seed=int(r.seed), fold=int(r.fold), path=REPO / r.checkpoint,
                        sha256=r.sha256))
    return out


def stage_eval(cell):
    layer, name = CELLS[cell]
    set_layer(layer, name)
    d = OUT / 'eval' / cell
    d.mkdir(parents=True, exist_ok=True)
    t = S30.fold_rows()
    X, y, g = t['X'][layer], t['y'], t['groups'].astype(str)
    srcs = sources(cell)
    for f in FOLDS:
        r = S30.roles(f)
        B, V = t['fold'] == r['B'], t['fold'] == r['V']
        op = d / f'orig_fold{f}.npz'
        if not op.exists():
            sc, pipe, a, C = S31.readers_orig(X[B], y[B], g[B], X[V])
            np.savez(op, y=y[V], groups=g[V], C=C, **sc)
        o = np.load(op)
        pipe = S31.lr(float(o['C'])).fit(X[B], y[B])
        a = X[B][y[B] == 1].mean(0) - X[B][y[B] == 0].mean(0)
        for s in srcs:
            if s['fold'] not in (None, f):
                continue
            p = d / f"fold{f}__{s['sae_id']}.npz"
            if p.exists():
                continue
            if 'sha256' in s and sha(s['path']) != s['sha256']:
                raise SystemExit(f"{s['sae_id']}: checkpoint hash mismatch")
            sae = SAEnp(s['path'])
            HB, l0B = sae.np(X[B])
            HV, l0V = sae.np(X[V])
            sc, C = S31.readers_sae(HB, y[B], g[B], HV, pipe, a)
            nmse = float(((X[V] - HV) ** 2).sum() / ((X[V] - X[V].mean(0)) ** 2).sum())
            np.savez(p, refit_C=C, task_nmse=nmse, l0_mean=float(np.r_[l0B, l0V].mean()), **sc)
    # OWT validation: NMSE, L0, spliced CE / KL per sequence (all SAEs)
    S31.LAYER = layer
    seqs, acts = S31.owt_val()
    Xall = torch.cat(acts)
    mu = Xall.mean(0)
    den = float(((Xall - mu) ** 2).sum())
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    orig_lp = []
    with torch.no_grad():
        for b in range(0, len(seqs), 16):
            ss = seqs[b:b + 16]
            Tm = max(len(q) for q in ss)
            toks = torch.full((len(ss), Tm), 50256, dtype=torch.long)
            for i, q in enumerate(ss):
                toks[i, :len(q)] = q
            orig_lp.append(model(toks.cuda()).float().log_softmax(-1).cpu())
    rows = []
    for s in srcs:
        p = d / f"owt__{s['sae_id']}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
            continue
        sae = SAEnp(s['path'])
        H, l0 = sae.np(Xall.numpy())
        sp = S31.splice_per_seq(model, seqs, lambda x: (sae.m(x)[0],), orig_lp)
        rr = dict(sae_id=s['sae_id'], arm=s['arm'], seed=s['seed'], fold=s['fold'],
                  owt_nmse=float(((Xall.numpy() - H) ** 2).sum() / den), owt_l0=float(l0.mean()),
                  owt_dce=float((sp[:, 1] - sp[:, 0]).sum() / sp[:, 3].sum()), owt_kl=float(sp[:, 2].sum() / sp[:, 3].sum()))
        p.write_text(json.dumps(rr))
        rows.append(rr)
    pd.DataFrame(rows).to_csv(d / 'owt_summary.csv', index=False)


def cell_table(cell, n_boot=2000, seed=20261121):
    d = OUT / 'eval' / cell
    orig = {f: dict(np.load(d / f'orig_fold{f}.npz')) for f in FOLDS}
    W = {f: T.group_bootstrap_counts(orig[f]['groups'], n_boot, np.random.default_rng(seed + f)) for f in FOLDS}
    gap = {}
    for s in sources(cell):
        for f in FOLDS:
            if s['fold'] not in (None, f):
                continue
            z = dict(np.load(d / f"fold{f}__{s['sae_id']}.npz"))
            y = orig[f]['y']
            for rd in S31.READERS:
                o = orig[f][S31.ORIG_OF[rd]]
                gap[(s['arm'], s['seed'], f, rd)] = (T.weighted_auroc(y, o) - T.weighted_auroc(y, z[rd]),
                                                     T.weighted_auroc(y, o, W[f]) - T.weighted_auroc(y, z[rd], W[f]))
    wts = {f: len(set(orig[f]['groups'])) for f in FOLDS}
    tot = sum(wts.values())
    return gap, {f: w / tot for f, w in wts.items()}


def stage_analyze():
    out = OUT / 'analysis'
    if out.exists():
        raise SystemExit('analysis exists; refusing to overwrite')
    out.mkdir(parents=True)
    rows, h1 = [], []
    for cell in CELLS:
        if not (OUT / 'eval' / cell / 'owt_summary.csv').exists():
            continue
        gap, w = cell_table(cell)
        for rd in S31.READERS:
            for arm, ref in (('tsp', 'ctrl'), ('tsp', 'rand'), ('ctrl', 'base'), ('tsp', 'base'), ('rand', 'ctrl')):
                if ref == 'base':
                    P, Bt = [], []
                    for s in SEEDS:
                        pt = sum(w[f] * (gap[(arm, s, f, rd)][0] - gap[('base', s, f, rd)][0]) for f in FOLDS)
                        bt = sum(w[f] * (gap[(arm, s, f, rd)][1] - gap[('base', s, f, rd)][1]) for f in FOLDS)
                        P.append(pt)
                        Bt.append(bt)
                    P, Bt = np.array(P), np.array(Bt)
                else:
                    P, Bt = S31.pooled(gap, w, arm, rd, SEEDS, FOLDS, ref)
                h = T.hierarchical(P, Bt.mean(0))
                abs_arm = np.mean([sum(w[f] * gap[(arm, s, f, rd)][0] for f in FOLDS) for s in SEEDS])
                rows.append(dict(cell=cell, reader=rd, contrast=f'{arm}-{ref}', abs_gap_arm=abs_arm, **h))
        o = pd.read_csv(OUT / 'eval' / cell / 'owt_summary.csv')
        base = o[o.arm == 'base'].set_index('seed')
        for arm in ARMS:
            a = o[o.arm == arm]
            rel = {m: float(np.mean([(r[m] - base.loc[r.seed, m]) / base.loc[r.seed, m] for r in a.to_dict('records')]))
                   for m in ('owt_nmse', 'owt_dce', 'owt_kl', 'owt_l0')}
            rows.append(dict(cell=cell, reader='owt_cost', contrast=f'{arm}-base(rel)', **rel,
                             l0_within_5pct=bool(all(abs((r['owt_l0'] - base.loc[r['seed'], 'owt_l0']) /
                                                         base.loc[r['seed'], 'owt_l0']) <= 0.05 for r in a.to_dict('records')))))
    df = pd.DataFrame(rows)
    prim = df[(df.reader == 'refit_logistic') & (df.contrast == 'tsp-ctrl')].copy()
    if len(prim):
        prim['p_holm'] = T.holm(prim.p.to_numpy())
        prim['label'] = [T.h1_label(r) for r in prim.to_dict('records')]
    df.to_csv(out / 'contrasts.csv', index=False)
    prim.to_csv(out / 'primary_tsp_vs_ctrl.csv', index=False)
    print(prim[['cell', 'estimate', 'ci_low', 'ci_high', 'p', 'p_holm', 'label', 'seeds_negative']].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gate', 'repro', 'train', 'eval', 'analyze'])
    ap.add_argument('--cell', choices=list(CELLS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True      # as the TSP pilot / TSP-R recipe
    {'gate': stage_gate, 'repro': stage_repro, 'analyze': stage_analyze}.get(a.stage, lambda: None)()
    if a.stage == 'train':
        stage_train(a.cell)
    if a.stage == 'eval':
        stage_eval(a.cell)


if __name__ == '__main__':
    main()
