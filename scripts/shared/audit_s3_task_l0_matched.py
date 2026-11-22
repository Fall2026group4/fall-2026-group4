#!/usr/bin/env python3
"""AUDIT-S3 (sensitivity, NOT confirmatory): re-evaluate the C1-C3 comparisons and the old-architecture ranking at
MATCHED task-level L0 (inference-time truncation; no retraining).

Problem (audit 2026-11-13; roadmap S3/E1): L0 is matched on OWT only. On the task prompts the arms differ strongly
(e.g. L1 persona: ReLU/JumpReLU/BatchTopK 208-216 vs TopK 128; L8 GoT: JumpReLU 63 vs TopK 128).

Protocol (pre-registered in results/shared/audit_sensitivity/preregistration.json):
  * For each layer L in {1, 8, 11} and task T in {repe_persona, got_cities}: k*_{L,T} = the minimum, over the six arm
    types (old ReLU/TopK/JumpReLU-3072, R2 TopK/BatchTopK/ReLU-12288) and their 3 seeds, of the natural mean L0 on the
    task's TRAIN rows. Every SAE is then truncated to mean L0 = k*: a global magnitude threshold tau on its latents,
    calibrated on the task TRAIN rows (tau = the (k* x n_train)-th largest positive latent value), applied unchanged to
    VALIDATION rows: z' = z * (z > tau), x_hat' = W_dec z' + b_dec. Truncation only removes latents, so no SAE is
    pushed above its natural sparsity. GoT metrics are scored at the informative layers 8 and 11 only (as in R2).
  * Metrics recomputed with unchanged project functions on VALIDATION rows: task NMSE and L0; persona and GoT
    re-trained logistic gap (M2.evaluate_source); persona pair-direction cosine (RLM-RP); M4 preservation T_rec/T_full
    at the L8 final-token site (persona prompts use the persona k*).
  * The R2 rule (scripts/shared/r2_comparisons.py:rule) is re-applied to C1-C3, and the old-family architecture
    ordering is recomputed. A validation class that changes between natural and matched L0 is flagged L0-SENSITIVE.
  * Achieved VALIDATION L0 must be within +-5 % of k*; otherwise the cell is flagged UNMATCHED and excluded.
No TEST row is read. Outputs: results/shared/audit_sensitivity/s3_task_l0_matched/ (refuses to overwrite).
Run from the repository root:  python scripts/shared/audit_s3_task_l0_matched.py [--smoke --out DIR]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

spec = importlib.util.spec_from_file_location('ladder', REPO / 'scripts/shared/sae_ladder_evaluation.py')
LAD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LAD)
M2, S24, S26 = LAD.M2, LAD.S24, LAD.S26
spec = importlib.util.spec_from_file_location('r2c', REPO / 'scripts/shared/r2_comparisons.py')
R2C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2C)

OUT = REPO / 'results/shared/audit_sensitivity/s3_task_l0_matched'
LADDER = REPO / 'results/openwebtext/r2_scaled_saes/ladder'
LAYERS = (1, 8, 11)
ARMS = {('old', 'topk'), ('old', 'relu'), ('old', 'jumprelu'), ('r2', 'topk'), ('r2', 'relu'), ('r2', 'batchtopk')}
COMP = {'C1': (('r2', 'topk'), ('old', 'topk')), 'C2': (('r2', 'relu'), ('old', 'relu')),
        'C3': (('r2', 'batchtopk'), ('old', 'jumprelu'))}
GOT_LAYERS = (8, 11)
TOL = 0.05


class Truncated(nn.Module):
    """Wraps an SAE; keeps only latents above tau. Same (reconstruction, z, pre) interface."""

    def __init__(self, sae, tau):
        super().__init__()
        self.sae, self.tau = sae, float(tau)
        self.decoder = sae.decoder

    def forward(self, x):
        _, z, pre = self.sae(x)
        z2 = z * (z > self.tau).to(z.dtype)
        rec = F.linear(z2, self.sae.decoder.weight) + self.sae.output_bias
        return rec, z2, pre


def natural_train_l0(sae, X, device):
    _, Z = LAD.encode_np(sae, X, device)
    return float((Z > 0).sum(1).mean()), Z


def calibrate_tau(Ztr, k):
    pos = Ztr[Ztr > 0]
    n_keep = int(round(k * Ztr.shape[0]))
    if n_keep >= len(pos):
        return 0.0
    return float(np.partition(pos, len(pos) - n_keep - 1)[len(pos) - n_keep - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.mkdir(parents=True)
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained('gpt2')
    tasks = M2.load_tasks()
    d, idx, y, y_hhp, g, sid, neardup, _ = S26.load_rows()
    S26.G.update(idx=idx, y=y, y_hhp=y_hhp, g=g, sid=sid, neardup=neardup)
    lad = pd.concat([pd.read_csv(p) for p in sorted(LADDER.glob('ladder_layer*.csv'))], ignore_index=True)
    rows, kstar_rows = [], []
    layers = (8,) if a.smoke else LAYERS
    for L in layers:
        rng = np.random.default_rng(M2.SEED + L)
        origs = {tn: M2.original_side(t, L, rng) for tn, t in tasks.items()}
        bidx = {tn: il.bootstrap_indices(t['groups'][t['split'] == 'validation'], M2.N_BOOT,
                                         np.random.default_rng(M2.SEED + 100 + L)) for tn, t in tasks.items()}
        Xr = np.load(S26.CACHE / f'final_layer{L}.npy').astype(np.float32)
        S26.G['X'] = Xr
        S26.G['orig'] = S26.original_side(Xr)
        srcs = [s for s in LAD.sources(L, ['old', 'r2']) if (s['family'], s['name']) in ARMS
                and (not a.smoke or s['seed'] == 42)]
        task_list = ['repe_persona'] + (['got_cities'] if L in GOT_LAYERS else [])
        # pass 1: natural TRAIN L0 per SAE and task
        nat = {}
        for src in srcs:
            sae = LAD.load(src, L, device)
            for tn in task_list:
                t = tasks[tn]
                nat[(tn, src['family'], src['name'], src['seed'])] = natural_train_l0(sae, t['X'][L][t['split'] == 'train'], device)[0]
            del sae
        kstar = {tn: min(v for k, v in nat.items() if k[0] == tn) for tn in task_list}
        for tn, k in kstar.items():
            kstar_rows.append(dict(layer=L, task=tn, k_star=k, argmin=min((v, k_) for k_, v in nat.items() if k_[0] == tn)[1][1:]))
        if L == 8:
            rows4 = S24.build_rows(tok, 'validation', S24.PERSONA, ['H', 'U', 'Hp'])
            eng = S24.Engine(model, L)
            vb = S24.clean_batches(eng, rows4, ['H', 'U', 'Hp'], 'final', device)
        # pass 2: matched evaluation
        for i, src in enumerate(srcs):
            base = LAD.load(src, L, device)
            key = (src['family'], src['name'], src['seed'])
            r = dict(layer=L, family=key[0], name=key[1], seed=key[2])
            for tn in task_list:
                t = tasks[tn]
                tr = t['split'] == 'train'
                _, Ztr = natural_train_l0(base, t['X'][L][tr], device)
                tau = calibrate_tau(Ztr, kstar[tn])
                sae = Truncated(base, tau).eval()
                H, Z = LAD.encode_np(sae, t['X'][L], device)
                rs, _ = M2.evaluate_source(dict(), t['X'][L], H, Z, t['y'], t['split'], t['groups'], origs[tn], bidx[tn],
                                           LAD.SEED + 1000 * L + i)
                va = t['split'] == 'validation'
                r.update({f'{tn}__k_star': kstar[tn], f'{tn}__tau': tau, f'{tn}__natural_train_l0': nat[(tn,) + key],
                          f'{tn}__matched_val_l0': float((Z[va] > 0).sum(1).mean()),
                          f'{tn}__task_nmse': rs['task_nmse'], f'{tn}__refit_logistic_gap': rs['refit_logistic_gap'],
                          f'{tn}__frozen_logistic_gap': rs['frozen_logistic_gap']})
                r[f'{tn}__matched'] = bool(abs(r[f'{tn}__matched_val_l0'] - kstar[tn]) / kstar[tn] <= TOL)
                if tn == 'repe_persona':
                    Hr, Zr = LAD.encode_np(sae, Xr, device)
                    rr = S26.evaluate({}, Hr, None, (Zr > 0).sum(1), float(((Zr[idx['all_va']] > 0).any(0) == 0).mean()))
                    r['rlm__pair_dir_cosine'] = rr['pair_dir_cosine']
                    if L == 8:
                        torch.manual_seed(LAD.SEED)
                        S24.GEN_CPU.manual_seed(LAD.SEED)
                        gen = torch.Generator(device=device).manual_seed(LAD.SEED)
                        dfp, dr, _ = S24.decompose(eng, sae, vb, 'persona', gen)
                        dfp = S24.add_derived(dfp)
                        s = [x for x in S24.summarise_decomp(dfp, dr, dict(kind='persona'), np.random.default_rng(LAD.SEED))
                             if x['template_set'] == 'T2_T4'][0]
                        r['m4__preservation_rec'] = s['preservation_rec']
            rows.append(r)
            print(f"L{L} {key}: " + ', '.join(f"{tn} k*={kstar[tn]:.1f} val L0={r[f'{tn}__matched_val_l0']:.1f}" for tn in task_list)
                  + f' ({time.time() - t0:.0f}s)', flush=True)
            del base
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(a.out / 'per_sae_matched.csv', index=False)
    pd.DataFrame(kstar_rows).to_csv(a.out / 'k_star.csv', index=False)
    if a.smoke:
        return
    # ---------------- comparisons: natural (committed ladder) vs matched
    metrics = [('repe_persona__refit_logistic_gap', False), ('repe_persona__frozen_logistic_gap', False),
               ('repe_persona__task_nmse', False), ('rlm__pair_dir_cosine', True), ('m4__preservation_rec', True),
               ('got_cities__refit_logistic_gap', False), ('got_cities__frozen_logistic_gap', False),
               ('got_cities__task_nmse', False)]
    comp = []
    for L in LAYERS:
        dm, dn = df[df.layer == L], lad[lad.layer == L]
        for cid, (new, ref) in COMP.items():
            for m, hib in metrics:
                if m not in dm or dm[m].isna().all():
                    continue
                task = m.split('__')[0]
                sel = lambda x, arm: x[(x.family == arm[0]) & (x.name == arm[1])].sort_values('seed')
                mn, mr = sel(dm, new), sel(dm, ref)
                ok = True
                if f'{task}__matched' in dm:
                    ok = bool(mn[f'{task}__matched'].all() and mr[f'{task}__matched'].all())
                rm = R2C.rule(mn[m], mr[m], hib)
                rn = R2C.rule(sel(dn, new)[m], sel(dn, ref)[m], hib) if m in dn else dict(cls='NOT IN LADDER')
                comp.append(dict(layer=L, comparison=cid, metric=m, matched_ok=ok,
                                 natural_cls=rn['cls'], natural_diff=rn.get('diff'), matched_cls=rm['cls'],
                                 matched_diff=rm.get('diff'), matched_ci_low=rm.get('ci95_low'),
                                 matched_ci_high=rm.get('ci95_high'), matched_pairs_better=rm.get('pairs_better'),
                                 l0_sensitive=bool(ok and rn['cls'] != rm['cls'] and rn['cls'] != 'NOT IN LADDER')))
    cp = pd.DataFrame(comp)
    cp.to_csv(a.out / 'comparisons_natural_vs_matched.csv', index=False)
    # old-family architecture ordering (mean over seeds)
    rank = []
    for L in LAYERS:
        for m, hib in metrics:
            for src_name, frame in (('natural', lad), ('matched', df)):
                x = frame[(frame.layer == L) & (frame.family == 'old')]
                if m not in x or x[m].isna().all():
                    continue
                mean = x.groupby('name')[m].mean().sort_values(ascending=not hib)
                rank.append(dict(layer=L, metric=m, l0=src_name, order=' > '.join(mean.index),
                                 **{f'mean_{k}': v for k, v in mean.items()}))
    rk = pd.DataFrame(rank)
    rk.to_csv(a.out / 'old_architecture_ordering.csv', index=False)
    info = dict(test_rows_used=0, n_saes=len(df), n_comparisons=len(cp), n_l0_sensitive=int(cp.l0_sensitive.sum()),
                n_unmatched=int((~cp.matched_ok).sum()), runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
