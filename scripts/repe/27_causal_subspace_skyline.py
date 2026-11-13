#!/usr/bin/env python3
"""R5: supervised causal skyline and random floor for the M4 sites (2026-11-10).

Pre-registered in results/repe/r5_causal_subspace_skyline/README.md. Question: how much of GPT-2's persona-instruction
effect can a rank-k intervention carry at each M4 site, when the subspace is (a) learned to maximise restoration on
TRAIN pairs (DAS-style skyline), (b) the top-k principal directions of the TRAIN differences, (c) the mass-mean
direction (k = 1), or (d) random (floor)? This is compared with the SAE top-k feature patches of M4 (same k, also
selected on TRAIN). All restoration numbers use the M4 definition T(v) = KL(p_H||p_U) - KL(p_H||p_{U + v}), with
v = P_k (x_H - x_U) added at the site, scored on VALIDATION T2-T4 pairs. No test rows exist in RepE-LM.

Run from the repository root: python scripts/repe/27_causal_subspace_skyline.py [--smoke DIR]
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
from src.evaluation import patching as pt  # noqa: E402

spec = importlib.util.spec_from_file_location('s24', REPO / 'scripts/repe/24_sae_patch_decomposition.py')
S24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S24)

OUT = REPO / 'results/repe/r5_causal_subspace_skyline'
M4 = REPO / 'results/repe/sae_patch_decomposition'
SITES = [(1, 'slot'), (8, 'final'), (11, 'final')]
KS = (1, 4, 16, 64)
DAS_SEEDS = (0, 1, 2)
STEPS, LR, BATCH_ROWS, N_RANDOM, SEED, INIT_NOISE = 400, 1e-2, 128, 20, 20261110, 0.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stack(batches, key):
    return torch.cat([b['x']['H'] - b['x']['U'] for b in batches]) if key == 'delta' else None


def restoration_grad(eng, base, pos, finals, lp_t, v):
    n = base.shape[0]
    ar = torch.arange(n, device=base.device)
    res = base.detach().clone()
    res = res.index_put((ar, pos), res[ar, pos] + v)
    cap = {}
    eng.m.run_with_hooks(res, start_at_layer=eng.L, return_type=None,
                         fwd_hooks=[('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x))])
    lq = eng.m.unembed(cap['h'][ar, finals][:, None])[:, 0].double().log_softmax(-1)
    return pt.kl(lp_t.clone(), lq)


def learn_das(eng, tb, k, seed, device, init):
    """Orthonormal U [d, k] (QR of a free matrix) minimising KL(p_H || p_{U + U U^T delta}) on TRAIN pairs.
    Initialised at the TRAIN-PCA basis plus seed-specific Gaussian noise (sd INIT_NOISE / sqrt(d)), so the learned
    subspace starts from the best unsupervised one (smoke test: random init under-fits PCA; README 1.2)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    A = (init.cpu() + INIT_NOISE * torch.randn(768, k, generator=g) / 768 ** 0.5).to(device).requires_grad_(True)
    opt = torch.optim.Adam([A], lr=LR)
    rows = [(bi, j) for bi, b in enumerate(tb) for j in range(len(b['rows']))]
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        bi = int(rng.integers(len(tb)))
        b = tb[bi]
        with torch.enable_grad():
            U, _ = torch.linalg.qr(A)
            d = (b['x']['H'] - b['x']['U']).float()
            v = (d @ U) @ U.T
            loss = restoration_grad(eng, b['resid']['U'], b['pos'], b['finals'], b['lp']['H'], v).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        U, _ = torch.linalg.qr(A)
    return U.detach()


@torch.no_grad()
def score(eng, vb, P_fn):
    """Per-pair restoration T on validation for v = P_fn(delta); returns DataFrame with T_full and T_v."""
    recs = []
    for b in vb:
        d = (b['x']['H'] - b['x']['U']).float()
        k0 = pt.kl(b['lp']['H'], b['lp']['U'])
        lf, lv = eng.patched(b['resid']['U'], b['pos'], b['finals'], [d, P_fn(d)])
        recs.append(pd.DataFrame(dict(statement_id=[r['statement_id'] for r in b['rows']],
                                      template=[r['template'] for r in b['rows']],
                                      T_full=(k0 - pt.kl(b['lp']['H'], lf)).cpu().numpy(),
                                      T_v=(k0 - pt.kl(b['lp']['H'], lv)).cpu().numpy())))
    return pd.concat(recs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    global STEPS, N_RANDOM
    out = a.smoke or OUT
    if a.smoke:
        STEPS, N_RANDOM = 30, 3
    else:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if not r_.endswith('README.md') and sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs; refusing to run.')
    for f in ('per_condition.csv', 'summary.csv', 'checks.json'):
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    device, t0 = 'cuda', time.time()
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained('gpt2')
    tr_rows = [r for r in S24.build_rows(tok, 'train', S24.PRIMARY, ['H', 'U']) if r['template'] in S24.PRIMARY]
    va_rows = [r for r in S24.build_rows(tok, 'validation', S24.PRIMARY, ['H', 'U']) if r['template'] in S24.PRIMARY]
    if a.smoke:
        tr_rows, va_rows = tr_rows[::10], va_rows[::5]
    rows, checks = [], dict(n_train_pairs=len(tr_rows), n_validation_pairs=len(va_rows), test_rows_used=0)
    rng = np.random.default_rng(SEED)
    for L, grp in SITES:
        eng = S24.Engine(model, L)
        tb = S24.clean_batches(eng, tr_rows, ['H', 'U'], grp, device)
        vb = S24.clean_batches(eng, va_rows, ['H', 'U'], grp, device)
        D = torch.cat([(b['x']['H'] - b['x']['U']).float() for b in tb])
        _, _, Vh = torch.linalg.svd(D, full_matrices=False)
        mm = D.mean(0)
        mm = (mm / mm.norm())[:, None]
        conds = [('identity_check', 768, None, lambda d: d)]
        for k in KS:
            Upca = Vh[:k].T
            conds.append(('pca_train_delta', k, None, lambda d, U=Upca: (d @ U) @ U.T))
            for s in DAS_SEEDS:
                U = learn_das(eng, tb, k, SEED + 100 * k + s, device, Upca)
                conds.append(('das_skyline', k, s, lambda d, U=U: (d @ U) @ U.T))
            for i in range(N_RANDOM):
                Q, _ = torch.linalg.qr(torch.as_tensor(rng.standard_normal((768, k)), dtype=torch.float32, device=device))
                conds.append(('random_subspace', k, i, lambda d, U=Q: (d @ U) @ U.T))
        conds.append(('massmean_direction', 1, None, lambda d: (d @ mm) @ mm.T))
        for name, k, s, fn in conds:
            df = score(eng, vb, fn)
            sid = df.statement_id.to_numpy()
            share, lo, hi = S24.ratio_ci(df.T_v.to_numpy(), df.T_full.to_numpy(), sid, np.random.default_rng(SEED))
            rows.append(dict(site=f'blocks.{L}.hook_resid_pre|{grp}', layer=L, condition=name, k=k, seed=s,
                             T_full_mean=float(df.groupby('statement_id').T_full.mean().mean()),
                             T_v_mean=float(df.groupby('statement_id').T_v.mean().mean()),
                             share=share, share_ci_low=lo, share_ci_high=hi))
        idc = [r for r in rows if r['condition'] == 'identity_check' and r['layer'] == L][0]
        checks[f'L{L}_identity_share'] = idc['share']
        print(f'L{L} done ({time.time() - t0:.0f}s)', flush=True)
    per = pd.DataFrame(rows)
    # summary with SAE feature patches (M4, same k, TRAIN-selected) and SAE whole reconstruction
    fs = pd.read_csv(M4 / 'features_summary.csv')
    hy = pd.read_csv(M4 / 'hypotheses.csv')
    summ = []
    for site, g in per.groupby('site'):
        tf = g.T_full_mean.iloc[0]
        f = fs[(fs.site == site) & (fs.template_set == 'T2_T4')]
        for k in KS:
            d = g[g.k == k]
            r = dict(site=site, k=k, T_full=tf,
                     das_share_mean=d[d.condition == 'das_skyline'].share.mean(),
                     das_share_sd=d[d.condition == 'das_skyline'].share.std(),
                     pca_share=d[d.condition == 'pca_train_delta'].share.iloc[0],
                     random_share_mean=d[d.condition == 'random_subspace'].share.mean(),
                     random_share_q95=d[d.condition == 'random_subspace'].share.quantile(0.95))
            for arch in ('relu', 'topk', 'jumprelu'):
                x = f[f.architecture == arch][f'F_attr_k{k}_mean'] / tf
                r[f'sae_{arch}_feature_patch_share_mean'] = x.mean()
                r[f'sae_{arch}_feature_patch_share_sd'] = x.std()
            if k == 1:
                r['massmean_share'] = g[g.condition == 'massmean_direction'].share.iloc[0]
            summ.append(r)
        for arch in ('relu', 'topk', 'jumprelu'):
            x = hy[(hy.site == site) & (hy.architecture == arch)].preservation_rec
            summ.append(dict(site=site, k='SAE full reconstruction', T_full=tf, **{f'sae_{arch}_rec_share_mean': x.mean()}))
    checks.update(runtime_s=time.time() - t0, script_sha256=sha(__file__),
                  identity_ok=bool(all(abs(v - 1) < 1e-6 for kk, v in checks.items() if kk.endswith('identity_share'))))
    per.to_csv(out / 'per_condition.csv', index=False)
    pd.DataFrame(summ).to_csv(out / 'summary.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2, default=float))
    print(json.dumps(checks, indent=1, default=float))
    if not a.smoke:
        print(pd.DataFrame(summ).round(3).to_string())


if __name__ == '__main__':
    main()
