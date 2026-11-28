#!/usr/bin/env python3
"""M4-G: SAE causal-faithfulness decomposition on Gemma-2-2B at the XC-1 sites (port of scripts/repe/24, M4). 2026-11-23.

Pre-registered in results/cross_model/m4_gemma_patch_decomposition/preregistration.json, frozen BEFORE the XC-1 full run
reads any VALIDATION pair. Hashes checked before the full run; refuses on mismatch or existing outputs.

Sites (rule fixed in advance):
  fixed  every XC-1 selected cell that is CONFIRMED on validation (results/cross_model/xc1_gemma_site_sweep/
         selected_sites.json); if none is confirmed, M4-G does not run (XC-1 stop rule).
  mid    amendment A1 ported: among XC-1 TRAIN-passing cells at layers {12, 16} (depth 0.50 / 0.65; GPT-2 used 6-8 =
         0.50-0.67) in groups {slot, final}, E_share closest to 0.5 (ties -> larger R_A); confirmed on VALIDATION T2-T4
         with original activations only (T_full CI low > 0 and T_full - S_full CI low > 0); else the next candidate;
         none -> fixed sites only. Runs before any SAE is loaded.
SAE      one Gemma Scope 16k canonical JumpReLU per site layer (no seeds).
Per pair (validation; persona T1-T4 and neutral N1-N2), at the site positions:
  Delta = x_H - x_U, Delta_rec = SAE(x_H) - SAE(x_U), Delta_err = Delta - Delta_rec;
  T(v) = KL(p_H||p_U) - KL(p_H||p_{U + v}); T_full, T_rec, T_err, I, Shapley phi_rec/phi_err, linearity T(Delta_rec/2);
  other direction (H - v), synonym (Hp toward H), neutral (Nb toward Na);
  controls: 20 isotropic matched-norm draws for Delta_rec and Delta_err, 20 decoder-span shuffles (as M4).
Hypotheses (PRIMARY = T2-T4; BH over sites):
  H4a sufficiency: max(p[T_rec - iso_rec], p[T_rec - shuf_rec]) BH q < 0.05 AND T_rec CI low > 0 AND
       mean T_rec > q95 of both control distributions.
  H4b rec vs err: phi_rec - phi_err > 0 (bootstrap p, BH).
  H4c (feature patches) is NOT run (declared omission: needs gradient attribution on TRAIN + OWT frequency stats).
  claim rule: one SAE per site, so a claim per site = that SAE passing; additive reading only if |I|/T_full CI high
  < 0.2, otherwise only the sufficiency (T_rec) and error (T_err) bounds are interpreted.
Run from the repository root: ../venv/bin/python scripts/cross_model/25_m4_gemma_patch_decomposition.py
  [--smoke --smoke-out D --smoke-sites L:group,...]   (smoke: TRAIN rows only, given sites)
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import patching as pt  # noqa: E402

OUT = REPO / 'results/cross_model/m4_gemma_patch_decomposition'
XC1 = REPO / 'results/cross_model/xc1_gemma_site_sweep'
DATA = REPO / 'data/repe/repe_lm'
PERSONA, NEUTRAL, PRIMARY = ['T1', 'T2', 'T3', 'T4'], ['N1', 'N2'], ['T2', 'T3', 'T4']
MID_LAYERS, MID_GROUPS = (12, 16), ('slot', 'final')
N_VEC, N_BOOT, SEED = 20, 2000, 20261125
HASHED = ['scripts/cross_model/25_m4_gemma_patch_decomposition.py', 'scripts/cross_model/24_xc1_gemma_site_sweep.py',
          'scripts/cross_model/02_matched_splicing.py', 'src/evaluation/pile_splicing.py', 'src/evaluation/patching.py',
          'src/models/backends.py', 'data/repe/repe_lm/prompts.csv', 'data/repe/repe_lm/templates.json']
OUTPUT_FILES = ['site_selection.json', 'per_pair.csv', 'decomposition_summary.csv', 'hypotheses.csv',
                'sanity_checks.json', 'control_draws.npz']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


X1 = _mod('scripts/cross_model/24_xc1_gemma_site_sweep.py', 'xc1')
NROW = X1.NROW


def build_rows(tok, split, templates, variants, bos=2):
    """Pairs (statement, template) with ids per variant; slot = the single differing token; drops otherwise."""
    d = pd.read_csv(DATA / 'prompts.csv')
    tm = json.loads((DATA / 'templates.json').read_text())
    d = d[(d.split == split) & d.template.isin(templates) & d.variant.isin(variants)]
    pos = {}
    for t in templates:
        w = tm[t]['variants'][variants[0]]['word']
        f = tm[t]['instruction'].format(w)
        a = [bos] + tok('USER: ' + f, add_special_tokens=False)['input_ids']
        b = [bos] + tok('USER: ' + f + ' ASSISTANT:', add_special_tokens=False)['input_ids']
        assert b[:len(a)] == a
        pos[t] = dict(asst_start=len(a), asst_end=len(b) - 1)
    rows, drops = [], 0
    for (sid, t), g in d.groupby(['statement_id', 'template']):
        v = g.set_index('variant')
        ids = {k: [bos] + tok(v.loc[k, 'prompt'], add_special_tokens=False)['input_ids'] for k in variants}
        n = len(ids[variants[0]])
        if any(len(ids[k]) != n for k in variants):
            drops += 1
            continue
        diffs = {i for k in variants[1:] for i in range(n) if ids[k][i] != ids[variants[0]][i]}
        if len(diffs) != 1:
            drops += 1
            continue
        rows.append(dict(statement_id=int(sid), template=t, ids=ids, final=n - 1, n_tokens=n, slot=diffs.pop(),
                         cont=[], **pos[t]))
    rows.sort(key=lambda r: (r['template'], r['n_tokens'], r['statement_id']))
    return rows, drops


def site_mask(rows, group, T):
    return X1.masks(rows, T)[group]


# ------------------------------------------------------------------------------------------------ core
def T_of(eng, base_ids, fin, L, m, vs, lpa, k0):
    """T(v) for each v in vs ([n, T, d] each, zero outside m)."""
    out = []
    for v in vs:
        lp, _, _ = eng.run(base_ids, fin, L, add=v * m[..., None])
        out.append(k0 - pt.kl(lpa, lp))
    return out


def decompose(eng, sae, rows_all, L, group, kind, gen, gcpu):
    a, b = ('H', 'U') if kind == 'persona' else ('Na', 'Nb')
    recs, draws, ident, zero = [], {'iso_rec': [], 'iso_err': [], 'shuf_rec': []}, 0.0, 0.0
    W = sae.sae.W_dec.float()                                           # [d_sae, d]
    for b0 in range(0, len(rows_all), NROW):
        rows = rows_all[b0:b0 + NROW]
        fin = [r['final'] for r in rows]
        A, B_ = [r['ids'][a] for r in rows], [r['ids'][b] for r in rows]
        lpa, _, ca = eng.run(A, fin, capture=[L])
        lpb, _, cb = eng.run(B_, fin, capture=[L])
        m = site_mask(rows, group, eng.t_fix)
        xa, xb = ca[L].float(), cb[L].float()
        n, T, D = xa.shape

        def sae_full(x):                                                # SAE at the masked positions only
            r = torch.zeros_like(x)
            z = torch.zeros(n, T, W.shape[0], device=x.device)
            xm = x[m]
            r[m] = sae(xm)[0].float()
            z[m] = sae.sae.encode(xm).float()
            return r, z
        ra, za = sae_full(xa)
        rb, zb = sae_full(xb)
        mm = m[..., None]
        d, dr = (xa - xb) * mm, (ra - rb) * mm
        de = d - dr
        ident = max(ident, float((((xa - ra) - (xb - rb)) * mm - de).abs().max()))   # e_H - e_U vs Delta - Delta_rec
        k0 = pt.kl(lpa, lpb)
        lf, lr, le, lh = T_of(eng, B_, fin, L, m, [d, dr, de, dr / 2], lpa, k0)
        z0 = T_of(eng, B_, fin, L, m, [torch.zeros_like(d)], lpa, k0)[0]
        zero = max(zero, float(z0.abs().max()))
        r = dict(statement_id=[x['statement_id'] for x in rows], template=[x['template'] for x in rows], kind=kind,
                 kl_base=k0.cpu().numpy(), T_full=lf.cpu().numpy(), T_rec=lr.cpu().numpy(), T_err=le.cpu().numpy(),
                 T_halfrec=lh.cpu().numpy(), norm_delta=d.norm(dim=-1).sum(-1).cpu().numpy(),
                 norm_rec=dr.norm(dim=-1).sum(-1).cpu().numpy(), norm_err=de.norm(dim=-1).sum(-1).cpu().numpy(),
                 cos_rec_delta=torch.nn.functional.cosine_similarity(dr.flatten(1), d.flatten(1), dim=-1).cpu().numpy())
        if kind == 'persona':
            kb = pt.kl(lpb, lpa)
            bf, br, be = T_of(eng, A, fin, L, m, [-d, -dr, -de], lpb, kb)
            r.update(TB_full=bf.cpu().numpy(), TB_rec=br.cpu().numpy(), TB_err=be.cpu().numpy())
            P = [x['ids']['Hp'] for x in rows]
            lps, _, cs = eng.run(P, fin, capture=[L])
            xs = cs[L].float()
            rs, _ = sae_full(xs)
            ds, dsr = (xa - xs) * mm, (ra - rs) * mm
            ks = pt.kl(lpa, lps)
            sf, sr, se = T_of(eng, P, fin, L, m, [ds, dsr, ds - dsr], lpa, ks)
            r.update(kl_syn=ks.cpu().numpy(), S_full=sf.cpu().numpy(), S_rec=sr.cpu().numpy(), S_err=se.cpu().numpy())
            dz = (za - zb)
            iso_r = [pt.random_like(dr, gen) for _ in range(N_VEC)]
            iso_e = [pt.random_like(de, gen) for _ in range(N_VEC)]
            shuf = []
            for _ in range(N_VEC):
                perm = torch.randperm(W.shape[0], generator=gcpu).to(dz.device)
                v = dz[..., perm] @ W
                shuf.append(v * (dr.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)))
            for name, vs in (('iso_rec', iso_r), ('iso_err', iso_e), ('shuf_rec', shuf)):
                draws[name].append(torch.stack(T_of(eng, B_, fin, L, m, vs, lpa, k0)).cpu().numpy())
        recs.append(pd.DataFrame(r))
    df = pd.concat(recs, ignore_index=True)
    return df, {k: (np.concatenate(v, 1) if v else None) for k, v in draws.items()}, ident, zero


def add_derived(df):
    df['I'] = df.T_full - df.T_rec - df.T_err
    df['phi_rec'] = 0.5 * (df.T_rec + df.T_full - df.T_err)
    df['phi_err'] = 0.5 * (df.T_err + df.T_full - df.T_rec)
    df['phi_diff'] = df.phi_rec - df.phi_err
    df['lin_dev'] = df.T_halfrec - 0.5 * df.T_rec
    if 'S_rec' in df:
        df['rec_minus_syn_rec'] = df.T_rec - df.S_rec
    return df


def ratio_ci(num, den, sid, rng):
    u, inv = np.unique(sid, return_inverse=True)
    nm = np.bincount(inv, num) / np.bincount(inv)
    dm = np.bincount(inv, den) / np.bincount(inv)
    idx = rng.integers(len(u), size=(N_BOOT, len(u)))
    r = nm[idx].mean(1) / dm[idx].mean(1)
    return float(nm.mean() / dm.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def per_statement_mean(v, sid):
    return float(pd.Series(v).groupby(sid).mean().mean())


def summarise(df, draws, meta, rng):
    out = []
    sets = {'T2_T4': df.template.isin(PRIMARY), 'T1': df.template == 'T1', 'all_T1_T4': df.template.isin(PERSONA)}
    if meta['kind'] == 'neutral':
        sets = {'neutral': df.template.isin(NEUTRAL)}
    for name, msk in sets.items():
        g = df[msk]
        if not len(g):
            continue
        sid = g.statement_id.to_numpy()
        r = dict(meta, template_set=name, n_pairs=int(msk.sum()))
        cols = ['kl_base', 'T_full', 'T_rec', 'T_err', 'I', 'phi_rec', 'phi_err', 'phi_diff', 'lin_dev']
        if meta['kind'] == 'persona':
            cols += ['TB_full', 'TB_rec', 'TB_err', 'kl_syn', 'S_full', 'S_rec', 'S_err', 'rec_minus_syn_rec']
        for c in cols:
            s = pt.stat_summary(g[c].to_numpy(), sid, N_BOOT, rng)
            r.update({f'{c}_{k}': v for k, v in s.items() if k in ('mean', 'ci_low', 'ci_high', 'p')})
        r['preservation_rec'], r['preservation_rec_ci_low'], r['preservation_rec_ci_high'] = \
            ratio_ci(g.T_rec.to_numpy(), g.T_full.to_numpy(), sid, rng)
        r['preservation_err'] = float(g.T_err.mean() / g.T_full.mean())
        r['shapley_share_rec'], r['shapley_share_rec_ci_low'], r['shapley_share_rec_ci_high'] = \
            ratio_ci(g.phi_rec.to_numpy(), g.T_full.to_numpy(), sid, rng)
        r['abs_I_over_T_full'], r['abs_I_ci_low'], r['abs_I_ci_high'] = \
            ratio_ci(g.I.abs().to_numpy(), g.T_full.to_numpy(), sid, rng)
        r['additive_reading_allowed'] = bool(r['abs_I_ci_high'] < 0.2)
        if meta['kind'] == 'persona':
            mm = msk.to_numpy()
            for k, v in draws.items():
                dm = np.array([per_statement_mean(v[i][mm], sid) for i in range(v.shape[0])])
                r[f'{k}_mean'], r[f'{k}_q95'] = float(dm.mean()), float(np.quantile(dm, 0.95))
                if k in ('iso_rec', 'shuf_rec'):
                    s = pt.stat_summary(g.T_rec.to_numpy() - v.mean(0)[mm], sid, N_BOOT, rng)
                    r[f'T_rec_minus_{k}_p'] = s['p']
                    r[f'T_rec_minus_{k}_ci_low'] = s['ci_low']
            r['T_rec_above_controls_q95'] = bool(r['T_rec_mean'] > max(r['iso_rec_q95'], r['shuf_rec_q95']))
        out.append(r)
    return out


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, p[o[i]] * n / (i + 1))
        q[o[i]] = run
    return q


# ------------------------------------------------------------------------------------------------ sites
def mid_site(eng, va, rng, cells_train):
    c = cells_train
    cand = c[c.layer.isin(MID_LAYERS) & c.group.isin(MID_GROUPS) & c.passes].copy()
    cand['dist'] = (cand.E_share - 0.5).abs()
    cand = cand.sort_values(['dist', 'R_A_mean'], ascending=[True, False])
    res = dict(rule='among XC-1 TRAIN-passing cells at layers {12,16} in {slot, final}: E_share closest to 0.5, ties -> '
                    'larger R_A; confirm on validation T2-T4 with original activations: T_full CI low > 0 and '
                    '(T_full - S_full) CI low > 0; else next; none -> fixed sites only',
               candidates=cand[['layer', 'group', 'E_share', 'R_A_mean', 'dist']].to_dict('records'), tried=[])
    rows_all = [x for x in va if x['template'] in PRIMARY]
    for _, r in cand.iterrows():
        L, g = int(r.layer), r.group
        recs = []
        for b0 in range(0, len(rows_all), NROW):
            rows = rows_all[b0:b0 + NROW]
            fin = [x['final'] for x in rows]
            H, U, P = ([x['ids'][k] for x in rows] for k in ('H', 'U', 'Hp'))
            lH, _, cH = eng.run(H, fin, capture=[L])
            lU, _, cU = eng.run(U, fin, capture=[L])
            lP, _, cP = eng.run(P, fin, capture=[L])
            m = site_mask(rows, g, eng.t_fix)
            tf = T_of(eng, U, fin, L, m, [(cH[L] - cU[L]).float()], lH, pt.kl(lH, lU))[0]
            ts = T_of(eng, P, fin, L, m, [(cH[L] - cP[L]).float()], lH, pt.kl(lH, lP))[0]
            recs.append(pd.DataFrame(dict(sid=[x['statement_id'] for x in rows], tf=tf.cpu().numpy(),
                                          ts=ts.cpu().numpy())))
        df = pd.concat(recs)
        s1 = pt.stat_summary(df.tf.to_numpy(), df.sid.to_numpy(), N_BOOT, rng)
        s2 = pt.stat_summary((df.tf - df.ts).to_numpy(), df.sid.to_numpy(), N_BOOT, rng)
        ok = bool(s1['ci_low'] > 0 and s2['ci_low'] > 0)
        res['tried'].append(dict(layer=L, group=g, T_full_val_T2_T4=s1, excess_over_synonym_val_T2_T4=s2, confirmed=ok))
        if ok:
            res['selected'] = dict(layer=L, group=g)
            return res
    res['selected'] = None
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    ap.add_argument('--smoke-sites', default='12:final,5:slot')
    a = ap.parse_args()
    global N_VEC, N_BOOT
    out = a.smoke_out if a.smoke else OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel in HASHED:
            if pr['hashes'].get(rel) != sha(REPO / rel):
                raise SystemExit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'run.log', 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    from transformers import AutoTokenizer
    from src.models.backends import GEMMA_ID, GEMMA_REV
    tok = AutoTokenizer.from_pretrained(GEMMA_ID, revision=GEMMA_REV)
    rng = np.random.default_rng(SEED)
    split = 'train' if a.smoke else 'validation'
    va, dp = build_rows(tok, split, PERSONA, ['H', 'U', 'Hp'])
    ne, dn = build_rows(tok, split, NEUTRAL, ['Na', 'Nb'])
    if a.smoke:
        keep = sorted({r['statement_id'] for r in va})[:12]
        va = [r for r in va if r['statement_id'] in keep]
        ne = [r for r in ne if r['statement_id'] in keep]
        N_VEC, N_BOOT = 3, 200
    allr = va + ne
    eng = X1.Eng(max(r['n_tokens'] for r in allr), max(r['n_tokens'] for r in allr))
    gen = torch.Generator(device='cuda').manual_seed(SEED)
    gcpu = torch.Generator().manual_seed(SEED)
    # ---------------- sites (no SAE loaded yet)
    if a.smoke:
        fixed = [dict(layer=int(s.split(':')[0]), group=s.split(':')[1], role='smoke') for s in a.smoke_sites.split(',')]
        mid = dict(selected=None, note='smoke: no mid-site rule')
    else:
        sel = json.loads((XC1 / 'selected_sites.json').read_text())
        fixed = [dict(layer=c['layer'], group=c['group'], role=c['role']) for c in sel['selected'] if c['confirmed']]
        if not fixed:
            (out / 'site_selection.json').write_text(json.dumps(dict(
                stop='XC-1 confirmed no selected cell: M4-G not justified (pre-registered stop rule)'), indent=1))
            log('XC-1 stop rule: no confirmed cell; M4-G not run.')
            return
        mid = mid_site(eng, va, rng, pd.read_csv(XC1 / 'cells_train.csv'))
    sites = list(fixed)
    if mid.get('selected') and (mid['selected']['layer'], mid['selected']['group']) not in \
            [(s['layer'], s['group']) for s in sites]:
        sites.append(dict(mid['selected'], role='mid (A1)'))
    (out / 'site_selection.json').write_text(json.dumps(dict(fixed=fixed, mid=mid, sites=sites), indent=1, default=float))
    log(f'sites {sites}')
    # ---------------- decomposition
    xb = _mod('scripts/cross_model/02_matched_splicing.py', 'xb02')
    summ, per_pair, draws_all, checks = [], [], {}, {}
    for s in sites:
        L, g = s['layer'], s['group']
        spec = [x for x in xb.sae_specs('gemma', L, ['canonical']) if x['family'] == 'gemmascope_16k_canonical'][0]
        sae = xb.load_sae('gemma', spec, 'cuda')
        meta0 = dict(layer=L, group=g, role=s['role'], depth=(L + 1) / 26, sae_id=spec['sae_id'],
                     sae_sha256=sha(spec['path']))
        for kind, rows in (('persona', va), ('neutral', ne)):
            df, dr, ident, zero = decompose(eng, sae, rows, L, g, kind, gen, gcpu)
            df = add_derived(df)
            df['layer'], df['group'] = L, g
            per_pair.append(df)
            meta = dict(meta0, kind=kind)
            summ += summarise(df, dr, meta, rng)
            checks[f'L{L}|{g}|{kind}'] = dict(identity_max_abs=ident, zero_patch_max_abs_T=zero,
                                              shapley_sum_max_err=float((df.phi_rec + df.phi_err - df.T_full).abs().max()))
            if kind == 'persona':
                for k, v in dr.items():
                    draws_all[f'L{L}__{g}__{k}'] = v
        log(f'site L{L} {g} done ({time.time() - t0:.0f} s)')
        del sae
        torch.cuda.empty_cache()
    sm = pd.DataFrame(summ)
    # cross-check T_full against XC-1 validation R_A (same pairs, all T1-T4)
    if not a.smoke:
        cv = pd.read_csv(XC1 / 'cells_validation.csv')
        for s in sites:
            ra = cv[(cv.layer == s['layer']) & (cv.group == s['group'])].R_A_mean.iloc[0]
            tf = sm[(sm.layer == s['layer']) & (sm.group == s['group']) & (sm.kind == 'persona') &
                    (sm.template_set == 'all_T1_T4')].T_full_mean.iloc[0]
            checks[f"L{s['layer']}|{s['group']}|T_full_vs_XC1_R_A_abs_diff"] = float(abs(ra - tf))
    checks['pass'] = all((v['identity_max_abs'] < 1e-3 and v['zero_patch_max_abs_T'] < 1e-6 and
                          v['shapley_sum_max_err'] < 1e-9) if isinstance(v, dict) else v < 1e-5
                         for k, v in checks.items())
    (out / 'sanity_checks.json').write_text(json.dumps(checks, indent=1))
    sm.to_csv(out / 'decomposition_summary.csv', index=False)
    pd.concat(per_pair, ignore_index=True).to_csv(out / 'per_pair.csv', index=False)
    np.savez_compressed(out / 'control_draws.npz', **draws_all)
    pr_ = sm[(sm.kind == 'persona') & (sm.template_set == 'T2_T4')].copy()
    pr_['H4a_p'] = pr_[['T_rec_minus_iso_rec_p', 'T_rec_minus_shuf_rec_p']].max(1)
    pr_['H4a_q'] = bh(pr_.H4a_p.to_numpy())
    pr_['H4a_pass'] = (pr_.H4a_q < 0.05) & (pr_.T_rec_ci_low > 0) & pr_.T_rec_above_controls_q95
    pr_['H4b_q'] = bh(pr_.phi_diff_p.to_numpy())
    pr_['H4b_pass'] = (pr_.H4b_q < 0.05) & (pr_.phi_diff_mean > 0)
    cols = ['layer', 'group', 'role', 'depth', 'T_full_mean', 'T_rec_mean', 'T_rec_ci_low', 'T_err_mean',
            'preservation_rec', 'preservation_rec_ci_low', 'preservation_rec_ci_high', 'shapley_share_rec',
            'abs_I_over_T_full', 'abs_I_ci_high', 'additive_reading_allowed', 'iso_rec_q95', 'shuf_rec_q95',
            'H4a_p', 'H4a_q', 'H4a_pass', 'phi_diff_mean', 'H4b_q', 'H4b_pass']
    pr_[cols].to_csv(out / 'hypotheses.csv', index=False)
    log(pr_[cols].to_string())
    log(json.dumps(dict(checks_pass=checks['pass'], runtime_s=time.time() - t0, n_forward=eng.n_forward,
                        finished_utc=datetime.now(timezone.utc).isoformat(), drops=dict(persona=dp, neutral=dn),
                        n_persona_pairs=len(va), n_neutral_pairs=len(ne))))
    if not checks['pass']:
        raise SystemExit('sanity checks failed (outputs kept for inspection; results not valid).')


if __name__ == '__main__':
    main()
