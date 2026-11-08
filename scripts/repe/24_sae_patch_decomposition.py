#!/usr/bin/env python3
"""M4: SAE causal-faithfulness decomposition at the M3 sites (+ the amended mid-depth site).

Pre-registered in results/repe/sae_patch_decomposition/preregistration.json (amendment A1 there).

For each site (layer L, position: persona slot or final token) and each of the 9 base SAEs at layer L, for every
matched pair (statement, template):
    Delta     = x_H - x_U                      original activation difference at the site
    Delta_rec = x_hat_H - x_hat_U              SAE-reconstructed difference
    Delta_err = e_H - e_U                      reconstruction-error difference   (identity Delta = Delta_rec + Delta_err)
    T(v)      = KL(p_H||p_U) - KL(p_H||p_{U + v at site})     restoration toward the honest output, nats
    T_full = T(Delta), T_rec = T(Delta_rec), T_err = T(Delta_err), I = T_full - T_rec - T_err
    Shapley: phi_rec = (T_rec + T_full - T_err)/2, phi_err = (T_err + T_full - T_rec)/2  (sum exactly to T_full)
    linearity: T(Delta_rec/2) vs T_rec/2
Controls: other direction (H - v, toward U), synonym pairs (H vs Hp), neutral one-word pairs (Na vs Nb), isotropic
matched-norm vectors for Delta_rec and Delta_err, and decoder-span shuffles (Delta_z permuted across features).
Features: per-pair patch of features F (U's features F set to H's values, error kept): v_F = W_dec[:, F] dz_F.
Ranking by first-order attribution on TRAIN (T2-T4) excluding position-tracking features; k chosen on TRAIN;
controls on VALIDATION: 200 random feature sets matched on OWT activation-frequency x |dz| deciles, 200 norm-matched
isotropic vectors, top-|dz| sets.
This measures what patching SAE-defined directions does to GPT-2's output. It does NOT show that GPT-2 uses SAE
features.

Run from repository root:  python scripts/repe/24_sae_patch_decomposition.py [--smoke --smoke-out DIR]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import patching as pt  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/repe/sae_patch_decomposition'
DATA = REPO / 'data/repe/repe_lm'
M3 = REPO / 'results/repe/causal_site_sweep'
OWT_VAL = REPO / 'data/activations/gpt2_small/validation'
PERSONA, NEUTRAL, PRIMARY = ['T1', 'T2', 'T3', 'T4'], ['N1', 'N2'], ['T2', 'T3', 'T4']
ARCHS, SEEDS = ('relu', 'topk', 'jumprelu'), (42, 1042, 2042)
FIXED_SITES = [(1, 'slot'), (11, 'final')]
K_GRID = (1, 4, 16, 64)
N_VEC, N_SET, N_BOOT, SEED, B, CHUNK, POS_CORR = 20, 200, 2000, 20261108, 128, 10, 0.3
GEN_CPU = torch.Generator().manual_seed(20261108)
_OWT_CACHE = {}
OUTPUT_FILES = ['mid_site_selection.json', 'per_pair_validation.csv', 'decomposition_summary.csv',
                'features_summary.csv', 'selected_features.json', 'hypotheses.csv', 'architecture_summary.csv',
                'sanity_checks.json', 'control_draws.npz']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data

def build_rows(tok, split, templates, variants):
    d = pd.read_csv(DATA / 'prompts.csv')
    assert set(d.split) <= {'train', 'validation'}
    tm = json.loads((DATA / 'templates.json').read_text())
    d = d[(d.split == split) & d.template.isin(templates)]
    rows = []
    for (sid, t), g in d.groupby(['statement_id', 'template']):
        v = g.set_index('variant')
        ids = {k: list(map(int, v.loc[k, 'token_ids'].split())) for k in variants}
        n = len(ids[variants[0]])
        assert all(len(x) == n for x in ids.values())
        rows.append(dict(statement_id=int(sid), template=t, ids=ids, n=n, final=n - 1, slot=int(tm[t]['slot_pos'])))
    rows.sort(key=lambda r: (r['template'], r['n'], r['statement_id']))
    return rows


def toks_of(rows, variant, device):
    T = max(r['n'] for r in rows)
    x = torch.full((len(rows), T), 50256, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :r['n']] = torch.tensor(r['ids'][variant])
    return x.to(device)


# --------------------------------------------------------------------------------------------- model

class Engine:
    def __init__(self, model, L):
        self.m, self.L = model, L
        self.hook = f'blocks.{L}.hook_resid_pre'

    @torch.inference_mode()
    def clean(self, toks, finals):
        cap = {}
        self.m.run_with_hooks(toks, return_type=None, fwd_hooks=[
            (self.hook, lambda x, hook: cap.__setitem__('r', x.clone())),
            ('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x))])
        ar = torch.arange(len(finals), device=toks.device)
        return cap['r'], self._logp(cap['h'], finals, ar)

    def _logp(self, h, finals, ar):
        return self.m.unembed(h[ar, finals][:, None])[:, 0].double().log_softmax(-1)

    @torch.inference_mode()
    def patched(self, base, pos, finals, vs):
        """base [n,T,d]; vs list of [n,d] vectors added at (row, pos). Returns list of final log-probs."""
        out = []
        for c0 in range(0, len(vs), CHUNK):
            chunk = vs[c0:c0 + CHUNK]
            r = len(chunk)
            res = base.repeat(r, 1, 1)
            n = base.shape[0]
            ar = torch.arange(n * r, device=base.device)
            res[ar, pos.repeat(r)] += torch.cat(chunk).to(res.dtype)
            cap = {}
            self.m.run_with_hooks(res, start_at_layer=self.L, return_type=None,
                                  fwd_hooks=[('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x))])
            lp = self._logp(cap['h'], finals.repeat(r), ar)
            out += list(lp.split(n))
        return out

    def grad_restoration(self, base, pos, finals, lp_target):
        """d KL(p_target || p_{base + v at pos}) / dv at v = 0, per row [n, d]."""
        v = torch.zeros(base.shape[0], base.shape[-1], device=base.device, requires_grad=True)
        with torch.enable_grad():
            res = base.detach().clone()
            ar = torch.arange(base.shape[0], device=base.device)
            res = res.index_put((ar, pos), res[ar, pos] + v)
            cap = {}
            self.m.run_with_hooks(res, start_at_layer=self.L, return_type=None,
                                  fwd_hooks=[('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x))])
            lq = self.m.unembed(cap['h'][ar, finals][:, None])[:, 0].double().log_softmax(-1)
            loss = pt.kl(lp_target.clone(), lq).sum()
            (g,) = torch.autograd.grad(loss, v)
        return g.detach()


# ------------------------------------------------------------------------------------- site handling

def positions(rows, group, device):
    return torch.tensor([r['slot'] if group == 'slot' else r['final'] for r in rows], device=device)


def clean_batches(eng, rows, variants, group, device):
    """Per batch: dict(rows, pos, finals, resid{variant}, lp{variant}, x{variant} at site)."""
    out = []
    for b0 in range(0, len(rows), B):
        rs = rows[b0:b0 + B]
        fin = torch.tensor([r['final'] for r in rs], device=device)
        pos = positions(rs, group, device)
        ar = torch.arange(len(rs), device=device)
        bt = dict(rows=rs, pos=pos, finals=fin, resid={}, lp={}, x={})
        for v in variants:
            r_, lp = eng.clean(toks_of(rs, v, device), fin)
            bt['resid'][v], bt['lp'][v], bt['x'][v] = r_, lp, r_[ar, pos].float()
        out.append(bt)
    return out


def select_mid_site(eng_by_layer, va_rows, device, rng):
    """Amendment A1: train rule on existing M3 numbers (no SAE), then validation confirmation on T2-T4."""
    c = pd.read_csv(M3 / 'cells_train.csv')
    cand = c[c.layer_index.isin([6, 7, 8]) & c.group.isin(['slot', 'final']) & c.passes].copy()
    cand['dist'] = (cand.E_share - 0.5).abs()
    cand = cand.sort_values(['dist', 'R_A_mean'], ascending=[True, False])
    res = dict(rule='among M3-passing train cells at resid_pre layers 6-8 in {slot, final}: E_share closest to 0.5 '
                    '(handover point), ties -> larger R_A; confirm on validation T2-T4: T_full CI low > 0 and '
                    '(T_full - T_full_synonym) CI low > 0; else try the next candidate in the same order; '
                    'if none confirms, proceed with the two fixed sites',
               candidates=cand[['site', 'group', 'E_share', 'R_A_mean', 'dist']].to_dict('records'), tried=[])
    for _, r in cand.iterrows():
        L = int(r.layer_index)
        eng = eng_by_layer(L)
        rows = [x for x in va_rows if x['template'] in PRIMARY]
        recs = []
        for bt in clean_batches(eng, rows, ['H', 'U', 'Hp'], r.group, device):
            d = bt['x']['H'] - bt['x']['U']
            ds = bt['x']['H'] - bt['x']['Hp']
            lA = eng.patched(bt['resid']['U'], bt['pos'], bt['finals'], [d])[0]
            lS = eng.patched(bt['resid']['Hp'], bt['pos'], bt['finals'], [ds])[0]
            k_hu, k_hp = pt.kl(bt['lp']['H'], bt['lp']['U']), pt.kl(bt['lp']['H'], bt['lp']['Hp'])
            tf = (k_hu - pt.kl(bt['lp']['H'], lA)).cpu().numpy()
            ts = (k_hp - pt.kl(bt['lp']['H'], lS)).cpu().numpy()
            recs.append(pd.DataFrame(dict(sid=[x['statement_id'] for x in bt['rows']], tf=tf, ts=ts)))
        df = pd.concat(recs)
        s1 = pt.stat_summary(df.tf.to_numpy(), df.sid.to_numpy(), N_BOOT, rng)
        s2 = pt.stat_summary((df.tf - df.ts).to_numpy(), df.sid.to_numpy(), N_BOOT, rng)
        ok = s1['ci_low'] > 0 and s2['ci_low'] > 0
        res['tried'].append(dict(site=r.site, group=r.group, T_full_val_T2_T4=s1, excess_over_synonym_val_T2_T4=s2,
                                 confirmed=bool(ok)))
        if ok:
            res['selected'] = dict(layer=L, group=r.group, site=r.site)
            return res
    res['selected'] = None
    return res


# ----------------------------------------------------------------------------------------- OWT stats

def owt_feature_stats(sae, L, device, max_tokens=None):
    if L in _OWT_CACHE:
        x, pos = _OWT_CACHE[L]
        return _owt_stats(sae, x, pos, device)
    xs, ps = [], []
    for p in sorted(OWT_VAL.glob('chunk_*.pt')):
        ch = torch.load(p, map_location='cpu', weights_only=False)
        a = ch['activations'][f'blocks.{L}.hook_resid_pre']
        for i, l in enumerate(ch['lengths']):
            l = int(l)
            if l > 1:
                xs.append(a[i, 1:l].float())
                ps.append(torch.arange(1, l, dtype=torch.float32))
    x, pos = torch.cat(xs), torch.cat(ps)
    _OWT_CACHE[L] = (x, pos)
    return _owt_stats(sae, x, pos, device)


def _owt_stats(sae, x, pos, device):
    with torch.no_grad():
        _, z, _ = sae(x.to(device))
    z = z.float()
    freq = (z > 0).float().mean(0)
    pc = pos.to(device) - pos.mean()
    zc = z - z.mean(0)
    corr = (zc * pc[:, None]).sum(0) / (zc.norm(dim=0) * pc.norm() + 1e-12)
    return freq.cpu().numpy(), corr.cpu().numpy()


# ------------------------------------------------------------------------------------- per-SAE analysis

def kl(a, b):
    return pt.kl(a, b)


def decompose(eng, sae, batches, kind, gen):
    """kind: 'persona' (H/U/Hp) or 'neutral' (Na/Nb). Returns per-pair DataFrame and per-draw sums."""
    W = sae.decoder.weight.detach().float()                      # [d, d_sae]
    recs, draws = [], {'iso_rec': [], 'iso_err': [], 'shuf_rec': []}
    ident = 0.0
    for bt in batches:
        a, b = ('H', 'U') if kind == 'persona' else ('Na', 'Nb')
        xa, xb = bt['x'][a], bt['x'][b]
        with torch.no_grad():
            ra, za, _ = sae(xa)
            rb, zb, _ = sae(xb)
        d, dr = xa - xb, ra.float() - rb.float()
        de = (xa - ra.float()) - (xb - rb.float())
        ident = max(ident, float((d - dr - de).abs().max()))
        base, pos, fin = bt['resid'][b], bt['pos'], bt['finals']
        lpa, lpb = bt['lp'][a], bt['lp'][b]
        k0 = kl(lpa, lpb)
        lf, lr, le, lh = eng.patched(base, pos, fin, [d, dr, de, dr / 2])
        r = dict(statement_id=[x['statement_id'] for x in bt['rows']], template=[x['template'] for x in bt['rows']],
                 kind=kind, kl_base=k0.cpu().numpy(),
                 T_full=(k0 - kl(lpa, lf)).cpu().numpy(), T_rec=(k0 - kl(lpa, lr)).cpu().numpy(),
                 T_err=(k0 - kl(lpa, le)).cpu().numpy(), T_halfrec=(k0 - kl(lpa, lh)).cpu().numpy(),
                 norm_delta=d.norm(dim=-1).cpu().numpy(), norm_rec=dr.norm(dim=-1).cpu().numpy(),
                 norm_err=de.norm(dim=-1).cpu().numpy(),
                 cos_rec_delta=torch.nn.functional.cosine_similarity(dr, d, dim=-1).cpu().numpy())
        if kind == 'persona':
            # other direction: subtract from H toward U
            kb = kl(lpb, lpa)
            bf, br, be = eng.patched(bt['resid'][a], pos, fin, [-d, -dr, -de])
            r.update(TB_full=(kb - kl(lpb, bf)).cpu().numpy(), TB_rec=(kb - kl(lpb, br)).cpu().numpy(),
                     TB_err=(kb - kl(lpb, be)).cpu().numpy())
            # synonym: Hp base restored toward H
            xs_ = bt['x']['Hp']
            with torch.no_grad():
                rs_, _, _ = sae(xs_)
            ds, dsr = xa - xs_, ra.float() - rs_.float()
            dse = ds - dsr
            ks = kl(lpa, bt['lp']['Hp'])
            sf, sr, se = eng.patched(bt['resid']['Hp'], pos, fin, [ds, dsr, dse])
            r.update(kl_syn=ks.cpu().numpy(), S_full=(ks - kl(lpa, sf)).cpu().numpy(),
                     S_rec=(ks - kl(lpa, sr)).cpu().numpy(), S_err=(ks - kl(lpa, se)).cpu().numpy())
            # matched-norm controls
            dz = (za - zb).float()
            iso_r = [pt.random_like(dr, gen) for _ in range(N_VEC)]
            iso_e = [pt.random_like(de, gen) for _ in range(N_VEC)]
            shuf = []
            for _ in range(N_VEC):
                perm = torch.randperm(dz.shape[1], generator=GEN_CPU).to(dz.device)
                v = dz[:, perm] @ W.T
                shuf.append(v * (dr.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)))
            for name, vs in (('iso_rec', iso_r), ('iso_err', iso_e), ('shuf_rec', shuf)):
                T = torch.stack([k0 - kl(lpa, l_) for l_ in eng.patched(base, pos, fin, vs)])   # [N_VEC, n]
                draws[name].append(T.cpu().numpy())
        recs.append(pd.DataFrame(r))
    df = pd.concat(recs, ignore_index=True)
    return df, {k: (np.concatenate(v, 1) if v else None) for k, v in draws.items()}, ident


def feature_patch_vectors(dz, W, feats):
    return dz[:, feats] @ W[:, feats].T


def feature_analysis(eng, sae, tr_batches, va_batches, freq, corr, gen, rng):
    W = sae.decoder.weight.detach().float()
    d_sae = W.shape[1]
    pos_feat = np.abs(corr) > POS_CORR
    # ---- TRAIN: attribution and |dz| on T2-T4
    attr, absdz, n = np.zeros(d_sae), np.zeros(d_sae), 0
    tr_cache = []
    for bt in tr_batches:
        with torch.no_grad():
            rh, zh, _ = sae(bt['x']['H'])
            ru, zu, _ = sae(bt['x']['U'])
        dz = (zh - zu).float()
        prim = torch.tensor([x['template'] in PRIMARY for x in bt['rows']], device=dz.device)
        g = eng.grad_restoration(bt['resid']['U'], bt['pos'], bt['finals'], bt['lp']['H']).float()
        # first-order restoration of feature j: -(dz_j W_j) . g
        a = -(dz * (g @ W))                              # [n, d_sae]
        attr += a[prim].sum(0).cpu().numpy()
        absdz += dz[prim].abs().sum(0).cpu().numpy()
        n += int(prim.sum())
        dr = ((rh - ru).float())
        tr_cache.append((bt, dz, dr, prim))
    attr, absdz = attr / n, absdz / n
    eligible = ~pos_feat & (absdz > 0)
    order_attr = [j for j in np.argsort(-attr) if eligible[j]]
    order_dz = [j for j in np.argsort(-absdz) if eligible[j]]

    def T_of(cache, feats_fn, prim_only):
        vals = []
        for bt, dz, dr, prim in cache:
            k0 = kl(bt['lp']['H'], bt['lp']['U'])
            vs = [feats_fn(dz, dr)]
            l_ = eng.patched(bt['resid']['U'], bt['pos'], bt['finals'], vs)[0]
            t = (k0 - kl(bt['lp']['H'], l_)).cpu().numpy()
            vals.append(t[prim.cpu().numpy()] if prim_only else t)
        return np.concatenate(vals)

    t_rec_train = T_of(tr_cache, lambda dz, dr: dr, True).mean()
    t_f_train = {k: T_of(tr_cache, lambda dz, dr, k=k: feature_patch_vectors(dz, W, order_attr[:k]), True).mean()
                 for k in K_GRID}
    k_star = next((k for k in K_GRID if t_f_train[k] >= 0.5 * t_rec_train), K_GRID[-1])
    # ---- VALIDATION
    va_cache = []
    for bt in va_batches:
        with torch.no_grad():
            rh, zh, _ = sae(bt['x']['H'])
            ru, zu, _ = sae(bt['x']['U'])
        prim = torch.tensor([x['template'] in PRIMARY for x in bt['rows']], device=zh.device)
        va_cache.append((bt, (zh - zu).float(), (rh - ru).float(), prim))
    sid = np.concatenate([[x['statement_id'] for x in bt['rows']] for bt, *_ in va_cache])
    tmpl = np.concatenate([[x['template'] for x in bt['rows']] for bt, *_ in va_cache])
    prim_all = np.isin(tmpl, PRIMARY)
    res = dict(k_star=k_star, t_rec_train_T2_T4=float(t_rec_train),
               t_F_train_T2_T4={k: float(v) for k, v in t_f_train.items()},
               n_position_features_excluded=int(pos_feat.sum()), n_eligible=int(eligible.sum()),
               top_features=[int(j) for j in order_attr[:64]],
               top_attr=[float(attr[j]) for j in order_attr[:64]])
    per = {}
    for k in K_GRID:
        per[f'F_attr_k{k}'] = T_of(va_cache, lambda dz, dr, k=k: feature_patch_vectors(dz, W, order_attr[:k]), False)
        per[f'F_absdz_k{k}'] = T_of(va_cache, lambda dz, dr, k=k: feature_patch_vectors(dz, W, order_dz[:k]), False)
    per['F_all'] = T_of(va_cache, lambda dz, dr: dz @ W.T, False)
    per['T_rec_check'] = T_of(va_cache, lambda dz, dr: dr, False)
    # ---- controls at k*
    sel = order_attr[:k_star]
    fdec = np.minimum((pd.Series(freq).rank(pct=True).to_numpy() * 10).astype(int), 9)
    ddec = np.minimum((pd.Series(absdz).rank(pct=True).to_numpy() * 10).astype(int), 9)
    pool = {}
    for j in np.flatnonzero(eligible):
        pool.setdefault((fdec[j], ddec[j]), []).append(j)
    rnd_sets = []
    for _ in range(N_SET):
        s = []
        for j in sel:
            cands = [c for c in pool.get((fdec[j], ddec[j]), []) if c not in sel and c not in s]
            if not cands:
                cands = [c for c in np.flatnonzero(eligible & (fdec == fdec[j])) if c not in sel and c not in s]
            s.append(int(rng.choice(cands)))
        rnd_sets.append(s)
    set_means = {'T2_T4': [], 'T1': []}
    iso_means = {'T2_T4': [], 'T1': []}
    for s in rnd_sets:
        t = T_of(va_cache, lambda dz, dr, s=s: feature_patch_vectors(dz, W, s), False)
        set_means['T2_T4'].append(per_statement_mean(t[prim_all], sid[prim_all]))
        set_means['T1'].append(per_statement_mean(t[tmpl == 'T1'], sid[tmpl == 'T1']))
    for _ in range(N_SET):
        t = T_of(va_cache, lambda dz, dr: pt.random_like(feature_patch_vectors(dz, W, sel), gen), False)
        iso_means['T2_T4'].append(per_statement_mean(t[prim_all], sid[prim_all]))
        iso_means['T1'].append(per_statement_mean(t[tmpl == 'T1'], sid[tmpl == 'T1']))
    per_df = pd.DataFrame(per).assign(statement_id=sid, template=tmpl)
    return res, per_df, {k: np.array(v) for k, v in set_means.items()}, {k: np.array(v) for k, v in iso_means.items()}


def per_statement_mean(v, sid):
    s = pd.Series(v).groupby(sid).mean()
    return float(s.mean())


# -------------------------------------------------------------------------------------------- stats

def ratio_ci(num, den, sid, rng):
    u, inv = np.unique(sid, return_inverse=True)
    nm = np.bincount(inv, num) / np.bincount(inv)
    dm = np.bincount(inv, den) / np.bincount(inv)
    idx = rng.integers(len(u), size=(N_BOOT, len(u)))
    r = nm[idx].mean(1) / dm[idx].mean(1)
    return float(nm.mean() / dm.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summarise_decomp(df, draws, meta, rng):
    out = []
    sets = {'T2_T4': df.template.isin(PRIMARY), 'T1': df.template == 'T1', 'all_T1_T4': df.template.isin(PERSONA)}
    if meta['kind'] == 'neutral':
        sets = {'neutral': df.template.isin(NEUTRAL)}
    for name, m in sets.items():
        g = df[m]
        sid = g.statement_id.to_numpy()
        r = dict(meta, template_set=name, n_pairs=int(m.sum()))
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
            r['syn_preservation_rec'] = float(g.S_rec.mean() / g.S_full.mean())
            r['TB_preservation_rec'] = float(g.TB_rec.mean() / g.TB_full.mean())
            mm = m.to_numpy()
            for k, v in draws.items():
                dm = np.array([per_statement_mean(v[i][mm], sid) for i in range(v.shape[0])])
                r[f'{k}_mean'], r[f'{k}_q95'] = float(dm.mean()), float(np.quantile(dm, 0.95))
                # paired: T_rec minus that pair's mean control draw
                if k in ('iso_rec', 'shuf_rec'):
                    s = pt.stat_summary(g.T_rec.to_numpy() - v.mean(0)[mm], sid, N_BOOT, rng)
                    r[f'T_rec_minus_{k}_p'] = s['p']
                    r[f'T_rec_minus_{k}_ci_low'] = s['ci_low']
            r['T_rec_above_controls_q95'] = bool(r['T_rec_mean'] > max(r['iso_rec_q95'], r['shuf_rec_q95']))
        out.append(r)
    return out


def add_derived(df):
    df['I'] = df.T_full - df.T_rec - df.T_err
    df['phi_rec'] = 0.5 * (df.T_rec + df.T_full - df.T_err)
    df['phi_err'] = 0.5 * (df.T_err + df.T_full - df.T_rec)
    df['phi_diff'] = df.phi_rec - df.phi_err
    df['lin_dev'] = df.T_halfrec - 0.5 * df.T_rec
    if 'S_rec' in df:
        df['rec_minus_syn_rec'] = df.T_rec - df.S_rec
    return df


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    global N_VEC, N_SET, N_BOOT
    out = a.smoke_out if a.smoke else OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel in ('scripts/repe/24_sae_patch_decomposition.py', 'src/evaluation/patching.py'):
            if pr['hashes'].get(rel) != sha(REPO / rel):
                raise SystemExit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
    else:
        N_VEC, N_SET, N_BOOT = 3, 5, 200
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = 'cuda'
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained('gpt2')
    rows = {(s, k): build_rows(tok, s, PERSONA if k == 'persona' else NEUTRAL,
                               ['H', 'U', 'Hp'] if k == 'persona' else ['Na', 'Nb'])
            for s in ('train', 'validation') for k in ('persona', 'neutral')}
    if a.smoke:
        rows = {k: v[::10] for k, v in rows.items()}
    engines = {}
    eng_by_layer = lambda L: engines.setdefault(L, Engine(model, L))

    # ---- Amendment A1: mid-depth site (no SAE involved)
    mid = select_mid_site(eng_by_layer, rows[('validation', 'persona')], device, rng)
    (out / 'mid_site_selection.json').write_text(json.dumps(mid, indent=2, default=float))
    sites = list(FIXED_SITES)
    if mid['selected']:
        sites.insert(1, (mid['selected']['layer'], mid['selected']['group']))
    print('sites', sites, flush=True)

    sanity = dict(max_identity_abs_err=0.0, max_start_at_layer_kl=0.0, max_abs_T_rec_vs_feature_all=0.0,
                  max_abs_shapley_sum_err=0.0, T_full_vs_M3=[])
    pair_frames, summ, feat_rows, feat_json, hyp_rows, draws_npz = [], [], [], {}, [], {}
    m3v = pd.read_csv(M3 / 'cells_validation.csv')
    for (L, group) in sites:
        eng = eng_by_layer(L)
        vb = clean_batches(eng, rows[('validation', 'persona')], ['H', 'U', 'Hp'], group, device)
        vn = clean_batches(eng, rows[('validation', 'neutral')], ['Na', 'Nb'], group, device)
        tb = clean_batches(eng, [r for r in rows[('train', 'persona')] if r['template'] in PRIMARY],
                           ['H', 'U'], group, device)
        # sanity: start_at_layer with v = 0 reproduces the clean U output
        bt = vb[0]
        l0 = eng.patched(bt['resid']['U'], bt['pos'], bt['finals'], [torch.zeros_like(bt['x']['U'])])[0]
        sanity['max_start_at_layer_kl'] = max(sanity['max_start_at_layer_kl'], float(kl(bt['lp']['U'], l0).abs().max()))
        for arch in ARCHS:
            for seed in SEEDS:
                sae, _ = fm.load_sae(uninterrupted_checkpoint(L, arch, seed), device)
                meta = dict(layer=L, group=group, site=f'blocks.{L}.hook_resid_pre|{group}', architecture=arch,
                            seed=seed)
                dfp, dr, ident = decompose(eng, sae, vb, 'persona', gen)
                dfn, _, ident2 = decompose(eng, sae, vn, 'neutral', gen)
                sanity['max_identity_abs_err'] = max(sanity['max_identity_abs_err'], ident, ident2)
                dfp, dfn = add_derived(dfp), add_derived(dfn)
                sanity['max_abs_shapley_sum_err'] = max(sanity['max_abs_shapley_sum_err'],
                                                        float((dfp.phi_rec + dfp.phi_err - dfp.T_full).abs().max()))
                for d_ in (dfp, dfn):
                    pair_frames.append(d_.assign(**meta))
                summ += summarise_decomp(dfp, dr, dict(meta, kind='persona'), rng)
                summ += summarise_decomp(dfn, {}, dict(meta, kind='neutral'), rng)
                key = f'L{L}_{group}_{arch}_{seed}'
                for k, v in dr.items():
                    draws_npz[f'{key}_{k}'] = v.astype(np.float32)
                # ---- features
                freq, corr = owt_feature_stats(sae, L, device)
                fres, fper, set_means, iso_means = feature_analysis(eng, sae, tb, vb, freq, corr, gen, rng)
                sanity['max_abs_T_rec_vs_feature_all'] = max(sanity['max_abs_T_rec_vs_feature_all'],
                                                             float((fper.F_all - fper.T_rec_check).abs().max()))
                feat_json[key] = fres
                for ts, mask in (('T2_T4', fper.template.isin(PRIMARY)), ('T1', fper.template == 'T1')):
                    g = fper[mask]
                    sid = g.statement_id.to_numpy()
                    t_rec = pt.stat_summary(g.T_rec_check.to_numpy(), sid, N_BOOT, rng)['mean']
                    fr = dict(meta, template_set=ts, k_star=fres['k_star'], T_rec=t_rec)
                    for c in [c for c in g.columns if c.startswith('F_')]:
                        s = pt.stat_summary(g[c].to_numpy(), sid, N_BOOT, rng)
                        fr[f'{c}_mean'], fr[f'{c}_ci_low'] = s['mean'], s['ci_low']
                    fk = fr[f"F_attr_k{fres['k_star']}_mean"]
                    fr['F_kstar_share_of_T_rec'] = fk / t_rec if t_rec else np.nan
                    fr['random_set_mean'], fr['random_set_q95'] = float(set_means[ts].mean()), float(np.quantile(set_means[ts], 0.95))
                    fr['iso_mean'], fr['iso_q95'] = float(iso_means[ts].mean()), float(np.quantile(iso_means[ts], 0.95))
                    fr['p_vs_random_sets'] = float((1 + (set_means[ts] >= fk).sum()) / (1 + len(set_means[ts])))
                    fr['p_vs_iso'] = float((1 + (iso_means[ts] >= fk).sum()) / (1 + len(iso_means[ts])))
                    feat_rows.append(fr)
                    draws_npz[f'{key}_{ts}_featsets'] = set_means[ts].astype(np.float32)
                print(f'{meta["site"]} {arch} {seed}: done ({time.time() - t0:.0f}s)', flush=True)
                del sae
        m3 = m3v[(m3v.site == f'blocks.{L}.hook_resid_pre') & (m3v.group == group)]
        if len(m3):
            mine = pd.concat([f for f in pair_frames if (f.layer.iloc[0] == L) and (f.group.iloc[0] == group)
                              and (f.kind.iloc[0] == 'persona')])
            mine = mine[(mine.architecture == 'relu') & (mine.seed == 42)]
            sanity['T_full_vs_M3'].append(dict(site=f'L{L} {group}', m4=float(mine.groupby('statement_id').T_full.mean().mean()),
                                               m3=float(m3.R_A_mean.iloc[0])))

    pairs = pd.concat(pair_frames, ignore_index=True)
    sm = pd.DataFrame(summ)
    fs = pd.DataFrame(feat_rows)
    # ---- hypotheses on the primary set (T2-T4), BH over SAE x site
    P = sm[(sm.kind == 'persona') & (sm.template_set == 'T2_T4')].copy()
    F = fs[fs.template_set == 'T2_T4'].set_index(['site', 'architecture', 'seed'])
    P['H4a_p'] = P[['T_rec_minus_iso_rec_p', 'T_rec_minus_shuf_rec_p']].max(1)
    P['H4b_p'] = P['phi_diff_p']
    P['H4c_p'] = [F.loc[(r.site, r.architecture, r.seed), 'p_vs_random_sets'] for r in P.itertuples()]
    for h in ('H4a', 'H4b', 'H4c'):
        P[f'{h}_q'] = pt_bh(P[f'{h}_p'].to_numpy())
    P['H4a_pass'] = (P.H4a_q < 0.05) & P.T_rec_above_controls_q95 & (P.T_rec_ci_low > 0)
    P['H4b_pass'] = (P.H4b_q < 0.05)
    kci = np.array([F.loc[(r.site, r.architecture, r.seed), f'F_attr_k{int(F.loc[(r.site, r.architecture, r.seed), "k_star"])}_ci_low'] > 0
                    for r in P.itertuples()], dtype=bool)
    P['H4c_pass'] = (P.H4c_q.to_numpy() < 0.05) & kci
    hyp = P[['site', 'architecture', 'seed', 'T_full_mean', 'T_rec_mean', 'T_err_mean', 'preservation_rec',
             'shapley_share_rec', 'abs_I_over_T_full', 'additive_reading_allowed', 'H4a_p', 'H4a_q', 'H4a_pass',
             'H4b_p', 'H4b_q', 'H4b_pass', 'H4c_p', 'H4c_q', 'H4c_pass']]
    arch = []
    for (site, ar_), g in hyp.groupby(['site', 'architecture']):
        d = dict(site=site, architecture=ar_)
        for c in ('T_full_mean', 'T_rec_mean', 'T_err_mean', 'preservation_rec', 'shapley_share_rec', 'abs_I_over_T_full'):
            d[c] = g[c].mean()
            d[c.replace('_mean', '') + '_sd'] = g[c].std()
        for h in ('H4a', 'H4b', 'H4c'):
            d[f'{h}_seeds_passing'] = int(g[f'{h}_pass'].sum())
            d[f'{h}_claim'] = bool(g[f'{h}_pass'].sum() >= 2)
        arch.append(d)
    sanity['T_full_identical_across_saes'] = float(pairs[pairs.kind == 'persona'].groupby(
        ['layer', 'group', 'statement_id', 'template']).T_full.agg(lambda v: v.max() - v.min()).max())
    sanity['pass'] = bool(sanity['max_identity_abs_err'] < 1e-3 and sanity['max_start_at_layer_kl'] < 1e-6
                          and sanity['max_abs_shapley_sum_err'] < 1e-9 and sanity['max_abs_T_rec_vs_feature_all'] < 1e-4
                          and sanity['T_full_identical_across_saes'] < 1e-9
                          and all(abs(x['m4'] - x['m3']) < 1e-5 for x in sanity['T_full_vs_M3']))
    sanity.update(runtime_s=time.time() - t0, test_rows_used=0, script_sha256=sha(__file__), sites=sites)
    pairs.to_csv(out / 'per_pair_validation.csv', index=False)
    sm.to_csv(out / 'decomposition_summary.csv', index=False)
    fs.to_csv(out / 'features_summary.csv', index=False)
    (out / 'selected_features.json').write_text(json.dumps(feat_json, indent=1, default=float))
    hyp.to_csv(out / 'hypotheses.csv', index=False)
    pd.DataFrame(arch).to_csv(out / 'architecture_summary.csv', index=False)
    (out / 'sanity_checks.json').write_text(json.dumps(sanity, indent=2, default=float))
    np.savez_compressed(out / 'control_draws.npz', **draws_npz)
    print(json.dumps(sanity, indent=1, default=float))
    print(pd.DataFrame(arch).round(3).to_string(index=False))


def pt_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


if __name__ == '__main__':
    main()
