#!/usr/bin/env python3
"""XC-1: reduced positive-control causal site sweep on ORIGINAL Gemma-2-2B (port of scripts/repe/23, M3). 2026-11-23.

Pre-registered in results/cross_model/xc1_gemma_site_sweep/preregistration.json (hashes checked before the full run;
refuses on mismatch or existing outputs). No SAE is used.

Port of scripts/repe/23 (GPT-2 M3) with these declared substitutions:
  model     HF Gemma-2-2B rev c5ebcd40, bf16, eager; site = output of model.model.layers[L] (resid_post).
  grid      REDUCED: L in {0, 2, 5, 8, 12, 16, 19, 23, 25} (9 of 26 layers, depth (L+1)/26 = 0.04 ... 1.0)
            x the same 5 position groups = 45 cells. Trivial sanity cell: (25, final) (= the output itself).
  TRAIN     a fixed seeded subset of 100 TRAIN statements x T1-T4 (<= 400 pairs) for screening and selection;
            random controls on a fixed 30-statement x 4-template subset (<= 120 pairs) of it.
  VALID     all VALIDATION statements x T1-T4 (confirmation of the selected cells; all cells descriptive).
  shapes    every forward pass has the identical shape [NROW, T_FIX] (bf16 kernel selection depends on shape; XC-2
            smoke showed 0.3-nat differences otherwise); continuation log-probs (C_A, secondary) use a second fixed
            shape [NROW, T_FIX_LONG] and are compared only with each other.
  logits    final-token logits = lm_head(final-norm hidden) with the soft-cap in fp32 (as XC-2 / XC-3).
Unchanged from M3: outcomes R_A (primary), R_B, R_S, R_W, matched-norm random, C_A; statistics (per-statement means,
2000-resample statement bootstrap, one-sided normal p); cell p = max(p[R_A>0], p[R_A-R_S>0], p[R_B>0]); Holm over all
cells; PASS = Holm p < 0.05 AND R_A > q95 of 200 random draws AND not trivial; select earliest / peak / latest;
CONFIRMED on validation if R_A CI low > 0; M4 justified iff >= 1 selected cell is confirmed.

Run from the repository root:  ../venv/bin/python scripts/cross_model/24_xc1_gemma_site_sweep.py [--smoke --smoke-out D]
--smoke uses TRAIN rows only (a different TRAIN subset stands in for validation); never reads VALIDATION or TEST.
"""
import argparse
import hashlib
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

OUT = REPO / 'results/cross_model/xc1_gemma_site_sweep'
DATA = REPO / 'data/repe/repe_lm'
FIT = ['T1', 'T2', 'T3', 'T4']
LAYERS = [0, 2, 5, 8, 12, 16, 19, 23, 25]
TRIVIAL = {(25, 'final')}
N_SCREEN, N_FULL, N_BOOT, SEED = 50, 200, 2000, 20261124
N_STMT_SCREEN, N_STMT_RANDOM = 100, 30
NROW = 96
HASHED = ['scripts/cross_model/24_xc1_gemma_site_sweep.py', 'src/evaluation/patching.py', 'src/models/backends.py',
          'data/repe/repe_lm/prompts.csv', 'data/repe/repe_lm/templates.json']
OUTPUT_FILES = ['cells_train.csv', 'cells_validation.csv', 'cells_validation_by_template.csv', 'selected_sites.json',
                'sanity_checks.json', 'random_draws.npz', 'pairs_info.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data
def build_pairs(tok, split, bos):
    d = pd.read_csv(DATA / 'prompts.csv')
    tm = json.loads((DATA / 'templates.json').read_text())
    d = d[(d.split == split) & d.template.isin(FIT)]
    pos = {}
    for t in FIT:
        f = tm[t]['instruction'].format(tm[t]['variants']['H']['word'])
        a = [bos] + tok('USER: ' + f, add_special_tokens=False)['input_ids']
        b = [bos] + tok('USER: ' + f + ' ASSISTANT:', add_special_tokens=False)['input_ids']
        assert b[:len(a)] == a, 'tokenisation of " ASSISTANT:" is not a clean suffix'
        pos[t] = dict(asst_start=len(a), asst_end=len(b) - 1)
    rows, drops = [], dict(length_or_multi_token_swap=0, continuation_not_clean=0)
    for (sid, t), g in d.groupby(['statement_id', 'template']):
        v = g.set_index('variant')
        ids = {k: [bos] + tok(v.loc[k, 'prompt'], add_special_tokens=False)['input_ids'] for k in ('H', 'U', 'Hp')}
        n = len(ids['H'])
        dHU = [i for i in range(n) if len(ids['U']) == n and ids['H'][i] != ids['U'][i]]
        dHP = [i for i in range(n) if len(ids['Hp']) == n and ids['H'][i] != ids['Hp'][i]]
        if not (len(ids['U']) == n == len(ids['Hp']) and len(dHU) == 1 and dHP == dHU):
            drops['length_or_multi_token_swap'] += 1
            continue
        slot = dHU[0]
        assert ids['H'][:pos[t]['asst_end'] + 1][:slot] == ids['U'][:slot]
        st = v.loc['H', 'statement']
        pre = v.loc['H', 'prefix']
        cont_txt = st[len(pre):] if st.startswith(pre) else ''
        cont = tok(cont_txt, add_special_tokens=False)['input_ids'] if cont_txt else []
        full = [bos] + tok(v.loc['H', 'prompt'] + cont_txt, add_special_tokens=False)['input_ids']
        if cont and full != ids['H'] + cont:
            drops['continuation_not_clean'] += 1
            cont = []
        rows.append(dict(statement_id=int(sid), template=t, ids=ids, cont=cont, final=n - 1, n_tokens=n, slot=slot,
                         **pos[t]))
    rows.sort(key=lambda r: (r['template'], r['n_tokens'], r['statement_id']))
    return rows, drops


def subsets(tr, rng):
    stm = sorted({r['statement_id'] for r in tr})
    screen = set(rng.choice(stm, min(N_STMT_SCREEN, len(stm)), replace=False).tolist())
    rand = set(rng.choice(sorted(screen), min(N_STMT_RANDOM, len(screen)), replace=False).tolist())
    rest = [s for s in stm if s not in screen]
    return screen, rand, rest


# ------------------------------------------------------------------------------------------------ engine
class Eng:
    """Gemma forward passes with ONE fixed shape per kind; patch = replace at mask, or add."""

    def __init__(self, t_fix, t_long):
        from src.models.backends import GemmaBackend
        self.bk = GemmaBackend('cuda')
        self.hf = self.bk.model
        self.cap = getattr(self.hf.config, 'final_logit_softcapping', None)
        self.t_fix, self.t_long = t_fix, t_long
        self.n_forward = 0

    def _ids(self, seqs, T):
        ids = torch.zeros(NROW, T, dtype=torch.long)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
        if len(seqs) < NROW:                                   # pad rows: copies of row 0 (results discarded)
            ids[len(seqs):] = ids[0]
        return ids.cuda()

    def _logits(self, h):
        lg = self.hf.lm_head(h).float()
        if self.cap:
            lg = torch.tanh(lg / self.cap) * self.cap
        return lg.double().log_softmax(-1)

    @torch.inference_mode()
    def run(self, seqs, finals, layer=None, mask=None, values=None, add=None, capture=(), long=False, conts=None):
        """seqs <= NROW id lists. mask [n, T] bool, values [n, T, d] (replace at mask), add [n, T, d].
        Returns (final log-probs [n, V] float64, continuation mean log-prob [n] or None, {layer: [n, T, d]})."""
        n = len(seqs)
        assert n <= NROW
        T = self.t_long if long else self.t_fix
        ids = self._ids(seqs, T)
        cap, hs = {}, []

        def pad_n(x):
            if x is None:
                return None
            if x.shape[1] < T:
                x = torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[1]) if x.dim() == 3 else (0, T - x.shape[1]))
            out = torch.zeros((NROW,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
            out[:n] = x
            return out
        mask_, values_, add_ = pad_n(mask), pad_n(values), pad_n(add)
        for L in sorted(set(capture) | ({layer} if layer is not None else set())):
            def hk(mod, inp, out, L=L):
                h = out[0] if isinstance(out, tuple) else out
                if L in capture:
                    cap[L] = h[:n].detach().clone()
                if L == layer:
                    h = h.clone()
                    if values_ is not None:
                        h[mask_] = values_.to(h.dtype)[mask_]
                    if add_ is not None:
                        h = (h.float() + add_.float()).to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return None
            hs.append(self.hf.model.layers[L].register_forward_hook(hk))
        try:
            hid = self.hf.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        finally:
            for x in hs:
                x.remove()
        self.n_forward += 1
        ar = torch.arange(n, device='cuda')
        fin = torch.tensor(finals, device='cuda')
        lp = self._logits(hid[ar, fin])
        cont = None
        if conts is not None:
            cont = torch.full((n,), float('nan'), dtype=torch.float64, device='cuda')
            for i in range(n):
                c = conts[i]
                if c:
                    lg = self._logits(hid[i, finals[i]:finals[i] + len(c)])
                    cont[i] = lg.gather(-1, torch.tensor(c, device='cuda')[:, None]).mean()
        return lp, cont, cap


def masks(rows, T):
    out = {}
    for g in pt.POSITION_GROUPS:
        m = torch.zeros(len(rows), T, dtype=torch.bool)
        for i, r in enumerate(rows):
            p = pt.group_positions(g, r['slot'], r['asst_start'], r['asst_end'], r['final'])
            if p:
                m[i, p] = True
        out[g] = m.cuda()
    return out


def partners(rows):
    keys = [(r['template'], r['n_tokens']) for r in rows]
    out = np.full(len(rows), -1)
    for i in range(len(rows)):
        same = [j for j in range(len(rows)) if keys[j] == keys[i] and rows[j]['statement_id'] != rows[i]['statement_id']]
        if same:
            later = [j for j in same if j > i]
            out[i] = (later or same)[0]
    return out


# ------------------------------------------------------------------------------------------------ sweep
def sweep(eng, rows_all, cells, n_draws, do_controls, gen, rand_ids, log):
    """Per-pair controls per cell (batches of NROW pairs) and per-draw random means over the random subset."""
    recs = []
    rand_sum = {c: np.zeros(n_draws) for c in cells}
    rand_sum_t = {c: {t: np.zeros(n_draws) for t in FIT} for c in cells}
    rand_n = {c: 0 for c in cells}
    rand_n_t = {c: {t: 0 for t in FIT} for c in cells}
    layers = sorted({c[0] for c in cells})
    T = eng.t_fix
    for b0 in range(0, len(rows_all), NROW):
        rows = rows_all[b0:b0 + NROW]
        fin = [r['final'] for r in rows]
        H, U, P = ([r['ids'][k] for r in rows] for k in ('H', 'U', 'Hp'))
        lH, _, cH = eng.run(H, fin, capture=layers)
        lU, _, cU = eng.run(U, fin, capture=layers)
        lP, _, _ = eng.run(P, fin)
        kHU, kUH, kHP = pt.kl(lH, lU), pt.kl(lU, lH), pt.kl(lH, lP)
        mk = masks(rows, T)
        part = partners(rows)
        ok = torch.tensor(part >= 0, device='cuda')
        tmpl = np.array([r['template'] for r in rows])
        sub = np.array([r['statement_id'] in rand_ids for r in rows])
        conts = [r['cont'] for r in rows]
        if do_controls:
            HL, UL = [r['ids']['H'] + r['cont'] for r in rows], [r['ids']['U'] + r['cont'] for r in rows]
            _, ccH, cHL = eng.run(HL, fin, capture=layers, long=True, conts=conts)
            _, ccU, _ = eng.run(UL, fin, long=True, conts=conts)
        for (L, g) in cells:
            m = mk[g]
            has = m.any(1).cpu().numpy()
            rec = dict(statement_id=[r['statement_id'] for r in rows], template=tmpl, layer=L, group=g,
                       kl_HU=kHU.cpu().numpy(), kl_HHp=kHP.cpu().numpy(), kl_UH=kUH.cpu().numpy(), has_positions=has)
            if do_controls:
                lA, _, _ = eng.run(U, fin, L, m, values=cH[L])
                lB, _, _ = eng.run(H, fin, L, m, values=cU[L])
                lS, _, _ = eng.run(P, fin, L, m, values=cH[L])
                rec['R_A'] = (kHU - pt.kl(lH, lA)).cpu().numpy()
                rec['R_B'] = (kUH - pt.kl(lU, lB)).cpu().numpy()
                rec['R_S'] = (kHP - pt.kl(lH, lS)).cpu().numpy()
                wv = cH[L][torch.tensor(np.maximum(part, 0), device='cuda')]
                lW, _, _ = eng.run(U, fin, L, m & ok[:, None], values=wv)
                rw = (kHU - pt.kl(lH, lW)).cpu().numpy()
                rw[part < 0] = np.nan
                rec['R_W'] = rw
                mL = torch.nn.functional.pad(m, (0, eng.t_long - T))
                _, ccA, _ = eng.run(UL, fin, L, mL, values=cHL[L], long=True, conts=conts)
                rec['C_A'] = ((ccH - ccU).abs() - (ccH - ccA).abs()).cpu().numpy()
            for col in ('R_A', 'R_B', 'R_S', 'R_W', 'C_A', 'kl_HU', 'kl_HHp', 'kl_UH'):
                if col in rec:
                    rec[col] = np.where(has, rec[col], np.nan)
            recs.append(pd.DataFrame(rec))
            rand_n[(L, g)] += int((has & sub).sum())
            for t in FIT:
                rand_n_t[(L, g)][t] += int((has & sub & (tmpl == t)).sum())
            si = np.flatnonzero(sub)
            if not len(si) or n_draws == 0:
                continue
            delta = ((cH[L] - cU[L]).float() * m[..., None])[torch.tensor(si, device='cuda')]
            per = max(1, NROW // len(si))
            Us = [U[i] for i in si]
            fs = [fin[i] for i in si]
            for d0 in range(0, n_draws, per):
                r_ = min(per, n_draws - d0)
                add = torch.cat([pt.random_like(delta, gen) for _ in range(r_)])
                lR, _, _ = eng.run(Us * r_, fs * r_, L, add=add)
                R = (kHU[si].repeat(r_) - pt.kl(lH[si].repeat(r_, 1), lR)).view(r_, len(si)).cpu().numpy()
                rand_sum[(L, g)][d0:d0 + r_] += R[:, has[si]].sum(1)
                for t in FIT:
                    rand_sum_t[(L, g)][t][d0:d0 + r_] += R[:, has[si] & (tmpl[si] == t)].sum(1)
        log(f'  pairs {min(b0 + NROW, len(rows_all))}/{len(rows_all)}  forwards {eng.n_forward}')
    df = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    rm = {c: v / max(rand_n[c], 1) for c, v in rand_sum.items()}
    rmt = {c: {t: v / max(rand_n_t[c][t], 1) for t, v in d.items()} for c, d in rand_sum_t.items()}
    return df, rm, rmt


def summarise(df, rm, rmt, rng, by_template=False):
    out = []
    keys = ['layer', 'group'] + (['template'] if by_template else [])
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        L, grp = int(k[0]), k[1]
        r = dict(zip(keys, k))
        r['layer_index'] = LAYERS.index(L)
        r['depth'] = (L + 1) / 26
        r['trivial_sanity_cell'] = (L, grp) in TRIVIAL
        r['n_pairs_with_positions'] = int(g.has_positions.sum())
        sid = g.statement_id.to_numpy()
        for col in ('R_A', 'R_B', 'R_S', 'R_W', 'C_A', 'kl_HU', 'kl_HHp'):
            if col in g:
                s = pt.stat_summary(g[col].to_numpy(), sid, N_BOOT, rng)
                r.update({f'{col}_{kk}': v for kk, v in s.items()})
        if 'R_A' in g:
            s = pt.stat_summary((g.R_A - g.R_S).to_numpy(), sid, N_BOOT, rng)
            r.update({f'R_A_minus_R_S_{kk}': v for kk, v in s.items()})
            r['E_share'] = r['R_A_mean'] / r['kl_HU_mean']
        v = rm[(L, grp)] if not by_template else rmt[(L, grp)][k[2]]
        r['random_mean'], r['random_q95'], r['n_random'] = float(np.mean(v)), float(np.quantile(v, 0.95)), len(v)
        out.append(r)
    return pd.DataFrame(out)


def select(cs):
    cs = cs.copy()
    cs['p_cell'] = cs[['R_A_p', 'R_A_minus_R_S_p', 'R_B_p']].max(1)
    cs['p_holm'] = pt.holm(cs.p_cell.to_numpy())
    cs['passes'] = (cs.p_holm < 0.05) & (cs.R_A_mean > cs.random_q95) & (cs.n_random >= N_FULL) & ~cs.trivial_sanity_cell
    p = cs[cs.passes]
    chosen = []
    if len(p):
        e = p.sort_values(['layer_index', 'R_A_mean'], ascending=[True, False]).iloc[0]
        k = p.sort_values('R_A_mean', ascending=False).iloc[0]
        l_ = p.sort_values(['layer_index', 'R_A_mean'], ascending=[False, False]).iloc[0]
        for role, r in (('earliest', e), ('peak', k), ('latest', l_)):
            key = (int(r.layer), r.group)
            hit = [c for c in chosen if (c['layer'], c['group']) == key]
            if hit:
                hit[0]['role'] += '+' + role
            else:
                chosen.append(dict(role=role, layer=key[0], group=key[1], depth=(key[0] + 1) / 26,
                                   R_A_train=float(r.R_A_mean), E_share=float(r.E_share)))
    return cs, chosen


def sanity(eng, rows):
    rows = rows[:NROW]
    fin = [r['final'] for r in rows]
    H, U = [r['ids']['H'] for r in rows], [r['ids']['U'] for r in rows]
    lH, _, cH = eng.run(H, fin, capture=[0, 12, 23, 25])
    lU, _, cU = eng.run(U, fin, capture=[12])
    kHU = pt.kl(lH, lU)
    real = torch.zeros(len(rows), eng.t_fix, dtype=torch.bool, device='cuda')
    for i, r in enumerate(rows):
        real[i, :r['n_tokens']] = True
    out = {}
    for L in (0, 12, 23):
        lA, _, _ = eng.run(U, fin, L, real, values=cH[L])
        out[f'all_positions_L{L}_max_kl_to_H'] = float(pt.kl(lH, lA).max())
    lZ, _, _ = eng.run(U, fin, 12, real, values=cU[12])
    out['zero_patch_max_abs_R'] = float((kHU - pt.kl(lH, lZ)).abs().max())
    m = masks(rows, eng.t_fix)['final']
    lT, _, _ = eng.run(U, fin, 25, m, values=cH[25])
    out['(25,final)_max_abs(R_A-kl_HU)'] = float(((kHU - pt.kl(lH, lT)) - kHU).abs().max())
    perm = list(range(len(rows)))[::-1]                     # the same rows in another batch position
    lHp, _, _ = eng.run([H[i] for i in perm], [fin[i] for i in perm])
    out['row_position_max_abs_logprob_diff'] = float((lHp - lH[perm]).abs().max())
    add0 = torch.zeros(len(rows), eng.t_fix, eng.bk.d_model, device='cuda')
    lA0, _, _ = eng.run(U, fin, 12, add=add0)
    out['zero_add_max_abs_logprob_diff'] = float((lA0 - lU).abs().max())
    out['pass'] = all(v < 1e-6 for k, v in out.items() if not k.startswith('all_positions')) and \
        all(v < 1e-5 for k, v in out.items() if k.startswith('all_positions'))
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    global N_SCREEN, N_FULL, N_BOOT
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
    tr_all, drop_tr = build_pairs(tok, 'train', 2)
    screen, rand_ids, rest = subsets(tr_all, rng)
    tr = [r for r in tr_all if r['statement_id'] in screen]
    if a.smoke:                                             # TRAIN only: validation stand-in = other TRAIN statements
        va, drop_va = [r for r in tr_all if r['statement_id'] in set(rest[:12])], {}
        tr = [r for r in tr if r['statement_id'] in set(sorted(screen)[:16])]
        rand_ids = {s for s in rand_ids if s in set(sorted(screen)[:16])} or set(sorted(screen)[:4])
        N_SCREEN, N_FULL, N_BOOT = 4, 8, 200
    else:
        va, drop_va = build_pairs(tok, 'validation', 2)
    allr = tr + va
    t_fix = max(r['n_tokens'] for r in allr)
    t_long = max(r['n_tokens'] + len(r['cont']) for r in allr)
    info = dict(n_train_pairs=len(tr), n_train_statements=len({r['statement_id'] for r in tr}),
                n_random_statements=len(rand_ids), n_validation_pairs=len(va),
                n_validation_statements=len({r['statement_id'] for r in va}), drops_train=drop_tr,
                drops_validation=drop_va, t_fix=t_fix, t_long=t_long, smoke=a.smoke,
                screen_statement_ids=sorted(int(s) for s in screen), random_statement_ids=sorted(int(s) for s in rand_ids))
    (out / 'pairs_info.json').write_text(json.dumps(info, indent=1))
    log(json.dumps({k: v for k, v in info.items() if 'ids' not in k}))
    eng = Eng(t_fix, t_long)
    gen = torch.Generator(device='cuda').manual_seed(SEED)
    checks = sanity(eng, tr)
    (out / 'sanity_checks.json').write_text(json.dumps(checks, indent=2))
    log(f'sanity {checks}')
    if not checks['pass']:
        raise SystemExit('sanity checks failed; stopping (hard gate).')
    cells = [(L, g) for L in LAYERS for g in pt.POSITION_GROUPS]
    log(f'screening {len(cells)} cells on {len(tr)} train pairs, {N_SCREEN} random draws')
    df_tr, rm, rmt = sweep(eng, tr, cells, N_SCREEN, True, gen, rand_ids, log)
    cs = summarise(df_tr, rm, rmt, rng)
    cand = cs[(cs.R_A_ci_low > 0) & (cs.R_A_mean > cs.random_q95) & ~cs.trivial_sanity_cell]
    cand_cells = [(int(L), g) for L, g in zip(cand.layer, cand.group)]
    log(f'{len(cand_cells)} candidates; {N_FULL - N_SCREEN} more random draws ({time.time() - t0:.0f} s)')
    draws = {f'L{L}|{g}|screen': v for (L, g), v in rm.items()}
    if cand_cells:
        _, rm2, rmt2 = sweep(eng, tr, cand_cells, N_FULL - N_SCREEN, False, gen, rand_ids, log)
        for c in cand_cells:
            rm[c] = np.concatenate([rm[c], rm2[c]])
            for t in FIT:
                rmt[c][t] = np.concatenate([rmt[c][t], rmt2[c][t]])
            draws[f'L{c[0]}|{c[1]}|extra'] = rm2[c]
    cs = summarise(df_tr, rm, rmt, rng)
    cs, chosen = select(cs)
    cs.to_csv(out / 'cells_train.csv', index=False)
    log(f'selected {chosen} ({time.time() - t0:.0f} s)')
    df_va, rmv0, rmtv0 = sweep(eng, va, cells, 1, True, gen, set(), log)       # all cells, controls only
    sel = [(c['layer'], c['group']) for c in chosen]
    va_ids = {r['statement_id'] for r in va}
    if sel:
        _, rmv, rmtv = sweep(eng, va, sel, N_FULL, False, gen, va_ids, log)
        for c in sel:
            rmv0[c], rmtv0[c] = rmv[c], rmtv[c]
    cv = summarise(df_va, rmv0, rmtv0, np.random.default_rng(SEED + 2))
    cv['random_controls_valid'] = [(int(L), g) in sel for L, g in zip(cv.layer, cv.group)]
    cv.to_csv(out / 'cells_validation.csv', index=False)
    summarise(df_va, rmv0, rmtv0, np.random.default_rng(SEED + 3), by_template=True).to_csv(
        out / 'cells_validation_by_template.csv', index=False)
    for c in chosen:
        v = cv[(cv.layer == c['layer']) & (cv.group == c['group'])].iloc[0]
        c.update(R_A_validation=float(v.R_A_mean), R_A_validation_ci=[float(v.R_A_ci_low), float(v.R_A_ci_high)],
                 R_B_validation=float(v.R_B_mean), R_S_validation=float(v.R_S_mean),
                 E_share_validation=float(v.E_share), random_q95_validation=float(v.random_q95),
                 confirmed=bool(v.R_A_ci_low > 0))
    res = dict(selected=chosen, n_passing_cells=int(cs.passes.sum()), n_candidates=len(cand_cells),
               M4_justified=any(c['confirmed'] for c in chosen),
               rule='Holm(max p of R_A>0, R_A-R_S>0, R_B>0) < 0.05 over 45 cells AND R_A > q95 of 200 matched-norm '
                    'random draws AND not a trivial sanity cell; earliest/peak/latest; confirmed on validation if '
                    'R_A CI low > 0',
               runtime_s=time.time() - t0, n_forward=eng.n_forward, script_sha256=sha(__file__),
               sanity_pass=checks['pass'], test_rows_used=0, finished_utc=datetime.now(timezone.utc).isoformat(),
               torch=torch.__version__, tf32=False, **{k: v for k, v in info.items() if 'ids' not in k})
    np.savez_compressed(out / 'random_draws.npz', **{k.replace('|', '__'): v for k, v in draws.items()})
    (out / 'selected_sites.json').write_text(json.dumps(res, indent=2, default=float))
    log(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
