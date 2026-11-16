#!/usr/bin/env python3
"""M3: where does ORIGINAL GPT-2 Small carry the effect of the corrected persona instruction (RepE-LM T1-T4)?

Pre-registered in results/repe/causal_site_sweep/preregistration.json. No SAE is used.

Sites: blocks.L.hook_resid_pre (L = 0..11) and blocks.11.hook_resid_post  x  position groups
       {slot, post_slot_instruction, assistant_span, prefix_excl_final, final}  = 65 cells.
Per (statement, template) pair, activation patching at one cell:
  R_A  H -> U   KL(p_H||p_U) - KL(p_H||p_{U<-H})        (restoration toward the honest output; primary, nats)
  R_B  U -> H   KL(p_U||p_H) - KL(p_U||p_{H<-U})        (other direction)
  R_S  synonym  KL(p_H||p_Hp) - KL(p_H||p_{Hp<-H})      (same-meaning one-token change at the same cell)
  R_W  wrong statement: U <- H of ANOTHER statement (same template and token length)
  R_rand matched-norm random: U + eta, ||eta|| = ||x_H - x_U|| per patched position (50 draws screening, 200 for
         candidates)
  C_A  continuation: |lp_H - lp_U| - |lp_H - lp_{U<-H}| (mean log-prob per continuation token)
p_H etc. are next-token distributions at the final prompt token.
TRAIN statements: screening and selection. VALIDATION: confirmation of selected sites (+ descriptive all cells).

Run from repository root:  python scripts/repe/23_positive_control_site_sweep.py [--smoke --smoke-out DIR]
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
from src.evaluation import patching as pt  # noqa: E402

OUT = REPO / 'results/repe/causal_site_sweep'
DATA = REPO / 'data/repe/repe_lm'
FIT = ['T1', 'T2', 'T3', 'T4']
SITES = [(f'blocks.{L}.hook_resid_pre', L) for L in range(12)] + [('blocks.11.hook_resid_post', 12)]
TRIVIAL = {('blocks.0.hook_resid_pre', 'slot'), ('blocks.11.hook_resid_post', 'final')}   # sanity checks only
N_SCREEN, N_FULL, N_BOOT, SEED, B, CHUNK = 50, 200, 2000, 20261108, 96, 10
N_RANDOM_PAIRS_TRAIN = 480   # fixed random subset (stratified by template) for the random controls: conservative
OUTPUT_FILES = ['cells_train.csv', 'cells_train_by_template.csv', 'cells_validation.csv',
                'cells_validation_by_template.csv', 'selected_sites.json', 'sanity_checks.json', 'random_draws.npz']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data

def build_pairs(tok, split):
    d = pd.read_csv(DATA / 'prompts.csv')
    assert set(d.split) <= {'train', 'validation'}
    tm = json.loads((DATA / 'templates.json').read_text())
    d = d[(d.split == split) & d.template.isin(FIT)]
    pos = {t: pt.template_positions(tok, tm[t]['instruction'].format(tm[t]['variants']['H']['word']), tm[t]['slot_pos'])
           for t in FIT}
    rows = []
    for (sid, t), g in d.groupby(['statement_id', 'template']):
        v = g.set_index('variant')
        ids = {k: list(map(int, v.loc[k, 'token_ids'].split())) for k in ('H', 'U', 'Hp')}
        p = pos[t]
        assert ids['H'][:p['asst_end'] + 1] == ([tok.bos_token_id] + tok.encode(
            'USER: ' + tm[t]['instruction'].format(tm[t]['variants']['H']['word']) + ' ASSISTANT:')), 'span mismatch'
        cont = v.loc['H', 'continuation_ids']
        cont = list(map(int, cont.split())) if isinstance(cont, str) else []
        rows.append(dict(statement_id=sid, template=t, ids=ids, cont=cont, final=int(v.loc['H', 'final_pos']),
                         n_tokens=int(v.loc['H', 'n_tokens']), **p))
    rows.sort(key=lambda r: (r['template'], r['n_tokens'], r['statement_id']))
    rng = np.random.default_rng(SEED + (0 if split == 'train' else 1))
    for t in FIT:
        ix = [i for i, r in enumerate(rows) if r['template'] == t]
        k = len(ix) if split != 'train' else min(len(ix), N_RANDOM_PAIRS_TRAIN // len(FIT))
        chosen = set(rng.choice(ix, k, replace=False).tolist())
        for i in ix:
            rows[i]['random_subset'] = i in chosen
    return rows


def batch_tensors(rows, device):
    T = max(r['n_tokens'] + len(r['cont']) for r in rows)
    toks = {k: torch.full((len(rows), T), 50256, dtype=torch.long) for k in ('H', 'U', 'Hp')}
    for i, r in enumerate(rows):
        for k in toks:
            s = r['ids'][k] + r['cont']
            toks[k][i, :len(s)] = torch.tensor(s)
    # wrong-statement partner: next row in the batch with the same template and token length, different statement
    partner = np.full(len(rows), -1)
    keys = [(r['template'], r['n_tokens']) for r in rows]
    for i in range(len(rows)):
        same = [j for j in range(len(rows)) if keys[j] == keys[i] and rows[j]['statement_id'] != rows[i]['statement_id']]
        if same:
            later = [j for j in same if j > i]
            partner[i] = (later or same)[0]
    return {k: v.to(device) for k, v in toks.items()}, torch.tensor(partner, device=device), T


def masks(rows, T, device):
    out = {}
    for gname in pt.POSITION_GROUPS:
        m = torch.zeros(len(rows), T, dtype=torch.bool)
        for i, r in enumerate(rows):
            p = pt.group_positions(gname, r['slot'], r['asst_start'], r['asst_end'], r['final'])
            if p:
                m[i, p] = True
        out[gname] = m.to(device)
    return out


# --------------------------------------------------------------------------------------------- model

class Runner:
    def __init__(self, model):
        self.m = model

    @torch.inference_mode()
    def run(self, toks, rows, site=None, mask=None, values=None, add=None, cache_sites=None, reps=1, want_cont=False):
        """Forward; returns (final log-probs [n, V], continuation mean log-prob [n], cache)."""
        captured = {}
        hooks = [('ln_final.hook_normalized', lambda x, hook: captured.__setitem__('h', x))]
        if site is not None:
            def patch(resid, hook):
                if values is not None:
                    resid[mask] = values[mask]
                else:
                    resid = resid + add
                return resid
            hooks.append((site, patch))
        cache = {}
        if cache_sites:
            for s in cache_sites:
                hooks.append((s, lambda x, hook, s=s: cache.__setitem__(s, x.clone())))
        self.m.run_with_hooks(toks, fwd_hooks=hooks, return_type=None)
        h = captured['h']
        n = h.shape[0]
        fin = torch.tensor([r['final'] for r in rows] * reps, device=h.device)
        ar = torch.arange(n, device=h.device)
        lp_final = self.m.unembed(h[ar, fin][:, None])[:, 0].double().log_softmax(-1)
        cont = torch.full((n,), float('nan'), dtype=torch.float64, device=h.device)
        for i, r in enumerate(rows * reps if want_cont else []):
            c = len(r['cont'])
            if c:
                lg = self.m.unembed(h[i, r['final']:r['final'] + c][None])[0].double().log_softmax(-1)
                tgt = toks[i, r['final'] + 1:r['final'] + 1 + c]
                cont[i] = lg.gather(-1, tgt[:, None]).mean()
        return lp_final, cont, cache


# ------------------------------------------------------------------------------------------- sweeps

def sweep(runner, rows_all, device, cells, n_draws, do_controls, gen):
    """Returns per-pair DataFrame of controls per cell and per-draw random sums."""
    recs = []
    rand_sum = {c: np.zeros(n_draws) for c in cells}
    rand_sum_t = {c: {t: np.zeros(n_draws) for t in FIT} for c in cells}
    rand_n = {c: 0 for c in cells}
    rand_n_t = {c: {t: 0 for t in FIT} for c in cells}
    sites = sorted({c[0] for c in cells}, key=lambda s: [x[0] for x in SITES].index(s))
    for b0 in range(0, len(rows_all), B):
        rows = rows_all[b0:b0 + B]
        toks, partner, T = batch_tensors(rows, device)
        mk = masks(rows, T, device)
        lH, cH, cacheH = runner.run(toks['H'], rows, cache_sites=sites, want_cont=True)
        lU, cU, cacheU = runner.run(toks['U'], rows, cache_sites=sites, want_cont=True)
        lP, cP, _ = runner.run(toks['Hp'], rows)
        kHU, kUH, kHP = pt.kl(lH, lU), pt.kl(lU, lH), pt.kl(lH, lP)
        tmpl = np.array([r['template'] for r in rows])
        for (site, g) in cells:
            m = mk[g]
            rec = dict(statement_id=[r['statement_id'] for r in rows], template=tmpl, site=site, group=g,
                       kl_HU=kHU.cpu().numpy(), kl_HHp=kHP.cpu().numpy(), kl_UH=kUH.cpu().numpy(),
                       has_positions=m.any(1).cpu().numpy())
            has = m.any(1).cpu().numpy()
            if do_controls:
                lA, cA, _ = runner.run(toks['U'], rows, site, m, values=cacheH[site], want_cont=True)
                lB, _, _ = runner.run(toks['H'], rows, site, m, values=cacheU[site])
                lS, _, _ = runner.run(toks['Hp'], rows, site, m, values=cacheH[site])
                rec['R_A'] = (kHU - pt.kl(lH, lA)).cpu().numpy()
                rec['R_B'] = (kUH - pt.kl(lU, lB)).cpu().numpy()
                rec['R_S'] = (kHP - pt.kl(lH, lS)).cpu().numpy()
                rec['C_A'] = ((cH - cU).abs() - (cH - cA).abs()).cpu().numpy()
                ok = partner >= 0
                wvals = cacheH[site][partner.clamp(min=0)]
                mw = m & ok[:, None]
                lW, _, _ = runner.run(toks['U'], rows, site, mw, values=wvals)
                rw = (kHU - pt.kl(lH, lW)).cpu().numpy()
                rw[~ok.cpu().numpy()] = np.nan
                rec['R_W'] = rw
            for col in ('R_A', 'R_B', 'R_S', 'R_W', 'C_A', 'kl_HU', 'kl_HHp', 'kl_UH'):
                if col in rec:
                    rec[col] = np.where(has, rec[col], np.nan)
            sub_ = np.array([r['random_subset'] for r in rows])
            rand_n[(site, g)] += int((has & sub_).sum())
            for t in FIT:
                rand_n_t[(site, g)][t] += int((has & sub_ & (tmpl == t)).sum())
            # matched-norm random, in chunks of CHUNK draws stacked on the batch axis
            recs.append(pd.DataFrame(rec))
            sub = np.array([r['random_subset'] for r in rows])
            if not sub.any():
                continue
            si = torch.tensor(np.flatnonzero(sub), device=device)
            rows_r, has_r, tm_r = [rows[i] for i in np.flatnonzero(sub)], has[sub], tmpl[sub]
            delta = ((cacheH[site] - cacheU[site]) * m[..., None])[si]
            for d0 in range(0, n_draws, CHUNK):
                r_ = min(CHUNK, n_draws - d0)
                add = torch.cat([pt.random_like(delta, gen) for _ in range(r_)])
                lR, _, _ = runner.run(toks['U'][si].repeat(r_, 1), rows_r, site, None, add=add, reps=r_)
                R = (kHU[si].repeat(r_) - pt.kl(lH[si].repeat(r_, 1), lR)).view(r_, len(rows_r)).cpu().numpy()
                rand_sum[(site, g)][d0:d0 + r_] += R[:, has_r].sum(1)
                for t in FIT:
                    rand_sum_t[(site, g)][t][d0:d0 + r_] += R[:, has_r & (tm_r == t)].sum(1)
        print(f'  pairs {min(b0 + B, len(rows_all))}/{len(rows_all)}', flush=True)
    df = pd.concat(recs, ignore_index=True)
    rand_mean = {c: v / max(rand_n[c], 1) for c, v in rand_sum.items()}
    rand_mean_t = {c: {t: v / max(rand_n_t[c][t], 1) for t, v in d.items()} for c, d in rand_sum_t.items()}
    return df, rand_mean, rand_mean_t


def summarise(df, rand_mean, rand_mean_t, rng, by_template=False):
    out = []
    keys = ['site', 'group'] + (['template'] if by_template else [])
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        site, grp = k[0], k[1]
        r = dict(zip(keys, k))
        r['layer_index'] = [s for s, _ in SITES].index(site)
        r['trivial_sanity_cell'] = (site, grp) in TRIVIAL
        r['n_pairs_with_positions'] = int(g.has_positions.sum())
        sid = g.statement_id.to_numpy()
        for col in ('R_A', 'R_B', 'R_S', 'R_W', 'C_A', 'kl_HU', 'kl_HHp'):
            if col in g:
                s = pt.stat_summary(g[col].to_numpy(), sid, N_BOOT, rng)
                for kk, v in s.items():
                    r[f'{col}_{kk}'] = v
        if 'R_A' in g:
            s = pt.stat_summary((g.R_A - g.R_S).to_numpy(), sid, N_BOOT, rng)
            r.update({f'R_A_minus_R_S_{kk}': v for kk, v in s.items()})
            r['E_share'] = r['R_A_mean'] / r['kl_HU_mean']
        rm = rand_mean[(site, grp)] if not by_template else rand_mean_t[(site, grp)][k[2]]
        r['random_mean'], r['random_q95'], r['n_random'] = float(np.mean(rm)), float(np.quantile(rm, 0.95)), len(rm)
        out.append(r)
    return pd.DataFrame(out)


def sanity(runner, rows, device):
    rows = rows[:B]
    toks, _, T = batch_tensors(rows, device)
    lH, _, _ = runner.run(toks['H'], rows)
    lU, _, cacheU = runner.run(toks['U'], rows, cache_sites=['blocks.6.hook_resid_pre'])
    kHU = pt.kl(lH, lU)
    out = {}
    real = torch.zeros(len(rows), T, dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        real[i, :r['n_tokens'] + len(r['cont'])] = True
    for L in (0, 6, 11):
        site = f'blocks.{L}.hook_resid_pre'
        _, _, cH = runner.run(toks['H'], rows, cache_sites=[site])
        lA, _, _ = runner.run(toks['U'], rows, site, real, values=cH[site])
        out[f'all_positions_L{L}_max_kl_to_H'] = float(pt.kl(lH, lA).max())
    lZ, _, _ = runner.run(toks['U'], rows, 'blocks.6.hook_resid_pre', real, values=cacheU['blocks.6.hook_resid_pre'])
    out['zero_patch_max_abs_R'] = float((kHU - pt.kl(lH, lZ)).abs().max())
    mk = masks(rows, T, device)
    for site, g in TRIVIAL:
        _, _, cH = runner.run(toks['H'], rows, cache_sites=[site])
        lA, _, _ = runner.run(toks['U'], rows, site, mk[g], values=cH[site])
        out[f'{site}|{g}_max_abs(R_A-kl_HU)'] = float(((kHU - pt.kl(lH, lA)) - kHU).abs().max())
    out['pass'] = all(v < 1e-5 for v in out.values())
    return out


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
            key = (r.site, r.group)
            if key not in [(c['site'], c['group']) for c in chosen]:
                chosen.append(dict(role=role, site=r.site, group=r.group, R_A_train=r.R_A_mean, E_share=r.E_share))
            else:
                [c for c in chosen if (c['site'], c['group']) == key][0]['role'] += '+' + role
    return cs, chosen


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    out = a.smoke_out if a.smoke else OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel in ('scripts/repe/23_positive_control_site_sweep.py', 'src/evaluation/patching.py'):
            if pr['hashes'].get(rel) != sha(REPO / rel):
                raise SystemExit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = 'cuda'
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    runner = Runner(model)
    gen = torch.Generator(device=device).manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained('gpt2')      # same tokenizer as script 20 (model.tokenizer adds BOS on encode)
    tr = build_pairs(tok, 'train')
    va = build_pairs(tok, 'validation')
    if a.smoke:
        global N_SCREEN, N_FULL, N_BOOT
        tr, va = tr[::12], va[::12]
        N_SCREEN, N_FULL, N_BOOT = 4, 8, 200
    cells = [(s, g) for s, _ in SITES for g in pt.POSITION_GROUPS]
    checks = sanity(runner, tr, device)
    (out / 'sanity_checks.json').write_text(json.dumps(checks, indent=2))
    print('sanity', checks, flush=True)

    print(f'screening {len(cells)} cells on {len(tr)} train pairs, {N_SCREEN} random draws', flush=True)
    df_tr, rm, rmt = sweep(runner, tr, device, cells, N_SCREEN, True, gen)
    cs = summarise(df_tr, rm, rmt, rng)
    cand = cs[(cs.R_A_ci_low > 0) & (cs.R_A_mean > cs.random_q95) & ~cs.trivial_sanity_cell]
    cand_cells = list(zip(cand.site, cand.group))
    print(f'{len(cand_cells)} candidates; {N_FULL - N_SCREEN} more random draws', flush=True)
    draws = {f'{s}|{g}|screen': v for (s, g), v in rm.items()}
    if cand_cells:
        _, rm2, rmt2 = sweep(runner, tr, device, cand_cells, N_FULL - N_SCREEN, False, gen)
        for c in cand_cells:
            rm[c] = np.concatenate([rm[c], rm2[c]])
            for t in FIT:
                rmt[c][t] = np.concatenate([rmt[c][t], rmt2[c][t]])
            draws[f'{c[0]}|{c[1]}|extra'] = rm2[c]
    cs = summarise(df_tr, rm, rmt, rng)
    cs, chosen = select(cs)
    cs.to_csv(out / 'cells_train.csv', index=False)
    ct = summarise(df_tr, rm, rmt, np.random.default_rng(SEED + 1), by_template=True)
    ct.to_csv(out / 'cells_train_by_template.csv', index=False)

    print(f'validation: all cells (controls), {len(chosen)} selected sites with {N_FULL} random draws', flush=True)
    # all cells, controls only (no random) for description
    df_va, rmv0, rmtv0 = sweep(runner, va, device, cells, 1, True, gen)
    sel_cells = [(c['site'], c['group']) for c in chosen]
    if sel_cells:
        _, rmv, rmtv = sweep(runner, va, device, sel_cells, N_FULL, False, gen)
        for c in sel_cells:
            rmv0[c], rmtv0[c] = rmv[c], rmtv[c]
    cv = summarise(df_va, rmv0, rmtv0, np.random.default_rng(SEED + 2))
    cv['random_controls_valid'] = [(s, g) in sel_cells for s, g in zip(cv.site, cv.group)]
    cv.to_csv(out / 'cells_validation.csv', index=False)
    cvt = summarise(df_va, rmv0, rmtv0, np.random.default_rng(SEED + 3), by_template=True)
    cvt.to_csv(out / 'cells_validation_by_template.csv', index=False)
    for c in chosen:
        v = cv[(cv.site == c['site']) & (cv.group == c['group'])].iloc[0]
        c.update(R_A_validation=v.R_A_mean, R_A_validation_ci=[v.R_A_ci_low, v.R_A_ci_high],
                 R_B_validation=v.R_B_mean, R_S_validation=v.R_S_mean, E_share_validation=v.E_share,
                 random_q95_validation=v.random_q95, confirmed=bool(v.R_A_ci_low > 0))
    res = dict(selected=chosen, n_passing_cells=int(cs.passes.sum()), n_candidates=len(cand_cells),
               M4_justified=any(c['confirmed'] for c in chosen),
               rule='Holm(max p of R_A>0, R_A-R_S>0, R_B>0) < 0.05 over 65 cells AND R_A > q95 of 200 matched-norm '
                    'random draws AND not a trivial sanity cell; earliest/peak/latest; confirmed on validation if '
                    'R_A CI low > 0',
               n_train_pairs=len(tr), n_validation_pairs=len(va), runtime_s=time.time() - t0,
               script_sha256=sha(__file__), sanity_pass=checks['pass'], test_rows_used=0)
    (out / 'selected_sites.json').write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(out / 'random_draws.npz', **{k.replace('.', '_').replace('|', '__'): v for k, v in draws.items()})
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
