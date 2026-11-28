#!/usr/bin/env python3
"""XC-3: steering parity. ONE persona-steering protocol for GPT-2 Small and Gemma-2-2B (2026-11-23).

Pre-registered in results/cross_model/xc3_steering_parity/preregistration.json (hashes of this script and the data
are checked before any non-smoke phase; the run refuses on mismatch).

Task     RepE-LM fit templates T1-T4, honest (H) vs untruthful (U) persona prompts (one-token swaps, gate K1).
Sites    depth-matched triplet T-B: GPT-2 blocks.L.hook_resid_pre L = 3/6/9  <->  Gemma resid_post L = 5/12/19.
         The FINAL prompt token only (where the persona effect is read out; M3 site for GPT-2 L11 'final').
Outcome  E(v) = 1 - sum_i KL(p_H,i || p_{U+v},i) / sum_i KL(p_H,i || p_U,i), p = next-token distribution at the final
         prompt token. E = 0: no change; E = 1: output equals the honest-persona output. Within-model normalised.
Arms     raw  : u_raw = unit(mean_H - mean_U) of ORIGINAL final-token activations (TRAIN, T1-T4, cached).
         rec  : u_rec = unit(mean SAE(x_H) - mean SAE(x_U)) (TRAIN), SAE = res-jb (GPT-2), Gemma Scope 16k canonical.
         patch: per-pair positive control, x_U <- x_H at the site (no norm constraint; ceiling).
Strength s = c * ||mean_H - mean_U||_TRAIN, c in GRID, chosen on VALIDATION as argmax E_raw per (model, layer);
         the same s is used for rec and every control (norm-matched).
Controls (at s) 200 Gaussian random unit directions; 200 random single alive SAE decoder columns (random sign).
Phases   --phase validation : gates, strength selection, descriptive controls -> validation outputs + selection.json
         --phase test       : one pass on RepE-LM TEST T1-T4 with the frozen selection -> test outputs.
         --smoke            : TRAIN rows only, tiny, written to --smoke-out (never VALIDATION/TEST).
Run from the repository root with ../venv/bin/python.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / 'results/cross_model/xc3_steering_parity'
PROMPTS = {'train': REPO / 'data/repe/repe_lm/prompts.csv', 'validation': REPO / 'data/repe/repe_lm/prompts.csv',
           'test': REPO / 'data/repe/repe_lm_test/prompts.csv'}
CACHE = {'gpt2': REPO / 'data/repe/activations/repe_lm/final_layer{L}.npy',
         'gemma': REPO / 'data/gemma/activations/repe_lm/final_layer{L}.npy'}
LAYERS = {'gpt2': [3, 6, 9], 'gemma': [5, 12, 19]}
FIT = ['T1', 'T2', 'T3', 'T4']
GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # = scripts/repe/13 STRENGTH_GRID (parity)
N_CTRL, SEED, N_BOOT, RETAIN = 200, 20261123, 2000, 0.8
BATCH = {'gpt2': 256, 'gemma': 192}
HASHED = ['scripts/cross_model/20_xc3_steering_parity.py', 'scripts/cross_model/02_matched_splicing.py',
          'src/evaluation/pile_splicing.py', 'src/models/backends.py', 'scripts/shared/sae_ladder_evaluation.py',
          'data/repe/repe_lm/prompts.csv', 'data/repe/repe_lm_test/prompts.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------- data
def pairs(model, split, tok_fn, bos):
    d = pd.read_csv(PROMPTS[split])
    d = d[(d.split == split) & d.template.isin(FIT) & d.variant.isin(['H', 'U'])]
    out, dropped = [], 0
    for (sid, t), g in d.groupby(['statement_id', 'template'], sort=True):
        v = g.set_index('variant')
        if model == 'gpt2':
            ids = {k: list(map(int, v.loc[k, 'token_ids'].split())) for k in ('H', 'U')}
            for k in ids:
                assert ids[k] == [bos] + tok_fn(v.loc[k, 'prompt']), 'GPT-2 stored ids differ from tokenizer'
        else:
            ids = {k: [bos] + tok_fn(v.loc[k, 'prompt']) for k in ('H', 'U')}
        n = len(ids['H'])
        diff = [i for i in range(min(n, len(ids['U']))) if ids['H'][i] != ids['U'][i]]
        if len(ids['U']) != n or len(diff) != 1:        # K1 under this model's tokenizer
            dropped += 1
            continue
        out.append(dict(statement_id=str(sid), template=t, H=ids['H'], U=ids['U'], n=n))
    return out, dropped


def train_rows(split_df_path=PROMPTS['train']):
    d = pd.read_csv(split_df_path)
    m = (d.split == 'train') & d.template.isin(FIT)
    return np.flatnonzero(m & (d.variant == 'H')), np.flatnonzero(m & (d.variant == 'U'))


# ------------------------------------------------------------------------------------------------ models
class Runner:
    """Next-token log-probs at each row's final token, with an optional additive edit at (layer, final token)."""

    def __init__(self, model_name, device='cuda'):
        self.name, self.device = model_name, device
        bk = _mod('src/models/backends.py', 'backends').make_backend(model_name, device)
        self.bk = bk
        if model_name == 'gpt2':
            self.tl = bk.model
            self.pad = 50256
        else:
            self.hf = bk.model
            self.cap = getattr(self.hf.config, 'final_logit_softcapping', None)
            self.pad = 0

    @torch.inference_mode()
    def logprobs(self, seqs, layer=None, add=None, replace=None, exact_hf=False):
        """seqs: list of id lists. add: [B, d] float vectors added at the final token of `layer`.
        replace: [B, d] values written at the final token. Returns float64 log-probs [B, V]."""
        B, T = len(seqs), max(map(len, seqs))
        ids = torch.full((B, T), self.pad, dtype=torch.long)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
        ids = ids.to(self.device)
        fin = torch.tensor([len(q) - 1 for q in seqs], device=self.device)
        ar = torch.arange(B, device=self.device)

        def edit(h):
            h = h.clone()
            if add is not None:
                h[ar, fin] = (h[ar, fin].float() + add).to(h.dtype)
            if replace is not None:
                h[ar, fin] = replace.to(h.dtype)
            return h
        if self.name == 'gpt2':
            hooks = [] if layer is None else [(f'blocks.{layer}.hook_resid_pre', lambda x, hook: edit(x))]
            lg = self.tl.run_with_hooks(ids, fwd_hooks=hooks)[ar, fin]
        else:
            hd = None
            if layer is not None:
                def hk(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = edit(h)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hd = self.hf.model.layers[layer].register_forward_hook(hk)
            try:   # right padding + causal attention: real positions never attend to padding
                hs = self.hf.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
            finally:
                if hd is not None:
                    hd.remove()
            if exact_hf:   # HF's own order: soft-cap in the model dtype (bf16); used only by the path sanity check
                lg = self.hf.lm_head(hs[ar, fin])
                if self.cap:
                    lg = torch.tanh(lg / self.cap) * self.cap
                lg = lg.float()
            else:          # outcome path: soft-cap in fp32 (more precise than HF's bf16 soft-cap)
                lg = self.hf.lm_head(hs[ar, fin]).float()
                if self.cap:
                    lg = torch.tanh(lg / self.cap) * self.cap
        return lg.double().log_softmax(-1)

    @torch.inference_mode()
    def capture(self, seqs, layer):
        """Site activation at each row's final token (float32 [B, d])."""
        got = {}
        B, T = len(seqs), max(map(len, seqs))
        ids = torch.full((B, T), self.pad, dtype=torch.long)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
        ids = ids.to(self.device)
        fin = torch.tensor([len(q) - 1 for q in seqs], device=self.device)
        ar = torch.arange(B, device=self.device)
        if self.name == 'gpt2':
            def h(x, hook):
                got['x'] = x[ar, fin].float().clone()
            self.tl.run_with_hooks(ids, fwd_hooks=[(f'blocks.{layer}.hook_resid_pre', h)], return_type=None)
        else:
            def hk(mod, inp, out):
                got['x'] = (out[0] if isinstance(out, tuple) else out)[ar, fin].float().clone()
            hd = self.hf.model.layers[layer].register_forward_hook(hk)
            try:
                self.hf.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            finally:
                hd.remove()
        return got['x']


def kl(lp, lq):
    """KL(p || q) per row, float64."""
    return (lp.exp() * (lp - lq)).sum(-1)


def batched(xs, b):
    for i in range(0, len(xs), b):
        yield i, xs[i:i + b]


# ---------------------------------------------------------------------------------------------- stats
def boot_E(klH_U, klH_X, groups, rng, n=N_BOOT):
    """Statement-clustered bootstrap of E = 1 - sum KL(H||X)/sum KL(H||U)."""
    ug, inv = np.unique(groups, return_inverse=True)
    a = np.bincount(inv, klH_U, len(ug))
    b = np.bincount(inv, klH_X, len(ug))
    idx = rng.integers(0, len(ug), (n, len(ug)))
    return 1 - b[idx].sum(1) / a[idx].sum(1)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


# ----------------------------------------------------------------------------------------------- main
def sae_for(model, L, device):
    xb = _mod('scripts/cross_model/02_matched_splicing.py', 'xb02')
    which = ['canonical'] if model == 'gemma' else ['resjb']
    spec = xb.sae_specs(model, L, which)
    spec = [s for s in spec if (s['family'] == 'gemmascope_16k_canonical' or s['family'] == 'resjb')][0]
    spec['layer'] = L
    w = xb.load_sae(model, spec, device)
    if model == 'gemma':
        enc = lambda x: w.sae.encode(x)
        dec = w.sae.W_dec                                   # [d_sae, d]
        sha_ = sha(spec['path'])
    else:
        enc = lambda x: w.sae(x)[1]
        dec = w.sae.decoder.weight.T                        # [d_sae, d]
        sha_ = sha(spec['src']['path']) if 'path' in spec['src'] else None
    return w, enc, dec.float(), spec['sae_id'], sha_


def run_model(model, phase, out, selection, smoke, device, log):
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    R = Runner(model, device)
    tok = R.bk.tokenize
    split = 'train' if smoke else phase
    P, dropped = pairs(model, split, tok, R.bk.bos_id)
    if smoke:
        P = P[::40]
    groups = np.array([p['statement_id'] for p in P])
    rH, rU = train_rows()
    gen = torch.Generator(device='cpu').manual_seed(SEED + (0 if model == 'gpt2' else 1))
    rng = np.random.default_rng(0)
    bs = BATCH[model]
    Hs, Us = [p['H'] for p in P], [p['U'] for p in P]
    lpH, lpU = [], []
    for _, b in batched(Hs, bs):
        lpH.append(R.logprobs(b))
    for _, b in batched(Us, bs):
        lpU.append(R.logprobs(b))
    klHU = torch.cat([kl(a, c) for a, c in zip(lpH, lpU)]).cpu().numpy()
    # sanity 1: final-token logit path == full forward (Gemma lm_head + softcap on final positions only)
    chk = Us[:8]
    if model == 'gemma':
        T_ = max(map(len, chk))
        ids_ = torch.zeros(len(chk), T_, dtype=torch.long)
        for i, q in enumerate(chk):
            ids_[i, :len(q)] = torch.tensor(q)
        with torch.inference_mode():
            full = R.hf(input_ids=ids_.to(device), attention_mask=torch.ones_like(ids_).to(device), use_cache=False).logits
        full = full[torch.arange(len(chk)), torch.tensor([len(q) - 1 for q in chk])].float().double().log_softmax(-1)
        path_err = float((full - R.logprobs(chk, exact_hf=True)).abs().max())
        info_extra = dict(fp32_softcap_vs_hf_bf16_max_abs_logprob=float((full - R.logprobs(chk)).abs().max()))
    else:
        with torch.inference_mode():
            T_ = max(map(len, chk))
            ids_ = torch.full((len(chk), T_), 50256, dtype=torch.long)
            for i, q in enumerate(chk):
                ids_[i, :len(q)] = torch.tensor(q)
            full = R.tl(ids_.to(device))[torch.arange(len(chk)), torch.tensor([len(q) - 1 for q in chk])].double().log_softmax(-1)
        path_err = float((full - R.logprobs(chk)).abs().max())
        info_extra = {}
    info = dict(**info_extra, model=model, split=split, n_pairs=len(P), n_statements=int(len(np.unique(groups))),
                k1_dropped=dropped, mean_kl_HU=float(klHU.mean()), sanity_logit_path_max_abs=path_err)
    log(f'[{model}] {split}: {len(P)} pairs ({dropped} dropped by K1), mean KL(H||U) {klHU.mean():.4f}')

    def E_of(vecs_or_fn, L, mode='add'):
        """Per-pair KL(H || U edited) for one condition. vecs: [d] (same for all) or callable(i0, batch) -> [b, d]."""
        res = []
        for i0, b in batched(Us, bs):
            v = vecs_or_fn(i0, len(b)) if callable(vecs_or_fn) else vecs_or_fn[None].expand(len(b), -1)
            lp = R.logprobs(b, L, add=v) if mode == 'add' else R.logprobs(b, L, replace=v)
            res.append(kl(lpH[i0 // bs], lp))
        return torch.cat(res).cpu().numpy()

    rows, per_pair, ctrl_store = [], {}, {}
    for L in LAYERS[model]:
        A = np.load(str(CACHE[model]).format(L=L), mmap_mode='r')
        aH, aU = np.asarray(A[rH], dtype=np.float32), np.asarray(A[rU], dtype=np.float32)
        ok = (np.abs(aH).sum(1) > 0) & (np.abs(aU).sum(1) > 0)     # rows K1 left unfilled are all-zero
        info[f'L{L}_train_pairs_for_direction'] = int(ok.sum())
        xH = torch.tensor(aH[ok], device=device)
        xU = torch.tensor(aU[ok], device=device)
        d_raw = xH.mean(0) - xU.mean(0)
        norm_raw = float(d_raw.norm())
        u_raw = d_raw / d_raw.norm()
        w, enc, dec, sae_id, sae_sha = sae_for(model, L, device)
        with torch.no_grad():
            rec = lambda x: torch.cat([w(x[i:i + 512])[0] for i in range(0, len(x), 512)])
            d_rec = rec(xH).mean(0) - rec(xU).mean(0)
            fired = torch.zeros(dec.shape[0], dtype=torch.bool, device=device)
            for x in (xH, xU):
                for i in range(0, len(x), 512):
                    fired |= (enc(x[i:i + 512]) > 0).any(0)
        u_rec = d_rec / d_rec.norm()
        cos = float(u_raw @ u_rec)
        alive = torch.nonzero(fired).flatten()
        log(f'[{model} L{L}] ||d_raw|| {norm_raw:.3f}  cos(raw, rec) {cos:.4f}  alive {len(alive)}  SAE {sae_id}')
        # sanity 2: a zero edit leaves every output unchanged
        z0 = E_of(torch.zeros(R.bk.d_model, device=device), L)
        info[f'L{L}_sanity_zero_edit_max_abs_kl_diff'] = float(np.abs(z0 - klHU).max())
        # site activations of H for the patch control
        xHsite = torch.cat([R.capture(b, L) for _, b in batched(Hs, bs)])
        kl_patch = E_of(lambda i0, n: xHsite[i0:i0 + n], L, mode='replace')
        if phase == 'validation' or smoke:
            grid_E = {}
            for c in GRID:
                k = E_of(c * norm_raw * u_raw, L)
                grid_E[c] = 1 - k.sum() / klHU.sum()
                per_pair[f'L{L}_raw_c{c}'] = k
            c_star = max(GRID, key=lambda c: grid_E[c])
        else:
            c_star = selection[model][str(L)]['c_star']
            grid_E = {}
        s = c_star * norm_raw
        k_raw = E_of(s * u_raw, L)
        k_rec = E_of(s * u_rec, L)
        per_pair.update({f'L{L}_raw': k_raw, f'L{L}_rec': k_rec, f'L{L}_patch': kl_patch})
        # controls
        n_ctrl = 4 if smoke else N_CTRL
        G = torch.randn(n_ctrl, R.bk.d_model, generator=gen).to(device)
        G = G / G.norm(dim=1, keepdim=True)
        pick = alive[torch.randint(len(alive), (n_ctrl,), generator=gen).to(device)]
        sign = (torch.randint(0, 2, (n_ctrl,), generator=gen) * 2 - 1).to(device).float()
        F = dec[pick] * sign[:, None]
        F = F / F.norm(dim=1, keepdim=True)
        fams = {'gaussian': G}
        if phase == 'test' or smoke:
            fams['sae_feature'] = F
        E_ctrl = {}
        for fam, V in fams.items():
            E_ctrl[fam] = np.array([1 - E_of(s * V[j], L).sum() / klHU.sum() for j in range(n_ctrl)])
            ctrl_store[f'L{L}_{fam}'] = E_ctrl[fam]
        E = lambda k: float(1 - k.sum() / klHU.sum())
        bt_raw, bt_rec = boot_E(klHU, k_raw, groups, rng), boot_E(klHU, k_rec, groups, rng)
        bt_ret = bt_rec / bt_raw
        r = dict(model=model, split=split, layer=L, depth_fraction=R.bk.depth_fraction(L), sae=sae_id, sae_sha256=sae_sha,
                 n_pairs=len(P), norm_raw=norm_raw, cos_raw_rec=cos, c_star=c_star, strength=s,
                 E_patch=E(kl_patch), E_patch_ci=list(np.quantile(boot_E(klHU, kl_patch, groups, rng), [.025, .975])),
                 E_raw=E(k_raw), E_raw_ci=list(np.quantile(bt_raw, [.025, .975])),
                 E_rec=E(k_rec), E_rec_ci=list(np.quantile(bt_rec, [.025, .975])),
                 retention=E(k_rec) / E(k_raw), retention_ci=list(np.quantile(bt_ret, [.025, .975])),
                 grid_E={str(k): float(v) for k, v in grid_E.items()})
        for fam, ec in E_ctrl.items():
            r[f'{fam}_mean'], r[f'{fam}_q95'], r[f'{fam}_max'] = float(ec.mean()), float(np.quantile(ec, .95)), float(ec.max())
            r[f'p_raw_vs_{fam}'] = float((1 + (ec >= r['E_raw']).sum()) / (1 + len(ec)))
            r[f'p_rec_vs_{fam}'] = float((1 + (ec >= r['E_rec']).sum()) / (1 + len(ec)))
        rows.append(r)
        log(f'[{model} L{L}] c*={c_star} E_patch {r["E_patch"]:.3f} E_raw {r["E_raw"]:.3f} E_rec {r["E_rec"]:.3f} '
            f'ret {r["retention"]:.3f} gauss q95 {r.get("gaussian_q95", float("nan")):.3f}')
        del w, dec
        torch.cuda.empty_cache()
    info['runtime_s'] = time.time() - t0
    np.savez_compressed(out / f'{model}_{split}_per_pair.npz', statement_id=groups, kl_HU=klHU,
                        template=np.array([p['template'] for p in P]), **per_pair)
    np.savez_compressed(out / f'{model}_{split}_controls.npz', **ctrl_store)
    del R
    torch.cuda.empty_cache()
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--models', default='gpt2,gemma')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    out = a.smoke_out if a.smoke else OUT / a.phase
    selection = None
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel in HASHED:
            if pr['hashes'].get(rel) != sha(REPO / rel):
                raise SystemExit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
        if a.phase == 'test':
            sel_p = OUT / 'validation/selection.json'
            selection = json.loads(sel_p.read_text())
    if (out / 'results.json').exists():
        raise SystemExit(f'{out}/results.json exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'run.log', 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()
    allrows, infos = [], {}
    for m in a.models.split(','):
        if (out / f'{m}_rows.json').exists():
            log(f'{m}: rows exist, reusing (resume)')
            allrows += json.loads((out / f'{m}_rows.json').read_text())
            continue
        rows, info = run_model(m, a.phase, out, selection, a.smoke, 'cuda', log)
        (out / f'{m}_rows.json').write_text(json.dumps(rows, indent=1, default=float))
        allrows += rows
        infos[m] = info
    res = dict(phase=a.phase, smoke=a.smoke, rows=allrows, info=infos, torch=torch.__version__,
               python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
               script_sha256=sha(__file__), written_utc=datetime.now(timezone.utc).isoformat())
    if a.phase == 'validation' and not a.smoke:
        sel = {}
        for r in allrows:
            sel.setdefault(r['model'], {})[str(r['layer'])] = dict(
                c_star=r['c_star'], gate_patch_ci_low_gt0=bool(r['E_patch_ci'][0] > 0))
        (out / 'selection.json').write_text(json.dumps(sel, indent=2))
        res['selection_sha256'] = sha(out / 'selection.json')
    if a.phase == 'test' and not a.smoke:
        res['selection_sha256'] = sha(OUT / 'validation/selection.json')
        fam = [r for r in allrows]
        for key in ('p_raw_vs_gaussian', 'p_raw_vs_sae_feature', 'p_rec_vs_gaussian', 'p_rec_vs_sae_feature'):
            adj = holm([r[key] for r in fam])
            for r, v in zip(fam, adj):
                r[key + '_holm'] = float(v)
        for r in fam:
            eligible = selection[r['model']][str(r['layer'])]['gate_patch_ci_low_gt0']
            r['eligible'] = eligible
            r['H1_raw_beats_controls'] = bool(eligible and r['p_raw_vs_gaussian_holm'] < .05 and r['p_raw_vs_sae_feature_holm'] < .05)
            r['H2_rec_beats_gaussian'] = bool(eligible and r['p_rec_vs_gaussian_holm'] < .05)
            lo, hi = r['retention_ci']
            r['H3_retention'] = ('not_eligible' if not eligible else
                                 'not_applicable (H1 not met)' if not r['H1_raw_beats_controls'] else
                                 'PRESERVED' if lo >= RETAIN else
                                 'DEGRADED' if hi < RETAIN else 'INCONCLUSIVE')
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    log('done')


if __name__ == '__main__':
    main()
