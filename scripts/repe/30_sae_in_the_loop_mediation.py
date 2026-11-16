#!/usr/bin/env python3
"""SIL: SAE-in-the-loop (mediated) persona intervention (Review #8 C3). Protocol: results/repe/sae_in_the_loop_mediation/.

Why a new design. Every M4 patch keeps the SAE error term, so it equals adding one vector to the raw activation
(paper section "What these patches are, mathematically"). It shows the SAE *selects* effect-carrying directions; it
does not show an effect *mediated* by the SAE. Here the SAE is put in the forward pass:

  spliced model M_D:  at layer 1 (intervention layer) and at every downstream layer in D, blocks.L.hook_resid_pre is
                      replaced at every non-BOS position by the SAE reconstruction x_hat = SAE_L(x) (error dropped).
  intervention:       at the layer-1 persona slot of the dishonest (U) prompt, the SAE latents are edited
                      z_U -> z_U with latents S set to z_H (S = all latents: full latent swap), decoded, and the
                      forward pass continues through M_D.
  effect:             T_S = KL(p^D_H || p^D_U) - KL(p^D_H || p^D_{U<-H}) at the final token (nats).

Configurations: D0 = {} (SAE only at the edit site), D1 = {8}, D2 = {8, 11}. In D1/D2 every path from the edit to
the output passes through the downstream SAEs' sparse codes at every position, so the effect cannot be written as a
fixed vector added to the raw activation; at the edit layer itself the edit is still x_hat_U -> x_hat_U + v (stated
in the README; not claimed to be more than that).

Controls (matched decoded norm, 20 draws each, at the same slot in the same spliced model):
  iso   x_hat_U + isotropic vector with ||v||; shuf  x_hat_U + W_dec(dz permuted over latents), rescaled to ||v||;
  neutral  the same latent swap for the neutral Na/Nb pairs (generic one-word change).
References: the original model's T_full (raw slot patch) and M4's error-kept T_rec on the same pairs.

Run (from the repository root):
    python scripts/repe/30_sae_in_the_loop_mediation.py --pilot OUTDIR     # R2 TopK s42 + old TopK s42, VALIDATION
    python scripts/repe/30_sae_in_the_loop_mediation.py --run              # pre-registered VALIDATION set
VALIDATION only; no test row is read.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import patching as pt  # noqa: E402

log = logging.getLogger('sil')


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAD = _load('scripts/shared/sae_ladder_evaluation.py', 'ladder')
S24 = LAD.S24
OUT = REPO / 'results/repe/sae_in_the_loop_mediation'
L_EDIT = 1
CONFIGS = {'D0': (), 'D1': (8,), 'D2': (8, 11)}
TEMPLATES = ['T2', 'T3', 'T4']          # M4's primary template set
N_VEC, SEED, B = 20, 20261113, 64
PILOT = [('r2', 'topk', 42), ('old', 'topk', 42)]
FAMILIES = [('r2', 'topk'), ('r2', 'batchtopk'), ('r2', 'relu'), ('old', 'topk'), ('old', 'relu'), ('old', 'jumprelu'),
            ('r2', 'topk_frozen_decoder')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kl(a, b):
    return pt.kl(a, b)


def sae_triplet(family, name, seed, device):
    """The same family/name/seed SAE at layers 1, 8, 11 (ladder sources/load)."""
    out = {}
    for L in (1, 8, 11):
        s = [x for x in LAD.sources(L, [family]) if x['name'] == name and x['seed'] == seed]
        if not s:
            return None
        out[L] = (LAD.load(s[0], L, device), s[0]['path'])
    return out


class Runner:
    def __init__(self, model, device):
        self.m, self.dev = model, device

    @torch.inference_mode()
    def final_lp(self, toks, finals, saes, splice, edit=None):
        """Forward with SAEs spliced at `splice` layers (non-BOS positions). edit = (slot_pos [n], fn(rec, z) -> rec)
        applied at L_EDIT. Returns final-token log-probs (float64)."""
        hooks = []
        n, T = toks.shape
        ar = torch.arange(n, device=self.dev)
        for L in splice:
            sae = saes[L][0]

            def h(resid, hook, sae=sae, L=L):
                x = resid[:, 1:].float()
                rec, z, _ = sae(x.reshape(-1, x.shape[-1]))
                rec = rec.float().reshape(x.shape)
                if edit is not None and L == L_EDIT:
                    pos, fn = edit
                    z = z.float().reshape(n, T - 1, -1)
                    rec[ar, pos - 1] = fn(rec[ar, pos - 1], z[ar, pos - 1])
                resid[:, 1:] = rec.to(resid.dtype)
                return resid
            hooks.append((f'blocks.{L}.hook_resid_pre', h))
        cap = {}
        hooks.append(('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x)))
        self.m.run_with_hooks(toks, return_type=None, fwd_hooks=hooks)
        return self.m.unembed(cap['h'][ar, finals][:, None])[:, 0].double().log_softmax(-1)

    @torch.inference_mode()
    def raw_patch_lp(self, toks, finals, pos, vec_fn):
        """Original model; at L_EDIT slot, resid -> vec_fn(resid_slot). Returns final log-probs."""
        n = toks.shape[0]
        ar = torch.arange(n, device=self.dev)
        cap = {}

        def h(resid, hook):
            resid[ar, pos] = vec_fn(resid[ar, pos].float()).to(resid.dtype)
            return resid
        self.m.run_with_hooks(toks, return_type=None, fwd_hooks=[
            (f'blocks.{L_EDIT}.hook_resid_pre', h), ('ln_final.hook_normalized', lambda x, hook: cap.__setitem__('h', x))])
        return self.m.unembed(cap['h'][ar, finals][:, None])[:, 0].double().log_softmax(-1)


def evaluate(run, rows, kind, saes, gen, gperm):
    a, b = ('H', 'U') if kind == 'persona' else ('Na', 'Nb')
    sae1 = saes[L_EDIT][0]
    W = sae1.decoder.weight.detach().float()
    recs = []
    for b0 in range(0, len(rows), B):
        rs = rows[b0:b0 + B]
        ta, tb = S24.toks_of(rs, a, run.dev), S24.toks_of(rs, b, run.dev)
        fin = torch.tensor([r['final'] for r in rs], device=run.dev)
        pos = torch.tensor([r['slot'] for r in rs], device=run.dev)
        ar = torch.arange(len(rs), device=run.dev)
        # raw slot activations at the edit layer (identical in the original and the spliced model: splicing starts there)
        xa = S24.Engine(run.m, L_EDIT).clean(ta, fin)[0][ar, pos].float()
        xb = S24.Engine(run.m, L_EDIT).clean(tb, fin)[0][ar, pos].float()
        with torch.no_grad():
            ra, za, _ = sae1(xa)
            rb, zb, _ = sae1(xb)
        ra, rb, za, zb = ra.float(), rb.float(), za.float(), zb.float()
        v = ra - rb                                   # the decoded latent swap, ||v|| used for matching
        dz = za - zb
        r = dict(statement_id=[x['statement_id'] for x in rs], template=[x['template'] for x in rs], kind=kind,
                 norm_v=v.norm(dim=-1).cpu().numpy(), norm_delta=(xa - xb).norm(dim=-1).cpu().numpy())
        # original-model references (M4 definitions at the L1 slot)
        lpa, lpb = run.final_lp(ta, fin, saes, ()), run.final_lp(tb, fin, saes, ())
        k0 = kl(lpa, lpb)
        r['kl_orig'] = k0.cpu().numpy()
        r['T_full_orig'] = (k0 - kl(lpa, run.raw_patch_lp(tb, fin, pos, lambda x: xa))).cpu().numpy()
        r['T_rec_errkept'] = (k0 - kl(lpa, run.raw_patch_lp(tb, fin, pos, lambda x: x + v))).cpu().numpy()
        for cname, down in CONFIGS.items():
            splice = (L_EDIT,) + down
            pa, pb = run.final_lp(ta, fin, saes, splice), run.final_lp(tb, fin, saes, splice)
            ks = kl(pa, pb)
            swap = run.final_lp(tb, fin, saes, splice, edit=(pos, lambda rec, z: rec + (za - z) @ W.T))
            r[f'{cname}_kl_spliced'] = ks.cpu().numpy()
            r[f'{cname}_T_S'] = (ks - kl(pa, swap)).cpu().numpy()
            if kind == 'persona':
                iso, shuf = [], []
                for _ in range(N_VEC):
                    e = pt.random_like(v, gen)
                    iso.append((ks - kl(pa, run.final_lp(tb, fin, saes, splice, edit=(pos, lambda rec, z, e=e: rec + e))))
                               .cpu().numpy())
                    perm = torch.randperm(dz.shape[1], generator=gperm).to(dz.device)
                    u = dz[:, perm] @ W.T
                    u = u * (v.norm(dim=-1, keepdim=True) / u.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                    shuf.append((ks - kl(pa, run.final_lp(tb, fin, saes, splice, edit=(pos, lambda rec, z, u=u: rec + u))))
                                .cpu().numpy())
                r[f'{cname}_iso_mean'] = np.mean(iso, 0)
                r[f'{cname}_iso_q95'] = np.quantile(iso, 0.95, axis=0)
                r[f'{cname}_shuf_mean'] = np.mean(shuf, 0)
                r[f'{cname}_shuf_q95'] = np.quantile(shuf, 0.95, axis=0)
        recs.append(pd.DataFrame(r))
    return pd.concat(recs, ignore_index=True)


def summarise(df, meta, rng):
    out = []
    for kind, g in df.groupby('kind'):
        sid = g.statement_id.to_numpy()
        r = dict(meta, kind=kind, n_pairs=len(g))
        for c in [c for c in g.columns if c.startswith(('kl_', 'T_', 'D0_', 'D1_', 'D2_'))]:
            s = pt.stat_summary(g[c].to_numpy(), sid, S24.N_BOOT, rng)
            r[f'{c}_mean'], r[f'{c}_ci_low'], r[f'{c}_ci_high'] = s['mean'], s['ci_low'], s['ci_high']
        for c in ['T_rec_errkept'] + [f'{d}_T_S' for d in CONFIGS]:
            r[f'{c}_over_T_full'], r[f'{c}_over_T_full_ci_low'], r[f'{c}_over_T_full_ci_high'] = \
                S24.ratio_ci(g[c].to_numpy(), g.T_full_orig.to_numpy(), sid, rng)
        if kind == 'persona':
            for d in CONFIGS:
                r[f'{d}_T_S_minus_iso_ci_low'] = pt.stat_summary((g[f'{d}_T_S'] - g[f'{d}_iso_mean']).to_numpy(), sid,
                                                                 S24.N_BOOT, rng)['ci_low']
                r[f'{d}_T_S_minus_shuf_ci_low'] = pt.stat_summary((g[f'{d}_T_S'] - g[f'{d}_shuf_mean']).to_numpy(), sid,
                                                                  S24.N_BOOT, rng)['ci_low']
                r[f'{d}_spliced_effect_present'] = bool(r[f'{d}_kl_spliced_ci_low'] > 0)
                r[f'{d}_mediated'] = bool(r[f'{d}_T_S_ci_low'] > 0 and r[f'{d}_T_S_minus_iso_ci_low'] > 0
                                          and r[f'{d}_T_S_minus_shuf_ci_low'] > 0)
        out.append(r)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument('--pilot', type=Path)
    gx.add_argument('--run', action='store_true')
    a = ap.parse_args()
    out = a.pilot if a.pilot else OUT / 'validation'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    if a.run:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel, h in pr['hashes'].items():
            if sha(REPO / rel) != h:
                raise SystemExit(f'{rel}: hash differs from the pre-registration; refusing to run.')
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained('gpt2')
    rows = {'persona': S24.build_rows(tok, 'validation', TEMPLATES, ['H', 'U']),
            'neutral': S24.build_rows(tok, 'validation', S24.NEUTRAL, ['Na', 'Nb'])}
    run = Runner(model, device)
    srcs = PILOT if a.pilot else [(f, n, s) for f, n in FAMILIES for s in (42, 1042, 2042)]
    out.mkdir(parents=True)
    pairs, summ = [], []
    for fam, name, seed in srcs:
        saes = sae_triplet(fam, name, seed, device)
        if saes is None:
            log.info('skip %s %s %s (not available at all three layers)', fam, name, seed)
            continue
        torch.manual_seed(SEED)
        gen = torch.Generator(device=device).manual_seed(SEED)
        gperm = torch.Generator().manual_seed(SEED)
        meta = dict(family=fam, name=name, seed=seed,
                    checkpoints={L: str(p).replace(str(Path.home()), '~').replace(str(REPO) + '/', '') for L, (_, p) in saes.items()})
        df = pd.concat([evaluate(run, rows[k], k, saes, gen, gperm) for k in ('persona', 'neutral')], ignore_index=True)
        for k, v in meta.items():
            if k != 'checkpoints':
                df[k] = v
        pairs.append(df)
        s = summarise(df, meta, np.random.default_rng(SEED))
        summ += s
        p = [x for x in s if x['kind'] == 'persona'][0]
        log.info('%s %s %s: T_full %.4f T_rec %.4f | D0 %.4f D1 %.4f D2 %.4f | spliced KL D2 %.4f (%.0fs)', fam, name, seed,
                 p['T_full_orig_mean'], p['T_rec_errkept_mean'], p['D0_T_S_mean'], p['D1_T_S_mean'], p['D2_T_S_mean'],
                 p['D2_kl_spliced_mean'], time.time() - t0)
        del saes
        torch.cuda.empty_cache()
    pd.concat(pairs, ignore_index=True).to_csv(out / 'per_pair.csv', index=False)
    pd.DataFrame(summ).to_csv(out / 'summary.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(dict(script_sha256=sha(__file__), mode='pilot' if a.pilot else 'run',
                                                     n_sources=len(pairs), runtime_s=time.time() - t0,
                                                     split='validation', test_rows_used=0), indent=2))


if __name__ == '__main__':
    main()
