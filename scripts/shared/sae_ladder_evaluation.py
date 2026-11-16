#!/usr/bin/env python3
"""Faithfulness ladder for any GPT-2 Small residual SAE (R1/E2 and R2 evaluation; 2026-11-10).

One code path for every SAE at a layer: our original base SAEs, the R2 data-scaled SAEs and the public res-jb SAEs.
Every metric is computed by imported, unchanged project code:
  * reconstruction: src/evaluation/faithfulness_metrics (OWT validation cache, BOS excluded)
  * behaviour: scripts/shared/spliced_reconstruction_evaluation.py (owt_eval, noise_like_error)
  * information: scripts/shared/m2_four_level_faithfulness.py (load_tasks, original_side, evaluate_source, PCA)
  * representation: scripts/repe/26_repe_lm_representation_preservation.py (load_rows, original_side, evaluate)
  * causal: scripts/repe/24_sae_patch_decomposition.py (M4 decompose/summarise at the layer's M4 site, reseeded per
    SAE as in E3; the feature analysis is not run)
New code: SAE loading (including the res-jb weight adapter) and orchestration. VALIDATION only; no test row is read.

Run from the repository root:
    python scripts/shared/sae_ladder_evaluation.py --layer 8 --sources old,resjb --out results/.../ladder_layer8_old_resjb.csv
    python scripts/shared/sae_ladder_evaluation.py --layer 8 --sources r2 --out ...
"""
import argparse
import glob
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
from src.evaluation import information_levels as il  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M2 = _load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
SP = _load('scripts/shared/spliced_reconstruction_evaluation.py', 'sp')
S24 = _load('scripts/repe/24_sae_patch_decomposition.py', 's24')
S26 = _load('scripts/repe/26_repe_lm_representation_preservation.py', 's26')
O3 = _load('scripts/openwebtext/03_training_sae.py', 'o3')

R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
R4 = REPO / 'results/openwebtext/r4_kl_finetune/checkpoints'
RESJB = Path.home() / '.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots'
SITE = {1: 'slot', 8: 'final', 11: 'final'}
ARCHS, SEEDS = ('relu', 'topk', 'jumprelu'), (42, 1042, 2042)
R2_NAMES = ('topk', 'batchtopk', 'relu', 'topk_w3072', 'topk_frozen_decoder')
SEED = 20261108
TASKS = ('got_cities', 'repe_persona', 'repe_neutral')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ sources
def resjb_path(L):
    hits = glob.glob(str(RESJB / f'*/blocks.{L}.hook_resid_pre/sae_weights.safetensors'))
    return Path(hits[0]) if hits else None


def load_resjb(L, device):
    """Map res-jb weights onto the in-house SAE class (it subtracts b_dec before encoding, like the public model)."""
    from safetensors.torch import load_file
    w = load_file(str(resjb_path(L)))
    sae = M2.fm.sae_class()(768, w['W_enc'].shape[1], 'relu', 128)
    with torch.no_grad():
        sae.encoder.weight.copy_(w['W_enc'].T)
        sae.encoder.bias.copy_(w['b_enc'])
        sae.decoder.weight.copy_(w['W_dec'].T)
        sae.output_bias.copy_(w['b_dec'])
    return sae.to(device).eval()


def sources(L, which):
    out = []
    if 'old' in which:
        out += [dict(family='old', name=a, seed=s, path=Path(uninterrupted_checkpoint(L, a, s)), arch=a) for a in ARCHS
                for s in SEEDS]
    if 'r2' in which:
        for n in R2_NAMES:
            for s in SEEDS:
                p = R2 / f'layer_{L}/{n}_seed{s}/checkpoint_final.pt'
                if p.exists():
                    out.append(dict(family='r2', name=n, seed=s, path=p, arch=None))
    if 'r2scaling' in which:          # data-scaling checkpoints, TopK and ReLU only (reconstruction and splicing)
        for n in ('topk', 'relu', 'batchtopk', 'topk_w3072'):
            for s in SEEDS:
                for b in ('10M', '25M', '50M'):
                    p = R2 / f'layer_{L}/{n}_seed{s}/checkpoint_{b}.pt'
                    if p.exists():
                        out.append(dict(family=f'r2_{b}', name=n, seed=s, path=p, arch=None))
    if 'r4' in which:                 # R4 fine-tunes (KL and MSE-control arms)
        for arm in ('kl', 'mse'):
            for s in SEEDS:
                p = R4 / f'layer_{L}/{arm}_topk_seed{s}/checkpoint.pt'
                if p.exists():
                    out.append(dict(family=f'r4_{arm}', name='topk', seed=s, path=p, arch=None))
    if 'resjb' in which and resjb_path(L) is not None:
        out.append(dict(family='resjb', name='resjb_relu_24576', seed=-1, path=resjb_path(L), arch='resjb'))
    return out


def load(src, L, device):
    if src['arch'] == 'resjb':
        return load_resjb(L, device)
    m, _ = fm.load_sae(src['path'], device, architecture=src['arch'])
    return m.eval()


@torch.no_grad()
def encode_np(sae, X, device, chunk=2048):
    """Chunked SAE pass: returns reconstruction and latents as float32 numpy arrays (CPU)."""
    Hs, Zs = [], []
    for b in torch.as_tensor(X).split(chunk):
        h, z, _ = sae(b.to(device))
        Hs.append(h.float().cpu())
        Zs.append(z.float().cpu())
    return torch.cat(Hs).numpy(), torch.cat(Zs).numpy()


class _OnDevice(list):
    """List of CPU tensors moved to the GPU on access (owt_eval indexes orig_cache[b // BATCH])."""

    def __init__(self, items, device):
        super().__init__(items)
        self.device = device

    def __getitem__(self, i):
        return super().__getitem__(i).to(self.device)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--sources', required=True, help='comma list of old,r2,r2scaling,resjb')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--light', action='store_true', help='reconstruction and splicing only')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    L, device, t0 = a.layer, 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    srcs = sources(L, a.sources.split(','))
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    # ---- shared inputs
    x_owt = O3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L)
    SP.BATCH = 8                     # smaller splice batches (GPU shared with training); results do not depend on it
    S24.CHUNK = 3                    # smaller M4 patch chunks, same reason
    spl = SP.Splicer(model, device)
    seqs, _ = SP.load_owt_validation()
    orig_lp = _OnDevice([t.cpu() for t in SP.owt_original(spl, seqs)], device)   # ~2.6 GB kept on the CPU
    if not a.light:
        tasks = M2.load_tasks()
        rng = np.random.default_rng(M2.SEED + L)               # same stream and task order as M2 -> identical originals
        origs = {tn: M2.original_side(t, L, rng) for tn, t in tasks.items()}
        bidx = {tn: il.bootstrap_indices(t['groups'][t['split'] == 'validation'], M2.N_BOOT,
                                         np.random.default_rng(M2.SEED + 100 + L)) for tn, t in tasks.items()}
        pca_stats, _ = M2.owt_pca(False)
        mu, cov = pca_stats[L]
        w_eig, V = il.pca_basis(mu, cov)
        xv_owt_np = M2.owt_val_layer(L)
        # RLM-RP state
        d, idx, y, y_hhp, g, sid, neardup, _ = S26.load_rows()
        S26.G.update(idx=idx, y=y, y_hhp=y_hhp, g=g, sid=sid, neardup=neardup)
        Xr = np.load(S26.CACHE / f'final_layer{L}.npy').astype(np.float32)
        S26.G['X'] = Xr
        S26.G['orig'] = S26.original_side(Xr)
        # M4 site
        site = SITE.get(L)
        if site:
            tok = AutoTokenizer.from_pretrained('gpt2')
            rows = {k: S24.build_rows(tok, 'validation', S24.PERSONA if k == 'persona' else S24.NEUTRAL,
                                      ['H', 'U', 'Hp'] if k == 'persona' else ['Na', 'Nb']) for k in ('persona', 'neutral')}
            eng = S24.Engine(model, L)
            vb = S24.clean_batches(eng, rows['persona'], ['H', 'U', 'Hp'], site, device)
            vn = S24.clean_batches(eng, rows['neutral'], ['Na', 'Nb'], site, device)
    out_rows = []
    partial = a.out.with_suffix('.partial.csv')
    for i, src in enumerate(srcs):
        sae = load(src, L, device)
        r = dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'],
                 checkpoint=str(src['path']).replace(str(Path.home()), '~').replace(str(REPO) + '/', ''),
                 checkpoint_sha256=sha(src['path']), d_sae=int(sae.decoder.weight.shape[1]))
        # reconstruction (OWT validation)
        xh, l0, act = fm.reconstruct(sae, x_owt, device)
        rec = fm.reconstruction_metrics(x_owt, xh, l0)
        r.update({f'owt_{k}': v for k, v in rec.items() if not isinstance(v, (list, dict))})
        r['owt_dead_fraction'] = float(1 - act.float().mean())
        # behaviour (OWT splicing) and same-norm noise reference
        o = SP.owt_eval(spl, seqs, L, SP.sae_fn(sae), orig_lp)
        r.update({f'splice_{k}': v for k, v in o.items()})
        gen = torch.Generator(device=device).manual_seed(SP.NOISE_SEED)
        on = SP.owt_eval(spl, seqs, L, SP.noise_like_error(sae, gen), orig_lp)
        r['splice_noise_delta_ce'], r['splice_noise_kl_mean'] = on['delta_ce'], on['kl_mean']
        r['splice_sae_vs_noise_delta_ce_ratio'] = o['delta_ce'] / on['delta_ce'] if on['delta_ce'] else np.nan
        if not a.light:
            # information (M2 readers), SAE and PCA at the SAE's own OWT validation NMSE
            k = il.pca_k_for_val_nmse(xv_owt_np, mu, V, r['owt_nmse'])
            r['pca_k_at_sae_nmse'] = int(k)
            for tn, t in tasks.items():
                H, Z = encode_np(sae, t['X'][L], device)
                rs, bs = M2.evaluate_source(dict(), t['X'][L], H, Z, t['y'], t['split'], t['groups'], origs[tn], bidx[tn],
                                            SEED + 1000 * L + i)
                r[f'{tn}__refit_logistic_gap_p'] = il.p_greater(bs['refit_logistic_gap'])
                Hp = il.pca_reconstruct(t['X'][L].astype(np.float64), mu, V, k).astype(np.float32)
                rp, _ = M2.evaluate_source(dict(), t['X'][L], Hp, None, t['y'], t['split'], t['groups'], origs[tn],
                                           bidx[tn], SEED + 1000 * L + i)
                for c in ('task_nmse', 'task_cosine', 'task_l0', 'frozen_logistic_gap', 'frozen_logistic_gap_ci_low',
                          'frozen_logistic_gap_ci_high', 'noise_frozen_logistic_gap', 'frozen_massmean_gap',
                          'refit_logistic_gap', 'refit_logistic_gap_ci_low', 'refit_logistic_gap_ci_high',
                          'refit_mlp_gap', 'refit_knn_gap', 'refit_sparse_z_gap', 'error_term_auroc'):
                    if c in rs:
                        r[f'{tn}__{c}'] = rs[c]
                r[f'{tn}__pca_refit_logistic_gap'] = rp['refit_logistic_gap']
                r[f'{tn}__sae_minus_pca_refit_gap'] = rs['refit_logistic_gap'] - rp['refit_logistic_gap']
            # representation (RLM-RP definitions)
            Hr, Zr = encode_np(sae, Xr, device)
            l0r = (Zr > 0).sum(1)
            va_act = (Zr[idx['all_va']] > 0).any(0)
            del Zr
            rr = S26.evaluate({}, Hr, None, l0r, float((~va_act).mean()))
            for c in ('pair_dir_cosine', 'pair_norm_retention', 'pair_error_share', 'massmean_dir_cosine',
                      'frozen_heldout_gap', 'refit_heldout_gap', 'frozen_syn_gap', 'repe_val_ev', 'repe_val_l0',
                      'repe_val_dead_fraction'):
                r[f'rlm__{c}'] = rr[c]
            # causal (M4 decomposition at the layer's M4 site)
            if site:
                torch.manual_seed(SEED)
                S24.GEN_CPU.manual_seed(SEED)
                gen = torch.Generator(device=device).manual_seed(SEED)
                rng4 = np.random.default_rng(SEED)
                dfp, dr, ident = S24.decompose(eng, sae, vb, 'persona', gen)
                dfp = S24.add_derived(dfp)
                summ = S24.summarise_decomp(dfp, dr, dict(kind='persona'), rng4)
                s = [x for x in summ if x['template_set'] == 'T2_T4'][0]
                for c in ('T_full_mean', 'T_rec_mean', 'T_err_mean', 'preservation_rec', 'preservation_rec_ci_low',
                          'preservation_rec_ci_high', 'preservation_err', 'shapley_share_rec', 'abs_I_over_T_full',
                          'rec_minus_syn_rec_mean', 'T_rec_above_controls_q95', 'iso_rec_q95', 'shuf_rec_q95'):
                    r[f'm4__{c}'] = s[c]
                r['m4__identity_err'] = ident
                r['m4__site'] = f'blocks.{L}.hook_resid_pre|{site}'
        out_rows.append(r)
        pd.DataFrame(out_rows).to_csv(partial, index=False)          # saved after every SAE
        print(f"L{L} {src['family']} {src['name']} {src['seed']}: ev {r['owt_explained_variance']:.4f} "
              f"l0 {r['owt_l0']:.1f} dCE {r['splice_delta_ce']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del sae
        torch.cuda.empty_cache()
    pd.DataFrame(out_rows).to_csv(a.out, index=False)
    partial.unlink()
    info = dict(script_sha256=sha(__file__), layer=L, sources=a.sources, n=len(out_rows), light=a.light,
                runtime_s=time.time() - t0, test_rows_used=0)
    a.out.with_suffix('.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
