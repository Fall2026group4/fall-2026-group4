#!/usr/bin/env python3
"""Phase-0 splicing for the PUBLIC GPT-2 Small SAEs, under the team's splicing pipeline (roadmap G9).

Run from repository root (GPU):
    python scripts/shared/public_sae_spliced_evaluation.py --saes public_relu
    python scripts/shared/public_sae_spliced_evaluation.py --saes public_topk
    (add --smoke for layer 6 only, written to .../smoke/)

WHAT. Exactly the measurements of scripts/shared/spliced_reconstruction_evaluation.py (its loaders, Splicer and
evaluators are imported, not copied), with the public SAELens SAEs in place of the in-house ones:
  public_relu  gpt2-small-res-jb, blocks.{L}.hook_resid_pre, layers 0-11 (d_sae 24 576)
  public_topk  gpt2-small-resid-post-v5-32k, blocks.{L-1}.hook_resid_post read at resid_pre L, layers 1-11
               (d_sae 32 768, k = 32). resid_post of block L-1 is the same tensor as resid_pre of block L.
For every layer: the SAE reconstruction and a norm-matched random-error control (same per-token error norm, random
direction), plus the layer controls (identity, fp16 round trip, mean ablation) so loss recovered can be computed:
  owt loss recovered = (CE_mean_ablation - CE_spliced) / (CE_mean_ablation - CE_original), higher is better.
Datasets and outcomes are those of the source script (OWT validation tokens; GoT validation statements; RepE
validation prompts). VALIDATION rows only. Category: behavioural faithfulness.

CAVEATS. The public SAEs are much wider (24 576 / 32 768 vs 3 072), were trained on different data and run at a
different L0, so they are NOT sparsity-matched to the in-house SAEs: report them as an external reference next to
results/*/spliced_evaluation/, never in an architecture ranking.

Existing outputs are never overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PUBLIC = {
    'public_relu': dict(release='gpt2-small-res-jb', sae_id='blocks.{src}.hook_resid_pre', offset=0, layers=range(12)),
    'public_topk': dict(release='gpt2-small-resid-post-v5-32k', sae_id='blocks.{src}.hook_resid_post', offset=1,
                        layers=range(1, 12)),
}
OUT_ROOT = REPO / 'results/shared/public_sae_spliced_evaluation'
NOISE_SEED = 20261110


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_public(kind, layer, device):
    from sae_lens import SAE
    spec = PUBLIC[kind]
    sae_id = spec['sae_id'].format(src=layer - spec['offset'])
    loaded = SAE.from_pretrained(release=spec['release'], sae_id=sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded          # older SAELens returns a tuple
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, f"{spec['release']}/{sae_id}"


def public_fn(sae):
    """x [m, 768] -> reconstruction, through the SAE's own encode/decode (handles its input normalisation)."""
    def fn(x):
        return sae.decode(sae.encode(x.to(sae.W_dec.dtype))).float()
    return fn


def public_noise_fn(sae, gen):
    rec = public_fn(sae)

    def fn(x):
        err = (x - rec(x)).norm(dim=-1, keepdim=True)
        e = torch.randn(x.shape, generator=gen, device=x.device)
        return x + e / e.norm(dim=-1, keepdim=True) * err
    return fn


@torch.no_grad()
def recon_metrics(sae, x, fm, batch=4096):
    outs, l0s = [], []
    for b in x.split(batch):
        z = sae.encode(b.to(sae.W_dec.dtype))
        outs.append(sae.decode(z).float().cpu())
        l0s.append((z > 0).sum(-1).float().cpu())
    m = fm.reconstruction_metrics(x.cpu(), torch.cat(outs), torch.cat(l0s))
    return {f'owt_val_recon__{k}': v for k, v in m.items()}


def loss_recovered(ce_mean, ce_spliced, ce_orig):
    """(CE_mean - CE_spliced) / (CE_mean - CE_orig) * 100; higher is better."""
    return 100.0 * (ce_mean - ce_spliced) / (ce_mean - ce_orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saes', choices=tuple(PUBLIC), required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.device != 'cuda' and not a.smoke:
        raise SystemExit('CUDA required for the full run (use --smoke for a CPU pipeline check)')
    spec = importlib.util.spec_from_file_location('splice', REPO / 'scripts/shared/spliced_reconstruction_evaluation.py')
    sp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp_mod)
    from transformer_lens import HookedTransformer
    fm = sp_mod.fm

    out = OUT_ROOT / a.saes / ('smoke' if a.smoke else '')
    if (out / 'per_run.csv').exists():
        raise SystemExit(f'{out}/per_run.csv exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    device = a.device
    t0 = time.time()
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sp = sp_mod.Splicer(model, device)
    owt_seqs, _ = sp_mod.load_owt_validation()
    got_v = sp_mod.load_got_validation(model)
    repe_v = sp_mod.load_repe_validation(model)
    means, n_train_tok = sp_mod.owt_train_means()
    owt_orig = sp_mod.owt_original(sp, owt_seqs)
    got_orig = sp_mod.got_original(sp, got_v)
    toks, lens = sp_mod.pad(list(repe_v.tokens), 50256)
    repe_orig = sp.logprobs(toks, lens)[torch.arange(len(repe_v), device=device), (lens - 1).to(device)]
    data = (owt_seqs, owt_orig, got_v, got_orig, repe_v, repe_orig)

    layers = [l for l in ([6] if a.smoke else range(12)) if l in PUBLIC[a.saes]['layers']]
    rows, layer_rows = [], []
    for layer in layers:
        for tag, fn in [('identity', lambda x: x.clone()), ('fp16_roundtrip', lambda x: x.half().float()),
                        ('mean_ablation', (lambda m: (lambda x: m.to(x.device).expand_as(x).clone()))(means[layer]))]:
            layer_rows.append({'layer': layer, **sp_mod.evaluate_all(sp, data, layer, fn, tag)})
        sae, source = load_public(a.saes, layer, device)
        gen = torch.Generator(device=device).manual_seed(NOISE_SEED + layer)
        base = {'layer': layer, 'family': a.saes, 'architecture': a.saes, 'seed': 0, 'source': source,
                'd_sae': int(sae.W_dec.shape[0])}
        base.update(recon_metrics(sae, sp_mod.fm_load_split(layer).to(device), fm))
        rows.append({**base, **sp_mod.evaluate_all(sp, data, layer, public_fn(sae), 'sae_reconstruction')})
        rows.append({**base, **sp_mod.evaluate_all(sp, data, layer, public_noise_fn(sae, gen), 'norm_matched_random_error')})
        del sae
        print(f'layer {layer} done ({time.time() - t0:.0f}s)', flush=True)

    df, lc = pd.DataFrame(rows), pd.DataFrame(layer_rows)
    mean_ce = lc[lc['intervention'] == 'mean_ablation'].set_index('layer')['owt__ce_spliced']
    df['owt__loss_recovered_pct_vs_mean_ablation'] = [
        loss_recovered(mean_ce[r.layer], r['owt__ce_spliced'], r['owt__ce_original']) for _, r in df.iterrows()]
    df.to_csv(out / 'per_run.csv', index=False)
    lc.to_csv(out / 'layer_controls.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'saes': a.saes, 'release': PUBLIC[a.saes]['release'],
        'script': 'scripts/shared/public_sae_spliced_evaluation.py', 'script_sha256': sha(__file__),
        'splicing_code_sha256': sha(REPO / 'scripts/shared/spliced_reconstruction_evaluation.py'),
        'data': 'validation only', 'test_rows_used': False, 'mean_ablation_source': f'OWT train cache, {n_train_tok} tokens',
        'status': 'validation finding; external reference, not sparsity-matched to the in-house SAEs',
        'runtime_seconds': round(time.time() - t0, 1), 'torch': torch.__version__, 'device': device}, indent=2))
    cols = ['layer', 'intervention', 'owt_val_recon__explained_variance', 'owt_val_recon__l0',
            'owt__loss_recovered_pct_vs_mean_ablation', 'owt__kl_mean', 'got__cities_truth_pair_accuracy']
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
