#!/usr/bin/env python3
"""Phase 0: spliced-in causal reconstruction evaluation (inference only, VALIDATION data only).

Run from repository root:
    python scripts/shared/spliced_reconstruction_evaluation.py              # 108 uninterrupted SAEs, all layers
    python scripts/shared/spliced_reconstruction_evaluation.py --bridge     # layer-6 historical vs uninterrupted

GPT-2 Small is run twice on the same tokens: unmodified, and with ``blocks.{L}.hook_resid_pre`` replaced by the SAE
reconstruction (the hook the SAEs were trained on). BOS is never replaced (it was excluded from SAE training).

Datasets, positions and outcomes (validation rows only; no test row is tokenized or run):
* OWT validation (100 x 128-token sequences, 12 700 non-BOS tokens). Splice: every non-BOS position.
  Outcomes over predictions made at non-BOS positions: next-token CE (original, spliced, delta), KL(orig || spliced),
  top-1 agreement, loss recovered vs mean-ablation = (CE_mean - CE_sae) / (CE_mean - CE_orig).
* GoT validation (cities, neg_cities, larger_than, smaller_than; 1 040 statements, 520 true/false pairs).
  Splice 'all' (every non-BOS statement position): statement log-likelihood, per-token CE/KL, and the task metric:
  within each group (one true + one false statement) margin = logp(true) - logp(false); pairwise accuracy =
  share of groups with margin > 0, reported per dataset (neg_cities / smaller_than invert the association). Splice 'final' (final token only, the probe site): KL of the next-token
  distribution at that position.
* RepE validation (90 statements x honest/untruthful persona prompt, ending in a statement prefix).
  Splice 'final' (the probe/steering site) and 'all'. Outcomes at the final position: KL(orig || spliced),
  delta logp of the true next statement token; persona-shift cosine = cos(p_H - p_U original, p_H - p_U spliced)
  over the vocabulary, mean over statements; correlation / MAE of the persona effect logp_H - logp_U on the next
  statement token; honesty effect = mean_true[logp_H - logp_U](next) - mean_false[...] (secondary outcome of repe/13;
  ~0 for GPT-2 itself, so descriptive only).

Controls: two unmodified passes (determinism); identity replacement (activation written back unchanged);
fp16 round-trip (precision of the cached SAE training data); mean-ablation (OWT-train mean of the layer);
norm-matched random error (x + e, ||e|| = ||x - SAE(x)|| per token, random direction, seeded) per SAE.
Cached activations are checked against freshly computed ones before anything is spliced.
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
from transformer_lens import HookedTransformer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

LAYERS = range(12)
ARCHS = ('relu', 'topk', 'jumprelu')
SEEDS = (42, 1042, 2042)
GOT_SETS = ('cities', 'neg_cities', 'larger_than', 'smaller_than')
GOT_DOMAIN = {'cities': 'geographic', 'neg_cities': 'geographic', 'larger_than': 'numerical', 'smaller_than': 'numerical'}
OUT = {'owt': REPO / 'results/openwebtext/spliced_evaluation',
       'got': REPO / 'results/geometry_of_truth/spliced_evaluation',
       'repe': REPO / 'results/repe/spliced_evaluation',
       'shared': REPO / 'results/shared/spliced_evaluation'}
OWT_FLAGS = REPO / 'results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/all_layers_per_run.csv'
BATCH = 25
NOISE_SEED = 20261107


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hook_name(layer):
    return f'blocks.{layer}.hook_resid_pre'


# ------------------------------------------------------------------ data (validation only)
def pad(seqs, pad_id):
    lengths = torch.tensor([len(s) for s in seqs])
    toks = torch.full((len(seqs), int(lengths.max())), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        toks[i, :len(s)] = torch.as_tensor(s)
    return toks, lengths


def load_owt_validation():
    root = REPO / 'data/activations/gpt2_small/validation'
    seqs, cached = [], {l: [] for l in LAYERS}
    for p in sorted(root.glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        for i, n in enumerate(c['lengths']):
            seqs.append(c['tokens'][i, :int(n)])
            if len(cached[0]) < 2:   # keep two sequences per layer for the cache check
                for l in LAYERS:
                    cached[l].append(c['activations'][hook_name(l)][i, :int(n)].float())
    return seqs, cached


def owt_train_means():
    """Mean non-BOS activation per layer over the OWT TRAIN cache (for mean-ablation)."""
    s = {l: torch.zeros(768, dtype=torch.float64) for l in LAYERS}
    n = 0
    for p in sorted((REPO / 'data/activations/gpt2_small/train').glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        for i, length in enumerate(c['lengths']):
            if int(length) > 1:
                for l in LAYERS:
                    s[l] += c['activations'][hook_name(l)][i, 1:int(length)].double().sum(0)
                n += int(length) - 1
    return {l: (v / n).float() for l, v in s.items()}, n


def load_got_validation(model):
    rows = []
    for name in GOT_SETS:
        df = pd.read_csv(REPO / f'data/got/{name}.csv')
        sp = pd.read_csv(REPO / f'data/got/splits/{name}_splits.csv').set_index('row_id')
        df['row_id'] = np.arange(len(df))
        df['split'] = sp.loc[df.row_id, 'split'].to_numpy()
        df['group_id'] = sp.loc[df.row_id, 'group_id'].astype(str).to_numpy()
        assert (sp.loc[df.row_id, 'label'].to_numpy() == df['label'].to_numpy()).all()
        df['dataset'] = name
        rows.append(df[df.split == 'val'])
    v = pd.concat(rows, ignore_index=True)
    v['domain'] = v.dataset.map(GOT_DOMAIN)
    g = v.groupby(['dataset', 'group_id']).label.agg(['nunique', 'size'])
    assert (g['nunique'] == 2).all() and (g['size'] == 2).all(), 'each GoT validation group must be one true + one false'
    v['tokens'] = [model.to_tokens(s, prepend_bos=True, move_to_device=False)[0] for s in v.statement]
    return v


def load_repe_validation(model):
    p = pd.read_csv(REPO / 'data/repe/persona_pairs.csv')
    v = p[p.split == 'validation'].sort_values(['statement_id', 'persona']).reset_index(drop=True)
    tok = model.tokenizer
    nxt, has = [], []
    for r in v.itertuples():
        full = tok.encode(r.statement, add_special_tokens=False)
        has.append(r.prefix_tokens < len(full))
        nxt.append(full[r.prefix_tokens] if r.prefix_tokens < len(full) else 0)
    v['next_token'], v['has_next'] = nxt, has
    v['tokens'] = [model.to_tokens(s, prepend_bos=True, move_to_device=False)[0] for s in v.prompt]
    assert (v.groupby('statement_id').persona.nunique() == 2).all()
    return v


# ------------------------------------------------------------------ splicing
class Splicer:
    """Builds a forward hook that rewrites resid_pre at selected positions."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    @torch.inference_mode()
    def logprobs(self, tokens, lengths, layer=None, fn=None, positions='all'):
        """Return float32 log-probs [n, T, vocab]; fn maps x [m, 768] -> replacement [m, 768]."""
        tokens = tokens.to(self.device)
        n, T = tokens.shape
        ar = torch.arange(T, device=self.device)[None]
        lens = lengths.to(self.device)[:, None]
        if positions == 'all':
            mask = (ar >= 1) & (ar < lens)
        else:  # final non-padding token only
            mask = ar == (lens - 1)
        hooks = []
        if fn is not None:
            def hook(resid, hook):
                resid[mask] = fn(resid[mask].float()).to(resid.dtype)
                return resid
            hooks = [(hook_name(layer), hook)]
        logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        return logits.float().log_softmax(-1)

    @torch.inference_mode()
    def resid(self, tokens, layer):
        _, cache = self.model.run_with_cache(tokens.to(self.device), names_filter=lambda nm: nm == hook_name(layer))
        return cache[hook_name(layer)].float()


def noise_like_error(sae, gen):
    def fn(x):
        rec = sae(x)[0]
        err_norm = (x - rec).norm(dim=-1, keepdim=True)
        e = torch.randn(x.shape, generator=gen, device=x.device)
        return x + e / e.norm(dim=-1, keepdim=True) * err_norm
    return fn


def sae_fn(sae):
    return lambda x: sae(x)[0]


# ------------------------------------------------------------------ per-dataset evaluators
def owt_eval(sp, seqs, layer, fn, orig_cache):
    """Token-level CE / KL / top-1 over predictions made at non-BOS positions."""
    ce_o = ce_s = kl = agree = 0.0
    n = 0
    for b in range(0, len(seqs), BATCH):
        toks, lens = pad(seqs[b:b + BATCH], 50256)
        lp_o = orig_cache[b // BATCH]
        lp_s = sp.logprobs(toks, lens, layer, fn, 'all')
        T = toks.shape[1]
        ar = torch.arange(T)[None]
        # prediction at position t (1 <= t <= len-2) of token t+1
        m = ((ar >= 1) & (ar <= lens[:, None] - 2)).to(sp.device)
        tgt = toks[:, 1:].to(sp.device)
        lo, ls, mm = lp_o[:, :-1], lp_s[:, :-1], m[:, :-1]
        ce_o += -lo.gather(-1, tgt[..., None])[..., 0][mm].sum().item()
        ce_s += -ls.gather(-1, tgt[..., None])[..., 0][mm].sum().item()
        kl += (lo.exp() * (lo - ls)).sum(-1)[mm].sum().item()
        agree += (lo.argmax(-1) == ls.argmax(-1))[mm].sum().item()
        n += int(mm.sum())
    return {'n_predictions': n, 'ce_original': ce_o / n, 'ce_spliced': ce_s / n, 'delta_ce': (ce_s - ce_o) / n,
            'kl_mean': kl / n, 'top1_agreement': agree / n}


def owt_original(sp, seqs):
    return [sp.logprobs(*pad(seqs[b:b + BATCH], 50256)) for b in range(0, len(seqs), BATCH)]   # fp32


def got_eval(sp, v, layer, fn, orig):
    """'all' splice: statement log-likelihood & truth margin; 'final' splice: next-token KL at the final position."""
    out = {}
    stmt_lp, tok_kl, tok_n, final_kl = [], 0.0, 0, []
    for b in range(0, len(v), 64):
        chunk = v.iloc[b:b + 64]
        toks, lens = pad(list(chunk.tokens), 50256)
        lo = orig['all'][b // 64]
        ls = sp.logprobs(toks, lens, layer, fn, 'all') if fn is not None else lo
        tgt = toks[:, 1:].to(sp.device)
        ar = torch.arange(toks.shape[1] - 1)[None]
        m = (ar <= lens[:, None] - 2).to(sp.device)          # predictions of statement tokens 1..len-1
        stmt_lp.append((ls[:, :-1].gather(-1, tgt[..., None])[..., 0] * m).sum(1).cpu())
        m_spliced = m & (ar >= 1).to(sp.device)                # predictions made at spliced positions
        tok_kl += (lo[:, :-1].exp() * (lo[:, :-1] - ls[:, :-1])).sum(-1)[m_spliced].sum().item()
        tok_n += int(m_spliced.sum())
        lf = sp.logprobs(toks, lens, layer, fn, 'final') if fn is not None else lo
        rows = torch.arange(len(chunk), device=sp.device)
        fin = (lens - 1).to(sp.device)
        a, c = lo[rows, fin], lf[rows, fin]
        final_kl.append((a.exp() * (a - c)).sum(-1).cpu())
    lp = torch.cat(stmt_lp).numpy()
    fk = torch.cat(final_kl).numpy()
    d = v[['dataset', 'domain', 'group_id', 'label']].copy()
    d['lp'], d['final_kl'] = lp, fk
    out['statement_kl_mean'] = tok_kl / tok_n
    # Pooled domains mix datasets whose labels are inverted relative to the factual association (neg_cities,
    # smaller_than), so the task metric is reported per dataset; pooled values are kept for completeness.
    for key, sub in ([('all', d)] + [(dom, d[d.domain == dom]) for dom in ('geographic', 'numerical')]
                     + [(nm, d[d.dataset == nm]) for nm in GOT_SETS]):
        piv = sub.pivot_table(index=['dataset', 'group_id'], columns='label', values='lp')
        margin = piv[1] - piv[0]
        out[f'{key}_truth_margin_mean'] = float(margin.mean())
        out[f'{key}_truth_pair_accuracy'] = float((margin > 0).mean())
        out[f'{key}_statement_logprob_mean'] = float(sub.lp.mean())
        out[f'{key}_final_token_kl_mean'] = float(sub.final_kl.mean())
    return out


def got_original(sp, v):
    return {'all': [sp.logprobs(*pad(list(v.iloc[b:b + 64].tokens), 50256)) for b in range(0, len(v), 64)]}


def repe_eval(sp, v, layer, fn, orig_lp, mode):
    toks, lens = pad(list(v.tokens), 50256)
    rows = torch.arange(len(v), device=sp.device)
    fin = (lens - 1).to(sp.device)
    ls = (sp.logprobs(toks, lens, layer, fn, mode)[rows, fin] if fn is not None else orig_lp)
    lo = orig_lp
    kl = (lo.exp() * (lo - ls)).sum(-1).cpu().numpy()
    nt = torch.tensor(v.next_token.values, device=sp.device)[:, None]
    lp_next_o, lp_next_s = lo.gather(1, nt)[:, 0].cpu().numpy(), ls.gather(1, nt)[:, 0].cpu().numpy()
    H = (v.persona == 'honest').to_numpy()
    U = ~H
    assert (v.statement_id.to_numpy()[H] == v.statement_id.to_numpy()[U]).all()

    Ht, Ut = torch.as_tensor(H, device=sp.device), torch.as_tensor(U, device=sp.device)
    has = v.has_next.to_numpy()[H]
    true = (v.factual_label.to_numpy()[H] == 1)

    def shift(lp):
        a, b = lp[Ht], lp[Ut]
        return (a.exp() * (a - b)).sum(-1).cpu().numpy()

    def shift_vec(lp):   # persona-induced change of the next-token distribution, per statement
        return lp[Ht].exp() - lp[Ut].exp()


    def honesty_effect(lpn):
        dlt = lpn[H] - lpn[U]
        return float(dlt[has & true].mean() - dlt[has & ~true].mean())

    return {'final_kl_mean': float(kl.mean()), 'delta_logp_next_token': float((lp_next_s - lp_next_o)[v.has_next.to_numpy()].mean()),
            # raw KL(p_H || p_U) totals are reported but NOT used as a preservation score: any perturbation that
            # differs between the two prompts inflates them (random-error control: x6.6).
            'persona_shift_kl_original': float(shift(lo).sum()), 'persona_shift_kl_spliced': float(shift(ls).sum()),
            'persona_shift_cosine': float(torch.nn.functional.cosine_similarity(shift_vec(lo), shift_vec(ls), dim=-1).mean()),
            'persona_next_token_effect_corr': float(np.corrcoef((lp_next_o[H] - lp_next_o[U])[has],
                                                                (lp_next_s[H] - lp_next_s[U])[has])[0, 1]),
            'persona_next_token_effect_mae': float(np.abs((lp_next_o[H] - lp_next_o[U]) - (lp_next_s[H] - lp_next_s[U]))[has].mean()),
            'honesty_effect_original': honesty_effect(lp_next_o), 'honesty_effect_spliced': honesty_effect(lp_next_s)}


# ------------------------------------------------------------------ main
def evaluate_all(sp, data, layer, fn, tag):
    owt_seqs, owt_orig, got_v, got_orig, repe_v, repe_orig = data
    r = {'owt': owt_eval(sp, owt_seqs, layer, fn, owt_orig) if fn is not None else None,
         'got': got_eval(sp, got_v, layer, fn, got_orig) if fn is not None else None,
         'repe_final': repe_eval(sp, repe_v, layer, fn, repe_orig, 'final') if fn is not None else None,
         'repe_all': repe_eval(sp, repe_v, layer, fn, repe_orig, 'all') if fn is not None else None}
    flat = {'intervention': tag}
    for k, d in r.items():
        for kk, vv in (d or {}).items():
            flat[f'{k}__{kk}'] = vv
    return flat


@torch.no_grad()
def owt_recon(sae, x):
    x_hat, l0, active = fm.reconstruct(sae, x, x.device)
    m = fm.reconstruction_metrics(x.cpu(), x_hat, l0)
    m['dead_fraction'] = fm.dead_fraction(active)
    return {f'owt_val_recon__{k}': v for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bridge', action='store_true')
    a = ap.parse_args()
    device = 'cuda'
    name = 'layer6_bridge.csv' if a.bridge else 'per_run.csv'
    outs = [OUT['shared'] / name] if a.bridge else [OUT[k] / name for k in ('owt', 'got', 'repe')]
    for o in outs:
        if o.exists():
            raise SystemExit(f'{o} exists; refusing to overwrite.')
        o.parent.mkdir(parents=True, exist_ok=True)
    OUT['shared'].mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sp = Splicer(model, device)

    owt_seqs, owt_cached = load_owt_validation()
    got_v = load_got_validation(model)
    repe_v = load_repe_validation(model)
    means, n_train_tok = owt_train_means()
    owt_val_x = {}
    controls = []

    # ---- pipeline checks: determinism + cached == fresh activations
    toks, lens = pad(owt_seqs[:BATCH], 50256)
    l1, l2 = sp.logprobs(toks, lens), sp.logprobs(toks, lens)
    controls.append({'check': 'determinism_two_unmodified_passes_owt', 'layer': -1,
                     'max_abs_logprob_diff': float((l1 - l2).abs().max())})
    gcache = {nm: torch.load(REPO / f'data/got/activations/gpt2_small/{nm}_final_token.pt', map_location='cpu',
                             weights_only=False) for nm in GOT_SETS}
    rcache = torch.load(REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt', map_location='cpu', weights_only=True)
    rp = pd.read_csv(REPO / 'data/repe/persona_pairs.csv')
    for layer in LAYERS:
        fresh = sp.resid(pad(owt_seqs[:2], 50256)[0], layer)
        err = max(float((fresh[i, :len(owt_seqs[i])].cpu() - owt_cached[layer][i]).abs().max()
                      / owt_cached[layer][i].abs().max()) for i in range(2))
        controls.append({'check': 'owt_cache_vs_fresh_rel_max_abs_diff (cache is fp16)', 'layer': layer, 'value': err})
        sub = got_v.iloc[:64]
        toks, lens = pad(list(sub.tokens), 50256)
        fr = sp.resid(toks, layer)[torch.arange(len(sub)), (lens - 1)].cpu()
        ca = torch.stack([gcache[r.dataset]['activations'][layer][r.row_id] for r in sub.itertuples()])
        controls.append({'check': 'got_cache_vs_fresh_max_abs_diff', 'layer': layer, 'value': float((fr - ca).abs().max())})
        rr = repe_v.iloc[:64]
        toks, lens = pad(list(rr.tokens), 50256)
        fr = sp.resid(toks, layer)[torch.arange(len(rr)), (lens - 1)].cpu()
        idx = [int(rp.index[rp.prompt == pr][0]) for pr in rr.prompt]
        ca = rcache['activations'][layer][idx].float()
        controls.append({'check': 'repe_cache_vs_fresh_max_abs_diff', 'layer': layer, 'value': float((fr - ca).abs().max())})
    bad = [c for c in controls if c['check'].startswith(('got_cache', 'repe_cache')) and c['value'] > 1e-3]
    bad += [c for c in controls if c['check'].startswith('owt_cache') and c['value'] > 1e-2]
    assert not bad, f'cached activations do not match fresh ones: {bad}'
    assert controls[0]['max_abs_logprob_diff'] < 1e-5, 'model is not deterministic'

    # ---- original outputs (computed once)
    owt_orig = owt_original(sp, owt_seqs)
    got_orig = got_original(sp, got_v)
    toks, lens = pad(list(repe_v.tokens), 50256)
    repe_orig = sp.logprobs(toks, lens)[torch.arange(len(repe_v), device=device), (lens - 1).to(device)]
    data = (owt_seqs, owt_orig, got_v, got_orig, repe_v, repe_orig)

    def owt_x(layer):
        if layer not in owt_val_x:
            owt_val_x[layer] = fm_load_split(layer)
        return owt_val_x[layer]

    rows = []
    layer_rows = []
    layers = [6] if a.bridge else list(LAYERS)
    for layer in layers:
        # per-layer controls
        if not a.bridge:
            for tag, fn in [('identity', lambda x: x.clone()), ('fp16_roundtrip', lambda x: x.half().float()),
                            ('mean_ablation', (lambda m: (lambda x: m.to(x.device).expand_as(x).clone()))(means[layer]))]:
                layer_rows.append({'layer': layer, **evaluate_all(sp, data, layer, fn, tag)})
        if a.bridge:
            ckpts = [('historical_resume_5000_to_15000', arch, 42,
                      REPO / f'results/openwebtext/convergence/layer_6/{arch}_l0_128_resume_5000_to_15000/checkpoint_step_15000.pt', None)
                     for arch in ARCHS]
            ckpts += [(f'historical_finetuned_{var}', arch, 42,
                       REPO / f'results/geometry_of_truth/sae_finetuning/layer_6_{arch}/{var}/finetuned_checkpoint.pt', arch)
                      for arch in ARCHS for var in ('reconstruction_only', 'truth_aware')]
            ckpts += [('uninterrupted', arch, s, uninterrupted_checkpoint(6, arch, s), None) for arch in ARCHS for s in SEEDS]
        else:
            ckpts = [('uninterrupted', arch, s, uninterrupted_checkpoint(layer, arch, s), None) for arch in ARCHS for s in SEEDS]
        for family, arch, seed, path, arch_override in ckpts:
            sae, _ = fm.load_sae(path, device, architecture=arch_override)
            gen = torch.Generator(device=device).manual_seed(NOISE_SEED + 1000 * layer + seed + ARCHS.index(arch))
            base = {'layer': layer, 'family': family, 'architecture': arch, 'seed': seed,
                    'checkpoint_path': Path(path).relative_to(REPO).as_posix(), 'checkpoint_sha256': sha(path)}
            base.update(owt_recon(sae, owt_x(layer).to(device)))
            if a.bridge:
                base.update(bridge_probe_metrics(sae, device))
            rows.append({**base, **evaluate_all(sp, data, layer, sae_fn(sae), 'sae_reconstruction')})
            rows.append({**base, **evaluate_all(sp, data, layer, noise_like_error(sae, gen), 'norm_matched_random_error')})
        print(f'layer {layer} done ({time.time() - t0:.0f}s)', flush=True)

    df = pd.DataFrame(rows)
    if not a.bridge:
        flags = pd.read_csv(OWT_FLAGS)[['layer', 'architecture', 'seed', 'actual_l0', 'matched_10pct', 'sparsity_flag']]
        df = df.merge(flags.rename(columns={'actual_l0': 'owt_test_l0'}), on=['layer', 'architecture', 'seed'], how='left')
    lc = pd.DataFrame(layer_rows)
    meta = {'family', 'architecture', 'seed', 'checkpoint_path', 'checkpoint_sha256', 'layer', 'intervention',
            'owt_test_l0', 'matched_10pct', 'sparsity_flag'}
    if a.bridge:
        df.to_csv(OUT['shared'] / name, index=False)
    else:
        for key, prefixes in [('owt', ('owt__', 'owt_val_recon__')), ('got', ('got__',)), ('repe', ('repe_final__', 'repe_all__'))]:
            cols = [c for c in df.columns if c in meta or c.startswith(prefixes)]
            df[cols].to_csv(OUT[key] / name, index=False)
            lcols = [c for c in lc.columns if c in meta or c.startswith(prefixes)]
            lc[lcols].to_csv(OUT[key] / 'layer_controls.csv', index=False)
        pd.DataFrame(controls).to_csv(OUT['shared'] / 'pipeline_checks.csv', index=False)
    info = {'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'mode': 'bridge' if a.bridge else 'all_layers',
            'script': 'scripts/shared/spliced_reconstruction_evaluation.py', 'script_sha256': sha(__file__),
            'metrics_module_sha256': sha(fm.__file__), 'data': 'validation only (OWT validation cache tokens; GoT val; RepE validation)',
            'test_rows_used': False, 'owt_val_sequences': len(owt_seqs), 'got_val_statements': len(got_v),
            'repe_val_prompts': len(repe_v), 'mean_ablation_source': f'OWT train cache, {n_train_tok} non-BOS tokens',
            'noise_seed': NOISE_SEED, 'runtime_seconds': round(time.time() - t0, 1),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (OUT['shared'] / ('run_info_bridge.json' if a.bridge else 'run_info.json')).write_text(json.dumps(info, indent=2))
    print(f'Saved ({time.time() - t0:.0f}s)')


_O3 = None
_PROBES = None


def bridge_probe_metrics(sae, device):
    """Layer-6 validation-only frozen-probe gaps + direction fidelity (same probes as the all-layer evaluation)."""
    global _PROBES
    if _PROBES is None:
        spec = importlib.util.spec_from_file_location('ev', REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py')
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        _PROBES = []
        for data in (ev.load_got()[0], ev.load_repe()[0]):
            for domain, d in data.items():
                X = d['X'][6]
                _PROBES.append((domain, d, X, ev.fit_original_probes(X, d['y'], d['split']), ev))
    out = {}
    for domain, d, X, probes, ev in _PROBES:
        y, split = d['y'], d['split']
        va, tr = split == 'validation', split == 'train'
        X_hat = fm.reconstruct(sae, X, device)[0].numpy()
        so, sr = ev.scores(probes, X), ev.scores(probes, X_hat)
        for pr in ('logistic', 'diffmean'):
            out[f'val_probe__{domain}_{pr}_auroc_gap'] = (fm.safe_auroc(y[va], so[pr][va]) - fm.safe_auroc(y[va], sr[pr][va]))
        rd = X_hat[tr & (y == 1)].astype(np.float64).mean(0) - X_hat[tr & (y == 0)].astype(np.float64).mean(0)
        out[f'val_probe__{domain}_direction_cosine'] = fm.direction_fidelity(probes['raw_direction'], rd)['direction_cosine']
        rec = fm.reconstruction_metrics(X[va], X_hat[va])
        out[f'val_probe__{domain}_relative_sq_error'] = rec['relative_sq_error']
    return out


def fm_load_split(layer):
    global _O3
    if _O3 is None:
        spec = importlib.util.spec_from_file_location('o3', REPO / 'scripts/openwebtext/03_training_sae.py')
        _O3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_O3)
    return _O3.load_split(REPO / 'data/activations/gpt2_small', 'validation', layer)


if __name__ == '__main__':
    main()
