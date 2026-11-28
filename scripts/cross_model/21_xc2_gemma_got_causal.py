#!/usr/bin/env python3
"""XC-2: the GoT causal truth-direction test, ported to Gemma-2-2B with Gemma Scope 16k (2026-11-23).

Port of dhruv-causal:scripts/geometry_of_truth/22_truth_direction_causal_test.py (commit 989d313; GPT-2, null result
with 600 controls). Protocol kept: data, pair construction, outcome, targets, positive control, controls, statistics,
families and criteria. Substitutions (pre-registered in results/cross_model/xc2_gemma_got_causal/preregistration.json):
  * model: Gemma-2-2B (HF, bf16, eager), site = output of model.model.layers[L] (resid_post), L = 8/12/16/21
    (depth (L+1)/26 = 0.35/0.50/0.65/0.85, matched to GPT-2 resid_pre 4/6/8/10 = 0.33/0.50/0.67/0.83);
  * SAEs: one Gemma Scope 16k canonical JumpReLU per layer (no seeds, so the "same sign in all seeds" rule is moot);
  * cached TRAIN activations: data/gemma/activations/got/cities_final_layer{L}.npy (XT-G0, statement-final token).

DATA. GoT cities; grouped splits data/got/splits/cities_splits.csv. Per city: the true and false statement are
  tokenized (Gemma, no BOS); prefix = shared tokens before the first difference; c / w = true / false next token.
OUTCOME. margin = log P(c | BOS+prefix) - log P(w | BOS+prefix) at the final prefix token; effect = mean over pairs of
  (margin with intervention - margin without); higher is better. Category: causal intervention.
TARGETS (TRAIN rows): mass_mean_direction; logistic_probe_direction (frozen original probe, C on validation);
  sae_reconstruction_direction, sae_top1_direction, sae_top10_direction. Each at +s and -s.
STRENGTH. s_L = multiple (0.05 ... 8) of ||train mass-mean difference|| maximising the mass-mean effect on VALIDATION.
POSITIVE CONTROL. unit(W_U[c] - W_U[w]) per pair at lambda * s_L, lambda in LAMBDA_GRID.
CONTROLS (600, norm s_L): Gaussian (raw + reconstruction directions); random alive latents with random sign (top-1);
  random 10-latent sets with the target coefficients (top-10).
STATISTICS. one-sided p = (1 + #{controls >= target})/(1 + N); two-sided = min(1, 2 min(upper, lower)); BH within the
  families A (28), C1 (8), C2 (12); 95 % bootstrap CI over test pairs (2000, seed 0).

Phases: --phase validation (strength selection -> validation/selection.json), --phase test (once). --smoke: TRAIN pairs.
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

LAYERS = (8, 12, 16, 21)
GPT2_MATCH = {8: 4, 12: 6, 16: 8, 21: 10}
DATA = REPO / 'data/got/cities.csv'
SPLITS = REPO / 'data/got/splits/cities_splits.csv'
ACTS = REPO / 'data/gemma/activations/got/cities_final_layer{L}.npy'
OUT = REPO / 'results/cross_model/xc2_gemma_got_causal'
N_CONTROLS, N_CONTROLS_SMOKE, N_VAL_CONTROLS = 600, 8, 50
CONTROL_SEED = 20261110                      # = dhruv 22
STRENGTH_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAMBDA_GRID = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
TOPK_SET, N_BOOT, ALPHA = 10, 2000, 0.05
CHUNK_SEQS = 2048
D = 2304
HASHED = ['scripts/cross_model/21_xc2_gemma_got_causal.py', 'scripts/cross_model/02_matched_splicing.py',
          'src/evaluation/pile_splicing.py', 'src/models/backends.py',
          'scripts/shared/all_layer_frozen_probe_evaluation.py', 'data/got/cities.csv',
          'data/got/splits/cities_splits.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------ pure helpers (identical definitions to dhruv 15/22)
def empirical_p_upper(t, c):
    c = np.asarray(c, float)
    return (1 + int(np.sum(c >= t))) / (1 + len(c))


def empirical_p_lower(t, c):
    c = np.asarray(c, float)
    return (1 + int(np.sum(c <= t))) / (1 + len(c))


def empirical_p_two_sided(t, c):
    return min(1.0, 2 * min(empirical_p_upper(t, c), empirical_p_lower(t, c)))


def bh_adjust(p):
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p, kind='stable')
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def unit(v):
    return v / v.norm()


def minimum_detectable_dose(lams, pvals, alpha=ALPHA):
    order = np.argsort(lams)
    lams, pvals = np.asarray(lams, float)[order], np.asarray(pvals, float)[order]
    best = float('nan')
    for lam, p in zip(lams[::-1], pvals[::-1]):
        if p <= alpha:
            best = float(lam)
        else:
            break
    return best


def divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return list(a[:i]), int(a[i]), int(b[i])
    return None


def build_pairs(df, splits, encode):
    df = df.copy()
    df['row_id'] = np.arange(len(df))
    sp = splits.set_index('row_id').loc[df['row_id']]
    if not (sp['label'].to_numpy() == df['label'].to_numpy()).all():
        raise ValueError('labels in the split file do not match the data')
    df['split'], df['group_id'] = sp['split'].to_numpy(), sp['group_id'].astype(str).to_numpy()
    rows, dropped = [], 0
    for g, d in df.groupby('group_id', sort=True):
        t, f = d[d['label'] == 1], d[d['label'] == 0]
        if len(t) != 1 or len(f) != 1 or t['split'].iloc[0] != f['split'].iloc[0]:
            dropped += 1
            continue
        div = divergence(encode(t['statement'].iloc[0]), encode(f['statement'].iloc[0]))
        if div is None or len(div[0]) == 0:
            dropped += 1
            continue
        rows.append({'group_id': g, 'split': t['split'].iloc[0], 'prefix': div[0], 'c': div[1], 'w': div[2]})
    return pd.DataFrame(rows), dropped


def bootstrap_mean_ci(v, n=N_BOOT, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def control_seed(layer, tag):
    h = hashlib.sha256(f'{CONTROL_SEED}|got-gemma|{layer}|{tag}'.encode()).digest()
    return CONTROL_SEED + int.from_bytes(h[:4], 'little') % 1_000_000


# ------------------------------------------------------------ Gemma runner
class PrefixRunner:
    def __init__(self, bk, pairs, device):
        self.hf, self.device = bk.model, device
        self.cap = getattr(self.hf.config, 'final_logit_softcapping', None)
        seqs = [[bk.bos_id] + list(p) for p in pairs['prefix']]
        self.lengths = torch.tensor([len(s) for s in seqs])
        T = int(self.lengths.max())
        self.tokens = torch.zeros((len(seqs), T), dtype=torch.long)      # right padding (causal: finals unaffected)
        for i, s in enumerate(seqs):
            self.tokens[i, :len(s)] = torch.tensor(s)
        self.c = torch.tensor(pairs['c'].to_numpy(), device=device)
        self.w = torch.tensor(pairs['w'].to_numpy(), device=device)
        self.n = len(seqs)

    @torch.inference_mode()
    def _fwd(self, tok, fin, layer, delta, capture=False, exact_hf=False):
        rows = torch.arange(len(tok), device=self.device)
        got, hd = {}, None
        if layer is not None:
            def hk(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if capture:
                    got['x'] = h[rows, fin].float().clone()
                if delta is None:
                    return None
                h = h.clone()
                h[rows, fin] = (h[rows, fin].float() + delta).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hd = self.hf.model.layers[layer].register_forward_hook(hk)
        try:
            hs = self.hf.model(input_ids=tok, attention_mask=torch.ones_like(tok), use_cache=False).last_hidden_state
        finally:
            if hd is not None:
                hd.remove()
        if capture:
            return got['x']
        if exact_hf:   # HF's order (soft-cap in bf16); only for the path check
            lg = self.hf.lm_head(hs[rows, fin])
            if self.cap:
                lg = torch.tanh(lg / self.cap) * self.cap
            return lg.float().log_softmax(-1)
        lg = self.hf.lm_head(hs[rows, fin]).float()      # outcome path: soft-cap in fp32
        if self.cap:
            lg = torch.tanh(lg / self.cap) * self.cap
        return lg.log_softmax(-1)

    def margins_many(self, layer, deltas):
        """deltas: list of [d] or [n, d] tensors -> [len(deltas), n] margins. EVERY forward has the identical shape
        [k * n, T] (short chunks are padded with zero deltas), so bf16 kernels are the same for targets, controls and
        the baseline (batch-shape-dependent bf16 rounding was 0.26-0.34 nats per pair in the first smoke)."""
        k = max(1, CHUNK_SEQS // self.n)
        tok = self.tokens.to(self.device).repeat(k, 1)
        fin = (self.lengths - 1).to(self.device).repeat(k)
        cc, ww = self.c.repeat(k), self.w.repeat(k)
        zero = torch.zeros(D, device=self.device)
        out = []
        for b in range(0, len(deltas), k):
            ds = list(deltas[b:b + k])
            m = len(ds)
            ds += [zero] * (k - m)
            Dl = torch.cat([d.expand(self.n, -1) if d.ndim == 1 else d for d in ds]).float()
            lp = self._fwd(tok, fin, layer, Dl)
            mg = (lp.gather(1, cc[:, None]) - lp.gather(1, ww[:, None]))[:, 0]
            out.append(mg.view(k, self.n)[:m].cpu())
        return torch.cat(out).numpy()

    def margins(self, layer, delta=None):
        """Baseline = zero delta at `layer` through the same fixed-shape path."""
        return self.margins_many(layer, [torch.zeros(D, device=self.device) if delta is None else delta])[0]

    @torch.inference_mode()
    def full_forward_check(self):
        tok = self.tokens[:64].to(self.device)
        fin = (self.lengths[:64] - 1).to(self.device)
        lg = self.hf(input_ids=tok, attention_mask=torch.ones_like(tok), use_cache=False).logits
        lp = lg[torch.arange(len(tok), device=self.device), fin].float().log_softmax(-1)
        full = (lp.gather(1, self.c[:64, None]) - lp.gather(1, self.w[:64, None]))[:, 0].cpu().numpy()
        lp2 = self._fwd(tok, fin, None, None, exact_hf=True)
        fast = (lp2.gather(1, self.c[:64, None]) - lp2.gather(1, self.w[:64, None]))[:, 0].cpu().numpy()
        return float(np.abs(full - fast).max())


def load_train(layer):
    sp = pd.read_csv(SPLITS)
    d = pd.read_csv(DATA)
    assert (sp['row_id'].to_numpy() == np.arange(len(d))).all() and (sp['label'].to_numpy() == d['label'].to_numpy()).all()
    X = np.asarray(np.load(str(ACTS).format(L=layer), mmap_mode='r'), dtype=np.float32)
    assert len(X) == len(d)
    split = np.where(sp['split'].to_numpy() == 'val', 'validation', sp['split'].to_numpy())
    return X, d['label'].to_numpy(dtype=np.int64), split


def load_sae(layer, device):
    xb = _mod('scripts/cross_model/02_matched_splicing.py', 'xb02')
    spec = [s for s in xb.sae_specs('gemma', layer, ['canonical']) if s['family'] == 'gemmascope_16k_canonical'][0]
    w = xb.load_sae('gemma', spec, device)
    return w.sae, spec['sae_id'], sha(spec['path'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    device = 'cuda'
    t0 = time.time()
    out = a.smoke_out if a.smoke else OUT / a.phase
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for rel in HASHED:
            if pr['hashes'].get(rel) != sha(REPO / rel):
                raise SystemExit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
    done_marker = out / ('selection.json' if a.phase == 'validation' else 'interventions.csv')
    if done_marker.exists():
        raise SystemExit(f'{done_marker} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    ev = _mod('scripts/shared/all_layer_frozen_probe_evaluation.py', 'frozen_eval')
    bk = _mod('src/models/backends.py', 'backends').make_backend('gemma', device)
    for p in bk.model.parameters():
        p.requires_grad_(False)
    pairs, dropped = build_pairs(pd.read_csv(DATA), pd.read_csv(SPLITS), bk.tokenize)
    n_controls = N_CONTROLS_SMOKE if a.smoke else N_CONTROLS
    W_U = bk.model.lm_head.weight.detach().float()         # [V, d]

    def directions(layer):
        X, y, split = load_train(layer)
        tr = split == 'train'
        d_raw = X[tr & (y == 1)].astype(np.float64).mean(0) - X[tr & (y == 0)].astype(np.float64).mean(0)
        probes = ev.fit_original_probes(X, y, split)
        sc, clf = probes['logistic'].steps[0][1], probes['logistic'].steps[-1][1]
        return X, y, split, d_raw, clf.coef_[0] / sc.scale_, probes['C']

    if a.phase == 'validation' or a.smoke:
        part = pairs[pairs['split'] == ('train' if a.smoke else 'val')].reset_index(drop=True)
        if a.smoke:
            part = part.iloc[:40]
        run = PrefixRunner(bk, part, device)
        base = run.margins(LAYERS[0])
        rows, chosen = [], {}
        for layer in LAYERS:
            base = run.margins(layer)
            _, _, _, d_raw, _, _ = directions(layer)
            norm = float(np.linalg.norm(d_raw))
            d_hat = unit(torch.tensor(d_raw, dtype=torch.float32)).to(device)
            g = torch.Generator().manual_seed(control_seed(layer, 'validation'))
            rand = [unit(torch.randn(D, generator=g)).to(device) for _ in range(4 if a.smoke else N_VAL_CONTROLS)]
            lrows = []
            for mult in STRENGTH_GRID:
                s = mult * norm
                ms = run.margins_many(layer, [s * d_hat] + [s * r for r in rand])
                eff = (ms - base[None]).mean(1)
                lrows.append(dict(layer=layer, strength_multiple=mult, strength_norm=s, effect_mass_mean=float(eff[0]),
                                  control_mean=float(eff[1:].mean()), control_sd=float(eff[1:].std(ddof=1)),
                                  control_max=float(eff[1:].max())))
                print(lrows[-1], flush=True)
            best = max(lrows, key=lambda r: r['effect_mass_mean'])
            chosen[str(layer)] = dict(strength_multiple=best['strength_multiple'], strength_norm=best['strength_norm'],
                                      mass_mean_difference_norm_train=norm)
            rows += lrows
        pd.DataFrame(rows).to_csv(out / 'validation_strength_selection.csv', index=False)
        info = dict(n_pairs={k: int(v) for k, v in pairs['split'].value_counts().items()}, pairs_dropped=int(dropped),
                    validation_baseline_pair_accuracy=float((base > 0).mean()),
                    validation_baseline_margin=float(base.mean()), strength_per_layer=chosen,
                    runtime_s=time.time() - t0, written_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(__file__))
        (out / 'validation_info.json').write_text(json.dumps(info, indent=2))
        (out / 'selection.json').write_text(json.dumps(chosen, indent=2))
        print(json.dumps(info, indent=1))
        if not a.smoke:
            return

    # ------------------------------------------------------------ test (or smoke continuation on TRAIN pairs)
    if a.smoke:
        sel = chosen
        test = pairs[pairs['split'] == 'train'].reset_index(drop=True).iloc[:40]
    else:
        sel = json.loads((OUT / 'validation/selection.json').read_text())
        test = pairs[pairs['split'] == 'test'].reset_index(drop=True)
    run = PrefixRunner(bk, test, device)
    base = run.margins(LAYERS[0])
    records, store, checks = [], {'baseline': base}, []
    sae_hashes = {}
    for layer in LAYERS:
        base = run.margins(layer)          # zero delta through the identical fixed-shape path
        store[f'baseline_L{layer}'] = base
        s = sel[str(layer)]['strength_norm']
        X, y, split, d_raw, d_probe, C = directions(layer)
        checks.append(dict(layer=layer, check='probe_C', value=C))
        zero = float(np.abs(run.margins(layer, torch.zeros(D, device=device)) - base).max())
        checks.append(dict(layer=layer, check='zero_delta_margin_max_abs', value=zero))
        fast = run.full_forward_check()
        checks.append(dict(layer=layer, check='final_token_path_vs_full_forward_margin_max_abs', value=fast))
        g_chk = torch.Generator().manual_seed(control_seed(layer, 'path_check'))
        v1 = 5.0 * unit(torch.randn(D, generator=g_chk)).to(device)
        k_ = CHUNK_SEQS // run.n
        rep = run.margins_many(layer, [v1] * 3 + [torch.zeros(D, device=device)] * k_)
        bc = float(max(np.abs(rep[0] - rep[1]).max(), np.abs(rep[0] - rep[2]).max(),
                       np.abs(rep[-1] - base).max(), np.abs(rep[k_] - base).max(),
                       np.abs(run.margins(layer, v1) - rep[0]).max()))
        checks.append(dict(layer=layer, check='same_delta_any_chunk_position_margin_max_abs', value=bc))
        if bc > 1e-4 or zero > 1e-4 or fast > 1e-4:
            raise SystemExit(f'layer {layer}: path consistency check failed ({bc}, {zero}, {fast})')

        def record(part, kind, name, group, detail, m, sign=0, key=None):
            eff = m - base
            records.append(dict(layer=layer, gpt2_matched_layer=GPT2_MATCH[layer], part=part, intervention_kind=kind,
                                intervention=name, control_group=group, detail=detail, sign=sign,
                                effect=float(eff.mean()), accuracy=float((m > 0).mean()),
                                baseline_accuracy=float((base > 0).mean()), n_pairs=len(base)))
            if key:
                store[key] = eff

        def gaussians(tag):
            g = torch.Generator().manual_seed(control_seed(layer, tag))
            return [unit(torch.randn(D, generator=g)).to(device) for _ in range(n_controls)]

        # A: graded positive control
        u = W_U[run.c] - W_U[run.w]
        u = u / u.norm(dim=-1, keepdim=True)
        rs = gaussians('positive')
        lams, ps = [], []
        for lam in LAMBDA_GRID:
            grp = f'L{layer}__positive_lam{lam}'
            ms = run.margins_many(layer, [lam * s * u] + [lam * s * r for r in rs])
            record('A', 'target', 'unembedding_contrast', grp, f'lambda={lam}', ms[0], key=grp)
            for mm in ms[1:]:
                record('A', 'control', 'gaussian', grp, f'lambda={lam}', mm)
            e = float((ms[0] - base).mean())
            lams.append(lam)
            ps.append(empirical_p_upper(e, (ms[1:] - base[None]).mean(1)))
        checks.append(dict(layer=layer, check='minimum_detectable_dose', value=minimum_detectable_dose(lams, ps)))

        # C: raw directions (+s, -s) with shared Gaussian controls
        grp = f'L{layer}__gaussian'
        d_hat = unit(torch.tensor(d_raw, dtype=torch.float32)).to(device)
        tg = []
        for name, v in (('mass_mean_direction', d_raw), ('logistic_probe_direction', d_probe)):
            v = unit(torch.tensor(v, dtype=torch.float32)).to(device)
            for sign in (1, -1):
                tg.append((name, sign, sign * s * v))
        # SAE directions
        sae, sae_id, sae_sha = load_sae(layer, device)
        sae_hashes[str(layer)] = dict(sae_id=sae_id, sha256=sae_sha)
        tr = split == 'train'
        Xtr = torch.tensor(X[tr], device=device)
        ytr = torch.tensor(y[tr], device=device)
        z = torch.cat([sae.encode(Xtr[i:i + 256]) for i in range(0, len(Xtr), 256)])
        rec = torch.cat([sae.decode(z[i:i + 256]) for i in range(0, len(z), 256)])
        Wd = sae.W_dec.T.float()                                   # [d, d_sae] (columns = latents, as dhruv's D)
        alive = torch.where((z > 0).any(0))[0].cpu()
        dmu = z[ytr == 1].mean(0) - z[ytr == 0].mean(0)
        top = torch.argsort(dmu.abs(), descending=True, stable=True)[:TOPK_SET]
        sae_t = {'sae_reconstruction_direction': (unit(rec[ytr == 1].mean(0) - rec[ytr == 0].mean(0)), grp),
                 'sae_top1_direction': (unit(torch.sign(dmu[top[0]]) * Wd[:, top[0]]), f'L{layer}__feat1'),
                 'sae_top10_direction': (unit(Wd[:, top] @ dmu[top]), f'L{layer}__feat10')}
        checks.append(dict(layer=layer, check='n_alive_latents_train', value=int(len(alive))))
        for name, (v, g_) in sae_t.items():
            checks.append(dict(layer=layer, check=f'cos_{name}_vs_mass_mean', value=float(v @ d_hat)))
        all_t = [(n_, sg, vec, grp) for n_, sg, vec in tg] + \
                [(n_, sg, sg * s * v, g_) for n_, (v, g_) in sae_t.items() for sg in (1, -1)]
        ms = run.margins_many(layer, [t[2] for t in all_t])
        for (n_, sg, _, g_), mm in zip(all_t, ms):
            record('C', 'target', n_, g_, '', mm, sign=sg, key=f'L{layer}__{n_}_{"pos" if sg > 0 else "neg"}')
        for mm in run.margins_many(layer, [s * r for r in gaussians('steer')]):
            record('C', 'control', 'gaussian', grp, '', mm)
        g = torch.Generator().manual_seed(control_seed(layer, 'gemmascope'))
        items = []
        for _ in range(n_controls):
            j = int(alive[torch.randint(len(alive), (1,), generator=g)])
            sg = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
            items.append(s * unit(sg * Wd[:, j]))
        for mm in run.margins_many(layer, items):
            record('C', 'control', 'random_latent_direction', f'L{layer}__feat1', '', mm)
        items = []
        for _ in range(n_controls):
            idx = alive[torch.randperm(len(alive), generator=g)[:TOPK_SET]].to(device)
            items.append(s * unit(Wd[:, idx] @ dmu[top]))
        for mm in run.margins_many(layer, items):
            record('C', 'control', 'random_latent_set_direction', f'L{layer}__feat10', '', mm)
        del sae, z, rec, Wd
        torch.cuda.empty_cache()
        pd.DataFrame(records).to_csv(out / 'interventions_partial.csv', index=False)
        print(f'done layer {layer} ({time.time() - t0:.0f}s)', flush=True)

    df = pd.DataFrame(records)
    pd.DataFrame(checks).to_csv(out / 'pipeline_checks.csv', index=False)
    np.savez_compressed(out / 'per_pair_effects.npz', **store)
    summary = summarise(df, store)
    summary.to_csv(out / 'summary.csv', index=False)
    df.to_csv(out / 'interventions.csv', index=False)
    crit = criteria(summary)
    (out / 'criteria.json').write_text(json.dumps(crit, indent=2, default=float))
    (out / 'test_run_info.json').write_text(json.dumps(dict(
        n_pairs=len(test), baseline_pair_accuracy=float((base > 0).mean()), baseline_margin=float(base.mean()),
        n_controls=n_controls, n_interventions=len(df), sae=sae_hashes, runtime_s=time.time() - t0,
        selection_sha256=None if a.smoke else sha(OUT / 'validation/selection.json'), script_sha256=sha(__file__),
        torch=torch.__version__, python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
        written_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    show = ['layer', 'part', 'intervention', 'sign', 'detail', 'effect', 'ci95_low', 'ci95_high', 'p_upper',
            'p_two_sided', 'p_bh']
    print(summary[[c for c in show if c in summary]].round(4).to_string(index=False))
    print(json.dumps(crit, indent=1, default=float))


def summarise(df, store):
    ctrl = df[df['intervention_kind'] == 'control']
    groups = {g: d['effect'].to_numpy() for g, d in ctrl.groupby('control_group')}
    rows = []
    for _, t in df[df['intervention_kind'] == 'target'].iterrows():
        layer = int(t['layer'])
        key = t['control_group'] if t['part'] == 'A' else f"L{layer}__{t['intervention']}_{'pos' if t['sign'] > 0 else 'neg'}"
        c = groups[t['control_group']]
        lo, hi = bootstrap_mean_ci(store[key])
        row = t.drop(labels=['intervention_kind']).to_dict()
        row.update(ci95_low=lo, ci95_high=hi, n_controls=len(c), control_mean=float(c.mean()),
                   control_sd=float(c.std(ddof=1)), control_q95=float(np.percentile(c, 95)),
                   z_vs_controls=float((t['effect'] - c.mean()) / c.std(ddof=1)),
                   p_upper=empirical_p_upper(t['effect'], c), p_lower=empirical_p_lower(t['effect'], c),
                   p_two_sided=empirical_p_two_sided(t['effect'], c))
        if t['part'] == 'A':
            row['family'], row['family_p'] = 'A_positive_control', 'p_upper'
        elif t['sign'] != 1:
            row['family'], row['family_p'] = None, None
        elif t['intervention'] in ('mass_mean_direction', 'logistic_probe_direction'):
            row['family'], row['family_p'] = 'C1_raw_directions', 'p_upper'
        else:
            row['family'], row['family_p'] = 'C2_sae_directions', 'p_two_sided'
        rows.append(row)
    out = pd.DataFrame(rows)
    out['p_bh'] = np.nan
    for fam, d in out[out['family'].notna()].groupby('family'):
        out.loc[d.index, 'p_bh'] = bh_adjust(d[d['family_p'].iloc[0]].to_numpy())
    return out


def criteria(S):
    res = {}
    for layer in LAYERS:
        L = S[S.layer == layer]
        r = {}
        for name in ('mass_mean_direction', 'logistic_probe_direction'):
            pos = L[(L.intervention == name) & (L.sign == 1)].iloc[0]
            neg = L[(L.intervention == name) & (L.sign == -1)].iloc[0]
            r[f'T1_{name}'] = bool(pos.p_bh <= ALPHA and np.sign(neg.effect) == -np.sign(pos.effect))
        for name in ('sae_reconstruction_direction', 'sae_top1_direction', 'sae_top10_direction'):
            pos = L[(L.intervention == name) & (L.sign == 1)].iloc[0]
            r[f'T2_{name}'] = bool(pos.p_bh <= ALPHA)
            r[f'T2_{name}_effect'] = float(pos.effect)
        res[str(layer)] = r
    return res


if __name__ == '__main__':
    main()
