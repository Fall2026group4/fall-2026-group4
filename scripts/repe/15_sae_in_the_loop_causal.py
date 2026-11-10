#!/usr/bin/env python3
"""SAE-in-the-loop causal test of the honest-persona shift, layers 2 / 6 / 10 (roadmap C1 + C3).

Two phases, run in order from the repository root (a GPU is needed for the test phase; --smoke runs anywhere):
    python scripts/repe/15_sae_in_the_loop_causal.py --phase validation
    python scripts/repe/15_sae_in_the_loop_causal.py --phase test
    (add --smoke to either phase for a 20-control pipeline check written to .../smoke/)

WHY. Script 13 added a *direction* to GPT-2 and compared it with random directions. It never put an SAE in the
forward pass, it ran at layer 6 only, on the historical SAEs, and its effects were ~0.5 % of the persona shift, so
its null has little power (scripts/repe/14_causal_reanalysis.py). Here the SAE is in the loop and the intervention
has the model's own natural size, so the effects are large enough to measure.

SETUP (same as script 13 unless stated).
  Model / site   GPT-2 Small, blocks.{L}.hook_resid_pre, FINAL prompt token, L in LAYERS.
  Prompts        RepE persona prompts; every intervention is applied to the UNTRUTHFUL prompt of a held-out
                 statement; the HONEST prompt of the same statement gives the behavioural target.
  SAEs           the 27 uninterrupted OWT SAEs at these layers: {relu, topk, jumprelu} x seeds {42, 1042, 2042}
                 (src/sae/checkpoint_registry.py:uninterrupted_checkpoint), d_sae 3072, target L0 128.
  Outcome        E(v) = 1 - sum_i KL(p_H,i || p_int,i) / sum_i KL(p_H,i || p_U,i), next-token distribution at the
                 final position. E = 0: no change; E = 1: the honest-persona output. Higher is better. Category: causal.

NOTATION for one statement: x_U, x_H = final-token activations of the untruthful / honest prompt;
  z = enc(x) SAE latents; D = decoder matrix; x_hat = D z + b; e = x - x_hat (the SAE error term).
  Exact identity (asserted at run time):  x_H = x_U + D (z_H - z_U) + (e_H - e_U).

PART A  graded positive control (power of the readout).
  x' = x_U + lam (x_H - x_U) for lam in LAMBDA_GRID. lam = 1 is script 13's positive control (activation patch).
  Control per lam: N random unit directions r, x' = x_U + lam ||x_H - x_U|| r (per-statement norm matched).
  Reported: E(lam), empirical p, and the minimum detectable dose = smallest lam whose p <= 0.05 for it and every
  larger lam. This states how small an effect this readout can see.

PART B  SAE-in-the-loop latent patching (the SAE is the causal variable; nothing is tuned).
  latent_all   x' = x_U + D (z_H - z_U)            every latent set to its honest value, SAE error kept from x_U
  error_term   x' = x_U + (e_H - e_U)              only what the SAE fails to reconstruct is patched
  latent_topk  x' = x_U + D_S (z_H - z_U)_S        only the k "honesty latents" S, k in TOPK_GRID
               S = k latents with the largest |mean_train(z | honest) - mean_train(z | untruthful)| (TRAIN rows only)
  Controls (N each):
    gaussian_matched         random unit direction scaled per statement to the norm of the target's own change
                             (for latent_all, error_term and each latent_topk)
    random_latents           the same patch on k random alive latents                       (latent_topk only)
    activation_matched       the same patch on k random alive latents whose mean train activation is close to that
                             of the selected ones (rank window +-MATCH_WINDOW)              (latent_topk only)
  Derived: mediated share = E(latent_all) / E(activation patch); concentration = E(latent_topk) / E(latent_all).

PART C  direction steering with both signs (script 13's protocol on the uninterrupted SAEs, plus a sign check).
  Strength s_L = multiple of ||train mean difference|| chosen on VALIDATION as the grid value maximising E of the
  raw honesty direction (script 13's rule), per layer. Targets, each at +s and -s: raw direction, SAE
  reconstruction direction, top-1 latent direction, top-10 latent direction. Controls: N Gaussian unit directions;
  per SAE N random single alive latents (random sign) and N random 10-latent sets carrying the target coefficients.

STATISTICS. One-sided empirical p = (1 + #{controls >= target}) / (1 + N) and two-sided
  p = min(1, 2 min(p_upper, p_lower)). Benjamini-Hochberg within each pre-registered family (written to
  preregistration.json). 95 % bootstrap CI over statements (2000 resamples, seed 0). SAE seeds are reported one by
  one and as mean +- sd; a claim about an architecture at a layer needs the same sign in all three seeds.

LEAKAGE GUARDS. Latents and directions are selected on TRAIN rows only; the strength on VALIDATION only; TEST rows
  are only scored, once, by --phase test after preregistration.json exists. Existing outputs are never overwritten.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LAYERS = (2, 6, 10)
ARCHS = ('relu', 'topk', 'jumprelu')
SEEDS = (42, 1042, 2042)
ACTS = REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt'
PAIRS = REPO / 'data/repe/persona_pairs.csv'
OUT = REPO / 'results/repe/steering/sae_in_the_loop'
N_CONTROLS = 600                  # 1/601 < 0.05/27: one test can pass Bonferroni over a 27-test family
N_CONTROLS_SMOKE = 20
CONTROL_SEED = 20261110           # fixed a priori (date this protocol was written)
LAMBDA_GRID = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
TOPK_GRID = (1, 10, 50)
MATCH_WINDOW = 25                 # activation-matched control: +-25 ranks in mean train activation
STRENGTH_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # = script 13
TOPK_SET = 10                     # = script 13 (Part C top-10 direction)
N_BOOT = 2000
ALPHA = 0.05
PARTS = ('graded', 'latent', 'steer')


# ---------------------------------------------------------------- pure functions (unit-tested)
def kl_rows(logp_ref, logp_other):
    """Row-wise KL(p_ref || p_other) from log-probabilities [n, vocab]."""
    return (logp_ref.exp() * (logp_ref - logp_other)).sum(-1)


def recovery(kl_after, kl_before):
    """E = 1 - sum KL_after / sum KL_before (aggregate ratio; robust to tiny per-item KLs)."""
    return 1.0 - float(np.sum(kl_after)) / float(np.sum(kl_before))


def empirical_p_upper(target, controls):
    controls = np.asarray(controls, dtype=float)
    return (1 + int(np.sum(controls >= target))) / (1 + len(controls))


def empirical_p_lower(target, controls):
    controls = np.asarray(controls, dtype=float)
    return (1 + int(np.sum(controls <= target))) / (1 + len(controls))


def empirical_p_two_sided(target, controls):
    return min(1.0, 2 * min(empirical_p_upper(target, controls), empirical_p_lower(target, controls)))


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values, same order as the input."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p.copy()
    if not np.isfinite(p).all():
        raise ValueError('bh_adjust: non-finite p-value')
    m = p.size
    order = np.argsort(p, kind='stable')
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def bootstrap_recovery_ci(kl_after, kl_before, n=N_BOOT, seed=0):
    kl_after, kl_before = np.asarray(kl_after, dtype=float), np.asarray(kl_before, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl_before), size=(n, len(kl_before)))
    vals = 1.0 - kl_after[idx].sum(1) / kl_before[idx].sum(1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_ratio_ci(kl_num, kl_den, kl_before, n=N_BOOT, seed=0):
    """CI of E(num) / E(den) over statements (same resamples); NaN bounds if E(den) changes sign in the resamples."""
    kl_num, kl_den, kl_before = (np.asarray(v, dtype=float) for v in (kl_num, kl_den, kl_before))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl_before), size=(n, len(kl_before)))
    e_num = 1.0 - kl_num[idx].sum(1) / kl_before[idx].sum(1)
    e_den = 1.0 - kl_den[idx].sum(1) / kl_before[idx].sum(1)
    point = recovery(kl_num, kl_before) / recovery(kl_den, kl_before)
    if (e_den <= 0).any():
        return point, float('nan'), float('nan')
    r = e_num / e_den
    return point, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def unit(v):
    return v / v.norm()


def select_honesty_latents(z_train, labels_train, k):
    """Indices (descending |difference|) and signed differences of the k latents with the largest
    |mean(z | honest) - mean(z | untruthful)| on TRAIN rows. labels: 1 honest, 0 untruthful."""
    labels_train = torch.as_tensor(labels_train)
    dmu = z_train[labels_train == 1].mean(0) - z_train[labels_train == 0].mean(0)
    top = torch.argsort(dmu.abs(), descending=True, stable=True)[:k]
    return top, dmu


def alive_latents(z_train):
    return torch.where((z_train > 0).any(0))[0]


def activation_rank_order(alive, mean_act):
    """Alive latents sorted by mean train activation (ascending); ties broken by index."""
    return sorted((int(a) for a in alive), key=lambda j: (float(mean_act[j]), j))


def activation_matched_set(selected, alive, mean_act, generator, window=MATCH_WINDOW, order=None):
    """For each selected latent, draw one alive latent whose rank in mean train activation is within +-window of
    it; never a selected latent, never the same latent twice. Returns a LongTensor of len(selected).
    ``order`` = activation_rank_order(alive, mean_act), passed in when many sets are drawn."""
    selected = [int(s) for s in selected]
    order = activation_rank_order(alive, mean_act) if order is None else order
    pos = {j: i for i, j in enumerate(order)}
    banned, out = set(selected), []
    for s in selected:
        if s not in pos:
            raise ValueError(f'selected latent {s} is not alive on train')
        w = window
        while True:
            lo, hi = max(0, pos[s] - w), min(len(order), pos[s] + w + 1)
            pool = [j for j in order[lo:hi] if j not in banned]
            if pool or (lo == 0 and hi == len(order)):
                break
            w *= 2
        if not pool:
            raise ValueError('not enough alive latents for an activation-matched control')
        j = pool[int(torch.randint(len(pool), (1,), generator=generator))]
        out.append(j)
        banned.add(j)
    return torch.tensor(out, dtype=torch.long)


def random_latent_set(alive, k, generator, exclude=()):
    pool = torch.tensor([int(a) for a in alive if int(a) not in set(int(e) for e in exclude)], dtype=torch.long)
    if len(pool) < k:
        raise ValueError('not enough alive latents')
    return pool[torch.randperm(len(pool), generator=generator)[:k]]


def latent_patch_delta(z_h, z_u, decoder_weight, idx=None):
    """D_S (z_H - z_U)_S for every statement. decoder_weight: [d_in, d_sae]. idx None = all latents."""
    dz = z_h - z_u
    if idx is None:
        return dz @ decoder_weight.T
    idx = torch.as_tensor(idx, dtype=torch.long, device=dz.device)
    return dz[:, idx] @ decoder_weight[:, idx].T


def decomposition_error(x_u, x_h, rec_u, rec_h, z_u, z_h, decoder_weight):
    """max |x_H - (x_U + D (z_H - z_U) + (e_H - e_U))|; ~0 up to float error for any linear-decoder SAE."""
    full = x_u + latent_patch_delta(z_h, z_u, decoder_weight) + ((x_h - rec_h) - (x_u - rec_u))
    return float((full - x_h).abs().max())


def norm_matched_delta(norms, direction):
    """Per-statement delta with the given norms [n] along one unit direction [d]."""
    return norms[:, None] * direction[None, :]


def minimum_detectable_dose(lams, pvals, alpha=ALPHA):
    """Smallest lam such that it and every larger lam have p <= alpha; NaN if the largest lam fails."""
    order = np.argsort(lams)
    lams, pvals = np.asarray(lams, dtype=float)[order], np.asarray(pvals, dtype=float)[order]
    best = float('nan')
    for lam, p in zip(lams[::-1], pvals[::-1]):
        if p <= alpha:
            best = float(lam)
        else:
            break
    return best


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- model running
class Runner:
    """Final-token interventions at one hook (script 13's Runner with the hook as a parameter)."""

    def __init__(self, model, prompts, hook, device):
        self.model, self.hook = model, hook
        toks = [model.to_tokens(p, prepend_bos=True, move_to_device=False)[0] for p in prompts]
        self.lengths = torch.tensor([t.numel() for t in toks])
        T = int(self.lengths.max())
        pad = model.tokenizer.eos_token_id
        self.tokens = torch.full((len(toks), T), pad, dtype=torch.long)
        for i, t in enumerate(toks):
            self.tokens[i, :t.numel()] = t   # right padding: causal model, final index unaffected
        self.tokens = self.tokens.to(device)
        self.final = (self.lengths - 1).to(device)
        self.rows = torch.arange(len(toks), device=device)

    @torch.inference_mode()
    def run(self, delta=None):
        """delta: [d] or [n, d] added to the final-token residual at the hook. Returns final-position log-probs."""
        def fn(resid, hook):
            resid[self.rows, self.final] = resid[self.rows, self.final] + delta.to(resid.dtype)
            return resid
        hooks = [] if delta is None else [(self.hook, fn)]
        logits = self.model.run_with_hooks(self.tokens, fwd_hooks=hooks)
        return logits[self.rows, self.final].float().log_softmax(-1)

    @torch.inference_mode()
    def final_resid(self):
        _, cache = self.model.run_with_cache(self.tokens, names_filter=lambda n: n == self.hook)
        return cache[self.hook][self.rows, self.final].float()


def control_seed(layer, tag):
    """Deterministic, order-independent seed per (layer, control group)."""
    h = hashlib.sha256(f'{CONTROL_SEED}|{layer}|{tag}'.encode()).digest()
    return CONTROL_SEED + int.from_bytes(h[:4], 'little') % 1_000_000


def load_inputs():
    pairs = pd.read_csv(PAIRS)
    acts = torch.load(ACTS, map_location='cpu', weights_only=True)
    if list(acts['personas']) != pairs['persona'].tolist():
        raise SystemExit('activation/prompt order mismatch')
    return pairs, acts


def split_runners(model, pairs, split, hook, device):
    sub = pairs[pairs['split'] == split]
    U = sub[sub['persona'] == 'untruthful'].sort_values('statement_id')
    H = sub[sub['persona'] == 'honest'].sort_values('statement_id')
    if not (U['statement_id'].values == H['statement_id'].values).all():
        raise SystemExit('honest / untruthful rows are not paired')
    return U, Runner(model, U['prompt'].tolist(), hook, device), Runner(model, H['prompt'].tolist(), hook, device)


def check_cache(acts, layer, split, runU):
    """Cached activations must equal freshly computed ones at the hook (guards hook / position bugs)."""
    splits, labels = np.array(acts['splits']), acts['persona_labels'].numpy()
    m = (splits == split) & (labels == 0)
    cached = acts['activations'][layer].float()[np.where(m)[0]]
    cached = cached[np.argsort(np.array(acts['statement_ids'])[m], kind='stable')]
    err = float((runU.final_resid().cpu() - cached).abs().max())
    if err >= 1e-3:
        raise SystemExit(f'layer {layer}: hook/position mismatch with cached activations ({err})')
    return err


def train_direction(acts, layer):
    X = acts['activations'][layer].float()
    labels, train = acts['persona_labels'].numpy(), np.array(acts['splits']) == 'train'
    d = X[train & (labels == 1)].mean(0) - X[train & (labels == 0)].mean(0)
    return d, X[train], labels[train]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--smoke', action='store_true', help=f'{N_CONTROLS_SMOKE} controls, 1 seed, writes to smoke/')
    ap.add_argument('--parts', nargs='+', choices=PARTS, default=list(PARTS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.device != 'cuda' and not args.smoke:
        raise SystemExit('CUDA required for the full run (use --smoke for a CPU pipeline check)')
    from transformer_lens import HookedTransformer
    from src.evaluation import faithfulness_metrics as fm
    from src.sae.checkpoint_registry import uninterrupted_checkpoint

    device = torch.device(args.device)
    n_controls = N_CONTROLS_SMOKE if args.smoke else N_CONTROLS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    out = (OUT / 'smoke') if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / 'preregistration.json'
    torch.manual_seed(0)

    pairs, acts = load_inputs()
    ckpts = {(l, a, s): uninterrupted_checkpoint(l, a, s) for l in LAYERS for a in ARCHS for s in seeds}
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------ validation: strengths + preregistration
    if args.phase == 'validation':
        if prereg_path.exists():
            raise SystemExit(f'{prereg_path} exists; refusing to overwrite.')
        rows, chosen = [], {}
        for layer in LAYERS:
            hook = f'blocks.{layer}.hook_resid_pre'
            U, runU, runH = split_runners(model, pairs, 'validation', hook, device)
            check_cache(acts, layer, 'validation', runU)
            d_raw, _, _ = train_direction(acts, layer)
            base_norm, d_hat = float(d_raw.norm()), unit(d_raw).to(device)
            logpH, logpU = runH.run(), runU.run()
            kl0 = kl_rows(logpH, logpU).cpu().numpy()
            gen = torch.Generator().manual_seed(control_seed(layer, 'validation'))
            val_rand = [unit(torch.randn(768, generator=gen)).to(device) for _ in range(50)]
            layer_rows = []
            for mult in STRENGTH_GRID:
                s = mult * base_norm
                e = recovery(kl_rows(logpH, runU.run(s * d_hat)).cpu().numpy(), kl0)
                ctrl = [recovery(kl_rows(logpH, runU.run(s * r)).cpu().numpy(), kl0) for r in val_rand]
                layer_rows.append(dict(layer=layer, strength_multiple=mult, strength_norm=s, E_raw_direction=e,
                                       control_mean=float(np.mean(ctrl)), control_sd=float(np.std(ctrl, ddof=1)),
                                       control_max=float(np.max(ctrl))))
                print(layer_rows[-1], flush=True)
            best = max(layer_rows, key=lambda r: r['E_raw_direction'])
            x_u, x_h = runU.final_resid(), runH.final_resid()
            e_patch = recovery(kl_rows(logpH, runU.run(x_h - x_u)).cpu().numpy(), kl0)
            chosen[str(layer)] = dict(strength_multiple=best['strength_multiple'], strength_norm=best['strength_norm'],
                                      mean_difference_norm_train=base_norm,
                                      validation_baseline_mean_KL=float(kl0.mean()),
                                      validation_activation_patch_E=e_patch, n_validation_statements=len(U))
            rows += layer_rows
        pd.DataFrame(rows).to_csv(out / 'validation_strength_selection.csv', index=False)
        n_sae = len(LAYERS) * len(ARCHS) * len(seeds)
        prereg = dict(
            written_utc=datetime.now(timezone.utc).isoformat(),
            script='scripts/repe/15_sae_in_the_loop_causal.py', script_sha256=sha256(__file__),
            pairs_sha256=sha256(PAIRS), activations_sha256=sha256(ACTS),
            checkpoint_sha256={f'{l}/{a}/{s}': sha256(p) for (l, a, s), p in ckpts.items()},
            layers=list(LAYERS), architectures=list(ARCHS), sae_seeds=list(seeds),
            hook='blocks.{layer}.hook_resid_pre', position='final prompt token',
            primary_outcome='E = 1 - sum KL(p_H||p_int) / sum KL(p_H||p_U), next-token at final position; higher is better',
            n_controls=n_controls, control_seed=CONTROL_SEED, lambda_grid=list(LAMBDA_GRID), topk_grid=list(TOPK_GRID),
            activation_match_window=MATCH_WINDOW, steering_strength_per_layer=chosen,
            steering_selection_rule='strength multiple maximising E of the raw honesty direction on validation (script 13 rule)',
            latent_selection='largest |mean_train(z|honest) - mean_train(z|untruthful)|, TRAIN rows only',
            significance='one-sided empirical p=(1+#controls>=target)/(1+n); two-sided p=min(1,2*min(upper,lower))',
            multiple_comparisons='Benjamini-Hochberg within each family below, alpha 0.05',
            families={
                'A_graded_patch': f'{len(LAYERS) * len(LAMBDA_GRID)} tests: activation patch at each lambda and layer vs norm-matched Gaussian (one-sided)',
                'B1_latent_all': f'{n_sae} tests: latent_all vs norm-matched Gaussian (one-sided)',
                'B2_error_term': f'{n_sae} tests: error_term vs norm-matched Gaussian (one-sided)',
                'B3_latent_topk_vs_random': f'{n_sae * len(TOPK_GRID)} tests: latent_topk vs random latents (one-sided)',
                'B4_latent_topk_vs_matched': f'{n_sae * len(TOPK_GRID)} tests: latent_topk vs activation-matched latents (one-sided)',
                'C1_raw_direction': f'{len(LAYERS)} tests: raw honesty direction (+s) vs Gaussian (one-sided)',
                'C2_sae_directions': f'{n_sae * 3} tests: SAE reconstruction / top-1 / top-10 direction (+s) vs their controls (two-sided)',
            },
            criteria={
                'H1_latents_mediate': 'for an architecture at a layer: latent_all BH-adjusted p <= 0.05 in all SAE seeds',
                'H2_concentration': 'latent_topk BH-adjusted p <= 0.05 vs activation-matched latents in all SAE seeds; report E(topk)/E(latent_all)',
                'H3_error_term': 'descriptive: E(error_term) vs E(latent_all) per architecture and layer (no threshold)',
                'H4_sign': 'top-1 latent direction: two-sided BH-adjusted p <= 0.05 with E < control mean at +s in all seeds = '
                           'replication of the layer-6 "away from honest output" observation; report E at -s beside it',
                'power': 'minimum detectable dose from Part A is reported for every layer before any null is interpreted',
            },
            ci=f'bootstrap over statements, {N_BOOT} resamples, seed 0',
            test_rows='scored once by --phase test; never used for selection',
            status_of_results='confirmed test result for the pre-registered families; anything else exploratory')
        prereg_path.write_text(json.dumps(prereg, indent=2))
        print(json.dumps({k: v for k, v in prereg.items() if k != 'checkpoint_sha256'}, indent=2))
        return

    # ------------------------------------------------------------ test
    if not prereg_path.exists():
        raise SystemExit('Run --phase validation first (preregistration missing).')
    if (out / 'interventions.csv').exists():
        raise SystemExit(f'{out}/interventions.csv exists; refusing to overwrite.')
    prereg = json.loads(prereg_path.read_text())
    if prereg['script_sha256'] != sha256(__file__):
        raise SystemExit('script changed since preregistration; re-run --phase validation in a new folder.')
    for (l, a, s), p in ckpts.items():
        if prereg['checkpoint_sha256'][f'{l}/{a}/{s}'] != sha256(p):
            raise SystemExit(f'checkpoint {p} changed since preregistration')

    records, kl_store, graded_rows, checks = [], {}, [], []
    for layer in LAYERS:
        hook = f'blocks.{layer}.hook_resid_pre'
        U, runU, runH = split_runners(model, pairs, 'test', hook, device)
        cache_err = check_cache(acts, layer, 'test', runU)
        logpH, logpU = runH.run(), runU.run()
        kl0 = kl_rows(logpH, logpU).cpu().numpy()
        kl_store[f'L{layer}__baseline'] = kl0
        x_u, x_h = runU.final_resid(), runH.final_resid()
        diff = x_h - x_u
        zero_err = float((runU.run(torch.zeros_like(x_u)) - logpU).abs().max())   # hook with zero delta = no-op
        checks.append(dict(layer=layer, check='cache_vs_fresh_max_abs', value=cache_err))
        checks.append(dict(layer=layer, check='zero_delta_logprob_max_abs', value=zero_err))

        def record(part, kind, name, group, arch, seed, detail, delta, sign=0, store=None, _layer=layer, _n=len(U)):
            kl = kl_rows(logpH, runU.run(delta)).cpu().numpy()
            records.append(dict(model='gpt2-small', layer=_layer, part=part, intervention_kind=kind, intervention=name,
                                control_group=group, sae_architecture=arch, sae_seed=seed, detail=detail, sign=sign,
                                mean_delta_norm=float(delta.norm(dim=-1).mean()) if delta.ndim == 2 else float(delta.norm()),
                                n_statements=_n, baseline_sum_KL=float(kl0.sum()), intervened_sum_KL=float(kl.sum()),
                                effect_E=recovery(kl, kl0)))
            if store:
                kl_store[store] = kl
            return records[-1]['effect_E']

        def gaussians(tag):
            g = torch.Generator().manual_seed(control_seed(layer, tag))
            return [unit(torch.randn(768, generator=g)).to(device) for _ in range(n_controls)]

        # ---------------- Part A: graded positive control
        if 'graded' in args.parts:
            rs = gaussians('graded')
            norms = diff.norm(dim=-1)
            lams, ps = [], []
            for lam in LAMBDA_GRID:
                grp = f'L{layer}__graded_lam{lam}'
                e = record('A', 'target', 'activation_patch', grp, '', '', f'lambda={lam}', lam * diff,
                           store=f'L{layer}__patch_lam{lam}')
                ctrl = [record('A', 'control', 'gaussian_matched', grp, '', '', f'lambda={lam}',
                               norm_matched_delta(lam * norms, r)) for r in rs]
                p = empirical_p_upper(e, ctrl)
                lams.append(lam)
                ps.append(p)
                graded_rows.append(dict(layer=layer, lam=lam, mean_patch_norm=float(lam * norms.mean()), effect_E=e,
                                        control_mean=float(np.mean(ctrl)), control_q95=float(np.percentile(ctrl, 95)),
                                        control_max=float(np.max(ctrl)), p_upper=p))
            mdd = minimum_detectable_dose(lams, ps)
            for r in graded_rows:
                if r['layer'] == layer:
                    r['minimum_detectable_dose'] = mdd
            print(f'layer {layer}: graded patch done, minimum detectable dose {mdd}', flush=True)
        elif f'L{layer}__patch_lam1.0' not in kl_store:
            record('A', 'target', 'activation_patch', f'L{layer}__patch_only', '', '', 'lambda=1.0', diff,
                   store=f'L{layer}__patch_lam1.0')

        # ---------------- Part C (raw direction): shared Gaussian controls at strength s
        d_raw, X_tr, y_tr = train_direction(acts, layer)
        s = prereg['steering_strength_per_layer'][str(layer)]['strength_norm']
        d_hat = unit(d_raw).to(device)
        if 'steer' in args.parts:
            grp = f'L{layer}__steer_gaussian'
            for sign in (+1, -1):
                record('C', 'target', 'raw_honesty_direction', grp, '', '', 'mean_diff_train', sign * s * d_hat,
                       sign=sign, store=f'L{layer}__raw_dir_{"pos" if sign > 0 else "neg"}')
            for r in gaussians('steer'):
                record('C', 'control', 'gaussian', grp, '', '', '', s * r)

        # ---------------- per SAE: Parts B and C
        for arch in ARCHS:
            for seed in seeds:
                sae, _ = fm.load_sae(ckpts[(layer, arch, seed)], device)
                key = f'L{layer}__{arch}_s{seed}'
                with torch.inference_mode():
                    rec_u, z_u, _ = sae(x_u)
                    rec_h, z_h, _ = sae(x_h)
                    rec_tr, z_tr, _ = sae(X_tr.to(device))
                D = sae.decoder.weight                                    # [768, d_sae], unit columns
                dec_err = decomposition_error(x_u, x_h, rec_u, rec_h, z_u, z_h, D)
                scale = float(x_h.abs().max())
                if dec_err > 1e-3 * max(1.0, scale):
                    raise SystemExit(f'{key}: decomposition identity violated ({dec_err}, scale {scale})')
                checks.append(dict(layer=layer, check=f'decomposition_max_abs_{arch}_s{seed}', value=dec_err))
                alive = alive_latents(z_tr).cpu()
                mean_act = z_tr.mean(0).cpu()
                rank_order = activation_rank_order(alive.tolist(), mean_act)
                top, dmu = select_honesty_latents(z_tr.cpu(), y_tr, max(max(TOPK_GRID), TOPK_SET))
                dmu = dmu.to(device)

                if 'latent' in args.parts:
                    targets = {'latent_all': latent_patch_delta(z_h, z_u, D),
                               'error_term': (x_h - rec_h) - (x_u - rec_u)}
                    for k in TOPK_GRID:
                        targets[f'latent_top{k}'] = latent_patch_delta(z_h, z_u, D, top[:k])
                    for name, delta in targets.items():
                        grp = f'{key}__{name}__gaussian'
                        record('B', 'target', name, grp, arch, seed,
                               'latents_' + '-'.join(map(str, top[:int(name[10:])].tolist())) if name.startswith('latent_top') else '',
                               delta, store=f'{key}__{name}')
                        nrm = delta.norm(dim=-1)
                        for r in gaussians(f'{arch}_{seed}_{name}'):
                            record('B', 'control', 'gaussian_matched', grp, arch, seed, '', norm_matched_delta(nrm, r))
                    for k in TOPK_GRID:
                        name = f'latent_top{k}'
                        g = torch.Generator().manual_seed(control_seed(layer, f'{arch}_{seed}_{name}_latents'))
                        for _ in range(n_controls):
                            idx = random_latent_set(alive, k, g, exclude=top[:k].tolist())
                            record('B', 'control', 'random_latents', f'{key}__{name}__random', arch, seed,
                                   '', latent_patch_delta(z_h, z_u, D, idx))
                        for _ in range(n_controls):
                            idx = activation_matched_set(top[:k].tolist(), None, mean_act, g, order=rank_order)
                            record('B', 'control', 'activation_matched_latents', f'{key}__{name}__matched', arch, seed,
                                   '', latent_patch_delta(z_h, z_u, D, idx))

                if 'steer' in args.parts:
                    y_t = torch.as_tensor(y_tr)
                    recon_dir = unit(rec_tr[(y_t == 1).to(device)].mean(0) - rec_tr[(y_t == 0).to(device)].mean(0))
                    top1_dir = unit(torch.sign(dmu[top[0]]) * D[:, top[0]])
                    top10 = top[:TOPK_SET].to(device)
                    top10_dir = unit(D[:, top10] @ dmu[top10])
                    for name, v, grp in (('sae_reconstruction_direction', recon_dir, f'L{layer}__steer_gaussian'),
                                         ('sae_top1_direction', top1_dir, f'{key}__steer_feat1'),
                                         ('sae_top10_direction', top10_dir, f'{key}__steer_feat10')):
                        for sign in (+1, -1):
                            record('C', 'target', name, grp, arch, seed, f'cos_raw={float(v @ d_hat):.4f}',
                                   sign * s * v, sign=sign, store=f'{key}__{name}_{"pos" if sign > 0 else "neg"}')
                    g = torch.Generator().manual_seed(control_seed(layer, f'{arch}_{seed}_steer'))
                    for _ in range(n_controls):
                        j = int(alive[torch.randint(len(alive), (1,), generator=g)])
                        sg = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                        record('C', 'control', 'random_latent_direction', f'{key}__steer_feat1', arch, seed,
                               f'latent_{j}', s * unit(sg * D[:, j]))
                    for _ in range(n_controls):
                        idx = alive[torch.randperm(len(alive), generator=g)[:TOPK_SET]].to(device)
                        record('C', 'control', 'random_latent_set_direction', f'{key}__steer_feat10', arch, seed,
                               '', s * unit(D[:, idx] @ dmu[top10]))
                del sae
                print(f'done {key}', flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out / 'interventions.csv', index=False)
    pd.DataFrame(graded_rows).to_csv(out / 'graded_positive_control.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'pipeline_checks.csv', index=False)
    np.savez_compressed(out / 'per_statement_kl_targets.npz', **kl_store)
    summary = summarise(df, kl_store)
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'test_run_info.json').write_text(json.dumps(dict(
        preregistration_script_sha256=prereg['script_sha256'], parts=args.parts, n_controls=n_controls,
        n_interventions=len(df), torch=torch.__version__, python=platform.python_version(),
        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        written_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    show = ['layer', 'part', 'intervention', 'sae_architecture', 'sae_seed', 'sign', 'effect_E', 'E_ci95_low',
            'E_ci95_high', 'control_group_kind', 'p_upper', 'p_two_sided', 'p_bh', 'share_of_activation_patch']
    print(summary[[c for c in show if c in summary]].round(4).to_string(index=False))


def family_of(row):
    """Pre-registered family and which p-value it uses (None = not in a family; reported unadjusted)."""
    if row['part'] == 'A':
        return 'A_graded_patch', 'p_upper'
    if row['part'] == 'B':
        if row['intervention'] == 'latent_all':
            return 'B1_latent_all', 'p_upper'
        if row['intervention'] == 'error_term':
            return 'B2_error_term', 'p_upper'
        if row['control_group_kind'] == 'random_latents':
            return 'B3_latent_topk_vs_random', 'p_upper'
        if row['control_group_kind'] == 'activation_matched_latents':
            return 'B4_latent_topk_vs_matched', 'p_upper'
        return None, None                                   # latent_topk vs Gaussian: descriptive
    if row['sign'] != 1:
        return None, None                                   # -s rows: descriptive sign check
    if row['intervention'] == 'raw_honesty_direction':
        return 'C1_raw_direction', 'p_upper'
    return 'C2_sae_directions', 'p_two_sided'


def summarise(df, kl_store):
    """One row per (target, control group): effect, CI, empirical p-values, BH within family, derived shares."""
    ctrl = df[df['intervention_kind'] == 'control']
    by_group = {g: (d['effect_E'].to_numpy(), d['intervention'].iloc[0]) for g, d in ctrl.groupby('control_group')}
    rows = []
    for _, t in df[df['intervention_kind'] == 'target'].iterrows():
        layer = int(t['layer'])
        if t['part'] == 'A':
            store = f"L{layer}__patch_lam{t['detail'].split('=')[1]}"
            groups = [t['control_group']]
        elif t['part'] == 'B':
            key = f"L{layer}__{t['sae_architecture']}_s{t['sae_seed']}"
            store = f"{key}__{t['intervention']}"
            groups = [t['control_group']]
            if t['intervention'].startswith('latent_top'):
                groups += [f"{store}__random", f"{store}__matched"]
        else:
            tag = 'pos' if t['sign'] > 0 else 'neg'
            if t['intervention'] == 'raw_honesty_direction':
                store = f'L{layer}__raw_dir_{tag}'
            else:
                store = f"L{layer}__{t['sae_architecture']}_s{t['sae_seed']}__{t['intervention']}_{tag}"
            groups = [t['control_group']]
        kl0 = kl_store[f'L{layer}__baseline']
        lo, hi = bootstrap_recovery_ci(kl_store[store], kl0)
        patch_key = f'L{layer}__patch_lam1.0'
        for g in groups:
            if g not in by_group:
                continue
            c, kind = by_group[g]
            row = t.drop(labels=['intervention_kind']).to_dict()
            row.update(control_group=g, control_group_kind=kind, n_controls=len(c), E_ci95_low=lo, E_ci95_high=hi,
                       control_mean=float(c.mean()), control_sd=float(c.std(ddof=1)),
                       control_q95=float(np.percentile(c, 95)), control_max=float(c.max()),
                       z_vs_controls=float((t['effect_E'] - c.mean()) / c.std(ddof=1)),
                       p_upper=empirical_p_upper(t['effect_E'], c), p_lower=empirical_p_lower(t['effect_E'], c),
                       p_two_sided=empirical_p_two_sided(t['effect_E'], c))
            if patch_key in kl_store and t['part'] == 'B':
                pt, rlo, rhi = bootstrap_ratio_ci(kl_store[store], kl_store[patch_key], kl0)
                row.update(share_of_activation_patch=pt, share_ci95_low=rlo, share_ci95_high=rhi)
                all_key = f"L{layer}__{t['sae_architecture']}_s{t['sae_seed']}__latent_all"
                if t['intervention'].startswith('latent_top') and all_key in kl_store:
                    pt, rlo, rhi = bootstrap_ratio_ci(kl_store[store], kl_store[all_key], kl0)
                    row.update(share_of_latent_all=pt, share_of_latent_all_ci95_low=rlo, share_of_latent_all_ci95_high=rhi)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    fams = out.apply(family_of, axis=1, result_type='expand')
    out['family'], out['family_p'] = fams[0], fams[1]
    out['p_bh'] = np.nan
    for fam, d in out[out['family'].notna()].groupby('family'):
        pcol = d['family_p'].iloc[0]
        out.loc[d.index, 'p_bh'] = bh_adjust(d[pcol].to_numpy())
    return out


if __name__ == '__main__':
    main()
