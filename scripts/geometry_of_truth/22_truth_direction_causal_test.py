#!/usr/bin/env python3
"""Causal test on Geometry of Truth: do truth directions and SAE truth latents make GPT-2 prefer the TRUE
completion? (roadmap C4). Layers 4 / 6 / 8 / 10, the 36 uninterrupted SAEs at those layers, 600 controls.

Two phases, run in order from the repository root:
    python scripts/geometry_of_truth/22_truth_direction_causal_test.py --phase validation
    python scripts/geometry_of_truth/22_truth_direction_causal_test.py --phase test
    (add --smoke to either phase: 20 controls, 1 SAE seed, written to .../smoke/)

DATA. GoT `cities` only (each city has one true and one false statement, "The city of X is in Y." vs
  "... is in Z."); grouped splits from data/got/splits/cities_splits.csv. neg_cities, larger_than and smaller_than
  are not used: their truth is not a next-token preference in this form.
  For each city, the two statements are tokenized and the FIRST POSITION WHERE THEY DIFFER is found (normally the
  country token; for e.g. " South Africa" vs " South Korea" it is the token after " South"). The shared prefix
  before it is the prompt; c = the true statement's token there, w = the false statement's token.

OUTCOME (behavioural, higher is better). margin = log P(c | prefix) - log P(w | prefix) at the final prefix position.
  effect = mean over test pairs of (margin with intervention - margin without). Category: causal.
  The intervention is added to blocks.{L}.hook_resid_pre at the final prefix token only.

TARGETS (estimated on TRAIN rows of the cached statement-final activations, i.e. at a DIFFERENT position from the
  intervention: the truth signal exists only after the country is read. Whether it transfers to the prefix
  position is part of the question.)
  mass_mean_direction        unit mean(x | true) - mean(x | false)
  logistic_probe_direction   raw-space direction of the frozen original probe (StandardScaler + LogisticRegression,
                             C chosen on validation; scripts/shared/all_layer_frozen_probe_evaluation.py)
  per SAE: sae_reconstruction_direction (mass-mean of reconstructions), sae_top1_direction and sae_top10_direction
           (decoder columns of the latents with the largest |mean_train(z | true) - mean_train(z | false)|)
  All at strength s_L and -s_L. s_L = multiple of ||train mass-mean difference|| maximising the effect of the
  mass-mean direction on VALIDATION pairs (the rule of scripts/repe/13).

POSITIVE CONTROL (oracle, graded). u_i = unit(W_U[:, c_i] - W_U[:, w_i]) per pair, at lambda * s_L for lambda in
  LAMBDA_GRID. It uses the answer, so it must move the margin; its minimum detectable dose says how small an
  effect this readout can see.

CONTROLS (N each, norm s_L): Gaussian unit directions (for the two raw directions and the reconstruction direction);
  per SAE random alive latents with random sign (for top-1) and random 10-latent sets carrying the target
  coefficients (for top-10).

STATISTICS. One-sided empirical p = (1 + #{controls >= target}) / (1 + N); two-sided = min(1, 2 min(upper, lower)).
  BH within the pre-registered families (preregistration.json). 95 % bootstrap CI over test pairs (2000, seed 0).
  An architecture-level claim needs the same sign in all SAE seeds.

LEAKAGE GUARDS. Directions and latents from TRAIN rows; strength from VALIDATION pairs; TEST pairs scored once by
  --phase test after the preregistration exists, with the script and checkpoint hashes unchanged. Never overwrites.
"""
import argparse
import hashlib
import importlib.util
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

LAYERS = (4, 6, 8, 10)
ARCHS = ('relu', 'topk', 'jumprelu')
SEEDS = (42, 1042, 2042)
DATA = REPO / 'data/got/cities.csv'
SPLITS = REPO / 'data/got/splits/cities_splits.csv'
ACTS = REPO / 'data/got/activations/gpt2_small/cities_final_token.pt'
OUT = REPO / 'results/geometry_of_truth/causal_truth_steering'
N_CONTROLS = 600
N_CONTROLS_SMOKE = 20
CONTROL_SEED = 20261110
STRENGTH_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAMBDA_GRID = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
TOPK_SET = 10
N_BOOT = 2000
ALPHA = 0.05
BATCH = 256


def _load_loop():
    spec = importlib.util.spec_from_file_location('repe15', REPO / 'scripts/repe/15_sae_in_the_loop_causal.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


loop = _load_loop()           # shared, unit-tested statistics and latent helpers
empirical_p_upper, empirical_p_lower = loop.empirical_p_upper, loop.empirical_p_lower
empirical_p_two_sided, bh_adjust, unit = loop.empirical_p_two_sided, loop.bh_adjust, loop.unit
select_latents, alive_latents, minimum_detectable_dose = loop.select_honesty_latents, loop.alive_latents, loop.minimum_detectable_dose


# ---------------------------------------------------------------- pure functions (unit-tested)
def divergence(tokens_true, tokens_false):
    """(prefix, c, w) for two token lists: prefix = shared tokens before the first difference. None if identical
    or if one is a prefix of the other."""
    n = min(len(tokens_true), len(tokens_false))
    for i in range(n):
        if tokens_true[i] != tokens_false[i]:
            return list(tokens_true[:i]), int(tokens_true[i]), int(tokens_false[i])
    return None


def build_pairs(df, splits, encode):
    """One row per city with both statements: prefix tokens (no BOS), true / false next token, split.
    encode: str -> list[int] (no BOS). Pairs without a usable divergence are dropped and counted."""
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
        rows.append({'group_id': g, 'split': t['split'].iloc[0], 'prefix': div[0], 'c': div[1], 'w': div[2],
                     'true_statement': t['statement'].iloc[0], 'false_statement': f['statement'].iloc[0]})
    return pd.DataFrame(rows), dropped


def bootstrap_mean_ci(values, n=N_BOOT, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def probe_direction_raw(pipeline):
    scaler, clf = pipeline.steps[0][1], pipeline.steps[-1][1]
    return clf.coef_[0] / scaler.scale_


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- model running
class PrefixRunner:
    """Margins log P(c) - log P(w) at the final prefix token, with an optional delta added there at one hook."""

    def __init__(self, model, pairs, device):
        self.model, self.device = model, device
        bos = model.tokenizer.bos_token_id if getattr(model.tokenizer, 'bos_token_id', None) is not None else 50256
        seqs = [[bos] + list(p) for p in pairs['prefix']]
        self.lengths = torch.tensor([len(s) for s in seqs])
        T = int(self.lengths.max())
        self.tokens = torch.full((len(seqs), T), bos, dtype=torch.long)
        for i, s in enumerate(seqs):
            self.tokens[i, :len(s)] = torch.tensor(s)      # right padding; causal model: final index unaffected
        self.c = torch.tensor(pairs['c'].to_numpy(), device=device)
        self.w = torch.tensor(pairs['w'].to_numpy(), device=device)

    @torch.inference_mode()
    def margins(self, hook=None, delta=None):
        out = []
        for b in range(0, len(self.tokens), BATCH):
            tok = self.tokens[b:b + BATCH].to(self.device)
            fin = (self.lengths[b:b + BATCH] - 1).to(self.device)
            rows = torch.arange(len(tok), device=self.device)
            hooks = []
            if delta is not None:
                d = delta if delta.ndim == 1 else delta[b:b + BATCH]

                def fn(resid, hook, _d=d, _rows=rows, _fin=fin):
                    resid[_rows, _fin] = resid[_rows, _fin] + _d.to(resid.dtype)
                    return resid
                hooks = [(hook, fn)]
            logits = self.model.run_with_hooks(tok, fwd_hooks=hooks)
            lp = logits[rows, fin].float().log_softmax(-1)
            out.append((lp.gather(1, self.c[b:b + BATCH, None]) - lp.gather(1, self.w[b:b + BATCH, None]))[:, 0])
        return torch.cat(out).cpu().numpy()

    @torch.inference_mode()
    def final_resid(self, hook):
        outs = []
        for b in range(0, len(self.tokens), BATCH):
            tok = self.tokens[b:b + BATCH].to(self.device)
            _, cache = self.model.run_with_cache(tok, names_filter=lambda n: n == hook)
            fin = (self.lengths[b:b + BATCH] - 1).to(self.device)
            outs.append(cache[hook][torch.arange(len(tok), device=self.device), fin].float())
        return torch.cat(outs)


def control_seed(layer, tag):
    h = hashlib.sha256(f'{CONTROL_SEED}|got|{layer}|{tag}'.encode()).digest()
    return CONTROL_SEED + int.from_bytes(h[:4], 'little') % 1_000_000


def load_train_activations(layer):
    cache = torch.load(ACTS, map_location='cpu', weights_only=False)
    splits = pd.read_csv(SPLITS).set_index('row_id')
    row_ids = np.asarray(cache['row_ids'], dtype=np.int64)
    labels = np.asarray(cache['labels'], dtype=np.int64)
    sp = splits.reindex(row_ids)
    if sp.isna().any().any() or not np.array_equal(labels, sp['label'].to_numpy(dtype=np.int64)):
        raise SystemExit('cities activation cache and split file disagree')
    split = np.where(sp['split'].to_numpy() == 'val', 'validation', sp['split'].to_numpy())
    X = {int(k): v for k, v in cache['activations'].items()}[layer].float().numpy()
    return X, labels, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.device != 'cuda' and not args.smoke:
        raise SystemExit('CUDA required for the full run (use --smoke for a CPU pipeline check)')
    from transformer_lens import HookedTransformer
    from src.evaluation import faithfulness_metrics as fm
    from src.sae.checkpoint_registry import uninterrupted_checkpoint
    spec = importlib.util.spec_from_file_location('frozen_eval', REPO / 'scripts/shared/all_layer_frozen_probe_evaluation.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    device = torch.device(args.device)
    n_controls = N_CONTROLS_SMOKE if args.smoke else N_CONTROLS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    out = (OUT / 'smoke') if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / 'preregistration.json'
    torch.manual_seed(0)

    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    encode = lambda s: model.to_tokens(s, prepend_bos=False, move_to_device=False)[0].tolist()
    pairs, dropped = build_pairs(pd.read_csv(DATA), pd.read_csv(SPLITS), encode)
    ckpts = {(l, a, s): uninterrupted_checkpoint(l, a, s) for l in LAYERS for a in ARCHS for s in seeds}

    def directions(layer):
        X, y, split = load_train_activations(layer)
        tr = split == 'train'
        d_raw = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
        probes = ev.fit_original_probes(X, y, split)
        return X, y, split, d_raw, probe_direction_raw(probes['logistic'])

    # ------------------------------------------------------------ validation: strength + preregistration
    if args.phase == 'validation':
        if prereg_path.exists():
            raise SystemExit(f'{prereg_path} exists; refusing to overwrite.')
        val = pairs[pairs['split'] == 'val'].reset_index(drop=True)
        run = PrefixRunner(model, val, device)
        base = run.margins()
        rows, chosen = [], {}
        for layer in LAYERS:
            hook = f'blocks.{layer}.hook_resid_pre'
            _, _, _, d_raw, _ = directions(layer)
            norm = float(np.linalg.norm(d_raw))
            d_hat = unit(torch.tensor(d_raw, dtype=torch.float32)).to(device)
            g = torch.Generator().manual_seed(control_seed(layer, 'validation'))
            rand = [unit(torch.randn(768, generator=g)).to(device) for _ in range(50)]
            lrows = []
            for mult in STRENGTH_GRID:
                s = mult * norm
                e = float((run.margins(hook, s * d_hat) - base).mean())
                ctrl = [float((run.margins(hook, s * r) - base).mean()) for r in rand]
                lrows.append(dict(layer=layer, strength_multiple=mult, strength_norm=s, effect_mass_mean=e,
                                  control_mean=float(np.mean(ctrl)), control_sd=float(np.std(ctrl, ddof=1)),
                                  control_max=float(np.max(ctrl))))
                print(lrows[-1], flush=True)
            best = max(lrows, key=lambda r: r['effect_mass_mean'])
            chosen[str(layer)] = dict(strength_multiple=best['strength_multiple'], strength_norm=best['strength_norm'],
                                      mass_mean_difference_norm_train=norm)
            rows += lrows
        pd.DataFrame(rows).to_csv(out / 'validation_strength_selection.csv', index=False)
        n_sae = len(LAYERS) * len(ARCHS) * len(seeds)
        prereg = dict(
            written_utc=datetime.now(timezone.utc).isoformat(),
            script='scripts/geometry_of_truth/22_truth_direction_causal_test.py', script_sha256=sha256(__file__),
            data_sha256=sha256(DATA), splits_sha256=sha256(SPLITS), activations_sha256=sha256(ACTS),
            checkpoint_sha256={f'{l}/{a}/{s}': sha256(p) for (l, a, s), p in ckpts.items()},
            n_pairs={k: int(v) for k, v in pairs['split'].value_counts().items()}, pairs_dropped=int(dropped),
            validation_baseline_pair_accuracy=float((base > 0).mean()), validation_baseline_margin=float(base.mean()),
            layers=list(LAYERS), architectures=list(ARCHS), sae_seeds=list(seeds),
            hook='blocks.{layer}.hook_resid_pre', position='final prefix token (before the first differing token)',
            outcome='mean change of log P(true token) - log P(false token); higher is better',
            n_controls=n_controls, control_seed=CONTROL_SEED, lambda_grid=list(LAMBDA_GRID),
            strength_per_layer=chosen, strength_rule='multiple maximising the mass-mean direction effect on validation',
            families={
                'A_positive_control': f'{len(LAYERS) * len(LAMBDA_GRID)} tests: unembedding contrast at each lambda vs Gaussian (one-sided)',
                'C1_raw_directions': f'{2 * len(LAYERS)} tests: mass-mean and probe direction (+s) vs Gaussian (one-sided)',
                'C2_sae_directions': f'{3 * n_sae} tests: reconstruction / top-1 / top-10 direction (+s) vs their controls (two-sided)'},
            criteria={
                'T1_truth_direction_steers': 'a raw direction at a layer: BH-adjusted one-sided p <= 0.05 at +s, and the effect at -s has the opposite sign',
                'T2_sae_directions': 'an SAE direction for an architecture at a layer: BH-adjusted two-sided p <= 0.05 with the same sign in all SAE seeds',
                'power': 'minimum detectable dose of the positive control reported per layer before any null is read'},
            ci=f'bootstrap over test pairs, {N_BOOT} resamples, seed 0', test_rows='scored once by --phase test',
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
    test = pairs[pairs['split'] == 'test'].reset_index(drop=True)
    run = PrefixRunner(model, test, device)
    base = run.margins()
    W_U = model.W_U.detach().float()
    records, store, checks = [], {'baseline': base}, []

    for layer in LAYERS:
        hook = f'blocks.{layer}.hook_resid_pre'
        s = prereg['strength_per_layer'][str(layer)]['strength_norm']
        X, y, split, d_raw, d_probe = directions(layer)
        zero = float(np.abs(run.margins(hook, torch.zeros(768, device=device)) - base).max())
        checks.append(dict(layer=layer, check='zero_delta_margin_max_abs', value=zero))

        def record(part, kind, name, group, arch, seed, detail, delta, sign=0, key=None, _layer=layer):
            m = run.margins(hook, delta)
            eff = m - base
            records.append(dict(layer=_layer, part=part, intervention_kind=kind, intervention=name, control_group=group,
                                sae_architecture=arch, sae_seed=seed, detail=detail, sign=sign,
                                effect=float(eff.mean()), accuracy=float((m > 0).mean()),
                                baseline_accuracy=float((base > 0).mean()), n_pairs=len(base)))
            if key:
                store[key] = eff
            return records[-1]['effect']

        def gaussians(tag):
            g = torch.Generator().manual_seed(control_seed(layer, tag))
            return [unit(torch.randn(768, generator=g)).to(device) for _ in range(n_controls)]

        # A: graded positive control (oracle unembedding contrast)
        u = W_U[:, run.c].T - W_U[:, run.w].T
        u = (u / u.norm(dim=-1, keepdim=True)).to(device)
        rs = gaussians('positive')
        lams, ps = [], []
        for lam in LAMBDA_GRID:
            grp = f'L{layer}__positive_lam{lam}'
            e = record('A', 'target', 'unembedding_contrast', grp, '', '', f'lambda={lam}', lam * s * u,
                       key=f'L{layer}__positive_lam{lam}')
            ctrl = [record('A', 'control', 'gaussian', grp, '', '', f'lambda={lam}', lam * s * r) for r in rs]
            lams.append(lam)
            ps.append(empirical_p_upper(e, ctrl))
        checks.append(dict(layer=layer, check='minimum_detectable_dose', value=minimum_detectable_dose(lams, ps)))

        # C: raw directions with shared Gaussian controls
        grp = f'L{layer}__gaussian'
        for name, v in (('mass_mean_direction', d_raw), ('logistic_probe_direction', d_probe)):
            v = unit(torch.tensor(v, dtype=torch.float32)).to(device)
            for sign in (1, -1):
                record('C', 'target', name, grp, '', '', '', sign * s * v, sign=sign,
                       key=f'L{layer}__{name}_{"pos" if sign > 0 else "neg"}')
        for r in gaussians('steer'):
            record('C', 'control', 'gaussian', grp, '', '', '', s * r)

        tr = split == 'train'
        d_hat = unit(torch.tensor(d_raw, dtype=torch.float32)).to(device)
        for arch in ARCHS:
            for seed in seeds:
                sae, _ = fm.load_sae(ckpts[(layer, arch, seed)], device)
                key = f'L{layer}__{arch}_s{seed}'
                with torch.inference_mode():
                    rec, z, _ = sae(torch.tensor(X[tr], device=device))
                ytr = torch.tensor(y[tr], device=device)
                D = sae.decoder.weight
                alive = alive_latents(z).cpu()
                top, dmu = select_latents(z.cpu(), y[tr], TOPK_SET)
                dmu = dmu.to(device)
                top_d = top.to(device)
                targets = {'sae_reconstruction_direction': (unit(rec[ytr == 1].mean(0) - rec[ytr == 0].mean(0)), grp),
                           'sae_top1_direction': (unit(torch.sign(dmu[top_d[0]]) * D[:, top_d[0]]), f'{key}__feat1'),
                           'sae_top10_direction': (unit(D[:, top_d] @ dmu[top_d]), f'{key}__feat10')}
                for name, (v, g_) in targets.items():
                    for sign in (1, -1):
                        record('C', 'target', name, g_, arch, seed, f'cos_mass_mean={float(v @ d_hat):.4f}',
                               sign * s * v, sign=sign, key=f'{key}__{name}_{"pos" if sign > 0 else "neg"}')
                g = torch.Generator().manual_seed(control_seed(layer, f'{arch}_{seed}'))
                for _ in range(n_controls):
                    j = int(alive[torch.randint(len(alive), (1,), generator=g)])
                    sg = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                    record('C', 'control', 'random_latent_direction', f'{key}__feat1', arch, seed, f'latent_{j}',
                           s * unit(sg * D[:, j]))
                for _ in range(n_controls):
                    idx = alive[torch.randperm(len(alive), generator=g)[:TOPK_SET]].to(device)
                    record('C', 'control', 'random_latent_set_direction', f'{key}__feat10', arch, seed, '',
                           s * unit(D[:, idx] @ dmu[top_d]))
                del sae
                print(f'done {key}', flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out / 'interventions.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'pipeline_checks.csv', index=False)
    np.savez_compressed(out / 'per_pair_effects.npz', **store)
    summary = summarise(df, store)
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'test_run_info.json').write_text(json.dumps(dict(
        n_test_pairs=len(test), baseline_pair_accuracy=float((base > 0).mean()), baseline_margin=float(base.mean()),
        n_controls=n_controls, n_interventions=len(df), torch=torch.__version__, python=platform.python_version(),
        device=str(device), written_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    show = ['layer', 'part', 'intervention', 'sae_architecture', 'sae_seed', 'sign', 'detail', 'effect', 'ci95_low',
            'ci95_high', 'p_upper', 'p_two_sided', 'p_bh']
    print(summary[[c for c in show if c in summary]].round(4).to_string(index=False))


def summarise(df, store):
    ctrl = df[df['intervention_kind'] == 'control']
    groups = {g: d['effect'].to_numpy() for g, d in ctrl.groupby('control_group')}
    rows = []
    for _, t in df[df['intervention_kind'] == 'target'].iterrows():
        layer = int(t['layer'])
        if t['part'] == 'A':
            key = f"L{layer}__positive_lam{t['detail'].split('=')[1]}"
        elif t['sae_architecture']:
            key = f"L{layer}__{t['sae_architecture']}_s{t['sae_seed']}__{t['intervention']}_{'pos' if t['sign'] > 0 else 'neg'}"
        else:
            key = f"L{layer}__{t['intervention']}_{'pos' if t['sign'] > 0 else 'neg'}"
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
        elif not t['sae_architecture']:
            row['family'], row['family_p'] = 'C1_raw_directions', 'p_upper'
        else:
            row['family'], row['family_p'] = 'C2_sae_directions', 'p_two_sided'
        rows.append(row)
    out = pd.DataFrame(rows)
    out['p_bh'] = np.nan
    for fam, d in out[out['family'].notna()].groupby('family'):
        out.loc[d.index, 'p_bh'] = bh_adjust(d[d['family_p'].iloc[0]].to_numpy())
    return out


if __name__ == '__main__':
    main()
