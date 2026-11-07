#!/usr/bin/env python3
"""Controlled causal test of the honesty direction and SAE honesty features (Task C).

Two phases, run in order from the repository root:
    python scripts/repe/13_causal_random_controls.py --phase validation
    python scripts/repe/13_causal_random_controls.py --phase test

PROTOCOL (fixed before the test split is touched; written to preregistration.json by the
validation phase and read back, unchanged, by the test phase)

Model / site    GPT-2 Small (TransformerLens), blocks.6.hook_resid_pre, FINAL prompt token
                (the same site at which the RepE honesty direction is estimated).
Prompts         RepE persona prompts "USER: Pretend you're an {honest|untruthful} person ...
                ASSISTANT: <statement prefix>". Interventions are applied to the UNTRUTHFUL
                prompt of each held-out statement; the HONEST prompt of the same statement
                defines the natural behavioural target.
Behaviour       Next-token distribution at the final position (actual model output).
Primary outcome E(v) = 1 - sum_i KL(p_H,i || p_U+v,i) / sum_i KL(p_H,i || p_U,i)
                ("fraction of the natural persona-induced output shift recovered").
                E = 0 for no intervention, E = 1 for a perfect honest-persona output.
Secondary       S(v) = mean_{true stmts} dlogp(next stmt token) - mean_{false stmts} dlogp(...)
                (does the intervention preferentially raise the next token of TRUE statements?)
Selection       Directions/features are estimated on the TRAIN split only.
                Intervention norm s is chosen on VALIDATION as the grid value maximising E of the
                raw honesty direction; the same s is then used for every intervention on TEST.
Positive ctrl   Per-example activation patch: replace the untruthful prompt's final-token
                layer-6 residual with the honest prompt's (known to carry the persona signal at
                that site). Its norm is the per-example difference norm (reported).
Controls        (i) 200 Gaussian random unit directions (seeded), norm s.
                (ii) per SAE: 200 random single alive SAE features (decoder column, random sign), norm s.
                (iii) per SAE: 200 random 10-feature sets carrying the target top-10 coefficients
                      on random alive features, norm s.
Significance    one-sided empirical p = (1 + #{controls with E >= E_target}) / (1 + n_controls);
                bootstrap 95% CI over statements (2000 resamples, seed 0) for target E.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

LAYER = 6
HOOK = f'blocks.{LAYER}.hook_resid_pre'
ACTS = Path('data/repe/activations/gpt2_small/repe_final_token.pt')
PAIRS = Path('data/repe/persona_pairs.csv')
OUT = Path('results/repe/steering/causal_random_controls')
TRAIN_SCRIPT = Path('scripts/openwebtext/03_training_sae.py')
ARCHS = ('relu', 'topk', 'jumprelu')
VARIANTS = ('pretrained', 'reconstruction_only', 'truth_aware')
N_CONTROLS = 200
CONTROL_SEED = 20261101           # date of the professor's review, fixed a priori
STRENGTH_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # multiples of ||mean difference|| (train)
TOPK_SET = 10


def sae_checkpoint(arch, variant):
    if variant == 'pretrained':
        return Path(f'results/openwebtext/convergence/layer_6/{arch}_l0_128_resume_5000_to_15000/'
                    'checkpoint_step_15000.pt')
    return Path(f'results/geometry_of_truth/sae_finetuning/layer_6_{arch}/{variant}/finetuned_checkpoint.pt')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sae(module, arch, variant, device):
    state = torch.load(sae_checkpoint(arch, variant), map_location='cpu', weights_only=False)['model_state_dict']
    sae = module.SAE(768, 3072, arch, 128, temperature=0.1).to(device)
    sae.load_state_dict(state, strict=True)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


# ---------------------------------------------------------------- core computations (unit-tested)
def kl_rows(logp_ref, logp_other):
    """Row-wise KL(p_ref || p_other) from log-probabilities [n, vocab]."""
    return (logp_ref.exp() * (logp_ref - logp_other)).sum(-1)


def recovery(kl_after, kl_before):
    """E = 1 - sum KL_after / sum KL_before (aggregate ratio; robust to tiny per-item KLs)."""
    return 1.0 - float(np.sum(kl_after)) / float(np.sum(kl_before))


def empirical_p(target, controls):
    controls = np.asarray(controls)
    return (1 + int(np.sum(controls >= target))) / (1 + len(controls))


def bootstrap_recovery_ci(kl_after, kl_before, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl_before), size=(n, len(kl_before)))
    vals = 1.0 - kl_after[idx].sum(1) / kl_before[idx].sum(1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- model running
class Runner:
    def __init__(self, model, prompts, device):
        self.model = model
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
    def run(self, delta=None, replace=None):
        """delta: [d] or [n, d] added at final token; replace: [n, d] overwrites final token."""
        def hook(resid, hook):
            if replace is not None:
                resid[self.rows, self.final] = replace.to(resid.dtype)
            elif delta is not None:
                resid[self.rows, self.final] = resid[self.rows, self.final] + delta.to(resid.dtype)
            return resid
        hooks = [] if (delta is None and replace is None) else [(HOOK, hook)]
        logits = self.model.run_with_hooks(self.tokens, fwd_hooks=hooks)
        return logits[self.rows, self.final].float().log_softmax(-1)

    @torch.inference_mode()
    def final_resid(self):
        _, cache = self.model.run_with_cache(self.tokens, names_filter=lambda n: n == HOOK)
        return cache[HOOK][self.rows, self.final].float()


def unit(v):
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--smoke', action='store_true', help='20 controls, writes to smoke dir')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    device = torch.device('cuda')
    n_controls = 20 if args.smoke else N_CONTROLS
    out = (OUT / 'smoke') if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / 'preregistration.json'
    torch.manual_seed(0)

    pairs = pd.read_csv(PAIRS)
    acts = torch.load(ACTS, map_location='cpu', weights_only=True)
    assert list(acts['personas']) == pairs['persona'].tolist(), 'activation/prompt order mismatch'
    X6 = acts['activations'][LAYER].float()
    labels = acts['persona_labels'].numpy()
    splits = np.array(acts['splits'])
    train = splits == 'train'
    d_raw = X6[train & (labels == 1)].mean(0) - X6[train & (labels == 0)].mean(0)
    base_norm = float(d_raw.norm())
    d_hat = unit(d_raw).to(device)

    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    eval_split = 'validation' if args.phase == 'validation' else 'test'
    sub = pairs[pairs['split'] == eval_split]
    U = sub[sub['persona'] == 'untruthful'].sort_values('statement_id')
    H = sub[sub['persona'] == 'honest'].sort_values('statement_id')
    assert (U['statement_id'].values == H['statement_id'].values).all()
    runU, runH = Runner(model, U['prompt'].tolist(), device), Runner(model, H['prompt'].tolist(), device)

    # sanity: cached activations equal freshly computed ones at the hook site (guards hook/position bugs)
    fresh = runU.final_resid().cpu()
    cached = X6[np.where((splits == eval_split) & (labels == 0))[0]]
    cached = cached[np.argsort(np.array(acts['statement_ids'])[(splits == eval_split) & (labels == 0)], kind='stable')]
    max_err = float((fresh - cached).abs().max())
    assert max_err < 1e-3, f'hook/position mismatch with cached activations: {max_err}'

    logpH = runH.run()
    logpU = runU.run()
    kl0 = kl_rows(logpH, logpU).cpu().numpy()

    # secondary outcome: next statement token (first continuation token)
    tok = model.tokenizer
    next_ids, has_next = [], []
    for _, r in U.iterrows():
        full = tok.encode(r['statement'], add_special_tokens=False)
        k = int(r['prefix_tokens'])
        has_next.append(k < len(full))
        next_ids.append(full[k] if k < len(full) else 0)
    next_ids = torch.tensor(next_ids, device=device)
    has_next = np.array(has_next)
    is_true = U['factual_label'].values == 1

    def secondary(logp):
        d = (logp.gather(1, next_ids[:, None]) - logpU.gather(1, next_ids[:, None]))[:, 0].cpu().numpy()
        return float(d[has_next & is_true].mean() - d[has_next & ~is_true].mean())

    def measure(logp):
        kl = kl_rows(logpH, logp).cpu().numpy()
        return recovery(kl, kl0), secondary(logp), kl

    # ---------------- SAE features selected on TRAIN only
    sae_mod_spec = importlib.util.spec_from_file_location('owt_sae', TRAIN_SCRIPT)
    sae_mod = importlib.util.module_from_spec(sae_mod_spec)
    sae_mod_spec.loader.exec_module(sae_mod)
    saes = {}
    Xtr = X6[train].to(device)
    ytr = labels[train]
    for arch in ARCHS:
        for variant in VARIANTS:
            sae = load_sae(sae_mod, arch, variant, device)
            with torch.inference_mode():
                rec, z, _ = sae(Xtr)
                rec_all, _, _ = sae(X6.to(device))
            dmu = (z[ytr == 1].mean(0) - z[ytr == 0].mean(0))
            alive = torch.where((z > 0).any(0))[0]
            top = torch.argsort(dmu.abs(), descending=True)[:TOPK_SET]
            W = sae.decoder.weight  # [768, 3072], unit columns
            recon_dir = (rec_all[torch.tensor(train & (labels == 1))].mean(0)
                         - rec_all[torch.tensor(train & (labels == 0))].mean(0))
            saes[(arch, variant)] = dict(
                top1=int(top[0]), top1_sign=float(torch.sign(dmu[top[0]])),
                top1_dir=unit(torch.sign(dmu[top[0]]) * W[:, top[0]]),
                top10=top.tolist(), top10_coef=dmu[top].cpu(),
                top10_dir=unit(W[:, top] @ dmu[top]),
                recon_dir=unit(recon_dir), W=W, alive=alive.cpu(),
                top1_cos_raw=float(unit(W[:, top[0]] * torch.sign(dmu[top[0]])) @ d_hat),
                top10_cos_raw=float(unit(W[:, top] @ dmu[top]) @ d_hat),
                recon_cos_raw=float(unit(recon_dir) @ d_hat),
                ckpt_sha256=sha256(sae_checkpoint(arch, variant)))

    # ---------------- validation phase: choose strength, write preregistration
    if args.phase == 'validation':
        rows = []
        gen = torch.Generator().manual_seed(CONTROL_SEED + 999)   # validation-only controls
        val_rand = [unit(torch.randn(768, generator=gen)).to(device) for _ in range(50)]
        for mult in STRENGTH_GRID:
            s = mult * base_norm
            e, sec, _ = measure(runU.run(delta=s * d_hat))
            ctrl = [measure(runU.run(delta=s * r))[0] for r in val_rand]
            rows.append(dict(strength_multiple=mult, strength_norm=s, E_raw_direction=e, S_raw_direction=sec,
                             control_mean=np.mean(ctrl), control_sd=np.std(ctrl, ddof=1),
                             control_max=np.max(ctrl)))
            print(rows[-1], flush=True)
        vdf = pd.DataFrame(rows)
        vdf.to_csv(out / 'validation_strength_selection.csv', index=False)
        best = vdf.loc[vdf['E_raw_direction'].idxmax()]
        pe, _, _ = measure(runU.run(replace=runH.final_resid()))
        prereg = dict(
            written_utc=datetime.now(timezone.utc).isoformat(),
            script_sha256=sha256(__file__),
            selection_rule='strength multiple maximising E of raw honesty direction on validation',
            chosen_strength_multiple=float(best['strength_multiple']),
            chosen_strength_norm=float(best['strength_norm']),
            mean_difference_norm_train=base_norm,
            validation_baseline_mean_KL=float(kl0.mean()),
            validation_positive_control_E=pe,
            primary_outcome='E = 1 - sum KL(p_H||p_U+v) / sum KL(p_H||p_U), next-token at final position',
            secondary_outcome='S = mean dlogp(next stmt token | true) - mean dlogp(next stmt token | false)',
            layer=LAYER, hook=HOOK, position='final prompt token',
            n_controls=N_CONTROLS, control_seed=CONTROL_SEED, topk_set=TOPK_SET,
            significance='one-sided empirical p=(1+#controls>=target)/(1+n)',
            ci='bootstrap over statements, 2000 resamples, seed 0',
            note='Protocol designed by the analysis team following the professor\'s request for 200 random '
                 'controls; no written professor specification was found in the repository.')
        prereg_path.write_text(json.dumps(prereg, indent=2))
        print(json.dumps(prereg, indent=2))
        return

    # ---------------- test phase
    if not prereg_path.exists():
        raise SystemExit('Run --phase validation first (preregistration missing).')
    prereg = json.loads(prereg_path.read_text())
    s = prereg['chosen_strength_norm']
    print(f'TEST phase: strength norm s={s:.4f} ({prereg["chosen_strength_multiple"]}x), n={len(U)} statements', flush=True)

    records, kl_store = [], {}

    def record(kind, name, arch, variant, feature, seed, strength, logp, group):
        e, sec, kl = measure(logp)
        records.append(dict(model='gpt2-small', layer=LAYER, hook=HOOK, position='final_prompt_token',
                            split='test', n_statements=len(U), intervention_kind=kind, intervention=name,
                            control_group=group, sae_architecture=arch, sae_variant=variant,
                            feature_or_direction=feature, random_seed=seed, strength_norm=strength,
                            baseline_sum_KL=float(kl0.sum()), intervened_sum_KL=float(kl.sum()),
                            effect_E=e, secondary_S=sec))
        return kl

    record('baseline', 'no_intervention', '', '', '', '', 0.0, logpU, '')
    patch = runH.final_resid()
    patch_norm = float((patch - runU.final_resid()).norm(dim=1).mean())
    kl_store['positive_control_patch'] = record('positive_control', 'activation_patch_honest_final_token',
                                                '', '', 'per-example x_H - x_U', '', patch_norm,
                                                runU.run(replace=patch), '')
    kl_store['raw_direction'] = record('target', 'raw_honesty_direction', '', '', 'mean_diff_train', '',
                                       s, runU.run(delta=s * d_hat), 'gaussian')
    # Gaussian controls
    gen = torch.Generator().manual_seed(CONTROL_SEED)
    for i in range(n_controls):
        r = unit(torch.randn(768, generator=gen)).to(device)
        record('control', f'gaussian_{i:03d}', '', '', f'gaussian_{i:03d}', CONTROL_SEED, s,
               runU.run(delta=s * r), 'gaussian')

    for (arch, variant), info in saes.items():
        key = f'{arch}_{variant}'
        kl_store[f'{key}__recon_dir'] = record('target', 'sae_reconstruction_direction', arch, variant,
                                               'mean_diff_of_reconstructions', '', s,
                                               runU.run(delta=s * info['recon_dir']), 'gaussian')
        kl_store[f'{key}__top1'] = record('target', 'sae_top1_feature', arch, variant,
                                          f"feature_{info['top1']}(sign {info['top1_sign']:+.0f})", '', s,
                                          runU.run(delta=s * info['top1_dir']), f'{key}__feat1')
        kl_store[f'{key}__top10'] = record('target', 'sae_top10_features', arch, variant,
                                           'features_' + '-'.join(map(str, info['top10'])), '', s,
                                           runU.run(delta=s * info['top10_dir']), f'{key}__feat10')
        seed = CONTROL_SEED + 1000 * (ARCHS.index(arch) * 3 + VARIANTS.index(variant) + 1)
        g = torch.Generator().manual_seed(seed)
        alive = info['alive']
        W = info['W']
        for i in range(n_controls):
            j = int(alive[torch.randint(len(alive), (1,), generator=g)])
            sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
            record('control', f'random_feature_{i:03d}', arch, variant, f'feature_{j}(sign {sign:+.0f})',
                   seed, s, runU.run(delta=s * unit(sign * W[:, j])), f'{key}__feat1')
        for i in range(n_controls):
            idx = alive[torch.randperm(len(alive), generator=g)[:TOPK_SET]]
            v = unit(W[:, idx.to(device)] @ info['top10_coef'].to(device))
            record('control', f'random_feature_set_{i:03d}', arch, variant,
                   'features_' + '-'.join(map(str, idx.tolist())), seed, s, runU.run(delta=s * v),
                   f'{key}__feat10')
        print(f'done {key}', flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out / 'causal_controls_200.csv', index=False)
    np.savez_compressed(out / 'per_statement_kl_targets.npz', baseline_kl=kl0, **kl_store)

    # ---------------- summary
    summ = []
    for _, t in df[df['intervention_kind'].isin(['target', 'positive_control'])].iterrows():
        row = t.to_dict()
        name = 'positive_control_patch' if t['intervention_kind'] == 'positive_control' else (
            'raw_direction' if t['intervention'] == 'raw_honesty_direction' else
            f"{t['sae_architecture']}_{t['sae_variant']}__" +
            {'sae_reconstruction_direction': 'recon_dir', 'sae_top1_feature': 'top1',
             'sae_top10_features': 'top10'}[t['intervention']])
        lo, hi = bootstrap_recovery_ci(kl_store[name], kl0)
        row.update(E_ci95_low=lo, E_ci95_high=hi)
        groups = ['gaussian'] if t['intervention_kind'] == 'positive_control' else [t['control_group']]
        if t['intervention'].startswith('sae_top'):
            groups.append('gaussian')
        for g in groups:
            c = df[(df['intervention_kind'] == 'control') & (df['control_group'] == g)]['effect_E'].values
            cs = df[(df['intervention_kind'] == 'control') & (df['control_group'] == g)]['secondary_S'].values
            tag = 'vs_gaussian' if g == 'gaussian' else 'vs_random_features'
            row.update({f'{tag}_n': len(c), f'{tag}_mean': c.mean(), f'{tag}_sd': c.std(ddof=1),
                        f'{tag}_q95': np.percentile(c, 95), f'{tag}_max': c.max(),
                        f'{tag}_z': (t['effect_E'] - c.mean()) / c.std(ddof=1),
                        f'{tag}_p_empirical': empirical_p(t['effect_E'], c),
                        f'{tag}_S_p_empirical': empirical_p(t['secondary_S'], cs)})
        if t['sae_architecture']:
            info = saes[(t['sae_architecture'], t['sae_variant'])]
            row['cosine_with_raw_direction'] = {'sae_reconstruction_direction': info['recon_cos_raw'],
                                                'sae_top1_feature': info['top1_cos_raw'],
                                                'sae_top10_features': info['top10_cos_raw']}[t['intervention']]
            row['sae_checkpoint_sha256'] = info['ckpt_sha256']
        else:
            row['cosine_with_raw_direction'] = 1.0 if t['intervention'] == 'raw_honesty_direction' else np.nan
        summ.append(row)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / 'causal_summary.csv', index=False)
    (out / 'test_run_info.json').write_text(json.dumps(dict(
        preregistration=prereg, script_sha256_at_test=sha256(__file__), n_test_statements=len(U),
        test_baseline_mean_KL=float(kl0.mean()), positive_control_mean_patch_norm=patch_norm,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
        written_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    cols = ['intervention', 'sae_architecture', 'sae_variant', 'effect_E', 'E_ci95_low', 'E_ci95_high',
            'vs_gaussian_p_empirical', 'vs_random_features_p_empirical', 'secondary_S', 'cosine_with_raw_direction']
    print(sdf[[c for c in cols if c in sdf]].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
