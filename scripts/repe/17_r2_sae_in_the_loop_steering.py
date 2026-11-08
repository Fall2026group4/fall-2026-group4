#!/usr/bin/env python3
"""R2: genuine SAE-in-the-loop intervention on honesty-selected SAE features, GPT-2 Small layer 6, final prompt token.

Run from repository root:
    python scripts/repe/17_r2_sae_in_the_loop_steering.py --smoke        # checks (pre-registration §8); smoke/<time>/
    python scripts/repe/17_r2_sae_in_the_loop_steering.py --prepare      # preregistration.json + hashes (refuses overwrite)
    python scripts/repe/17_r2_sae_in_the_loop_steering.py --validation   # validation stage: all k, all controls, k choice
    python scripts/repe/17_r2_sae_in_the_loop_steering.py --test         # REFUSES: test stage deferred (needs approval)

Pre-registration: results/repe/r2_sae_in_the_loop_steering_layer6/README.md (§0-§11 as drafted, §12 addendum).

Intervention (live forward pass, TransformerLens hook at blocks.6.hook_resid_pre, final prompt token):
    z = SAE.encode(x); z'_F = z_F + alpha * (mu_F - z_F) (other latents unchanged; TopK: after top-k, no re-selection)
    error kept (primary):   x' = x + W_dec[:, F] (z'_F - z_F)            (alpha = 0 -> x' = x exactly)
    no error (secondary):   x' = W_dec z' + b                           (alpha = 0 -> reconstruction splice)
The edit depends on the encoder output z(x) at run time (not a fixed direction), except for features inactive on a
prompt (z_F = 0), where it reduces to alpha * mu_F * W_dec[:, F]; the activity rate is reported.
Feature selection uses TRAIN only: candidates active in >= 5 % of the 834 train rows; paired effect size
d_f = mean(z_H - z_U) / sd(z_H - z_U) over the 417 train pairs; top-|d| sets of size k in {1, 4, 16};
mu^H / mu^U = train persona means of z. Validation statement 326 excluded. Test rows are never read by this script.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C13 = load('scripts/repe/13_causal_random_controls.py', 'c13')   # Runner, kl_rows, recovery, CI, empirical p
LAYER, HOOK = 6, 'blocks.6.hook_resid_pre'
ARCHS, SEEDS, KS = ('relu', 'topk', 'jumprelu'), (42, 1042, 2042), (1, 4, 16)
ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
N_CONTROLS, CONTROL_SEED, N_BOOT, MIN_ACTIVE = 200, 20261108, 2000, 0.05
EXCLUDED = (326,)
OUT = REPO / 'results/repe/r2_sae_in_the_loop_steering_layer6'
R1CK = REPO / 'results/repe/sae_finetuning/r1_honesty_preservation_layer6'
CACHE = REPO / 'data/repe/activations/gpt2_small/repe_final_token.pt'
PAIRS = REPO / 'data/repe/persona_pairs.csv'
CROSS = REPO / 'data/repe/cross_wording_validation.csv'
VERIFIED = ('scripts/repe/17_r2_sae_in_the_loop_steering.py', 'scripts/repe/13_causal_random_controls.py',
            'data/repe/persona_pairs.csv', 'data/repe/cross_wording_validation.csv',
            'results/repe/r2_sae_in_the_loop_steering_layer6/preregistration.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sae_list(include_ft=True):
    out = [('base', a, s, Path(uninterrupted_checkpoint(LAYER, a, s))) for a in ARCHS for s in SEEDS]
    if include_ft:   # secondary, EXPLORATORY (pre-registration §6): R1 fine-tunes
        out += [(f'EXPLORATORY_{arm}', a, s, R1CK / f'{arm}_{a}_seed{s}' / 'finetuned_checkpoint.pt')
                for arm in ('r_recon', 'r_honesty') for a in ARCHS for s in SEEDS]
    return out


# ------------------------------------------------------------------ data
def train_data():
    c = torch.load(CACHE, map_location='cpu', weights_only=True)
    sid, split, per = np.array(c['statement_ids']), np.array(c['splits']), np.array(c['personas'])
    X = c['activations'][LAYER].float()
    ids = np.unique(sid[split == 'train'])
    h = np.array([np.flatnonzero((sid == i) & (per == 'honest'))[0] for i in ids])
    u = np.array([np.flatnonzero((sid == i) & (per == 'untruthful'))[0] for i in ids])
    assert (split[h] == 'train').all() and (split[u] == 'train').all()
    fl = c['factual_labels'].numpy()
    return dict(XH=X[h], XU=X[u], fact=fl[h], ids=ids)


def eval_prompts(which):
    df = pd.read_csv(PAIRS if which == 'val' else CROSS)
    df = df[(df.split == 'validation') & ~df.statement_id.isin(EXCLUDED)]
    U = df[df.persona == 'untruthful'].sort_values('statement_id')
    H = df[df.persona == 'honest'].sort_values('statement_id')
    assert (U.statement_id.values == H.statement_id.values).all() and len(U) == 89
    return U, H


# ------------------------------------------------------------------ feature selection (train only)
@torch.no_grad()
def select_features(sae, td, device):
    zH, zU = sae(td['XH'].to(device))[1], sae(td['XU'].to(device))[1]
    act = torch.cat([zH, zU]).gt(0).float().mean(0)
    diff = zH - zU
    sd = diff.std(0)
    d = torch.where(sd > 0, diff.mean(0) / sd.clamp_min(1e-12), torch.zeros_like(sd))
    cand = torch.where((act >= MIN_ACTIVE) & (sd > 0))[0]
    order = cand[torch.argsort(d[cand].abs(), descending=True)]
    # unrelated control: factual label (true vs false), Cohen's d within each persona, averaged
    t = torch.as_tensor(td['fact'] == 1, device=device)
    dd = []
    for z in (zH, zU):
        a, b = z[t], z[~t]
        pooled = ((a.var(0) + b.var(0)) / 2).sqrt()
        dd.append(torch.where(pooled > 0, (a.mean(0) - b.mean(0)) / pooled.clamp_min(1e-12), torch.zeros_like(pooled)))
    dfact = (dd[0] + dd[1]) / 2
    sel = set(order[:max(KS)].tolist())
    ucand = torch.tensor([i for i in cand.tolist() if i not in sel], device=device)
    uorder = ucand[torch.argsort(dfact[ucand].abs(), descending=True)]
    z_all = torch.cat([zH, zU])
    return dict(order=order, cand=cand, d=d, act=act, muH=zH.mean(0), muU=zU.mean(0),
                mu_true=z_all[torch.cat([t, t])].mean(0), unrel=uorder, dfact=dfact)


def edit(sae, x, F, mu, alpha, keep_error=True):
    _, z, _ = sae(x)
    zF = z[:, F]
    newF = zF + alpha * (mu[F] - zF)
    W = sae.decoder.weight
    if keep_error:
        return x + (newF - zF) @ W[:, F].T, z
    z2 = z.clone()
    z2[:, F] = newF
    return z2 @ W.T + sae.output_bias, z


# ------------------------------------------------------------------ evaluation context per prompt set
class PromptSet:
    def __init__(self, model, which, device, train_probes):
        self.U, self.H = eval_prompts(which)
        self.rU = C13.Runner(model, self.U.prompt.tolist(), device)
        self.rH = C13.Runner(model, self.H.prompt.tolist(), device)
        self.xU, self.xH = self.rU.final_resid(), self.rH.final_resid()
        self.lpU, self.lpH = self.rU.run(), self.rH.run()
        self.kl0 = C13.kl_rows(self.lpH, self.lpU).cpu().numpy()          # U->H baseline
        self.kl0_HU = C13.kl_rows(self.lpU, self.lpH).cpu().numpy()       # H->U baseline
        tok = model.tokenizer
        nxt, has = [], []
        for r in self.U.itertuples():
            full = tok.encode(r.statement, add_special_tokens=False)
            has.append(r.prefix_tokens < len(full))
            nxt.append(full[r.prefix_tokens] if r.prefix_tokens < len(full) else 0)
        self.next = torch.tensor(nxt, device=device)
        self.has = np.array(has)
        self.true = self.U.factual_label.to_numpy() == 1
        self.probes = train_probes
        self.n = len(self.U)

    def outcome(self, lp, direction='UH'):
        if direction == 'UH':
            kl = C13.kl_rows(self.lpH, lp).cpu().numpy()
            e = C13.recovery(kl, self.kl0)
            d = (lp.gather(1, self.next[:, None]) - self.lpU.gather(1, self.next[:, None]))[:, 0].cpu().numpy()
            s = float(d[self.has & self.true].mean() - d[self.has & ~self.true].mean())
            top1 = float((lp.argmax(-1) == self.lpH.argmax(-1)).float().mean())
            ent = float((-(lp.exp() * lp).sum(-1) + (self.lpU.exp() * self.lpU).sum(-1)).mean())
            return dict(E=e, S=s, top1_agree_honest=top1, entropy_change=ent), kl
        kl = C13.kl_rows(self.lpU, lp).cpu().numpy()
        return dict(E=C13.recovery(kl, self.kl0_HU)), kl

    def probe_shift(self, x_new, x_from, x_to):
        out = {}
        for name, f in self.probes.items():
            a, b, c = f(x_new), f(x_from), f(x_to)
            out[f'probe_shift_{name}'] = float((a - b).mean() / (c - b).mean())
        return out


def fit_train_probes(td):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = torch.cat([td['XH'], td['XU']]).numpy()
    y = np.r_[np.ones(len(td['XH'])), np.zeros(len(td['XU']))]
    d = (td['XH'].double().mean(0) - td['XU'].double().mean(0))
    d = (d / d.norm()).float()
    log = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(X, y)
    return {'massmean': lambda x: (x.float().cpu() @ d).numpy(),
            'logistic': lambda x: log.decision_function(x.float().cpu().numpy())}, d


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def paired_boot_diff(kl_a, kl_b, kl0, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl0), size=(n, len(kl0)))
    ea = 1 - kl_a[idx].sum(1) / kl0[idx].sum(1)
    eb = 1 - kl_b[idx].sum(1) / kl0[idx].sum(1)
    return float(np.percentile(ea - eb, 2.5)), float(np.percentile(ea - eb, 97.5))


# ------------------------------------------------------------------ main stages
@torch.inference_mode()
def run_sae(tag, arch, seed, path, model, sets, td, device, n_controls, owt, full, rows, ctrl_rows, feat_rows, k_list):
    sae, _ = fm.load_sae(path, device, architecture=arch)
    for p in sae.parameters():
        p.requires_grad_(False)
    fs = select_features(sae, td, device)
    for r_, f in enumerate(fs['order'][:max(KS)].tolist()):
        feat_rows.append(dict(sae=tag, architecture=arch, seed=seed, rank=r_ + 1, feature=f, d=float(fs['d'][f]),
                              train_active_rate=float(fs['act'][f]), muH=float(fs['muH'][f]), muU=float(fs['muU'][f])))
    for r_, f in enumerate(fs['unrel'][:max(KS)].tolist()):
        feat_rows.append(dict(sae=tag, architecture=arch, seed=seed, rank=r_ + 1, feature=f, d=float(fs['d'][f]),
                              d_factual=float(fs['dfact'][f]), train_active_rate=float(fs['act'][f]),
                              muH=float(fs['muH'][f]), muU=float(fs['muU'][f]), kind='unrelated_factual'))
    gen_seed = CONTROL_SEED + 7919 * (ARCHS.index(arch) * 10 + SEEDS.index(seed))
    base = dict(sae=tag, architecture=arch, seed=seed, checkpoint_sha256=sha(path))
    for k in k_list:
        F = fs['order'][:k]
        g = torch.Generator().manual_seed(gen_seed + k)
        pool = torch.tensor([i for i in fs['cand'].tolist() if i not in set(F.tolist())])
        rand_sets = [pool[torch.randperm(len(pool), generator=g)[:k]].to(device) for _ in range(n_controls)]
        rand_dirs = [C13.unit(torch.randn(768, generator=g)).to(device) for _ in range(n_controls)]
        for sname, S in sets.items():
            kl_store = {}
            for alpha in ALPHAS:
                for keep in ((True, False) if (full and sname == 'val') else (True,)):
                    xn, z = edit(sae, S.xU, F, fs['muH'], alpha, keep)
                    res, kl = S.outcome(S.rU.run(replace=xn))
                    lo, hi = C13.bootstrap_recovery_ci(kl, S.kl0, n=N_BOOT)
                    dx = (xn - S.xU).norm(dim=1)
                    z2 = sae(xn)[1]
                    want = (alpha * (fs['muH'][F] - z[:, F]))
                    got = z2[:, F] - z[:, F]
                    m = want.abs() > 1e-6
                    nf = torch.ones(z.shape[1], dtype=torch.bool, device=device)
                    nf[F] = False
                    rows.append(dict(base, k=k, set=sname, direction='U->H', alpha=alpha, error_kept=keep, **res,
                                     E_ci_lo=lo, E_ci_hi=hi, mean_delta_norm=float(dx.mean()),
                                     feature_active_rate=float((z[:, F] > 0).float().mean()),
                                     reencode_target_attainment=float((got[m] / want[m]).mean()) if m.any() else np.nan,
                                     reencode_other_rel_change=float((z2[:, nf] - z[:, nf]).abs().sum() / z[:, nf].abs().sum().clamp_min(1e-12)),
                                     **S.probe_shift(xn, S.xU, S.xH)))
                    if keep:
                        kl_store[alpha] = kl
                        if alpha == 1.0:
                            dx_t = dx
            if full and sname == 'val':
                for alpha in ALPHAS:     # H->U (secondary)
                    xn, _ = edit(sae, S.xH, F, fs['muU'], alpha, True)
                    res, kl = S.outcome(S.rH.run(replace=xn), 'HU')
                    lo, hi = C13.bootstrap_recovery_ci(kl, S.kl0_HU, n=N_BOOT)
                    rows.append(dict(base, k=k, set=sname, direction='H->U', alpha=alpha, error_kept=True, **res,
                                     E_ci_lo=lo, E_ci_hi=hi))
                # specificity: U->H edit applied to honest prompts
                xn, _ = edit(sae, S.xH, F, fs['muH'], 1.0, True)
                klspec = C13.kl_rows(S.lpH, S.rH.run(replace=xn)).cpu().numpy()
                # unrelated features clamped to their true-statement mean (alpha = 1)
                Fu = fs['unrel'][:k]
                xu_, _ = edit(sae, S.xU, Fu, fs['mu_true'], 1.0, True)
                ru, klu = S.outcome(S.rU.run(replace=xu_))
                # raw mass-mean direction, norm matched per prompt to the target edit
                xr = S.xU + dx_t[:, None] * td['dhat'].to(device)[None]
                rr, klr = S.outcome(S.rU.run(replace=xr))
                # reconstruction splice (no edit, no error) and positive control
                rec_, kl_rec = S.outcome(S.rU.run(replace=sae(S.xU)[0]))
                pos, klp = S.outcome(S.rU.run(replace=S.xH))
                slo, shi = paired_boot_diff(kl_store[1.0], klu, S.kl0)
                rows.append(dict(base, k=k, set=sname, direction='U->H', alpha=1.0, error_kept=True, kind='extras',
                                 specificity_mean_KL_on_honest=float(klspec.mean()),
                                 specificity_KL_ratio_to_persona_KL=float(klspec.sum() / S.kl0.sum()),
                                 E_unrelated=ru['E'], E_raw_direction_normmatched=rr['E'], E_recon_splice=rec_['E'],
                                 E_positive_control=pos['E'], specificity_diff_ci_lo=slo, specificity_diff_ci_hi=shi,
                                 unrelated_mean_delta_norm=float((xu_ - S.xU).norm(dim=1).mean())))
            # controls at alpha = 1 (U->H, error kept)
            for i, Fr in enumerate(rand_sets):
                xn, _ = edit(sae, S.xU, Fr, fs['muH'], 1.0, True)
                ctrl_rows.append(dict(base, k=k, set=sname, family='C2_random_features', i=i,
                                      E=S.outcome(S.rU.run(replace=xn))[0]['E']))
            if full and sname == 'val':
                for i, u in enumerate(rand_dirs):
                    xn = S.xU + dx_t[:, None] * u[None]
                    ctrl_rows.append(dict(base, k=k, set=sname, family='C3_matched_norm_directions', i=i,
                                          E=S.outcome(S.rU.run(replace=xn))[0]['E']))
        if full and owt is not None:    # general damage on OWT validation: edit at all non-BOS positions
            sp, Sp, seqs, orig = owt

            def fn_for(FF):
                return lambda x: edit(sae, x, FF, fs['muH'], 1.0, True)[0]
            o = sp.owt_eval(Sp, seqs, LAYER, fn_for(F), orig)
            ctrl_rows.append(dict(base, k=k, set='owt', family='damage_target', i=-1, delta_ce=o['delta_ce'],
                                  kl=o['kl_mean'], top1=o['top1_agreement']))
            for i, Fr in enumerate(rand_sets):
                o = sp.owt_eval(Sp, seqs, LAYER, fn_for(Fr), orig)
                ctrl_rows.append(dict(base, k=k, set='owt', family='damage_C2_random_features', i=i,
                                      delta_ce=o['delta_ce'], kl=o['kl_mean'], top1=o['top1_agreement']))


@torch.inference_mode()
def smoke():
    device = 'cuda'
    out = OUT / 'smoke' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    td = train_data()
    probes, td['dhat'] = fit_train_probes(td)
    S = PromptSet(model, 'val', device, probes)
    checks = {}
    # cached vs fresh activations (validation rows of the cache, same order)
    c = torch.load(CACHE, map_location='cpu', weights_only=True)
    sid, split, per = np.array(c['statement_ids']), np.array(c['splits']), np.array(c['personas'])
    idx = [np.flatnonzero((sid == i) & (per == 'untruthful'))[0] for i in S.U.statement_id]
    checks['fresh_vs_cached_max_abs'] = float((S.xU.cpu() - c['activations'][LAYER][idx].float()).abs().max())
    # hook: only the final position changes; positions before it are bit-identical
    toks = S.rU.tokens
    lg0 = model(toks)
    lg1 = model.run_with_hooks(toks, fwd_hooks=[(HOOK, lambda r, hook: (r.__setitem__((S.rU.rows, S.rU.final),
                                                                                       r[S.rU.rows, S.rU.final] + 1.0), r)[1])])
    pre = torch.arange(toks.shape[1], device=device)[None] < S.rU.final[:, None]
    checks['earlier_positions_max_abs_logit_change'] = float((lg1 - lg0).abs()[pre].max())
    checks['final_position_changed'] = bool((lg1 - lg0).abs()[S.rU.rows, S.rU.final].max() > 0)
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    for arch in ARCHS:
        path = Path(uninterrupted_checkpoint(LAYER, arch, 42))
        sae, _ = fm.load_sae(path, device, architecture=arch)
        fs = select_features(sae, td, device)
        F = fs['order'][:4]
        x0, _ = edit(sae, S.xU, F, fs['muH'], 0.0, True)
        lp0 = S.rU.run(replace=x0)
        checks[f'{arch}_alpha0_identity_max_abs_logprob'] = float((lp0 - S.lpU).abs().max())
        checks[f'{arch}_alpha0_identity_E'] = S.outcome(lp0)[0]['E']
        # no-error alpha=0 equals the shared splicer's final-token reconstruction splice
        xr, _ = edit(sae, S.xU, F, fs['muH'], 0.0, False)
        mine = S.rU.run(replace=xr)
        toks2, lens2 = sp.pad([model.to_tokens(p, prepend_bos=True, move_to_device=False)[0] for p in S.U.prompt], 50256)
        ref = sp.Splicer(model, device).logprobs(toks2, lens2, LAYER, sp.sae_fn(sae), 'final')
        ref = ref[torch.arange(S.n, device=device), (lens2 - 1).to(device)]
        checks[f'{arch}_noerror_alpha0_vs_shared_splicer_max_abs'] = float((mine - ref).abs().max())
        x1, z = edit(sae, S.xU, F, fs['muH'], 1.0, True)
        z2 = sae(x1)[1]
        checks[f'{arch}_reencode_mean_abs_err_to_target_zF'] = float((z2[:, F] - fs['muH'][F]).abs().mean())
        checks[f'{arch}_reencode_mean_abs_zF_before'] = float((z[:, F] - fs['muH'][F]).abs().mean())
        res = S.outcome(S.rU.run(replace=x1))[0]
        checks[f'{arch}_E_alpha1_k4'] = res['E']
        checks[f'{arch}_finite'] = bool(np.isfinite(list(res.values())).all())
        checks[f'{arch}_selected_k4'] = F.tolist()
        checks[f'{arch}_feature_active_rate_val'] = float((z[:, F] > 0).float().mean())
    checks['gpu_peak_mb'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    checks['pass'] = bool(checks['fresh_vs_cached_max_abs'] < 1e-3 and checks['earlier_positions_max_abs_logit_change'] == 0
                          and checks['final_position_changed']
                          and all(checks[f'{a}_alpha0_identity_max_abs_logprob'] < 1e-4 for a in ARCHS)
                          and all(checks[f'{a}_noerror_alpha0_vs_shared_splicer_max_abs'] < 1e-4 for a in ARCHS)
                          and all(checks[f'{a}_finite'] for a in ARCHS) and checks['gpu_peak_mb'] < 10240)
    (out / 'smoke_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


def prepare():
    for p in ('preregistration.json', 'pretraining_hashes.json'):
        if (OUT / p).exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
    prereg = {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': 'R2 SAE-in-the-loop honesty-feature intervention, GPT-2 Small layer 6, final prompt token',
        'documents': ['README.md §0-§11 as drafted (commit d17bb00)', 'README.md §12 pre-run addendum'],
        'stage_now': 'VALIDATION only (user decision 2026-11-08: validation now, test later). No test row is read.',
        'saes_primary': {f'{a}/{s}': sha(uninterrupted_checkpoint(LAYER, a, s)) for a in ARCHS for s in SEEDS},
        'saes_secondary_exploratory': 'R1 r_recon and r_honesty (18): targets + 200 random-feature controls, validation only',
        'intervention': 'z_F <- z_F + alpha (mu_F - z_F); error kept (primary) / no error (secondary); final token',
        'alphas': ALPHAS, 'primary_alphas': [1.0, -1.0], 'k_grid': KS, 'min_train_active_rate': MIN_ACTIVE,
        'feature_score': 'paired d over 417 train pairs; top-|d|', 'targets': 'train persona means of z',
        'k_selection_rule': 'per architecture, k maximising mean E(alpha=1, U->H, error kept) over 3 seeds on validation '
                            '(default wording); frozen for the test stage',
        'controls': {'C2': f'{N_CONTROLS} random candidate-feature sets of size k (target set excluded), clamped to their own '
                           'honest means, alpha 1', 'C3': f'{N_CONTROLS} Gaussian unit directions, per-prompt norm = target ||dx||',
                     'unrelated': 'top-k features by factual-label d (within persona, averaged; target set excluded), clamped '
                                  'to their true-statement mean, alpha 1',
                     'raw_direction': 'train mass-mean persona direction, per-prompt norm-matched',
                     'positive': 'final-token activation patch x_H -> U prompt', 'recon_splice': 'decode(z), no error',
                     'damage': 'alpha-1 edit at all non-BOS positions of the 100 OWT validation sequences (128 tokens) vs '
                               'the 200 C2 sets', 'specificity': 'U->H edit applied to honest prompts'},
        'control_seed': CONTROL_SEED, 'bootstrap': f'{N_BOOT} statement resamples, seed 0',
        'outcome': 'E = 1 - sum KL(p_H||p_U\') / sum KL(p_H||p_U), next-token at final position (6c definition)',
        'test_criteria_unchanged': 'README §7 (to be applied once, on test, after approval)',
        'validation_report': 'the §7 criteria are also computed on validation, DESCRIPTIVELY, for every k and for the '
                             'selected k (optimistic, since k is selected on the same data); wording check on the '
                             'cross-wording validation set replaces criterion 5 at this stage',
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    hashes = {p: sha(REPO / p) for p in VERIFIED}
    hashes['results/repe/r2_sae_in_the_loop_steering_layer6/README.md'] = sha(OUT / 'README.md')
    (OUT / 'pretraining_hashes.json').write_text(json.dumps({'recorded': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             'sha256': hashes}, indent=2))
    print(json.dumps(hashes, indent=2))


@torch.inference_mode()
def validation():
    hashes = json.loads((OUT / 'pretraining_hashes.json').read_text())['sha256']
    bad = [p for p in VERIFIED if sha(REPO / p) != hashes[p]]
    if bad:
        raise SystemExit(f'files changed since pre-registration: {bad}')
    for p in ('validation_targets.csv', 'validation_controls.csv'):
        if (OUT / p).exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
    device = 'cuda'
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    td = train_data()
    probes, td['dhat'] = fit_train_probes(td)
    sets = {'val': PromptSet(model, 'val', device, probes), 'cw': PromptSet(model, 'cw', device, probes)}
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    Sp = sp.Splicer(model, device)
    seqs, _ = sp.load_owt_validation()
    owt = (sp, Sp, seqs, sp.owt_original(Sp, seqs))
    rows, ctrl, feats = [], [], []
    for tag, arch, seed, path in sae_list():
        full = tag == 'base'
        run_sae(tag, arch, seed, path, model, sets if full else {'val': sets['val']}, td, device, N_CONTROLS,
                owt if full else None, full, rows, ctrl, feats, KS)
        print(f'{tag} {arch} {seed} done ({time.time() - t0:.0f}s)', flush=True)
    T, C = pd.DataFrame(rows), pd.DataFrame(ctrl)
    T.to_csv(OUT / 'validation_targets.csv', index=False)
    C.to_csv(OUT / 'validation_controls.csv', index=False)
    pd.DataFrame(feats).to_csv(OUT / 'feature_selection.csv', index=False)
    pos = {s: S.outcome(S.rU.run(replace=S.xH))[0]['E'] for s, S in sets.items()}
    summarise(T, C, pos)
    (OUT / 'validation_run_info.json').write_text(json.dumps({
        'script_sha256': sha(__file__), 'stage': 'validation', 'test_rows_read': False,
        'n_val_statements': sets['val'].n, 'n_cw_statements': sets['cw'].n,
        'val_baseline_mean_KL_HU': float(sets['val'].kl0.mean()), 'positive_control_E': pos,
        'owt_sequences': len(seqs), 'gpu_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))


def summarise(T, C, pos):
    tg = T[(T.direction == 'U->H') & (T.error_kept == True) & T.get('kind', pd.Series(index=T.index)).isna()]  # noqa: E712
    ex = T[T.get('kind', pd.Series(index=T.index)) == 'extras']
    summ = []
    for (sae, arch, seed, k), g in tg.groupby(['sae', 'architecture', 'seed', 'k']):
        for st in sorted(g.set.unique()):
            a1 = g[(g.set == st) & (g.alpha == 1.0)].iloc[0]
            am = g[(g.set == st) & (g.alpha == -1.0)].iloc[0]
            r = dict(sae=sae, architecture=arch, seed=seed, k=k, set=st, E_alpha1=a1.E, E_alpha1_ci_lo=a1.E_ci_lo,
                     E_alpha1_ci_hi=a1.E_ci_hi, E_alpha_m1=am.E, E_alpha_m1_ci_lo=am.E_ci_lo, E_alpha_m1_ci_hi=am.E_ci_hi,
                     S_alpha1=a1.S, top1_agree_honest_alpha1=a1.top1_agree_honest, feature_active_rate=a1.feature_active_rate,
                     mean_delta_norm_alpha1=a1.mean_delta_norm, probe_shift_massmean_alpha1=a1.probe_shift_massmean,
                     probe_shift_logistic_alpha1=a1.probe_shift_logistic, reencode_target_attainment=a1.reencode_target_attainment,
                     E_fraction_of_positive=a1.E / pos[st])
            for fam, tagf in (('C2_random_features', 'C2'), ('C3_matched_norm_directions', 'C3')):
                c = C[(C.sae == sae) & (C.architecture == arch) & (C.seed == seed) & (C.k == k) & (C.set == st) & (C.family == fam)].E.values
                if len(c):
                    r.update({f'{tagf}_mean': c.mean(), f'{tagf}_q95': np.percentile(c, 95),
                              f'{tagf}_p': C13.empirical_p(a1.E, c), f'{tagf}_n': len(c)})
            e = ex[(ex.sae == sae) & (ex.architecture == arch) & (ex.seed == seed) & (ex.k == k) & (ex.set == st)]
            if len(e):
                e = e.iloc[0]
                r.update(E_unrelated=e.E_unrelated, specificity_diff=a1.E - e.E_unrelated,
                         specificity_diff_ci_lo=e.specificity_diff_ci_lo, specificity_diff_ci_hi=e.specificity_diff_ci_hi,
                         E_raw_direction_normmatched=e.E_raw_direction_normmatched, E_recon_splice=e.E_recon_splice,
                         specificity_KL_ratio_on_honest=e.specificity_KL_ratio_to_persona_KL)
            d = C[(C.sae == sae) & (C.architecture == arch) & (C.seed == seed) & (C.k == k) & (C.set == 'owt')]
            if len(d) and st == 'val':
                tgt = d[d.family == 'damage_target'].delta_ce.iloc[0]
                cc = d[d.family == 'damage_C2_random_features'].delta_ce.values
                r.update(owt_delta_ce_target=tgt, owt_delta_ce_C2_q95=np.percentile(cc, 95),
                         owt_delta_ce_C2_mean=cc.mean(), owt_kl_target=d[d.family == 'damage_target'].kl.iloc[0])
            summ.append(r)
    Sdf = pd.DataFrame(summ)
    base = (Sdf.sae == 'base') & (Sdf.set == 'val')
    for k in KS:
        m = base & (Sdf.k == k)
        q = bh(np.r_[Sdf.loc[m, 'C2_p'].values, Sdf.loc[m, 'C3_p'].values])
        n = m.sum()
        Sdf.loc[m, 'C2_q_BH'], Sdf.loc[m, 'C3_q_BH'] = q[:n], q[n:]
    Sdf.to_csv(OUT / 'validation_summary.csv', index=False)
    # k selection (per architecture, mean E(alpha=1) over seeds; validation, default wording, base SAEs)
    ksel = {}
    for arch in ARCHS:
        mm = Sdf[base & (Sdf.architecture == arch)].groupby('k').E_alpha1.mean()
        ksel[arch] = {'k': int(mm.idxmax()), 'mean_E_alpha1_by_k': {int(a): float(b) for a, b in mm.items()}}
    (OUT / 'k_selection.json').write_text(json.dumps(ksel, indent=2))
    # descriptive validation criteria (README §7 applied to validation; optimistic for the selected k)
    crit = {}
    for arch in ARCHS:
        crit[arch] = {}
        for k in KS:
            g = Sdf[base & (Sdf.architecture == arch) & (Sdf.k == k)].set_index('seed')
            cw = Sdf[(Sdf.sae == 'base') & (Sdf.set == 'cw') & (Sdf.architecture == arch) & (Sdf.k == k)].set_index('seed')
            c1 = int(((g.C2_q_BH < 0.05) & (g.C3_q_BH < 0.05)).sum()) >= 2
            c2 = int((g.E_alpha_m1_ci_hi < 0).sum()) >= 2
            c3 = int((g.specificity_diff_ci_lo > 0).sum()) >= 2
            c4 = bool((g.owt_delta_ce_target <= g.owt_delta_ce_C2_q95).all())
            c5 = int((cw.E_alpha1 > cw.C2_q95).sum()) >= 2
            verdict = 'SUPPORTED' if all((c1, c2, c3, c4, c5)) else ('NOT SUPPORTED' if not c1 else 'INCONCLUSIVE')
            crit[arch][f'k={k}'] = {'selected_k': k == ksel[arch]['k'], '1_effect': c1, '2_sign_reversal': c2,
                                    '3_specificity': c3, '4_no_damage_confound': c4, '5_wording_cw_validation': c5,
                                    'VALIDATION_DESCRIPTIVE_VERDICT': verdict}
    (OUT / 'validation_criteria_descriptive.json').write_text(json.dumps(crit, indent=2))
    print(json.dumps({a: {k: v['VALIDATION_DESCRIPTIVE_VERDICT'] for k, v in c.items()} for a, c in crit.items()}, indent=2))
    print(json.dumps(ksel, indent=2))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ('smoke', 'prepare', 'validation', 'test'):
        g.add_argument(f'--{f}', action='store_true')
    a = ap.parse_args()
    if a.test:
        raise SystemExit('R2 test stage is DEFERRED (user decision 2026-11-08); it needs explicit approval.')
    {'smoke': smoke, 'prepare': prepare, 'validation': validation}[next(k for k in ('smoke', 'prepare', 'validation') if getattr(a, k))]()


if __name__ == '__main__':
    main()
