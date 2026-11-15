#!/usr/bin/env python3
"""E3: causal preservation of the IMP-L layer-8 fine-tuned SAEs, with the M4 protocol at the layer-8 final token.

Pre-registered in results/repe/e3_finetuned_causal_preservation/README.md (section 1) and preregistration.json
(written by --prepare, with SHA-256 hashes of the scripts, the data and the 36 checkpoints, before any E3 result).

36 SAEs at blocks.8.hook_resid_pre: 9 base (uninterrupted 15k-step OWT SAEs) + 27 IMP-L fine-tunes (owt_only,
method_a, kl) x relu/topk/jumprelu x seeds 42/1042/2042. No training. RepE-LM validation pairs only; the feature
ranking uses TRAIN T2-T4, exactly as in M4. Every metric is computed by the M4 functions imported unchanged from
scripts/repe/24_sae_patch_decomposition.py: the SAE reconstruction is inserted into GPT-2's forward pass at the site
(start_at_layer from the clean residual), not a steering direction.

Implementation details fixed before any result (see preregistration.json "implementation"):
  * every SAE gets the same random seed (generator, numpy rng and the CPU shuffle generator are re-seeded per SAE),
    so control draws are paired across arms; M4 used one running stream, so its p-values/CIs are not reproduced
    bit for bit, only its deterministic quantities;
  * the M4 reproduction check compares the deterministic columns of hypotheses.csv (T_full/T_rec/T_err means,
    preservation_rec, shapley_share_rec, |I|/T_full) for the 9 base SAEs, tolerance 1e-6; it runs first and the
    script stops before any fine-tuned SAE if it fails;
  * H4a/H4b/H4c use Benjamini-Hochberg over the 36 SAEs (M4: over its 27 SAE x site cells).

Run from the repository root:
    python scripts/repe/25_e3_finetuned_causal_preservation.py --prepare
    python scripts/repe/25_e3_finetuned_causal_preservation.py --smoke --smoke-out DIR     (mechanics only)
    python scripts/repe/25_e3_finetuned_causal_preservation.py --run
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
from src.evaluation import patching as pt  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S24_REL = 'scripts/repe/24_sae_patch_decomposition.py'
S24 = _load(S24_REL, 's24')

OUT = REPO / 'results/repe/e3_finetuned_causal_preservation'
M4 = REPO / 'results/repe/sae_patch_decomposition'
IMPL = REPO / 'results/geometry_of_truth/multilayer_finetune'
CKPT = REPO / 'results/geometry_of_truth/sae_finetuning/multilayer_finetune/layer_8'
L, GROUP = 8, 'final'
SITE = f'blocks.{L}.hook_resid_pre|{GROUP}'
ARMS = ('base', 'owt_only', 'method_a', 'kl')
ARCHS, SEEDS = S24.ARCHS, S24.SEEDS
SEED = S24.SEED
T_975_DF2 = 4.302652729911275
DET_COLS = ['T_full_mean', 'T_rec_mean', 'T_err_mean', 'preservation_rec', 'shapley_share_rec', 'abs_I_over_T_full']
OUTPUT_FILES = ['per_pair_validation.csv', 'decomposition_summary.csv', 'features_summary.csv', 'hypotheses.csv',
                'site_reconstruction.csv', 'comparisons.csv', 'criteria.json', 'sanity_checks.json',
                'selected_features.json', 'control_draws.npz']
HASHED = [S24_REL, 'scripts/repe/25_e3_finetuned_causal_preservation.py', 'src/evaluation/patching.py',
          'src/evaluation/faithfulness_metrics.py', 'data/repe/repe_lm/prompts.csv',
          'results/repe/e3_finetuned_causal_preservation/README.md',
          'results/geometry_of_truth/multilayer_finetune/criteria.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ckpt_path(arm, arch, seed):
    if arm == 'base':
        return Path(uninterrupted_checkpoint(L, arch, seed))
    return CKPT / f'{arch}_{arm}_seed{seed}' / 'finetuned_checkpoint.pt'


def all_saes():
    return [(arm, arch, seed, ckpt_path(arm, arch, seed)) for arm in ARMS for arch in ARCHS for seed in SEEDS]


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


# ------------------------------------------------------------------------------------------- prepare

def prepare():
    pr_path = OUT / 'preregistration.json'
    if pr_path.exists():
        raise SystemExit(f'{pr_path} exists; refusing to overwrite.')
    impl8 = pd.read_csv(IMPL / 'per_sae_layer8.csv')
    ck = {}
    for arm, arch, seed, p in all_saes():
        if not p.exists():
            raise SystemExit(f'missing checkpoint {p}')
        h = sha(p)
        row = impl8[(impl8.arm == arm) & (impl8.architecture == arch) & (impl8.seed == seed)]
        assert len(row) == 1 and row.checkpoint_sha256.iloc[0] == h, f'{p}: hash differs from IMP-L evaluation'
        ck[rel(p)] = h
    crit = json.loads((IMPL / 'criteria.json').read_text())
    guards = {arch: {'method_a': crit[f'layer8_{arch}']['method_a']['guards'],
                     'kl': crit[f'layer8_{arch}']['kl']['guards']} for arch in ARCHS}
    pr = {
        'experiment': 'E3 causal preservation of the IMP-L layer-8 fine-tuned SAEs (M4 protocol, L8 final token)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'approved_by_user': '2026-11-10 ("Complete causal faithfulness evaluation (E3). This is the highest-priority '
                            'new experiment ... Check the existing E3 preregistration before implementation. Do not '
                            'change its success criteria after observing results.")',
        'evidence_label': 'VALIDATION (RepE-LM validation pairs; TRAIN T2-T4 for feature ranking and k only; test not loaded)',
        'design_source': 'results/repe/e3_finetuned_causal_preservation/README.md section 1 (draft of 2026-11-09, '
                         'unchanged); this file adds hashes and implementation details only',
        'site': SITE,
        'saes': '9 base + 27 IMP-L layer-8 fine-tunes (owt_only, method_a, kl) x relu/topk/jumprelu x 42/1042/2042',
        'hypotheses': {
            'E3-H1': 'method_a raises preservation_rec = T_rec/T_full, consistent vs base AND vs owt_only',
            'E3-H2': 'kl raises preservation_rec over method_a, consistent',
            'E3-H3': 'specificity: a gain in H1/H2 must also appear in rec_minus_syn_rec = T_rec - S_rec (persona minus '
                     'synonym), same pairs and rule; otherwise labelled non-specific'},
        'consistent_rule': 'better on all 3 seeds AND |mean paired difference| > 2 x sd (IMP-L rule)',
        'classes': 'IMP-L definitions: NOT CREDITABLE if any inherited guard failed; REGRESSION if consistently worse '
                   '(H1: vs base; H2: vs method_a); PRELIMINARY IMPROVEMENT if the criterion is met; else NO '
                   'IMPROVEMENT. CONFIRMED is not used.',
        'guards_inherited_from_impl_criteria_json': guards,
        'stated_in_advance': 'kl relu and kl jumprelu at layer 8 failed the IMP-L L0 guard, so their E3-H2 class is '
                             'NOT CREDITABLE a priori; only topk can earn a creditable kl class',
        'secondary': ['shapley_share_rec (up)', 'preservation_err = T_err/T_full (down)', 'abs_I_over_T_full (down)',
                      'rec_minus_syn_rec_mean (up)', 'H4a/H4b/H4c passes (M4 definitions, BH over the 36 SAEs)',
                      'k_star and F_kstar_share_of_T_rec', 'T_full (must be identical across SAEs)',
                      'site reconstruction on RepE-LM validation prompts at L8 final: MSE, EV, cosine, L0 (reported)'],
        'implementation': {
            'code': 'M4 functions imported unchanged (Engine, build_rows, clean_batches, decompose, add_derived, '
                    'summarise_decomp, feature_analysis, owt_feature_stats)',
            'seeding': 'generator, numpy rng and the CPU shuffle generator re-seeded to 20261108 before every SAE',
            'reproduction_check': 'deterministic hypotheses.csv columns of the 9 base SAEs vs M4 L8 final, tol 1e-6; '
                                  'checked before any fine-tuned SAE is evaluated; abort on failure',
            'bh': 'H4a/H4b/H4c q-values over the 36 SAEs',
            'statistics': 'per architecture, paired by seed; mean, sd, 95 % t-interval (t = 4.303, df 2), per-seed values',
            'smoke': 'run on 1/10 of the pairs with 3-5 control draws into a scratch directory, mechanics only; no '
                     'hypothesis value is inspected'},
        'test_rows_used': 0,
        'hashes': {r_: sha(REPO / r_) for r_ in HASHED},
        'checkpoint_hashes': ck,
    }
    pr_path.write_text(json.dumps(pr, indent=2))
    print(f'wrote {pr_path} ({len(ck)} checkpoints hashed)')


# ------------------------------------------------------------------------------------------ evaluation

def reseed(device):
    torch.manual_seed(SEED)
    S24.GEN_CPU.manual_seed(SEED)
    return torch.Generator(device=device).manual_seed(SEED), np.random.default_rng(SEED)


@torch.no_grad()
def site_reconstruction(sae, batches):
    """RepE-LM validation prompts (H, U, Hp) at the site: MSE, EV, cosine and L0 of the SAE (reported only)."""
    xs = torch.cat([bt['x'][v] for bt in batches for v in ('H', 'U', 'Hp')])
    r, z, _ = sae(xs)
    r = r.float()
    err = ((xs - r) ** 2).sum(-1)
    return dict(site_n_tokens=int(len(xs)), site_mse=float(err.mean() / xs.shape[1]),
                site_ev=float(1 - err.sum() / ((xs - xs.mean(0)) ** 2).sum()),
                site_cosine=float(torch.nn.functional.cosine_similarity(r, xs, dim=-1).mean()),
                site_l0=float((z > 0).float().sum(-1).mean()),
                site_dead_fraction=float(((z > 0).sum(0) == 0).float().mean()))


def evaluate_sae(eng, sae, vb, vn, tb, meta, device, sanity, store):
    gen, rng = reseed(device)
    dfp, dr, ident = S24.decompose(eng, sae, vb, 'persona', gen)
    dfn, _, ident2 = S24.decompose(eng, sae, vn, 'neutral', gen)
    sanity['max_identity_abs_err'] = max(sanity['max_identity_abs_err'], ident, ident2)
    dfp, dfn = S24.add_derived(dfp), S24.add_derived(dfn)
    sanity['max_abs_shapley_sum_err'] = max(sanity['max_abs_shapley_sum_err'],
                                            float((dfp.phi_rec + dfp.phi_err - dfp.T_full).abs().max()))
    for d_ in (dfp, dfn):
        store['pairs'].append(d_.assign(**meta))
    store['summ'] += S24.summarise_decomp(dfp, dr, dict(meta, kind='persona'), rng)
    store['summ'] += S24.summarise_decomp(dfn, {}, dict(meta, kind='neutral'), rng)
    key = f"{meta['arm']}_{meta['architecture']}_{meta['seed']}"
    for k, v in dr.items():
        store['draws'][f'{key}_{k}'] = v.astype(np.float32)
    freq, corr = S24.owt_feature_stats(sae, L, device)
    fres, fper, set_means, iso_means = S24.feature_analysis(eng, sae, tb, vb, freq, corr, gen, rng)
    sanity['max_abs_T_rec_vs_feature_all'] = max(sanity['max_abs_T_rec_vs_feature_all'],
                                                 float((fper.F_all - fper.T_rec_check).abs().max()))
    store['feat_json'][key] = fres
    for ts, mask in (('T2_T4', fper.template.isin(S24.PRIMARY)), ('T1', fper.template == 'T1')):
        g = fper[mask]
        sid = g.statement_id.to_numpy()
        t_rec = pt.stat_summary(g.T_rec_check.to_numpy(), sid, S24.N_BOOT, rng)['mean']
        fr = dict(meta, template_set=ts, k_star=fres['k_star'], T_rec=t_rec)
        for c in [c for c in g.columns if c.startswith('F_')]:
            s = pt.stat_summary(g[c].to_numpy(), sid, S24.N_BOOT, rng)
            fr[f'{c}_mean'], fr[f'{c}_ci_low'] = s['mean'], s['ci_low']
        fk = fr[f"F_attr_k{fres['k_star']}_mean"]
        fr['F_kstar_share_of_T_rec'] = fk / t_rec if t_rec else np.nan
        fr['random_set_mean'], fr['random_set_q95'] = float(set_means[ts].mean()), float(np.quantile(set_means[ts], 0.95))
        fr['iso_mean'], fr['iso_q95'] = float(iso_means[ts].mean()), float(np.quantile(iso_means[ts], 0.95))
        fr['p_vs_random_sets'] = float((1 + (set_means[ts] >= fk).sum()) / (1 + len(set_means[ts])))
        fr['p_vs_iso'] = float((1 + (iso_means[ts] >= fk).sum()) / (1 + len(iso_means[ts])))
        store['feat'].append(fr)
        store['draws'][f'{key}_{ts}_featsets'] = set_means[ts].astype(np.float32)
    store['site'].append(dict(meta, **site_reconstruction(sae, vb)))


def hypotheses(sm, fs):
    P = sm[(sm.kind == 'persona') & (sm.template_set == 'T2_T4')].copy()
    F = fs[fs.template_set == 'T2_T4'].set_index(['arm', 'architecture', 'seed'])
    idx = list(zip(P.arm, P.architecture, P.seed))
    P['H4a_p'] = P[['T_rec_minus_iso_rec_p', 'T_rec_minus_shuf_rec_p']].max(1)
    P['H4b_p'] = P['phi_diff_p']
    P['H4c_p'] = [F.loc[i, 'p_vs_random_sets'] for i in idx]
    for h in ('H4a', 'H4b', 'H4c'):
        P[f'{h}_q'] = S24.pt_bh(P[f'{h}_p'].to_numpy())
    P['H4a_pass'] = (P.H4a_q < 0.05) & P.T_rec_above_controls_q95 & (P.T_rec_ci_low > 0)
    P['H4b_pass'] = P.H4b_q < 0.05
    kci = np.array([F.loc[i, f"F_attr_k{int(F.loc[i, 'k_star'])}_ci_low"] > 0 for i in idx], dtype=bool)
    P['H4c_pass'] = (P.H4c_q.to_numpy() < 0.05) & kci
    P['k_star'] = [int(F.loc[i, 'k_star']) for i in idx]
    P['F_kstar_share_of_T_rec'] = [F.loc[i, 'F_kstar_share_of_T_rec'] for i in idx]
    keep = ['arm', 'architecture', 'seed', 'checkpoint', 'T_full_mean', 'T_rec_mean', 'T_rec_ci_low', 'T_rec_ci_high',
            'T_err_mean', 'I_mean', 'preservation_rec', 'preservation_rec_ci_low', 'preservation_rec_ci_high',
            'preservation_err', 'shapley_share_rec', 'shapley_share_rec_ci_low', 'shapley_share_rec_ci_high',
            'abs_I_over_T_full', 'additive_reading_allowed', 'S_full_mean', 'S_rec_mean', 'rec_minus_syn_rec_mean',
            'rec_minus_syn_rec_ci_low', 'rec_minus_syn_rec_ci_high', 'syn_preservation_rec', 'TB_preservation_rec',
            'iso_rec_mean', 'iso_rec_q95', 'shuf_rec_mean', 'shuf_rec_q95', 'T_rec_above_controls_q95',
            'H4a_p', 'H4a_q', 'H4a_pass', 'H4b_p', 'H4b_q', 'H4b_pass', 'H4c_p', 'H4c_q', 'H4c_pass',
            'k_star', 'F_kstar_share_of_T_rec']
    P['effect_shortfall'] = P.T_full_mean - P.T_rec_mean
    return P[keep + ['effect_shortfall']]


HIGHER = {'preservation_rec': True, 'shapley_share_rec': True, 'rec_minus_syn_rec_mean': True, 'T_rec_mean': True,
          'preservation_err': False, 'abs_I_over_T_full': False, 'effect_shortfall': False,
          'F_kstar_share_of_T_rec': True, 'k_star': False, 'site_mse': False, 'site_ev': True, 'site_cosine': True,
          'site_l0': None}
PAIRS = [('owt_only', 'base'), ('method_a', 'base'), ('method_a', 'owt_only'), ('kl', 'base'), ('kl', 'owt_only'),
         ('kl', 'method_a')]


def compare(hyp, site):
    df = hyp.merge(site[['arm', 'architecture', 'seed', 'site_mse', 'site_ev', 'site_cosine', 'site_l0']],
                   on=['arm', 'architecture', 'seed'])
    comps, crit = [], {}
    impl = json.loads((IMPL / 'criteria.json').read_text())
    for arch, sub in df.groupby('architecture'):
        piv = {a: sub[sub.arm == a].set_index('seed').sort_index() for a in ARMS}
        for a_, b_ in PAIRS:
            for mtr in HIGHER:
                dd = piv[a_][mtr].astype(float) - piv[b_][mtr].astype(float)
                m, s = dd.mean(), dd.std()
                comps.append(dict(architecture=arch, comparison=f'{a_} - {b_}', metric=mtr,
                                  higher_is_better=HIGHER[mtr], base_mean=piv[b_][mtr].mean(),
                                  arm_mean=piv[a_][mtr].mean(), mean_diff=m, sd_diff=s,
                                  ci95_low=m - T_975_DF2 * s / np.sqrt(3), ci95_high=m + T_975_DF2 * s / np.sqrt(3),
                                  pct_change=100 * m / piv[b_][mtr].mean() if piv[b_][mtr].mean() else np.nan,
                                  **{f'seed{k}': dd.get(k) for k in SEEDS}))

        def up(a_, b_, mtr):
            dd = piv[a_][mtr] - piv[b_][mtr]
            return {'pass': bool((dd > 0).all() and dd.mean() > 2 * dd.std()), 'paired_diffs_by_seed': dd.round(6).tolist()}

        def down(a_, b_, mtr):
            dd = piv[a_][mtr] - piv[b_][mtr]
            return bool((dd < 0).all() and -dd.mean() > 2 * dd.std())

        def classify(primary, g, wrong):
            if not all(g.values()):
                return 'NOT CREDITABLE (guard failed)'
            if wrong:
                return 'REGRESSION'
            return 'PRELIMINARY IMPROVEMENT' if primary else 'NO IMPROVEMENT'

        gma = impl[f'layer8_{arch}']['method_a']['guards']
        gkl = impl[f'layer8_{arch}']['kl']['guards']
        p = 'preservation_rec'
        h1 = {'vs_base': up('method_a', 'base', p), 'vs_owt_only': up('method_a', 'owt_only', p)}
        h1_pass = h1['vs_base']['pass'] and h1['vs_owt_only']['pass']
        h2 = up('kl', 'method_a', p)
        spec1 = {'vs_base': up('method_a', 'base', 'rec_minus_syn_rec_mean'),
                 'vs_owt_only': up('method_a', 'owt_only', 'rec_minus_syn_rec_mean')}
        spec2 = up('kl', 'method_a', 'rec_minus_syn_rec_mean')
        c1 = classify(h1_pass, gma, down('method_a', 'base', p))
        c2 = classify(h2['pass'], gkl, down('kl', 'method_a', p))
        crit[f'layer8_{arch}'] = {
            'E3_H1_method_a': {'criterion': h1, 'guards': gma, 'class': c1,
                               'H3_specificity': spec1,
                               'specific': bool(spec1['vs_base']['pass'] and spec1['vs_owt_only']['pass'])
                               if c1 == 'PRELIMINARY IMPROVEMENT' else None},
            'E3_H2_kl': {'criterion': h2, 'guards': gkl, 'class': c2, 'H3_specificity': spec2,
                         'specific': bool(spec2['pass']) if c2 == 'PRELIMINARY IMPROVEMENT' else None},
            'descriptive_not_criteria': {
                'kl_vs_base_up': up('kl', 'base', p), 'kl_vs_owt_only_up': up('kl', 'owt_only', p),
                'owt_only_vs_base_up': up('owt_only', 'base', p),
                'method_a_vs_base_down': down('method_a', 'base', p), 'kl_vs_base_down': down('kl', 'base', p)}}
    return pd.DataFrame(comps), crit


def run(out, smoke):
    if not smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if r_.endswith('README.md'):
                continue          # README section 2 is written after the run; section 1 is frozen by its hash below
            if sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs from preregistration.json; refusing to run.')
        for r_, h in pr['checkpoint_hashes'].items():
            if sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: checkpoint hash differs; refusing to run.')
    else:
        S24.N_VEC, S24.N_SET, S24.N_BOOT = 3, 5, 200
    for f in OUTPUT_FILES:
        if (out / f).exists():
            raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = 'cuda'
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    params_before = hashlib.sha256(b''.join(p_.detach().cpu().numpy().tobytes() for p_ in model.parameters())).hexdigest()
    tok = AutoTokenizer.from_pretrained('gpt2')
    rows = {(s, k): S24.build_rows(tok, s, S24.PERSONA if k == 'persona' else S24.NEUTRAL,
                                   ['H', 'U', 'Hp'] if k == 'persona' else ['Na', 'Nb'])
            for s in ('train', 'validation') for k in ('persona', 'neutral')}
    if smoke:
        rows = {k: v[::10] for k, v in rows.items()}
    eng = S24.Engine(model, L)
    vb = S24.clean_batches(eng, rows[('validation', 'persona')], ['H', 'U', 'Hp'], GROUP, device)
    vn = S24.clean_batches(eng, rows[('validation', 'neutral')], ['Na', 'Nb'], GROUP, device)
    tb = S24.clean_batches(eng, [r for r in rows[('train', 'persona')] if r['template'] in S24.PRIMARY],
                           ['H', 'U'], GROUP, device)
    sanity = dict(max_identity_abs_err=0.0, max_start_at_layer_kl=0.0, max_abs_T_rec_vs_feature_all=0.0,
                  max_abs_shapley_sum_err=0.0, n_validation_persona_pairs=len(rows[('validation', 'persona')]),
                  n_validation_neutral_pairs=len(rows[('validation', 'neutral')]),
                  n_train_T2_T4_pairs=sum(r['template'] in S24.PRIMARY for r in rows[('train', 'persona')]),
                  site_hook=eng.hook, site_position='final token (n - 1) of each equal-length pair')
    bt = vb[0]
    l0 = eng.patched(bt['resid']['U'], bt['pos'], bt['finals'], [torch.zeros_like(bt['x']['U'])])[0]
    sanity['max_start_at_layer_kl'] = float(S24.kl(bt['lp']['U'], l0).abs().max())
    # positions: every pair's patch position must be its last real token
    sanity['positions_are_final_token'] = bool(all(int(p) == r['n'] - 1 for b in vb for p, r in zip(b['pos'], b['rows'])))
    store = dict(pairs=[], summ=[], feat=[], feat_json={}, draws={}, site=[])
    for i, (arm, arch, seed, path) in enumerate(all_saes()):
        sae, _ = fm.load_sae(path, device, architecture=arch)
        meta = dict(layer=L, group=GROUP, site=SITE, arm=arm, architecture=arch, seed=seed, checkpoint=rel(path))
        evaluate_sae(eng, sae, vb, vn, tb, meta, device, sanity, store)
        print(f'{arm} {arch} {seed}: done ({time.time() - t0:.0f}s)', flush=True)
        del sae
        if i == 8 and not smoke:          # the 9 base SAEs are first: reproduce M4 before any fine-tuned SAE
            sm0 = pd.DataFrame(store['summ'])
            mine = sm0[(sm0.kind == 'persona') & (sm0.template_set == 'T2_T4')].set_index(['architecture', 'seed'])
            m4 = pd.read_csv(M4 / 'hypotheses.csv')
            m4 = m4[m4.site == SITE].set_index(['architecture', 'seed'])
            err = float(max((mine.loc[m4.index, c] - m4[c]).abs().max() for c in DET_COLS))
            sanity['m4_reproduction_max_abs_err'] = err
            sanity['m4_reproduction_pass'] = bool(err < 1e-6)
            print('M4 reproduction max abs err', err, flush=True)
            if err >= 1e-6:
                (out / 'sanity_checks.json').write_text(json.dumps(sanity, indent=2, default=float))
                raise SystemExit('M4 reproduction failed; stopping before any fine-tuned SAE.')
    params_after = hashlib.sha256(b''.join(p_.detach().cpu().numpy().tobytes() for p_ in model.parameters())).hexdigest()
    pairs = pd.concat(store['pairs'], ignore_index=True)
    sm, fs, site = pd.DataFrame(store['summ']), pd.DataFrame(store['feat']), pd.DataFrame(store['site'])
    hyp = hypotheses(sm, fs)
    comps, crit = compare(hyp, site)
    sanity['T_full_identical_across_saes'] = float(pairs[pairs.kind == 'persona'].groupby(
        ['statement_id', 'template']).T_full.agg(lambda v: v.max() - v.min()).max())
    sanity['n_saes_evaluated'] = int(len(hyp))
    sanity['gpt2_parameters_unchanged'] = params_before == params_after
    sanity['pass'] = bool(sanity['max_identity_abs_err'] < 1e-3 and sanity['max_start_at_layer_kl'] < 1e-6
                          and sanity['max_abs_shapley_sum_err'] < 1e-9 and sanity['max_abs_T_rec_vs_feature_all'] < 1e-4
                          and sanity['T_full_identical_across_saes'] < 1e-9 and sanity['positions_are_final_token']
                          and sanity['gpt2_parameters_unchanged'] and (smoke or sanity['m4_reproduction_pass']))
    sanity.update(runtime_s=time.time() - t0, test_rows_used=0, script_sha256=sha(__file__),
                  gpu_peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), smoke=smoke)
    pairs.to_csv(out / 'per_pair_validation.csv', index=False)
    sm.to_csv(out / 'decomposition_summary.csv', index=False)
    fs.to_csv(out / 'features_summary.csv', index=False)
    hyp.to_csv(out / 'hypotheses.csv', index=False)
    site.to_csv(out / 'site_reconstruction.csv', index=False)
    comps.to_csv(out / 'comparisons.csv', index=False)
    (out / 'criteria.json').write_text(json.dumps(crit, indent=2, default=float))
    (out / 'selected_features.json').write_text(json.dumps(store['feat_json'], indent=1, default=float))
    (out / 'sanity_checks.json').write_text(json.dumps(sanity, indent=2, default=float))
    np.savez_compressed(out / 'control_draws.npz', **store['draws'])
    print(json.dumps(sanity, indent=1, default=float))
    if smoke:
        print('smoke: mechanics only; hypothesis values not printed')
    else:
        print(json.dumps({k: {h: v[h]['class'] for h in ('E3_H1_method_a', 'E3_H2_kl')} for k, v in crit.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--run', action='store_true')
    ap.add_argument('--smoke-out', type=Path)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.smoke:
        if a.smoke_out is None:
            raise SystemExit('--smoke needs --smoke-out')
        run(a.smoke_out, True)
    else:
        run(OUT, False)


if __name__ == '__main__':
    main()
