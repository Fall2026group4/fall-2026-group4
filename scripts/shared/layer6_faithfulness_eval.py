#!/usr/bin/env python3
"""Reusable validation-only per-SAE evaluation at GPT-2 layer 6 (factored from
scripts/geometry_of_truth/15_method_a_evaluation.py, which is left unchanged) plus downstream reconstruction.

    ctx = build_context(device)        # probes on ORIGINAL activations, cached original model outputs
    row = evaluate_sae(sae, ctx)       # dict of metrics

All probes are fitted on ORIGINAL activations (GoT: train groups B, logistic C chosen on original validation, MLP x5,
mass-mean; RepE: train). No GoT/RepE test row is read. Splicing replaces blocks.6.hook_resid_pre at non-BOS positions.
Downstream NMSE (new): on OWT validation, NMSE_k = sum ||a_k(spliced) - a_k(orig)||^2 / sum ||a_k(orig) - mean||^2
over spliced positions, for k = blocks.7..11.hook_resid_pre.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402

LAYER = 6
DOWNSTREAM = tuple(range(7, 12))
MLP_SEEDS = (42, 123, 456, 789, 2026)


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_context(device='cuda'):
    P11 = load('scripts/geometry_of_truth/11_truth_aware_pilot.py', 'p11')
    g07 = load('scripts/geometry_of_truth/07_independent_truth_probe.py', 'g07')
    sp = load('scripts/shared/spliced_reconstruction_evaluation.py', 'splice')
    ev = load('scripts/shared/all_layer_frozen_probe_evaluation.py', 'ev')
    o3 = load('scripts/openwebtext/03_training_sae.py', 'o3')
    got = {}
    for domain in P11.DOMAINS:
        X, m = P11.load_domain(domain)
        y = m.label.to_numpy()
        B, va = (m.half == 'B').to_numpy(), (m.split == 'val').to_numpy()
        assert not (m.split == 'test').to_numpy()[B | va].any()
        _, CB, logB = P11.select_C(X, y, B, va)
        dB = X[B & (y == 1)].astype(np.float64).mean(0) - X[B & (y == 0)].astype(np.float64).mean(0)
        mlps = []
        for s in MLP_SEEDS:
            g07.SEED = s
            torch.manual_seed(s)
            mlps.append(g07.train_probe(X[B], y[B], device))
        got[domain] = dict(X=X, y=y, B=B, va=va, logB=logB, CB=CB, dB=dB, mlps=mlps,
                           o_log=logB.decision_function(X[va]), o_dm=X[va] @ (dB / np.linalg.norm(dB)),
                           o_mlp=[g07.probe_scores(p, s_, X[va], device) for p, s_ in mlps])
    rd = ev.load_repe()[0]['honesty']
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    S = sp.Splicer(model, device)
    owt_seqs, _ = sp.load_owt_validation()
    got_v = sp.load_got_validation(model)
    repe_v = sp.load_repe_validation(model)
    owt_orig = sp.owt_original(S, owt_seqs)
    got_orig = sp.got_original(S, got_v)
    toks, lens = sp.pad(list(repe_v.tokens), 50256)
    repe_orig = S.logprobs(toks, lens)[torch.arange(len(repe_v), device=device), (lens - 1).to(device)]
    return dict(device=device, got=got, g07=g07, sp=sp, ev=ev, rd=rd,
                repe_probes=ev.fit_original_probes(rd['X'][LAYER], rd['y'], rd['split']),
                model=model, S=S, owt_seqs=owt_seqs, got_v=got_v, repe_v=repe_v, owt_orig=owt_orig, got_orig=got_orig,
                repe_orig=repe_orig,
                cities_orig=sp.got_eval(S, got_v, 0, lambda x: x.clone(), got_orig)['cities_truth_pair_accuracy'],
                owt_val=o3.load_split(REPO / 'data/activations/gpt2_small', 'validation', LAYER).to(device))


@torch.no_grad()
def downstream_nmse(sae, ctx, batch=25):
    """Per-layer NMSE of downstream residuals (blocks.7..11.hook_resid_pre) on OWT validation when layer 6 is spliced."""
    sp, model, dev = ctx['sp'], ctx['model'], ctx['device']
    names = [f'blocks.{k}.hook_resid_pre' for k in DOWNSTREAM]
    err = {n: 0.0 for n in names}
    tot = {n: 0.0 for n in names}
    s1 = {n: 0.0 for n in names}
    cnt = 0
    for b in range(0, len(ctx['owt_seqs']), batch):
        toks, lens = sp.pad(ctx['owt_seqs'][b:b + batch], 50256)
        toks = toks.to(dev)
        ar = torch.arange(toks.shape[1], device=dev)[None]
        mask = (ar >= 1) & (ar < lens.to(dev)[:, None])
        _, co = model.run_with_cache(toks, names_filter=lambda n: n in names)
        store = {}

        def h6(r, hook):
            r[mask] = sae(r[mask].float())[0].to(r.dtype)
            return r

        def keep(r, hook):
            store[hook.name] = r.clone()
            return r
        model.run_with_hooks(toks, fwd_hooks=[(f'blocks.{LAYER}.hook_resid_pre', h6)] + [(n, keep) for n in names])
        for n in names:
            a, ah = co[n][mask].double(), store[n][mask].double()
            err[n] += (ah - a).square().sum().item()
            tot[n] += a.square().sum().item()
            s1[n] = s1[n] + a.sum(0) if torch.is_tensor(s1[n]) else a.sum(0)
        cnt += int(mask.sum())
    out = {}
    for k, n in zip(DOWNSTREAM, names):
        var = tot[n] - (s1[n].square().sum().item() / cnt)   # sum ||a - mean||^2
        out[f'downstream_nmse_L{k}'] = err[n] / var
    out['downstream_nmse_mean'] = float(np.mean([out[f'downstream_nmse_L{k}'] for k in DOWNSTREAM]))
    return out


def evaluate_sae(sae, ctx):
    dev, g07, sp, ev, rd = ctx['device'], ctx['g07'], ctx['sp'], ctx['ev'], ctx['rd']
    r = {}
    for domain, d in ctx['got'].items():
        X, y, va, B = d['X'], d['y'], d['va'], d['B']
        Xh, l0_all, _ = fm.reconstruct(sae, X, dev)
        Xh = Xh.numpy()
        rec = fm.reconstruction_metrics(X[va], Xh[va], l0_all[torch.as_tensor(va)])
        yv = y[va]
        s_log = d['logB'].decision_function(Xh[va])
        mlp_gaps = [fm.safe_auroc(yv, o) - fm.safe_auroc(yv, g07.probe_scores(p, s_, Xh[va], dev))
                    for (p, s_), o in zip(d['mlps'], d['o_mlp'])]
        refit = make_pipeline(StandardScaler(), LogisticRegression(C=d['CB'], max_iter=3000, random_state=42)).fit(Xh[B], y[B])
        dBh = Xh[B & (y == 1)].astype(np.float64).mean(0) - Xh[B & (y == 0)].astype(np.float64).mean(0)
        pre = f'got_{domain}_'
        r.update({pre + 'indep_logistic_gap': fm.safe_auroc(yv, d['o_log']) - fm.safe_auroc(yv, s_log),
                  pre + 'indep_mlp_gap_mean': float(np.mean(mlp_gaps)), pre + 'indep_mlp_gap_sd': float(np.std(mlp_gaps, ddof=1)),
                  pre + 'indep_diffmean_gap': fm.safe_auroc(yv, d['o_dm']) - fm.safe_auroc(yv, Xh[va] @ (d['dB'] / np.linalg.norm(d['dB']))),
                  pre + 'indep_logistic_score_corr': float(np.corrcoef(d['o_log'], s_log)[0, 1]),
                  pre + 'DIAG_refit_on_recon_val_auroc': fm.safe_auroc(yv, refit.decision_function(Xh[va])),
                  pre + 'direction_cosine_B': fm.direction_fidelity(d['dB'], dBh)['direction_cosine'],
                  pre + 'val_relative_sq_error': rec['relative_sq_error'], pre + 'val_cosine': rec['cosine'],
                  pre + 'val_l0': rec['l0']})
    Xr = rd['X'][LAYER]
    Xrh, l0r, _ = fm.reconstruct(sae, Xr, dev)
    Xrh = Xrh.numpy()
    rv, rtr, yr = rd['split'] == 'validation', rd['split'] == 'train', rd['y']
    so, sr = ev.scores(ctx['repe_probes'], Xr), ev.scores(ctx['repe_probes'], Xrh)
    dr = Xrh[rtr & (yr == 1)].astype(np.float64).mean(0) - Xrh[rtr & (yr == 0)].astype(np.float64).mean(0)
    rrec = fm.reconstruction_metrics(Xr[rv], Xrh[rv], l0r[torch.as_tensor(rv)])
    r.update(repe_logistic_gap=fm.safe_auroc(yr[rv], so['logistic'][rv]) - fm.safe_auroc(yr[rv], sr['logistic'][rv]),
             repe_diffmean_gap=fm.safe_auroc(yr[rv], so['diffmean'][rv]) - fm.safe_auroc(yr[rv], sr['diffmean'][rv]),
             repe_direction_cosine=fm.direction_fidelity(ctx['repe_probes']['raw_direction'], dr)['direction_cosine'],
             repe_val_relative_sq_error=rrec['relative_sq_error'], repe_val_l0=rrec['l0'])
    xh, l0, act = fm.reconstruct(sae, ctx['owt_val'], dev)
    rec = fm.reconstruction_metrics(ctx['owt_val'].cpu(), xh, l0)
    r.update(owt_val_ev=rec['explained_variance'], owt_val_mse=rec['mse'], owt_val_cosine=rec['cosine'],
             owt_val_l0=rec['l0'], owt_val_dead_fraction=fm.dead_fraction(act))
    S, fn = ctx['S'], sp.sae_fn(sae)
    o = sp.owt_eval(S, ctx['owt_seqs'], LAYER, fn, ctx['owt_orig'])
    gg = sp.got_eval(S, ctx['got_v'], LAYER, fn, ctx['got_orig'])
    rp = sp.repe_eval(S, ctx['repe_v'], LAYER, fn, ctx['repe_orig'], 'final')
    r.update(splice_owt_delta_ce=o['delta_ce'], splice_owt_kl=o['kl_mean'], splice_owt_top1=o['top1_agreement'],
             splice_got_statement_kl=gg['statement_kl_mean'],
             splice_got_cities_pref_change=gg['cities_truth_pair_accuracy'] - ctx['cities_orig'],
             splice_repe_final_kl=rp['final_kl_mean'], splice_repe_persona_shift_cosine=rp['persona_shift_cosine'])
    r.update(downstream_nmse(sae, ctx))
    return r
