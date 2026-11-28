#!/usr/bin/env python3
"""XF: fresh, never-used cities-style truth TEST built from geonames (2026-11-23).

Pre-registered in results/cross_model/xf_fresh_cities_test/preregistration.json. Every non-build stage checks the
pre-registered hashes and refuses on mismatch; every stage refuses to overwrite its outputs.

Why. The GoT cities TEST rows were exposed during GPT-2 method decisions (audit 2026-11-21 row 23). XF builds a new
cities family from got-repo/datasets/geonames.csv with the GoT template ("The city of X is in Y.") and the GoT
country pool, using only cities that do not occur in GoT cities / neg_cities. Its TEST split is read once.

Stages (run from the repository root with ../venv/bin/python, in order):
  build   text only: filters, leakage checks, one TRUE and one FALSE statement per city, a salted-hash city-level
          DEV/TEST split. No model is run. Prints DEV label balance only.
  smoke   the full extraction + reader + statistics pipeline on GoT cities TRAIN rows only (pseudo-DEV / pseudo-TEST
          halves by city); never touches the fresh family.
  gate    DEV rows only: original-activation grouped-CV logistic AUROC per (model, layer); eligible if >= 0.70.
  test    the single TEST pass (needs gate.json committed): readers fitted on DEV, scored on TEST, for every SAE.
  analyze statistics and criteria from the test outputs (no model).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import tspr_stats as T  # noqa: E402

OUT = REPO / 'results/cross_model/xf_fresh_cities_test'
DATA = REPO / 'data/got/fresh/geonames_cities'
GEONAMES = REPO.parent / 'got-repo/datasets/geonames.csv'
SALT = 'XF-geonames-2026-11-23'
POP_LO, POP_HI = 200_000, 500_000
AMBIG_POP = 10_000             # a name held by another-country place with >= this population is ambiguous
CAP_PER_COUNTRY = 30
DEV_FRAC = 0.40
GATE_AUROC = 0.70
CEILING = 0.995                # original TEST AUROC CI upper bound at or above this: preservation label uninformative
N_BOOT, BOOT_SEED = 2000, 20261124
PUBLIC = {'gpt2': (3, 6, 9), 'gemma': (5, 12, 19)}          # triplet T-B (as XC-3)
TSP_CELLS = {'L8_topk_tspr': 8, 'L11_topk': 11, 'L8_batchtopk': 8, 'L8_relu': 8}
GPT2_LAYERS = sorted(set(PUBLIC['gpt2']) | set(TSP_CELLS.values()))
# Geonames English country name -> GoT cities.csv country string (learned from the 639 GoT cities found in geonames).
COUNTRY_MAP = {
    'Congo': 'the Republic of the Congo', 'Congo, Democratic Republic of the': 'the Democratic Republic of the Congo',
    'Dominican Republic': 'the Dominican Republic', 'Iran, Islamic Rep. of': 'Iran',
    "Korea, Dem. People's Rep. of": 'North Korea', 'Korea, Republic of': 'South Korea',
    'Moldova, Republic of': 'Moldova', 'Philippines': 'the Philippines', 'Russian Federation': 'Russia',
    'Sudan, The Republic of': 'Sudan', 'Syrian Arab Republic': 'Syria', 'Tanzania, United Republic of': 'Tanzania',
    'United Arab Emirates': 'the United Arab Emirates', 'United States': 'the United States',
    'Venezuela, Bolivarian Rep. of': 'Venezuela', 'Viet Nam': 'Vietnam'}
PROJECT_TEXT = ['data/got/cities.csv', 'data/got/neg_cities.csv', 'data/got/larger_than.csv',
                'data/got/smaller_than.csv', 'data/repe/facts_clean.csv']
HASHED = ['scripts/cross_model/23_xf_fresh_cities_test.py', 'scripts/cross_model/02_matched_splicing.py',
          'src/evaluation/pile_splicing.py', 'src/models/backends.py', 'scripts/shared/sae_ladder_evaluation.py',
          'src/evaluation/tspr_stats.py', 'src/evaluation/faithfulness_metrics.py',
          'scripts/geometry_of_truth/25_truth_subspace_finetune.py',
          'scripts/geometry_of_truth/30_tsp_r_phase1_training.py',
          'scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py',
          'scripts/geometry_of_truth/32_tsp_generalisation.py',
          'data/got/fresh/geonames_cities/statements.csv', 'data/got/fresh/geonames_cities/build_info.json',
          'data/got/cities.csv', 'data/got/splits/cities_splits.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s.lower())


def hfrac(key):
    return int(hashlib.sha256(f'{SALT}|{key}'.encode()).hexdigest()[:12], 16) / 16 ** 12


# ----------------------------------------------------------------------------------------------- build (text only)
def stage_build():
    if DATA.exists():
        sys.exit(f'{DATA} exists; refusing to overwrite.')
    c = pd.read_csv(REPO / 'data/got/cities.csv')
    nc = pd.read_csv(REPO / 'data/got/neg_cities.csv')
    pool = sorted(set(c.correct_country))
    g = pd.read_csv(GEONAMES)
    g['country'] = g['Country name EN'].map(lambda x: COUNTRY_MAP.get(x, x))
    used = {norm(x) for x in set(c.city) | set(nc.city)}
    countries = {norm(x) for x in pool} | {norm(x) for x in g['Country name EN'].dropna()}
    log = {}
    cand = g[(g.Population >= POP_LO) & (g.Population < POP_HI)].copy()
    log['in_population_band'] = len(cand)
    cand = cand[cand.country.isin(pool)]
    log['country_in_got_pool'] = len(cand)
    cand = cand[cand.Name == cand['ASCII Name']]
    cand = cand[cand.Name.str.fullmatch(r"[A-Za-z][A-Za-z .'\-]*[A-Za-z]")]
    log['ascii_simple_name'] = len(cand)
    cand['key'] = cand.Name.map(norm)
    cand = cand[~cand.key.isin(used)]
    log['not_in_got_cities_or_neg_cities'] = len(cand)
    cand = cand[~cand.key.isin(countries)]
    log['not_a_country_name'] = len(cand)
    big = g[g.Population >= AMBIG_POP].assign(key=lambda d: d.Name.map(norm))
    ncty = big.groupby('key').country.nunique()
    cand = cand[cand.key.map(lambda k: ncty.get(k, 1) <= 1)]
    cand = cand.sort_values('Population', ascending=False).drop_duplicates('key')
    log['unambiguous_unique_name'] = len(cand)
    cand['h'] = cand.key.map(hfrac)
    cand = cand.sort_values('h').groupby('country', group_keys=False).head(CAP_PER_COUNTRY)
    log[f'after_cap_{CAP_PER_COUNTRY}_per_country'] = len(cand)
    rng = np.random.default_rng(int(hashlib.sha256(SALT.encode()).hexdigest()[:8], 16))
    rows = []
    for r in cand.sort_values('key').itertuples():
        wrong = [x for x in pool if x != r.country]
        f = wrong[int(rng.integers(len(wrong)))]
        split = 'DEV' if hfrac('split|' + r.key) < DEV_FRAC else 'TEST'
        for lab, cty in ((1, r.country), (0, f)):
            rows.append(dict(statement=f'The city of {r.Name} is in {cty}.', label=lab, city=r.Name, country=cty,
                             correct_country=r.country, population=int(r.Population), geoname_id=int(r._1),
                             split=split))
    df = pd.DataFrame(rows)
    # leakage / integrity checks (all hard)
    proj = set()
    for rel in PROJECT_TEXT:
        p = REPO / rel
        if p.exists():
            proj |= {norm(s) for s in pd.read_csv(p).statement}
    checks = dict(
        statement_dup_with_project_text=int(df.statement.map(norm).isin(proj).sum()),
        city_overlap_with_got=int(df.city.map(norm).isin(used).sum()),
        duplicate_statements=int(df.statement.duplicated().sum()),
        cities_in_both_splits=int(df.groupby('city').split.nunique().gt(1).sum()),
        true_false_per_city_ok=bool((df.groupby('city').label.agg(['sum', 'count']) == [1, 2]).all().all()))
    assert checks['statement_dup_with_project_text'] == 0 and checks['city_overlap_with_got'] == 0
    assert checks['duplicate_statements'] == 0 and checks['cities_in_both_splits'] == 0
    assert checks['true_false_per_city_ok']
    DATA.mkdir(parents=True)
    df.to_csv(DATA / 'statements.csv', index=False)
    dev = df[df.split == 'DEV']
    info = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(GEONAMES.relative_to(REPO.parent)),
                source_sha256=sha(GEONAMES), salt=SALT, population_band=[POP_LO, POP_HI], ambiguity_pop=AMBIG_POP,
                cap_per_country=CAP_PER_COUNTRY, dev_frac=DEV_FRAC, filter_counts=log, checks=checks,
                n_cities=int(df.city.nunique()), n_statements=len(df), n_countries=int(df.correct_country.nunique()),
                dev=dict(cities=int(dev.city.nunique()), statements=len(dev), label_mean=float(dev.label.mean())),
                test=dict(cities=int(df[df.split == 'TEST'].city.nunique()), statements=int((df.split == 'TEST').sum())),
                note='TEST labels are balanced by construction (one TRUE + one FALSE per city); no TEST row is '
                     'tokenised, embedded or scored before the test stage.')
    (DATA / 'build_info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ----------------------------------------------------------------------------------------------- activations
@torch.no_grad()
def extract_gpt2(model, texts, layers, bs=32):
    """BOS + statement, final token, blocks.L.hook_resid_pre (as scripts/geometry_of_truth/31 extract)."""
    hooks = {f'blocks.{L}.hook_resid_pre': L for L in layers}
    out = {L: [] for L in layers}
    for b in range(0, len(texts), bs):
        seqs = [model.to_tokens(s, prepend_bos=True, move_to_device=False)[0] for s in texts[b:b + bs]]
        lens = torch.tensor([len(s) for s in seqs])
        pad = torch.full((len(seqs), int(lens.max())), model.tokenizer.eos_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :len(s)] = s
        _, cache = model.run_with_cache(pad.cuda(), names_filter=lambda n: n in hooks, return_type=None)
        for h, L in hooks.items():
            out[L].append(cache[h][torch.arange(len(seqs)), (lens - 1).cuda()].float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


@torch.no_grad()
def extract_gemma(bk, texts, layers, bs=32):
    """BOS + statement, final token, output of model.model.layers[L] (as scripts/gemma/10 final_batches)."""
    out = {L: [] for L in layers}
    for s in range(0, len(texts), bs):
        seqs = [[bk.bos_id] + bk.tokenize(t) for t in texts[s:s + bs]]
        Tm = max(map(len, seqs))
        ids = torch.zeros(len(seqs), Tm, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
            am[i, :len(q)] = 1
        cap, hs = {}, []
        for L in layers:
            def h(m, i, o, L=L):
                cap[L] = (o[0] if isinstance(o, tuple) else o).detach()
            hs.append(bk.layers[L].register_forward_hook(h))
        try:
            bk.model(input_ids=ids.to(bk.device), attention_mask=am.to(bk.device), use_cache=False)
        finally:
            for x in hs:
                x.remove()
        last = am.sum(1) - 1
        for L in layers:
            out[L].append(cap[L][torch.arange(len(seqs)), last].float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


def activations(texts, out_npz):
    if out_npz.exists():
        z = np.load(out_npz)
        return {k: z[k] for k in z.files}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    A = {f'gpt2_L{L}': v for L, v in extract_gpt2(m, texts, GPT2_LAYERS).items()}
    del m
    torch.cuda.empty_cache()
    from src.models.backends import GemmaBackend
    bk = GemmaBackend('cuda')
    A |= {f'gemma_L{L}': v for L, v in extract_gemma(bk, texts, list(PUBLIC['gemma'])).items()}
    del bk
    torch.cuda.empty_cache()
    np.savez(out_npz, **A)
    return A


# ----------------------------------------------------------------------------------------------- SAEs
def public_sae(model, L):
    xb = _mod('scripts/cross_model/02_matched_splicing.py', 'xb02')
    spec = [s for s in xb.sae_specs(model, L, ['canonical'] if model == 'gemma' else ['resjb'])
            if s['family'] in ('gemmascope_16k_canonical', 'resjb')][0]
    spec['layer'] = L
    w = xb.load_sae(model, spec, 'cuda')
    p = spec['path'] if model == 'gemma' else spec['src'].get('path')
    return w, spec['sae_id'], (sha(p) if p else None)


def np_recon(fn, X, bs=2048):
    H, L0 = [], []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            h, l0 = fn(torch.as_tensor(X[b:b + bs], device='cuda', dtype=torch.float32))
            H.append(h.float().cpu().numpy())
            L0.append(l0.float().cpu().numpy())
    return np.concatenate(H), np.concatenate(L0)


def tsp_sources(limit=None):
    """(cell, layer, src dict, loader) for every surviving TSP-R L8 TopK and TI-1 SAE (base, ctrl, tsp, rand)."""
    S32 = _mod('scripts/geometry_of_truth/32_tsp_generalisation.py', 's32')
    S31 = S32.S31
    out = []
    for s in S31.sae_list():
        if s['arm'] in ('base', 'ctrl', 'tsp', 'rand'):
            out.append(('L8_topk_tspr', 8, s, lambda s=s: S31.verified(s)))
    for cell in ('L11_topk', 'L8_batchtopk', 'L8_relu'):
        for s in S32.sources(cell):
            if s['arm'] in ('base', 'ctrl', 'tsp', 'rand'):
                def load(s=s):
                    if 'sha256' in s and sha(s['path']) != s['sha256']:
                        raise SystemExit(f"{s['sae_id']}: checkpoint hash mismatch")
                    return S32.SAEnp(s['path'])
                out.append((cell, TSP_CELLS[cell], s, load))
    if limit:                                   # smoke only: the first `limit` seeds of each cell
        keep = []
        for cell in TSP_CELLS:
            seeds = sorted({int(o[2]['seed']) for o in out if o[0] == cell})[:limit]
            keep += [o for o in out if o[0] == cell and int(o[2]['seed']) in seeds]
        out = keep
    return out, S31


# ----------------------------------------------------------------------------------------------- core pass
def score_pass(A_dev, y_dev, g_dev, A_te, y_te, g_te, out, eligible, tsp_limit=None, log=print):
    """Fit readers on DEV, score TEST. Writes one npz per (model, layer) original and per SAE."""
    S31 = _mod('scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py', 's31x')
    out.mkdir(parents=True, exist_ok=True)
    pipes = {}
    for key in A_dev:
        if key not in eligible:
            continue
        Xd, Xt = A_dev[key], A_te[key]
        C = S31.pick_C(Xd, y_dev, g_dev)
        pipe = S31.lr(C).fit(Xd, y_dev)
        a = Xd[y_dev == 1].mean(0) - Xd[y_dev == 0].mean(0)
        pipes[key] = (pipe, a)
        np.savez(out / f'orig__{key}.npz', y=y_te, groups=g_te, logistic=pipe.decision_function(Xt), massmean=Xt @ a,
                 C=C, auroc=roc_auc_score(y_te, pipe.decision_function(Xt)))
        log(f'[orig] {key}: C {C}, AUROC {roc_auc_score(y_te, pipe.decision_function(Xt)):.4f}')

    def readers(key, fn, sid, meta):
        p = out / f'sae__{sid}.npz'
        if p.exists():
            return
        pipe, a = pipes[key]
        Hd, l0d = np_recon(fn, A_dev[key])
        Ht, l0t = np_recon(fn, A_te[key])
        Cr = S31.pick_C(Hd, y_dev, g_dev)
        Xt = A_te[key]
        nmse = float(((Xt - Ht) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum())
        np.savez(p, refit_logistic=S31.lr(Cr).fit(Hd, y_dev).decision_function(Ht),
                 frozen_logistic=pipe.decision_function(Ht), frozen_massmean=Ht @ a, refit_C=Cr,
                 l0_mean=float(np.r_[l0d, l0t].mean()), nmse_test=nmse, meta=json.dumps(meta))

    for model, layers in PUBLIC.items():
        for L in layers:
            key = f'{model}_L{L}'
            if key not in pipes:
                log(f'[public] {key}: not eligible, skipped')
                continue
            w, sid, s_ = public_sae(model, L)
            readers(key, w, f'public_{key}', dict(family='public', model=model, layer=L, sae_id=sid, sha256=s_))
            del w
            torch.cuda.empty_cache()
    srcs, _ = tsp_sources(tsp_limit)
    t0 = time.time()
    for i, (cell, L, s, load) in enumerate(srcs):
        key = f'gpt2_L{L}'
        if key not in pipes:
            continue
        sid = f"tsp_{cell}__{s['sae_id']}"
        if (out / f'sae__{sid}.npz').exists():
            continue
        sae = load()
        fn = (lambda x, sae=sae: sae(x)) if hasattr(sae, 'k') else (lambda x, sae=sae: (lambda r: (r[0], (r[1] > 0).sum(-1)))(sae.m(x)))
        readers(key, fn, sid, dict(family='tsp', cell=cell, layer=L, sae_id=s['sae_id'], arm=s['arm'],
                                   seed=int(s['seed']), fold=None if s['fold'] is None else int(s['fold']),
                                   sha256=s.get('sha256') or sha(s['path'])))
        del sae
        if i % 20 == 0:
            log(f'[tsp] {i + 1}/{len(srcs)} {sid} ({time.time() - t0:.0f} s)')


# ----------------------------------------------------------------------------------------------- statistics
def pct_ci(b):
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyze(d, res_path):
    if res_path.exists():
        sys.exit(f'{res_path} exists; refusing to overwrite.')
    origs = {p.name[6:-4]: dict(np.load(p, allow_pickle=True)) for p in sorted(d.glob('orig__*.npz'))}
    any_o = next(iter(origs.values()))
    y, groups = any_o['y'], any_o['groups'].astype(str)
    W = T.group_bootstrap_counts(groups, N_BOOT, np.random.default_rng(BOOT_SEED))
    ob = {k: {rd: (T.weighted_auroc(y, o[rd]), T.weighted_auroc(y, o[rd], W)) for rd in ('logistic', 'massmean')}
          for k, o in origs.items()}
    gaps, meta = {}, {}
    for p in sorted(d.glob('sae__*.npz')):
        z = dict(np.load(p, allow_pickle=True))
        m = json.loads(str(z['meta']))
        key = f"{m.get('model', 'gpt2')}_L{m['layer']}"
        sid = p.name[5:-4]
        meta[sid] = dict(m, l0_mean=float(z['l0_mean']), nmse_test=float(z['nmse_test']), refit_C=float(z['refit_C']))
        for rd, orig_rd in (('refit_logistic', 'logistic'), ('frozen_logistic', 'logistic'), ('frozen_massmean', 'massmean')):
            pt, bt = ob[key][orig_rd]
            gaps[(sid, rd)] = (pt - T.weighted_auroc(y, z[rd]), bt - T.weighted_auroc(y, z[rd], W))
    res = dict(n_test_statements=int(len(y)), n_test_cities=int(len(set(groups))), boot=N_BOOT,
               original={k: dict(auroc=v['logistic'][0], ci=pct_ci(v['logistic'][1]), C=float(origs[k]['C']))
                         for k, v in ob.items()}, public={}, tsp={})
    for sid, m in meta.items():
        if m['family'] != 'public':
            continue
        cell = {}
        for rd in ('refit_logistic', 'frozen_logistic', 'frozen_massmean'):
            pt, bt = gaps[(sid, rd)]
            lo, hi = pct_ci(bt)
            lab = 'LOSS' if lo > 0 else ('NON-INFERIOR (gap < SESOI)' if hi < T.SESOI else 'INCONCLUSIVE')
            okey = f"{m['model']}_L{m['layer']}"
            if res['original'][okey]['ci'][1] >= CEILING:     # pre-declared: no headroom, label uninformative
                lab = 'CEILING (uninformative; ' + lab + ')'
            cell[rd] = dict(gap=pt, ci=[lo, hi], label=lab)
        res['public'][sid] = dict(meta=m, **cell)
    prim = []
    for cell in TSP_CELLS:
        ms = {sid: m for sid, m in meta.items() if m['family'] == 'tsp' and m['cell'] == cell}
        if not ms:
            continue
        seeds = sorted({m['seed'] for m in ms.values()})

        def per_seed(arm, rd, ref=None):
            P, Bt = [], []
            for s in seeds:
                def avg(a):
                    ids = [sid for sid, m in ms.items() if m['arm'] == a and m['seed'] == s]
                    return (np.mean([gaps[(i, rd)][0] for i in ids]), np.mean([gaps[(i, rd)][1] for i in ids], 0))
                pa, ba = avg(arm)
                if ref:
                    pr, br = avg(ref)
                    pa, ba = pa - pr, ba - br
                P.append(pa)
                Bt.append(ba)
            return T.hierarchical(np.array(P), np.array(Bt).mean(0))

        c = dict(seeds=seeds, n_saes=len(ms),
                 l0_mean_by_arm={a: float(np.mean([m['l0_mean'] for m in ms.values() if m['arm'] == a]))
                                 for a in ('base', 'ctrl', 'tsp', 'rand')})
        for rd in ('refit_logistic', 'frozen_logistic'):
            for a, r in (('tsp', 'ctrl'), ('tsp', 'rand'), ('tsp', 'base'), ('ctrl', 'base')):
                c[f'{a}-{r} {rd}_gap'] = per_seed(a, rd, r)
            for a in ('base', 'ctrl', 'tsp'):
                c[f'{a} absolute {rd}_gap'] = per_seed(a, rd)
        res['tsp'][cell] = c
        prim.append((cell, c['tsp-ctrl refit_logistic_gap']))
    if prim:
        ph = T.holm(np.array([h['p'] for _, h in prim]))
        for (cell, h), p in zip(prim, ph):
            res['tsp'][cell]['primary'] = dict(estimate=h['estimate'], ci=[h['ci_low'], h['ci_high']], p=h['p'],
                                               p_holm=float(p), label=T.h1_label(h),
                                               supported=bool(T.h1_label(h).startswith('SUPPORTED') and p < 0.05))
    res['meta_per_sae'] = meta
    res_path.write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------------------------- stages
def check_hashes():
    pr = json.loads((OUT / 'preregistration.json').read_text())
    for rel in HASHED:
        if pr['hashes'].get(rel) != sha(REPO / rel):
            sys.exit(f'{rel}: hash differs from the pre-registered hash; refusing to run.')
    return pr


def committed(rel):
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
    return tracked and clean


def gate_auroc(X, y, g):
    from sklearn.model_selection import GroupKFold
    S31 = _mod('scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py', 's31g')
    best = (-1.0, None)
    for C in S31.C_GRID:
        sc = np.zeros(len(y))
        for a, b in GroupKFold(3).split(X, y, g):
            sc[b] = S31.lr(C).fit(X[a], y[a]).decision_function(X[b])
        v = roc_auc_score(y, sc)
        if v > best[0] + 1e-12:
            best = (v, C)
    return best


def stage_smoke(out, tsp_limit):
    """GoT cities TRAIN rows only; pseudo-DEV / pseudo-TEST halves by city. Never reads the fresh family."""
    if (out / 'smoke_results.json').exists():
        sys.exit(f'{out} has results; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(REPO / 'data/got/cities.csv')
    sp = pd.read_csv(REPO / 'data/got/splits/cities_splits.csv')
    split_col = 'split' if 'split' in sp.columns else sp.columns[-1]
    tr = c.loc[sp[split_col].astype(str).str.lower().eq('train').to_numpy()].reset_index(drop=True)
    tr = tr[tr.city.map(lambda x: hfrac('smoke|' + norm(x)) < 0.5)].reset_index(drop=True)   # half of TRAIN, quick
    half = tr.city.map(lambda x: hfrac('smoke2|' + norm(x)) < 0.5).to_numpy()
    A = activations(tr.statement.tolist(), out / 'acts.npz')
    y, g = tr.label.to_numpy(), tr.city.map(norm).to_numpy()
    gate = {k: gate_auroc(v[half], y[half], g[half]) for k, v in A.items()}
    elig = {k for k, (auc, _) in gate.items() if auc >= GATE_AUROC}
    score_pass({k: v[half] for k, v in A.items()}, y[half], g[half], {k: v[~half] for k, v in A.items()}, y[~half],
               g[~half], out / 'scores', elig, tsp_limit=tsp_limit)
    res = analyze(out / 'scores', out / 'smoke_results.json')
    print(json.dumps(dict(gate=gate, eligible=sorted(elig), original=res['original'],
                          public={k: v['refit_logistic'] for k, v in res['public'].items()},
                          tsp={k: v.get('primary') for k, v in res['tsp'].items()}), indent=1, default=float))


def load_split(split):
    df = pd.read_csv(DATA / 'statements.csv')
    return df[df.split == split].reset_index(drop=True)


def stage_gate():
    check_hashes()
    p = OUT / 'gate' / 'gate.json'
    if p.exists():
        sys.exit('gate.json exists; refusing to overwrite.')
    p.parent.mkdir(parents=True, exist_ok=True)
    dev = load_split('DEV')
    t0 = time.time()
    A = activations(dev.statement.tolist(), OUT / 'gate' / 'dev_acts.npz')
    y, g = dev.label.to_numpy(), dev.city.map(norm).to_numpy()
    gate = {}
    for k, X in A.items():
        auc, C = gate_auroc(X, y, g)
        gate[k] = dict(auroc_grouped_cv=auc, C=C, eligible=bool(auc >= GATE_AUROC))
        print(k, gate[k], flush=True)
    p.write_text(json.dumps(dict(rule=f'eligible if grouped-CV (3-fold by city) logistic AUROC on DEV original '
                                      f'activations >= {GATE_AUROC}', n_dev=len(dev), cells=gate,
                                 seconds=time.time() - t0, test_rows_used=0), indent=1))


def stage_test():
    pr = check_hashes()
    rel = 'results/cross_model/xf_fresh_cities_test/gate/gate.json'
    if not committed(rel):
        sys.exit('gate.json must be committed before TEST is read.')
    d = OUT / 'test'
    if (d / 'test_info.json').exists():
        sys.exit('test pass already done; refusing to repeat.')
    d.mkdir(parents=True, exist_ok=True)
    gate = json.loads((REPO / rel).read_text())['cells']
    elig = {k for k, v in gate.items() if v['eligible']}
    dev, te = load_split('DEV'), load_split('TEST')
    t0 = time.time()
    Ad = activations(dev.statement.tolist(), OUT / 'gate' / 'dev_acts.npz')
    At = activations(te.statement.tolist(), d / 'test_acts.npz')
    logf = open(d / 'run.log', 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()
    score_pass(Ad, dev.label.to_numpy(), dev.city.map(norm).to_numpy(), At, te.label.to_numpy(),
               te.city.map(norm).to_numpy(), d / 'scores', elig, log=log)
    (d / 'test_info.json').write_text(json.dumps(dict(
        finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.time() - t0, eligible=sorted(elig),
        n_test=len(te), prereg_sha256=sha(OUT / 'preregistration.json'),
        head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        torch=torch.__version__, tf32=False), indent=1))


def stage_analyze():
    check_hashes()
    res = analyze(OUT / 'test' / 'scores', OUT / 'test' / 'results.json')
    print(json.dumps(dict(original=res['original'],
                          public={k: {rd: v[rd] for rd in ('refit_logistic', 'frozen_logistic')}
                                  for k, v in res['public'].items()},
                          tsp={k: v.get('primary') for k, v in res['tsp'].items()}), indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'smoke', 'gate', 'test', 'analyze'])
    ap.add_argument('--smoke-out', type=Path, default=OUT / 'smoke')
    ap.add_argument('--tsp-limit', type=int, default=2)
    a = ap.parse_args()
    if a.stage == 'build':
        stage_build()
    elif a.stage == 'smoke':
        stage_smoke(a.smoke_out, a.tsp_limit)
    else:
        {'gate': stage_gate, 'test': stage_test, 'analyze': stage_analyze}[a.stage]()


if __name__ == '__main__':
    main()
