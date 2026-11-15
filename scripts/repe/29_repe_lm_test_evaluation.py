#!/usr/bin/env python3
"""H1-TEST: the single pre-registered RepE-LM held-out evaluation (protocol frozen 2026-11-13).

Protocol: results/repe/repe_lm_test_protocol/{README.md,preregistration.json} (hypotheses HT1-HT4).

Every metric is computed by imported, UNCHANGED project code, exactly as in the validation ladder
(scripts/shared/sae_ladder_evaluation.py):
  * HT2 representation (pair_dir_cosine, ...): scripts/repe/26_*.py `original_side` / `evaluate`;
  * HT3 information (repe_persona refit gap): scripts/shared/m2_four_level_faithfulness.py `original_side` /
    `evaluate_source`;
  * HT1 causal (T_rec/T_full): scripts/repe/24_*.py `Engine` / `clean_batches` / `decompose` / `add_derived` /
    `summarise_decomp`, seeded as in the ladder;
  * comparisons: scripts/shared/r2_comparisons.py `rule` (the R2 rule);
  * SAE sources and loading: the ladder's `sources` / `load`.
New code is limited to: building the evaluation row sets from the TEST prompt file, extracting final-token
activations for those prompts, HT4 (original-activation probe transfer), Holm, and orchestration.

Fitting: every probe, direction and regularisation constant is fitted on RepE-LM TRAIN rows only
(data/repe/repe_lm/prompts.csv split == train; cached activations) and applied frozen to the evaluation rows.
The imported functions select evaluation rows by the in-memory split name 'validation'; here that name denotes the
EVALUATION rows (TEST rows in --run, VALIDATION rows in --smoke). No TEST row is ever used for fitting.

Modes (run from the repository root):
    python scripts/repe/29_repe_lm_test_evaluation.py --smoke OUTDIR   # VALIDATION rows only; checks reproduction
    python scripts/repe/29_repe_lm_test_evaluation.py --run            # TEST, once; hash-guarded
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

log = logging.getLogger('h1_test')


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAD = _load('scripts/shared/sae_ladder_evaluation.py', 'ladder')
M2, S24, S26 = LAD.M2, LAD.S24, LAD.S26
R2C = _load('scripts/shared/r2_comparisons.py', 'r2c')

TRAIN_PROMPTS = REPO / 'data/repe/repe_lm/prompts.csv'
TEST_PROMPTS = REPO / 'data/repe/repe_lm_test/prompts.csv'
CACHE = REPO / 'data/repe/activations/repe_lm'
PROTO = REPO / 'results/repe/repe_lm_test_protocol'
OUT = REPO / 'results/repe/repe_lm_test_evaluation'
LADDER = REPO / 'results/openwebtext/r2_scaled_saes/ladder'
VAL_COMP = REPO / 'results/openwebtext/r2_scaled_saes/comparisons.csv'
LAYERS = (1, 8, 11)
SITE = {1: 'slot', 8: 'final', 11: 'final'}
SOURCES = 'old,r2,r4,resjb'
PERSONA, HELDOUT_M1, HELDOUT_TEST = ['T1', 'T2', 'T3', 'T4'], ['T5', 'T6'], ['T7', 'T8']
NEUTRAL_M1, NEUTRAL_TEST = ['N1', 'N2'], ['N3', 'N4']
SEED = LAD.SEED
N_BOOT = 1000
FLOOR = 0.55
# confirmatory cells: (hypothesis, metric, comparison, layer)
HT = ([('HT1', 'm4__preservation_rec', c, 8) for c in ('C1', 'C3')]
      + [('HT2', 'rlm__pair_dir_cosine', c, L) for c in ('C1', 'C2', 'C3') for L in LAYERS]
      + [('HT3', 'repe_persona__refit_logistic_gap', c, 8) for c in ('C1', 'C3')])
RLM_COLS = ('pair_dir_cosine', 'pair_norm_retention', 'pair_error_share', 'massmean_dir_cosine', 'frozen_logistic_gap',
            'refit_logistic_gap', 'frozen_heldout_gap', 'refit_heldout_gap', 'frozen_syn_gap', 'frozen_neutral_gap',
            'refit_neutral_gap', 'task_l0', 'task_nmse', 'repe_val_ev', 'repe_val_l0', 'orig_persona_logistic_auroc',
            'orig_heldout_auroc', 'orig_neutral_auroc')
M2_COLS = ('task_nmse', 'task_l0', 'frozen_logistic_gap', 'frozen_logistic_gap_ci_low', 'frozen_logistic_gap_ci_high',
           'noise_frozen_logistic_gap', 'refit_logistic_gap', 'refit_logistic_gap_ci_low', 'refit_logistic_gap_ci_high',
           'error_term_auroc')
M4_COLS = ('T_full_mean', 'T_full_ci_low', 'T_rec_mean', 'T_err_mean', 'preservation_rec', 'preservation_rec_ci_low',
           'preservation_rec_ci_high', 'preservation_err', 'shapley_share_rec', 'abs_I_over_T_full', 'kl_base_mean',
           'kl_syn_mean', 'rec_minus_syn_rec_mean', 'iso_rec_q95', 'shuf_rec_q95', 'T_rec_above_controls_q95',
           'T_rec_minus_iso_rec_p', 'T_rec_minus_shuf_rec_p')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data

def frame(mode):
    """Combined frame: RepE-LM TRAIN rows (file order) + evaluation rows (file order), with a cache index."""
    d = pd.read_csv(TRAIN_PROMPTS)
    assert set(d.split) <= {'train', 'validation'}
    d['cache_row'] = np.arange(len(d))
    if mode == 'smoke':                         # validation rows, original file order (train and validation interleaved)
        return d.reset_index(drop=True)
    t = pd.read_csv(TEST_PROMPTS)
    assert set(t.split) == {'test'}
    assert not set(t.statement_id) & set(d.statement_id), 'statement crosses train/test'
    assert not set(t.norm_group) & set(d.norm_group), 'norm group crosses train/test'
    t['cache_row'] = -1
    return pd.concat([d[d.split == 'train'], t], ignore_index=True)


def eval_split(mode):
    return 'validation' if mode == 'smoke' else 'test'


def row_index(D, mode):
    """The index sets of S26.load_rows, with the evaluation split in place of validation, plus HT4 sets."""
    tr, va = (D.split == 'train').to_numpy(), (D.split == eval_split(mode)).to_numpy()
    persona = D.template.isin(PERSONA).to_numpy()
    hu = D.variant.isin(['H', 'U']).to_numpy()
    f = np.flatnonzero
    idx = dict(p_tr=f(persona & hu & tr), p_va=f(persona & hu & va),
               syn_va=f(persona & D.variant.isin(['Hp', 'U']).to_numpy() & va),
               hhp_va=f(persona & D.variant.isin(['H', 'Hp']).to_numpy() & va),
               ho_va=f(D.template.isin(HELDOUT_M1).to_numpy() & va),
               n_tr=f(D.template.isin(NEUTRAL_M1).to_numpy() & tr), n_va=f(D.template.isin(NEUTRAL_M1).to_numpy() & va),
               all_va=f(va),
               ho_test_only=f(D.template.isin(HELDOUT_TEST).to_numpy() & va),
               n_test_only=f(D.template.isin(NEUTRAL_TEST).to_numpy() & va))
    pv = D[persona & va & hu]
    piv = pv.reset_index().pivot(index='pair_id', columns='variant', values='index')
    idx['pair_H'], idx['pair_U'] = piv['H'].to_numpy(), piv['U'].to_numpy()
    return idx


def build_rows(D, mode, templates, variants):
    """S24.build_rows on the combined frame's evaluation rows (slot positions from the prompt file)."""
    d = D[(D.split == eval_split(mode)) & D.template.isin(templates)]
    rows = []
    for (sid, t), g in d.groupby(['statement_id', 'template']):
        v = g.set_index('variant')
        ids = {k: list(map(int, v.loc[k, 'token_ids'].split())) for k in variants}
        n = len(ids[variants[0]])
        assert all(len(x) == n for x in ids.values())
        slot = v.slot_pos.unique()
        assert len(slot) == 1
        rows.append(dict(statement_id=int(sid), template=t, ids=ids, n=n, final=n - 1, slot=int(slot[0])))
    rows.sort(key=lambda r: (r['template'], r['n'], r['statement_id']))
    return rows


@torch.inference_mode()
def extract_final(model, D, rows_mask, layers, device, batch=64):
    """blocks.L.hook_resid_pre at each prompt's final token, original GPT-2 (as scripts/repe/21_*.py)."""
    sub = D[rows_mask]
    out = {L: np.zeros((len(sub), 768), np.float32) for L in layers}
    names = {f'blocks.{L}.hook_resid_pre' for L in layers}
    seqs = [list(map(int, s.split())) for s in sub.token_ids]
    fpos = sub.final_pos.to_numpy()
    for b0 in range(0, len(seqs), batch):
        sq = seqs[b0:b0 + batch]
        T = max(map(len, sq))
        toks = torch.full((len(sq), T), 50256, dtype=torch.long)
        for i, s in enumerate(sq):
            toks[i, :len(s)] = torch.tensor(s)
        _, cache = model.run_with_cache(toks.to(device), names_filter=lambda n: n in names)
        ar = torch.arange(len(sq), device=device)
        fp = torch.tensor(fpos[b0:b0 + batch], device=device)
        for L in layers:
            out[L][b0:b0 + len(sq)] = cache[f'blocks.{L}.hook_resid_pre'][ar, fp].float().cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------ HT4

def ht4(X, D, idx, y, g, o, rng):
    """Original-activation persona probe (fit on TRAIN T1-T4) transferred to held-out wordings and neutral swaps."""
    out = {}
    for name, rows in (('persona_T1_T4', idx['p_va']), ('heldout_T5_T6', idx['ho_va']),
                       ('test_only_T7_T8', idx['ho_test_only']), ('neutral_N1_N2', idx['n_va']),
                       ('neutral_N3_N4', idx['n_test_only'])):
        if not len(rows):
            continue
        s = o['pipe'].decision_function(X[rows])
        bi = il.bootstrap_indices(g[rows], N_BOOT, rng)
        lo, hi = il.ci(il.boot_auroc(y[rows], s, bi))
        out[name] = dict(n=int(len(rows)), auroc=il.auroc(y[rows], s), ci_low=lo, ci_high=hi)
    return out


# ------------------------------------------------------------------------------------------------ main loop

def evaluate_layer(L, mode, model, D, idx, srcs, device, partial):
    y, g = D.label.to_numpy(), D.norm_group.to_numpy()
    sid = D.statement_id.to_numpy()
    # activations: TRAIN rows from the M1 cache; evaluation rows extracted now
    cache = np.load(CACHE / f'final_layer{L}.npy').astype(np.float32)
    X = np.zeros((len(D), 768), np.float32)
    trm = (D.split == 'train').to_numpy()
    X[trm] = cache[D.cache_row.to_numpy()[trm]]
    vam = (D.split == eval_split(mode)).to_numpy()
    X[vam] = extract_final(model, D, vam, [L], device)[L]
    checks = {}
    if mode == 'smoke':
        checks['live_vs_cache_max_abs'] = float(np.abs(X[vam] - cache[D.cache_row.to_numpy()[vam]]).max())
    # RLM-RP state (S26), exactly as the ladder sets it
    S26.G.clear()
    S26.G.update(idx=idx, y=y, y_hhp=(D.variant == 'H').astype(int).to_numpy(), g=g, sid=sid, neardup=set(), X=X)
    S26.G['orig'] = S26.original_side(X)
    # M2 repe_persona task (persona T1-T4 H/U rows); evaluation rows carry the in-memory name 'validation'
    pm = (D.template.isin(PERSONA) & D.variant.isin(['H', 'U'])).to_numpy() & (trm | vam)
    pidx = np.flatnonzero(pm)
    task = dict(X={L: X[pidx]}, y=y[pidx], groups=g[pidx],
                split=np.where(D.split.to_numpy()[pidx] == 'train', 'train', 'validation'))
    orig2 = M2.original_side(task, L, np.random.default_rng(M2.SEED + L))
    bidx = il.bootstrap_indices(task['groups'][task['split'] == 'validation'], M2.N_BOOT,
                                np.random.default_rng(M2.SEED + 100 + L))
    # HT4 (original activations only)
    h4 = ht4(X, D, idx, y, g, S26.G['orig'], np.random.default_rng(SEED + L))
    # M4 site
    site = SITE[L]
    from transformers import AutoTokenizer  # noqa: F401  (rows come from the prompt file's token ids)
    rows = build_rows(D, mode, PERSONA, ['H', 'U', 'Hp'])
    if mode == 'smoke':
        tok = AutoTokenizer.from_pretrained('gpt2')
        ref = S24.build_rows(tok, 'validation', S24.PERSONA, ['H', 'U', 'Hp'])
        checks['m4_rows_identical_to_S24'] = bool(ref == rows)
    eng = S24.Engine(model, L)
    vb = S24.clean_batches(eng, rows, ['H', 'U', 'Hp'], site, device)
    out = []
    for i, src in enumerate(srcs):
        sae = LAD.load(src, L, device)
        r = dict(layer=L, family=src['family'], name=src['name'], seed=src['seed'],
                 checkpoint=str(src['path']).replace(str(Path.home()), '~').replace(str(REPO) + '/', ''),
                 checkpoint_sha256=sha(src['path']), split=eval_split(mode).upper())
        # HT3: M2 evaluate_source on the persona task
        H, Z = LAD.encode_np(sae, task['X'][L], device)
        rs, _ = M2.evaluate_source(dict(), task['X'][L], H, Z, task['y'], task['split'], task['groups'], orig2, bidx,
                                   SEED + 1000 * L + i)
        for c in M2_COLS:
            if c in rs:
                r[f'repe_persona__{c}'] = rs[c]
        # HT2: RLM-RP evaluate
        Hr, Zr = LAD.encode_np(sae, X, device)
        l0r = (Zr > 0).sum(1)
        va_act = (Zr[idx['all_va']] > 0).any(0)
        del Zr
        rr = S26.evaluate({}, Hr, None, l0r, float((~va_act).mean()))
        for c in RLM_COLS:
            r[f'rlm__{c}'] = rr.get(c, np.nan)
        # HT1: M4 decomposition, seeded as in the ladder
        torch.manual_seed(SEED)
        S24.GEN_CPU.manual_seed(SEED)
        gen = torch.Generator(device=device).manual_seed(SEED)
        rng4 = np.random.default_rng(SEED)
        dfp, dr, ident = S24.decompose(eng, sae, vb, 'persona', gen)
        dfp = S24.add_derived(dfp)
        s = [x for x in S24.summarise_decomp(dfp, dr, dict(kind='persona'), rng4) if x['template_set'] == 'T2_T4'][0]
        for c in M4_COLS:
            r[f'm4__{c}'] = s.get(c, np.nan)
        r['m4__identity_err'] = ident
        r['m4__site'] = f'blocks.{L}.hook_resid_pre|{site}'
        out.append(r)
        pd.DataFrame(out).to_csv(partial, index=False)
        log.info('L%d %s %s %s: cos %.4f refit %.4f pres %.4f', L, src['family'], src['name'], src['seed'],
                 r['rlm__pair_dir_cosine'], r['repe_persona__refit_logistic_gap'], r['m4__preservation_rec'])
        del sae
        torch.cuda.empty_cache()
    return out, h4, checks


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[ix]))
        adj[ix] = run
    return adj


def welch_p(new, ref):
    from scipy import stats
    return float(stats.ttest_ind(np.asarray(new, float), np.asarray(ref, float), equal_var=False).pvalue)


def comparisons(df):
    rows = []
    for L in LAYERS:
        d = df[df.layer == L]
        for cid, new, ref, desc in R2C.COMPARISONS:
            nv = d[(d.family == new[0]) & (d.name == new[1])].sort_values('seed')
            rv = d[(d.family == ref[0]) & (d.name == ref[1])].sort_values('seed')
            for m, hib, lev in R2C.METRICS:
                if m not in d or not len(nv) or not len(rv) or nv[m].isna().all():
                    continue
                res = R2C.rule(nv[m], rv[m], hib)
                rows.append(dict(layer=L, comparison=cid, metric=m, level=lev, higher_is_better=hib,
                                 welch_p=welch_p(nv[m], rv[m]), **res))
    return pd.DataFrame(rows)


def confirmatory(comp, val):
    rows = []
    for h, m, c, L in HT:
        t = comp[(comp.metric == m) & (comp.comparison == c) & (comp.layer == L)].iloc[0]
        v = val[(val.metric == m) & (val.comparison == c) & (val.layer == L)].iloc[0]
        rows.append(dict(hypothesis=h, metric=m, comparison=c, layer=L, val_ref=v.ref_mean, val_new=v.new_mean,
                         val_diff=v['diff'], val_cls=v.cls, test_ref=t.ref_mean, test_new=t.new_mean, test_diff=t['diff'],
                         test_ci95_low=t.ci95_low, test_ci95_high=t.ci95_high, test_pairs_better=t.pairs_better,
                         test_welch_p=t.welch_p, test_cls_unadjusted=t.cls))
    c = pd.DataFrame(rows)
    c['test_welch_p_holm'] = holm(c.test_welch_p)
    c['confirmed_cell'] = (c.test_cls_unadjusted == 'IMPROVEMENT') & (c.test_welch_p_holm < 0.05)
    return c


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument('--smoke', type=Path, help='VALIDATION reproduction check; output directory (outside results/)')
    gx.add_argument('--run', action='store_true', help='the single TEST pass')
    a = ap.parse_args()
    mode = 'smoke' if a.smoke else 'test'
    out = a.smoke if a.smoke else OUT
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite.')
    if mode == 'test':
        pr = json.loads((PROTO / 'preregistration.json').read_text())
        ad = PROTO / 'preregistration_addendum.json'      # frozen before any TEST row is read; adds this script's hash
        if not ad.exists():
            raise SystemExit('pre-registration addendum missing; refusing to read TEST rows.')
        hashes = dict(pr['hashes'], **json.loads(ad.read_text())['hashes'])
        if 'scripts/repe/29_repe_lm_test_evaluation.py' not in hashes:
            raise SystemExit('script 29 is not hashed in the pre-registration; refusing to read TEST rows.')
        for rel, h in hashes.items():
            if sha(REPO / rel) != h:
                raise SystemExit(f'{rel}: hash differs from the pre-registration; refusing to run.')
    t0 = time.time()
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    D = frame(mode)
    idx = row_index(D, mode)
    out.mkdir(parents=True)
    partial = out / 'per_sae.partial.csv'
    rows, h4all, checks = [], {}, {}
    for L in LAYERS:
        srcs = LAD.sources(L, SOURCES.split(','))
        if mode == 'smoke':   # two SAEs per layer: enough to check reproduction of the ladder
            srcs = [s for s in srcs if (s['family'], s['name'], s['seed']) in {('old', 'topk', 42), ('r2', 'topk', 42)}]
        r, h4, ck = evaluate_layer(L, mode, model, D, idx, srcs, device, partial)
        rows += r
        h4all[f'L{L}'] = h4
        checks[f'L{L}'] = ck
    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_sae.csv', index=False)
    partial.unlink()
    (out / 'ht4_probe_transfer.json').write_text(json.dumps(h4all, indent=2))
    info = dict(mode=mode, script_sha256=sha(__file__), n_saes=len(df), runtime_s=time.time() - t0,
                eval_rows=int((D.split == eval_split(mode)).sum()), test_rows_used=int((D.split == 'test').sum()),
                fitted_on='RepE-LM TRAIN only', checks=checks)
    if mode == 'smoke':
        lad = pd.concat([pd.read_csv(p) for p in sorted(LADDER.glob('ladder_layer*.csv'))], ignore_index=True)
        rep = []
        for _, r in df.iterrows():
            l_ = lad[(lad.layer == r.layer) & (lad.family == r.family) & (lad.name == r['name']) & (lad.seed == r.seed)].iloc[0]
            for m in ('rlm__pair_dir_cosine', 'rlm__frozen_heldout_gap', 'rlm__refit_heldout_gap',
                      'repe_persona__refit_logistic_gap', 'repe_persona__frozen_logistic_gap', 'm4__preservation_rec',
                      'm4__T_full_mean'):
                rep.append(dict(layer=r.layer, family=r.family, name=r['name'], seed=r.seed, metric=m,
                                smoke=r[m], ladder=l_[m], abs_diff=abs(r[m] - l_[m])))
        rep = pd.DataFrame(rep)
        rep.to_csv(out / 'reproduction_vs_ladder.csv', index=False)
        info['max_abs_diff_vs_ladder'] = rep.groupby('metric').abs_diff.max().to_dict()
    else:
        comp = comparisons(df)
        comp.to_csv(out / 'comparisons.csv', index=False)
        conf = confirmatory(comp, pd.read_csv(VAL_COMP))
        conf.to_csv(out / 'confirmatory_cells.csv', index=False)
        info['hypotheses'] = {h: bool(conf[conf.hypothesis == h].confirmed_cell.all()) for h in ('HT1', 'HT2', 'HT3')}
    (out / 'checks.json').write_text(json.dumps(info, indent=2, default=float))
    log.info('done in %.0f s: %s', time.time() - t0, json.dumps(info, default=float)[:2000])


if __name__ == '__main__':
    main()
