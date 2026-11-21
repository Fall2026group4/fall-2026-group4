#!/usr/bin/env python3
"""AUDIT-S1 (sensitivity, NOT confirmatory): add statement / document sampling uncertainty to the seed-only R2 rule.

Problem (audit 2026-11-13): the R2 rule (scripts/shared/r2_comparisons.py:rule) is a Welch test over 3 vs 3 seed
means, all computed on one fixed evaluation set. Its CI covers "retrain with another seed", not "evaluate on other
statements / documents".

What this script does (VALIDATION rows only; no TEST row is read):
  1. For the C1-C3 arms at layers 1, 8, 11 (old TopK/ReLU/JumpReLU-3072, R2 TopK/ReLU/BatchTopK-12288; 3 seeds each)
     it recomputes, per statement (RepE-LM) or per document (OWT), the quantities behind
       HT1  m4__preservation_rec      (L8 only; ratio of statement means of T_rec and T_full, T2-T4; clusters = statement)
       HT2  rlm__pair_dir_cosine      (clusters = norm_group = statement)
       HT3  repe_persona__refit_logistic_gap (clusters = norm_group)
       M5   owt_nmse, splice_delta_ce (clusters = OWT validation document)
     with the unchanged project functions, and checks each per-SAE value reproduces the committed ladder.
  2. Paired cluster bootstrap (B = 2000, same resamples for all SAEs of a layer): D_b = mean_seeds(new) -
     mean_seeds(ref); v_stmt = var(D_b) is the cluster-sampling variance of the comparison.
  3. Combined uncertainty (pre-registered in results/shared/audit_sensitivity/preregistration.json):
       se = sqrt(se_seed^2 + v_stmt),  se_seed^2 = s_new^2/3 + s_ref^2/3,
       df = min(Satterthwaite(se_seed^2 with Welch df, v_stmt with n_clusters - 1), Welch df)  [conservative],
       CI = D +- t_{0.975, df} se, p = two-sided t.
     Applied (a) on VALIDATION directly and (b) to the committed TEST seed statistics (H1-TEST and M5), with
     v_stmt scaled by n_clusters_val / n_clusters_test (a transfer assumption, stated as such).
  4. Holm over the 13 H1 cells and, separately, over the 18 M5 cells.

Outputs: results/shared/audit_sensitivity/s1_statement_uncertainty/ (refuses to overwrite).
Run from the repository root:  python scripts/shared/audit_s1_statement_uncertainty.py [--smoke]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import information_levels as il  # noqa: E402

spec = importlib.util.spec_from_file_location('ladder', REPO / 'scripts/shared/sae_ladder_evaluation.py')
LAD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LAD)
M2, SP, S24, S26, O3 = LAD.M2, LAD.SP, LAD.S24, LAD.S26, LAD.O3

OUT = REPO / 'results/shared/audit_sensitivity/s1_statement_uncertainty'
LADDER = REPO / 'results/openwebtext/r2_scaled_saes/ladder'
H1_TEST = REPO / 'results/repe/repe_lm_test_evaluation/confirmatory_cells.csv'
M5_TEST = REPO / 'results/shared/m5_confirmatory_test/comparisons.csv'
LAYERS = (1, 8, 11)
ARMS = {('old', 'topk'), ('old', 'relu'), ('old', 'jumprelu'), ('r2', 'topk'), ('r2', 'relu'), ('r2', 'batchtopk')}
COMP = {'C1': (('r2', 'topk'), ('old', 'topk')), 'C2': (('r2', 'relu'), ('old', 'relu')),
        'C3': (('r2', 'batchtopk'), ('old', 'jumprelu'))}
B, BOOT_SEED = 2000, 20261113
N_TEST_STATEMENTS, N_TEST_DOCS = 86, 100


def owt_doc_counts():
    """Non-BOS token count per OWT validation document, in O3.load_split order."""
    counts = []
    for p in sorted((REPO / 'data/activations/gpt2_small/validation').glob('chunk_*.pt')):
        ch = torch.load(p, map_location='cpu', weights_only=False)
        counts += [int(n) - 1 for n in ch['lengths'] if int(n) > 1]
    return np.array(counts)


@torch.inference_mode()
def per_doc_ce(sp, seqs, layer=None, fn=None, batch=8):
    """Per-document summed next-token CE over the owt_eval prediction mask, and the number of predictions."""
    ce, n = [], []
    for b in range(0, len(seqs), batch):
        toks, lens = SP.pad(seqs[b:b + batch], 50256)
        lp = sp.logprobs(toks, lens, layer, fn, 'all')
        T = toks.shape[1]
        ar = torch.arange(T)[None]
        m = ((ar >= 1) & (ar <= lens[:, None] - 2)).to(sp.device)[:, :-1]
        tgt = toks[:, 1:].to(sp.device)
        nll = -lp[:, :-1].gather(-1, tgt[..., None])[..., 0]
        ce += (nll * m).sum(1).double().cpu().tolist()
        n += m.sum(1).cpu().tolist()
        del lp
    return np.array(ce), np.array(n)


def cluster_index(groups):
    u, inv = np.unique(groups, return_inverse=True)
    return u, inv


def boot_weights(n_clusters, rng):
    """Multinomial cluster counts: [B, n_clusters] (resampling clusters with replacement)."""
    return np.stack([np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters) for _ in range(B)])


def combine(d, se_seed2, df_seed, v_stmt, n_clusters):
    se = np.sqrt(se_seed2 + v_stmt)
    num = (se_seed2 + v_stmt) ** 2
    den = (se_seed2 ** 2 / df_seed if df_seed > 0 else 0) + v_stmt ** 2 / (n_clusters - 1)
    df_satt = num / den if den > 0 else np.inf
    # Conservative primary df: never above the seed-only Welch df, so the combined CI is never narrower than the
    # frozen seed-only CI (Satterthwaite alone can raise df when the large-df cluster term is added).
    df = min(df_satt, df_seed) if df_seed > 0 else df_satt
    h = stats.t.ppf(0.975, df) * se
    p = 2 * stats.t.sf(abs(d) / se, df) if se > 0 else 0.0
    p_satt = 2 * stats.t.sf(abs(d) / se, df_satt) if se > 0 else 0.0
    return dict(se_total=se, df_total=df, df_satterthwaite=df_satt, ci_low=d - h, ci_high=d + h, p=p,
                p_satterthwaite=p_satt)


def welch_parts(new, ref):
    new, ref = np.asarray(new, float), np.asarray(ref, float)
    sn, sr = new.var(ddof=1) / len(new), ref.var(ddof=1) / len(ref)
    df = (sn + sr) ** 2 / (sn ** 2 / (len(new) - 1) + sr ** 2 / (len(ref) - 1)) if sn + sr > 0 else np.inf
    return sn + sr, df


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[ix]))
        adj[ix] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='layer 8, seed 42 arms only, into the scratch dir given by --out')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.mkdir(parents=True)
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained('gpt2')
    lad = pd.concat([pd.read_csv(p) for p in sorted(LADDER.glob('ladder_layer*.csv'))], ignore_index=True)
    sp = SP.Splicer(model, device)
    seqs, _ = SP.load_owt_validation()
    ce_o, n_o = per_doc_ce(sp, seqs)
    doc_counts = owt_doc_counts()
    tasks = M2.load_tasks()
    d, idx, y, y_hhp, g, sid, neardup, _ = S26.load_rows()
    S26.G.update(idx=idx, y=y, y_hhp=y_hhp, g=g, sid=sid, neardup=neardup)
    per_sae, checks, boots = [], [], {}
    layers = (8,) if a.smoke else LAYERS
    for L in layers:
        rng_o = np.random.default_rng(M2.SEED + L)
        origs = {tn: M2.original_side(t, L, rng_o) for tn, t in tasks.items()}
        tp = tasks['repe_persona']
        tr, va = tp['split'] == 'train', tp['split'] == 'validation'
        yv, gv = tp['y'][va], tp['groups'][va]
        o_auc = origs['repe_persona']['auroc']['logistic']
        o_sc = origs['repe_persona']['scores']['logistic']
        Xr = np.load(S26.CACHE / f'final_layer{L}.npy').astype(np.float32)
        x_owt = O3.load_split(REPO / 'data/activations/gpt2_small', 'validation', L).numpy()
        doc_of_tok = np.repeat(np.arange(len(doc_counts)), doc_counts)
        assert len(doc_of_tok) == len(x_owt)
        xm = x_owt.mean(0)
        var_doc = np.bincount(doc_of_tok, ((x_owt - xm) ** 2).sum(1))
        pair_groups = g[idx['pair_H']]
        if L == 8:
            rows4 = S24.build_rows(tok, 'validation', S24.PERSONA, ['H', 'U', 'Hp'])
            eng = S24.Engine(model, L)
            vb = S24.clean_batches(eng, rows4, ['H', 'U', 'Hp'], 'final', device)
        srcs = [s for s in LAD.sources(L, ['old', 'r2']) if (s['family'], s['name']) in ARMS
                and (not a.smoke or s['seed'] == 42)]
        layer_vals = {}
        for i, src in enumerate(srcs):
            sae = LAD.load(src, L, device)
            key = (src['family'], src['name'], src['seed'])
            ref = lad[(lad.layer == L) & (lad.family == src['family']) & (lad.name == src['name']) & (lad.seed == src['seed'])].iloc[0]
            v = {}
            # OWT NMSE per document
            H, _ = LAD.encode_np(sae, x_owt, device)
            sse_doc = np.bincount(doc_of_tok, ((x_owt - H) ** 2).sum(1))
            v['owt_nmse'] = (sse_doc, var_doc)
            # splice delta CE per document
            ce_s, n_s = per_doc_ce(sp, seqs, L, SP.sae_fn(sae))
            assert (n_s == n_o).all()
            v['splice_delta_ce'] = (ce_s - ce_o, n_o.astype(float))
            # HT2 pair cosine per pair
            Hr, _ = LAD.encode_np(sae, Xr, device)
            D = (Xr[idx['pair_H']] - Xr[idx['pair_U']]).astype(np.float64)
            Dh = (Hr[idx['pair_H']] - Hr[idx['pair_U']]).astype(np.float64)
            nd, nh = np.linalg.norm(D, axis=1), np.linalg.norm(Dh, axis=1)
            ok = nd > 1e-4
            cos = np.where(ok, np.sum(D * Dh, 1) / (nd * nh + 1e-12), np.nan)
            v['rlm__pair_dir_cosine'] = cos
            # HT3 refit persona gap: scores per row
            Hp_, _ = LAD.encode_np(sae, tp['X'][L], device)
            pipe, _ = il.fit_logistic(Hp_[tr], tp['y'][tr], tp['groups'][tr])
            v['repe_persona__refit_logistic_gap'] = pipe.decision_function(Hp_[va])
            # HT1 M4 per pair (L8)
            if L == 8:
                torch.manual_seed(LAD.SEED)
                S24.GEN_CPU.manual_seed(LAD.SEED)
                gen = torch.Generator(device=device).manual_seed(LAD.SEED)
                dfp, _, _ = S24.decompose(eng, sae, vb, 'persona', gen)
                dfp = dfp[dfp.template.isin(S24.PRIMARY)]
                v['m4__preservation_rec'] = (dfp.T_rec.to_numpy(), dfp.T_full.to_numpy(), dfp.statement_id.to_numpy())
            layer_vals[key] = v
            # reproduction checks (point estimates)
            pt = dict(owt_nmse=sse_doc.sum() / var_doc.sum(),
                      splice_delta_ce=(ce_s - ce_o).sum() / n_o.sum(),
                      rlm__pair_dir_cosine=float(np.nanmean(cos)),
                      repe_persona__refit_logistic_gap=o_auc - il.auroc(yv, v['repe_persona__refit_logistic_gap']))
            if L == 8:
                Tr, Tf, s_ = v['m4__preservation_rec']
                ss = pd.DataFrame(dict(r=Tr, f=Tf, s=s_)).groupby('s').mean()
                pt['m4__preservation_rec'] = ss.r.mean() / ss.f.mean()
            for m, val in pt.items():
                checks.append(dict(layer=L, family=key[0], name=key[1], seed=key[2], metric=m, recomputed=val,
                                   ladder=float(ref[m]), abs_diff=abs(val - float(ref[m]))))
                per_sae.append(dict(layer=L, family=key[0], name=key[1], seed=key[2], metric=m, value=val))
            print(f'L{L} {key}: {json.dumps({k: round(x, 5) for k, x in pt.items()})} ({time.time() - t0:.0f}s)', flush=True)
            del sae
            torch.cuda.empty_cache()
        # ---------------- paired cluster bootstrap per metric
        rng = np.random.default_rng(BOOT_SEED + L)
        u_doc, inv_doc = cluster_index(np.arange(len(doc_counts)))
        Wd = boot_weights(len(u_doc), rng)
        u_seq = np.arange(len(seqs))
        Ws = boot_weights(len(u_seq), rng)
        u_pg, inv_pg = cluster_index(pair_groups)
        Wp = boot_weights(len(u_pg), rng)
        u_g, inv_g = cluster_index(gv)
        Wg = boot_weights(len(u_g), rng)
        mb = [np.flatnonzero(inv_g == c) for c in range(len(u_g))]
        if L == 8:
            sid4 = next(iter(layer_vals.values()))['m4__preservation_rec'][2]
            u_s4, inv_s4 = cluster_index(sid4)
            W4 = boot_weights(len(u_s4), rng)
        # AUROC resample index lists (one per bootstrap draw, shared by all SAEs)
        idx_g = [np.concatenate([np.repeat(mb[c], k) for c, k in enumerate(w) if k]) for w in Wg]
        o_boot = np.array([il.auroc(yv[ii], o_sc[ii]) for ii in idx_g])
        for key, v in layer_vals.items():
            bt = {}
            sse_doc, var_doc_ = v['owt_nmse']
            bt['owt_nmse'] = (Wd @ sse_doc) / (Wd @ var_doc_)
            dce, nn = v['splice_delta_ce']
            bt['splice_delta_ce'] = (Ws @ dce) / (Ws @ nn)
            cos = v['rlm__pair_dir_cosine']
            okm = np.isfinite(cos)
            s_c = np.bincount(inv_pg[okm], cos[okm], minlength=len(u_pg))
            n_c = np.bincount(inv_pg[okm], minlength=len(u_pg))
            bt['rlm__pair_dir_cosine'] = (Wp @ s_c) / (Wp @ n_c)
            sc = v['repe_persona__refit_logistic_gap']
            bt['repe_persona__refit_logistic_gap'] = o_boot - np.array([il.auroc(yv[ii], sc[ii]) for ii in idx_g])
            if L == 8:
                Tr, Tf, _ = v['m4__preservation_rec']
                nr = np.bincount(inv_s4, Tr) / np.bincount(inv_s4)
                nf = np.bincount(inv_s4, Tf) / np.bincount(inv_s4)
                bt['m4__preservation_rec'] = (W4 @ nr) / (W4 @ nf)
            boots[(L,) + key] = bt
        ncl = dict(owt_nmse=len(u_doc), splice_delta_ce=len(u_seq), rlm__pair_dir_cosine=len(u_pg),
                   repe_persona__refit_logistic_gap=len(u_g))
        if L == 8:
            ncl['m4__preservation_rec'] = len(u_s4)
        boots[('ncl', L)] = ncl
    ck = pd.DataFrame(checks)
    ck.to_csv(a.out / 'reproduction_checks.csv', index=False)
    pd.DataFrame(per_sae).to_csv(a.out / 'per_sae_validation.csv', index=False)
    # ---------------- comparisons
    hib = {'owt_nmse': False, 'splice_delta_ce': False, 'rlm__pair_dir_cosine': True,
           'repe_persona__refit_logistic_gap': False, 'm4__preservation_rec': True}
    val_rows, vstmt = [], {}
    ps = pd.DataFrame(per_sae)
    for L in layers:
        for cid, (new, ref) in COMP.items():
            for m in hib:
                if m == 'm4__preservation_rec' and L != 8:
                    continue
                kn = [k for k in boots if k[0] == L and len(k) == 4 and (k[1], k[2]) == new]
                kr = [k for k in boots if k[0] == L and len(k) == 4 and (k[1], k[2]) == ref]
                if not kn or not kr:
                    continue
                Db = np.mean([boots[k][m] for k in kn], 0) - np.mean([boots[k][m] for k in kr], 0)
                v_stmt = float(np.var(Db, ddof=1))
                nv = ps[(ps.layer == L) & (ps.family == new[0]) & (ps.name == new[1]) & (ps.metric == m)].value.to_numpy()
                rv = ps[(ps.layer == L) & (ps.family == ref[0]) & (ps.name == ref[1]) & (ps.metric == m)].value.to_numpy()
                dd = nv.mean() - rv.mean()
                ncl = boots[('ncl', L)][m]
                vstmt[(L, cid, m)] = (v_stmt, ncl)
                row = dict(layer=L, comparison=cid, metric=m, higher_is_better=hib[m], diff=dd, n_clusters=ncl,
                           v_stmt=v_stmt)
                if len(nv) > 1 and len(rv) > 1:
                    se2, dfw = welch_parts(nv, rv)
                    row.update(se_seed=np.sqrt(se2), welch_df=dfw, **combine(dd, se2, dfw, v_stmt, ncl))
                    row['stmt_share_of_variance'] = v_stmt / (se2 + v_stmt)
                val_rows.append(row)
    val = pd.DataFrame(val_rows)
    val.to_csv(a.out / 'validation_combined.csv', index=False)
    if a.smoke:
        print(ck.groupby('metric').abs_diff.max())
        return
    # ---------------- transfer to the committed TEST seed statistics
    h1 = pd.read_csv(H1_TEST)
    tcomp = pd.read_csv(REPO / 'results/repe/repe_lm_test_evaluation/comparisons.csv')
    out = []
    for _, r in h1.iterrows():
        t = tcomp[(tcomp.metric == r.metric) & (tcomp.comparison == r.comparison) & (tcomp.layer == r.layer)].iloc[0]
        v_val, ncl = vstmt[(r.layer, r.comparison, r.metric)]
        v_test = v_val * ncl / N_TEST_STATEMENTS
        se2 = t.ref_sd ** 2 / 3 + t.new_sd ** 2 / 3
        c = combine(t['diff'], se2, t.welch_df, v_test, N_TEST_STATEMENTS)
        hb = bool(t.higher_is_better)
        excl = (c['ci_low'] > 0) if hb else (c['ci_high'] < 0)
        out.append(dict(hypothesis=r.hypothesis, metric=r.metric, comparison=r.comparison, layer=r.layer,
                        test_diff=t['diff'], seed_only_ci_low=t.ci95_low, seed_only_ci_high=t.ci95_high,
                        seed_only_holm_p=r.test_welch_p_holm, confirmed_cell_frozen=bool(r.confirmed_cell),
                        v_stmt_val=v_val, v_stmt_test=v_test, se_seed=np.sqrt(se2), **c, pairs_better=t.pairs_better,
                        ci_excludes_0=bool(excl)))
    h = pd.DataFrame(out)
    h['p_holm'] = holm(h.p)
    h['supported_under_seed_plus_statement'] = h.ci_excludes_0 & (h.pairs_better == 9) & (h.p_holm < 0.05)
    h.to_csv(a.out / 'h1_test_cells_adjusted.csv', index=False)
    m5 = pd.read_csv(M5_TEST)
    out = []
    for L in LAYERS:
        for cid in COMP:
            for m in ('owt_nmse', 'splice_delta_ce'):
                t = m5[(m5.layer == L) & (m5.comparison == cid) & (m5.metric == m)].iloc[0]
                v_val, ncl = vstmt[(L, cid, m)]
                v_test = v_val * ncl / N_TEST_DOCS
                se2 = t.ref_sd ** 2 / 3 + t.new_sd ** 2 / 3
                c = combine(t['diff'], se2, t.welch_df, v_test, N_TEST_DOCS)
                out.append(dict(layer=L, comparison=cid, metric=m, test_diff=t['diff'], seed_only_ci_low=t.ci95_low,
                                seed_only_ci_high=t.ci95_high, frozen_cls=t.cls, v_stmt_test=v_test, se_seed=np.sqrt(se2),
                                **c, pairs_better=t.pairs_better, ci_excludes_0=bool(c['ci_high'] < 0)))
    m = pd.DataFrame(out)
    m['p_holm'] = holm(m.p)
    m['supported_under_seed_plus_document'] = m.ci_excludes_0 & (m.pairs_better == 9) & (m.p_holm < 0.05)
    m.to_csv(a.out / 'm5_test_cells_adjusted.csv', index=False)
    info = dict(script=str(Path(__file__).relative_to(REPO)), B=B, boot_seed=BOOT_SEED, test_rows_used=0,
                max_reproduction_abs_diff=ck.groupby('metric').abs_diff.max().to_dict(),
                n_h1_supported=int(h.supported_under_seed_plus_statement.sum()),
                n_m5_supported=int(m.supported_under_seed_plus_document.sum()), runtime_s=time.time() - t0)
    (a.out / 'checks.json').write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))


if __name__ == '__main__':
    main()
