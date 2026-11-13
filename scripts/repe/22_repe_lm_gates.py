#!/usr/bin/env python3
"""M1 step 3: gates K1-K7 for the RepE-LM task. TRAIN statements decide every gate; VALIDATION is descriptive only.

Pre-registered in results/repe/repe_lm_gates/preregistration.json (pass rules copied in GATES below).

  K1 construction invariants   token level (script 20) + activation level (script 21)             PASS/FAIL
  K2 masked bag-of-words       slot token removed + template/length/label/final-token features   AUROC == 0.5 +- 1e-9
     + single-feature shortcut AUROCs (length, slot position, final position, prefix length,
       template, factual label, statement id, final token)                                      each == 0.5 +- 1e-9
  K3 unmasked bag-of-words     positive control (slot token included)                              AUROC >= 0.99
  K4 neutral-pair floor        N1/N2 decodability and cross-pair transfer, per layer              descriptive floor
  K5 semantic transfer         leave-one-persona-pair-out (T1-T4), final token, logistic          >= 0.70 at some
                               layer >= 1 with bootstrap CI low > K4 transfer floor at that layer
  K6 leakage                   train/validation duplicates (exact normalised, token Jaccard > 0.8) none
                               + leave-one-template-out AUROC and grouped-vs-ungrouped CV          reported
  K7 behavioural gate          final-token KL(H||U) vs KL(H||Hp), per train statement averaged     Wilcoxon p < 0.01
                               over T1-T4                                                         and median ratio >= 1.5

Run from repository root:
    python scripts/repe/22_repe_lm_gates.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data/repe/repe_lm'
ACT = REPO / 'data/repe/activations/repe_lm'
RES = REPO / 'results/repe/repe_lm_gates'
AB = REPO / 'data/repe/splits/r1_train_AB_groups.csv'
N_LAYERS, N_BOOT, SEED, C = 12, 1000, 20261108, 0.1
FIT = ['T1', 'T2', 'T3', 'T4']
OUTPUT_FILES = ['gates.json', 'k2_k3_bow_and_shortcuts.csv', 'probe_auroc_by_layer.csv', 'k5_lopo_by_template.csv',
                'k7_behaviour_summary.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else np.nan


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def massmean_scores(Xtr, ytr, Xte):
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    return Xte @ w


# ------------------------------------------------------------------------------ K2 / K3

def bow_gates(d):
    tr = d[(d.split == 'train') & d.template.isin(FIT) & d.variant.isin(['H', 'U'])].reset_index(drop=True)
    toks = tr.token_ids.str.split()
    masked = [' '.join(t[:sp] + t[sp + 1:]) for t, sp in zip(toks, tr.slot_pos)]
    full = [' '.join(t) for t in toks]
    extra = pd.get_dummies(tr.template).astype(float)
    extra['prefix_tokens'] = tr.prefix_tokens
    extra['factual_label'] = tr.factual_label
    extra['n_tokens'] = tr.n_tokens
    final_tok = pd.get_dummies(toks.map(lambda t: 'f' + t[-1])).astype(float)
    gkf = GroupKFold(5)
    out = []
    for name, docs in [('K2_masked_bow', masked), ('K3_unmasked_bow', full)]:
        X = hstack([CountVectorizer(token_pattern=r'\S+').fit_transform(docs),
                    csr_matrix(extra.to_numpy()), csr_matrix(final_tok.to_numpy())]).tocsr()
        sc = np.zeros(len(tr))
        for a, b in gkf.split(X, tr.label, tr.norm_group):
            m = LogisticRegression(C=1.0, max_iter=5000, solver='liblinear').fit(X[a], tr.label[a])
            sc[b] = m.decision_function(X[b])
        out.append(dict(check=name, feature='bag-of-words + template + lengths + factual label + final token',
                        auroc=auroc(tr.label, sc), n=len(tr)))
    singles = dict(n_tokens=tr.n_tokens, slot_pos=tr.slot_pos, final_pos=tr.final_pos, prefix_tokens=tr.prefix_tokens,
                   template=tr.template.map({t: i for i, t in enumerate(FIT)}), factual_label=tr.factual_label,
                   statement_id=tr.statement_id, final_token=toks.map(lambda t: int(t[-1])))
    for k, v in singles.items():
        out.append(dict(check='K2_single_feature_shortcut', feature=k, auroc=auroc(tr.label, v.to_numpy(float)),
                        n=len(tr)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- probes

def load_acts(pos):
    return [np.load(ACT / f'{pos}_layer{L}.npy') for L in range(N_LAYERS)]


def grouped_cv_auroc(X, y, groups, grouped=True):
    sc = np.zeros(len(y))
    split = GroupKFold(5).split(X, y, groups) if grouped else KFold(5, shuffle=True, random_state=SEED).split(X)
    for a, b in split:
        sc[b] = probe().fit(X[a], y[a]).decision_function(X[b])
    return auroc(y, sc)


def boot_mean_auroc(parts, rng):
    """parts: list of (y, score, statement_ids). Bootstrap statements (shared ids across parts); mean AUROC."""
    ids = np.unique(np.concatenate([p[2] for p in parts]))
    vals = []
    for _ in range(N_BOOT):
        draw = rng.choice(ids, len(ids), replace=True)
        cnt = pd.Series(draw).value_counts()
        a = []
        for y, s, sid in parts:
            w = pd.Series(sid).map(cnt).fillna(0).to_numpy()
            rep = np.repeat(np.arange(len(y)), w.astype(int))
            a.append(auroc(y[rep], s[rep]))
        vals.append(np.nanmean(a))
    return np.percentile(vals, [2.5, 97.5])


def probe_gates(d, rng):
    ab = pd.read_csv(AB).set_index('statement_id').half
    tr = d[d.split == 'train'].copy()
    tr['half'] = tr.statement_id.map(ab)
    assert tr.half.notna().all(), 'train statement without an A/B half'
    va = d[d.split == 'validation']
    rows, lopo_rows = [], []
    for pos in ('final', 'slot'):
        acts = load_acts(pos)
        for L in range(N_LAYERS):
            X = acts[L]
            per = dict(position=pos, layer=L)
            # persona decodability within T1-T4 (grouped CV over statements), and ungrouped for leakage contrast
            m = (d.split == 'train') & d.template.isin(FIT) & d.variant.isin(['H', 'U'])
            per['persona_grouped_cv'] = grouped_cv_auroc(X[m], d.label[m].to_numpy(), d.norm_group[m].to_numpy())
            per['persona_ungrouped_cv'] = grouped_cv_auroc(X[m], d.label[m].to_numpy(), None, grouped=False)
            # K4 floor: neutral decodability (grouped CV) and cross-pair transfer N1 <-> N2 (A -> B and B -> A)
            mn = (d.split == 'train') & (d.template_role == 'neutral')
            per['K4_neutral_grouped_cv'] = grouped_cv_auroc(X[mn], d.label[mn].to_numpy(), d.norm_group[mn].to_numpy())
            tf = []
            for src, dst in [('N1', 'N2'), ('N2', 'N1')]:
                for h_fit, h_te in [('A', 'B'), ('B', 'A')]:
                    a = tr.index[(tr.template == src) & (tr.half == h_fit)]
                    b = tr.index[(tr.template == dst) & (tr.half == h_te)]
                    s = probe().fit(X[a], d.label[a]).decision_function(X[b])
                    v = auroc(d.label[b], s)
                    tf.append(max(v, 1 - v))
            per['K4_neutral_transfer_floor'] = float(np.mean(tf))
            # K5 leave-one-persona-pair-out (= K6 leave-one-template-out), logistic and mass-mean
            parts, parts_mm = [], []
            for held in FIT:
                rest = [t for t in FIT if t != held]
                for h_fit, h_te in [('A', 'B'), ('B', 'A')]:
                    a = tr.index[tr.template.isin(rest) & tr.variant.isin(['H', 'U']) & (tr.half == h_fit)]
                    b = tr.index[(tr.template == held) & tr.variant.isin(['H', 'U']) & (tr.half == h_te)]
                    s = probe().fit(X[a], d.label[a]).decision_function(X[b])
                    smm = massmean_scores(X[a], d.label[a].to_numpy(), X[b])
                    parts.append((d.label[b].to_numpy(), s, d.statement_id[b].to_numpy()))
                    parts_mm.append((d.label[b].to_numpy(), smm, d.statement_id[b].to_numpy()))
                    lopo_rows.append(dict(position=pos, layer=L, held_out=held, fit_half=h_fit,
                                          auroc_logistic=auroc(d.label[b], s), auroc_massmean=auroc(d.label[b], smm)))
            per['K5_lopo_logistic'] = float(np.mean([auroc(y, s) for y, s, _ in parts]))
            per['K5_lopo_massmean'] = float(np.mean([auroc(y, s) for y, s, _ in parts_mm]))
            if pos == 'final':
                lo, hi = boot_mean_auroc(parts, rng)
                per['K5_lopo_logistic_ci_low'], per['K5_lopo_logistic_ci_high'] = lo, hi
            # persona probe (all T1-T4 train) applied to neutral pairs: should not separate them
            a = tr.index[tr.template.isin(FIT) & tr.variant.isin(['H', 'U'])]
            pp = probe().fit(X[a], d.label[a])
            for t in ('N1', 'N2'):
                b = tr.index[tr.template == t]
                per[f'persona_probe_on_{t}'] = auroc(d.label[b], pp.decision_function(X[b]))
            # descriptive validation: T1-T4 and held-out T5-T6 (no fitting on these rows)
            for t in FIT + ['T5', 'T6']:
                b = va.index[(va.template == t) & va.variant.isin(['H', 'U'])]
                per[f'val_{t}'] = auroc(d.label[b], pp.decision_function(X[b]))
            rows.append(per)
            print(pos, L, {k: round(v, 3) for k, v in per.items() if isinstance(v, float)}, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(lopo_rows)


# ------------------------------------------------------------------------------- K7

def behaviour_gate():
    b = pd.read_csv(RES / 'k7_behaviour_rows.csv')
    out, summ = {}, []
    for split in ('train', 'validation'):
        f = b[(b.split == split) & b.template.isin(FIT)]
        per = f.groupby('statement_id')[['kl_H_U', 'kl_H_Hp']].mean()
        ratio = per.kl_H_U / per.kl_H_Hp
        w = stats.wilcoxon(per.kl_H_U, per.kl_H_Hp, alternative='greater')
        c = f.dropna(subset=['cont_lp_H'])
        cper = c.assign(dU=(c.cont_lp_U - c.cont_lp_H).abs(), dHp=(c.cont_lp_Hp - c.cont_lp_H).abs()) \
            .groupby('statement_id')[['dU', 'dHp']].mean()
        wc = stats.wilcoxon(cper.dU, cper.dHp, alternative='greater')
        sign = c.assign(d=c.cont_lp_U - c.cont_lp_H).groupby('factual_label').d.mean().to_dict()
        n = b[(b.split == split) & (b.role == 'neutral')]
        out[split] = dict(
            n_statements=int(len(per)), median_kl_H_U=float(per.kl_H_U.median()),
            median_kl_H_Hp=float(per.kl_H_Hp.median()), median_ratio=float(ratio.median()),
            frac_statements_U_gt_Hp=float((per.kl_H_U > per.kl_H_Hp).mean()), wilcoxon_p=float(w.pvalue),
            neutral_median_kl_Na_Nb=float(n.groupby('statement_id').kl_Na_Nb.mean().median()),
            continuation_n=int(len(cper)), continuation_median_absdiff_U=float(cper.dU.median()),
            continuation_median_absdiff_Hp=float(cper.dHp.median()), continuation_wilcoxon_p=float(wc.pvalue),
            continuation_mean_signed_U_minus_H_by_factual_label=sign)
        for t, g in b[b.split == split].groupby('template'):
            r = dict(split=split, template=t, n=len(g))
            for col in [c_ for c_ in g.columns if c_.startswith('kl_')]:
                if g[col].notna().any():
                    r[f'median_{col}'] = float(g[col].median())
            summ.append(r)
        if split == 'validation':
            for t in ('T5', 'T6'):
                g = b[(b.split == split) & (b.template == t)]
                out[split][f'heldout_{t}_median_kl_H_U'] = float(g.kl_H_U.median())
    tr = out['train']
    tr['pass'] = bool(tr['wilcoxon_p'] < 0.01 and tr['median_ratio'] >= 1.5)
    return out, pd.DataFrame(summ)


def main():
    for f in OUTPUT_FILES:
        if (RES / f).exists():
            raise SystemExit(f'{RES / f} exists; refusing to overwrite.')
    pr = json.loads((RES / 'preregistration.json').read_text())
    rel = str(Path(__file__).resolve().relative_to(REPO))
    if pr['hashes'].get(rel) != sha(__file__):
        raise SystemExit('script hash differs from the pre-registered hash; refusing to run.')
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(DATA / 'prompts.csv')
    assert set(d.split) <= {'train', 'validation'}
    tmpl = json.loads((DATA / 'templates.json').read_text())
    d['slot_pos'] = d.template.map({t: v['slot_pos'] for t, v in tmpl.items()})
    ver = json.loads((DATA / 'split_verification.json').read_text())
    k1a = json.loads((RES / 'k1_activation_invariants.json').read_text())

    bow = bow_gates(d)
    bow.to_csv(RES / 'k2_k3_bow_and_shortcuts.csv', index=False)
    k2_vals = bow[bow.check.str.startswith('K2')].auroc
    k3 = float(bow[bow.check == 'K3_unmasked_bow'].auroc.iloc[0])

    pl, lopo = probe_gates(d, rng)
    pl.to_csv(RES / 'probe_auroc_by_layer.csv', index=False)
    lopo.to_csv(RES / 'k5_lopo_by_template.csv', index=False)
    fin = pl[pl.position == 'final']
    ok = fin[(fin.layer >= 1) & (fin.K5_lopo_logistic >= 0.70) &
             (fin.K5_lopo_logistic_ci_low > fin.K4_neutral_transfer_floor)]
    k7, k7s = behaviour_gate()
    k7s.to_csv(RES / 'k7_behaviour_summary.csv', index=False)

    gates = dict(
        K1=dict(token_level=ver['K1_token_level']['pass'], activation_level=k1a['pass'],
                n_token_pairs=ver['K1_token_level']['n_compared_pairs'], n_activation_pairs=k1a['n_pairs_checked'],
                max_abs_diff_before_slot_by_layer=k1a['max_abs_diff_before_slot_by_layer'],
                min_abs_diff_at_slot_layer0=k1a['min_abs_diff_at_slot_layer0'],
                status='PASS' if ver['K1_token_level']['pass'] and k1a['pass'] else 'FAIL'),
        K2=dict(max_abs_deviation_from_0_5=float((k2_vals - 0.5).abs().max()),
                status='PASS' if (k2_vals - 0.5).abs().max() <= 1e-9 else 'FAIL'),
        K3=dict(auroc=k3, status='PASS' if k3 >= 0.99 else 'FAIL'),
        K4=dict(status='REPORTED (floor, not a gate)',
                neutral_grouped_cv_by_layer=fin.set_index('layer').K4_neutral_grouped_cv.round(4).to_dict(),
                neutral_transfer_floor_by_layer=fin.set_index('layer').K4_neutral_transfer_floor.round(4).to_dict()),
        K5=dict(lopo_logistic_by_layer=fin.set_index('layer').K5_lopo_logistic.round(4).to_dict(),
                lopo_ci_low_by_layer=fin.set_index('layer').K5_lopo_logistic_ci_low.round(4).to_dict(),
                passing_layers=ok.layer.tolist(),
                status='PASS' if len(ok) else 'FAIL (relabel task "instruction-token information")'),
        K6=dict(duplicates=ver['K6_duplicates'], status='PASS' if ver['K6_duplicates']['pass'] else 'FAIL',
                leave_one_template_out='= K5 (each T1-T4 template has its own word pair)',
                grouped_vs_ungrouped_cv_final=fin.set_index('layer')[['persona_grouped_cv', 'persona_ungrouped_cv']]
                .round(4).to_dict('index')),
        K7=dict(train=k7['train'], validation_descriptive=k7['validation'],
                status='PASS' if k7['train']['pass'] else 'FAIL (skip M3/M4)'),
    )
    stop = []
    if gates['K1']['status'] != 'PASS' or gates['K2']['status'] != 'PASS':
        stop.append('K1/K2 failed: fix construction; no further analysis')
    if gates['K5']['status'] != 'PASS':
        stop.append('K5 failed: task relabelled "instruction-token information"; M2 only')
    if gates['K7']['status'] != 'PASS':
        stop.append('K7 failed: skip M3/M4')
    gates['decision'] = dict(M2_allowed=gates['K1']['status'] == 'PASS' and gates['K2']['status'] == 'PASS'
                             and gates['K3']['status'] == 'PASS',
                             M3_allowed=gates['K1']['status'] == 'PASS' and gates['K2']['status'] == 'PASS'
                             and gates['K7']['status'] == 'PASS',
                             notes=stop)
    gates['script_sha256'] = sha(__file__)
    (RES / 'gates.json').write_text(json.dumps(gates, indent=2, default=float))
    print(bow.to_string(index=False))
    print(json.dumps({k: v.get('status') for k, v in gates.items() if isinstance(v, dict)}, indent=1))
    print(json.dumps(gates['K7'], indent=1, default=float))
    print(json.dumps(gates['decision'], indent=1))


if __name__ == '__main__':
    main()
