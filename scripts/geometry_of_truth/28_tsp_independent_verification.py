#!/usr/bin/env python3
"""TSP-VERIFY: independent re-computation of the TSP pilot (2026-11-14). VALIDATION only; no training; no test row.

Re-implements, without importing the pilot's evaluation code (scripts 24/26/27, M2, information_levels, the ladder,
the splicing module or the SAE class), every number the pilot's verdict rests on, and adds robustness checks:

  V1  GoT geographic (cities + neg_cities), group-B readers, validation cities:
      frozen logistic gap, re-fit logistic gap, re-fit MLP gap (3 and 10 network seeds), error-only AUROC,
      GoT NMSE, L0. Own loader (raw .pt + split CSVs + A/B CSV), own TopK forward from the state dict, sklearn probes.
  V2  Probe-fitting robustness: re-fit logistic at each fixed C in {0.01, 0.1, 1}; wider MLP (256 hidden).
  V3  Geometry: share of each frozen-probe direction inside the truth / random / PCA subspaces; relative error of each
      SAE inside S, inside span(frozen probe) and outside S.
  V4  Wording split: gaps scored separately on affirmative (cities) and negated (neg_cities) validation rows, and a
      cross-wording transfer probe (fitted on affirmative group-B rows only, scored on negated validation rows).
  V5  Different template and entity type: GoT numerical (larger_than + smaller_than), group-B probes, validation
      rows. The subspace S (from geographic group A) is never re-estimated, so this is a transfer test.
  V6  OWT validation reconstruction (NMSE, EV, L0; BOS excluded) and spliced CE / KL with an own hook (all OWT
      validation sequences, positions >= 1 replaced).
  V7  Statistics from the re-computed values: paired seed differences, own shared grouped bootstrap (2000 resamples
      of validation cities), the pre-registered se / t(2) / Holm rule, alternative uncertainty models, and a
      power / precision analysis.

Outputs: results/geometry_of_truth/truth_subspace_preservation/verification/ (refuses to overwrite).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
PILOT = ROOT / 'pilot'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints/layer_8'
R4 = REPO / 'results/openwebtext/r4_kl_finetune/checkpoints/layer_8'
LAYER, SEEDS, ARMS = 8, (42, 1042, 2042), ('ctrl', 'tsp', 'rand', 'pca')
C_GRID, N_BOOT, BSEED = (0.01, 0.1, 1.0), 2000, 777


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data
def load_task(names):
    ab = pd.read_csv(REPO / 'data/got/splits/train_AB_groups_truth_aware_pilot.csv')
    half_of = dict(zip(ab.group_id.astype(str), ab.half))
    X, y, split, grp, part = [], [], [], [], []
    for name in names:
        d = torch.load(REPO / f'data/got/activations/gpt2_small/{name}_final_token.pt', map_location='cpu',
                       weights_only=False)
        sp = pd.read_csv(REPO / f'data/got/splits/{name}_splits.csv').set_index('row_id').loc[d['row_ids'].numpy()]
        assert np.array_equal(sp.label.to_numpy(), d['labels'].numpy())
        keep = sp.split.isin(['train', 'val']).to_numpy()                       # test rows dropped here
        X.append(d['activations'][LAYER].float().numpy()[keep])
        y.append(sp.label.to_numpy()[keep])
        split.append(sp.split.to_numpy()[keep])
        grp.append(sp.group_id.astype(str).to_numpy()[keep])
        part.append(np.full(keep.sum(), name))
    X, y, split, grp, part = map(np.concatenate, (X, y, split, grp, part))
    half = np.array([half_of.get(g, '') if s == 'train' else '' for g, s in zip(grp, split)])
    assert 'test' not in set(split)
    assert not set(grp[split == 'train']) & set(grp[split == 'val'])
    return dict(X=X, y=y, A=(split == 'train') & (half == 'A'), B=(split == 'train') & (half == 'B'),
                V=split == 'val', g=grp, part=part)


# ------------------------------------------------------------------------------------------------ SAE
class TopK:
    """TopK SAE from a modern_v1 state dict (scale already folded): x_hat = W_d topk(relu(W_e (x - b) + b_e)) + b."""

    def __init__(self, path, device):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        s, cfg = ck['model_state_dict'], ck['config']
        assert cfg['architecture'] == 'topk' and cfg['format'] == 'modern_v1'
        self.k = int(cfg['k'])
        self.We, self.be = s['encoder.weight'].to(device), s['encoder.bias'].to(device)
        self.Wd, self.b = s['decoder.weight'].to(device), s['output_bias'].to(device)

    @torch.no_grad()
    def __call__(self, x):
        pre = torch.relu((x - self.b) @ self.We.T + self.be)
        v, i = pre.topk(self.k, -1)
        z = torch.zeros_like(pre).scatter(-1, i, v)
        return z @ self.Wd.T + self.b, (z > 0).sum(-1).float()

    def np(self, X, device, bs=4096):
        H, L0 = [], []
        for b in range(0, len(X), bs):
            h, l0 = self(torch.as_tensor(X[b:b + bs], device=device))
            H.append(h.cpu().numpy())
            L0.append(l0.cpu().numpy())
        return np.concatenate(H), np.concatenate(L0)


def sources():
    out = [('r2_base', s, R2 / f'topk_seed{s}/checkpoint_final.pt') for s in SEEDS]
    out += [('r4_mse', s, R4 / f'mse_topk_seed{s}/checkpoint.pt') for s in SEEDS]
    out += [(f'tsp_{a}', s, PILOT / f'checkpoints/layer_{LAYER}/{a}_topk_seed{s}/checkpoint.pt') for a in ARMS for s in SEEDS]
    return out


# ------------------------------------------------------------------------------------------------ probes
def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=0))


def pick_C(X, y, g):
    best = (-1, None)
    for C in C_GRID:
        sc = np.zeros(len(y))
        for a, b in GroupKFold(3).split(X, y, g):
            sc[b] = lr(C).fit(X[a], y[a]).decision_function(X[b])
        v = roc_auc_score(y, sc)
        if v > best[0] + 1e-12:
            best = (v, C)
    return best[1]


def mlp(Xtr, ytr, Xte, seeds, hidden=64):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    return np.mean([MLPClassifier((hidden,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=s)
                    .fit(a, ytr).predict_proba(b)[:, 1] for s in seeds], 0)


def boot_idx(groups, n, rng):
    u, inv = np.unique(groups, return_inverse=True)
    mem = [np.flatnonzero(inv == k) for k in range(len(u))]
    return [np.concatenate([mem[j] for j in rng.integers(len(u), size=len(u))]) for _ in range(n)]


def bauc(y, s, idx):
    return np.array([roc_auc_score(y[i], s[i]) for i in idx], np.float32)


def raw_dir(pipe):
    return pipe[-1].coef_[0] / pipe[0].scale_


class Readers:
    """Original-side readers of one task (group-B rows) and per-source scoring on validation rows."""

    def __init__(self, t, rng, full=True):
        self.t, self.full = t, full
        B, V = t['B'], t['V']
        X, y, g = t['X'], t['y'], t['g']
        self.yv = y[V]
        self.C = pick_C(X[B], y[B], g[B])
        self.pipe = lr(self.C).fit(X[B], y[B])
        self.a = raw_dir(self.pipe)
        self.orig = dict(logistic=self.pipe.decision_function(X[V]))
        if full:
            self.orig['mlp3'] = mlp(X[B], y[B], X[V], (0, 1, 2))
            self.orig['mlp10'] = mlp(X[B], y[B], X[V], range(10))
            self.orig['mlp256'] = mlp(X[B], y[B], X[V], range(5), 256)
            for C in C_GRID:
                self.orig[f'logistic_C{C}'] = lr(C).fit(X[B], y[B]).decision_function(X[V])
        self.auc = {k: roc_auc_score(self.yv, v) for k, v in self.orig.items()}
        self.idx = boot_idx(g[V], N_BOOT, rng)
        self.ob = {k: bauc(self.yv, v, self.idx) for k, v in self.orig.items()}

    def score(self, H, l0):
        t, B, V = self.t, self.t['B'], self.t['V']
        X, y, g = t['X'], t['y'], t['g']
        Xv, Hv = X[V], H[V]
        E = Xv - Hv
        r = dict(task_nmse=float(((E ** 2).sum()) / ((Xv - Xv.mean(0)) ** 2).sum()), task_l0=float(l0[V].mean()))
        s = dict(frozen_logistic=self.pipe.decision_function(Hv))
        r['error_term_auroc'] = roc_auc_score(self.yv, E @ self.a)
        Cr = pick_C(H[B], y[B], g[B])
        r['refit_C'] = Cr
        s['refit_logistic'] = lr(Cr).fit(H[B], y[B]).decision_function(Hv)
        if self.full:
            s['refit_mlp3'] = mlp(H[B], y[B], Hv, (0, 1, 2))
            s['refit_mlp10'] = mlp(H[B], y[B], Hv, range(10))
            s['refit_mlp256'] = mlp(H[B], y[B], Hv, range(5), 256)
            for C in C_GRID:
                s[f'refit_logistic_C{C}'] = lr(C).fit(H[B], y[B]).decision_function(Hv)
        boots = {}
        for k, v in s.items():
            ref = k.split('_', 1)[1]
            r[f'{k}_gap'] = self.auc[ref] - roc_auc_score(self.yv, v)
            boots[f'{k}_gap'] = self.ob[ref] - bauc(self.yv, v, self.idx)
        boots['error_term_auroc'] = bauc(self.yv, E @ self.a, self.idx)
        return r, boots


def wording(t, H):
    """Gaps on affirmative vs negated validation rows (pooled frozen probe) and a cross-wording transfer probe."""
    X, y, B, V, part, g = t['X'], t['y'], t['B'], t['V'], t['part'], t['g']
    pipe = lr(pick_C(X[B], y[B], g[B])).fit(X[B], y[B])
    out = {}
    for name in ('cities', 'neg_cities'):
        m = V & (part == name)
        out[f'frozen_gap_{name}'] = roc_auc_score(y[m], pipe.decision_function(X[m])) - \
            roc_auc_score(y[m], pipe.decision_function(H[m]))
    Ba, Vn = B & (part == 'cities'), V & (part == 'neg_cities')
    C = pick_C(X[Ba], y[Ba], g[Ba])
    o = lr(C).fit(X[Ba], y[Ba])
    out['xwording_orig_auroc'] = roc_auc_score(y[Vn], o.decision_function(X[Vn]))
    out['xwording_frozen_gap'] = out['xwording_orig_auroc'] - roc_auc_score(y[Vn], o.decision_function(H[Vn]))
    Ch = pick_C(H[Ba], y[Ba], g[Ba])
    out['xwording_refit_gap'] = out['xwording_orig_auroc'] - roc_auc_score(
        y[Vn], lr(Ch).fit(H[Ba], y[Ba]).decision_function(H[Vn]))
    return out


# ------------------------------------------------------------------------------------------------ OWT
def owt_val():
    seqs, acts = [], []
    for p in sorted((REPO / 'data/activations/gpt2_small/validation').glob('chunk_*.pt')):
        c = torch.load(p, map_location='cpu', weights_only=False)
        A = c['activations'][f'blocks.{LAYER}.hook_resid_pre']
        for i, n in enumerate(c['lengths']):
            n = int(n)
            seqs.append(c['tokens'][i, :n])
            acts.append(A[i, 1:n].float())                                   # BOS excluded
    return seqs, torch.cat(acts)


@torch.no_grad()
def splice(model, seqs, sae, device, orig_lp=None, bs=16):
    hook = f'blocks.{LAYER}.hook_resid_pre'
    ce_o = ce_s = kl = 0.0
    n = 0
    lps = []
    for b in range(0, len(seqs), bs):
        ss = seqs[b:b + bs]
        T = max(len(s) for s in ss)
        toks = torch.full((len(ss), T), 50256, dtype=torch.long)
        lens = torch.tensor([len(s) for s in ss])
        for i, s in enumerate(ss):
            toks[i, :len(s)] = s
        toks, lens = toks.to(device), lens.to(device)
        ar = torch.arange(T, device=device)[None]
        rep = (ar >= 1) & (ar < lens[:, None])

        def fn(resid, hook):
            resid[rep] = sae(resid[rep].float())[0].to(resid.dtype)
            return resid
        lo = model(toks).float().log_softmax(-1) if orig_lp is None else orig_lp[b // bs].to(device)
        if orig_lp is None:
            lps.append(lo.cpu())
        ls = model.run_with_hooks(toks, fwd_hooks=[(hook, fn)]).float().log_softmax(-1)
        m = ((ar >= 1) & (ar <= lens[:, None] - 2))[:, :-1]
        tgt = toks[:, 1:, None]
        ce_o += -lo[:, :-1].gather(-1, tgt)[..., 0][m].sum().item()
        ce_s += -ls[:, :-1].gather(-1, tgt)[..., 0][m].sum().item()
        kl += (lo[:, :-1].exp() * (lo[:, :-1] - ls[:, :-1])).sum(-1)[m].sum().item()
        n += int(m.sum())
    return dict(splice_delta_ce=(ce_s - ce_o) / n, splice_kl=kl / n, splice_n=n), lps


# ------------------------------------------------------------------------------------------------ statistics
T2 = stats.t.ppf(0.975, 2)


def compare(dx, dr, bx, br):
    d = np.asarray(dx) - np.asarray(dr)
    se_seed = d.std(ddof=1) / np.sqrt(len(d))
    v_stmt = float(np.var(bx.mean(0) - br.mean(0), ddof=1)) if bx is not None else 0.0
    se = np.sqrt(se_seed ** 2 + v_stmt)
    m = d.mean()
    df_sat = se ** 4 / (se_seed ** 4 / (len(d) - 1)) if se_seed > 0 else np.inf   # v_stmt from 2000 reps ~ known
    return dict(mean_diff=m, per_seed=d.round(4).tolist(), sd_seed_diff=d.std(ddof=1), se_seed=se_seed,
                se_stmt=np.sqrt(v_stmt), se=se, stmt_share_of_var=v_stmt / se ** 2 if se > 0 else np.nan,
                ci_low=m - T2 * se, ci_high=m + T2 * se, p=2 * stats.t.sf(abs(m) / se, 2),
                df_satterthwaite=df_sat, p_satterthwaite=2 * stats.t.sf(abs(m) / se, df_sat),
                p_seed_only=2 * stats.t.sf(abs(m) / se_seed, 2),
                p_stmt_only=2 * stats.norm.sf(abs(m) / np.sqrt(v_stmt)) if v_stmt > 0 else np.nan,
                seeds_better=int((d < 0).sum()))


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    out, run = np.empty_like(ps), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def power(delta, sd_seed, se_stmt_now, n_groups_now, n_seeds, n_groups, m=2, n_sim=20000, rng=None):
    """P(pass) of the frozen rule generalised to n seeds (all n favour X, t(n-1) CI excludes 0, Holm over m taken as
    Bonferroni-worst-case alpha/m) when the true effect is delta; statement se scales as 1/sqrt(groups)."""
    rng = rng or np.random.default_rng(0)
    s_t = se_stmt_now * np.sqrt(n_groups_now / n_groups)
    common = rng.normal(0, s_t, n_sim)                                       # shared statement sample
    d = delta + common[:, None] + rng.normal(0, sd_seed, (n_sim, n_seeds))
    mean, sd = d.mean(1), d.std(1, ddof=1)
    se = np.sqrt(sd ** 2 / n_seeds + s_t ** 2)
    tcrit = stats.t.ppf(1 - 0.05 / (2 * m), n_seeds - 1)
    ok_all = (d < 0).all(1)
    ok_ci = mean + tcrit * se < 0
    return dict(p_pass_all_seeds_rule=float((ok_all & ok_ci).mean()), p_pass_ci_only=float(ok_ci.mean()),
                p_all_seeds_favour=float(ok_all.mean()), expected_halfwidth=float(
                    tcrit * np.sqrt(sd_seed ** 2 / n_seeds + s_t ** 2)))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'verification')
    ap.add_argument('--skip-splice', action='store_true')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; refusing to overwrite.')
    a.out.mkdir(parents=True)
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    geo, num = load_task(('cities', 'neg_cities')), load_task(('larger_than', 'smaller_than'))
    info = dict(n_val_geo=int(geo['V'].sum()), n_B_geo=int(geo['B'].sum()), n_A_geo=int(geo['A'].sum()),
                n_val_groups_geo=len(set(geo['g'][geo['V']])), n_val_num=int(num['V'].sum()),
                n_B_num=int(num['B'].sum()), n_val_groups_num=len(set(num['g'][num['V']])), test_rows_used=0)
    print(info, flush=True)
    rg = Readers(geo, np.random.default_rng(BSEED))
    rn = Readers(num, np.random.default_rng(BSEED + 1), full=False)
    info['orig_auroc_geo'], info['orig_auroc_num'], info['orig_C_geo'] = rg.auc, rn.auc, rg.C
    P = np.load(PILOT / 'subspaces.npz')
    a_dir = {'geo_B_logistic': rg.a / np.linalg.norm(rg.a), 'num_B_logistic': rn.a / np.linalg.norm(rn.a)}
    geom = {f'{k}_share_in_{s}': float(np.linalg.norm(P[s].T @ v) ** 2) for k, v in a_dir.items() for s in P.files}
    geom['chance_share_rank8'] = 8 / 768
    geom['subspace_names'] = P.files
    print(geom, flush=True)
    seqs, owt = owt_val()
    owt_dev = owt.to(device)
    owt_var = float(((owt_dev - owt_dev.mean(0)) ** 2).sum())
    model = None
    if not a.skip_splice:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained('gpt2-small', device=device).eval()
        _, orig_lp = splice(model, seqs, lambda x: (x, None), device)
    rows, boots = [], {}
    for fam, seed, path in sources():
        sae = TopK(path, device)
        r = dict(family=fam, seed=seed, checkpoint=str(path.relative_to(REPO)), sha256=sha(path))
        H, l0 = sae.np(geo['X'], device)
        rr, bb = rg.score(H, l0)
        r.update({f'geo__{k}': v for k, v in rr.items()})
        boots.update({f'{fam}|{seed}|geo__{k}': v for k, v in bb.items()})
        r.update({f'geo__{k}': v for k, v in wording(geo, H).items()})
        Ev = geo['X'][geo['V']] - H[geo['V']]
        for nm in P.files:
            Q = P[nm]
            dev = geo['X'][geo['V']] - geo['X'][geo['A']].mean(0)
            r[f'geo__relerr_in_{nm}'] = float(np.linalg.norm(Ev @ Q) ** 2 / np.linalg.norm(dev @ Q) ** 2)
        av = a_dir['geo_B_logistic']
        r['geo__relerr_in_probe_dir'] = float(((Ev @ av) ** 2).sum() / (((geo['X'][geo['V']] - geo['X'][geo['A']].mean(0)) @ av) ** 2).sum())
        Hn, l0n = sae.np(num['X'], device)
        rr, bb = rn.score(Hn, l0n)
        r.update({f'num__{k}': v for k, v in rr.items()})
        boots.update({f'{fam}|{seed}|num__{k}': v for k, v in bb.items()})
        hs, l0s = [], []
        for b in range(0, len(owt_dev), 8192):
            h, l = sae(owt_dev[b:b + 8192])
            hs.append(((owt_dev[b:b + 8192] - h) ** 2).sum().item())
            l0s.append(l.sum().item())
        r['owt_nmse'] = sum(hs) / owt_var
        r['owt_ev'] = 1 - r['owt_nmse']
        r['owt_l0'] = sum(l0s) / len(owt_dev)
        if model is not None:
            r.update(splice(model, seqs, sae, device, orig_lp)[0])
        rows.append(r)
        print(f"{fam} {seed}: frozen {r['geo__frozen_logistic_gap']:.4f} refit {r['geo__refit_logistic_gap']:.4f} "
              f"mlp3 {r['geo__refit_mlp3_gap']:.4f} mlp10 {r['geo__refit_mlp10_gap']:.4f} err {r['geo__error_term_auroc']:.3f} "
              f"| num frozen {r['num__frozen_logistic_gap']:.4f} refit {r['num__refit_logistic_gap']:.4f} "
              f"| owt {r['owt_nmse']:.4f} l0 {r['owt_l0']:.1f} dCE {r.get('splice_delta_ce', np.nan):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del sae
    df = pd.DataFrame(rows)
    df.to_csv(a.out / 'per_sae.csv', index=False)
    np.savez_compressed(a.out / 'bootstrap.npz', **boots)

    # ---- agreement with the pilot's own outputs
    g = pd.read_csv(PILOT / 'eval/got_independent.csv')
    lad = pd.read_csv(PILOT / 'eval/ladder_layer8.csv')
    agree = []
    for ours, theirs, src in [('geo__frozen_logistic_gap', 'frozen_logistic_gap', g), ('geo__refit_logistic_gap', 'refit_logistic_gap', g),
                              ('geo__refit_mlp3_gap', 'refit_mlp_gap', g), ('geo__error_term_auroc', 'error_term_auroc', g),
                              ('geo__task_nmse', 'task_nmse', g), ('geo__task_l0', 'task_l0', g),
                              ('owt_nmse', 'owt_nmse', lad), ('owt_l0', 'owt_l0', lad),
                              ('splice_delta_ce', 'splice_delta_ce', lad), ('splice_kl', 'splice_kl_mean', lad)]:
        if ours not in df:
            continue
        m = df.merge(src[['family', 'seed', theirs]], on=['family', 'seed'], suffixes=('', '_pilot'))
        diff = (m[ours] - m[theirs if theirs != ours else f'{theirs}_pilot']).abs()
        agree.append(dict(metric=ours, pilot_column=theirs, max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean())))
    pd.DataFrame(agree).to_csv(a.out / 'agreement_with_pilot.csv', index=False)

    # ---- statistics
    def vals(fam, col):
        return df[df.family == fam].set_index('seed').loc[list(SEEDS), col].to_numpy(float)

    def bst(fam, col):
        k = f'{fam}|{SEEDS[0]}|{col}'
        return np.stack([boots[f'{fam}|{s}|{col}'] for s in SEEDS]) if k in boots else None
    metrics = [c for c in df.columns if c.startswith(('geo__', 'num__')) and c.endswith(('_gap', 'error_term_auroc', 'task_nmse'))
               and 'xwording_orig' not in c] + ['owt_nmse', 'owt_l0'] + (['splice_delta_ce', 'splice_kl'] if model else [])
    comps = [('tsp_tsp', 'tsp_ctrl'), ('tsp_tsp', 'tsp_rand'), ('tsp_tsp', 'tsp_pca'), ('tsp_ctrl', 'r2_base'),
             ('tsp_rand', 'tsp_ctrl'), ('tsp_pca', 'tsp_ctrl'), ('tsp_tsp', 'r2_base'), ('tsp_ctrl', 'r4_mse')]
    st = []
    for x, rf in comps:
        for mtr in metrics:
            st.append(dict(arm=x, ref=rf, metric=mtr, mean_x=vals(x, mtr).mean(), sd_x=vals(x, mtr).std(ddof=1),
                           mean_ref=vals(rf, mtr).mean(), sd_ref=vals(rf, mtr).std(ddof=1),
                           **compare(vals(x, mtr), vals(rf, mtr), bst(x, mtr), bst(rf, mtr))))
    st = pd.DataFrame(st)
    pr = st[(st.arm == 'tsp_tsp') & (st.ref == 'tsp_ctrl') & st.metric.isin(['geo__frozen_logistic_gap', 'geo__refit_logistic_gap'])].copy()
    pr['p_holm'] = holm(pr.p.to_numpy())
    pr['pass'] = (pr.seeds_better == 3) & (pr.ci_high < 0) & (pr.p_holm < 0.05)
    st.to_csv(a.out / 'comparisons.csv', index=False)
    pr.to_csv(a.out / 'primary_recomputed.csv', index=False)

    # ---- power / precision
    pw = []
    for _, row in pr.iterrows():
        for delta_name, delta in (('observed', row.mean_diff), ('half_observed', row.mean_diff / 2)):
            for n_s in (3, 5, 8, 10, 15):
                for n_g in (info['n_val_groups_geo'], 2 * info['n_val_groups_geo'], 500, 1000):
                    pw.append(dict(metric=row.metric, delta=delta_name, delta_value=delta, n_seeds=n_s, n_val_groups=n_g,
                                   **power(delta, row.sd_seed_diff, row.se_stmt, info['n_val_groups_geo'], n_s, n_g)))
    pd.DataFrame(pw).to_csv(a.out / 'power.csv', index=False)
    info.update(geometry=geom, runtime_s=time.time() - t0, script_sha256=sha(__file__),
                primary=pr[['metric', 'mean_diff', 'ci_low', 'ci_high', 'p', 'p_holm', 'seeds_better', 'pass',
                            'stmt_share_of_var', 'p_satterthwaite', 'p_seed_only']].to_dict('records'))
    (a.out / 'verification_info.json').write_text(json.dumps(info, indent=2, default=float))
    print(pr.to_string(), flush=True)
    print(pd.DataFrame(agree).to_string(), flush=True)


if __name__ == '__main__':
    main()
