#!/usr/bin/env python3
"""E1: TopK trained at task-level matched sparsity (layers 1, 8, 11). Protocol: results/shared/e1_task_l0_matched_topk/
README.md (+ preregistration.json). Evidence on completion: VALIDATION; no TEST row is read anywhere.

Stages (outputs refuse to be overwritten; every stage after 'prepare' requires the frozen pre-registration hashes):
  prepare  k* per (layer, task, X) = round(mean over X's 3 base SAEs of the mean task L0 on TRAIN prompts); split checks
  gate     (G1) training reproduction: TopK@128 L8 seed 42 retrained with the original train_one must equal the
           committed uninterrupted checkpoint (max |param diff| <= 1e-6);
           (G2) evaluation reproduction: M2 evaluate_source on the committed L8 TopK seed-42 SAE, with M2's exact
           seeds / job index / bootstrap, must reproduce per_cell.csv (|diff| <= 1e-6);
           (G3) M4 decompose on the same SAE at the L8 final site must reproduce preservation_rec (|diff| <= 1e-6).
  train    36 TopK SAEs (12 k* x 3 seeds; identical (L, k, seed) trained once) with the ORIGINAL train_one and the
           uninterrupted hyper-parameters (width 3072, Adam 3e-4, batch 256, 15,000 steps, OWT train cache)
  eval     M2 information metrics (all three M2 tasks), PCA at matched task NMSE, M4 causal metrics for the
           RepE-matched SAEs at that layer's M4 site, OWT validation reconstruction and splicing
  analyze  E1-H1 classes (Welch 95 % CI over seeds, BH over the frozen family), E1-H2 (paired IMP-L rule)
Run: python scripts/shared/e1_task_l0_matched_topk.py {prepare|gate|train|eval|analyze}
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
from scipy import stats
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.evaluation import information_levels as il  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

OUT = REPO / 'results/shared/e1_task_l0_matched_topk'
CK_SCRATCH = Path('/mnt/scratch/checkpoints/e1')
CK_EBS = REPO / 'results/openwebtext/e1_task_matched_topk'
DATA = REPO / 'data/activations/gpt2_small'
LAYERS, X_ARCHS, MATCH_TASKS, SEEDS = (1, 8, 11), ('relu', 'jumprelu'), ('got_cities', 'repe_persona'), (42, 1042, 2042)
M4_SITE = {1: 'slot', 8: 'final', 11: 'final'}
TOL = 1e-6
# Amendment E1-A1 (2026-11-22): probe fitting (M2 evaluate_source, M4 decompose, PCA references) runs with BLAS limited to
# one thread, as inside M2's loky workers; multi-threaded BLAS changes summation order and moved AUROC gaps by ~1e-4.
BLAS_THREADS = 1
PRIMARY = {'got_cities': ['refit_logistic_gap'], 'repe_persona': ['refit_logistic_gap', 'preservation_rec']}


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_freeze():
    pr = json.loads((OUT / 'preregistration.json').read_text())
    for rel, h in pr['hashes'].items():
        if sha(REPO / rel) != h:
            raise SystemExit(f'{rel}: hash differs from the frozen pre-registration; refusing to run.')
    return pr


def refuse(*paths):
    for p in paths:
        if Path(p).exists():
            raise SystemExit(f'{p} exists; refusing to overwrite')


def train_args(layer, seed, steps=15000):
    # identical to scripts/openwebtext/13_consistent_protocol_retrain.py (the 108 uninterrupted baselines)
    return argparse.Namespace(layer=layer, features=3072, batch_size=256, steps=steps, lr=3e-4, seed=seed,
                              temperature=0.1, initial_sparsity_coeff=0.001, min_sparsity_coeff=1e-7,
                              max_sparsity_coeff=100.0, adjust_every=25, adjust_rate=0.5, match_tolerance=0.2,
                              log_every=2500)


def ckpt_path(root, L, k, seed):
    return root / f'layer_{L}' / f'topk_k{k}_seed{seed}' / 'checkpoint_step_15000.pt'


def codes(sae, X, device):
    with torch.no_grad():
        rec, z, _ = sae(torch.as_tensor(X).to(device))
    return rec.float().cpu().numpy(), z.float().cpu().numpy()


# ------------------------------------------------------------------------------------------------ prepare
def stage_prepare():
    refuse(OUT / 'k_star.json', OUT / 'pretraining_checks.json')
    M2 = load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
    device = 'cuda'
    tasks = M2.load_tasks()
    checks = dict(test_rows_used=0)
    for tn, t in tasks.items():
        assert 'test' not in set(t['split'])
        tr, va = t['split'] == 'train', t['split'] == 'validation'
        checks[tn] = dict(n_train=int(tr.sum()), n_validation=int(va.sum()),
                          group_overlap=len(set(t['groups'][tr]) & set(t['groups'][va])))
        assert checks[tn]['group_overlap'] == 0
    ks = []
    for L in LAYERS:
        for tn in MATCH_TASKS:
            t = tasks[tn]
            Xtr = t['X'][L][t['split'] == 'train']
            for X in X_ARCHS:
                per = []
                for s in SEEDS:
                    sae, _ = fm.load_sae(uninterrupted_checkpoint(L, X, s), device)
                    _, z = codes(sae, Xtr, device)
                    per.append(float((z > 0).sum(1).mean()))
                l0 = float(np.mean(per))
                k = int(max(1, round(l0)))
                ok = abs(k - l0) / l0 <= 0.10
                ks.append(dict(layer=L, task=tn, X=X, task_l0_train_per_seed=per, task_l0_train=l0, k_star=k,
                               within_10pct=bool(ok)))
                if not ok:
                    raise SystemExit(f'k* {k} not within 10 % of task L0 {l0:.2f} at L{L} {tn} {X}: E1 stops')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'k_star.json').write_text(json.dumps(ks, indent=1))
    (OUT / 'pretraining_checks.json').write_text(json.dumps(checks, indent=1))
    print(pd.DataFrame(ks)[['layer', 'task', 'X', 'task_l0_train', 'k_star']].to_string(index=False))


# --------------------------------------------------------------------------------------------------- gate
def m2_layer_context(M2, tasks, L):
    """M2's exact per-layer originals and bootstrap indices (same rng sequence as M2.main)."""
    rng = np.random.default_rng(M2.SEED + L)
    origs = {tn: M2.original_side(t, L, rng) for tn, t in tasks.items()}
    bidx = {tn: il.bootstrap_indices(t['groups'][t['split'] == 'validation'], M2.N_BOOT,
                                     np.random.default_rng(M2.SEED + 100 + L)) for tn, t in tasks.items()}
    return origs, bidx


def m2_job_index(M2, tasks, arch, seed, tn):
    """Index of (arch, seed, task) in M2.main's per-layer job list (pca jobs per task, then seeds x tasks)."""
    nt = len(tasks)
    a, si, ti = M2.ARCHS.index(arch), M2.SEEDS.index(seed), list(tasks).index(tn)
    return a * (nt + len(M2.SEEDS) * nt) + nt + si * nt + ti


def m2_eval(M2, tasks, L, origs, bidx, sae, tn, seed_index, device, meta):
    t = tasks[tn]
    H, Z = codes(sae, t['X'][L], device)
    r, _ = M2.evaluate_source(meta, t['X'][L], H, Z, t['y'], t['split'], t['groups'], M2.strip(origs[tn]), bidx[tn],
                              M2.SEED + 1000 * L + seed_index)
    return r


class M4Ctx:
    def __init__(self, device):
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer
        self.M4 = load('scripts/repe/24_sae_patch_decomposition.py', 'm4')
        self.model = HookedTransformer.from_pretrained('gpt2-small', device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained('gpt2')
        self.vrows = self.M4.build_rows(tok, 'validation', self.M4.PERSONA, ['H', 'U', 'Hp'])
        self.device, self.batches = device, {}

    def preservation(self, sae, L):
        M4 = self.M4
        if L not in self.batches:
            eng = M4.Engine(self.model, L)
            self.batches[L] = (eng, M4.clean_batches(eng, self.vrows, ['H', 'U', 'Hp'], M4_SITE[L], self.device))
        eng, vb = self.batches[L]
        gen = torch.Generator(device=self.device).manual_seed(M4.SEED)
        dfp, dr, _ = M4.decompose(eng, sae, vb, 'persona', gen)
        dfp = M4.add_derived(dfp)
        meta = dict(layer=L, group=M4_SITE[L], kind='persona')
        s = M4.summarise_decomp(dfp, dr, meta, np.random.default_rng(M4.SEED))
        return [x for x in s if x.get('template_set') == 'T2_T4'][0]


def stage_gate():
    check_freeze()
    refuse(OUT / 'gate.json')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    TR = load('scripts/openwebtext/03_training_sae.py', 'owt_train')
    M2 = load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
    res, t0 = {}, time.time()
    # G1 training reproduction
    L, s = 8, 42
    train, val = TR.load_split(DATA, 'train', L), TR.load_split(DATA, 'validation', L)
    model, _ = TR.train_one('topk', 128, train, val, train_args(L, s), torch.device(device))
    ref = torch.load(uninterrupted_checkpoint(L, 'topk', s), map_location='cpu', weights_only=False)['model_state_dict']
    new = {k: v.cpu() for k, v in model.state_dict().items()}
    diff = max(float((new[k].float() - ref[k].float()).abs().max()) for k in ref)
    res['G1_training_reproduction'] = dict(max_abs_param_diff=diff, passed=diff <= TOL, seconds=round(time.time() - t0))
    del model, train, val
    # G2 M2 evaluation reproduction (committed L8 TopK seed 42, all three tasks); E1-A1: single-threaded BLAS
    blas = threadpool_limits(limits=BLAS_THREADS)
    tasks = M2.load_tasks()
    origs, bidx = m2_layer_context(M2, tasks, L)
    sae, _ = fm.load_sae(uninterrupted_checkpoint(L, 'topk', s), device)
    pc = pd.read_csv(REPO / 'results/shared/m2_four_level_faithfulness/per_cell.csv')
    g2 = {}
    for tn in tasks:
        i = m2_job_index(M2, tasks, 'topk', s, tn)
        r = m2_eval(M2, tasks, L, origs, bidx, sae, tn, i, device,
                    dict(layer=L, task=tn, source='sae', architecture='topk', seed=s, pca_k=np.nan, matched=True,
                         live_latents=np.nan))
        c = pc[(pc.layer == L) & (pc.task == tn) & (pc.source == 'sae') & (pc.architecture == 'topk') & (pc.seed == s)]
        assert len(c) == 1
        g2[tn] = {m: float(abs(r[m] - c.iloc[0][m])) for m in ('refit_logistic_gap', 'frozen_logistic_gap', 'task_nmse',
                                                               'refit_mlp_gap')}
    mx = max(max(v.values()) for v in g2.values())
    res['G2_m2_eval_reproduction'] = dict(abs_diff=g2, max_abs_diff=mx, passed=mx <= TOL)
    # G3 M4 reproduction at the L8 final site
    m4 = M4Ctx(device)
    p = m4.preservation(sae, L)
    ds = pd.read_csv(REPO / 'results/repe/sae_patch_decomposition/decomposition_summary.csv')
    c = ds[(ds.layer == L) & (ds.group == 'final') & (ds.architecture == 'topk') & (ds.seed == s) &
           (ds.kind == 'persona') & (ds.template_set == 'T2_T4')]
    assert len(c) == 1
    d3 = float(abs(p['preservation_rec'] - c.iloc[0].preservation_rec))
    res['G3_m4_reproduction'] = dict(abs_diff_preservation_rec=d3, passed=d3 <= TOL)
    blas.unregister()
    res['blas_threads_for_probes'] = BLAS_THREADS
    res['all_passed'] = all(v['passed'] for v in res.values() if isinstance(v, dict))
    (OUT / 'gate.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if not res['all_passed']:
        raise SystemExit('E1 GATE FAILED: stopping (no counted run)')


# -------------------------------------------------------------------------------------------------- train
def cells():
    return json.loads((OUT / 'k_star.json').read_text())


def stage_train():
    check_freeze()
    g = json.loads((OUT / 'gate.json').read_text())
    if not g.get('all_passed'):
        raise SystemExit('gate not passed')
    TR = load('scripts/openwebtext/03_training_sae.py', 'owt_train')
    device = torch.device('cuda')
    ledger = OUT / 'ledger.csv'
    rows = pd.read_csv(ledger).to_dict('records') if ledger.exists() else []
    done = {(r['layer'], r['k'], r['seed']) for r in rows}
    for L in LAYERS:
        ks = sorted({c['k_star'] for c in cells() if c['layer'] == L})
        train = val = None
        for k in ks:
            for s in SEEDS:
                if (L, k, s) in done:
                    continue
                ck = ckpt_path(CK_SCRATCH, L, k, s)
                refuse(ck)
                if train is None:
                    train, val = TR.load_split(DATA, 'train', L), TR.load_split(DATA, 'validation', L)
                t0 = time.time()
                model, vm = TR.train_one('topk', k, train, val, train_args(L, s), device)
                ck.parent.mkdir(parents=True, exist_ok=True)
                args = train_args(L, s)
                torch.save({'model_state_dict': model.cpu().state_dict(),
                            'config': {**vars(args), 'architecture': 'topk', 'target_l0': k, 'd_in': 768,
                                       'procedure': 'e1_task_l0_matched_topk_2026-11-21'},
                            'validation_metrics': vm}, ck)
                rows.append(dict(layer=L, k=k, seed=s, checkpoint=str(ck), sha256=sha(ck), val_l0=vm['l0'],
                                 val_explained_variance=vm['explained_variance'], seconds=round(time.time() - t0, 1)))
                pd.DataFrame(rows).to_csv(ledger, index=False)
                print(f'L{L} k={k} s={s}: EV {vm["explained_variance"]:.4f} L0 {vm["l0"]:.1f} ({rows[-1]["seconds"]}s)',
                      flush=True)
                del model
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------- eval
def stage_eval():
    with threadpool_limits(limits=BLAS_THREADS):          # E1-A1
        _stage_eval()


def _stage_eval():
    check_freeze()
    refuse(OUT / 'eval_m2.csv', OUT / 'eval_m4.csv', OUT / 'eval_owt.csv')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    M2 = load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
    M5 = load('scripts/shared/m5_confirmatory_test.py', 'm5')
    SP, TR = M5.SP, load('scripts/openwebtext/03_training_sae.py', 'owt_train')
    led = pd.read_csv(OUT / 'ledger.csv')
    tasks = M2.load_tasks()
    pca_stats, _ = M2.owt_pca(False)
    m4 = M4Ctx(device)
    SP.BATCH = 8
    spl = SP.Splicer(m4.model, device)
    seqs = M5.load_owt('validation')
    orig_lp = M5.LAD._OnDevice([t.cpu() for t in SP.owt_original(spl, seqs)], device)
    m2rows, m4rows, owtrows = [], [], []
    for L in LAYERS:
        origs, bidx = m2_layer_context(M2, tasks, L)
        mu, cov = pca_stats[L]
        _, V = il.pca_basis(mu, cov)
        xval = TR.load_split(DATA, 'validation', L)
        persona_ks = {c['k_star'] for c in cells() if c['layer'] == L and c['task'] == 'repe_persona'}
        for j, r in enumerate(led[led.layer == L].sort_values(['k', 'seed']).itertuples()):
            assert sha(r.checkpoint) == r.sha256
            sae, _ = fm.load_sae(r.checkpoint, device)
            base = dict(layer=L, source='sae', architecture=f'topk_k{r.k}', seed=int(r.seed), k=int(r.k))
            for ti, tn in enumerate(tasks):
                rr = m2_eval(M2, tasks, L, origs, bidx, sae, tn, 500 + 10 * j + ti, device,
                             dict(base, task=tn, pca_k=np.nan, matched=True, live_latents=np.nan))
                m2rows.append(rr)
                t = tasks[tn]
                va = t['split'] == 'validation'
                kp = il.pca_k_for_val_nmse(t['X'][L][va].astype(np.float64), mu, V, rr['task_nmse'])
                Hp = il.pca_reconstruct(t['X'][L].astype(np.float64), mu, V, kp).astype(np.float32)
                rp, _ = M2.evaluate_source(dict(base, task=tn, source='pca_task_nmse_matched', pca_k=kp, matched=True,
                                                live_latents=np.nan),
                                           t['X'][L], Hp, None, t['y'], t['split'], t['groups'],
                                           M2.strip(origs[tn]), bidx[tn], M2.SEED + 1000 * L + 900 + 10 * j + ti)
                m2rows.append(rp)
            if int(r.k) in persona_ks:
                p = m4.preservation(sae, L)
                m4rows.append(dict(base, site=M4_SITE[L], **{k_: p[k_] for k_ in p if k_ not in base}))
            xh, l0, act = fm.reconstruct(sae, xval, device)
            o = dict(base, **{f'owt_{m}': v for m, v in fm.reconstruction_metrics(xval, xh, l0).items()})
            o['owt_dead_fraction'] = float((np.asarray(act) == 0).mean())
            o.update({f'splice_{m}': v for m, v in SP.owt_eval(spl, seqs, L, SP.sae_fn(sae), orig_lp).items()})
            owtrows.append(o)
            print(f"L{L} k={r.k} s={r.seed}: EV {o['owt_explained_variance']:.3f} dCE {o['splice_delta_ce']:.4f}",
                  flush=True)
            del sae
            torch.cuda.empty_cache()
    pd.DataFrame(m2rows).to_csv(OUT / 'eval_m2.csv', index=False)
    pd.DataFrame(m4rows).to_csv(OUT / 'eval_m4.csv', index=False)
    pd.DataFrame(owtrows).to_csv(OUT / 'eval_owt.csv', index=False)


# ------------------------------------------------------------------------------------------------ analyze
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se > 0 else 1.0
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    p = 2 * stats.t.sf(abs(d) / se, df) if se > 0 else (0.0 if d != 0 else 1.0)
    return d, d - h, d + h, p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[o[i]] * m / (i + 1))
        q[o[i]] = prev
    return q


def stage_analyze():
    check_freeze()
    refuse(OUT / 'analysis')
    pc = pd.read_csv(REPO / 'results/shared/m2_four_level_faithfulness/per_cell.csv')
    ds = pd.read_csv(REPO / 'results/repe/sae_patch_decomposition/decomposition_summary.csv')
    e2, e4, eo = (pd.read_csv(OUT / f) for f in ('eval_m2.csv', 'eval_m4.csv', 'eval_owt.csv'))
    e2 = e2[e2.source == 'sae']

    def vals(L, tn, arch, metric, k=None):
        if metric == 'preservation_rec':
            if k is None:
                g = ds[(ds.layer == L) & (ds.group == M4_SITE[L]) & (ds.architecture == arch) & (ds.kind == 'persona')
                       & (ds.template_set == 'T2_T4')]
            else:
                g = e4[(e4.layer == L) & (e4.k == k)]
            return g.sort_values('seed')[metric].to_numpy()
        if k is None:
            g = pc[(pc.layer == L) & (pc.task == tn) & (pc.source == 'sae') & (pc.architecture == arch)]
        else:
            g = e2[(e2.layer == L) & (e2.task == tn) & (e2.k == k)]
        return g.sort_values('seed')[metric].to_numpy()

    rows, h2 = [], []
    for c in cells():
        L, tn, X, k = c['layer'], c['task'], c['X'], c['k_star']
        degenerate = bool((eo[(eo.layer == L) & (eo.k == k)].owt_explained_variance < 0.5).any())
        for m in PRIMARY[tn]:
            x, t128, tk = vals(L, tn, X, m), vals(L, tn, 'topk', m), vals(L, tn, 'topk', m, k)
            assert len(x) == len(t128) == len(tk) == 3, (L, tn, X, m)
            d0, l0_, h0, p0 = welch(x, t128)
            d1, l1, h1, p1 = welch(x, tk)
            rows.append(dict(layer=L, task=tn, X=X, k_star=k, metric=m, diff_128=d0, ci128_lo=l0_, ci128_hi=h0,
                             p128=p0, diff_kstar=d1, cik_lo=l1, cik_hi=h1, pk=p1, degenerate_sparsity=degenerate))
            dd = tk - t128                           # E1-H2: paired by seed (IMP-L rule)
            cls = ('DIFFERS (k* < 128 side)' if (np.sign(dd) == np.sign(dd.mean())).all() and abs(dd.mean()) > 2 * dd.std(ddof=1)
                   else 'NO CLEAR CHANGE')
            h2.append(dict(layer=L, task=tn, X=X, k_star=k, metric=m, mean_paired_diff=float(dd.mean()),
                           sd=float(dd.std(ddof=1)), per_seed=dd.tolist(), cls=cls))
    R = pd.DataFrame(rows)
    R['q_kstar'] = bh(R.pk)
    R['q_128'] = bh(R.p128)

    def klass(r, use_bh):
        """Frozen E1-H1 classes (README §1.1). use_bh: the k* CI must also survive BH (q < 0.05)."""
        ex128 = r.ci128_lo > 0 or r.ci128_hi < 0
        exk = (r.cik_lo > 0 or r.cik_hi < 0) and (r.q_kstar < 0.05 if use_bh else True)
        if exk:
            return 'PERSISTS' if np.sign(r.diff_kstar) == np.sign(r.diff_128) else 'REVERSED'
        return 'ATTENUATED' if ex128 else 'NO DIFFERENCE'
    R['cls'] = [klass(r, False) for r in R.itertuples()]
    R['cls_bh'] = [klass(r, True) for r in R.itertuples()]
    sig128 = R[(R.ci128_lo > 0) | (R.ci128_hi < 0)]
    share = float((sig128.cls_bh == 'PERSISTS').mean()) if len(sig128) else float('nan')
    summary = dict(n_tests=len(R), n_significant_at_128=len(sig128), share_persists=share,
                   robust=bool(len(sig128) and share >= 0.75), class_counts=R.cls.value_counts().to_dict(),
                   class_counts_bh=R.cls_bh.value_counts().to_dict(),
                   n_degenerate_cells=int(R.degenerate_sparsity.sum()))
    out = OUT / 'analysis'
    out.mkdir()
    R.to_csv(out / 'e1_h1_classes.csv', index=False)
    pd.DataFrame(h2).to_csv(out / 'e1_h2_topk_kstar_vs_128.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'gate', 'train', 'eval', 'analyze'])
    a = ap.parse_args()
    {'prepare': stage_prepare, 'gate': stage_gate, 'train': stage_train, 'eval': stage_eval,
     'analyze': stage_analyze}[a.stage]()


if __name__ == '__main__':
    main()
