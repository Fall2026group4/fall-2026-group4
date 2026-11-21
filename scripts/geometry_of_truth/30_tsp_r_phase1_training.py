#!/usr/bin/env python3
"""TSP-R Phase 1 training: 5 new R2 TopK L8 base SAEs + 120 fold-wise TSP fine-tunes (8 seeds x 5 arms x 3 folds).

Recipes are NOT re-implemented. They are reused from the hash-verified frozen scripts:
  * base SAEs: scripts/openwebtext/17_streaming_sae_training.py `train` (R2 recipe, 100M tokens, layer 8, topk only).
    Two module constants are patched: SEEDS (its per-seed batch generators are built from it) and the output folder
    (pilot-mode save path, so nothing is written into the historical R2 folder).
  * fine-tunes: scripts/geometry_of_truth/25_truth_subspace_finetune.py `run_one` (the exact pilot recipe). Only the
    GoT-A rows (fold-specific) and the subspace (fold-specific; plus the new label-shuffled arm) change; module
    constant R2 is pointed at the new base folder for the new seeds.

Stages (run from the repository root):
  plan        dry run, no SAE training: builds the matrix, the fold row sets and the 5 x 3 subspaces, checks hashes,
              paths, disk; writes plan files (refuses to overwrite).
  repro       (needs approval) re-trains pilot ctrl seed 42 on the pilot A rows into a SMOKE folder and compares it
              bit-for-bit with the pilot checkpoint: proves the wrapper reproduces the frozen recipe.
  base        (needs approval) trains the 5 new base SAEs jointly into a staging folder, renamed on success.
  base_qc     inference-only QC of the new bases against the original 3 (EV window, L0, dead fraction).
  finetune    (needs approval) runs the 120 fine-tunes in seed-major order; resumable via the run ledger.
Every training stage first verifies the frozen pre-registration hashes and refuses to overwrite any checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sae import truth_subspace as ts  # noqa: E402

ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
OUT = ROOT / 'replication'
CK = OUT / 'checkpoints/layer_8'
PREREG = OUT / 'preregistration.json'
LEDGER = OUT / 'run_ledger.csv'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
NEWBASE = REPO / 'results/openwebtext/r2_tspr_base_seeds'
FOLDS_CSV = REPO / 'data/got/fresh/splits/cities_crossfit_folds.csv'
TOK = REPO / 'data/openwebtext/stream_tokens/shard79_chunks.npy'
LAYER, NAME = 8, 'topk'
OLD_SEEDS, NEW_SEEDS = (42, 1042, 2042), (3042, 4042, 5042, 6042, 7042)
SEEDS = OLD_SEEDS + NEW_SEEDS
ARMS = ('ctrl', 'tsp', 'rand', 'pca', 'shuf')
FOLDS = (0, 1, 2)
RANK, SUB_SEED = 8, 20261114
BASE_TOKENS = 99_999_744                      # R2 preregistration tokens_per_sae
MIN_FREE_GB = 20
FROZEN = ['scripts/geometry_of_truth/25_truth_subspace_finetune.py', 'scripts/openwebtext/17_streaming_sae_training.py',
          'src/sae/truth_subspace.py', 'src/sae/modern_sae.py', 'src/evaluation/faithfulness_metrics.py',
          'scripts/geometry_of_truth/24_truth_subspace_diagnostics.py',
          'scripts/geometry_of_truth/03_original_truth_probes.py',
          'data/got/fresh/splits/cities_crossfit_folds.csv', 'data/got/splits/cities_splits.csv',
          'data/got/splits/neg_cities_splits.csv', 'data/openwebtext/stream_tokens/shard79_chunks.npy',
          'data/openwebtext/stream_tokens/shard78_chunks.npy']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_path(seed):
    root = R2 if seed in OLD_SEEDS else NEWBASE / 'checkpoints'
    return root / f'layer_{LAYER}/{NAME}_seed{seed}/checkpoint_final.pt'


def ckpt_path(fold, arm, seed):
    return CK / f'fold{fold}/{arm}_{NAME}_seed{seed}/checkpoint.pt'


def matrix():
    rows = []
    for s in SEEDS:                      # seed-major: an early stop still leaves complete paired seed blocks
        for f in FOLDS:
            for a in ARMS:
                rows.append(dict(run_id=f'f{f}_{a}_s{s}', order=len(rows), seed=s, fold=f, arm=a,
                                 base_seed_batch='r2_original' if s in OLD_SEEDS else 'tspr_new',
                                 base_checkpoint=str(base_path(s).relative_to(REPO)),
                                 checkpoint=str(ckpt_path(f, a, s).relative_to(REPO))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ data / subspaces
def fold_rows():
    """GoT geographic train+val rows (test never loaded) with fold roles. Fold f: V = f, B = f+1, A = f+2."""
    t = mod('scripts/geometry_of_truth/24_truth_subspace_diagnostics.py', 'tspd').load_got([LAYER])
    folds = pd.read_csv(FOLDS_CSV)
    fold_of = dict(zip(folds.group_id.astype(str), folds.fold))
    g = t['groups'].astype(str)
    assert set(g) == set(fold_of), 'fold file must cover exactly the train+val cities groups'
    t['fold'] = np.array([fold_of[x] for x in g])
    return t


def roles(f):
    return dict(V=f, B=(f + 1) % 3, A=(f + 2) % 3)


def shuffled_labels(y, groups, pol, seed):
    """Truth labels permuted within each (group, polarity) cell: per city and polarity the true/false pair is swapped
    with probability 1/2, so the label is decoupled from the truth direction but the rows, groups and estimator match."""
    rng = np.random.default_rng(seed)
    ys = y.copy()
    cells = pd.Series(np.arange(len(y))).groupby([groups, pol]).indices
    for idx in cells.values():
        ys[idx] = y[idx][rng.permutation(len(idx))]
    return ys


def subspaces_for_fold(t, f):
    A = t['fold'] == roles(f)['A']
    XA, yA, pA, gA = t['X'][LAYER][A], t['y'][A], t['pol'][A], t['groups'][A]
    s0 = SUB_SEED + LAYER + 1000 * (f + 1)
    P_tsp, info = ts.estimate_truth_subspace(XA, yA, pA, gA, r=RANK, seed=s0)
    ysh = shuffled_labels(yA, gA, pA, s0 + 1)
    P_shuf, info_s = ts.estimate_truth_subspace(XA, ysh, pA, gA, r=RANK, seed=s0)
    P = dict(tsp=P_tsp, rand=ts.random_subspace(XA.shape[1], RANK, s0 + 2), pca=ts.pca_subspace(XA, RANK), shuf=P_shuf)
    meta = dict(fold=f, roles=roles(f), A_rows=int(A.sum()), A_groups=int(len(set(gA))), subspace_seed=s0,
                shuf_perm_seed=s0 + 1, rand_seed=s0 + 2, tsp_info=info, shuf_info=info_s,
                shuf_label_agreement=float((ysh == yA).mean()))
    return P, XA, meta


def overlaps(P):
    """Mean squared principal-angle cosine between subspaces (1 = identical, r/d = 0.0104 = chance)."""
    k = list(P)
    return {f'{a}~{b}': round(float(np.linalg.norm(P[a].T @ P[b]) ** 2 / RANK), 4)
            for i, a in enumerate(k) for b in k[i + 1:]}


# ------------------------------------------------------------------------------------------------ stages
def plan():
    refuse = [OUT / 'plan_matrix.csv', OUT / 'subspaces.npz', OUT / 'plan_info.json']
    for p in refuse:
        if p.exists():
            sys.exit(f'refusing to overwrite {p}')
    OUT.mkdir(parents=True, exist_ok=True)
    M = matrix()
    assert len(M) == 120 and M.run_id.is_unique and M.checkpoint.is_unique
    exist = [c for c in M.checkpoint if (REPO / c).exists()]
    assert not exist, f'checkpoints already exist: {exist[:3]}'
    assert not NEWBASE.exists(), f'{NEWBASE} exists'
    old = {s: sha(base_path(s)) for s in OLD_SEEDS}
    t = fold_rows()
    fold_info, P_all, cross = [], {}, {}
    for f in FOLDS:
        P, XA, meta = subspaces_for_fold(t, f)
        r = roles(f)
        for role, k in r.items():
            meta[f'{role}_groups'] = int(len(set(t['groups'][t['fold'] == k])))
            meta[f'{role}_rows'] = int((t['fold'] == k).sum())
        meta['subspace_overlaps'] = overlaps(P)
        fold_info.append(meta)
        for a, v in P.items():
            P_all[f'fold{f}_{a}'] = v
    for a in ('tsp', 'shuf', 'pca'):
        cross[a] = {f'fold{i}~fold{j}': round(float(np.linalg.norm(P_all[f'fold{i}_{a}'].T @ P_all[f'fold{j}_{a}']) ** 2
                                                    / RANK), 4) for i in FOLDS for j in FOLDS if i < j}
    # Pilot S overlap (pilot A = half A of the original train split) for reference.
    pilot = np.load(ROOT / 'pilot/subspaces.npz')
    cross['tsp_vs_pilot'] = {f'fold{f}': round(float(np.linalg.norm(P_all[f'fold{f}_tsp'].T @ pilot['tsp']) ** 2 / RANK), 4)
                             for f in FOLDS}
    np.savez(OUT / 'subspaces.npz', **P_all)
    M.to_csv(OUT / 'plan_matrix.csv', index=False)
    free_gb = shutil.disk_usage(REPO).free / 2 ** 30
    ck_mb = (REPO / M.base_checkpoint.iloc[0]).stat().st_size / 2 ** 20
    info = dict(stage='plan (dry run; no SAE trained)', runs=len(M), seeds=list(SEEDS), new_seeds=list(NEW_SEEDS),
                arms=list(ARMS), folds=list(FOLDS), fold_info=fold_info, cross_fold_subspace_overlap=cross,
                existing_base_sha256=old, disk_free_gb=round(free_gb, 1), checkpoint_mb=round(ck_mb, 1),
                expected_new_storage_gb=round((120 + 5) * ck_mb / 1024, 2),
                frozen_hashes={p: sha(REPO / p) for p in FROZEN}, script_sha256=sha(__file__),
                subspaces_sha256=sha(OUT / 'subspaces.npz'), test_rows_loaded=0)
    (OUT / 'plan_info.json').write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps({k: info[k] for k in ('runs', 'disk_free_gb', 'expected_new_storage_gb')}),
          json.dumps(fold_info, default=float)[:1500], json.dumps(cross), sep='\n')


def check_frozen():
    pr = json.loads(PREREG.read_text())
    for rel, h in pr['hashes'].items():
        if sha(REPO / rel) != h:
            sys.exit(f'{rel}: hash differs from the frozen pre-registration; refusing to run.')
    if shutil.disk_usage(REPO).free / 2 ** 30 < MIN_FREE_GB:
        sys.exit('less than 20 GB free; refusing to run.')
    return pr


def gpt2(device):
    import torch
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    h = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    torch.backends.cuda.matmul.allow_tf32 = True
    return model, h


def save_atomic(obj, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def repro():
    import torch
    check_frozen()
    out = OUT / 'smoke_repro/ctrl_topk_seed42.pt'
    if out.exists():
        sys.exit(f'refusing to overwrite {out}')
    m25 = mod('scripts/geometry_of_truth/25_truth_subspace_finetune.py', 'tsp25')
    t = mod('scripts/geometry_of_truth/24_truth_subspace_diagnostics.py', 'tspd').load_got([LAYER])
    A = (t['split'] == 'train') & (t['half'] == 'A')
    model, _ = gpt2('cuda')
    chunks = np.load(TOK, mmap_mode='r')[m25.START_CHUNK:]
    steps = m25.TOKENS // (m25.SEQ_BATCH * 127)
    sae, cfg, log = m25.run_one(model, 42, 'ctrl', None, t['X'][LAYER][A], chunks, 'cuda', steps)
    save_atomic(dict(model_state_dict=sae.state_dict(), config=cfg), out)
    ref = torch.load(ROOT / 'pilot/checkpoints/layer_8/ctrl_topk_seed42/checkpoint.pt', map_location='cpu',
                     weights_only=False)['model_state_dict']
    diff = max(float((sae.state_dict()[k] - ref[k]).abs().max()) for k in ref)
    (OUT / 'smoke_repro/repro.json').write_text(json.dumps(dict(max_abs_weight_diff=diff, ok=diff == 0.0), indent=2))
    print('repro max |diff|', diff)
    if diff != 0.0:
        sys.exit('wrapper does not reproduce the pilot recipe bit-for-bit; stop.')


def base():
    check_frozen()
    staging = NEWBASE.with_name(NEWBASE.name + '_staging')
    if NEWBASE.exists():
        sys.exit(f'{NEWBASE} exists; refusing to overwrite.')
    if staging.exists():
        sys.exit(f'{staging} exists (interrupted run?): move it aside manually, then rerun from scratch.')
    m17 = mod('scripts/openwebtext/17_streaming_sae_training.py', 'r2train')
    m17.SEEDS = NEW_SEEDS
    m17.train(staging, (LAYER,), [('topk', 'topk', m17.WIDTH, NEW_SEEDS)], BASE_TOKENS, True)
    os.replace(staging, NEWBASE)


def base_qc():
    """Inference-only QC of the new bases against the 3 original R2 TopK L8 seeds, on the identical monitor set
    (shard 79, first 800 chunks) logged by the same training code. Pass: final EV within +-0.005 of the original
    range, L0 = 128, dead fraction <= 0.01, GPT-2 unchanged."""
    if (OUT / 'base_qc.json').exists():
        sys.exit('base_qc.json exists; refusing to overwrite.')
    def final(csv):
        d = pd.read_csv(csv)
        m = d[(d.monitor == True) & (d.layer == LAYER) & (d.name == NAME)]  # noqa: E712
        return m[m.step == m.step.max()].set_index('seed')[['ev', 'l0', 'dead']]
    old = final(REPO / 'results/openwebtext/r2_scaled_saes/training_monitor.csv').loc[list(OLD_SEEDS)]
    new = final(NEWBASE / 'training_monitor.csv').loc[list(NEW_SEEDS)]
    info = json.loads((NEWBASE / 'training_info.json').read_text())
    lo, hi = old.ev.min() - 0.005, old.ev.max() + 0.005
    ok = bool(new.ev.between(lo, hi).all() and (new.l0 == 128).all() and (new.dead <= 0.01).all()
              and info['gpt2_parameters_unchanged'] and info['tokens_per_sae'] == BASE_TOKENS)
    out = dict(old=old.reset_index().to_dict('records'), new=new.reset_index().to_dict('records'), ev_window=[lo, hi],
               tokens_per_sae=info['tokens_per_sae'], gpt2_unchanged=info['gpt2_parameters_unchanged'],
               new_sha256={s: sha(base_path(s)) for s in NEW_SEEDS}, **{'pass': ok})
    (OUT / 'base_qc.json').write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


def finetune():
    import torch
    pr = check_frozen()
    for s in NEW_SEEDS:
        if not base_path(s).exists():
            sys.exit(f'missing new base {base_path(s)}; run the base stage first.')
    qc = json.loads((OUT / 'base_qc.json').read_text()) if (OUT / 'base_qc.json').exists() else None
    if qc is None or not qc.get('pass'):
        sys.exit('base_qc.json missing or failed; the base QC must pass before fine-tuning.')
    M = pd.read_csv(OUT / 'plan_matrix.csv')
    led = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=['run_id', 'sha256'])
    done = dict(zip(led.run_id, led.sha256))
    m25 = mod('scripts/geometry_of_truth/25_truth_subspace_finetune.py', 'tsp25')
    t = fold_rows()
    P_all = np.load(OUT / 'subspaces.npz')
    assert sha(OUT / 'subspaces.npz') == pr['subspaces_sha256']
    model, h0 = gpt2('cuda')
    chunks = np.load(TOK, mmap_mode='r')[m25.START_CHUNK:]
    steps = m25.TOKENS // (m25.SEQ_BATCH * 127)
    t0 = time.time()
    for r in M.itertuples():
        p = REPO / r.checkpoint
        if r.run_id in done:
            if not p.exists() or sha(p) != done[r.run_id]:
                sys.exit(f'{r.run_id}: ledger hash mismatch or missing file; manual review required.')
            continue
        if p.exists():
            sys.exit(f'{p} exists without a ledger entry; manual review required (never overwritten).')
        A = t['fold'] == roles(r.fold)['A']
        m25.R2 = R2 if r.seed in OLD_SEEDS else NEWBASE / 'checkpoints'
        P = None if r.arm == 'ctrl' else P_all[f'fold{r.fold}_{r.arm}']
        ts0 = time.time()
        sae, cfg, log = m25.run_one(model, int(r.seed), r.arm, P, t['X'][LAYER][A], chunks, 'cuda', steps)
        last = log[-1]
        if not (abs(last['l0_owt'] - 128) < 1e-6 and abs(last['l0_got'] - 128) < 1e-6):
            sys.exit(f'{r.run_id}: L0 != 128; stopping rule.')
        cfg.update(procedure='tspr_phase1_2026-11-15', fold=int(r.fold), fold_roles=roles(r.fold),
                   a_rows=int(A.sum()), subspace_key=None if P is None else f'fold{r.fold}_{r.arm}',
                   subspaces_sha256=pr['subspaces_sha256'])
        save_atomic(dict(model_state_dict=sae.state_dict(), config=cfg, monitor=log), p)
        row = dict(run_id=r.run_id, seed=r.seed, fold=r.fold, arm=r.arm, checkpoint=r.checkpoint, sha256=sha(p),
                   runtime_s=round(time.time() - ts0, 1), **{k: last[k] for k in ('nmse_owt', 'nmse_got', 'sub',
                                                                                 'l0_owt', 'l0_got')},
                   finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        pd.DataFrame([row]).to_csv(LEDGER, mode='a', header=not LEDGER.exists(), index=False)
        print(f'[{r.order + 1}/120] {r.run_id} nmse_owt {last["nmse_owt"]:.4f} sub {last["sub"]:.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if shutil.disk_usage(REPO).free / 2 ** 30 < MIN_FREE_GB:
            sys.exit('disk below 20 GB; stopping.')
    h1 = hashlib.sha256(b''.join(q.detach().cpu().numpy().tobytes() for q in model.parameters())).hexdigest()
    if h0 != h1:
        sys.exit('GPT-2 parameters changed; stopping rule.')
    print('all 120 fine-tunes complete')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['plan', 'repro', 'base', 'base_qc', 'finetune'])
    {'plan': plan, 'repro': repro, 'base': base, 'base_qc': base_qc, 'finetune': finetune}[ap.parse_args().stage]()
