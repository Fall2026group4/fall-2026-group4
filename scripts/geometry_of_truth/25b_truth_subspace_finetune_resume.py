#!/usr/bin/env python3
"""TSP pilot RECOVERY wrapper (2026-11-14). Completes the runs missing after the interrupted pilot.

The pilot (script 25) was interrupted after 9 of 12 fine-tunes when the Claude session ended (~17:56 UTC); the
missing runs were seed 2042 x {tsp, rand, pca}. This wrapper does NOT modify script 25: it imports it unchanged
(hash-checked against the frozen pre-registration) and calls its own ``run_one`` with its own constants, so the
missing runs use the exact frozen configuration.

Before training it verifies:
  1. every hash in preregistration.json (README excepted) matches, including script 25 and src/sae/truth_subspace.py;
  2. the saved pilot/subspaces.npz equals a fresh recomputation of the subspaces from group A (determinism);
  3. each of the 9 existing checkpoints has the frozen configuration (arm, lr, tokens, mu, lam, rank, subspace seed,
     procedure, source checkpoint and its sha256);
  4. (``--verify-repro``) re-training one completed run (ctrl, seed 2042) into a scratch directory reproduces the
     stored checkpoint (max |weight difference| reported; GPU kernels may be non-deterministic).
Existing checkpoints are never overwritten. GPT-2 parameter hashes are checked before and after.
Writes pilot/finetune_info.json (with a recovery note) and pilot/finetune_monitor_recovery.csv.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
PILOT = ROOT / 'pilot'
FROZEN = REPO / 'scripts/geometry_of_truth/25_truth_subspace_finetune.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_frozen():
    pr = json.loads((ROOT / 'preregistration.json').read_text())
    for rel, h in pr['hashes'].items():
        if not rel.endswith('README.md') and sha(REPO / rel) != h:
            raise SystemExit(f'{rel}: hash differs from the frozen pre-registration; refusing to run.')
    spec = importlib.util.spec_from_file_location('tsp25', FROZEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, pr


def ckpt_path(m, arm, seed, root=PILOT):
    return root / f'checkpoints/layer_{m.LAYER}/{arm}_{m.NAME}_seed{seed}/checkpoint.pt'


def expected_cfg(m, arm, seed):
    src = m.R2 / f'layer_{m.LAYER}/{m.NAME}_seed{seed}/checkpoint_final.pt'
    steps = m.TOKENS // (m.SEQ_BATCH * 127)
    return dict(finetune_arm=arm, finetune_tokens=steps * m.SEQ_BATCH * 127, finetune_lr=m.LR, mu=m.MU,
                lam=(0.0 if arm == 'ctrl' else m.LAM), subspace_rank=(0 if arm == 'ctrl' else m.RANK),
                subspace_seed=m.SUB_SEED + m.LAYER, procedure='tsp_pilot_2026-11-14',
                source_checkpoint=str(src.relative_to(REPO)), source_sha256=sha(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-repro', type=Path, help='scratch dir: retrain ctrl seed 2042 and compare')
    a = ap.parse_args()
    m, pr = load_frozen()
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    report = dict(frozen_hashes_ok=True, frozen_script_sha256=sha(FROZEN))
    # ---- 2. subspaces
    t = m._diag().load_got([m.LAYER])
    A = (t['split'] == 'train') & (t['half'] == 'A')
    XA = t['X'][m.LAYER][A]
    P, info = m.subspaces(XA, t['y'][A], t['pol'][A], t['groups'][A])
    saved = np.load(PILOT / 'subspaces.npz')
    diffs = {k: float(np.abs(saved[k] - P[k]).max()) for k in P}
    if max(diffs.values()) > 1e-10:
        raise SystemExit(f'subspaces differ from the saved pilot subspaces: {diffs}')
    report['subspace_max_abs_diff'] = diffs
    # ---- 3. existing checkpoints
    done, missing = [], []
    for seed in m.SEEDS:
        for arm in m.ARMS:
            p = ckpt_path(m, arm, seed)
            if p.exists():
                cfg = torch.load(p, map_location='cpu', weights_only=False)['config']
                exp = expected_cfg(m, arm, seed)
                bad = {k: (cfg.get(k), v) for k, v in exp.items() if cfg.get(k) != v}
                if bad:
                    raise SystemExit(f'{p}: configuration differs from the frozen one: {bad}')
                done.append((seed, arm))
            else:
                missing.append((seed, arm))
    report.update(existing_runs=[f'{a_}_seed{s}' for s, a_ in done], missing_runs=[f'{a_}_seed{s}' for s, a_ in missing])
    print(f'existing {len(done)} (config verified), missing {missing}', flush=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    before = hashlib.sha256(b''.join(p_.detach().cpu().numpy().tobytes() for p_ in model.parameters())).hexdigest()
    chunks = np.load(m.TOK, mmap_mode='r')[m.START_CHUNK:]
    steps = m.TOKENS // (m.SEQ_BATCH * 127)
    # ---- 4. reproduction check of one completed run (scratch only)
    if a.verify_repro:
        sae, cfg, log = m.run_one(model, 2042, 'ctrl', None, XA, chunks, device, steps)
        ref = torch.load(ckpt_path(m, 'ctrl', 2042), map_location='cpu', weights_only=False)['model_state_dict']
        mx = max(float((sae.state_dict()[k] - ref[k]).abs().max()) for k in ref)
        report['repro_ctrl_seed2042'] = dict(max_abs_weight_diff=mx, final_nmse_owt=log[-1]['nmse_owt'],
                                            final_nmse_got=log[-1]['nmse_got'])
        print(f'repro ctrl seed2042: max |dW| = {mx:.3e}; nmse_owt {log[-1]["nmse_owt"]:.4f} '
              f'nmse_got {log[-1]["nmse_got"]:.4f}', flush=True)
        a.verify_repro.mkdir(parents=True, exist_ok=True)
        (a.verify_repro / 'repro_check.json').write_text(json.dumps(report['repro_ctrl_seed2042'], indent=2))
    # ---- training of the missing runs
    logs, ckpts = [], {}
    for seed, arm in missing:
        p = ckpt_path(m, arm, seed)
        if p.exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
        sae, cfg, log = m.run_one(model, seed, arm, None if arm == 'ctrl' else P[arm], XA, chunks, device, steps)
        cfg['recovery'] = 'trained by 25b_truth_subspace_finetune_resume.py after the interrupted pilot'
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
        logs += log
        print(f'seed {seed} {arm}: nmse_owt {log[-1]["nmse_owt"]:.4f} nmse_got {log[-1]["nmse_got"]:.4f} '
              f'sub {log[-1]["sub"]:.4f} l0 {log[-1]["l0_owt"]:.1f}/{log[-1]["l0_got"]:.1f} ({time.time() - t0:.0f}s)',
              flush=True)
    after = hashlib.sha256(b''.join(p_.detach().cpu().numpy().tobytes() for p_ in model.parameters())).hexdigest()
    for seed in m.SEEDS:
        for arm in m.ARMS:
            p = ckpt_path(m, arm, seed)
            ckpts[str(p.relative_to(REPO))] = sha(p)
    import pandas as pd
    pd.DataFrame(logs).to_csv(PILOT / 'finetune_monitor_recovery.csv', index=False)
    if (PILOT / 'finetune_info.json').exists():
        raise SystemExit('finetune_info.json exists; refusing to overwrite.')
    (PILOT / 'finetune_info.json').write_text(json.dumps(dict(
        layer=m.LAYER, name=m.NAME, seeds=list(m.SEEDS), arms=list(m.ARMS), steps=steps,
        tokens_per_run=steps * m.SEQ_BATCH * 127, lr=m.LR, mu=m.MU, lam=m.LAM, rank=m.RANK, subspace_info=info,
        n_got_A_rows=int(A.sum()), test_rows_used=0, gpt2_parameters_unchanged=(before == after), checkpoints=ckpts,
        recovery=dict(note='Pilot interrupted after 9/12 runs (session end ~17:56 UTC 2026-11-14). The 9 existing '
                           'checkpoints were kept unchanged; the 3 missing runs were trained by script 25b, which '
                           'calls the hash-verified frozen script 25 run_one. Monitor rows of the 9 interrupted runs '
                           'were held in memory and lost; their final values survive in the pilot log.', **report),
        runtime_s_recovery=time.time() - t0, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9), indent=2))


if __name__ == '__main__':
    main()
