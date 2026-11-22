#!/usr/bin/env python3
"""TSP pilot: truth-subspace-preserving fine-tune of R2 TopK-12288 at fixed L0 (2026-11-14). VALIDATION-only study.

Arms (identical tokens, steps, lr, seeds, base checkpoint; they differ ONLY in the subspace term):
  ctrl : L = NMSE_OWT + mu * NMSE_GoT-A                                  (Method-A analogue on the R2 base)
  tsp  : ctrl + lam * ||P_S^T (x - x_hat)||^2 / ||P_S^T (x - mean_A)||^2  on GoT-A rows, S = truth subspace from A
  rand : ctrl + lam * same term with a Haar-random subspace of the same rank (geometry-free control)
  pca  : ctrl + lam * same term with the top principal subspace of GoT-A (label-free, high-variance control)
TopK keeps L0 = 128 exactly on OWT, GoT and RepE prompts (sparsity matched by construction).

Recipe copied from R4 (scripts/openwebtext/18_kl_finetune_fixed_l0.py): R2 normalised coordinates (scale unfolded,
then folded back), Adam lr 2e-5, clip 1, decoder renormalised every step, 32 x 127 OWT tokens per step from shard 79
chunks >= 10,000 (the same 5M tokens R4 used). GoT-A rows (train group A of
data/got/splits/train_AB_groups_truth_aware_pilot.csv) are the ONLY task rows used; group B, validation and test are
never seen in training. The subspace basis and its seed are saved with each checkpoint.

Outputs: results/geometry_of_truth/truth_subspace_preservation/pilot/ (refuses to overwrite; checks frozen hashes).
"""
import argparse
import copy
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
from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae import truth_subspace as ts  # noqa: E402

ROOT = REPO / 'results/geometry_of_truth/truth_subspace_preservation'
OUT = ROOT / 'pilot'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
TOK = REPO / 'data/openwebtext/stream_tokens/shard79_chunks.npy'
LAYER, NAME, SEEDS, ARMS = 8, 'topk', (42, 1042, 2042), ('ctrl', 'tsp', 'rand', 'pca')
SEQ_BATCH, TOKENS, LR, START_CHUNK = 32, 5_000_000, 2e-5, 10_000
RANK, MU, LAM, SUB_SEED = 8, 1.0, 1.0, 20261114


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _diag():
    spec = importlib.util.spec_from_file_location('tspd', REPO / 'scripts/geometry_of_truth/24_truth_subspace_diagnostics.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def subspaces(XA, yA, polA, gA):
    P, info = ts.estimate_truth_subspace(XA, yA, polA, gA, r=RANK, seed=SUB_SEED + LAYER)
    r = P.shape[1]
    return dict(tsp=P, rand=ts.random_subspace(XA.shape[1], r, SUB_SEED + LAYER), pca=ts.pca_subspace(XA, r)), info


def run_one(model, seed, arm, P, XA, chunks, device, steps):
    src = R2 / f'layer_{LAYER}/{NAME}_seed{seed}/checkpoint_final.pt'
    sae, ck = fm.load_sae(src, device)
    s = ck['config']['input_scale_folded']
    sae.fold_scale(1.0 / s)
    sae.train()
    torch.manual_seed(seed)
    params = [p for p in sae.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    hook = f'blocks.{LAYER}.hook_resid_pre'
    xa = torch.as_tensor(XA, device=device) * s
    xa_c = xa - xa.mean(0)
    got_den = xa_c.pow(2).sum()
    Pt = None if P is None else torch.as_tensor(P, dtype=torch.float32, device=device)
    sub_den = None if P is None else (xa_c @ Pt).pow(2).sum(-1).mean()
    log = []
    for step in range(steps):
        toks = torch.as_tensor(chunks[step * SEQ_BATCH:(step + 1) * SEQ_BATCH].astype(np.int64), device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook, stop_at_layer=LAYER + 1)
            x = cache[hook][:, 1:].float().reshape(-1, xa.shape[1]) * s
        rec, z, _ = sae(x)
        nmse_owt = (rec - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        reca, za, _ = sae(xa)
        nmse_got = (reca - xa).pow(2).sum() / got_den
        loss = nmse_owt + MU * nmse_got
        sub = torch.zeros((), device=device)
        if Pt is not None:
            sub = ts.subspace_loss(xa, reca, Pt, sub_den)
            loss = loss + LAM * sub
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sae.normalize_decoder()
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step} ({arm}, seed {seed}); stopping.')
        if step % 100 == 0 or step == steps - 1:
            log.append(dict(seed=seed, arm=arm, step=step, nmse_owt=float(nmse_owt), nmse_got=float(nmse_got),
                            sub=float(sub), l0_owt=float((z > 0).float().sum(-1).mean()),
                            l0_got=float((za > 0).float().sum(-1).mean())))
    sae.eval()
    out = copy.deepcopy(sae).cpu()
    out.fold_scale(s)
    cfg = dict(ck['config'], finetune_arm=arm, finetune_tokens=steps * SEQ_BATCH * 127, finetune_lr=LR, mu=MU,
               lam=(0.0 if P is None else LAM), subspace_rank=(0 if P is None else int(P.shape[1])),
               subspace_seed=SUB_SEED + LAYER, source_checkpoint=str(src.relative_to(REPO)), source_sha256=sha(src),
               procedure='tsp_pilot_2026-11-14')
    return out, cfg, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    out_dir = a.smoke or OUT
    if not a.smoke:
        pr = json.loads((ROOT / 'preregistration.json').read_text())
        for rel, h in pr['hashes'].items():
            if not rel.endswith('README.md') and sha(REPO / rel) != h:
                raise SystemExit(f'{rel}: hash differs from the frozen pre-registration; refusing to run.')
    if (out_dir / 'finetune_info.json').exists():
        raise SystemExit('outputs exist; refusing to overwrite.')
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    t = _diag().load_got([LAYER])
    A = (t['split'] == 'train') & (t['half'] == 'A')
    XA = t['X'][LAYER][A]
    P, info = subspaces(XA, t['y'][A], t['pol'][A], t['groups'][A])
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    chunks = np.load(TOK, mmap_mode='r')[START_CHUNK:]
    steps = (TOKENS // (SEQ_BATCH * 127)) if not a.smoke else 30
    seeds = (42,) if a.smoke else SEEDS
    logs, ckpts = [], {}
    np.savez(out_dir / 'subspaces.npz' if out_dir.mkdir(parents=True, exist_ok=True) is None else None, **P)
    for seed in seeds:
        for arm in ARMS:
            sae, cfg, log = run_one(model, seed, arm, None if arm == 'ctrl' else P[arm], XA, chunks, device, steps)
            p = out_dir / f'checkpoints/layer_{LAYER}/{arm}_{NAME}_seed{seed}/checkpoint.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise SystemExit(f'{p} exists; refusing to overwrite.')
            torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
            ckpts[str(p.relative_to(REPO)) if not a.smoke else str(p)] = sha(p)
            logs += log
            print(f'seed {seed} {arm}: nmse_owt {log[-1]["nmse_owt"]:.4f} nmse_got {log[-1]["nmse_got"]:.4f} '
                  f'sub {log[-1]["sub"]:.4f} l0 {log[-1]["l0_owt"]:.1f}/{log[-1]["l0_got"]:.1f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    after = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    import pandas as pd
    pd.DataFrame(logs).to_csv(out_dir / 'finetune_monitor.csv', index=False)
    (out_dir / 'finetune_info.json').write_text(json.dumps(dict(
        layer=LAYER, name=NAME, seeds=list(seeds), arms=list(ARMS), steps=steps, tokens_per_run=steps * SEQ_BATCH * 127,
        lr=LR, mu=MU, lam=LAM, rank=RANK, subspace_info=info, n_got_A_rows=int(A.sum()), test_rows_used=0,
        gpt2_parameters_unchanged=(before == after), checkpoints=ckpts, runtime_s=time.time() - t0,
        gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9), indent=2))


if __name__ == '__main__':
    main()
