#!/usr/bin/env python3
"""R4: short KL+MSE fine-tune of R2 TopK SAEs at fixed L0, with an MSE-only control (2026-11-10).

Pre-registered in results/openwebtext/r4_kl_finetune/README.md. Method after Karvonen 2025 (arXiv 2503.17272):
a short end-to-end fine-tune on ~5 % of the training budget. TopK keeps L0 = 128 exactly, which removes the L0-drift
confound that made our IMP-L output-KL results for ReLU/JumpReLU not creditable.

Arms (same tokens, same steps, same lr):
  kl   : loss = NMSE(x, x_hat) + alpha * KL(p_orig || p_spliced), SAE spliced at every non-BOS position of layer L
  mse  : loss = NMSE(x, x_hat)                                   (continued-training control)
alpha is set per SAE so that alpha * KL = NMSE on the first batch (scale matching, as in IMP-L).
Data: OpenWebText shard 79 chunks from index 10,000 on (disjoint from R2 training shard 78, the R2 monitor slice
[:800] and our validation/test documents). Training happens in the normalised coordinates of R2 (scale unfolded,
then folded back).

Run: python scripts/openwebtext/18_kl_finetune_fixed_l0.py [--smoke DIR]
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import faithfulness_metrics as fm  # noqa: E402

OUT = REPO / 'results/openwebtext/r4_kl_finetune'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
TOK = REPO / 'data/openwebtext/stream_tokens/shard79_chunks.npy'
LAYERS, NAMES, SEEDS, ARMS = (8, 11), ('topk',), (42, 1042, 2042), ('kl', 'mse')
SEQ_BATCH, TOKENS, LR, START_CHUNK = 32, 5_000_000, 2e-5, 10_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_one(model, L, name, seed, arm, chunks, device, smoke):
    src = R2 / f'layer_{L}/{name}_seed{seed}/checkpoint_final.pt'
    sae, ck = fm.load_sae(src, device)
    s = ck['config']['input_scale_folded']
    sae.fold_scale(1.0 / s)                      # back to the normalised training coordinates
    sae.train()
    params = [p for p in sae.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    hook = f'blocks.{L}.hook_resid_pre'
    steps = (TOKENS // (SEQ_BATCH * 127)) if not smoke else 20
    alpha, log = None, []
    for step in range(steps):
        toks = torch.as_tensor(chunks[step * SEQ_BATCH:(step + 1) * SEQ_BATCH].astype(np.int64), device=device)
        with torch.no_grad():
            logits_o, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook)
            lp_o = logits_o[:, 1:].float().log_softmax(-1)           # predictions after the BOS position
            x = cache[hook][:, 1:].float()
        xs = x * s
        rec, z, _ = sae(xs.reshape(-1, xs.shape[-1]))
        nmse = (rec - xs.reshape(-1, xs.shape[-1])).pow(2).sum() / (xs - xs.mean((0, 1))).pow(2).sum()
        loss = nmse
        kl = torch.zeros((), device=device)
        if arm == 'kl':
            resid = cache[hook].clone()
            resid = torch.cat([resid[:, :1], (rec / s).view_as(x).to(resid.dtype)], 1)
            logits_s = model(resid, start_at_layer=L)
            lp_s = logits_s[:, 1:].float().log_softmax(-1)
            kl = (lp_o.exp() * (lp_o - lp_s)).sum(-1).mean()
            if alpha is None:
                alpha = float(nmse.detach() / kl.detach().clamp_min(1e-8))
            loss = nmse + alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sae.normalize_decoder()
        if step % 100 == 0 or step == steps - 1:
            log.append(dict(layer=L, name=name, seed=seed, arm=arm, step=step, nmse=float(nmse), kl=float(kl),
                            alpha=alpha, l0=float((z > 0).float().sum(-1).mean())))
    sae.eval()
    out = copy.deepcopy(sae).cpu()
    out.fold_scale(s)
    cfg = dict(ck['config'], finetune_arm=arm, finetune_tokens=steps * SEQ_BATCH * 127, finetune_lr=LR,
               finetune_alpha=alpha, source_checkpoint=str(src.relative_to(REPO)), source_sha256=sha(src),
               procedure='r4_kl_finetune_2026-11-10')
    return out, cfg, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    out_dir = a.smoke or OUT
    if not a.smoke:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if not r_.endswith('README.md') and sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs; refusing to run.')
    if (out_dir / 'finetune_info.json').exists():
        raise SystemExit('outputs exist; refusing to overwrite.')
    device, t0 = 'cuda', time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    chunks = np.load(TOK, mmap_mode='r')[START_CHUNK:]
    logs, ckpts = [], {}
    layers = (8,) if a.smoke else LAYERS
    seeds = (42,) if a.smoke else SEEDS
    for L in layers:
        for name in NAMES:
            for seed in seeds:
                for arm in ARMS:
                    sae, cfg, log = run_one(model, L, name, seed, arm, chunks, device, bool(a.smoke))
                    p = out_dir / f'checkpoints/layer_{L}/{arm}_{name}_seed{seed}/checkpoint.pt'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if p.exists():
                        raise SystemExit(f'{p} exists; refusing to overwrite.')
                    torch.save(dict(model_state_dict=sae.state_dict(), config=cfg), p)
                    ckpts[str(p.relative_to(REPO)) if not a.smoke else str(p)] = sha(p)
                    logs += log
                    print(f'L{L} {name} {seed} {arm}: final nmse {log[-1]["nmse"]:.4f} kl {log[-1]["kl"]:.4f} '
                          f'l0 {log[-1]["l0"]:.1f} ({time.time() - t0:.0f}s)', flush=True)
    after = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    pd.DataFrame(logs).to_csv(out_dir / 'finetune_monitor.csv', index=False)
    (out_dir / 'finetune_info.json').write_text(json.dumps(dict(
        script_sha256=sha(__file__), checkpoints=ckpts, runtime_s=time.time() - t0, gpt2_parameters_unchanged=before == after,
        tokens_per_run=(TOKENS // (SEQ_BATCH * 127)) * SEQ_BATCH * 127, lr=LR, data=f'shard 79 chunks from {START_CHUNK}',
        smoke=bool(a.smoke)), indent=2))


if __name__ == '__main__':
    main()
