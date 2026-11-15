#!/usr/bin/env python3
"""R2: data-scaled SAE training on streamed GPT-2 Small residual activations (2026-11-10).

Problem: our original SAEs saw 101,600 unique tokens (about 38 passes) at width 3072. Published GPT-2 SAEs use
hundreds of millions of tokens (res-jb: 300M, 24,576 latents). This script trains SAEs on about 100M fresh
OpenWebText tokens with published recipes (src/sae/modern_sae.py), at target L0 = 128 (the matched value of our study).

Data: OpenWebText parquet shard 78 (training) and shard 79 (training monitor only). Documents whose text matches any
of our 1000 sampled documents (data/openwebtext/{train,validation,test}.parquet) are excluded, by exact and
normalised-prefix hash. Each document is tokenised, split into 127-token chunks and prefixed with BOS; BOS is
excluded from SAE training. Our OWT validation/test caches are never read here.

Activations for layers 1, 8 and 11 (blocks.L.hook_resid_pre) come from one GPT-2 forward pass, go into a shuffled
fp16 GPU buffer per layer, and each seed draws its own batches from it (architectures of the same seed share batches).

Run from the repository root:
    python scripts/openwebtext/17_streaming_sae_training.py --prepare-data
    python scripts/openwebtext/17_streaming_sae_training.py --pilot OUTDIR        (layer 8, seed 42, 5M tokens)
    python scripts/openwebtext/17_streaming_sae_training.py --run                 (pre-registered configuration)
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sae.modern_sae import ModernSAE  # noqa: E402

SHARDS = REPO / 'data/openwebtext/stream_shards/plain_text'
TOK = REPO / 'data/openwebtext/stream_tokens'
OUT = REPO / 'results/openwebtext/r2_scaled_saes'
CKPT = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints'
TRAIN_SHARD, MONITOR_SHARD = 78, 79
CTX, BOS, D = 128, 50256, 768
LAYERS, SEEDS = (1, 8, 11), (42, 1042, 2042)
TARGET_L0, WIDTH = 128, 12288
# (name, architecture, width, seeds)
CONFIGS = [('topk', 'topk', WIDTH, SEEDS), ('batchtopk', 'batchtopk', WIDTH, SEEDS),
           ('relu', 'relu', WIDTH, SEEDS),   # modern JumpReLU abandoned in the pilot (README 1.2)
           ('topk_w3072', 'topk', 3072, SEEDS), ('topk_frozen_decoder', 'topk_frozen_decoder', WIDTH, (42,))]
BATCH, LR, WARMUP, DECAY_FRAC, CLIP = 4096, 2e-4, 500, 0.2, 1.0
BUFFER, LM_BATCH = 1 << 19, 128
AUX_ALPHA, AUX_K, DEAD_TOKENS = 1 / 32, 512, 5_000_000
CTRL_ETA = 0.01
THR_LR_MULT = 1.0
JUMP_EPS = 0.05
LAM_MIN, LAM_MAX = 1e-3, 1e3
SAVE_AT = (10_000_000, 25_000_000, 50_000_000)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ data
def _norm(t):
    return re.sub(r'\s+', ' ', t).strip().lower()


def excluded_hashes():
    ex, pre = set(), set()
    for s in ('train', 'validation', 'test'):
        for t in pd.read_parquet(REPO / f'data/openwebtext/{s}.parquet').text:
            ex.add(hashlib.sha1(t.encode()).hexdigest())
            pre.add(hashlib.sha1(_norm(t)[:300].encode()).hexdigest())
    return ex, pre


def prepare_data():
    from transformers import AutoTokenizer
    TOK.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained('gpt2')
    ex, pre = excluded_hashes()
    info = {}
    for shard in (TRAIN_SHARD, MONITOR_SHARD):
        out = TOK / f'shard{shard}_chunks.npy'
        if out.exists():
            raise SystemExit(f'{out} exists; refusing to overwrite.')
        df = pd.read_parquet(SHARDS / f'train-{shard:05d}-of-00080.parquet')
        keep = [not (hashlib.sha1(t.encode()).hexdigest() in ex or hashlib.sha1(_norm(t)[:300].encode()).hexdigest() in pre)
                for t in df.text]
        texts = df.text[keep].tolist()
        rng = np.random.default_rng(20261110 + shard)
        order = rng.permutation(len(texts))
        chunks = []
        for b in range(0, len(order), 2000):
            enc = tok([texts[i] for i in order[b:b + 2000]])['input_ids']
            for ids in enc:
                for c in range(0, len(ids) - (CTX - 1) + 1, CTX - 1):
                    chunks.append([BOS] + ids[c:c + CTX - 1])
        arr = np.asarray(chunks, dtype=np.uint16)
        arr = arr[rng.permutation(len(arr))]
        np.save(out, arr)
        info[shard] = dict(documents_in_shard=len(df), excluded_matching_our_sample=int(len(df) - sum(keep)),
                           documents_used=len(texts), chunks=int(len(arr)), non_bos_tokens=int(len(arr) * (CTX - 1)),
                           sha256=sha(out))
        print(shard, info[shard], flush=True)
    (TOK / 'prepare_info.json').write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------------------------------------ model side
class ActivationStream:
    """Yields [layer -> fp16 tensor of non-BOS activations] for LM_BATCH chunks at a time, in file order."""

    def __init__(self, model, chunks, layers, device):
        self.m, self.chunks, self.layers, self.dev = model, chunks, layers, device
        self.pos, self.tokens_read = 0, 0
        self.names = [f'blocks.{L}.hook_resid_pre' for L in layers]

    @torch.inference_mode()
    def next(self):
        if self.pos + LM_BATCH > len(self.chunks):
            raise StopIteration('token stream exhausted')
        toks = torch.as_tensor(self.chunks[self.pos:self.pos + LM_BATCH].astype(np.int64), device=self.dev)
        self.pos += LM_BATCH
        _, cache = self.m.run_with_cache(toks, names_filter=lambda n: n in self.names, stop_at_layer=max(self.layers) + 1)
        self.tokens_read += LM_BATCH * (CTX - 1)
        return {L: cache[f'blocks.{L}.hook_resid_pre'][:, 1:].reshape(-1, D).to(torch.float16) for L in self.layers}


class Buffer:
    def __init__(self, stream, layers, device):
        self.s, self.layers = stream, layers
        self.buf = {L: torch.empty(BUFFER, D, dtype=torch.float16, device=device) for L in layers}
        self.fill(torch.arange(BUFFER, device=device))

    def fill(self, idx):
        n, got = len(idx), 0
        while got < n:
            a = self.s.next()
            take = min(n - got, len(a[self.layers[0]]))
            for L in self.layers:
                self.buf[L][idx[got:got + take]] = a[L][:take]
            got += take


# ------------------------------------------------------------------------------------------------ training
class Run:
    def __init__(self, name, arch, width, seed, L, scale, bdec, device, total_steps):
        torch.manual_seed(seed * 1000 + L)
        self.name, self.arch, self.seed, self.L, self.scale = name, arch, seed, L, scale
        self.sae = ModernSAE(D, width, arch, k=TARGET_L0, jump_eps=JUMP_EPS).to(device)
        with torch.no_grad():
            self.sae.output_bias.copy_(bdec)
        thr = [p for n, p in self.sae.named_parameters() if n == 'log_threshold']
        rest = [p for n, p in self.sae.named_parameters() if p.requires_grad and n != 'log_threshold']
        groups = [dict(params=rest, lr_mult=1.0)] + ([dict(params=thr, lr_mult=THR_LR_MULT)] if thr else [])
        self.opt = torch.optim.Adam(groups, lr=LR)
        self.lam = {'relu': 0.1, 'jumprelu': 0.01}.get(arch, 0.0)
        self.last_fired = torch.zeros(width, dtype=torch.long, device=device)
        self.total, self.ema_l0, self.step, self.bt_ema = total_steps, None, 0, None

    def lr(self):
        s, T = self.step, self.total
        if s < WARMUP:
            return LR * (s + 1) / WARMUP
        d0 = int(T * (1 - DECAY_FRAC))
        return LR if s < d0 else LR * max(0.0, (T - s) / (T - d0))

    @torch.no_grad()
    def init_threshold(self, x):
        """JumpReLU: start each latent's threshold at its (1 - target/width) pre-activation quantile, so L0 starts
        near the target (the published init of 0.001 cannot be moved far enough in our step budget)."""
        if self.arch != 'jumprelu':
            return
        pre = self.sae.pre(x)
        q = torch.quantile(pre.float(), 1 - TARGET_L0 / self.sae.d_sae, dim=0).clamp_min(1e-3)
        self.sae.log_threshold.copy_(q.log())

    def train_step(self, x):
        sae = self.sae
        sae.train()
        for g in self.opt.param_groups:
            g['lr'] = self.lr() * g['lr_mult']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rec, z, pre = sae(x)
        rec, z, pre = rec.float(), z.float(), pre.float()
        err = x - rec
        mse = err.pow(2).sum(-1).mean()
        loss = mse
        l0 = (z > 0).float().sum(-1).mean()
        if self.arch == 'relu':
            loss = loss + self.lam * (z * sae.decoder.weight.norm(dim=0)).sum(-1).mean()
        elif self.arch == 'jumprelu':
            loss = loss + self.lam * sae.l0_surrogate(pre)
        fired = (z > 0).any(0)
        self.last_fired[fired] = self.step
        aux = torch.zeros((), device=x.device)
        if self.arch in ('topk', 'batchtopk', 'topk_frozen_decoder'):
            dead = (self.step - self.last_fired) * BATCH > DEAD_TOKENS
            nd = int(dead.sum())
            if nd > 0:
                kk = min(AUX_K, nd)
                p = F.relu(pre).masked_fill(~dead, 0.0)
                v, i = p.topk(kk, dim=-1)
                za = torch.zeros_like(p).scatter(-1, i, v)
                e_hat = F.linear(za, sae.decoder.weight)
                e = err.detach()
                aux = (e - e_hat).pow(2).sum(-1).mean() / e.pow(2).sum(-1).mean().clamp_min(1e-8) * mse.detach()
                loss = loss + AUX_ALPHA * aux
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in sae.parameters() if p.requires_grad], CLIP)
        self.opt.step()
        sae.normalize_decoder()
        with torch.no_grad():
            l0v = float(l0)
            self.ema_l0 = l0v if self.ema_l0 is None else 0.99 * self.ema_l0 + 0.01 * l0v
            if self.arch in ('relu', 'jumprelu'):
                self.lam = min(LAM_MAX, max(LAM_MIN, self.lam * math.exp(CTRL_ETA * max(-1.0, min(1.0, math.log(max(l0v, 1e-3) / TARGET_L0))))))
            if self.arch == 'batchtopk':
                kept = z[z > 0]
                t = float(kept.min()) if kept.numel() else 0.0
                self.bt_ema = t if self.bt_ema is None else 0.99 * self.bt_ema + 0.01 * t
                sae.bt_threshold.fill_(self.bt_ema)
        self.step += 1
        return dict(loss=float(loss), mse=float(mse), l0=l0v, aux=float(aux), lam=self.lam)

    @torch.no_grad()
    def monitor(self, xm):
        """xm: scaled monitor activations (float32). EV, L0 and dead fraction in eval mode."""
        self.sae.eval()
        err = var = l0 = 0.0
        active = torch.zeros(self.sae.d_sae, dtype=torch.bool, device=xm.device)
        mu = xm.mean(0)
        for b in xm.split(8192):
            rec, z, _ = self.sae(b)
            err += (b - rec).pow(2).sum().item()
            var += (b - mu).pow(2).sum().item()
            l0 += (z > 0).sum().item()
            active |= (z > 0).any(0)
        return dict(ev=1 - err / var, l0=l0 / len(xm), dead=float((~active).float().mean()))

    def save(self, path, tokens, extra):
        import copy
        m = copy.deepcopy(self.sae).cpu()
        m.fold_scale(self.scale)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit(f'{path} exists; refusing to overwrite.')
        torch.save(dict(model_state_dict=m.state_dict(),
                        config=dict(m.config(), layer=self.L, seed=self.seed, name=self.name, target_l0=TARGET_L0,
                                    tokens_trained=tokens, input_scale_folded=self.scale, lambda_final=self.lam,
                                    procedure='r2_streaming_2026-11-10', **extra)), path)


def train(out, layers, configs, total_tokens, pilot):
    from transformer_lens import HookedTransformer
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not pilot:
        for f in ('training_monitor.csv', 'training_info.json'):
            if (out / f).exists():
                raise SystemExit(f'{out / f} exists; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    ckdir = (out / 'checkpoints') if pilot else CKPT
    t0 = time.time()
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    params_before = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    train_chunks = np.load(TOK / f'shard{TRAIN_SHARD}_chunks.npy', mmap_mode='r')
    mon_chunks = np.load(TOK / f'shard{MONITOR_SHARD}_chunks.npy', mmap_mode='r')[:800]
    total_steps = total_tokens // BATCH
    # monitor activations (raw) and per-layer input scale from the first 200k training tokens
    mon = ActivationStream(model, mon_chunks, layers, device)
    xm = {L: [] for L in layers}
    for _ in range(len(mon_chunks) // LM_BATCH):
        a = mon.next()
        for L in layers:
            xm[L].append(a[L].float())
    xm = {L: torch.cat(v) for L, v in xm.items()}
    stream = ActivationStream(model, train_chunks, layers, device)
    buf = Buffer(stream, layers, device)
    scale, bdec = {}, {}
    for L in layers:
        x = buf.buf[L][:200_000].float()
        scale[L] = float(math.sqrt(D / x.pow(2).sum(-1).mean().item()))
        bdec[L] = (x * scale[L]).mean(0)
    runs = [Run(name, arch, width, seed, L, scale[L], bdec[L], device, total_steps)
            for L in layers for (name, arch, width, seeds) in configs for seed in seeds]
    for r in runs:
        r.init_threshold(buf.buf[r.L][:8192].float() * scale[r.L])
    gens = {(L, s): torch.Generator(device=device).manual_seed(7919 * s + L) for L in layers for s in SEEDS}
    log, saved = [], set()
    print(f'{len(runs)} SAEs; {total_steps} steps; setup {time.time() - t0:.0f}s', flush=True)
    step = 0
    while step < total_steps:
        # one round: each (layer, seed) draws batches with its own generator; then half the buffer is refreshed
        for _ in range(BUFFER // 2 // BATCH):
            if step >= total_steps:
                break
            for L in layers:
                for s in SEEDS:
                    rs = [r for r in runs if r.L == L and r.seed == s]
                    if not rs:
                        continue
                    idx = torch.randint(BUFFER, (BATCH,), generator=gens[(L, s)], device=device)
                    x = buf.buf[L][idx].float() * scale[L]
                    for r in rs:
                        st = r.train_step(x)
                        if step % 250 == 0:
                            log.append(dict(step=step, tokens=step * BATCH, layer=L, name=r.name, seed=s, **st))
            step += 1
            tok = step * BATCH
            if step % 2500 == 0 or step == total_steps:
                for r in runs:
                    m = r.monitor(xm[r.L] * scale[r.L])
                    log.append(dict(step=step, tokens=tok, layer=r.L, name=r.name, seed=r.seed, monitor=True, **m,
                                    lam=r.lam))
                last = [x for x in log if x.get('monitor') and x['step'] == step]
                print(f'step {step}/{total_steps} ({time.time() - t0:.0f}s, lm tokens {stream.tokens_read:,}): ' +
                      '; '.join(f"L{x['layer']} {x['name']}/{x['seed']} ev {x['ev']:.3f} l0 {x['l0']:.0f} dead {x['dead']:.3f}"
                                for x in last if x['seed'] == 42), flush=True)
            for b in SAVE_AT:
                if tok >= b and b not in saved and not pilot:
                    for r in runs:
                        r.save(ckdir / f'layer_{r.L}/{r.name}_seed{r.seed}/checkpoint_{b // 1_000_000}M.pt', b, {})
                    saved.add(b)
        idx = torch.randperm(BUFFER, device=device)[:BUFFER // 2]
        try:
            buf.fill(idx)
        except StopIteration:
            print('stream exhausted; stopping early', flush=True)
            break
    for r in runs:
        r.save(ckdir / f'layer_{r.L}/{r.name}_seed{r.seed}/checkpoint_final.pt', step * BATCH, {})
    params_after = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    pd.DataFrame(log).to_csv(out / 'training_monitor.csv', index=False)
    info = dict(script_sha256=sha(__file__), modern_sae_sha256=sha(REPO / 'src/sae/modern_sae.py'), layers=list(layers),
                configs=[list(c[:3]) + [list(c[3])] for c in configs], steps=step, tokens_per_sae=step * BATCH,
                lm_tokens_read=stream.tokens_read, unique_train_tokens_available=int(len(train_chunks) * (CTX - 1)),
                input_scale=scale, runtime_s=time.time() - t0, gpt2_parameters_unchanged=params_before == params_after,
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), pilot=pilot,
                hyper=dict(batch=BATCH, lr=LR, warmup=WARMUP, decay_frac=DECAY_FRAC, clip=CLIP, buffer=BUFFER,
                           thr_lr_mult=THR_LR_MULT, jump_eps=JUMP_EPS, lam_bounds=[LAM_MIN, LAM_MAX], autocast='bf16 forward', aux_alpha=AUX_ALPHA, aux_k=AUX_K, dead_tokens=DEAD_TOKENS, ctrl_eta=CTRL_ETA,
                           target_l0=TARGET_L0))
    (out / 'training_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-data', action='store_true')
    g.add_argument('--pilot', type=Path)
    g.add_argument('--run', action='store_true')
    ap.add_argument('--pilot-tokens', type=int, default=5_000_000)
    ap.add_argument('--pilot-configs', default='')
    a = ap.parse_args()
    if a.prepare_data:
        prepare_data()
    elif a.pilot:
        cfg = [(n, ar, w, (42,)) for (n, ar, w, _) in CONFIGS if n != 'topk_frozen_decoder'
               and (not a.pilot_configs or n in a.pilot_configs.split(','))]
        train(a.pilot, (8,), cfg, a.pilot_tokens, True)
    else:
        pr = json.loads((OUT / 'preregistration.json').read_text())
        for r_, h in pr['hashes'].items():
            if not r_.endswith('README.md') and sha(REPO / r_) != h:
                raise SystemExit(f'{r_}: hash differs from preregistration.json; refusing to run.')
        train(OUT, LAYERS, CONFIGS, pr['tokens_per_sae'], False)


if __name__ == '__main__':
    main()
