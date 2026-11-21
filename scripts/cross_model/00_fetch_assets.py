#!/usr/bin/env python3
"""Fetch the pinned external assets for the final-week cross-model work (2026-11-21) and record their hashes.

Everything goes to the Hugging Face cache (re-downloadable, never committed). Writes
results/cross_model/assets/asset_manifest_<batch>.json; refuses to overwrite an existing manifest.

Batches:
  b1: Gemma Scope 16k canonical SAEs, all 26 layers (repo revision fd571b47, as pinned by M6/R3);
      res-jb GPT-2 SAEs, layers 0-11; Pile validation file (M6 pin); OWT shard 00077 (fresh cross-model OWT sample).
Run from the repository root:  python scripts/cross_model/00_fetch_assets.py --batch b1
"""
import argparse, hashlib, json, time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
GS_REPO, GS_REV = 'google/gemma-scope-2b-pt-res', 'fd571b47c1c64851e9b1989792367b9babb4af63'
CANONICAL_L0 = [105, 102, 141, 59, 124, 68, 70, 69, 71, 73, 77, 80, 82, 84, 84, 78, 78, 77, 74, 73, 71, 70, 72, 75,
                73, 116]  # sae_lens gemma-scope-2b-pt-res-canonical, width_16k (sae_lens 6.54.12)
PILE = ('monology/pile-uncopyrighted', 'val.jsonl.zst', '3be90335b66f24456a5d6659d9c8d208c0357119')
OWT = ('Skylion007/openwebtext', 'plain_text/train-00077-of-00080.parquet')
RESJB = 'jbloom/GPT2-Small-SAEs-Reformatted'
LADDER_16K = {5: [18, 34, 143, 309], 12: [22, 41, 176, 445], 19: [23, 40, 137, 279]}  # non-canonical, sae_lens 6.54.12


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def get(rows, **kw):
    t = time.time()
    p = hf_hub_download(**kw)
    rows.append({**{k: v for k, v in kw.items()}, 'local_path': str(Path(p)).replace(str(Path.home()), '~'),
                 'bytes': Path(p).stat().st_size, 'sha256': sha(p), 'seconds': round(time.time() - t, 1)})
    print(rows[-1]['filename'], rows[-1]['bytes'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=['b1', 'b2a'])
    a = ap.parse_args()
    out = REPO / f'results/cross_model/assets/asset_manifest_{a.batch}.json'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    rows = []
    if a.batch == 'b2a':        # 16k L0 ladder at the T_B layers (sparsity-mechanism analysis; XB/XT-G batch 2)
        for L, l0s in LADDER_16K.items():
            for l0 in l0s:
                get(rows, repo_id=GS_REPO, filename=f'layer_{L}/width_16k/average_l0_{l0}/params.npz', revision=GS_REV)
        out.write_text(json.dumps({'batch': a.batch, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': rows},
                                  indent=1))
        return
    get(rows, repo_id=PILE[0], repo_type='dataset', filename=PILE[1], revision=PILE[2])
    get(rows, repo_id=OWT[0], repo_type='dataset', filename=OWT[1])
    for L in range(12):
        for fn in ('cfg.json', 'sae_weights.safetensors'):
            get(rows, repo_id=RESJB, filename=f'blocks.{L}.hook_resid_pre/{fn}')
    for L, l0 in enumerate(CANONICAL_L0):
        get(rows, repo_id=GS_REPO, filename=f'layer_{L}/width_16k/average_l0_{l0}/params.npz', revision=GS_REV)
    out.write_text(json.dumps({'batch': a.batch, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': rows},
                              indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
