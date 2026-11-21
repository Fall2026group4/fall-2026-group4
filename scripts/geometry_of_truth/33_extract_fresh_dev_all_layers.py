#!/usr/bin/env python3
"""XT-G amendment A2 support: GPT-2 Small final-token activations (blocks.L.hook_resid_pre, L = 0..11) for the DEV
rows of companies_true_false and common_claim_true_false. Uses exactly the rows Gemma used
(data/gemma/activations/got_fresh/{name}_dev_rows.csv); TEST rows are never read. Output (gitignored):
data/got/activations/gpt2_small/fresh_dev_all_layers/{name}_dev_final_layer{L}.npy (float32). Refuses to overwrite.
Run: python scripts/geometry_of_truth/33_extract_fresh_dev_all_layers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'data/got/activations/gpt2_small/fresh_dev_all_layers'
FAMS = ('companies_true_false', 'common_claim_true_false')


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f'{OUT} not empty; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    m = HookedTransformer.from_pretrained('gpt2-small', device='cuda').eval()
    for name in FAMS:
        rows = pd.read_csv(REPO / f'data/gemma/activations/got_fresh/{name}_dev_rows.csv')
        assert set(rows.split) == {'DEV'}
        raw = pd.read_csv(next((REPO / 'data/got/fresh').glob(f'raw_*/{name}.csv')))
        st = raw.statement.iloc[rows.row_id.to_numpy()].tolist()
        assert (raw.label.iloc[rows.row_id.to_numpy()].to_numpy() == rows.label.to_numpy()).all()
        buf = np.zeros((12, len(st), 768), np.float32)
        for s in range(0, len(st), 64):
            seqs = [m.to_tokens(x, prepend_bos=True, move_to_device=False)[0] for x in st[s:s + 64]]
            T = max(len(q) for q in seqs)
            toks = torch.full((len(seqs), T), 50256, dtype=torch.long)
            for i, q in enumerate(seqs):
                toks[i, :len(q)] = q
            last = torch.tensor([len(q) - 1 for q in seqs])
            with torch.no_grad():
                _, cache = m.run_with_cache(toks.cuda(), names_filter=lambda n: n.endswith('hook_resid_pre'))
            for L in range(12):
                buf[L, s:s + len(seqs)] = cache[f'blocks.{L}.hook_resid_pre'][torch.arange(len(seqs)), last].cpu().numpy()
        for L in range(12):
            np.save(OUT / f'{name}_dev_final_layer{L}.npy', buf[L])
        print(name, len(st), flush=True)


if __name__ == '__main__':
    main()
