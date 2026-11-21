#!/usr/bin/env python3
"""XT-G0 step 1: ORIGINAL Gemma-2-2B (frozen) activations on our task data, mirroring the GPT-2 caches (2026-11-21).

Site: output of model.model.layers[L] (resid_post), L = 0..25, HF bf16 eager (as M6/XB). Stored as float16
(lossless range for these values; bf16 has fewer mantissa bits than fp16). Outputs are gitignored
(data/gemma/activations/). Refuses to overwrite.

  got/{name}_final_layer{L}.npy   GoT cities / neg_cities / larger_than / smaller_than (all rows, as the GPT-2 cache;
                                  test rows are dropped at load time and never evaluated before the final pass),
                                  BOS + statement, final token.
  got_fresh/{name}_dev_final_layer{L}.npy   companies_true_false, common_claim_true_false: DEV rows ONLY
                                  (TSP-R Phase 0 split; TEST rows stay locked).
  repe_lm/{final,slot}_layer{L}.npy  row order = data/repe/repe_lm/prompts.csv (train + validation only);
                                  slot = the position where the pair members differ under the Gemma tokenizer.
  owt/train_pca_layer{L}.npz      mean, eigenvalues, eigenvectors of OWT TRAIN (the same 800 documents as the GPT-2
                                  cache, one window per document: BOS + first 127 Gemma tokens), BOS excluded.
  owt/val_layer{L}.npy            OWT VALIDATION windows (same 100 documents), BOS excluded, concatenated.
Gates written to results/gemma/xtg_task_baselines/gates/: K1 (activation invariants before the slot), K7 (final-token
KL of persona vs neutral swaps), extraction info with hashes.
Run: python scripts/gemma/10_extract_task_activations.py [--smoke]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.backends import GemmaBackend  # noqa: E402

ACT = REPO / 'data/gemma/activations'
GATES = REPO / 'results/gemma/xtg_task_baselines/gates'
NL, D = 26, 2304
GOT = ('cities', 'neg_cities', 'larger_than', 'smaller_than')
FRESH = ('companies_true_false', 'common_claim_true_false')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def final_batches(bk, texts, bs=32):
    """Yield (row indices, acts [NL, b, D] at the final token) for BOS + text."""
    for s in range(0, len(texts), bs):
        seqs = [[bk.bos_id] + bk.tokenize(t) for t in texts[s:s + bs]]
        T = max(map(len, seqs))
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
            am[i, :len(q)] = 1
        out = run_masked(bk, ids, am)
        last = am.sum(1) - 1
        yield s, np.stack([out[L][torch.arange(len(seqs)), last].float().cpu().numpy() for L in range(NL)])


@torch.no_grad()
def run_masked(bk, ids, am):
    cap = {}
    hs = []
    for L in range(NL):
        def h(m, i, o, L=L):
            cap[L] = (o[0] if isinstance(o, tuple) else o).detach()
        hs.append(bk.layers[L].register_forward_hook(h))
    try:
        out = bk.model(input_ids=ids.to(bk.device), attention_mask=am.to(bk.device), use_cache=False)
    finally:
        for x in hs:
            x.remove()
    cap['logits'] = out.logits
    return cap


def save_layers(prefix, arr_by_layer):
    for L in range(NL):
        np.save(f'{prefix}_layer{L}.npy', arr_by_layer[L].astype(np.float16))


def do_got(bk, smoke):
    (ACT / 'got').mkdir(parents=True, exist_ok=True)
    info = {}
    for name in GOT:
        df = pd.read_csv(REPO / f'data/got/{name}.csv')
        if smoke:
            df = df.head(64)
        buf = np.zeros((NL, len(df), D), np.float32)
        for s, a in final_batches(bk, df.statement.tolist()):
            buf[:, s:s + a.shape[1]] = a
        save_layers(str(ACT / f'got/{name}_final'), buf)
        info[name] = dict(n=len(df), source_sha256=sha(REPO / f'data/got/{name}.csv'))
    (ACT / 'got_fresh').mkdir(parents=True, exist_ok=True)
    for name in FRESH:
        raw = pd.read_csv(next((REPO / 'data/got/fresh').glob(f'raw_*/{name}.csv')))
        sp = pd.read_csv(REPO / f'data/got/fresh/splits/{name}_splits.csv')
        sp = sp[(sp.split == 'DEV') & (sp.excluded.isna() | (sp.excluded == False))]  # noqa: E712
        df = raw.iloc[sp.row_id.to_numpy()].reset_index(drop=True)
        if smoke:
            df, sp = df.head(64), sp.head(64)
        assert 'TEST' not in set(sp.split)
        buf = np.zeros((NL, len(df), D), np.float32)
        for s, a in final_batches(bk, df.statement.tolist()):
            buf[:, s:s + a.shape[1]] = a
        save_layers(str(ACT / f'got_fresh/{name}_dev_final'), buf)
        sp[['row_id', 'group_id', 'split']].assign(label=df.label.to_numpy()).to_csv(
            ACT / f'got_fresh/{name}_dev_rows.csv', index=False)
        info[name] = dict(n_dev=len(df), split_sha256=sha(REPO / f'data/got/fresh/splits/{name}_splits.csv'))
    return info


def do_repe(bk, smoke):
    (ACT / 'repe_lm').mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(REPO / 'data/repe/repe_lm/prompts.csv')
    assert set(d.split) <= {'train', 'validation'}
    groups = list(d.groupby(['statement_id', 'template']).groups.items())
    if smoke:
        groups = groups[:40]
    fin = np.zeros((NL, len(d), D), np.float32)
    slot = np.zeros((NL, len(d), D), np.float32)
    slot_pos = np.full(len(d), -1)
    k1 = dict(n_pairs=0, n_len_mismatch=0, n_multi_diff=0, max_abs_before_slot_by_layer=[0.0] * NL,
              min_abs_at_slot_layer0=np.inf)
    k7 = []
    for (sid, tmpl), idx in groups:
        idx = list(idx)
        seqs = [[bk.bos_id] + bk.tokenize(p) for p in d.prompt.iloc[idx]]
        n = len(seqs[0])
        if any(len(q) != n for q in seqs):
            k1['n_len_mismatch'] += 1
            continue
        diff = [t for t in range(n) if len({q[t] for q in seqs}) > 1]
        if len(diff) != 1:
            k1['n_multi_diff'] += 1
            continue
        sp = diff[0]
        ids = torch.tensor(seqs)
        cap = run_masked(bk, ids, torch.ones_like(ids))
        for L in range(NL):
            h = cap[L].float()
            fin[L, idx] = h[:, -1].cpu().numpy()
            slot[L, idx] = h[:, sp].cpu().numpy()
            if sp > 0:
                k1['max_abs_before_slot_by_layer'][L] = max(k1['max_abs_before_slot_by_layer'][L],
                                                            float((h[:, :sp] - h[:1, :sp]).abs().max()))
        k1['min_abs_at_slot_layer0'] = min(k1['min_abs_at_slot_layer0'],
                                           float((cap[0][1:, sp].float() - cap[0][:1, sp].float()).abs().max(-1).values.min()))
        k1['n_pairs'] += len(idx) - 1
        slot_pos[idx] = sp
        lp = torch.log_softmax(cap['logits'][:, -1].float(), -1)
        var = d.variant.iloc[idx].tolist()
        ref = var.index('H') if 'H' in var else var.index('Na')
        for j, v in enumerate(var):
            if j != ref:
                kl = float((lp[ref].exp() * (lp[ref] - lp[j])).sum())
                k7.append(dict(statement_id=sid, template=tmpl, ref=var[ref], other=v, kl=kl,
                               split=d.split.iloc[idx[0]]))
    save_layers(str(ACT / 'repe_lm/final'), fin)
    save_layers(str(ACT / 'repe_lm/slot'), slot)
    np.save(ACT / 'repe_lm/slot_pos.npy', slot_pos)
    return k1, pd.DataFrame(k7), dict(n_rows=len(d), prompts_sha256=sha(REPO / 'data/repe/repe_lm/prompts.csv'),
                                      n_rows_filled=int((slot_pos >= 0).sum()))


def owt_windows(bk, split, smoke):
    df = pd.read_parquet(REPO / f'data/openwebtext/{split}.parquet')
    if smoke:
        df = df.head(16)
    for s in range(0, len(df), 16):
        seqs = [[bk.bos_id] + bk.tokenize(t)[:127] for t in df.text.iloc[s:s + 16]]
        T = max(map(len, seqs))
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
            am[i, :len(q)] = 1
        cap = run_masked(bk, ids, am)
        m = am.bool().clone()
        m[:, 0] = False
        yield {L: cap[L][m.to(bk.device)].double().cpu() for L in range(NL)}


def do_owt(bk, smoke):
    (ACT / 'owt').mkdir(parents=True, exist_ok=True)
    s1 = {L: torch.zeros(D, dtype=torch.float64) for L in range(NL)}
    s2 = {L: torch.zeros(D, D, dtype=torch.float64) for L in range(NL)}
    n = 0
    for chunk in owt_windows(bk, 'train', smoke):
        for L in range(NL):
            x = chunk[L]
            s1[L] += x.sum(0)
            s2[L] += x.T @ x
        n += chunk[0].shape[0]
    for L in range(NL):
        mu = s1[L] / n
        cov = s2[L] / n - torch.outer(mu, mu)
        w, V = torch.linalg.eigh(cov)
        w, V = w.flip(0), V.flip(1)
        np.savez(ACT / f'owt/train_pca_layer{L}.npz', mu=mu.numpy(), eigvals=w.numpy(), eigvecs=V.float().numpy(), n=n)
    vals = {L: [] for L in range(NL)}
    for chunk in owt_windows(bk, 'validation', smoke):
        for L in range(NL):
            vals[L].append(chunk[L].float().numpy())
    for L in range(NL):
        np.save(ACT / f'owt/val_layer{L}.npy', np.concatenate(vals[L]).astype(np.float16))
    return dict(n_train_tokens=int(n), n_val_tokens=int(sum(v.shape[0] for v in vals[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    global ACT, GATES
    if a.smoke:
        ACT, GATES = ACT / '_smoke', GATES.parent / 'smoke_gates'
    if (GATES / 'extraction_info.json').exists():
        raise SystemExit('extraction already done; refusing to overwrite')
    GATES.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    bk = GemmaBackend()
    info = dict(got=do_got(bk, a.smoke))
    print('got done', time.time() - t0, flush=True)
    k1, k7, info['repe_lm'] = do_repe(bk, a.smoke)
    print('repe done', time.time() - t0, flush=True)
    info['owt'] = do_owt(bk, a.smoke)
    k1['pass'] = bool(k1['n_len_mismatch'] == 0 and k1['n_multi_diff'] == 0 and
                      max(k1['max_abs_before_slot_by_layer']) <= 1e-2 and k1['min_abs_at_slot_layer0'] > 0)
    (GATES / 'k1_gemma.json').write_text(json.dumps(k1, indent=1, default=float))
    k7.to_csv(GATES / 'k7_gemma_rows.csv', index=False)
    info.update(minutes=round((time.time() - t0) / 60, 2), script_sha256=sha(__file__),
                backends_sha256=sha(REPO / 'src/models/backends.py'), dtype_stored='float16', site='resid_post')
    (GATES / 'extraction_info.json').write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(dict(k1=k1, info=info), indent=1, default=float))


if __name__ == '__main__':
    main()
