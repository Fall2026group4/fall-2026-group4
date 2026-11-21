#!/usr/bin/env python3
"""XB step 1: build the shared-text-span document samples used by BOTH models (2026-11-21).

Corpora (pre-registration: results/cross_model/xb_matched_splicing/preregistration.json):
  * pile:   M6's committed 512 eval + 64 reference documents (line indices from M6's sample_manifest.csv, origin/main
            bf25b40), re-read from the pinned Pile validation file and verified against M6's text SHA-256.
  * owt_xm: a NEW sample from OpenWebText shard 00077 (never used by any of our SAE training: R2 streamed shards
            00078/00079). Order = numpy.random.default_rng(20261121).permutation(n_rows). A document is eligible if
            it has >= 32 text tokens under both tokenizers and its text SHA-256 matches none of our 1,000-document
            OWT sample and none of shards 00078/00079. First 512 eligible -> eval, next 64 -> reference.

Context rule (primary, 'shared span'): each document is cut to the longest character prefix that has <= 1023 text
tokens under BOTH tokenizers (BOS is prepended separately, so <= 1024 inputs), i.e. both models see identical text.
The secondary 'own-1024' rule (each model's own first 1023 text tokens) is recorded too; for Gemma on the Pile it
is exactly M6's input.

Outputs (refuses to overwrite): results/cross_model/xb_matched_splicing/samples/{corpus}_manifest.csv with ids,
hashes, the span end and the token counts of both tokenizers. No text is written.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

OUT = REPO / 'results/cross_model/xb_matched_splicing/samples'
M6 = REPO / 'results/cross_model/xb_matched_splicing/inputs/m6'
MAXTXT = 1023
SEED_OWT = 20261121
HF = Path.home() / '.cache/huggingface/hub'


def sha_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def sha_ids(ids):
    return ps.sha256_token_ids(ids)


def tokenizers():
    from transformers import AutoTokenizer, GPT2TokenizerFast
    g2 = GPT2TokenizerFast.from_pretrained('gpt2')
    gm = AutoTokenizer.from_pretrained('google/gemma-2-2b', revision='c5ebcd40d208330abc697524c919956e692655cf')
    return g2, gm


def span_end(text, toks):
    """Largest c such that text[:c] has <= MAXTXT tokens under every tokenizer (checked by re-tokenising)."""
    c = len(text)
    for t in toks:
        off = t(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        if len(off) > MAXTXT:
            c = min(c, off[MAXTXT - 1][1])
    for _ in range(200):
        if all(len(t(text[:c], add_special_tokens=False)['input_ids']) <= MAXTXT for t in toks):
            return c
        c -= 1
    raise RuntimeError('span search did not converge')


def row(corpus, set_name, rank, doc_id, source, text, g2, gm):
    c = span_end(text, (g2, gm))
    span = text[:c]
    i2 = g2(span, add_special_tokens=False)['input_ids']
    im = gm(span, add_special_tokens=False)['input_ids']
    own_m = gm(text, add_special_tokens=False)['input_ids'][:MAXTXT]
    own_2 = g2(text, add_special_tokens=False)['input_ids'][:MAXTXT]
    return dict(corpus=corpus, set=set_name, rank=rank, doc_id=doc_id, source=source, n_chars=len(text),
                text_sha256=sha_text(text), span_chars=c, span_sha256=sha_text(span),
                gpt2_span_tokens=len(i2), gemma_span_tokens=len(im), gpt2_span_ids_sha256=sha_ids(i2),
                gemma_span_ids_sha256=sha_ids(im), gpt2_own_tokens=len(own_2), gemma_own_tokens=len(own_m),
                gemma_own_ids_sha256=sha_ids([2] + own_m))


def pile_text_map(line_ids):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('monology/pile-uncopyrighted', 'val.jsonl.zst', repo_type='dataset',
                        revision='3be90335b66f24456a5d6659d9c8d208c0357119')
    objs = ps.load_lines(Path(p), line_ids)
    return {i: (o['text'], o['meta']['pile_set_name']) for i, o in objs.items()}


def build_pile(g2, gm):
    man = pd.read_csv(M6 / 'sample_manifest.csv')
    texts = pile_text_map(man.line_index.tolist())
    rows = []
    for r in man.itertuples():
        t, src = texts[r.line_index]
        if sha_text(t) != r.text_sha256:
            raise SystemExit(f'Pile line {r.line_index}: text hash differs from M6 manifest')
        out = row('pile', r.set, r.rank, int(r.line_index), src, t, g2, gm)
        if out['gemma_own_ids_sha256'] != r.token_ids_sha256:
            raise SystemExit(f'Pile line {r.line_index}: Gemma own-1024 ids differ from M6 manifest')
        rows.append(out)
    return pd.DataFrame(rows)


def build_owt(g2, gm):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('Skylion007/openwebtext', 'plain_text/train-00077-of-00080.parquet', repo_type='dataset')
    df = pd.read_parquet(p, columns=['text'])
    excl = set()
    for s in ('train', 'validation', 'test'):
        excl |= {sha_text(t) for t in pd.read_parquet(REPO / f'data/openwebtext/{s}.parquet').text}
    for sh in ('78', '79'):
        f = REPO / f'data/openwebtext/stream_shards/plain_text/train-000{sh}-of-00080.parquet'
        excl |= {sha_text(t) for t in pd.read_parquet(f, columns=['text']).text}
    order = np.random.default_rng(SEED_OWT).permutation(len(df))
    rows, n_ex, n_short, n_dup = [], 0, 0, 0
    for pos in order:
        t = df.text.iloc[int(pos)]
        n_ex += 1
        h = sha_text(t)
        if h in excl:
            n_dup += 1
            continue
        if min(len(g2(t[:20000], add_special_tokens=False)['input_ids']),
               len(gm(t[:20000], add_special_tokens=False)['input_ids'])) < 32:
            n_short += 1
            continue
        excl.add(h)
        k = len(rows)
        rows.append(row('owt_xm', 'eval' if k < 512 else 'reference', k if k < 512 else k - 512, int(pos),
                        'openwebtext', t, g2, gm))
        if len(rows) == 576:
            break
    info = dict(shard='train-00077-of-00080.parquet', n_rows=len(df), seed=SEED_OWT, examined=n_ex,
                skipped_short=n_short, skipped_duplicate_of_our_data=n_dup, excluded_hashes=len(excl) - 576,
                order_head_sha256=hashlib.sha256(order[:n_ex].astype('<i8').tobytes()).hexdigest())
    return pd.DataFrame(rows), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=['pile', 'owt_xm'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'{a.corpus}_manifest.csv'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    g2, gm = tokenizers()
    t0 = time.time()
    info = {}
    if a.corpus == 'pile':
        df = build_pile(g2, gm)
    else:
        df, info = build_owt(g2, gm)
    assert df.groupby('set').size().to_dict() == {'eval': 512, 'reference': 64}
    assert df.doc_id.is_unique
    assert (df.gpt2_span_tokens <= MAXTXT).all() and (df.gemma_span_tokens <= MAXTXT).all()
    df.to_csv(out, index=False)
    info.update(corpus=a.corpus, seconds=round(time.time() - t0, 1), manifest_sha256=ps.sha256_file(out),
                span_tokens_gpt2_mean=float(df.gpt2_span_tokens.mean()),
                span_tokens_gemma_mean=float(df.gemma_span_tokens.mean()),
                gemma_tokens_per_gpt2_token=float(df.gemma_span_tokens.sum() / df.gpt2_span_tokens.sum()))
    (OUT / f'{a.corpus}_sample_info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
