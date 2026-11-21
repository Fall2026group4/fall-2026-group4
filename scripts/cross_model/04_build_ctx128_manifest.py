#!/usr/bin/env python3
"""XB amendment A4: build the 128-token shared-span manifests (2026-11-21).

For every document of the frozen XB manifest (`samples/<corpus>_manifest.csv`; same documents, same eval/reference
sets and ranks), the shared span is re-cut to the longest prefix that has <= 127 text tokens under BOTH tokenizers
(BOS is prepended separately, so <= 128 model inputs). This is the identical rule as the 1024 shared span
(`01_build_xm_samples.span_end`), with the cap at the GPT-2 SAEs' training context (R2 CTX = 128; res-jb 128).

Checks: the 128 span is a prefix of the frozen 1024 span; its text hash and both tokenizations are recorded; the frozen
manifests are read only. Output refuses to overwrite: samples/<corpus>_ctx128_manifest.csv (+ _ctx128_sample_info.json).

Run: python scripts/cross_model/04_build_ctx128_manifest.py --corpus pile|owt_xm
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('xm01', REPO / 'scripts/cross_model/01_build_xm_samples.py')
XM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(XM)
SAMPLES = REPO / 'results/cross_model/xb_matched_splicing/samples'
MAXTXT_128 = 127


def doc_texts(corpus, man):
    spec2 = importlib.util.spec_from_file_location('xb02', REPO / 'scripts/cross_model/02_matched_splicing.py')
    xb = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(xb)
    return xb.doc_texts(corpus, man)


def build(man, texts, g2, gm):
    XM.MAXTXT = MAXTXT_128                      # span_end and row() read the module-level cap
    out = []
    for r in man.itertuples():
        text = texts[int(r.doc_id)]
        assert XM.sha_text(text) == r.text_sha256, f'doc {r.doc_id}: text differs from the frozen manifest'
        row = XM.row(r.corpus, r.set, int(r.rank), int(r.doc_id), r.source, text, g2, gm)
        assert int(row['span_chars']) <= int(r.span_chars), f'doc {r.doc_id}: 128 span longer than the 1024 span'
        row['span1024_chars'] = int(r.span_chars)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=['pile', 'owt_xm'])
    a = ap.parse_args()
    src = SAMPLES / f'{a.corpus}_manifest.csv'
    out = SAMPLES / f'{a.corpus}_ctx128_manifest.csv'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    t0 = time.time()
    man = pd.read_csv(src)
    g2, gm = XM.tokenizers()
    df = build(man, doc_texts(a.corpus, man), g2, gm)
    assert df.groupby('set').size().to_dict() == {'eval': 512, 'reference': 64}
    assert (df.doc_id.values == man.doc_id.values).all() and (df['rank'].values == man['rank'].values).all()
    assert (df.gpt2_span_tokens <= MAXTXT_128).all() and (df.gemma_span_tokens <= MAXTXT_128).all()
    df.to_csv(out, index=False)
    info = dict(corpus=a.corpus, amendment='A4', source_manifest=src.name,
                source_manifest_sha256=XM.ps.sha256_file(src), manifest_sha256=XM.ps.sha256_file(out),
                max_text_tokens=MAXTXT_128, seconds=round(time.time() - t0, 1),
                span_tokens_gpt2_mean=float(df.gpt2_span_tokens.mean()),
                span_tokens_gemma_mean=float(df.gemma_span_tokens.mean()),
                n_gpt2_binds=int((df.gpt2_span_tokens == MAXTXT_128).sum()),
                n_short_docs=int(((df.gpt2_span_tokens < MAXTXT_128) & (df.gemma_span_tokens < MAXTXT_128)).sum()))
    (SAMPLES / f'{a.corpus}_ctx128_sample_info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
