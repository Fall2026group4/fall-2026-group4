#!/usr/bin/env python3
"""M6 step 1: build the pre-registered Pile validation sample (CPU only; tokenizer, no model weights).

Pre-registration: results/gemma2_pile/gemma_scope_splicing/README.md + preregistration.json (commit b39dcc2).

What it does, in order:
 1. Reads the design from preregistration.json (fields listed in src/evaluation/pile_splicing.REQUIRED_PREREG_FIELDS).
 2. Downloads ``val.jsonl.zst`` from the pinned dataset revision (or uses --data-file) and checks its size and
    SHA-256 against the pre-registration (gate V1). A mismatch stops the script.
 3. Counts the documents, builds the seeded permutation, and selects the first n_eval eligible documents
    (>= min_text_tokens text tokens) as the evaluation set and the next n_reference as the reference set.
 4. Writes sample_manifest.csv (identifiers and hashes only, never the text) and sample_info.json (provenance).

Gemma-2-2B is gated on Hugging Face: the tokenizer download needs an accepted licence and a token in the
environment (HF_TOKEN or `huggingface-cli login`). The script never prints or stores the token.

Run from repository root:
    python scripts/gemma/02_pile_sample_manifest.py            # pre-registered sample -> gemma_scope_splicing/
    python scripts/gemma/02_pile_sample_manifest.py --smoke    # 8 eval + 4 reference -> gemma_scope_splicing/smoke/
Refuses to overwrite existing outputs.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

EXP = REPO / 'results/gemma2_pile/gemma_scope_splicing'
PREREG = EXP / 'preregistration.json'
SMOKE_N_EVAL, SMOKE_N_REF = 8, 4


def rel(p: Path) -> str:
    """Repository-relative path for records (no absolute home paths in committed outputs)."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true', help='tiny sample (8 eval + 4 reference) written to smoke/')
    ap.add_argument('--data-file', type=Path, default=None,
                    help='use an already downloaded val.jsonl.zst (still SHA-256 checked)')
    args = ap.parse_args()

    t0 = time.time()
    cfg = ps.load_preregistration(PREREG)
    data, samp, tokcfg = cfg['data'], cfg['sampling'], cfg['tokenization']
    n_eval = SMOKE_N_EVAL if args.smoke else int(samp['n_eval'])
    n_ref = SMOKE_N_REF if args.smoke else int(samp['n_reference'])
    out_dir = EXP / 'smoke' if args.smoke else EXP
    manifest_path, info_path = out_dir / 'sample_manifest.csv', out_dir / 'sample_info.json'
    clash = [rel(p) for p in (manifest_path, info_path) if p.exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite existing outputs: {clash}')

    # ---- 1. data file: pinned revision, size and SHA-256 (gate V1, part 1)
    if args.data_file is not None:
        data_path = args.data_file
    else:
        from huggingface_hub import hf_hub_download
        print(f"downloading {data['hf_dataset']}/{data['file']} @ {data['revision'][:12]} ...", flush=True)
        data_path = Path(hf_hub_download(repo_id=data['hf_dataset'], filename=data['file'], repo_type='dataset',
                                         revision=data['revision']))
    size = data_path.stat().st_size
    if size != int(data['bytes']):
        raise SystemExit(f"V1 FAILED: {data['file']} has {size} bytes, pre-registered {data['bytes']}")
    print('hashing data file ...', flush=True)
    digest = ps.sha256_file(data_path)
    if digest != data['sha256']:
        raise SystemExit(f"V1 FAILED: SHA-256 {digest} != pre-registered {data['sha256']}")
    print('V1 data file: size and SHA-256 match the pre-registration', flush=True)

    # ---- 2. tokenizer at the pre-registered revision (Amendment 1); the latest revision is never looked up
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok_revision = tokcfg['revision']
    tok = AutoTokenizer.from_pretrained(tokcfg['tokenizer'], revision=tok_revision)
    if tok.bos_token_id is None:
        raise SystemExit('tokenizer has no BOS token')
    tok_file_sha = {name: ps.sha256_file(Path(hf_hub_download(tokcfg['tokenizer'], name, revision=tok_revision)))
                    for name in ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
                                 'special_tokens_map.json')}

    def text_ids(text: str):
        return tok.encode(text, add_special_tokens=False)

    # ---- 3. deterministic order and selection
    print('counting documents ...', flush=True)
    n_lines = ps.count_lines_jsonl_zst(data_path)
    order = ps.permutation_order(n_lines, int(samp['seed']))
    source_field = data['source_label_field'].split('.')
    objs = {}

    def load_texts(indices):
        objs.update(ps.load_lines(data_path, [i for i in indices if i not in objs]))
        return {i: objs[i]['text'] for i in indices}

    eval_ids, ref_ids, sel_info, texts = ps.select_from_corpus(
        order, load_texts, lambda t: len(text_ids(t)), n_eval=n_eval, n_reference=n_ref,
        min_text_tokens=int(samp['min_text_tokens']))
    if set(eval_ids) & set(ref_ids):  # V1, part 2 (also asserted inside select_documents)
        raise SystemExit('V1 FAILED: evaluation and reference sets overlap')
    position = {int(i): p for p, i in enumerate(order[:sel_info['candidates_loaded']])}

    # ---- 4. manifest rows (hashes and identifiers only)
    rows = []
    for set_name, ids in (('eval', eval_ids), ('reference', ref_ids)):
        for rank, idx in enumerate(ids):
            rows.append(ps.manifest_row(
                set_name=set_name, rank=rank, perm_position=position[idx], line_index=idx,
                source=ps.get_field(objs[idx], source_field), text=texts[idx], text_ids=text_ids(texts[idx]),
                bos_id=tok.bos_token_id, max_tokens_including_bos=int(tokcfg['max_tokens_including_bos'])))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ps.MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    import transformers
    import zstandard
    info = {
        'experiment': cfg.get('experiment'),
        'status': 'SMOKE (pipeline check, not a result)' if args.smoke else 'pre-registered sample',
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data': {'hf_dataset': data['hf_dataset'], 'revision': data['revision'], 'file': data['file'],
                 'bytes': size, 'sha256': digest, 'n_documents': n_lines},
        'sampling': {'seed': int(samp['seed']), 'rng': 'numpy.random.default_rng(seed).permutation(n_documents)',
                     'min_text_tokens': int(samp['min_text_tokens']), 'n_eval': n_eval, 'n_reference': n_ref,
                     'smoke': bool(args.smoke), **sel_info,
                     'selected_order_sha256': hashlib.sha256(
                         np.asarray(eval_ids + ref_ids, dtype='<i8').tobytes()).hexdigest()},
        'gate_V1': {'size_match': True, 'sha256_match': True, 'disjoint': True,
                    'n_eval': len(eval_ids), 'n_reference': len(ref_ids)},
        'tokenizer': {'hf_id': tokcfg['tokenizer'], 'revision_preregistered': tok_revision,
                      'bos_token_id': tok.bos_token_id, 'file_sha256': tok_file_sha,
                      'max_tokens_including_bos': int(tokcfg['max_tokens_including_bos'])},
        'eval_sources': dict(Counter(r['pile_set_name'] for r in rows if r['set'] == 'eval').most_common()),
        'eval_truncated': sum(r['truncated'] for r in rows if r['set'] == 'eval'),
        'hashes': {'preregistration.json': ps.sha256_file(PREREG),
                   'README.md': ps.sha256_file(EXP / 'README.md'),
                   rel(Path(__file__)): ps.sha256_file(Path(__file__)),
                   'src/evaluation/pile_splicing.py': ps.sha256_file(REPO / 'src/evaluation/pile_splicing.py'),
                   rel(manifest_path): ps.sha256_file(manifest_path)},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'transformers': transformers.__version__, 'zstandard': zstandard.__version__},
        'runtime_s': round(time.time() - t0, 1),
    }
    info_path.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {rel(manifest_path)} ({len(rows)} rows) and {rel(info_path)}')
    print(f"eval sources: {info['eval_sources']}; truncated eval docs: {info['eval_truncated']}")


if __name__ == '__main__':
    main()
