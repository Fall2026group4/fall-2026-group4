#!/usr/bin/env python3
"""XB step 2: matched cross-model SAE splicing (GPT-2 Small and Gemma-2-2B under one protocol; 2026-11-21).

Pre-registration: results/cross_model/xb_matched_splicing/preregistration.json (frozen before any output).
Re-uses M6's validated helpers unchanged (src/evaluation/pile_splicing.py, byte-identical to origin/main):
JumpReLUSAE, masks, norm-matched random error, reconstruction/behaviour sums. GPT-2 SAEs are loaded with the
existing ladder loader (scripts/shared/sae_ladder_evaluation.py: old in-house, R2, res-jb).

One document per forward pass (batch 1, no padding), as M6. Per document: one original pass capturing every layer,
then one pass per (layer, condition). Output: per-document sums in long format, appended per document so a run can
resume (``--resume``); a finished run refuses to be overwritten.

Run from the repository root, e.g.
  python scripts/cross_model/02_matched_splicing.py --model gemma --corpus pile --context shared --saes canonical
  python scripts/cross_model/02_matched_splicing.py --model gpt2 --corpus owt_xm --context shared --saes resjb,old,r2
  add --smoke for 8 documents and 2 layers (writes to smoke/)
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402
from src.models.backends import make_backend  # noqa: E402

XB = REPO / 'results/cross_model/xb_matched_splicing'
GS = Path.home() / '.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/snapshots'
CANON = [105, 102, 141, 59, 124, 68, 70, 69, 71, 73, 77, 80, 82, 84, 84, 78, 78, 77, 74, 73, 71, 70, 72, 75, 73, 116]
TRIPLETS = {'gpt2': [1, 3, 6, 8, 9, 11], 'gemma': [1, 5, 12, 16, 19, 23]}
MODEL_SALT = {'gpt2': 11, 'gemma': 29}


# ------------------------------------------------------------------------------------------------- SAE sources
_LAD = None


def _ladder():
    global _LAD
    if _LAD is not None:
        return _LAD
    spec = importlib.util.spec_from_file_location('ladder', REPO / 'scripts/shared/sae_ladder_evaluation.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _LAD = m
    return m


class _Wrap:
    """Uniform SAE callable: float32 [N, d] -> (x_hat float32 [N, d], L0 per token [N])."""

    def __init__(self, sae, kind):
        self.sae, self.kind, self.d_sae = sae, kind, None

    @torch.no_grad()
    def __call__(self, x):
        if self.kind == 'gemmascope':
            self.d_sae = int(self.sae.W_enc.shape[1])
            return self.sae(x)
        h, z, _ = self.sae(x)
        self.d_sae = int(z.shape[-1])
        return h.float(), (z != 0).sum(-1)


def gemma_scope_path(L, width, l0):
    hits = glob.glob(str(GS / f'*/layer_{L}/width_{width}/average_l0_{l0}/params.npz'))
    if not hits:
        raise FileNotFoundError(f'Gemma Scope layer {L} width {width} L0 {l0} not cached')
    return Path(hits[0])


def sae_specs(model, layer, which):
    """List of dicts: family, sae_id, seed, d_sae, loader()."""
    out = []
    if model == 'gemma':
        if 'canonical' in which:
            p = gemma_scope_path(layer, '16k', CANON[layer])
            out.append(dict(family='gemmascope_16k_canonical', sae_id=f'layer_{layer}/width_16k/average_l0_{CANON[layer]}',
                            seed=-1, d_sae=16384, path=p))
        for tok in which:
            if tok.startswith('ladder:'):      # ladder:<width>:<l0> pairs given explicitly
                _, w, l0 = tok.split(':')
                p = gemma_scope_path(layer, w, l0)
                out.append(dict(family=f'gemmascope_{w}', sae_id=f'layer_{layer}/width_{w}/average_l0_{l0}', seed=-1,
                                d_sae={'16k': 16384, '32k': 32768, '65k': 65536}[w], path=p))
        return out
    lad = _ladder()
    srcs = lad.sources(layer, [w for w in which if w in ('old', 'r2', 'resjb')])
    for s in srcs:
        fam = s['family']
        name = s['name'] if fam != 'resjb' else 'resjb_relu_24576'
        out.append(dict(family=fam if fam != 'old' else f'old_{s["name"]}', sae_id=f'{fam}/{name}', seed=s['seed'],
                        d_sae=None, src=s, lad=lad))
    return out


def load_sae(model, spec, device):
    if model == 'gemma':
        params = dict(np.load(spec['path']))
        return _Wrap(ps.JumpReLUSAE(params, 2304, spec['d_sae'], device=device), 'gemmascope')
    return _Wrap(spec['lad'].load(spec['src'], spec['layer'], device), 'inhouse')


# --------------------------------------------------------------------------------------------------- documents
def doc_texts(corpus, man):
    if corpus == 'pile':
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('monology/pile-uncopyrighted', 'val.jsonl.zst', repo_type='dataset',
                            revision='3be90335b66f24456a5d6659d9c8d208c0357119')
        objs = ps.load_lines(Path(p), man.doc_id.tolist())
        return {i: o['text'] for i, o in objs.items()}
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('Skylion007/openwebtext', 'plain_text/train-00077-of-00080.parquet', repo_type='dataset')
    df = pd.read_parquet(p, columns=['text'])
    return {int(i): df.text.iloc[int(i)] for i in man.doc_id}


def doc_ids(bk, model, context, text, r):
    if context == 'shared':
        span = text[:int(r.span_chars)]
        ids = bk.tokenize(span)
        want = r.gpt2_span_ids_sha256 if model == 'gpt2' else r.gemma_span_ids_sha256
        if ps.sha256_token_ids(ids) != want:
            raise SystemExit(f'doc {r.doc_id}: span token ids differ from the manifest')
        return [bk.bos_id] + ids
    ids = [bk.bos_id] + bk.tokenize(text)[:1023]
    if model == 'gemma' and ps.sha256_token_ids(ids) != r.gemma_own_ids_sha256:
        raise SystemExit(f'doc {r.doc_id}: own-1024 ids differ from the manifest')
    return ids


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gpt2', 'gemma'])
    ap.add_argument('--corpus', required=True, choices=['pile', 'owt_xm'])
    ap.add_argument('--context', required=True, choices=['shared', 'own'])
    ap.add_argument('--saes', required=True, help='gemma: canonical[,ladder:<w>:<l0>...]; gpt2: resjb,old,r2')
    ap.add_argument('--layers', default='all', help="'all', 'triplets' or a comma list")
    ap.add_argument('--tag', default='main')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    run = f'{a.model}_{a.corpus}_{a.context}_{a.tag}'
    out_dir = XB / ('smoke' if a.smoke else 'runs') / run
    rows_path, done_path = out_dir / 'per_doc.csv', out_dir / 'DONE.json'
    if done_path.exists():
        raise SystemExit(f'{run} finished already; refusing to overwrite')
    if rows_path.exists() and not a.resume:
        raise SystemExit(f'{rows_path} exists; pass --resume to continue')
    out_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(XB / f'samples/{a.corpus}_manifest.csv')
    ev, ref = man[man.set == 'eval'].sort_values('rank'), man[man.set == 'reference'].sort_values('rank')
    bk = make_backend(a.model)
    nL = bk.n_layers
    if a.layers == 'all':
        layers = list(range(nL))
    elif a.layers == 'triplets':
        layers = TRIPLETS[a.model]
    else:
        layers = [int(x) for x in a.layers.split(',')]
    if a.smoke:
        layers, ev, ref = layers[:2] if a.layers == 'all' else layers[:2], ev.head(8), ref.head(8)
    which = a.saes.split(',')
    texts = doc_texts(a.corpus, pd.concat([ev, ref]))
    dev = bk.device

    # SAE specs per layer. Amendment A2: Gemma runs layer-major (one layer's SAEs on the GPU at a time, documents in
    # the inner loop, original pass repeated per layer); GPT-2 runs document-major with all SAEs resident.
    specs = {}
    for L in layers:
        sp = sae_specs(a.model, L, which)
        for s in sp:
            s['layer'] = L
        specs[L] = sp
    n_sae = sum(len(v) for v in specs.values())
    blocks = [[L] for L in layers]          # A2 (extended to GPT-2, A3): layer-major for both models
    print(f'{run}: {len(layers)} layers, {n_sae} SAEs, {len(ev)} eval docs, {len(blocks)} blocks', flush=True)

    # mean ablation vectors from the reference set
    sums = {L: torch.zeros(bk.d_model, dtype=torch.float64, device=dev) for L in layers}
    cnt = 0
    for r in ref.itertuples():
        ids = torch.tensor([doc_ids(bk, a.model, a.context, texts[r.doc_id], r)])
        _, cap = bk.run(ids, capture=set(layers))
        for L in layers:
            sums[L] += cap[L][0, 1:].double().sum(0)
        cnt += ids.shape[1] - 1
        del cap
    means = {L: (sums[L] / cnt).float() for L in layers}

    # resume (A1 + A2): a (document, layer) cell is done only if its mean-ablation row and its sum_x entry exist;
    # rows of incomplete cells (and control rows of documents whose first block is incomplete) are dropped.
    sumx_path = out_dir / 'sum_x.npz'
    sumx = dict(np.load(sumx_path)) if sumx_path.exists() else {}
    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        ma = prev[prev.cond == 'mean_ablation']
        done = {(int(r), int(L)) for r, L in zip(ma['rank'], ma.layer) if f'{int(r)}_{int(L)}' in sumx}
        first = blocks[0][0]
        keep = [((int(r), int(L)) in done) if c not in ('original_repeat', 'identity') else ((int(r), first) in done)
                for r, L, c in zip(prev['rank'], prev.layer, prev.cond)]
        if not all(keep):
            prev[keep].to_csv(rows_path, index=False)
            print(f'resume: dropped {len(keep) - sum(keep)} rows of incomplete cells', flush=True)
        print(f'resume: {len(done)} (doc, layer) cells done', flush=True)
    t0 = time.time()
    header = not rows_path.exists()
    for bi, blk in enumerate(blocks):
        todo = [r for r in ev.itertuples() if not all((int(r.rank), L) in done for L in blk)]
        if not todo:
            continue
        saes = {L: [(s, load_sae(a.model, s, dev)) for s in specs[L]] for L in blk}
        for k, r in enumerate(ev.itertuples()):
            if all((int(r.rank), L) in done for L in blk):
                continue
            ids_list = doc_ids(bk, a.model, a.context, texts[r.doc_id], r)
            ids = torch.tensor([ids_list])
            T = ids.shape[1]
            am = torch.ones_like(ids)
            smask = ps.splice_mask(am)[0].to(dev)
            pmask = ps.prediction_mask(am).to(dev)
            ids_d = ids.to(dev)
            logits0, cap = bk.run(ids, capture=set(blk))
            ps.assert_finite('logits0', logits0)
            rows = []

            def beh(logits_new, cond, L, fam='-', sid='-', seed=-1, extra=None):
                b_ = ps.behaviour_sums(logits0, logits_new, ids_d, pmask)
                d = dict(rank=int(r.rank), doc_id=int(r.doc_id), layer=L, depth=bk.depth_fraction(L), family=fam,
                         sae_id=sid, seed=seed, cond=cond, n_tok=int(T - 1), **b_)
                if extra:
                    d.update(extra)
                rows.append(d)

            if k < 8 and bi == 0:   # determinism and plumbing controls on the first 8 documents
                lr, _ = bk.run(ids)
                beh(lr, 'original_repeat', -1)
                Lc = blk[len(blk) // 2]
                li, _ = bk.run(ids, splices={Lc: [(0, lambda x: x, smask)]})
                beh(li, 'identity', Lc)
            for L in blk:
                if (int(r.rank), L) in done:
                    continue
                x = cap[L][0][smask].float()
                gen = torch.Generator().manual_seed(MODEL_SALT[a.model] * 1_000_003 + (1000 + L) * 10_007 + int(r.rank))
                u = torch.randn(x.shape, generator=gen, dtype=torch.float32)
                u = (u / u.norm(dim=-1, keepdim=True)).to(dev)
                m = means[L]
                lm, _ = bk.run(ids, splices={L: [(0, lambda z, m=m: m.expand_as(z), smask)]})
                beh(lm, 'mean_ablation', L)
                for s, sae in saes[L]:
                    xh, l0 = sae(x)
                    ps.assert_finite('x_hat', xh)
                    rec = ps.reconstruction_sums(x, xh, l0)
                    rec.pop('sum_x')
                    extra = {**rec, 'd_sae': sae.d_sae}
                    ls, _ = bk.run(ids, splices={L: [(0, lambda z, xh=xh: xh, smask)]})
                    beh(ls, 'sae', L, s['family'], s['sae_id'], s['seed'], extra)
                    xr = x + u * (x - xh).norm(dim=-1, keepdim=True)
                    lr_, _ = bk.run(ids, splices={L: [(0, lambda z, xr=xr: xr, smask)]})
                    beh(lr_, 'random', L, s['family'], s['sae_id'], s['seed'],
                        {'err_norm_sum': float((x - xh).norm(dim=-1).sum())})
                sumx[f'{r.rank}_{L}'] = x.double().sum(0).cpu().numpy()
            df = pd.DataFrame(rows)
            ps.assert_finite_values('row', {c: float(df[c].sum()) for c in ('ce_orig_sum', 'ce_new_sum', 'kl_sum')})
            df.to_csv(rows_path, mode='a', header=header, index=False)
            header = False
            for L in blk:
                done.add((int(r.rank), L))
            if (k + 1) % 16 == 0 or k == len(ev) - 1:
                np.savez(sumx_path, **sumx)
                print(f'block {bi + 1}/{len(blocks)} doc {k + 1}/{len(ev)}  {(time.time() - t0) / 60:.1f} min', flush=True)
        del saes
        torch.cuda.empty_cache()
    np.savez(sumx_path, **sumx)
    info = dict(run=run, args=vars(a), layers=layers, n_sae=n_sae, minutes=round((time.time() - t0) / 60, 2),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                sae_ids={L: [s['sae_id'] for s in specs[L]] for L in layers},
                script_sha256=ps.sha256_file(Path(__file__)), backends_sha256=ps.sha256_file(REPO / 'src/models/backends.py'),
                pile_splicing_sha256=ps.sha256_file(REPO / 'src/evaluation/pile_splicing.py'),
                prereg_sha256=ps.sha256_file(XB / 'preregistration.json'))
    done_path.write_text(json.dumps(info, indent=1, default=str))
    print('DONE', run, flush=True)


if __name__ == '__main__':
    main()
