#!/usr/bin/env python3
"""Corrected Gemma-2-2B truth-direction steering sweep (Task D).

Re-examines the observation that steering at +0.5 and -0.5 both increased "TRUE"
judgments (aditi:scripts/run_gemma_baseline.py, results/tables/gemma/cities/judgment_summary.csv).

Differences from the original script (each is a deliberate, documented correction):
  * Grouped-by-city splits from data/got/splits/cities_splits.csv (original used a random row split,
    which leaks cities between train and test).
  * Continuous behavioural outcome: logit difference LD = logit(" TRUE") - logit(" FALSE") at the
    final position, in addition to %judged TRUE and accuracy.
  * Symmetric, fine-grained strength grid including the small values requested.
  * Norm-matched random-direction controls at every strength (original had none).
  * Strengths reported both as multiples of mean activation norm (original unit) and as multiples of
    the true/false separation along the direction (||mu_true - mu_false||).
  * Two direction constructions: 'bare' (original: last token of the bare statement) and
    'context' (last token of the actual judgment prompt), to test the context-mismatch hypothesis.
  * Two position modes: 'segment' (original: every token of the test statement + suffix) and 'final'
    (only the last token, where the answer is read out).
The original layer (11, resid_post) and few-shot prompts are kept so results are comparable.

Run from repository root:
    python scripts/gemma/01_steering_sweep.py            # full
    python scripts/gemma/01_steering_sweep.py --smoke    # tiny check
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

JUDGE_SHOTS = ("The city of Paris is in France. This statement is: TRUE\n"
               "The city of Paris is in Japan. This statement is: FALSE\n"
               "The city of Tokyo is in Japan. This statement is: TRUE\n"
               "The city of Tokyo is in Brazil. This statement is: FALSE\n")
SUFFIX = " This statement is:"
GRID = (-0.5, -0.25, -0.10, -0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.5)
OUT = Path('results/gemma/steering_sweep_layer11')
SEED = 20261101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=11)
    ap.add_argument('--n-random', type=int, default=20)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    out = OUT / 'smoke' if a.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(SEED)

    df = pd.read_csv('data/got/cities.csv')
    sp = pd.read_csv('data/got/splits/cities_splits.csv')
    assert (sp['row_id'].values == np.arange(len(df))).all() and (sp['label'].values == df['label'].values).all()
    df['split'] = sp['split'].values
    if a.smoke:
        df = pd.concat([g.head(24) for _, g in df.groupby('split')])
        a.n_random = 2

    # Backend: Hugging Face transformers, loaded directly onto the GPU in bf16.
    # TransformerLens (used by the original script) materialises the model on the CPU and is OOM-killed on
    # this 15 GB host (14 GB RSS observed, see logs). The output of HF decoder layer L is exactly TL's
    # blocks.L.hook_resid_post (Gemma uses RMSNorm, so TL weight processing does not change the residual
    # stream). Eager attention is used because Gemma-2 applies attention-logit soft-capping.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b', dtype=torch.bfloat16, device_map=device,
                                                 attn_implementation='eager')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = a.layer
    hook = f'blocks.{L}.hook_resid_post (HF model.model.layers[{L}] output)'
    layer_module = model.model.layers[L]
    d_model = model.config.hidden_size

    def ids(text, bos=True):
        return tok(text, add_special_tokens=bos)['input_ids']

    t_ids, f_ids = ids(' TRUE', False), ids(' FALSE', False)
    assert len(t_ids) == 1 and len(f_ids) == 1, (t_ids, f_ids)
    t_id, f_id = int(t_ids[0]), int(f_ids[0])

    shots_tok = ids(JUDGE_SHOTS)
    prompts = [JUDGE_SHOTS + s.strip() + SUFFIX for s in df['statement']]
    seg = []
    for p in prompts:
        tk = ids(p)
        assert tk[:len(shots_tok)] == shots_tok, 'tokenisation boundary differs between shots and prompt'
        seg.append(len(tk) - len(shots_tok))
    df['prompt'], df['seg_len'] = prompts, seg

    state = {}

    def layer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if state.get('capture'):
            state['captured'] = hs[:, -1, :].float().cpu()
        fn = state.get('fn')
        if fn is None:
            return None
        hs = fn(hs)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    layer_module.register_forward_hook(layer_hook)

    def forward(texts):
        enc = tok(texts, return_tensors='pt', padding=True).to(device)
        return model(**enc).logits[:, -1, :].float()

    @torch.no_grad()
    def last_resid(texts):
        rows = []
        state.update(capture=True, fn=None)
        for i in range(0, len(texts), a.batch):
            forward(texts[i:i + a.batch])
            rows.append(state['captured'])
        state['capture'] = False
        return torch.cat(rows)

    def make_hook(vec, seg_lens, mode):
        def fn(resid):
            T = resid.shape[1]
            v = vec.to(resid.dtype)
            if mode == 'final':
                resid = resid.clone()
                resid[:, -1, :] = resid[:, -1, :] + v
                return resid
            start = T - seg_lens.to(resid.device)
            mask = (torch.arange(T, device=resid.device)[None, :] >= start[:, None]).to(resid.dtype)
            return resid + mask[:, :, None] * v
        return fn

    @torch.no_grad()
    def judge(sub, vec=None, mode='segment'):
        lds = []
        for i in range(0, len(sub), a.batch):
            b = sub.iloc[i:i + a.batch]
            state['fn'] = None if vec is None else make_hook(vec, torch.tensor(b['seg_len'].values), mode)
            logits = forward(b['prompt'].tolist())
            lds.append((logits[:, t_id] - logits[:, f_id]).cpu())
        state['fn'] = None
        return torch.cat(lds).numpy()

    # padding sanity check: batched (left-padded) vs single-prompt logit difference
    probe = df[df['split'] == 'val'].iloc[:4]
    batched = judge(probe)
    single = np.array([judge(probe.iloc[[k]])[0] for k in range(len(probe))])
    pad_err = float(np.abs(batched - single).max())
    print('left-padding max |LD batched - single| =', pad_err, flush=True)

    tr, va = df['split'] == 'train', df['split'] == 'val'
    y = df['label'].values
    directions, seps = {}, {}
    bare = last_resid(df['statement'].str.strip().tolist())
    ctx = last_resid(df['prompt'].tolist())
    for name, A in (('bare', bare), ('context', ctx)):
        d = A[tr.values & (y == 1)].mean(0) - A[tr.values & (y == 0)].mean(0)
        directions[name] = (d / d.norm()).to(device)
        seps[name] = float(d.norm())
        proj = (A @ (d / d.norm())).numpy()
        print(f'{name}: separation ||mu_T-mu_F||={seps[name]:.2f}, '
              f'val AUROC of projection={roc_auc_score(y[va], proj[va]):.4f}', flush=True)
    act_norm = float(bare.norm(dim=1).mean())          # original strength unit
    ctx_norm = float(ctx.norm(dim=1).mean())
    cos_bc = float(directions['bare'] @ directions['context'])
    print(f'mean act norm bare={act_norm:.1f} ctx-final={ctx_norm:.1f}; cos(bare,context)={cos_bc:.3f}', flush=True)

    g = torch.Generator().manual_seed(SEED)
    randoms = [torch.randn(d_model, generator=g) for _ in range(a.n_random)]
    randoms = [(r / r.norm()).to(device) for r in randoms]

    rows = []
    for split in ('val', 'test'):
        sub = df[df['split'] == split].reset_index(drop=True)
        ys = sub['label'].values
        base = judge(sub)

        def summarise(ld, **kw):
            judged = ld > 0
            return dict(split=split, n=len(sub), **kw,
                        mean_LD=float(ld.mean()), mean_LD_true_stmts=float(ld[ys == 1].mean()),
                        mean_LD_false_stmts=float(ld[ys == 0].mean()),
                        delta_mean_LD=float(ld.mean() - base.mean()),
                        judged_true_pct=100 * float(judged.mean()),
                        accuracy_pct=100 * float((judged == (ys == 1)).mean()),
                        auroc_LD=float(roc_auc_score(ys, ld)))
        rows.append(summarise(base, direction='none', mode='none', alpha_act_norm=0.0,
                              strength_norm=0.0, strength_in_separations=0.0, control_id=''))
        for mode in ('segment', 'final'):
            for alpha in GRID:
                if alpha == 0:
                    continue
                s = alpha * act_norm
                for name in ('bare', 'context'):
                    ld = judge(sub, s * directions[name], mode)
                    rows.append(summarise(ld, direction=name, mode=mode, alpha_act_norm=alpha, strength_norm=s,
                                          strength_in_separations=s / seps[name], control_id=''))
                for k, r in enumerate(randoms):
                    ld = judge(sub, s * r, mode)
                    rows.append(summarise(ld, direction='random', mode=mode, alpha_act_norm=alpha, strength_norm=s,
                                          strength_in_separations=float('nan'), control_id=f'random_{k:02d}'))
                print(f'{split} {mode} alpha={alpha:+.3f} done', flush=True)
                pd.DataFrame(rows).to_csv(out / 'gemma_steering_sweep_all.csv', index=False)

    res = pd.DataFrame(rows)
    res.to_csv(out / 'gemma_steering_sweep_all.csv', index=False)
    tgt = res[res['direction'].isin(['none', 'bare', 'context'])]
    rnd = res[res['direction'] == 'random'].groupby(['split', 'mode', 'alpha_act_norm']).agg(
        random_mean_delta_LD=('delta_mean_LD', 'mean'), random_sd_delta_LD=('delta_mean_LD', 'std'),
        random_min_delta_LD=('delta_mean_LD', 'min'), random_max_delta_LD=('delta_mean_LD', 'max'),
        random_mean_judged_true_pct=('judged_true_pct', 'mean'),
        random_mean_accuracy_pct=('accuracy_pct', 'mean'), n_random=('delta_mean_LD', 'size')).reset_index()
    summary = tgt.merge(rnd, on=['split', 'mode', 'alpha_act_norm'], how='left')
    summary['z_vs_random'] = (summary['delta_mean_LD'] - summary['random_mean_delta_LD']) / summary['random_sd_delta_LD']
    summary.to_csv(out / 'gemma_steering_sweep.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps(dict(
        model='google/gemma-2-2b (HF transformers, bf16, eager attention)', layer=L, hook=hook, grid=GRID,
        strength_unit='alpha x mean ||resid_post|| at last token of bare statements (original unit)',
        act_norm=act_norm, ctx_final_act_norm=ctx_norm, separation=seps, cos_bare_context=cos_bc,
        n_random=a.n_random, seed=SEED, true_token=t_id, false_token=f_id, left_padding_check_max_abs=pad_err,
        splits={s: int((df['split'] == s).sum()) for s in ('train', 'val', 'test')},
        prompt_format='4-shot judgment prompt; ' + repr(SUFFIX), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
        written_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    print(summary[summary['split'] == 'test'][['mode', 'direction', 'alpha_act_norm', 'strength_in_separations',
                                              'delta_mean_LD', 'random_mean_delta_LD', 'random_sd_delta_LD',
                                              'judged_true_pct', 'accuracy_pct']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
