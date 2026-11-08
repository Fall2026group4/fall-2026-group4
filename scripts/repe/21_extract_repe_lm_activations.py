#!/usr/bin/env python3
"""M1 step 2: run ORIGINAL GPT-2 Small (frozen) on the RepE-LM prompts (train + validation only).

Outputs
  data/repe/activations/repe_lm/{final,slot}_layer{L}.npy     float32 [n_rows, 768], blocks.L.hook_resid_pre, L=0..11
      (gitignored; row order = data/repe/repe_lm/prompts.csv)
  results/repe/repe_lm_gates/k1_activation_invariants.json     K1 activation-level checks (all positions, all layers)
  results/repe/repe_lm_gates/k7_behaviour_rows.csv             per (statement, template) output effects for K7

K1 (activation level), for every pair (H vs U, H vs Hp, Na vs Nb):
  * positions before the persona slot: identical residual at every layer (layer 0 exactly; others |diff| <= 1e-4),
  * the slot position differs at layer 0 (the swapped token).
K7 quantities, per (statement, template):
  * final-token KL(p_ref || p_other) in nats for other in {U, Hp} (persona) or {Nb} (neutral),
  * continuation: mean log-prob per token of the statement's remaining tokens (teacher forced) under each variant.

Run from repository root:
    python scripts/repe/21_extract_repe_lm_activations.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / 'data/repe/repe_lm/prompts.csv'
ACT = REPO / 'data/repe/activations/repe_lm'
RES = REPO / 'results/repe/repe_lm_gates'
N_LAYERS, D = 12, 768
STATEMENTS_PER_BATCH = 8
TOL = 1e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    outs = [ACT / f'{p}_layer{L}.npy' for p in ('final', 'slot') for L in range(N_LAYERS)]
    outs += [RES / 'k1_activation_invariants.json', RES / 'k7_behaviour_rows.csv']
    for f in outs:
        if f.exists():
            raise SystemExit(f'{f} exists; refusing to overwrite.')
    pr = json.loads((RES / 'preregistration.json').read_text())
    for rel, h in pr['hashes'].items():
        if rel.endswith('.py') and 'extract' in rel and h != sha(REPO / rel):
            raise SystemExit(f'{rel} hash differs from the pre-registered hash; refusing to run.')
    d = pd.read_csv(PROMPTS)
    assert set(d.split) <= {'train', 'validation'}, 'test rows present: refusing'
    tmpl = json.loads((REPO / 'data/repe/repe_lm/templates.json').read_text())
    d['slot_pos'] = d.template.map({t: v['slot_pos'] for t, v in tmpl.items()})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    names = {f'blocks.{L}.hook_resid_pre' for L in range(N_LAYERS)}
    pad_id = model.tokenizer.eos_token_id

    final = np.zeros((N_LAYERS, len(d), D), np.float32)
    slot = np.zeros((N_LAYERS, len(d), D), np.float32)
    k1 = dict(max_abs_diff_before_slot_layer0=0.0, max_abs_diff_before_slot_by_layer=[0.0] * N_LAYERS,
              min_abs_diff_at_slot_layer0=np.inf, n_pairs_checked=0, n_violations=0)
    beh = []
    ref_of = {'fit': 'H', 'heldout': 'H', 'neutral': 'Na'}
    stmts = d.statement_id.unique()
    for b0 in range(0, len(stmts), STATEMENTS_PER_BATCH):
        sub = d[d.statement_id.isin(stmts[b0:b0 + STATEMENTS_PER_BATCH])]
        seqs = []
        for _, r in sub.iterrows():
            ids = list(map(int, r.token_ids.split()))
            cont = list(map(int, r.continuation_ids.split())) if isinstance(r.continuation_ids, str) else []
            seqs.append(ids + cont)            # continuation appended (teacher forcing); causal => no effect on prompt
        T = max(map(len, seqs))
        toks = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            toks[i, :len(s)] = torch.tensor(s)
        with torch.inference_mode():
            logits, cache = model.run_with_cache(toks.to(device), names_filter=lambda n: n in names)
            logp = logits.float().log_softmax(-1)
        idx = sub.index.to_numpy()
        fpos = torch.tensor(sub.final_pos.to_numpy(), device=device)
        spos = torch.tensor(sub.slot_pos.to_numpy(), device=device)
        ar = torch.arange(len(sub), device=device)
        resid = [cache[f'blocks.{L}.hook_resid_pre'].float() for L in range(N_LAYERS)]
        for L in range(N_LAYERS):
            final[L, idx] = resid[L][ar, fpos].cpu().numpy()
            slot[L, idx] = resid[L][ar, spos].cpu().numpy()
        local = {rid: i for i, rid in enumerate(idx)}
        for (pid, tid), g in sub.groupby(['pair_id', 'template']):
            role = g.template_role.iloc[0]
            ref = g[g.variant == ref_of[role]].iloc[0]
            ir = local[ref.name]
            sp, fp = int(ref.slot_pos), int(ref.final_pos)
            ncont = int(ref.n_continuation)
            row = dict(statement_id=int(ref.statement_id), split=ref.split, template=tid, role=role,
                       factual_label=int(ref.factual_label), n_continuation=ncont)

            def cont_lp(i):
                if ncont == 0:
                    return np.nan
                tgt = toks[i, fp + 1:fp + 1 + ncont].to(device)
                return float(logp[i, fp:fp + ncont].gather(-1, tgt[:, None]).mean())
            row[f'cont_lp_{ref.variant}'] = cont_lp(ir)
            for _, o in g[g.variant != ref.variant].iterrows():
                io = local[o.name]
                k1['n_pairs_checked'] += 1
                for L in range(N_LAYERS):
                    dif = (resid[L][ir, :sp] - resid[L][io, :sp]).abs().max().item()
                    k1['max_abs_diff_before_slot_by_layer'][L] = max(k1['max_abs_diff_before_slot_by_layer'][L], dif)
                    if (L == 0 and dif != 0.0) or dif > TOL:
                        k1['n_violations'] += 1
                at = (resid[0][ir, sp] - resid[0][io, sp]).abs().max().item()
                k1['min_abs_diff_at_slot_layer0'] = min(k1['min_abs_diff_at_slot_layer0'], at)
                p_ref, p_o = logp[ir, fp], logp[io, fp]
                row[f'kl_{ref.variant}_{o.variant}'] = float((p_ref.exp() * (p_ref - p_o)).sum())
                row[f'cont_lp_{o.variant}'] = cont_lp(io)
            beh.append(row)
        print(f'statements {min(b0 + STATEMENTS_PER_BATCH, len(stmts))}/{len(stmts)}', flush=True)
        del cache, logits, logp, resid

    k1['max_abs_diff_before_slot_layer0'] = k1['max_abs_diff_before_slot_by_layer'][0]
    k1['tolerance_layers_ge_1'] = TOL
    k1['pass'] = bool(k1['n_violations'] == 0 and k1['min_abs_diff_at_slot_layer0'] > 0)
    ACT.mkdir(parents=True, exist_ok=True)
    for L in range(N_LAYERS):
        np.save(ACT / f'final_layer{L}.npy', final[L])
        np.save(ACT / f'slot_layer{L}.npy', slot[L])
    meta = dict(prompts_sha256=sha(PROMPTS), n_rows=len(d), hook='blocks.{L}.hook_resid_pre', layers=N_LAYERS,
                positions=['final (last prompt token)', 'slot (persona/neutral token)'], dtype='float32',
                model='gpt2-small (transformer_lens default processing, same as SAE training)', device=device,
                torch=torch.__version__)
    (ACT / 'meta.json').write_text(json.dumps(meta, indent=2))
    k1['activation_meta'] = meta
    (RES / 'k1_activation_invariants.json').write_text(json.dumps(k1, indent=2))
    pd.DataFrame(beh).to_csv(RES / 'k7_behaviour_rows.csv', index=False)
    print(json.dumps({k: v for k, v in k1.items() if k != 'activation_meta'}, indent=1))


if __name__ == '__main__':
    main()
