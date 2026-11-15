#!/usr/bin/env python3
"""Integration test of script 31 (TSP-R Phase 1 evaluation) on existing checkpoints. No training; no TEST data.

Setup, in a gitignored smoke folder (replication/smoke_eval/):
  * 2 seeds (42 as "existing", 1042 as "new"; both R2 TopK L8 bases);
  * arms ctrl / rand / pca / shuf and the base point to the unchanged R2 base checkpoint of their seed;
  * arm tsp points to a null-decoder copy (decoder weights 0: constant reconstruction, all information lost);
  * the ladder CSV is the pilot's R2-base ladder rows copied to every arm (ladder guards then compare equal rows).
For speed the MLP readers use 1 net instead of 10 / 5 (the reader function itself is unchanged).

Expected, and asserted:
  1. every comparison among {base, ctrl, rand, pca, shuf} is exactly 0 with se 0, for every reader;
  2. tsp - ctrl gaps are > 0 for every reader (null decoder loses everything); H1 not rejected; label HARM;
  3. OWT NMSE guard fails for the degraded tsp (trade-off flagged); L0 guard passes;
  4. the TEST and final stages refuse to run because the analysis is not committed.
Writes replication/eval_integration_test.json.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('s31', REPO / 'scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py')
m31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m31)
S30 = m31.S30
REAL = S30.OUT
SMOKE = REAL / 'smoke_eval'
RESUME = '--resume' in sys.argv          # reuse the smoke folder's cached per-SAE npz files (setup is redone)
if SMOKE.exists() and not RESUME:
    sys.exit(f'{SMOKE} exists; move it aside or pass --resume.')
SMOKE.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 1042)
R2 = S30.R2
base = {s: R2 / f'layer_8/topk_seed{s}/checkpoint_final.pt' for s in SEEDS}

# degraded copies for the tsp arm
noisy = {}
for s in SEEDS:
    ck = torch.load(base[s], map_location='cpu', weights_only=False)
    # Null decoder: x_hat = output bias for every input, so all information is provably lost (data processing).
    # (Attempt 1 used decoder noise; it raised NMSE 0.096 -> 0.225 yet slightly improved re-fit gaps, so it was not a
    # guaranteed known effect. Kept in smoke_eval_attempt1/.)
    ck['model_state_dict']['decoder.weight'] = torch.zeros_like(ck['model_state_dict']['decoder.weight'])
    noisy[s] = SMOKE / f'null_decoder_seed{s}.pt'
    if not noisy[s].exists():
        torch.save(ck, noisy[s])

# patch the module constants to the smoke design
S30.SEEDS, S30.OLD_SEEDS, S30.NEW_SEEDS = SEEDS, (42,), (1042,)
S30.base_path = lambda s: base[s]
S30.LEDGER = SMOKE / 'run_ledger.csv'
m31.OUT, m31.EV, m31.AN = SMOKE, SMOKE / 'eval', SMOKE / 'eval/analysis'
_mlp = m31.mlp
m31.mlp = lambda Xtr, ytr, Xte, n, hidden: _mlp(Xtr, ytr, Xte, 1, hidden)
shutil.copy(REAL / 'preregistration.json', SMOKE / 'preregistration.json')
M = pd.read_csv(REAL / 'plan_matrix.csv')
M = M[M.seed.isin(SEEDS)].copy()
M['checkpoint'] = [str((noisy[s] if a == 'tsp' else base[s]).relative_to(REPO)) for s, a in zip(M.seed, M.arm)]
M.to_csv(SMOKE / 'plan_matrix.csv', index=False)
pd.DataFrame(dict(run_id=M.run_id, sha256=[S30.sha(REPO / c) for c in M.checkpoint])).to_csv(S30.LEDGER, index=False)
(SMOKE / 'plan_info.json').write_text(json.dumps(dict(existing_base_sha256={'42': S30.sha(base[42])})))
(SMOKE / 'base_qc.json').write_text(json.dumps({'new_sha256': {'1042': S30.sha(base[1042])}, 'pass': True}))
lad = pd.read_csv(REPO / 'results/geometry_of_truth/truth_subspace_preservation/pilot/eval/ladder_layer8.csv')
lad = lad[(lad.family == 'r2_base') & lad.seed.isin(SEEDS)]
(SMOKE / 'eval').mkdir(exist_ok=True)
pd.concat([lad.assign(family=a) for a in ('base',) + S30.ARMS]).to_csv(SMOKE / 'eval/ladder_fold0.csv', index=False)

m31.stage_cities()
m31.stage_owt()
m31.stage_analyze()

AN = m31.AN
comp = pd.read_csv(AN / 'comparisons_cities.csv')
same = {'base', 'ctrl', 'rand', 'pca', 'shuf'}
z = comp[comp.arm.isin(same) & comp.ref.isin(same)]
ok1 = bool(len(z) > 0 and (z.estimate.abs() == 0).all() and (z.se.abs() == 0).all())
tc = comp[(comp.arm == 'tsp') & (comp.ref == 'ctrl')].set_index('metric')
ok2 = bool((tc.ci_low > 0).all())
prim = json.loads((AN / 'primary.json').read_text())
ok2 = ok2 and not prim['gatekeeping']['H1']['rejected'] and prim['H1_label'] == 'HARM'
gd = pd.read_csv(AN / 'cost_guards.csv').set_index('guard')
ok3 = bool(not gd.loc['owt_nmse relative (tsp/ctrl - 1)', 'passed'] and gd.loc['L0 = 128 exactly on OWT, GoT, RepE', 'passed']
           and prim['trade_off'])
refused = []
for st in (m31.stage_test, m31.stage_final):
    try:
        st()
        refused.append(False)
    except SystemExit as e:
        refused.append('must be committed' in str(e) or 'not committed' in str(e))
ok4 = all(refused)
out = dict(identical_arms_exactly_zero=ok1, degraded_tsp_worse_and_H1_not_rejected=ok2,
           nmse_guard_flags_degraded_and_l0_passes=ok3, test_and_final_refuse_before_commit=ok4,
           tsp_minus_ctrl=tc[['estimate', 'ci_low', 'ci_high', 'p']].to_dict('index'), H1_label=prim['H1_label'],
           n_identical_comparisons=len(z), script31_sha256=S30.sha(REPO / 'scripts/geometry_of_truth/31_tsp_r_phase1_evaluation.py'),
           stats_sha256=S30.sha(REPO / 'src/evaluation/tspr_stats.py'), passed=bool(ok1 and ok2 and ok3 and ok4))
(REAL / 'eval_integration_test.json').write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float))
sys.exit(0 if out['passed'] else 1)
