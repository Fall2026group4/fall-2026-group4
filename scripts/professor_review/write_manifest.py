#!/usr/bin/env python3
"""Write experiment_manifest.csv for every experiment completed in the professor-review corrections."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

OUT = Path('results/professor_review_corrections')


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else 'MISSING'


env = f"torch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
gem = json.loads((OUT / 'gemma' / 'run_info.json').read_text()) if (OUT / 'gemma' / 'run_info.json').exists() else {}
pre = json.loads((OUT / 'causal' / 'preregistration.json').read_text())
rows = [
    dict(task='A', experiment='OWT consistent-protocol retrain + test eval',
         script='scripts/openwebtext/13_consistent_protocol_retrain.py', model='gpt2-small',
         sae='in-house relu/topk/jumprelu, d_sae=3072, target L0=128', layers='2,6,10',
         data='OWT (Skylion007/openwebtext, seed-42 sample) train 101600 / val 12700 / test 12700 tokens',
         seeds='42,1042,2042', optimizer='Adam lr 3e-4, batch 256, 15000 steps, clip 10, uninterrupted',
         intervention_strengths='', eval_n='12700 test tokens', status='completed',
         outputs='owt_consistent_protocol/, reconstruction_comparison.csv', environment=env),
    dict(task='B', experiment='Matched vs unmatched sparsity re-analysis',
         script='scripts/professor_review/analyze_owt_corrections.py', model='gpt2-small',
         sae='in-house, 3072', layers='0-11 (sweep), 2/6/10 (15k)', data='existing sweep (validation) + 15k (test)',
         seeds='42 (sweep)', optimizer='sweep: 1000 steps', intervention_strengths='',
         eval_n='108 sweep runs + 9 original + 27 corrected', status='completed (re-analysis only)',
         outputs='sparsity_comparison.csv, sparsity_matched_cells.csv', environment=env),
    dict(task='C', experiment='200-random-control causal test (honesty direction & SAE features)',
         script='scripts/repe/13_causal_random_controls.py', model='gpt2-small',
         sae='9 layer-6 variants (3 arch x pretrained/recon-only/truth-aware)', layers='6 (hook_resid_pre, final token)',
         data='RepE persona prompts: train 417 stmts (selection), val 90 (strength), test 90 (evaluation)',
         seeds=f"controls {pre['control_seed']} (+per-SAE offsets); bootstrap 0", optimizer='n/a (inference)',
         intervention_strengths=f"norm {pre['chosen_strength_norm']:.4f} ({pre['chosen_strength_multiple']}x ||mean diff||)",
         eval_n='90 test statements x 3830 interventions', status='completed',
         outputs='causal/, causal_controls_200.csv, causal_summary.csv', environment=env),
    dict(task='D', experiment='Gemma-2-2B truth-direction steering sweep with random controls',
         script='scripts/gemma/01_steering_sweep.py', model='google/gemma-2-2b (bf16)',
         sae='none', layers=f"{gem.get('layer', 11)} (hook_resid_post)",
         data=f"GoT cities grouped splits {gem.get('splits', '')}", seeds=str(gem.get('seed', '')),
         optimizer='n/a', intervention_strengths=str(gem.get('grid', '')) + ' x mean act norm',
         eval_n='val + test statements', status='completed' if gem else 'not completed',
         outputs='gemma/, gemma_steering_sweep.csv', environment=env),
    dict(task='E', experiment='GoT layer-10 legacy reproduction + frozen-probe comparison',
         script='scripts/professor_review/got_layer10_frozen_probe.py', model='gpt2-small',
         sae='SAELens gpt2-small-res-jb blocks.10.hook_resid_pre (d_sae 24576)', layers='10',
         data='GoT cities, city-grouped 70/30 (1046/450)', seeds='split 42 (+1..5), probe 42',
         optimizer='LogisticRegression max_iter 2000', intervention_strengths='', eval_n='450 test statements',
         status='completed', outputs='got_layer10/, got_faithfulness.csv', environment=env),
    dict(task='F/G', experiment='RepE test-split + unified fine-tuning comparison on identical held-out sets',
         script='scripts/professor_review/finetuning_repe_got_eval.py', model='gpt2-small',
         sae='9 layer-6 variants', layers='6', data='OWT test; GoT test (452 geo / 596 num); RepE val 180 / test 180 prompts; cross-wording 180',
         seeds='probe 42', optimizer='n/a', intervention_strengths='', eval_n='see data', status='completed',
         outputs='repe_representation.csv, finetuning_comparison.csv, got_faithfulness.csv', environment=env),
    dict(task='I', experiment='Unit tests', script='src/tests/test_professor_review.py', model='', sae='', layers='',
         data='synthetic', seeds='0,1', optimizer='', intervention_strengths='', eval_n='8 tests',
         status='completed (8 passed)', outputs='', environment=env),
]
for r in rows:
    r['script_sha256_16'] = sha(r['script'])
pd.DataFrame(rows).to_csv(OUT / 'experiment_manifest.csv', index=False)
print(pd.DataFrame(rows)[['task', 'status', 'script_sha256_16']].to_string(index=False))
