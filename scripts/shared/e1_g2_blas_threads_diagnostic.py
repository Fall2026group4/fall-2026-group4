"""Diagnostic only (not an experiment; writes nothing in the repo): does E1's G2 reproduce M2 exactly when BLAS runs
single-threaded, as inside M2's loky workers? Uses E1's frozen helper functions unchanged."""
import importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
from threadpoolctl import threadpool_limits
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('e1', REPO / 'scripts/shared/e1_task_l0_matched_topk.py'); E1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E1)
from src.evaluation import faithfulness_metrics as fm
from src.sae.checkpoint_registry import uninterrupted_checkpoint
M2 = E1.load('scripts/shared/m2_four_level_faithfulness.py', 'm2')
pc = pd.read_csv(REPO / 'results/shared/m2_four_level_faithfulness/per_cell.csv')
L, s = 8, 42
with threadpool_limits(limits=1):
    tasks = M2.load_tasks(); origs, bidx = E1.m2_layer_context(M2, tasks, L)
    sae, _ = fm.load_sae(uninterrupted_checkpoint(L, 'topk', s), 'cuda')
    for tn in tasks:
        r = E1.m2_eval(M2, tasks, L, origs, bidx, sae, tn, E1.m2_job_index(M2, tasks, 'topk', s, tn), 'cuda',
                       dict(layer=L, task=tn, source='sae', architecture='topk', seed=s, pca_k=np.nan, matched=True, live_latents=np.nan))
        c = pc[(pc.layer == L) & (pc.task == tn) & (pc.source == 'sae') & (pc.architecture == 'topk') & (pc.seed == s)].iloc[0]
        print(tn, {m: float(abs(r[m] - c[m])) for m in ('refit_logistic_gap', 'frozen_logistic_gap', 'task_nmse', 'refit_mlp_gap')}, flush=True)
