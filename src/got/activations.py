"""Cache GPT-2 residual-stream activations for the Geometry of Truth statements.

Reuses the RepE setup: GPT-2 Small in TransformerLens, 13 positions
(blocks.0..11.hook_resid_pre + blocks.11.hook_resid_post). The prompt is the
statement itself, so 'mean' averages every token except the BOS token.
"""

import numpy as np
import pandas as pd

from src.repe.activations import D_MODEL, HOOKS, load_model  # noqa: F401  (load_model re-exported)

import logging

logger = logging.getLogger(__name__)


def cache_statements(model, df):
    """Activations for every statement, one at a time.

    Returns last [n, 13, 768] (final token, the period), mean [n, 13, 768] (all non-BOS tokens),
    and meta (df plus n_tokens).
    """
    import torch

    n = len(df)
    last = np.zeros((n, len(HOOKS), D_MODEL), dtype=np.float32)
    mean = np.zeros_like(last)
    n_tokens = []
    hook_set = set(HOOKS)
    with torch.no_grad():
        for i, statement in enumerate(df["statement"]):
            tokens = model.to_tokens(statement, prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_set)
            for j, hook in enumerate(HOOKS):
                acts = cache[hook][0].detach().cpu().numpy()
                last[i, j] = acts[-1]
                mean[i, j] = acts[1:].mean(axis=0)
            n_tokens.append(int(tokens.shape[1]))
            if (i + 1) % 500 == 0 or i + 1 == n:
                logger.info(f"    cached {i + 1}/{n}")
    meta = df.reset_index(drop=True).copy()
    meta["n_tokens"] = n_tokens
    return last, mean, meta


def save_set(directory, name, last, mean, meta):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{name}.npz", last=last, mean=mean)
    meta.to_csv(directory / f"{name}.csv", index=False)


def load_set(directory, name):
    data = np.load(directory / f"{name}.npz")
    return data["last"], data["mean"], pd.read_csv(directory / f"{name}.csv")