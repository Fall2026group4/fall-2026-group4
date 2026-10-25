"""Stage 6a: baseline steering with the stage-2 truth directions (no SAE).

Readout: for each subject's built-in true/false pair,
    score = log P(true sentence) - log P(false sentence)
Steering adds  alpha x unit direction x (typical residual norm at that layer)
to every token except BOS, at blocks.L.hook_resid_pre.
    effect   = mean score with steering - mean score without   (> 0 = pushed towards 'true')
    fluency  = change in mean per-token log-prob of both sentences (< 0 = text got less likely)
"""

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

STEER_DATASETS = ["cities", "neg_cities", "larger_than", "smaller_than"]
TWIN = {"cities": "neg_cities", "neg_cities": "cities", "larger_than": "smaller_than",
        "smaller_than": "larger_than"}


def test_pairs(meta, n_pairs=60, seed=0):
    """(true_statement, false_statement) for test subjects that have both versions."""
    test = meta[meta["split"] == "test"]
    rows = []
    for _, g in test.groupby("subject"):
        if g["label"].nunique() == 2:
            rows.append((g.loc[g["label"] == 1, "statement"].iloc[0], g.loc[g["label"] == 0, "statement"].iloc[0]))
    rng = np.random.default_rng(seed)
    if len(rows) > n_pairs:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n_pairs, replace=False))]
    return rows


def layer_norms(acts):
    """Typical size of the residual stream per layer (mean L2 norm at the last token)."""
    return np.linalg.norm(acts, axis=2).mean(axis=0)


def _sentence_logprob(model, tokens, hooks):
    import torch

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        logp = torch.log_softmax(logits[0], dim=-1)
    n = int(tokens.shape[1])
    total = sum(float(logp[i, int(tokens[0, i + 1])]) for i in range(n - 1))
    return total, total / (n - 1)


def _hooks(layer, vector):
    """Add `vector` at blocks.<layer>.hook_resid_pre on every token except BOS."""
    if vector is None:
        return []

    def add(resid, hook):
        resid[:, 1:, :] += vector
        return resid

    return [(f"blocks.{layer}.hook_resid_pre", add)]


def score_pairs(model, pairs, layer=None, vector=None):
    """Mean true-minus-false score and mean per-token log-prob over all pairs."""
    hooks = _hooks(layer, vector)
    gaps, fluency = [], []
    for true_s, false_s in pairs:
        t_total, t_tok = _sentence_logprob(model, model.to_tokens(true_s, prepend_bos=True), hooks)
        f_total, f_tok = _sentence_logprob(model, model.to_tokens(false_s, prepend_bos=True), hooks)
        gaps.append(t_total - f_total)
        fluency.append((t_tok + f_tok) / 2)
    return float(np.mean(gaps)), float(np.mean(fluency)), float(np.mean(np.array(gaps) > 0))


def make_vector(direction, norm, alpha):
    import torch

    unit = direction / (np.linalg.norm(direction) + 1e-12)
    return torch.tensor(alpha * norm * unit, dtype=torch.float32)


def steering_grid(model, pairs_by_dataset, directions, norms, layers, alphas, sources, n_random=2, seed=0):
    """Run every dataset x direction source x layer x alpha; return one row per setting.

    directions[name] : [13, 768] mass-mean directions fitted on that dataset
    sources          : which directions to try, from {"own", "twin", "random"}
    """
    rng = np.random.default_rng(seed)
    randoms = [rng.normal(size=768) for _ in range(n_random)]
    rows = []
    for name, pairs in pairs_by_dataset.items():
        base_gap, base_flu, base_acc = score_pairs(model, pairs)
        rows.append({"dataset": name, "source": "none", "layer": None, "alpha": 0.0,
                     "mean_gap": round(base_gap, 4), "pair_accuracy": round(base_acc, 4),
                     "effect": 0.0, "fluency_change": 0.0})
        options = []
        if "own" in sources:
            options.append(("own", lambda L, n=name: directions[n][L]))
        if "twin" in sources:
            options.append(("twin", lambda L, n=name: directions[TWIN[n]][L]))
        if "random" in sources:
            options += [(f"random_{i + 1}", lambda L, r=r: r) for i, r in enumerate(randoms)]
        for source, get_dir in options:
            for layer in layers:
                for alpha in alphas:
                    vec = make_vector(get_dir(layer), norms[name][layer], alpha)
                    gap, flu, acc = score_pairs(model, pairs, layer, vec)
                    rows.append({"dataset": name, "source": source, "layer": f"L{layer}", "alpha": alpha,
                                 "mean_gap": round(gap, 4), "pair_accuracy": round(acc, 4),
                                 "effect": round(gap - base_gap, 4), "fluency_change": round(flu - base_flu, 4)})
            logger.info(f"    {name}: {source} done")
    return pd.DataFrame(rows)