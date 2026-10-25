"""Statistics for the steering-against-random-controls tests (no torch needed).

The causal test compares one chosen direction (the honesty direction, or one
SAE feature) with many random directions of the same size. The empirical
p-value is the share of directions (counting the chosen one) that are at least
as strong as the chosen one. With n random controls the smallest p-value that
can ever be reported is 1 / (n + 1): 20 controls give 0.048, 100 give 0.0099,
200 give 0.0050. A null result from a test that cannot reach p < 0.05 or
p < 0.005 says very little, so every summary also reports that floor.
"""

import numpy as np


def min_attainable_p(n_random):
    """Smallest empirical p-value possible with n_random random controls."""
    return 1.0 / (n_random + 1)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, float)
    n, order = len(p), np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def rank_summary(directional, own_key, rng, test_mask, extra=None, n_boot=1000):
    """Compare one direction's per-item effect with the random directions' effects.

    directional : {name: per-item directional effect}; the key own_key is the chosen one.
    """
    own = directional[own_key]
    rnd = np.array([abs(v.mean()) for k, v in directional.items() if k != own_key])
    boot = rng.integers(0, len(own), size=(n_boot, len(own)))
    ci = np.percentile(own[boot].mean(1), [2.5, 97.5])
    out = dict(own_directional=float(own.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
               own_directional_test=float(own[test_mask].mean()),
               random_abs_mean=float(rnd.mean()), random_abs_max=float(rnd.max()),
               own_rank=int(1 + (rnd > abs(own.mean())).sum()), n_directions=len(rnd) + 1,
               n_random=len(rnd), min_attainable_p=min_attainable_p(len(rnd)),
               p_value=(1 + (rnd >= abs(own.mean())).sum()) / (len(rnd) + 1))
    out.update(extra or {})
    return out
