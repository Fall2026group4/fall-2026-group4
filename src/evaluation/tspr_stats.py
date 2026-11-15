"""Statistics for TSP-R Phase 1 (frozen pre-registration:
results/geometry_of_truth/truth_subspace_preservation/replication/preregistration.json).

Pure functions only (numpy / scipy); unit-tested in src/tests/test_tspr_stats.py.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

SESOI = 0.010


# ------------------------------------------------------------------------------------------------ bootstrap AUROC
def group_bootstrap_counts(groups, n_boot, rng):
    """(n_boot, n) row multiplicities of a cluster bootstrap that resamples whole groups with replacement."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    picks = rng.integers(len(u), size=(n_boot, len(u)))
    gc = np.zeros((n_boot, len(u)), np.int32)
    for b in range(n_boot):
        np.add.at(gc[b], picks[b], 1)
    return gc[:, inv]


def weighted_auroc(y, s, W=None):
    """AUROC (ties count 1/2) of scores s for labels y under row weights W ((B, n) or (n,)); returns (B,) or float."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, np.float64)
    single = W is None or np.ndim(W) == 1
    W = np.ones((1, len(s))) if W is None else np.atleast_2d(np.asarray(W, np.float64))
    o = np.argsort(s, kind='mergesort')
    s_o, y_o, W_o = s[o], y[o], W[:, o]
    starts = np.flatnonzero(np.r_[True, s_o[1:] != s_o[:-1]])
    pos = np.add.reduceat(W_o * y_o, starts, axis=1)
    neg = np.add.reduceat(W_o * ~y_o, starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_below + 0.5 * neg)).sum(1)
    out = num / (pos.sum(1) * neg.sum(1))
    return float(out[0]) if single else out


# ------------------------------------------------------------------------------------------------ hierarchical se
def _p(m, se, df):
    """Two-sided t p-value; se = 0 (identical sources) gives p = 1 if m = 0, else 0."""
    if se > 0:
        return float(2 * stats.t.sf(abs(m) / se, df))
    return 1.0 if m == 0 else 0.0


def hierarchical(per_seed, boot_seed_mean, alpha=0.05):
    """Seed + statement uncertainty of a mean over seeds.

    per_seed: (S,) per-seed estimates (each already pooled over folds).
    boot_seed_mean: (B,) the seed-averaged estimate under each shared statement resample.
    se = sqrt(sd_seed^2 / S + var_boot); Satterthwaite df = se^4 / (se_seed^4 / (S - 1)) (statement part known)."""
    d = np.asarray(per_seed, np.float64)
    S = len(d)
    m = float(d.mean())
    se_seed = float(d.std(ddof=1) / np.sqrt(S))
    v_stmt = float(np.var(np.asarray(boot_seed_mean, np.float64), ddof=1)) if boot_seed_mean is not None else 0.0
    se = float(np.sqrt(se_seed ** 2 + v_stmt))
    df = se ** 4 / (se_seed ** 4 / (S - 1)) if se_seed > 0 else np.inf
    tq = stats.t.ppf(1 - alpha / 2, df)
    t90 = stats.t.ppf(0.95, df)
    p = _p(m, se, df)
    return dict(estimate=m, per_seed=d.round(5).tolist(), se_seed=se_seed, se_stmt=float(np.sqrt(v_stmt)), se=se,
                stmt_share=v_stmt / se ** 2 if se > 0 else np.nan, df=float(df), ci_low=m - tq * se, ci_high=m + tq * se,
                ci90_low=m - t90 * se, ci90_high=m + t90 * se, upper95_one_sided=m + t90 * se,
                lower95_one_sided=m - t90 * se, p=p, p_pilot_rule_t=_p(m, se, S - 1),
                ci_low_pilot_rule=m - stats.t.ppf(1 - alpha / 2, S - 1) * se,
                ci_high_pilot_rule=m + stats.t.ppf(1 - alpha / 2, S - 1) * se,
                seeds_negative=int((d < 0).sum()), n_seeds=S)


def two_way_bootstrap(per_seed_boot, n_draw, rng):
    """per_seed_boot: (S, B) per-seed estimates under each statement resample. Each draw resamples seeds with
    replacement and picks one statement resample uniformly; returns the 2.5 / 97.5 percentiles of the mean."""
    S, B = per_seed_boot.shape
    s_idx = rng.integers(S, size=(n_draw, S))
    b_idx = rng.integers(B, size=n_draw)
    vals = per_seed_boot[s_idx, b_idx[:, None]].mean(1)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ------------------------------------------------------------------------------------------------ multiplicity
def holm(ps):
    ps = np.asarray(ps, np.float64)
    o = np.argsort(ps)
    out, run = np.empty_like(ps), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def gatekeeping(h1, h2, h3, alpha=0.05):
    """Fixed sequence: H1 gate (two-sided p, rejection only if estimate < 0); then Holm over H2, H3 (same rule).
    Each h is a dict with 'estimate' and 'p'. Returns adjusted p-values and rejections."""
    rej1 = h1['p'] < alpha and h1['estimate'] < 0
    hp = holm([h2['p'], h3['p']])
    adj2, adj3 = max(h1['p'], hp[0]), max(h1['p'], hp[1])
    rej2 = rej1 and adj2 < alpha and h2['estimate'] < 0
    rej3 = rej1 and adj3 < alpha and h3['estimate'] < 0
    return dict(H1=dict(p_adj=h1['p'], rejected=bool(rej1)), H2=dict(p_adj=float(adj2), rejected=bool(rej2)),
                H3=dict(p_adj=float(adj3), rejected=bool(rej3)))


# ------------------------------------------------------------------------------------------------ labels
def h1_label(h, sesoi=SESOI):
    if h['ci_high'] < 0:
        return 'SUPPORTED (meaningful)' if h['estimate'] <= -sesoi else 'SUPPORTED, below SESOI'
    if h['ci_low'] > 0:
        return 'HARM'
    if h['ci90_low'] > -sesoi and h['ci90_high'] < sesoi:
        return 'NO MEANINGFUL EFFECT'
    return 'INCONCLUSIVE'


def specificity_label(g):
    if not g['H1']['rejected']:
        return 'not tested (H1 not rejected)'
    if g['H2']['rejected'] and g['H3']['rejected']:
        return 'truth-specific'
    if g['H3']['rejected']:
        return 'geometry-specific only'
    if g['H2']['rejected']:
        return 'label-specific vs shuf only (rand not rejected)'
    return 'not truth-specific'


def non_inferiority(h, margin, higher_is_worse=True):
    """Pass if the one-sided 95 % bound stays within the margin (upper <= margin, or lower >= -margin)."""
    if higher_is_worse:
        return dict(bound=h['upper95_one_sided'], margin=margin, passed=bool(h['upper95_one_sided'] <= margin))
    return dict(bound=h['lower95_one_sided'], margin=-margin, passed=bool(h['lower95_one_sided'] >= -margin))
