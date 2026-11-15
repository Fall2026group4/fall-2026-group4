"""M3/M4 activation-patching helpers: position groups for the RepE-LM prompts, KL outcomes, restoration, statistics.

Restoration (absolute nats, larger = the patched site carries more of the effect):
    R_A (H -> U)  = KL(p_H || p_U) - KL(p_H || p_{U <- H at site})
    R_B (U -> H)  = KL(p_U || p_H) - KL(p_U || p_{H <- U at site})
    R_S (synonym) = KL(p_H || p_Hp) - KL(p_H || p_{Hp <- H at site})
"""
import numpy as np
import torch

POSITION_GROUPS = ('slot', 'post_slot_instruction', 'assistant_span', 'prefix_excl_final', 'final')


def template_positions(tok, instruction_filled, slot):
    """Token indices (BOS = 0) of the instruction end and the ' ASSISTANT:' span for one filled template."""
    bos = [tok.bos_token_id]
    a = bos + tok.encode('USER: ' + instruction_filled)
    b = bos + tok.encode('USER: ' + instruction_filled + ' ASSISTANT:')
    assert b[:len(a)] == a, 'tokenisation of " ASSISTANT:" is not a clean suffix'
    return dict(slot=slot, asst_start=len(a), asst_end=len(b) - 1)


def group_positions(group, slot, asst_start, asst_end, final_pos):
    """Positions patched for one position group (may be empty)."""
    if group == 'slot':
        return [slot]
    if group == 'post_slot_instruction':
        return list(range(slot + 1, asst_start))
    if group == 'assistant_span':
        return list(range(asst_start, asst_end + 1))
    if group == 'prefix_excl_final':
        return list(range(asst_end + 1, final_pos))
    if group == 'final':
        return [final_pos]
    raise ValueError(group)


def kl(logp, logq):
    """KL(p || q) over the last axis, from log-probabilities (float64)."""
    logp, logq = logp.double(), logq.double()
    return (logp.exp() * (logp - logq)).sum(-1)


def stat_summary(values, groups, n_boot, rng):
    """Mean over statements (pairs first averaged within statement), bootstrap over statements.
    Returns mean, se, CI and one-sided normal-approximation p for mean > 0."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ok = np.isfinite(values)
    if not ok.any():
        return dict(mean=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=1.0, n_statements=0)
    u, inv = np.unique(groups[ok], return_inverse=True)
    per = np.bincount(inv, values[ok]) / np.bincount(inv)
    boots = per[rng.integers(len(per), size=(n_boot, len(per)))].mean(1)
    m, se = float(per.mean()), float(boots.std(ddof=1))
    from scipy.stats import norm
    p = float(norm.sf(m / se)) if se > 0 else (0.0 if m > 0 else 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=m, se=se, ci_low=float(lo), ci_high=float(hi), p=p, n_statements=int(len(per)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def random_like(delta, gen):
    """Random directions with the same per-position norm as delta (zero where delta is zero)."""
    g = torch.randn(delta.shape, generator=gen, device=delta.device, dtype=delta.dtype)
    g = g / g.norm(dim=-1, keepdim=True)
    return g * delta.norm(dim=-1, keepdim=True)
