"""Stage 6 (RepE): steering with the honesty direction and with SAE honesty features.

Readout (same method as GoT 6a-6c), on minimal true/false pairs from the RepE facts
(a false statement that is a <= 3-word edit of a true one):
    gap = log P(true sentence) - log P(false sentence)
    effect = mean gap with steering - mean gap without (> 0 = pushed towards the true version)
    directional (per pair) = (gap at +alpha - gap at -alpha) / 2, with + = towards 'honest'
Every result is also reported on the TEST-split subset (pairs whose true statement is in test).

Parts:
  a  scan: honesty directions at layers 1-11, alpha +-0.5
       mean_diff_matched : fitted here on the 4 length-matched sets' TRAIN pairs (no token confound)
       mean_diff_original, repe_pca : Stage 2 directions (original set)
  b  control: mean_diff_matched vs 20 random directions at layers 6, 10, 11
  c  SAE features: each SAE's top honesty feature vs 20 frequency-matched random features

Run from the repo root:
    python scripts/run_repe_s6.py --part a --quick
    python scripts/run_repe_s6.py --part a       (then b, then c)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_repe_s4 as s4  # noqa: E402  (data + SAE loading)
from src.repe.activations import load_model  # noqa: E402

PAIRS_PATH = ROOT / "data" / "repe" / "minimal_pairs.csv"
DIR_PATH = ROOT / "data" / "repe" / "directions" / "original_last.npz"
TAB_DIR = ROOT / "results" / "tables" / "repe"
MATCHED = [s for s in s4.SETS if s.startswith("matched_")]
SCAN_LAYERS = list(range(1, 12))
CTRL_LAYERS = [6, 10, 11]
ALPHAS = [-0.5, 0.5]
N_RANDOM = 20


# ---------- readout ----------
def load_pairs(n_max=None):
    p = pd.read_csv(PAIRS_PATH)
    p = p[(p["words_changed"] <= 3) & (p["similarity"] >= 0.7)].reset_index(drop=True)
    if n_max:
        p = p.head(n_max)
    pairs = list(zip(p["true_statement"], p["false_statement"]))
    test_mask = (p["true_split"].astype(str) == "test").to_numpy()
    return pairs, test_mask


def _hooks(layer, vec):
    if vec is None:
        return []

    def add(resid, hook):
        resid[:, 1:, :] += vec  # every token except BOS
        return resid

    return [(f"blocks.{layer}.hook_resid_pre", add)]


def sentence_logprob(model, text, hooks):
    toks = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=hooks) if hooks else model(toks)
        lp = torch.log_softmax(logits[0], dim=-1)
        tok_lp = lp[:-1].gather(1, toks[0, 1:, None]).squeeze(1)
    return float(tok_lp.sum()), float(tok_lp.mean())


def pair_scores(model, pairs, layer=None, vec=None):
    hooks = _hooks(layer, vec)
    gaps, flu = [], []
    for t, f in pairs:
        tt, tm = sentence_logprob(model, t, hooks)
        ft, fm = sentence_logprob(model, f, hooks)
        gaps.append(tt - ft)
        flu.append((tm + fm) / 2)
    return np.array(gaps), np.array(flu)


def make_vec(direction, norm, alpha):
    u = direction / (np.linalg.norm(direction) + 1e-12)
    return torch.tensor(alpha * norm * u, dtype=torch.float32)


# ---------- directions ----------
def honesty_directions():
    z = np.load(DIR_PATH)
    dirs = {"mean_diff_original": z["mean_diff"], "repe_pca": z["repe_pca"]}
    diffs = []
    for n in MATCHED:
        s = s4.load_set(n)
        tr = s["meta"]["split"].astype(str).to_numpy()[s["h"]] == "train"
        diffs.append(s["x"][s["h"][tr]] - s["x"][s["u"][tr]])
    dirs["mean_diff_matched"] = np.concatenate(diffs).mean(0)  # [13, 768], honest minus dishonest
    ref = dirs["mean_diff_matched"]
    for k in ("mean_diff_original", "repe_pca"):  # make every direction point towards 'honest'
        for L in range(ref.shape[0]):
            if dirs[k][L] @ ref[L] < 0:
                dirs[k][L] = -dirs[k][L]
    return dirs


def layer_norms():
    x = s4.load_set("original")["x"]
    return np.linalg.norm(x, axis=2).mean(axis=0)


# ---------- summaries ----------
def bh_adjust(p):
    p = np.asarray(p, float)
    n, order = len(p), np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def rank_summary(directional, own_key, rng, test_mask, extra=None, n_boot=1000):
    own = directional[own_key]
    rnd = np.array([abs(v.mean()) for k, v in directional.items() if k != own_key])
    boot = rng.integers(0, len(own), size=(n_boot, len(own)))
    ci = np.percentile(own[boot].mean(1), [2.5, 97.5])
    out = dict(own_directional=float(own.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
               own_directional_test=float(own[test_mask].mean()),
               random_abs_mean=float(rnd.mean()), random_abs_max=float(rnd.max()),
               own_rank=int(1 + (rnd > abs(own.mean())).sum()), n_directions=len(rnd) + 1,
               p_value=(1 + (rnd >= abs(own.mean())).sum()) / (len(rnd) + 1))
    out.update(extra or {})
    return out


def run_sources(model, pairs, layer, sources, norms, alphas, base):
    """sources: {name: direction[768]} -> grid rows and per-pair directional effect."""
    rows, directional = [], {}
    g0, f0 = base
    for name, direction in sources.items():
        gaps = {}
        for a in alphas:
            g, f = pair_scores(model, pairs, layer, make_vec(direction, norms[layer], a))
            gaps[a] = g
            rows.append(dict(source=name, layer=layer, alpha=a, effect=float(g.mean() - g0.mean()),
                             fluency_change=float(f.mean() - f0.mean()),
                             pair_accuracy=float((g > 0).mean())))
        directional[name] = (gaps[max(alphas)] - gaps[min(alphas)]) / 2
    return rows, directional


# ---------- parts ----------
def part_a(model, pairs, test_mask, norms, quick):
    dirs = honesty_directions()
    names = ["mean_diff_matched"] if quick else list(dirs)
    layers = [6] if quick else SCAN_LAYERS
    for L in [6, 10, 11]:
        c = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  L{L} cosine with mean_diff_matched: original {c(dirs['mean_diff_original'][L], dirs['mean_diff_matched'][L]):.2f}"
              f"  repe_pca {c(dirs['repe_pca'][L], dirs['mean_diff_matched'][L]):.2f}")
    base = pair_scores(model, pairs)
    rows, summary = [], []
    for L in layers:
        r, d = run_sources(model, pairs, L, {n: dirs[n][L] for n in names}, norms, ALPHAS, base)
        rows += r
        for n in names:
            summary.append(dict(source=n, layer=L, directional=float(d[n].mean()),
                                directional_test=float(d[n][test_mask].mean())))
        print(f"    L{L} done")
    s = pd.DataFrame(summary)
    print("\nDirectional effect (+ = pushing towards 'honest' raises preference for the TRUE version):")
    print(s.pivot_table(index="layer", columns="source", values="directional").round(3).to_string())
    return {"repe_s6a_scan_grid": pd.DataFrame(rows), "repe_s6a_scan_summary": s}


def part_b(model, pairs, test_mask, norms, quick):
    rng = np.random.default_rng(0)
    own = honesty_directions()["mean_diff_matched"]
    n_rand = 3 if quick else N_RANDOM
    randoms = [rng.normal(size=768) for _ in range(n_rand)]
    base = pair_scores(model, pairs)
    rows, summary = [], []
    for L in ([6] if quick else CTRL_LAYERS):
        sources = {"own": own[L], **{f"random_{i + 1}": r for i, r in enumerate(randoms)}}
        r, d = run_sources(model, pairs, L, sources, norms, ALPHAS, base)
        rows += r
        summary.append(dict(layer=L, **rank_summary(d, "own", rng, test_mask)))
        print(f"    L{L}: directional {summary[-1]['own_directional']:+.3f}  rank "
              f"{summary[-1]['own_rank']}/{summary[-1]['n_directions']}")
    s = pd.DataFrame(summary)
    s["p_adjusted_bh"] = bh_adjust(s["p_value"])
    print("\nHonesty direction vs random directions (rank 1 = strongest):")
    print(s.round(3).to_string(index=False))
    return {"repe_s6b_random20_grid": pd.DataFrame(rows), "repe_s6b_random20_summary": s}


@torch.no_grad()
def encode(sae, x, batch=256):
    zs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dtype=sae.W_enc.dtype)
        zs.append(sp.csr_matrix(sae.encode(xb).float().numpy()))
    return sp.vstack(zs).tocsr()


def choose_feature(zh, zu, n_rand, rng, min_freq=0.01):
    both = sp.vstack([zh, zu]).tocsr()
    freq = np.asarray((both > 0).astype(np.float32).mean(0)).ravel()
    diff = (zh - zu).toarray()
    score = diff.mean(0) / (diff.std(0) + 1e-6)
    score[freq < min_freq] = 0
    j = int(np.argmax(np.abs(score)))
    sign = 1.0 if score[j] > 0 else -1.0
    idx = np.arange(len(freq))
    cand = idx[(freq >= freq[j] / 2) & (freq <= freq[j] * 2) & (idx != j)]
    if len(cand) < n_rand:
        order = np.argsort(np.abs(np.log(freq + 1e-9) - np.log(freq[j])))
        cand = order[order != j][: n_rand * 3]
    return j, sign, float(score[j]), float(freq[j]), [int(r) for r in rng.choice(cand, n_rand, replace=False)]


def part_c(model, pairs, test_mask, norms, quick):
    rng = np.random.default_rng(0)
    n_rand = 3 if quick else N_RANDOM
    sets = {n: s4.load_set(n) for n in MATCHED}
    base = pair_scores(model, pairs)
    rows, summary = [], []
    for t in (["relu"] if quick else ["relu", "topk"]):
        for L in ([6] if quick else CTRL_LAYERS):
            sid = f"blocks.{L}.hook_resid_pre" if t == "relu" else f"blocks.{L - 1}.hook_resid_post"
            sae = s4.load_sae(s4.RELEASES[t], sid)
            W = sae.W_dec.detach().float().numpy()
            xh, xu = [], []
            for s in sets.values():
                tr = s["meta"]["split"].astype(str).to_numpy()[s["h"]] == "train"
                xh.append(s["x"][s["h"][tr], L, :])
                xu.append(s["x"][s["u"][tr], L, :])
            zh, zu = encode(sae, np.concatenate(xh)), encode(sae, np.concatenate(xu))
            j, sign, eff, freq, randoms = choose_feature(zh, zu, n_rand, rng)
            sources = {"own": sign * W[j], **{f"random_{i + 1}": W[r] for i, r in enumerate(randoms)}}
            r, d = run_sources(model, pairs, L, sources, norms, ALPHAS, base)
            rows += [dict(sae_type=t, **x) for x in r]
            summary.append(dict(sae_type=t, layer=L, feature=j, sign=sign, train_effect_size=eff,
                                firing_freq=freq, **rank_summary(d, "own", rng, test_mask)))
            print(f"    {t} L{L}: feature {j} (freq {freq:.2f})  directional "
                  f"{summary[-1]['own_directional']:+.3f}  rank {summary[-1]['own_rank']}/{summary[-1]['n_directions']}")
            del sae
    s = pd.DataFrame(summary)
    s["p_adjusted_bh"] = bh_adjust(s["p_value"])
    print("\nSAE honesty feature vs frequency-matched random features (rank 1 = strongest):")
    print(s.round(3).to_string(index=False))
    return {"repe_s6c_grid": pd.DataFrame(rows), "repe_s6c_summary": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c"], required=True)
    ap.add_argument("--quick", action="store_true", help="layer 6 only, few directions, 30 pairs")
    args = ap.parse_args()

    pairs, test_mask = load_pairs(30 if args.quick else None)
    print(f"{len(pairs)} minimal pairs ({test_mask.sum()} with the true statement in the test split)")
    model = load_model()
    norms = layer_norms()
    g0, _ = pair_scores(model, pairs)
    print(f"Baseline (no steering): GPT-2 prefers the TRUE version in {100 * (g0 > 0).mean():.1f}% of pairs "
          f"(test subset {100 * (g0[test_mask] > 0).mean():.1f}%), mean gap {g0.mean():+.3f}\n")

    t0 = time.time()
    tables = {"a": part_a, "b": part_b, "c": part_c}[args.part](model, pairs, test_mask, norms, args.quick)
    print(f"\nPart {args.part} took {time.time() - t0:.0f}s")
    if args.quick:
        print("Quick test only, nothing saved.")
        return
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(TAB_DIR / f"{name}.csv", index=False)
        print(f"Saved results/tables/repe/{name}.csv")


if __name__ == "__main__":
    main()