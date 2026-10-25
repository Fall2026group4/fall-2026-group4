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
  c  SAE features: each SAE's top honesty feature vs frequency-matched random features
  d  positive control + second behavioural outcome (Review #6): the model's own log P(' True') - log P(' False')
       after the prompt '<statement> True or False? Answer:'. Positive control = the unembedding contrast direction
       W_U[' True'] - W_U[' False'], which MUST move this outcome; honesty direction and random directions are compared with it.

Run from the repo root:
    python scripts/run_repe_s6.py --part a --quick
    python scripts/run_repe_s6.py --part a       (then b, c, d)
    python scripts/run_repe_s6.py --part b --n-random 20     (fast rerun with the old 20 controls)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

logger = logging.getLogger("run_repe_s6")
logging.basicConfig(level=logging.INFO, format="%(message)s")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_repe_s4 as s4  # noqa: E402  (data + SAE loading)
from src.repe.activations import load_model  # noqa: E402
from src.repe.stats import bh_adjust, rank_summary  # noqa: E402

PAIRS_PATH = ROOT / "data" / "repe" / "minimal_pairs.csv"
DIR_PATH = ROOT / "data" / "repe" / "directions" / "original_last.npz"
TAB_DIR = ROOT / "results" / "tables" / "repe"
MATCHED = [s for s in s4.SETS if s.startswith("matched_")]
SCAN_LAYERS = list(range(1, 12))
CTRL_LAYERS = [6, 10, 11]
ALPHAS = [-0.5, 0.5]
N_RANDOM = 200  # Review #6: >= 200 random controls so the smallest attainable p is below 0.005
TF_PROMPT = "{statement}\nTrue or False? Answer:"


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


def tf_gaps(model, prompts, layer=None, vec=None):
    """Behavioural outcome: log P(' True') - log P(' False') as the next token after each prompt."""
    hooks = _hooks(layer, vec)
    tid, fid = model.to_single_token(" True"), model.to_single_token(" False")
    out = []
    for p in prompts:
        toks = model.to_tokens(p, prepend_bos=True)
        with torch.no_grad():
            logits = model.run_with_hooks(toks, fwd_hooks=hooks) if hooks else model(toks)
            lp = torch.log_softmax(logits[0, -1], dim=-1)
        out.append(float(lp[tid] - lp[fid]))
    return np.array(out)


def part_d(model, pairs, test_mask, norms, quick):
    """Positive control and a model-output outcome (Instructor Review #6).

    Outcome: log P(' True') - log P(' False') after '<statement> True or False? Answer:'.
    Positive control: the unembedding contrast W_U[' True'] - W_U[' False'] pushes that outcome by
    construction, so it MUST rank first against the random directions; if it does not, the test
    cannot detect effects and a null for the honesty direction would mean nothing.
    """
    rng = np.random.default_rng(0)
    n_rand = 3 if quick else N_RANDOM
    prompts = [TF_PROMPT.format(statement=t) for t, _ in pairs] + [TF_PROMPT.format(statement=f) for _, f in pairs]
    mask = np.concatenate([test_mask, test_mask])
    tid, fid = model.to_single_token(" True"), model.to_single_token(" False")
    positive = (model.W_U[:, tid] - model.W_U[:, fid]).detach().float().cpu().numpy()
    own = honesty_directions()["mean_diff_matched"]
    randoms = [rng.normal(size=768) for _ in range(n_rand)]
    base = tf_gaps(model, prompts)
    n = len(pairs)
    logger.info(f"  baseline log P(True) - log P(False): true statements {base[:n].mean():+.3f}, "
                f"false statements {base[n:].mean():+.3f}")
    rows, summary = [], []
    for L in ([6] if quick else CTRL_LAYERS):
        sources = {"honesty_direction": own[L], "positive_control": positive,
                   **{f"random_{i + 1}": r for i, r in enumerate(randoms)}}
        directional = {}
        for name, direction in sources.items():
            g = {a: tf_gaps(model, prompts, L, make_vec(direction, norms[L], a)) for a in ALPHAS}
            for a in ALPHAS:
                rows.append(dict(source=name, layer=L, alpha=a, effect=float(g[a].mean() - base.mean())))
            directional[name] = (g[max(ALPHAS)] - g[min(ALPHAS)]) / 2
        for own_key in ("honesty_direction", "positive_control"):
            subset = {k: v for k, v in directional.items() if k == own_key or k.startswith("random_")}
            summary.append(dict(layer=L, comparison=own_key, **rank_summary(subset, own_key, rng, mask)))
            r = summary[-1]
            logger.info(f"    L{L} {own_key}: directional {r['own_directional']:+.3f}  rank {r['own_rank']}/{r['n_directions']}"
                        f"  p {r['p_value']:.4f}")
    s = pd.DataFrame(summary)
    s["p_adjusted_bh"] = s.groupby("comparison")["p_value"].transform(bh_adjust)
    pc = s[s["comparison"] == "positive_control"]
    logger.info("\nPositive control: " + ("PASSED (ranks first at every layer)" if (pc["own_rank"] == 1).all()
                                        else "FAILED, the test cannot be trusted until this is fixed"))
    logger.info(s.round(4).to_string(index=False))
    return {"repe_s6d_tf_grid": pd.DataFrame(rows), "repe_s6d_tf_summary": s}


def main():
    global N_RANDOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c", "d"], required=True)
    ap.add_argument("--n-random", type=int, default=None,
                    help=f"random controls for parts b, c, d (default {N_RANDOM}; 20 reproduces the earlier run)")
    ap.add_argument("--quick", action="store_true", help="layer 6 only, few directions, 30 pairs")
    args = ap.parse_args()
    if args.n_random:
        N_RANDOM = args.n_random

    pairs, test_mask = load_pairs(30 if args.quick else None)
    print(f"{len(pairs)} minimal pairs ({test_mask.sum()} with the true statement in the test split)")
    model = load_model()
    norms = layer_norms()
    g0, _ = pair_scores(model, pairs)
    print(f"Baseline (no steering): GPT-2 prefers the TRUE version in {100 * (g0 > 0).mean():.1f}% of pairs "
          f"(test subset {100 * (g0[test_mask] > 0).mean():.1f}%), mean gap {g0.mean():+.3f}\n")

    t0 = time.time()
    tables = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[args.part](model, pairs, test_mask, norms, args.quick)
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