"""Stage 6c: steering GPT-2 with SAE features (truth feature vs random features).

Main readout (same as 6a/6b): per test pair, gap = log P(true sentence) - log P(false sentence).
Truth feature: chosen on the TRAIN split only, per dataset, SAE type and layer
  (largest standardized true-minus-false difference among features firing on >= 1% of statements).
Control: random features from the same SAE with similar firing frequency (within a factor of 2).
Directional effect per pair = (gap at +alpha - gap at -alpha) / 2.
Extras: test AUROC of the chosen feature, a vocabulary check (does the feature's direction
directly push the word ' true' over ' false'?), the 'This statement is' judgment, BH-adjusted
p-values, bootstrap CIs, and a fixed-prompt text grid.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from src.got import steering

import logging

logger = logging.getLogger(__name__)

RELEASES = {"relu": "gpt2-small-res-jb", "topk": "gpt2-small-resid-post-v5-32k"}
JUDGE_SUFFIX = " This statement is"


# ---------- SAEs ----------
def sae_id_for(sae_type, layer):
    """SAE reading blocks.<layer>.hook_resid_pre (TopK: the previous block's resid_post, same point)."""
    return f"blocks.{layer}.hook_resid_pre" if sae_type == "relu" else f"blocks.{layer - 1}.hook_resid_post"


def load_sae(sae_type, layer):
    from sae_lens import SAE
    out = SAE.from_pretrained(RELEASES[sae_type], sae_id_for(sae_type, layer), device="cpu")
    sae = out[0] if isinstance(out, tuple) else out
    sae.eval()
    return sae


def encode(sae, x, batch=256):
    import torch
    zs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(dtype=sae.W_enc.dtype)
            zs.append(sp.csr_matrix(sae.encode(xb).float().numpy()))
    return sp.vstack(zs).tocsr()


# ---------- feature choice (train split only) ----------
def choose_features(z_train, y_train, n_random, rng, min_freq=0.01):
    freq = np.asarray((z_train > 0).astype(np.float32).mean(0)).ravel()
    m1 = np.asarray(z_train[np.where(y_train == 1)[0]].mean(0)).ravel()
    m0 = np.asarray(z_train[np.where(y_train == 0)[0]].mean(0)).ravel()
    mean = np.asarray(z_train.mean(0)).ravel()
    ex2 = np.asarray(z_train.multiply(z_train).mean(0)).ravel()
    sd = np.sqrt(np.maximum(ex2 - mean ** 2, 0))
    d = (m1 - m0) / (sd + 1e-6)
    d[freq < min_freq] = 0
    j = int(np.argmax(np.abs(d)))
    sign = 1.0 if d[j] > 0 else -1.0
    idx = np.arange(len(freq))
    cand = idx[(freq >= freq[j] / 2) & (freq <= freq[j] * 2) & (idx != j)]
    if len(cand) < n_random:  # fall back to the features closest in frequency
        order = np.argsort(np.abs(np.log(freq + 1e-9) - np.log(freq[j])))
        cand = order[order != j][: n_random * 3]
    randoms = rng.choice(cand, size=n_random, replace=False)
    return j, sign, float(d[j]), float(freq[j]), [int(r) for r in randoms]


# ---------- readouts ----------
def pair_gaps(model, pairs, layer=None, vector=None):
    """Per-pair true-minus-false log-prob gap and mean per-token log-prob (fluency)."""
    hooks = steering._hooks(layer, vector)
    gaps, flu = [], []
    for true_s, false_s in pairs:
        t_tot, t_tok = steering._sentence_logprob(model, model.to_tokens(true_s, prepend_bos=True), hooks)
        f_tot, f_tok = steering._sentence_logprob(model, model.to_tokens(false_s, prepend_bos=True), hooks)
        gaps.append(t_tot - f_tot)
        flu.append((t_tok + f_tok) / 2)
    return np.array(gaps), np.array(flu)


def judgment_scores(model, statements, layer=None, vector=None):
    """logit(' true') - logit(' false') after '<statement> This statement is'."""
    import torch
    t_id, f_id = model.to_single_token(" true"), model.to_single_token(" false")
    hooks = steering._hooks(layer, vector)
    out = []
    with torch.no_grad():
        for s in statements:
            toks = model.to_tokens(s + JUDGE_SUFFIX, prepend_bos=True)
            logits = model.run_with_hooks(toks, fwd_hooks=hooks) if hooks else model(toks)
            out.append(float(logits[0, -1, t_id] - logits[0, -1, f_id]))
    return np.array(out)


def judgment_baseline(model, data):
    rows = []
    for name, d in data.items():
        statements = [s for pair in d["pairs"] for s in pair]
        labels = np.array([1, 0] * len(d["pairs"]))
        s = judgment_scores(model, statements)
        rows.append(dict(dataset=name, auroc=roc_auc_score(labels, s),
                         says_true_pct=100 * float((s > 0).mean()),
                         correct_pct=100 * float(((s > 0).astype(int) == labels).mean())))
    return pd.DataFrame(rows)


def vocab_effect(model, direction):
    """Direct push of a unit direction on ' true' vs ' false' through the unembedding (approximate)."""
    import torch
    u = torch.tensor(direction / (np.linalg.norm(direction) + 1e-12), dtype=model.W_U.dtype)
    with torch.no_grad():
        logits = u @ model.W_U
    return float(logits[model.to_single_token(" true")] - logits[model.to_single_token(" false")])


def generate(model, prompt, layer=None, vector=None, n_tokens=20):
    import torch
    hooks = steering._hooks(layer, vector)
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        # no KV cache, so the steering vector is added at every generated token too
        out = model.generate(prompt, max_new_tokens=n_tokens, do_sample=False,
                             use_past_kv_cache=False, verbose=False)
    return out[len(prompt):].replace("\n", " ")


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


# ---------- the experiment ----------
def run(model, data, sae_types, layers, alphas, n_random, text_layers, text_alphas,
        n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    base = {name: pair_gaps(model, d["pairs"]) for name, d in data.items()}
    choice_rows, grid_rows, summary_rows, text_rows = [], [], [], []
    a_lo, a_hi = min(alphas), max(alphas)

    for t in sae_types:
        for layer in layers:
            sae = load_sae(t, layer)
            W_dec = sae.W_dec.detach().float().numpy()
            for name, d in data.items():
                x = d["last"][:, layer, :]
                tr, te, y = d["tr"], d["te"], d["y"]
                z_tr, z_te = encode(sae, x[tr]), encode(sae, x[te])
                j, sign, eff, freq, randoms = choose_features(z_tr, y[tr], n_random, rng)
                auc = roc_auc_score(y[te], sign * z_te[:, j].toarray().ravel())
                feats = [("truth", j, sign)] + [(f"random_{i + 1}", r, 1.0) for i, r in enumerate(randoms)]
                vocab = {src: vocab_effect(model, sgn * W_dec[fid]) for src, fid, sgn in feats}
                vr = np.array([abs(v) for k, v in vocab.items() if k != "truth"])
                choice_rows.append(dict(
                    sae_type=t, layer=layer, dataset=name, feature=j, sign=sign,
                    train_effect_size=eff, firing_freq=freq, test_auroc=auc,
                    vocab_true_minus_false=vocab["truth"], vocab_random_abs_mean=vr.mean(),
                    vocab_rank_among_randoms=int(1 + (vr > abs(vocab["truth"])).sum())))

                g0, f0 = base[name]
                directional = {}
                for src, fid, sgn in feats:
                    gaps = {}
                    for a in alphas:
                        vec = steering.make_vector(sgn * W_dec[fid], d["norms"][layer], a)
                        g, f = pair_gaps(model, d["pairs"], layer, vec)
                        gaps[a] = g
                        grid_rows.append(dict(sae_type=t, layer=layer, dataset=name, source=src,
                                              feature=fid, alpha=a, effect=float(g.mean() - g0.mean()),
                                              fluency_change=float(f.mean() - f0.mean()),
                                              pair_accuracy=float((g > 0).mean())))
                    directional[src] = (gaps[a_hi] - gaps[a_lo]) / 2

                # secondary readout: the 'This statement is' judgment, truth feature only
                statements = [s for pair in d["pairs"] for s in pair]
                jud = {a: judgment_scores(model, statements, layer,
                                          steering.make_vector(sign * W_dec[j], d["norms"][layer], a))
                       for a in (a_lo, a_hi)}

                own = directional["truth"]
                rnd = np.array([abs(v.mean()) for k, v in directional.items() if k != "truth"])
                boot = rng.integers(0, len(own), size=(n_boot, len(own)))
                ci = np.percentile(own[boot].mean(1), [2.5, 97.5])
                summary_rows.append(dict(
                    sae_type=t, layer=layer, dataset=name, feature=j, test_auroc=auc,
                    own_directional=float(own.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
                    random_abs_mean=float(rnd.mean()), random_abs_max=float(rnd.max()),
                    own_rank=int(1 + (rnd > abs(own.mean())).sum()), n_features=len(rnd) + 1,
                    p_value=(1 + (rnd >= abs(own.mean())).sum()) / (len(rnd) + 1),
                    judgment_shift=float((jud[a_hi].mean() - jud[a_lo].mean()) / 2),
                    vocab_true_minus_false=vocab["truth"]))
                logger.info(f"    {t} L{layer} {name}: feature {j} (test AUROC {auc:.2f})  "
                      f"directional {own.mean():+.3f}  rank {summary_rows[-1]['own_rank']}/{len(rnd) + 1}")

                if layer in text_layers:
                    for prompt in d["text_prompts"]:
                        for a in text_alphas:
                            vec = None if a == 0 else steering.make_vector(sign * W_dec[j], d["norms"][layer], a)
                            text_rows.append(dict(sae_type=t, layer=layer, dataset=name, feature=j,
                                                  prompt=prompt, alpha=a,
                                                  continuation=generate(model, prompt, layer, vec)))
            del sae

    summary = pd.DataFrame(summary_rows)
    summary["p_adjusted_bh"] = bh_adjust(summary["p_value"].to_numpy())
    return (pd.DataFrame(choice_rows), pd.DataFrame(grid_rows), summary, pd.DataFrame(text_rows))