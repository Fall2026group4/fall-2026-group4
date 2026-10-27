"""
SAE-Faithful: Gemma-2-2B baseline.

Step 1. Layer table: for every layer, how well does the truth direction separate
        true from false statements (AUROC)? Best layer is picked on validation.
Step 2. Next-word test: with a few example sentences in front, does Gemma name
        the right country? Does pushing the truth direction change that?
Step 3. True/false judgment test: "<statement> This statement is: TRUE/FALSE".
        Does pushing the truth direction flip Gemma's judgment?

Steering is tried at several strengths (--alphas), in both directions, and is
applied ONLY to the test sentence, never to the example sentences in front of it.

Data: data/got/<dataset>.csv with columns statement, label (1 true, 0 false),
      and optionally split (train / val / test).

Outputs (in --out/<dataset>/):
  layer_table.csv / .md     all layers, best layer marked
  nextword_summary.csv      next-word accuracy per steering strength
  judgment_summary.csv      % judged TRUE and accuracy per steering strength
  nextword_examples.csv     example completions
  judgment_examples.csv     example judgments

Run:
  python scripts/run_gemma_baseline.py --dataset cities --data_dir data/got --out results/tables/gemma
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

NEXTWORD_SHOTS = ("The city of Paris is in France. "
                  "The city of Tokyo is in Japan. "
                  "The city of Toronto is in Canada.")

JUDGE_SHOTS = ("The city of Paris is in France. This statement is: TRUE\n"
               "The city of Paris is in Japan. This statement is: FALSE\n"
               "The city of Tokyo is in Japan. This statement is: TRUE\n"
               "The city of Tokyo is in Brazil. This statement is: FALSE\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--data_dir", default="data/got")
    p.add_argument("--dataset", default="cities")
    p.add_argument("--steer_layer", type=int, default=None,
                   help="layer to steer at (default: best layer from the table)")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                   help="steering strengths, as a fraction of the typical activation size")
    p.add_argument("--n_show", type=int, default=10, help="examples to save and print")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="results/gemma")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_data(path, seed):
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    if "split" not in df.columns:
        idx = np.arange(len(df))
        tr, rest = train_test_split(idx, test_size=0.4, stratify=df["label"], random_state=seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=df["label"].iloc[rest],
                                  random_state=seed)
        df["split"] = "train"
        df.loc[va, "split"] = "val"
        df.loc[te, "split"] = "test"
    return df


@torch.no_grad()
def cache_last_token(model, statements, batch):
    """[n_layers, n_statements, d_model] activations at the last token."""
    n = model.cfg.n_layers
    names = {f"blocks.{l}.hook_resid_post" for l in range(n)}
    out = [[] for _ in range(n)]
    for i in range(0, len(statements), batch):
        _, cache = model.run_with_cache(statements[i:i + batch], names_filter=lambda x: x in names)
        for l in range(n):
            out[l].append(cache[f"blocks.{l}.hook_resid_post"][:, -1, :].float().cpu().numpy())
        del cache
    return np.stack([np.concatenate(o) for o in out])


def steer_hook(layer, vec, seg_lens):
    """Add vec only to the last seg_lens[b] positions of each prompt (the test sentence).
    Prompts are left-padded, so the test sentence always sits at the end.
    During generation with a KV cache each step has length 1: those new tokens
    come after the test sentence, so they are steered too."""
    def fn(resid, hook):
        T = resid.shape[1]
        v = vec.to(resid.dtype)
        if T == 1:
            return resid + v
        pos = torch.arange(T, device=resid.device)
        start = T - seg_lens.to(resid.device)
        mask = (pos[None, :] >= start[:, None]).to(resid.dtype)
        return resid + mask[:, :, None] * v
    return (f"blocks.{layer}.hook_resid_post", fn)


@torch.no_grad()
def last_logprobs(model, prompts, seg_lens, batch, layer, vec):
    """Log-probabilities at the last position. vec=None means no steering."""
    rows = []
    for i in range(0, len(prompts), batch):
        hooks = [] if vec is None else [steer_hook(layer, vec, seg_lens[i:i + batch])]
        with model.hooks(fwd_hooks=hooks):
            logits = model(prompts[i:i + batch], return_type="logits")
        rows.append(logits[:, -1, :].float().log_softmax(-1).cpu())
    return torch.cat(rows)


def segment_lengths(model, prompts, shots):
    """Number of tokens after the example sentences, i.e. the test sentence."""
    n_shots = model.to_tokens(shots).shape[1]
    return torch.tensor([model.to_tokens(p).shape[1] - n_shots for p in prompts])


def first_token(model, text):
    return model.to_tokens(text, prepend_bos=False)[0, 0].item()


def main():
    args = parse_args()
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Run nvidia-smi and fix the driver first.")
    device = "cuda"

    model = HookedTransformer.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                              device=device, default_padding_side="left")
    model.eval()

    df = load_data(os.path.join(args.data_dir, f"{args.dataset}.csv"), args.seed)
    y, split = df["label"].values, df["split"].values
    tr, va, te = split == "train", split == "val", split == "test"

    # ================================================= Step 1: layer table
    print(f"Caching {len(df)} statements from '{args.dataset}' ...")
    acts = cache_last_token(model, df["statement"].tolist(), args.batch)

    rows, directions = [], []
    for l in range(model.cfg.n_layers):
        X = acts[l]
        d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)   # truth direction
        directions.append(d / np.linalg.norm(d))
        rows.append(dict(layer=l,
                         val_auroc=round(roc_auc_score(y[va], X[va] @ d), 3),
                         test_auroc=round(roc_auc_score(y[te], X[te] @ d), 3)))
    table = pd.DataFrame(rows)
    best = int(table.loc[table["val_auroc"].idxmax(), "layer"])
    table["best"] = np.where(table["layer"] == best, "<-- best", "")
    table.to_csv(os.path.join(out_dir, "layer_table.csv"), index=False)
    with open(os.path.join(out_dir, "layer_table.md"), "w") as f:
        f.write("| Layer | Val AUROC | Test AUROC |\n|---|---|---|\n")
        for r in rows:
            cells = [str(r["layer"]), f'{r["val_auroc"]:.3f}', f'{r["test_auroc"]:.3f}']
            if r["layer"] == best:
                cells = [f"**{c}**" for c in cells]
            f.write("| " + " | ".join(cells) + " |\n")
    print("\nLayer table (best layer chosen on validation):")
    print(table.to_string(index=False))

    L = best if args.steer_layer is None else args.steer_layer
    unit = torch.tensor(directions[L], dtype=torch.float32, device=device)
    act_size = float(np.linalg.norm(acts[L], axis=1).mean())
    print(f"\nSteering at layer {L} (typical activation size {act_size:.1f})")

    conditions = [("no_steering", 0.0)]
    for a in args.alphas:
        conditions += [(f"toward_false_{a}", -a), (f"toward_true_{a}", a)]

    def vec_for(signed_alpha):
        return None if signed_alpha == 0 else unit * signed_alpha * act_size

    test_df = df[te].reset_index(drop=True)

    # ================================================= Step 2: next-word test
    nw = test_df[test_df["label"] == 1].copy()
    nw["prefix"] = nw["statement"].str.strip().str.rstrip(".").str.rsplit(" is in ", n=1).str[0] + " is in"
    nw["answer"] = nw["statement"].str.strip().str.rstrip(".").str.rsplit(" is in ", n=1).str[1]
    nw = nw.dropna(subset=["answer"]).reset_index(drop=True)
    nw_prompts = [NEXTWORD_SHOTS + " " + p for p in nw["prefix"]]
    nw_seg = segment_lengths(model, nw_prompts, NEXTWORD_SHOTS)
    ans_ids = torch.tensor([first_token(model, " " + a) for a in nw["answer"]])

    nw_rows, nw_pred = [], {}
    for name, a in conditions:
        lp = last_logprobs(model, nw_prompts, nw_seg, args.batch, L, vec_for(a))
        top = lp.argmax(-1)
        nw_pred[name] = [model.to_string([t]).strip() for t in top.tolist()]
        nw_rows.append(dict(condition=name, strength=a,
                            correct_country_pct=round(100 * (top == ans_ids).float().mean().item(), 1),
                            avg_prob_correct=round(lp.gather(1, ans_ids[:, None]).exp().mean().item(), 3)))
    nw_summary = pd.DataFrame(nw_rows)
    nw_summary.to_csv(os.path.join(out_dir, "nextword_summary.csv"), index=False)
    print(f"\nNext-word test ({len(nw)} true test statements, with example sentences in front)")
    print(nw_summary.to_string(index=False))

    max_a = max(args.alphas)
    show = nw.head(args.n_show)
    ex_rows = []
    for _, r in show.iterrows():
        row = {"prompt": r["prefix"], "correct": r["answer"]}
        for name, a in [(f"toward_false_{max_a}", -max_a), ("no_steering", 0.0),
                        (f"toward_true_{max_a}", max_a)]:
            p = NEXTWORD_SHOTS + " " + r["prefix"]
            v = vec_for(a)
            hooks = [] if v is None else [steer_hook(L, v, segment_lengths(model, [p], NEXTWORD_SHOTS))]
            with model.hooks(fwd_hooks=hooks):
                text = model.generate(p, max_new_tokens=6, do_sample=False, verbose=False,
                                      use_past_kv_cache=True)
            row[name] = text[len(p):].split("\n")[0].strip()
        ex_rows.append(row)
    nw_ex = pd.DataFrame(ex_rows)
    nw_ex.to_csv(os.path.join(out_dir, "nextword_examples.csv"), index=False)
    print(f"\nNext-word examples (strength {max_a}):")
    print(nw_ex.to_string(index=False))

    # ================================================= Step 3: true/false judgment test
    jd_prompts = [JUDGE_SHOTS + s.strip() + " This statement is:" for s in test_df["statement"]]
    jd_seg = segment_lengths(model, jd_prompts, JUDGE_SHOTS)
    t_id, f_id = first_token(model, " TRUE"), first_token(model, " FALSE")
    labels = test_df["label"].values

    jd_rows, jd_says_true = [], {}
    for name, a in conditions:
        lp = last_logprobs(model, jd_prompts, jd_seg, args.batch, L, vec_for(a))
        says_true = (lp[:, t_id] > lp[:, f_id]).numpy()
        jd_says_true[name] = says_true
        jd_rows.append(dict(condition=name, strength=a,
                            judged_true_pct=round(100 * says_true.mean(), 1),
                            accuracy_pct=round(100 * (says_true == (labels == 1)).mean(), 1)))
    jd_summary = pd.DataFrame(jd_rows)
    jd_summary.to_csv(os.path.join(out_dir, "judgment_summary.csv"), index=False)
    print(f"\nTrue/false judgment test ({len(test_df)} test statements, "
          f"{100 * labels.mean():.0f}% actually true)")
    print(jd_summary.to_string(index=False))

    jd_ex = pd.DataFrame({
        "statement": test_df["statement"],
        "actually": np.where(labels == 1, "TRUE", "FALSE"),
        f"toward_false_{max_a}": np.where(jd_says_true[f"toward_false_{max_a}"], "TRUE", "FALSE"),
        "no_steering": np.where(jd_says_true["no_steering"], "TRUE", "FALSE"),
        f"toward_true_{max_a}": np.where(jd_says_true[f"toward_true_{max_a}"], "TRUE", "FALSE"),
    })
    jd_ex.to_csv(os.path.join(out_dir, "judgment_examples.csv"), index=False)
    print(f"\nJudgment examples (strength {max_a}):")
    print(jd_ex.head(args.n_show).to_string(index=False))

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()