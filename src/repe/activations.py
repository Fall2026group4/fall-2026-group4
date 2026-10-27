"""Run prompts through GPT-2 (TransformerLens) and keep residual-stream activations.

13 positions per prompt: blocks.0..11.hook_resid_pre (where the SAEs attach)
plus blocks.11.hook_resid_post (the final residual).
"""

import numpy as np

HOOKS = [f"blocks.{layer}.hook_resid_pre" for layer in range(12)] + ["blocks.11.hook_resid_post"]
POSITIONS = [f"L{layer}" for layer in range(12)] + ["final"]
D_MODEL = 768


def load_model(device="cpu"):
    """GPT-2 Small in TransformerLens (downloads ~500 MB the first time)."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    return model


def count_tokens(model, text):
    """Number of tokens GPT-2 sees for a prompt, including the BOS token."""
    return int(model.to_tokens(text, prepend_bos=True).shape[1])


def cache_prompt_set(model, prompts_df):
    """Activations for every prompt, one prompt at a time (no padding issues).

    Returns:
      last : [n_prompts, 13, 768] at the LAST token (the final period)
      mean : [n_prompts, 13, 768] averaged over the STATEMENT's tokens only
      meta : prompts_df plus n_tokens and n_statement_tokens
    """
    import torch

    n = len(prompts_df)
    last = np.zeros((n, len(HOOKS), D_MODEL), dtype=np.float32)
    mean = np.zeros_like(last)
    n_tokens, n_statement = [], []
    hook_set = set(HOOKS)

    with torch.no_grad():
        for i, (prompt, statement) in enumerate(zip(prompts_df["prompt"], prompts_df["statement"])):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            k = int(model.to_tokens(" " + statement, prepend_bos=False).shape[1])
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_set)
            for j, hook in enumerate(HOOKS):
                acts = cache[hook][0].detach().cpu().numpy()  # [n_tokens, 768]
                last[i, j] = acts[-1]
                mean[i, j] = acts[-k:].mean(axis=0)
            n_tokens.append(int(tokens.shape[1]))
            n_statement.append(k)
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"    cached {i + 1}/{n}")

    meta = prompts_df.copy()
    meta["n_tokens"] = n_tokens
    meta["n_statement_tokens"] = n_statement
    return last, mean, meta


def save_prompt_set(directory, name, last, mean, meta):
    """Save <name>.npz (activations) and <name>.csv (one row per prompt, same order)."""
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{name}.npz", last=last, mean=mean)
    meta.to_csv(directory / f"{name}.csv", index=False)


def load_prompt_set(directory, name):
    """Load what save_prompt_set wrote: (last, mean, meta)."""
    import pandas as pd

    data = np.load(directory / f"{name}.npz")
    return data["last"], data["mean"], pd.read_csv(directory / f"{name}.csv")