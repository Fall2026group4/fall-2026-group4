"""GPT-2 activation extraction utilities.

Reusable functions for loading GPT-2 Small and caching the
last-token residual-stream activations across transformer layers.
"""

import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
N_LAYERS = 12


def load_gpt2_small(device="cpu"):
    """Load GPT-2 Small with TransformerLens."""
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
    )

    # GPT-2 has no pad token by default.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model.eval()
    return model


def extract_last_token_residuals(
    model,
    texts,
    batch_size=32,
    device="cpu",
):
    """Extract last-token hook_resid_pre activations for layers 0-11.

    Returns
    -------
    torch.Tensor
        Shape: [n_examples, 12, 768]
    """

    hook_names = [
        f"blocks.{layer}.hook_resid_pre"
        for layer in range(N_LAYERS)
    ]

    all_batches = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = model.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Match the baseline notebook by explicitly prepending BOS.
        bos = torch.full(
            (input_ids.shape[0], 1),
            model.tokenizer.bos_token_id,
            dtype=input_ids.dtype,
            device=device,
        )

        bos_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=device,
        )

        tokens = torch.cat([bos, input_ids], dim=1)
        attention_mask = torch.cat(
            [bos_mask, attention_mask],
            dim=1,
        )

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=hook_names,
                return_type=None,
            )

        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            tokens.shape[0],
            device=device,
        )

        layer_activations = []

        for hook_name in hook_names:
            resid = cache[hook_name]
            last_resid = resid[
                batch_indices,
                last_indices,
                :
            ]

            layer_activations.append(
                last_resid.detach().cpu()
            )

        # [batch, 12, 768]
        batch_tensor = torch.stack(
            layer_activations,
            dim=1,
        )

        all_batches.append(batch_tensor)

        del cache

    return torch.cat(all_batches, dim=0)