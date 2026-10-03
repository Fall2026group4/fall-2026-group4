"""SAE utilities for the Geometry of Truth analysis."""

import torch
from sae_lens import SAE


SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.10.hook_resid_pre"
SAE_LAYER = 10


def load_geometry_sae(device="cpu"):
    """Load the pretrained GPT-2 Small ReLU SAE used for Geometry analysis."""

    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
    )

    return sae


def encode_residual_activations(
    sae,
    activations,
    batch_size=64,
):
    """Encode residual-stream activations into SAE feature activations.

    Parameters
    ----------
    sae
        Loaded SAE Lens SAE.

    activations : torch.Tensor
        Shape [n_examples, 768].

    batch_size : int
        Number of examples encoded at once.

    Returns
    -------
    torch.Tensor
        SAE feature activations with shape
        [n_examples, d_sae].
    """

    if activations.ndim != 2:
        raise ValueError(
            "activations must have shape [examples, hidden_dim]"
        )

    encoded_batches = []

    with torch.inference_mode():
        for start in range(
            0,
            activations.shape[0],
            batch_size,
        ):
            batch = activations[
                start:start + batch_size
            ]

            features = sae.encode(batch)

            encoded_batches.append(
                features.detach().cpu()
            )

    return torch.cat(
        encoded_batches,
        dim=0,
    )


def select_sae_layer(
    cached_activations,
    layer=SAE_LAYER,
):
    """Select one GPT-2 residual layer from Stage 1 cached activations.

    Expected input shape:
    [examples, 12, 768]
    """

    if cached_activations.ndim != 3:
        raise ValueError(
            "cached_activations must have shape "
            "[examples, layers, hidden_dim]"
        )

    return cached_activations[:, layer, :]