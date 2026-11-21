"""Model-agnostic residual-stream backends for the final-week cross-model work (2026-11-21).

One interface for GPT-2 Small and Gemma-2-2B. Both expose:
  * ``tokenize(text) -> list[int]``: text token ids **without** BOS (caller prepends ``bos_id``);
  * ``n_layers``, ``d_model``, ``bos_id``, ``name``;
  * ``depth_fraction(L)``: fraction of transformer blocks applied before the SAE site;
  * ``run(ids, splices, capture)``: one forward pass on a [B, T] batch. ``splices`` maps a layer to a list of
    ``(row, fn, mask)``: at that layer the activation of batch row ``row`` at positions ``mask`` [T] is replaced by
    ``fn(x[mask])``. ``capture`` is a set of layers whose (pre-splice) activations of every row are returned.
    Returns (logits [B, T, V], {layer: activations [B, T, d]}).

Hook conventions (each SAE family's own training convention; see the 2026-11-21 audit §3.9):
  * GPT-2: TransformerLens ``HookedTransformer.from_pretrained('gpt2-small')`` with default processing,
    site ``blocks.L.hook_resid_pre`` (L = 0..11). Gate G-HOOK (2026-11-21): this equals the raw HF
    ``hidden_states[L]`` minus its per-token mean over d_model (relative difference ~1e-7); logits identical.
    Depth fraction L / 12.
  * Gemma: Hugging Face ``Gemma2ForCausalLM`` (bf16, eager attention), site = output of ``model.model.layers[L]``
    (= ``blocks.L.hook_resid_post``, L = 0..25), exactly as M6/R3. Depth fraction (L + 1) / 26.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple

import torch

GEMMA_ID, GEMMA_REV = 'google/gemma-2-2b', 'c5ebcd40d208330abc697524c919956e692655cf'


def _apply(hidden: torch.Tensor, items) -> torch.Tensor:
    new = hidden.clone()
    for row, fn, mask in items:
        x = hidden[row][mask]
        new[row, mask] = fn(x.float()).to(hidden.dtype)
    return new


class GPT2Backend:
    name = 'gpt2'
    n_layers = 12
    d_model = 768
    bos_id = 50256

    def __init__(self, device='cuda'):
        from transformer_lens import HookedTransformer
        from transformers import GPT2TokenizerFast
        self.model = HookedTransformer.from_pretrained('gpt2-small', device=device).eval()
        self.tok = GPT2TokenizerFast.from_pretrained('gpt2')
        self.device = device

    def tokenize(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)['input_ids']

    def offsets(self, text: str):
        return self.tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']

    @staticmethod
    def depth_fraction(L: int) -> float:
        return L / 12

    @torch.no_grad()
    def run(self, ids: torch.Tensor, splices: Dict[int, list] = None, capture: Set[int] = frozenset()):
        splices = splices or {}
        cap: Dict[int, torch.Tensor] = {}
        hooks = []
        for L in sorted(set(capture) | set(splices)):
            def hook(act, hook, L=L):
                if L in capture:
                    cap[L] = act.detach().clone()
                if L in splices and splices[L]:
                    return _apply(act, splices[L])
                return None
            hooks.append((f'blocks.{L}.hook_resid_pre', hook))
        logits = self.model.run_with_hooks(ids.to(self.device), fwd_hooks=hooks)
        return logits, cap


class GemmaBackend:
    name = 'gemma'
    n_layers = 26
    d_model = 2304
    bos_id = 2

    def __init__(self, device='cuda', dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(GEMMA_ID, revision=GEMMA_REV)
        self.model = AutoModelForCausalLM.from_pretrained(GEMMA_ID, revision=GEMMA_REV, dtype=dtype,
                                                          attn_implementation='eager').to(device).eval()
        self.layers = self.model.model.layers
        assert len(self.layers) == 26
        self.device = device

    def tokenize(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)['input_ids']

    def offsets(self, text: str):
        return self.tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']

    @staticmethod
    def depth_fraction(L: int) -> float:
        return (L + 1) / 26

    @torch.no_grad()
    def run(self, ids: torch.Tensor, splices: Dict[int, list] = None, capture: Set[int] = frozenset()):
        splices = splices or {}
        cap: Dict[int, torch.Tensor] = {}
        handles = []
        for L in sorted(set(capture) | set(splices)):
            def hook(module, inputs, output, L=L):
                hidden = output[0] if isinstance(output, tuple) else output
                if L in capture:
                    cap[L] = hidden.detach().clone()
                if L in splices and splices[L]:
                    new = _apply(hidden, splices[L])
                    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
                return None
            handles.append(self.layers[L].register_forward_hook(hook))
        try:
            ids = ids.to(self.device)
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return out.logits, cap


def make_backend(name: str, device='cuda'):
    return {'gpt2': GPT2Backend, 'gemma': GemmaBackend}[name](device=device)
