"""Shared, testable helpers for M6: Gemma Scope SAE reconstruction and splicing on The Pile.

Pre-registration: results/gemma2_pile/gemma_scope_splicing/README.md and preregistration.json (commit b39dcc2).

Part 1 (Commit 2A), *sampling*: deterministic selection of the evaluation and reference documents, BOS +
truncation, and the hashes that make the sample reproducible.

Part 2 (Commit 2B), *SAE, splicing and metrics*: pre-registration checks for the evaluation fields, the Gemma Scope
JumpReLU SAE (encode/decode, shape validation), position masks (BOS and padding never touched), the forward hook
that captures and optionally replaces a decoder layer's output (``resid_post``), the controls, per-document
reconstruction and behaviour sums, EV from sums, the document bootstrap and the H-G1 statistic.

Design rules (from the pre-registration):
* Documents are identified by their 0-based line number in the pinned ``val.jsonl.zst``.
* Order = ``numpy.random.default_rng(seed).permutation(n_lines)``. Walking that order, a document is *eligible* if it
  has at least ``min_text_tokens`` text tokens (BOS not counted). The first ``n_eval`` eligible documents form the
  evaluation set, the next ``n_reference`` eligible documents the reference set. The two sets are therefore disjoint
  by construction; :func:`select_documents` also asserts it.
* Model input per document = [BOS] + the first ``max_tokens_including_bos - 1`` text tokens.
* Hashes: SHA-256 of the UTF-8 text, and SHA-256 of the token ids used (little-endian int32 bytes).

Only numpy and the standard library are imported at module level, so the sampling tests need no torch or
transformers. ``zstandard`` is imported lazily inside :func:`iter_jsonl_zst`; torch is imported inside the Part 2
functions that need it.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

# Fields of preregistration.json that the sampling code depends on. Missing fields are an error, not a default.
REQUIRED_PREREG_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ('data', 'hf_dataset'),
    ('data', 'revision'),
    ('data', 'file'),
    ('data', 'bytes'),
    ('data', 'sha256'),
    ('data', 'source_label_field'),
    ('data', 'test_file_used'),
    ('sampling', 'seed'),
    ('sampling', 'min_text_tokens'),
    ('sampling', 'n_eval'),
    ('sampling', 'n_reference'),
    ('sampling', 'disjoint'),
    ('tokenization', 'tokenizer'),
    ('tokenization', 'revision'),
    ('tokenization', 'prepend_bos'),
    ('tokenization', 'max_tokens_including_bos'),
)


# ----------------------------------------------------------------------------------------------------------- config
def get_field(cfg: dict, path: Sequence[str]):
    """Return cfg[path[0]][path[1]]...; raise KeyError naming the full path if any level is missing."""
    node = cfg
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"preregistration field missing: {'.'.join(path[:i + 1])}")
        node = node[key]
    return node


def load_preregistration(path: Path) -> dict:
    """Load preregistration.json and check that every field the sampling code reads is present and sane."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    for field in REQUIRED_PREREG_FIELDS:
        get_field(cfg, field)
    if cfg['data']['test_file_used'] is not False:
        raise ValueError('preregistration says the Pile test file is used; M6 must never use it')
    if cfg['sampling']['disjoint'] is not True:
        raise ValueError('preregistration must require disjoint evaluation and reference sets')
    rev = str(cfg['tokenization']['revision'])
    if len(rev) != 40 or any(c not in '0123456789abcdef' for c in rev):
        raise ValueError('tokenization.revision must be a full 40-character commit SHA (Amendment 1), '
                         f'not {rev!r}')
    if cfg['tokenization']['prepend_bos'] is not True:
        raise ValueError('preregistration must prepend BOS')
    for key in ('n_eval', 'n_reference', 'min_text_tokens'):
        if int(cfg['sampling'][key]) <= 0:
            raise ValueError(f'sampling.{key} must be positive')
    if int(cfg['tokenization']['max_tokens_including_bos']) < 2:
        raise ValueError('max_tokens_including_bos must be at least 2 (BOS + one text token)')
    return cfg


# ----------------------------------------------------------------------------------------------------------- hashes
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks (the Pile validation file is ~338 MB)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of the document text encoded as UTF-8."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_token_ids(ids: Sequence[int]) -> str:
    """SHA-256 of token ids as little-endian int32 bytes (Gemma's vocabulary, 256 000, fits in int32)."""
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int32).max):
        raise ValueError('token id out of int32 range')
    return hashlib.sha256(arr.astype('<i4').tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------- corpus
def iter_jsonl_zst(path: Path) -> Iterator[Tuple[int, dict]]:
    """Yield (0-based line index, parsed JSON object) for every line of a .jsonl.zst file, streaming."""
    import zstandard  # lazy: only needed when the real corpus is read

    with open(path, 'rb') as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding='utf-8')
        for i, line in enumerate(text):
            yield i, json.loads(line)


def count_lines_jsonl_zst(path: Path) -> int:
    """Number of documents (lines) in a .jsonl.zst file."""
    n = 0
    for n, _ in enumerate(iter_jsonl_zst(path), start=1):
        pass
    return n


def load_lines(path: Path, wanted: Iterable[int]) -> Dict[int, dict]:
    """Return {line index: parsed object} for the requested line indices (one streaming pass)."""
    wanted = set(int(i) for i in wanted)
    out: Dict[int, dict] = {}
    if not wanted:
        return out
    last = max(wanted)
    for i, obj in iter_jsonl_zst(path):
        if i in wanted:
            out[i] = obj
        if i >= last:
            break
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f'{len(missing)} requested line indices are beyond the end of the file')
    return out


# -------------------------------------------------------------------------------------------------------- selection
def permutation_order(n_lines: int, seed: int) -> np.ndarray:
    """The pre-registered document order: numpy.random.default_rng(seed).permutation(n_lines).

    The selected line indices are also saved in the manifest, which is the authoritative record of the sample
    (NumPy does not guarantee identical Generator streams across all future versions; the version is recorded).
    """
    if n_lines <= 0:
        raise ValueError('n_lines must be positive')
    return np.random.default_rng(int(seed)).permutation(int(n_lines))


def select_documents(order: Iterable[int], n_text_tokens: Callable[[int], int], *, n_eval: int,
                     n_reference: int, min_text_tokens: int) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Walk ``order``; keep documents with >= min_text_tokens text tokens; first n_eval -> eval, next n_reference -> ref.

    ``n_text_tokens(line_index)`` returns the number of text tokens (BOS excluded) of that document.
    Returns (eval line indices, reference line indices, counters). Raises if the order runs out before both sets
    are full. Disjointness is guaranteed by construction (each index is assigned once) and asserted.
    """
    eval_ids: List[int] = []
    ref_ids: List[int] = []
    seen = set()
    examined = skipped = 0
    for idx in order:
        idx = int(idx)
        if idx in seen:
            raise ValueError(f'line index {idx} appears twice in the order; the order must be a permutation')
        seen.add(idx)
        examined += 1
        if n_text_tokens(idx) < min_text_tokens:
            skipped += 1
            continue
        if len(eval_ids) < n_eval:
            eval_ids.append(idx)
        else:
            ref_ids.append(idx)
        if len(eval_ids) == n_eval and len(ref_ids) == n_reference:
            break
    if len(eval_ids) < n_eval or len(ref_ids) < n_reference:
        raise ValueError(f'order exhausted: {len(eval_ids)}/{n_eval} eval, {len(ref_ids)}/{n_reference} reference '
                         f'after examining {examined} documents')
    assert not set(eval_ids) & set(ref_ids), 'evaluation and reference sets overlap'
    return eval_ids, ref_ids, {'examined': examined, 'skipped_too_short': skipped}


def select_from_corpus(order: np.ndarray, load_texts: Callable[[Sequence[int]], Dict[int, str]],
                       count_text_tokens: Callable[[str], int], *, n_eval: int, n_reference: int,
                       min_text_tokens: int, initial_candidates: int = 2048):
    """Run :func:`select_documents` over ``order`` without loading the whole corpus.

    Texts are loaded for a prefix of the order; if the prefix holds too few eligible documents, the prefix is
    doubled and selection is redone from the start. The result therefore depends only on ``order`` and the
    eligibility rule, never on the prefix size. Returns (eval ids, ref ids, counters, {line index: text}).
    """
    n_cand = min(int(initial_candidates), len(order))
    while True:
        prefix = [int(i) for i in order[:n_cand]]
        texts = load_texts(prefix)
        counts: Dict[int, int] = {}

        def n_tok(i: int) -> int:
            if i not in counts:
                counts[i] = int(count_text_tokens(texts[i]))
            return counts[i]

        try:
            eval_ids, ref_ids, info = select_documents(prefix, n_tok, n_eval=n_eval, n_reference=n_reference,
                                                       min_text_tokens=min_text_tokens)
        except ValueError:
            if n_cand >= len(order):
                raise
            n_cand = min(2 * n_cand, len(order))
            continue
        info['candidates_loaded'] = n_cand
        return eval_ids, ref_ids, info, texts


# ----------------------------------------------------------------------------------------------------- tokenization
def truncate_with_bos(text_ids: Sequence[int], bos_id: int, max_tokens_including_bos: int) -> Tuple[List[int], bool]:
    """[BOS] + the first (max - 1) text tokens. Returns (ids used, whether the text was truncated)."""
    if max_tokens_including_bos < 2:
        raise ValueError('max_tokens_including_bos must be at least 2')
    keep = max_tokens_including_bos - 1
    ids = [int(bos_id)] + [int(t) for t in text_ids[:keep]]
    return ids, len(text_ids) > keep


def manifest_row(*, set_name: str, rank: int, perm_position: int, line_index: int, source: str, text: str,
                 text_ids: Sequence[int], bos_id: int, max_tokens_including_bos: int) -> dict:
    """One row of sample_manifest.csv: identifiers and hashes, never the text itself."""
    used, truncated = truncate_with_bos(text_ids, bos_id, max_tokens_including_bos)
    return {
        'set': set_name,
        'rank': int(rank),
        'perm_position': int(perm_position),
        'line_index': int(line_index),
        'pile_set_name': source,
        'n_chars': len(text),
        'text_sha256': sha256_text(text),
        'n_text_tokens': len(text_ids),
        'n_tokens_used': len(used),
        'truncated': bool(truncated),
        'token_ids_sha256': sha256_token_ids(used),
    }


MANIFEST_COLUMNS = ['set', 'rank', 'perm_position', 'line_index', 'pile_set_name', 'n_chars', 'text_sha256',
                    'n_text_tokens', 'n_tokens_used', 'truncated', 'token_ids_sha256']


# ===================================================================================================================
# Part 2 (Commit 2B): Gemma Scope SAE, splicing and metrics
# ===================================================================================================================
# Fields of preregistration.json that the evaluation (script 03) reads, in addition to REQUIRED_PREREG_FIELDS.
REQUIRED_PREREG_FIELDS_EVAL: Tuple[Tuple[str, ...], ...] = (
    ('model', 'hf_id'), ('model', 'revision'), ('model', 'dtype'), ('model', 'attn_implementation'),
    ('hook', 'site'), ('layers',),
    ('sae', 'hf_repo'), ('sae', 'hf_revision'), ('sae', 'sae_lens_release'), ('sae', 'architecture'),
    ('sae', 'd_in'), ('sae', 'd_sae'),
    ('sae', 'compute_dtype'), ('sae', 'ids'), ('sae', 'substitution_allowed'),
    ('positions', 'behaviour_predictions'), ('conditions',),
    ('seeds', 'random_error'), ('seeds', 'bootstrap'), ('bootstrap', 'n_resamples'),
    ('low_n_source_threshold_docs',), ('validity_gates', 'V3_determinism_identity_max_mean_kl_nats'),
)
CONDITIONS = ['original', 'original_repeat', 'sae_splice', 'norm_matched_random_error', 'mean_ablation', 'identity']
SAE_PARAM_KEYS = ('W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold')


def _is_full_sha(value) -> bool:
    s = str(value)
    return len(s) == 40 and all(c in '0123456789abcdef' for c in s)


def load_preregistration_eval(path: Path) -> dict:
    """:func:`load_preregistration` plus every field script 03 reads; inconsistencies are errors, never defaults."""
    cfg = load_preregistration(path)
    for field in REQUIRED_PREREG_FIELDS_EVAL:
        get_field(cfg, field)
    for field in (('model', 'revision'), ('sae', 'hf_revision')):
        if not _is_full_sha(get_field(cfg, field)):
            raise ValueError(f"{'.'.join(field)} must be a full 40-character commit SHA")
    if cfg['model']['revision'] != cfg['tokenization']['revision']:
        raise ValueError('model.revision and tokenization.revision differ (Amendment 1 pins both to one commit)')
    if cfg['hook']['site'] != 'resid_post':
        raise ValueError("hook.site must be 'resid_post'")
    if cfg['sae']['architecture'] != 'jumprelu':
        raise ValueError("sae.architecture must be 'jumprelu'")
    if cfg['sae']['substitution_allowed'] is not False:
        raise ValueError('sae.substitution_allowed must be false (gate V5)')
    layers = [int(x) for x in cfg['layers']]
    ids = cfg['sae']['ids']
    if sorted(int(k) for k in ids) != sorted(layers):
        raise ValueError(f'sae.ids keys {sorted(ids)} do not match layers {layers}')
    for layer in layers:
        if not str(ids[str(layer)]).startswith(f'layer_{layer}/'):
            raise ValueError(f'sae.ids[{layer}] = {ids[str(layer)]!r} is not an SAE for layer {layer}')
    if list(cfg['conditions']) != CONDITIONS:
        raise ValueError(f'conditions must be exactly {CONDITIONS}')
    if float(cfg['validity_gates']['V3_determinism_identity_max_mean_kl_nats']) <= 0:
        raise ValueError('V3 threshold must be positive')
    random_error_seed(cfg, layers[0])  # the seed rule must be parseable
    return cfg


def random_error_seed(cfg: dict, layer: int) -> int:
    """Pre-registered seed for the norm-matched random error at ``layer`` ('1000 + layer')."""
    rule = str(cfg['seeds']['random_error']).replace(' ', '')
    if rule != '1000+layer':
        raise ValueError(f"unexpected seeds.random_error rule {cfg['seeds']['random_error']!r}")
    return 1000 + int(layer)


# ---------------------------------------------------------------------------------------------------- Gemma Scope SAE
def validate_sae_params(params: dict, d_model: int, d_sae: int) -> None:
    """Check keys and shapes of a Gemma Scope ``params.npz`` against the model width and the pre-registered d_sae."""
    missing = [k for k in SAE_PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f'SAE params missing keys: {missing}')
    expected = {'W_enc': (d_model, d_sae), 'W_dec': (d_sae, d_model), 'b_enc': (d_sae,), 'b_dec': (d_model,),
                'threshold': (d_sae,)}
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape:
            raise ValueError(f'SAE {key} has shape {tuple(params[key].shape)}, expected {shape}')


class JumpReLUSAE:
    """Gemma Scope JumpReLU SAE, computed in float32.

    encode: f = ReLU(x W_enc + b_enc) * 1[x W_enc + b_enc > threshold]   (strict '>' at the threshold)
    decode: x_hat = f W_dec + b_dec
    No input centring by b_dec and no activation normalisation (sae-lens: apply_b_dec_to_input=False,
    normalize_activations=None for these SAEs). Equivalence with sae-lens is *checked* in the smoke run, not assumed.
    """

    def __init__(self, params: dict, d_model: int, d_sae: int, device='cpu'):
        import torch
        validate_sae_params(params, d_model, d_sae)
        for k in SAE_PARAM_KEYS:
            setattr(self, k, torch.as_tensor(np.asarray(params[k]), dtype=torch.float32, device=device))

    def encode(self, x):
        import torch
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def __call__(self, x):
        """x: float32 [N, d_model] -> (x_hat float32 [N, d_model], L0 per token int [N])."""
        f = self.encode(x)
        return self.decode(f), (f != 0).sum(-1)


# ------------------------------------------------------------------------------------------------------------ masks
def splice_mask(attention_mask):
    """Bool [B, T]: True at real (attention_mask = 1), non-BOS positions. Assumes BOS at position 0, right padding."""
    m = attention_mask.bool().clone()
    m[:, 0] = False
    return m


def prediction_mask(attention_mask):
    """Bool [B, T]: True at position t if 1 <= t <= len-2, i.e. t is not BOS and the target t+1 is a real token."""
    am = attention_mask.bool()
    pm = am.new_zeros(am.shape)
    if am.shape[1] > 2:
        pm[:, 1:-1] = am[:, 2:]
    return pm


# ----------------------------------------------------------------------------------------------------- forward hook
class LayerSplice:
    """Forward hook on one decoder layer (its output = ``resid_post``).

    Every pass stores a *copy* of the layer's original output in ``captured`` (a copy, so no later in-place operation
    in the model can change it). If ``fn`` is set, the output at ``mask`` positions is replaced by
    ``fn(original[mask])``; all other positions (BOS, padding) are returned unchanged.
    Works whether the layer returns a tensor or a tuple whose first element is the hidden state.
    """

    def __init__(self, module):
        self.captured = None
        self.fn = None
        self.mask = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.captured = hidden.detach().clone()
        if self.fn is None:
            return None
        if self.mask is None or self.mask.shape != hidden.shape[:2]:
            raise ValueError('splice mask missing or of the wrong shape')
        new = hidden.clone()
        new[self.mask] = self.fn(hidden[self.mask]).to(hidden.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def set(self, fn=None, mask=None):
        self.fn, self.mask = fn, mask

    def remove(self):
        self.handle.remove()


# --------------------------------------------------------------------------------------------------------- controls
def norm_matched_random_error(x, x_hat, generator):
    """x + ||x - x_hat|| * u per token, u an isotropic Gaussian direction drawn on CPU from ``generator`` (float32)."""
    import torch
    e = torch.randn(x.shape, generator=generator, dtype=torch.float32)
    e = (e / e.norm(dim=-1, keepdim=True)).to(x.device)
    return x + e * (x - x_hat).norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------------------------- per-document sums
def reconstruction_sums(x, x_hat, l0) -> dict:
    """Sums over one document's evaluated tokens (float64). sum_x is a [d] numpy vector, needed for EV's mean."""
    import torch
    x, x_hat = x.double(), x_hat.double()
    return {
        'n_tok': int(x.shape[0]),
        'sse': float((x - x_hat).square().sum()),
        'sum_sq': float(x.square().sum()),
        'cos_sum': float(torch.nn.functional.cosine_similarity(x, x_hat, dim=-1, eps=1e-8).sum()),
        'l0_sum': float(l0.double().sum()),
        'sum_x': x.sum(0).cpu().numpy(),
    }


def ev_from_sums(sse, sum_sq, sum_x, n_tok) -> float:
    """EV = 1 - SSE / sum ||x - x_bar||^2 over the pooled documents, x_bar = their own token mean.

    Same definition as src/evaluation/faithfulness_metrics.reconstruction_metrics on the concatenated tokens.
    """
    sse, sum_sq, n_tok = np.asarray(sse, float), np.asarray(sum_sq, float), np.asarray(n_tok, float)
    sum_x = np.asarray(sum_x, float)
    n = n_tok.sum()
    mean = sum_x.sum(0) / n
    total = sum_sq.sum() - n * float(mean @ mean)
    return float(1 - sse.sum() / max(total, 1e-12))


def behaviour_sums(logits_orig, logits_new, input_ids, pred_mask, chunk: int = 128) -> dict:
    """Sums over predictions at ``pred_mask`` positions (float64): CE original / new, KL(orig || new), top-1 agree.

    Position t predicts token t+1. Log-softmax is taken in float32 in chunks of positions (Gemma's vocabulary is
    256 000, so a full-sequence float32 copy would need ~1 GB per 1024 tokens).
    """
    import torch
    out = {'n_pred': 0, 'ce_orig_sum': 0.0, 'ce_new_sum': 0.0, 'kl_sum': 0.0, 'top1_agree': 0}
    T = input_ids.shape[1]
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        m = pred_mask[:, s:e]
        if not bool(m.any()):
            continue
        lo = torch.log_softmax(logits_orig[:, s:e].float(), -1)
        ln = torch.log_softmax(logits_new[:, s:e].float(), -1)
        tgt = input_ids[:, s + 1:e + 1].unsqueeze(-1)
        out['n_pred'] += int(m.sum())
        out['ce_orig_sum'] += float(-lo.gather(-1, tgt)[..., 0][m].double().sum())
        out['ce_new_sum'] += float(-ln.gather(-1, tgt)[..., 0][m].double().sum())
        out['kl_sum'] += float((lo.exp() * (lo - ln)).sum(-1)[m].double().sum())
        out['top1_agree'] += int((lo.argmax(-1) == ln.argmax(-1))[m].sum())
    return out


# -------------------------------------------------------------------------------------------- bootstrap and H-G1
def bootstrap_indices(n_docs: int, n_resamples: int, seed: int) -> np.ndarray:
    """[n_resamples, n_docs] document indices; one matrix shared by all statistics of a set (paired resampling)."""
    return np.random.default_rng(int(seed)).integers(0, n_docs, size=(int(n_resamples), int(n_docs)))


def bootstrap_ci(stat: Callable[[np.ndarray], float], idx: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """Percentile interval of ``stat(document indices)`` over the resamples in ``idx``."""
    vals = np.array([stat(row) for row in idx], dtype=float)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def hg1_verdict(lo: float, hi: float) -> str:
    """Pre-registered H-G1 decision: supported if lo > 0, contradicted if hi < 0, otherwise inconclusive."""
    if lo > 0:
        return 'supported'
    if hi < 0:
        return 'contradicted'
    return 'inconclusive'


def h_g1(dce_sae_doc_mean, dce_random_doc_mean, idx: np.ndarray, levels=(0.95, 1 - 0.05 / 3)) -> dict:
    """H-G1 at one layer: d = mean over documents of (doc-mean dCE_SAE - doc-mean dCE_random), bootstrap CIs.

    The verdict uses the 95 % interval; the Bonferroni (98.33 %) interval is reported as a sensitivity check.
    """
    d = np.asarray(dce_sae_doc_mean, float) - np.asarray(dce_random_doc_mean, float)
    out = {'n_docs': int(d.size), 'mean_d': float(d.mean())}
    for level in levels:
        lo, hi = bootstrap_ci(lambda r: float(d[r].mean()), idx, level)
        tag = f'{100 * level:.2f}'.rstrip('0').rstrip('.')
        out[f'ci{tag}_lo'], out[f'ci{tag}_hi'] = lo, hi
    lo95, hi95 = out['ci95_lo'], out['ci95_hi']
    out['verdict'] = hg1_verdict(lo95, hi95)
    out['verdict_bonferroni'] = hg1_verdict(out['ci98.33_lo'], out['ci98.33_hi'])
    return out


# ------------------------------------------------------------------------- engineering safeguards (A1, A2, A6, H4a)
class NumericsError(RuntimeError):
    """A NaN or inf was found where only finite values are valid (acceptance requirement 4)."""


def assert_finite(name: str, *tensors) -> None:
    """Raise NumericsError if any tensor holds a NaN or inf."""
    import torch
    for i, t in enumerate(tensors):
        if not bool(torch.isfinite(t).all()):
            raise NumericsError(f'non-finite values in {name} (tensor {i}, shape {tuple(t.shape)}, dtype {t.dtype})')


def assert_finite_values(name: str, values: dict) -> None:
    """Raise NumericsError if any numeric value of a dict (e.g. per-document sums) is NaN or inf."""
    bad = [k for k, v in values.items() if isinstance(v, (int, float, np.floating, np.integer))
           and not np.isfinite(float(v))]
    if bad:
        raise NumericsError(f'non-finite values in {name}: {bad}')


def nonfinite_columns(table, columns=None) -> List[str]:
    """Numeric columns of a pandas DataFrame (or the given subset) that contain NaN or inf."""
    cols = list(columns) if columns is not None else [c for c in table.columns
                                                     if np.issubdtype(table[c].dtype, np.number)]
    return [c for c in cols if not np.isfinite(table[c].to_numpy(dtype=float)).all()]


def git_provenance(repo: Path, paths: Sequence[str]) -> dict:
    """HEAD commit of ``repo`` and which of ``paths`` differ from it (modified, staged or untracked)."""
    import subprocess

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout

    try:
        commit = git('rev-parse', 'HEAD').strip()
        status = git('status', '--porcelain', '--untracked-files=all', '--', *paths)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}', 'commit': None, 'dirty_paths': None}
    dirty = sorted({line[3:].strip() for line in status.splitlines() if line.strip()})
    return {'available': True, 'commit': commit, 'dirty_paths': dirty}


def check_pinned_commit(found, pinned: str, what: str) -> str:
    """'verified' if the loaded object's commit equals the pinned one, 'unverifiable' if the library does not expose
    it; raise ValueError on a mismatch (a different revision would be a protocol deviation)."""
    if not found:
        return 'unverifiable'
    if str(found) != str(pinned):
        raise ValueError(f'{what} was loaded from commit {found}, but the pre-registration pins {pinned}')
    return 'verified'


def logit_precision_sums(logits_model, head_input, w_out, softcap, input_ids, pred_mask, chunk: int = 128) -> dict:
    """Measure (do not change) the precision of the model's bf16 logits (decision H4a, a documented limitation).

    The reference recomputes the same output head in float32 from the head's own input:
    ref = softcap * tanh((head_input @ w_out^T) / softcap) (no softcap if ``softcap`` is None). Returns sums over
    ``pred_mask`` positions: KL(P_ref || P_model), |CE_model - CE_ref|, top-1 disagreements, and the max |logit diff|.
    """
    import torch
    out = {'n_pred': 0, 'kl_ref_vs_model_sum': 0.0, 'abs_ce_diff_sum': 0.0, 'top1_disagree': 0,
           'max_abs_logit_diff': 0.0}
    T = input_ids.shape[1]
    w = w_out.float()
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        m = pred_mask[:, s:e]
        if not bool(m.any()):
            continue
        ref = head_input[:, s:e].float() @ w.T
        if softcap is not None:
            ref = softcap * torch.tanh(ref / softcap)
        lm = logits_model[:, s:e].float()
        lr, lp = torch.log_softmax(ref, -1), torch.log_softmax(lm, -1)
        tgt = input_ids[:, s + 1:e + 1].unsqueeze(-1)
        out['n_pred'] += int(m.sum())
        out['kl_ref_vs_model_sum'] += float((lr.exp() * (lr - lp)).sum(-1)[m].double().sum())
        out['abs_ce_diff_sum'] += float((lr.gather(-1, tgt)[..., 0] - lp.gather(-1, tgt)[..., 0]).abs()[m]
                                        .double().sum())
        out['top1_disagree'] += int((lr.argmax(-1) != lp.argmax(-1))[m].sum())
        out['max_abs_logit_diff'] = max(out['max_abs_logit_diff'], float((ref - lm).abs()[m].max()))
    return out
