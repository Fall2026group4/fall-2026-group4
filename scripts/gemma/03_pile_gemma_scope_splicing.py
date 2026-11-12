#!/usr/bin/env python3
"""M6 step 2: Gemma Scope SAE reconstruction and splicing on the pre-registered Pile sample (GPU, inference only).

Pre-registration: results/gemma2_pile/gemma_scope_splicing/README.md + preregistration.json
(commits b39dcc2, fa6f2ce). Sample: sample_manifest.csv + sample_info.json from scripts/gemma/02 (commit 39ed943).

Per evaluation document (batch size 1, so there is never any padding in the real run):
  original pass (captures resid_post at every layer; gate V2: hook output == hidden_states[L + 1])
  original_repeat pass (gate V3, determinism)
  for each layer L in the pre-registration, one pass per condition, only layer L modified at non-BOS positions:
      sae_splice                 x -> SAE(x)
      norm_matched_random_error  x -> x + ||x - SAE(x)|| * u        (u random unit vector, seed 1000 + L)
      mean_ablation              x -> mean of layer-L activations over the 64 reference documents (non-BOS)
      identity                   x -> x                              (gate V3, hook plumbing)
Reconstruction sums (EV, cosine, relative error, L0) come from the original pass; behaviour sums (CE, KL, top-1) from
each spliced pass against the original logits. Aggregates are token-weighted, overall and per Pile source, with
document-bootstrap CIs; H-G1 per layer.

Gates (pre-registered): V1 sample integrity (+ re-tokenization hashes), V2 hook equivalence, V3 identity/repeat
mean KL <= threshold, V4 SAE sanity (binding in the smoke run), V5 pinned SAE ids only.
  * V1, V2, V5: stop immediately in both modes (gates_FAILED.json).
  * V3, V4: --smoke stops immediately; the full run computes the complete aggregate, then on failure writes
    gates_FAILED.json, marks every output table INVALID, keeps the outputs for diagnosis and exits with code 2.
  * sae-lens cross-check (--smoke only): a DIAGNOSTIC, not a gate. sae-lens may load its default revision, so a
    difference beyond rtol 1e-4 stops the smoke run for investigation (crosscheck_MISMATCH.json) with the revisions
    involved recorded; it is not by itself a failure of the pinned loader.

Both modes read the committed sample (sample_manifest.csv + sample_info.json, commit a34d975) and never regenerate it.

Engineering safeguards (no change to the scientific design; review 2026-11-12, items A1-A11):
  A1 every activation, reconstruction, logits tensor, per-document sum and output column must be finite, else
     numerics_FAILED.json and stop.  A2 git commit + dirty state of the code/inputs recorded; the full run refuses
     a dirty tree.  A3 deterministic algorithms (CUBLAS_WORKSPACE_CONFIG set before torch loads; warn-only, and any
     determinism warning is recorded).  A4 captured activations are copies.  A5 use_cache=False.  A6 the loaded
     model/tokenizer commit is checked against the pin.  A7 outputs are written to .partial/ and moved into place
     only when complete; run_info.json (with SHA-256 of every output) is moved last.  A8 --out-tag for a byte-level
     smoke rerun.  A9 GPU capability, bf16 support and logits dtype recorded; --smoke also measures the bf16 logit
     precision against a float32 recomputation of the same head (decision H4a: a documented limitation, not used
     in any result).  A10 INVALID banner; cache scan cannot crash the smoke run.  A11 run the suite with
     `python -m pytest -q -rs src/tests` and require zero skips.

Run from repository root (CUDA required):
    python scripts/gemma/03_pile_gemma_scope_splicing.py --smoke                 # first 8 eval + 4 reference docs -> smoke/
    python scripts/gemma/03_pile_gemma_scope_splicing.py --smoke --out-tag rerun # same, -> smoke_rerun/ (A8)
    python scripts/gemma/03_pile_gemma_scope_splicing.py                         # full pre-registered run
Refuses to overwrite existing outputs.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # A3: must be set before torch initialises CUDA/cuBLAS

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

EXP = REPO / 'results/gemma2_pile/gemma_scope_splicing'
PREREG = EXP / 'preregistration.json'
OUTPUTS = ['per_doc_metrics.csv', 'per_doc_sum_x.npz', 'summary_by_layer.csv', 'summary_by_layer_source.csv',
           'controls.csv', 'h_g1.csv', 'gates.json']
RUN_INFO = 'run_info.json'  # moved into place last (A7): its presence means the output set is complete
STOP_FILES = ['gates_FAILED.json', 'crosscheck_MISMATCH.json', 'numerics_FAILED.json', 'provenance_FAILED.json']
STAGING = '.partial'
# Code and inputs whose committed state the results depend on (A2)
TRACKED_INPUTS = ['scripts/gemma/03_pile_gemma_scope_splicing.py', 'src/evaluation/pile_splicing.py',
                  'results/gemma2_pile/gemma_scope_splicing/preregistration.json',
                  'results/gemma2_pile/gemma_scope_splicing/README.md',
                  'results/gemma2_pile/gemma_scope_splicing/sample_manifest.csv',
                  'results/gemma2_pile/gemma_scope_splicing/sample_info.json']
# Implementation constants that are NOT in the pre-registration (approved 2026-11-12; README section 2):
SAE_LENS_CROSSCHECK_RTOL = 1e-4  # smoke DIAGNOSTIC: max|x_hat_ours - x_hat_saelens| / max|x_hat_saelens|
V4_L0_FACTOR = 2.0               # pre-registered: measured L0 within a factor of 2 of the release's average_l0
SMOKE_N_EVAL, SMOKE_N_REF = 8, 4  # --smoke: first documents (by rank) of the committed sample; outputs are SMOKE


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).name


def write_stop(out_dir: Path, filename: str, payload: dict, msg: str):
    """Record everything known so far in ``filename`` and stop. Stop files are never results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps({'message': msg, **payload}, indent=2, default=str) + '\n',
                                    encoding='utf-8')
    raise SystemExit(f'{msg}  (details in {rel(out_dir / filename)})')


def fail(out_dir: Path, gates: dict, gate: str, msg: str):
    """A failed pre-registered gate stops the run."""
    gates[gate] = {**gates.get(gate, {}), 'pass': False, 'message': msg}
    write_stop(out_dir, 'gates_FAILED.json', {'gates': gates}, f'{gate} FAILED: {msg}')


def stop_for_investigation(out_dir: Path, gates: dict, msg: str):
    """Smoke-only diagnostic stop (not a gate failure): record everything and stop so the cause can be investigated."""
    write_stop(out_dir, 'crosscheck_MISMATCH.json', {'gates': gates}, f'DIAGNOSTIC STOP: {msg}')


def check_finite(out_dir: Path, gates: dict, name: str, *tensors):
    """A1: stop on any NaN/inf (numerics_FAILED.json)."""
    try:
        ps.assert_finite(name, *tensors)
    except ps.NumericsError as exc:
        write_stop(out_dir, 'numerics_FAILED.json', {'gates': gates}, f'NUMERICS FAILED: {exc}')


def check_finite_values(out_dir: Path, gates: dict, name: str, values: dict):
    try:
        ps.assert_finite_values(name, values)
    except ps.NumericsError as exc:
        write_stop(out_dir, 'numerics_FAILED.json', {'gates': gates}, f'NUMERICS FAILED: {exc}')


def release_l0(sae_id: str) -> float:
    m = re.search(r'average_l0_(\d+)$', sae_id)
    if not m:
        raise ValueError(f'cannot read average_l0 from {sae_id!r}')
    return float(m.group(1))


# ------------------------------------------------------------------------------------------------------ aggregation
def summarize(df: pd.DataFrame, sum_x: np.ndarray, sum_x_row: dict, ranks, layers, n_boot: int, seed: int) -> list:
    """One summary row per layer for the documents ``ranks`` (token-weighted; bootstrap over those documents).

    ``sum_x_row`` maps a document rank to its row in ``sum_x`` [n_docs, n_layers, d_model].
    """
    ranks = list(ranks)
    idx = ps.bootstrap_indices(len(ranks), n_boot, seed)
    rows = []
    for li, layer in enumerate(layers):
        def arr(cond, col):
            s = df[(df.layer == layer) & (df.condition == cond)].set_index('rank').loc[ranks, col]
            return s.to_numpy(dtype=float)

        n_tok, sse, sum_sq = arr('sae_splice', 'n_tok'), arr('sae_splice', 'sse'), arr('sae_splice', 'sum_sq')
        cos, l0 = arr('sae_splice', 'cos_sum'), arr('sae_splice', 'l0_sum')
        sx = sum_x[[sum_x_row[r] for r in ranks], li]
        n_pred, ce_o = arr('sae_splice', 'n_pred'), arr('sae_splice', 'ce_orig_sum')
        ce = {c: arr(c, 'ce_new_sum') for c in ('sae_splice', 'norm_matched_random_error', 'mean_ablation')}
        kl, top1 = arr('sae_splice', 'kl_sum'), arr('sae_splice', 'top1_agree')

        stats = {
            'explained_variance': lambda r: ps.ev_from_sums(sse[r], sum_sq[r], sx[r], n_tok[r]),
            'cosine': lambda r: cos[r].sum() / n_tok[r].sum(),
            'relative_sq_error': lambda r: sse[r].sum() / sum_sq[r].sum(),
            'l0': lambda r: l0[r].sum() / n_tok[r].sum(),
            'delta_ce_sae': lambda r: (ce['sae_splice'][r].sum() - ce_o[r].sum()) / n_pred[r].sum(),
            'delta_ce_random': lambda r: (ce['norm_matched_random_error'][r].sum() - ce_o[r].sum()) / n_pred[r].sum(),
            'delta_ce_mean_ablation': lambda r: (ce['mean_ablation'][r].sum() - ce_o[r].sum()) / n_pred[r].sum(),
            'kl_sae': lambda r: kl[r].sum() / n_pred[r].sum(),
            'top1_agreement_sae': lambda r: top1[r].sum() / n_pred[r].sum(),
            'loss_recovered': lambda r: ((ce['mean_ablation'][r].sum() - ce['sae_splice'][r].sum())
                                         / (ce['mean_ablation'][r].sum() - ce_o[r].sum())),
            'ratio_dce_sae_over_random': lambda r: ((ce['sae_splice'][r].sum() - ce_o[r].sum())
                                                    / (ce['norm_matched_random_error'][r].sum() - ce_o[r].sum())),
        }
        everything = np.arange(len(ranks))
        row = {'layer': layer, 'n_docs': len(ranks), 'n_tokens': int(n_tok.sum()), 'n_predictions': int(n_pred.sum()),
               'ce_original': ce_o.sum() / n_pred.sum()}
        for name, fn in stats.items():
            row[name] = float(fn(everything))
            row[f'{name}_ci95_lo'], row[f'{name}_ci95_hi'] = ps.bootstrap_ci(fn, idx, 0.95)
        row['nmse'] = 1 - row['explained_variance']
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true',
                    help='first 8 eval + 4 reference docs of the committed sample, write to smoke/, run the sae-lens check')
    ap.add_argument('--out-tag', default=None,
                    help='(--smoke only) write to smoke_<tag>/ instead of smoke/, e.g. for a byte-level rerun check (A8)')
    args = ap.parse_args()
    t_start = time.time()
    if args.out_tag is not None and (not args.smoke or not re.fullmatch(r'[A-Za-z0-9_-]+', args.out_tag)):
        raise SystemExit('--out-tag needs --smoke and may contain only letters, digits, _ and -')

    cfg = ps.load_preregistration_eval(PREREG)
    layers = [int(x) for x in cfg['layers']]
    out_dir = (EXP / ('smoke' if args.out_tag is None else f'smoke_{args.out_tag}')) if args.smoke else EXP
    clash = [rel(out_dir / f) for f in OUTPUTS + [RUN_INFO, STAGING] + STOP_FILES if (out_dir / f).exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite existing outputs or stop files: {clash}')

    # A2: code provenance. A full run must come from a clean, committed tree so code and commit correspond.
    provenance = ps.git_provenance(REPO, TRACKED_INPUTS)
    print(f"code provenance: commit {provenance.get('commit')}, dirty: {provenance.get('dirty_paths')}", flush=True)
    if not args.smoke and (not provenance['available'] or provenance['dirty_paths']):
        raise SystemExit(f'full run refused: code/inputs must be committed and unmodified ({provenance})')

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required (no CPU fallback for M6)')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # A3: deterministic algorithms. warn_only so an op without a deterministic kernel is recorded rather than
    # crashing the run; gate V3 then tests determinism empirically (repeat and identity passes).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    determinism_warnings = []
    _show = warnings.showwarning

    def _record(message, category, filename, lineno, file=None, line=None):
        if 'determinis' in str(message).lower():
            determinism_warnings.append(f'{category.__name__}: {message}')
        _show(message, category, filename, lineno, file, line)

    warnings.showwarning = _record
    gates = {}

    # ---------------------------------------------------------------- V1: sample integrity and re-tokenization
    # Always the committed, authoritative sample (commit a34d975); it is never regenerated here.
    manifest_path, info_path = EXP / 'sample_manifest.csv', EXP / 'sample_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    man = pd.read_csv(manifest_path)
    v1 = {'manifest_sha256_matches_sample_info':
          ps.sha256_file(manifest_path) == info['hashes'].get(rel(manifest_path)),
          'sample_info_gate_V1': all(info['gate_V1'][k] for k in ('size_match', 'sha256_match', 'disjoint')),
          'data_sha256_matches_prereg': info['data']['sha256'] == cfg['data']['sha256'],
          'tokenizer_revision_matches_prereg':
              info['tokenizer']['revision_preregistered'] == cfg['tokenization']['revision']}
    ev_docs = man[man.set == 'eval'].sort_values('rank')
    ref_docs = man[man.set == 'reference'].sort_values('rank')
    v1['counts'] = (len(ev_docs) == int(cfg['sampling']['n_eval'])
                    and len(ref_docs) == int(cfg['sampling']['n_reference']))
    v1['disjoint'] = not set(ev_docs.line_index) & set(ref_docs.line_index)
    gates['V1'] = v1
    if not all(v1.values()):
        fail(out_dir, gates, 'V1', f'sample integrity check failed: {v1}')
    if args.smoke:  # V1 above ran on the full sample; the smoke run then evaluates only its first documents
        ev_docs, ref_docs = ev_docs.head(SMOKE_N_EVAL), ref_docs.head(SMOKE_N_REF)
        man = man[man.line_index.isin(set(ev_docs.line_index) | set(ref_docs.line_index))]
        gates['V1']['smoke_subset'] = {'n_eval': SMOKE_N_EVAL, 'n_reference': SMOKE_N_REF}

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data = cfg['data']
    data_path = Path(hf_hub_download(repo_id=data['hf_dataset'], filename=data['file'], repo_type='dataset',
                                     revision=data['revision']))
    if ps.sha256_file(data_path) != data['sha256']:
        fail(out_dir, gates, 'V1', 'data file SHA-256 differs from the pre-registration')
    objs = ps.load_lines(data_path, man.line_index.tolist())
    tok_rev = cfg['tokenization']['revision']
    tok = AutoTokenizer.from_pretrained(cfg['tokenization']['tokenizer'], revision=tok_rev)
    loaded_commits = {}
    try:  # A6: the tokenizer actually loaded must come from the pinned commit
        loaded_commits['tokenizer'] = ps.check_pinned_commit(
            getattr(tok, 'init_kwargs', {}).get('_commit_hash'), tok_rev, 'tokenizer')
    except ValueError as exc:
        write_stop(out_dir, 'provenance_FAILED.json', {'gates': gates}, f'PROVENANCE FAILED: {exc}')
    max_tok = int(cfg['tokenization']['max_tokens_including_bos'])
    token_ids = {}
    for r in man.itertuples():
        text = objs[r.line_index]['text']
        ids, _ = ps.truncate_with_bos(tok.encode(text, add_special_tokens=False), tok.bos_token_id, max_tok)
        if ps.sha256_text(text) != r.text_sha256 or ps.sha256_token_ids(ids) != r.token_ids_sha256:
            fail(out_dir, gates, 'V1', f'text or token hash mismatch at line {r.line_index}')
        token_ids[r.line_index] = ids
    gates['V1']['retokenized_hashes_match'] = True
    gates['V1']['pass'] = True
    print(f'V1 pass: {len(ev_docs)} eval + {len(ref_docs)} reference documents, hashes verified', flush=True)

    # ---------------------------------------------------------------- model at the pinned revision
    model = AutoModelForCausalLM.from_pretrained(cfg['model']['hf_id'], revision=cfg['model']['revision'],
                                                 dtype=torch.bfloat16, attn_implementation='eager', device_map=device)
    model.eval()
    mcfg = model.config
    try:  # A6: the weights actually loaded must come from the pinned commit
        loaded_commits['model'] = ps.check_pinned_commit(getattr(mcfg, '_commit_hash', None),
                                                         cfg['model']['revision'], 'model')
    except ValueError as exc:
        write_stop(out_dir, 'provenance_FAILED.json', {'gates': gates}, f'PROVENANCE FAILED: {exc}')
    print(f'loaded commits: {loaded_commits}', flush=True)
    # A9: hardware and precision facts
    try:
        bf16_native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch without the argument
        bf16_native = 'unknown'
    hardware = {'gpu': torch.cuda.get_device_name(0),
                'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(0))),
                'bf16_native': bf16_native, 'bf16_supported_incl_emulation': torch.cuda.is_bf16_supported(),
                'model_param_dtype': str(next(model.parameters()).dtype)}
    print(f'hardware: {hardware}', flush=True)
    d_model, d_sae = int(cfg['sae']['d_in']), int(cfg['sae']['d_sae'])
    if mcfg.hidden_size != d_model or max(layers) >= mcfg.num_hidden_layers - 1:
        raise SystemExit(f'model/pre-registration mismatch: hidden {mcfg.hidden_size} vs d_in {d_model}, '
                         f'{mcfg.num_hidden_layers} layers vs requested {layers} (last layer excluded: final norm)')

    # ---------------------------------------------------------------- V5: pinned SAEs only
    saes, sae_files = {}, {}
    for layer in layers:
        sae_id = cfg['sae']['ids'][str(layer)]
        try:
            p = Path(hf_hub_download(repo_id=cfg['sae']['hf_repo'], filename=f'{sae_id}/params.npz',
                                     revision=cfg['sae']['hf_revision']))
        except Exception as exc:  # noqa: BLE001  any failure means the pinned SAE is unavailable
            fail(out_dir, gates, 'V5', f'pinned SAE {sae_id} could not be loaded ({type(exc).__name__}); '
                                       'no substitution allowed: stop and ask the user')
        with np.load(p) as z:
            params = {k: z[k] for k in z.files}
        saes[layer] = ps.JumpReLUSAE(params, d_model, d_sae, device=device)
        sae_files[sae_id] = ps.sha256_file(p)
    gates['V5'] = {'pass': True, 'params_npz_sha256': sae_files}

    hooks = {layer: ps.LayerSplice(model.model.layers[layer]) for layer in layers}

    def run(ids, hidden_states=False):
        with torch.inference_mode():
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=hidden_states,
                         use_cache=False)  # A5

    def as_input(line_index):
        return torch.tensor([token_ids[line_index]], device=device)

    # ---------------------------------------------------------------- smoke only: sae-lens cross-check (DIAGNOSTIC)
    # Not a pre-registered validity gate. sae-lens may load its default revision, so a mismatch is reported and
    # investigated; it is not by itself evidence that the pinned loader is wrong.
    if args.smoke:
        import sae_lens
        from huggingface_hub import model_info, scan_cache_dir
        from sae_lens import SAE
        ids = as_input(int(ev_docs.iloc[0].line_index))
        run(ids)
        check = {}
        for layer in layers:
            sae_id = cfg['sae']['ids'][str(layer)]
            x = hooks[layer].captured[0, 1:].float()
            loaded = SAE.from_pretrained(release=cfg['sae']['sae_lens_release'], sae_id=sae_id, device=device)
            ref_sae = loaded[0] if isinstance(loaded, tuple) else loaded
            with torch.inference_mode():
                f_ref = ref_sae.encode(x)
                x_ref = ref_sae.decode(f_ref).float()
                f_ours = saes[layer].encode(x)
                x_ours = saes[layer].decode(f_ours)
            max_abs = float((x_ours - x_ref).abs().max())
            check[str(layer)] = {'sae_id': sae_id, 'max_abs_diff': max_abs,
                                 'max_abs_diff_rel': max_abs / float(x_ref.abs().max()),
                                 'active_latent_mismatch_frac': float(((f_ours != 0) != (f_ref != 0)).float().mean()),
                                 'l0_ours': float((f_ours != 0).sum(-1).float().mean()),
                                 'l0_sae_lens': float((f_ref != 0).sum(-1).float().mean())}
            del ref_sae
        # which cached revisions of the SAE repo hold the files sae-lens used (ours is pinned; sae-lens may use main)
        wanted = {f"{cfg['sae']['ids'][str(layer)]}/params.npz" for layer in layers}
        cached = []
        try:  # A10: a cache-scan problem must not crash the run; it is recorded instead
            for repo in scan_cache_dir().repos:
                if repo.repo_id != cfg['sae']['hf_repo']:
                    continue
                for rev in repo.revisions:
                    files = {Path(f.file_path).relative_to(rev.snapshot_path).as_posix() for f in rev.files}
                    cached.append({'revision': rev.commit_hash, 'refs': sorted(rev.refs),
                                   'pinned_files_present': sorted(files & wanted)})
        except Exception as exc:  # noqa: BLE001
            cached = [{'error': f'cache scan failed: {type(exc).__name__}: {exc}'}]
        diag = {'kind': 'diagnostic (not a pre-registered gate)', 'rtol': SAE_LENS_CROSSCHECK_RTOL,
                'sae_lens_version': sae_lens.__version__, 'pinned_revision': cfg['sae']['hf_revision'],
                'repo_main_revision_now': model_info(cfg['sae']['hf_repo']).sha,
                'cached_revisions_with_these_files': cached, 'by_layer': check}
        diag['all_within_rtol'] = all(v['max_abs_diff_rel'] <= SAE_LENS_CROSSCHECK_RTOL for v in check.values())
        gates['diagnostic_sae_lens_crosscheck'] = diag
        print(f'sae-lens cross-check (diagnostic): {json.dumps(diag, default=str)}', flush=True)
        if not diag['all_within_rtol']:
            stop_for_investigation(out_dir, gates, 'sae-lens output differs from the pinned loader beyond rtol '
                                                   f'{SAE_LENS_CROSSCHECK_RTOL}: investigate (compare revisions above) '
                                                   'before continuing')

    # ---------------------------------------------------------------- reference means (mean ablation)
    sums = {layer: torch.zeros(d_model, dtype=torch.float64, device=device) for layer in layers}
    n_ref_tok = 0
    t_ref = time.time()
    for r in ref_docs.itertuples():
        run(as_input(r.line_index))
        for layer in layers:
            check_finite(out_dir, gates, f'reference activations, layer {layer}', hooks[layer].captured)
            sums[layer] += hooks[layer].captured[0, 1:].double().sum(0)
        n_ref_tok += len(token_ids[r.line_index]) - 1
    means = {layer: (sums[layer] / n_ref_tok).float() for layer in layers}
    check_finite(out_dir, gates, 'mean-ablation vectors', *means.values())
    sec_per_ref_doc = (time.time() - t_ref) / max(len(ref_docs), 1)

    # ---------------------------------------------------------------- evaluation loop
    gens = {layer: torch.Generator().manual_seed(ps.random_error_seed(cfg, layer)) for layer in layers}
    rows, v2_checks = [], 0
    sum_x = np.zeros((len(ev_docs), len(layers), d_model))
    sum_x_row = {int(r): i for i, r in enumerate(ev_docs['rank'])}
    # H4a (smoke only): measure the bf16 logit precision; the head's input is captured by a pre-hook on the head itself
    head_capture, precision = {}, []
    head_hook = None
    if args.smoke:
        head = model.get_output_embeddings()
        head_hook = head.register_forward_pre_hook(lambda m, a: head_capture.__setitem__('x', a[0].detach().clone()))
        softcap = getattr(mcfg, 'final_logit_softcapping', None)

    t_loop = time.time()
    logits_dtype = None
    for k, doc in enumerate(ev_docs.itertuples()):
        ids = as_input(doc.line_index)
        am = torch.ones_like(ids)
        smask, pmask = ps.splice_mask(am), ps.prediction_mask(am)
        base = {'rank': int(doc.rank), 'line_index': int(doc.line_index), 'pile_set_name': doc.pile_set_name}
        for h in hooks.values():
            h.set(None)

        out = run(ids, hidden_states=True)
        for layer in layers:  # V2: the tensor given to the SAE is the resid_post of layer L
            if not torch.equal(hooks[layer].captured, out.hidden_states[layer + 1]):
                fail(out_dir, gates, 'V2', f'hook output != hidden_states[{layer + 1}] (document line {doc.line_index})')
            v2_checks += 1
        logits_o = out.logits
        logits_dtype = str(logits_o.dtype)
        check_finite(out_dir, gates, f'original logits (line {doc.line_index})', logits_o)
        check_finite(out_dir, gates, f'original activations (line {doc.line_index})',
                     *[hooks[layer].captured for layer in layers])
        x_orig = {layer: hooks[layer].captured[smask].float() for layer in layers}
        del out
        if args.smoke:
            with torch.inference_mode():
                precision.append(ps.logit_precision_sums(logits_o, head_capture['x'], head.weight, softcap, ids, pmask))
            head_capture.clear()
        rows.append({**base, 'layer': -1, 'condition': 'original', **ps.behaviour_sums(logits_o, logits_o, ids, pmask)})
        logits_r = run(ids).logits
        check_finite(out_dir, gates, f'repeat logits (line {doc.line_index})', logits_r)
        rows.append({**base, 'layer': -1, 'condition': 'original_repeat',
                     **ps.behaviour_sums(logits_o, logits_r, ids, pmask)})
        del logits_r
        check_finite_values(out_dir, gates, f'behaviour sums (line {doc.line_index})', rows[-1])

        for li, layer in enumerate(layers):
            sae = saes[layer]
            x_hat, l0 = sae(x_orig[layer])
            check_finite(out_dir, gates, f'SAE reconstruction, layer {layer} (line {doc.line_index})', x_hat)
            recon = ps.reconstruction_sums(x_orig[layer], x_hat, l0)
            sum_x[k, li] = recon.pop('sum_x')
            check_finite_values(out_dir, gates, f'reconstruction sums, layer {layer}', recon)
            if not np.isfinite(sum_x[k, li]).all():
                write_stop(out_dir, 'numerics_FAILED.json', {'gates': gates},
                           f'NUMERICS FAILED: sum_x non-finite, layer {layer} (line {doc.line_index})')
            conditions = {
                'sae_splice': lambda h, s=sae: s(h.float())[0],
                'norm_matched_random_error':
                    lambda h, s=sae, g=gens[layer]: ps.norm_matched_random_error(h.float(), s(h.float())[0], g),
                'mean_ablation': lambda h, m=means[layer]: m.expand(h.shape[0], -1),
                'identity': lambda h: h,
            }
            for cond, fn in conditions.items():
                hooks[layer].set(fn, smask)
                logits_n = run(ids).logits
                hooks[layer].set(None)
                check_finite(out_dir, gates, f'{cond} logits, layer {layer} (line {doc.line_index})', logits_n)
                row = {**base, 'layer': layer, 'condition': cond, **ps.behaviour_sums(logits_o, logits_n, ids, pmask)}
                check_finite_values(out_dir, gates, f'{cond} behaviour sums, layer {layer}', row)
                if cond == 'sae_splice':
                    row.update(recon)
                rows.append(row)
                del logits_n
        del logits_o
        if k == 0 or (k + 1) % 16 == 0 or k + 1 == len(ev_docs):
            per_doc = (time.time() - t_loop) / (k + 1)
            print(f'{k + 1}/{len(ev_docs)} documents, {per_doc:.1f} s/document', flush=True)
    gates['V2'] = {'pass': True, 'n_checks': v2_checks}
    sec_per_eval_doc = (time.time() - t_loop) / max(len(ev_docs), 1)
    if head_hook is not None:
        head_hook.remove()
    logit_precision = None
    if precision:  # H4a: a documented limitation, reported only; nothing here enters a result
        tot = {k: sum(p[k] for p in precision) for k in ('n_pred', 'kl_ref_vs_model_sum', 'abs_ce_diff_sum',
                                                         'top1_disagree')}
        logit_precision = {
            'what': 'model bf16 logits vs float32 recomputation of the same head from its own input (smoke documents)',
            'logits_dtype': logits_dtype, 'final_logit_softcapping': softcap, 'n_predictions': tot['n_pred'],
            'mean_kl_fp32_vs_bf16': tot['kl_ref_vs_model_sum'] / tot['n_pred'],
            'mean_abs_ce_diff': tot['abs_ce_diff_sum'] / tot['n_pred'],
            'top1_disagreement_rate': tot['top1_disagree'] / tot['n_pred'],
            'max_abs_logit_diff': max(p['max_abs_logit_diff'] for p in precision)}
        print(f'logit precision (H4a limitation, diagnostic): {logit_precision}', flush=True)
    # projection for the pre-registered sample (512 eval + 64 reference), from this run's per-document timings
    projected_h = (sec_per_eval_doc * int(cfg['sampling']['n_eval'])
                   + sec_per_ref_doc * int(cfg['sampling']['n_reference'])) / 3600
    print(f'timing: {sec_per_eval_doc:.1f} s/eval doc, {sec_per_ref_doc:.1f} s/reference doc, peak GPU memory '
          f'{torch.cuda.max_memory_allocated() / 1e9:.2f} GB; projected full run (512 + 64 docs): {projected_h:.2f} h',
          flush=True)

    # ---------------------------------------------------------------- aggregation
    df = pd.DataFrame(rows)
    n_boot, seed = int(cfg['bootstrap']['n_resamples']), int(cfg['seeds']['bootstrap'])
    ranks_all = ev_docs['rank'].astype(int).tolist()
    by_layer = pd.DataFrame(summarize(df, sum_x, sum_x_row, ranks_all, layers, n_boot, seed))
    by_layer.insert(1, 'release_average_l0', [release_l0(cfg['sae']['ids'][str(x)]) for x in layers])
    src_rows = []
    low_n = int(cfg['low_n_source_threshold_docs'])
    for source, grp in ev_docs.groupby('pile_set_name'):
        for row in summarize(df, sum_x, sum_x_row, grp['rank'].astype(int).tolist(), layers, n_boot, seed):
            src_rows.append({'pile_set_name': source, 'low_n': row['n_docs'] < low_n, **row})
    by_source = pd.DataFrame(src_rows)

    thr = float(cfg['validity_gates']['V3_determinism_identity_max_mean_kl_nats'])
    ctrl, h_rows = [], []
    rep = df[df.condition == 'original_repeat']
    rep_kl = rep.kl_sum.sum() / rep.n_pred.sum()
    for layer in layers:
        ident = df[(df.layer == layer) & (df.condition == 'identity')]
        ident_kl = ident.kl_sum.sum() / ident.n_pred.sum()
        ctrl.append({'layer': layer, 'identity_mean_kl': ident_kl,
                     'identity_max_doc_mean_kl': float((ident.kl_sum / ident.n_pred).max()),
                     'repeat_mean_kl': rep_kl, 'repeat_max_doc_mean_kl': float((rep.kl_sum / rep.n_pred).max()),
                     'v3_threshold': thr, 'v3_pass': bool(ident_kl <= thr and rep_kl <= thr)})

        def doc_mean_dce(cond):
            s = df[(df.layer == layer) & (df.condition == cond)].set_index('rank').loc[ranks_all]
            return ((s.ce_new_sum - s.ce_orig_sum) / s.n_pred).to_numpy()

        h_rows.append({'layer': layer, **ps.h_g1(doc_mean_dce('sae_splice'), doc_mean_dce('norm_matched_random_error'),
                                                 ps.bootstrap_indices(len(ranks_all), n_boot, seed))})
    controls, hg1 = pd.DataFrame(ctrl), pd.DataFrame(h_rows)
    # A1: every numeric output must be finite (checked before any gate is evaluated or anything is written)
    nonfinite = {
        'per_doc_metrics (behaviour)': ps.nonfinite_columns(df, ['n_pred', 'ce_orig_sum', 'ce_new_sum', 'kl_sum',
                                                                 'top1_agree']),
        'per_doc_metrics (reconstruction)': ps.nonfinite_columns(df[df.condition == 'sae_splice'],
                                                                 ['n_tok', 'sse', 'sum_sq', 'cos_sum', 'l0_sum']),
        'summary_by_layer': ps.nonfinite_columns(by_layer), 'summary_by_layer_source': ps.nonfinite_columns(by_source),
        'controls': ps.nonfinite_columns(controls), 'h_g1': ps.nonfinite_columns(hg1),
        'per_doc_sum_x': [] if np.isfinite(sum_x).all() else ['sum_x']}
    nonfinite = {k: v for k, v in nonfinite.items() if v}
    if nonfinite:
        write_stop(out_dir, 'numerics_FAILED.json', {'gates': gates, 'nonfinite_columns': nonfinite},
                   f'NUMERICS FAILED: non-finite values in output tables {nonfinite}')
    gates['V3'] = {'pass': bool(controls.v3_pass.all()), 'threshold': thr,
                   'by_layer': controls[['layer', 'identity_mean_kl', 'repeat_mean_kl']].to_dict('records')}
    v4 = {str(r.layer): {'l0': r.l0, 'release_average_l0': r.release_average_l0, 'ev': r.explained_variance,
                         'pass': bool(r.release_average_l0 / V4_L0_FACTOR <= r.l0 <= r.release_average_l0 * V4_L0_FACTOR
                                      and r.explained_variance > 0)} for r in by_layer.itertuples()}
    gates['V4'] = {'pass': all(v['pass'] for v in v4.values()), 'factor': V4_L0_FACTOR, 'by_layer': v4,
                   'binding': 'smoke run' if args.smoke else 'reported (binding check was the smoke run)'}
    if args.smoke:
        for gate in ('V3', 'V4'):
            if not gates[gate]['pass']:
                fail(out_dir, gates, gate, f'{gate} failed in the smoke run: {gates[gate]}')

    # ---------------------------------------------------------------- outputs
    # Only the pre-registered gates V1-V5 decide validity; the sae-lens cross-check is a diagnostic.
    all_pass = all(gates[g].get('pass', False) for g in ('V1', 'V2', 'V3', 'V4', 'V5'))
    if not all_pass:  # full run only (smoke stopped above): keep outputs for diagnosis, mark them INVALID
        for table in (df, by_layer, by_source, controls, hg1):
            table.insert(0, 'status', 'INVALID: validity gate failed (see gates_FAILED.json); do not interpret')
    # A7: write everything into .partial/ first; files are moved into place only when the set is complete
    staging = out_dir / STAGING
    staging.mkdir(parents=True, exist_ok=False)
    df.to_csv(staging / 'per_doc_metrics.csv', index=False)
    np.savez_compressed(staging / 'per_doc_sum_x.npz', sum_x=sum_x, ranks=np.asarray(ranks_all), layers=np.asarray(layers))
    by_layer.to_csv(staging / 'summary_by_layer.csv', index=False)
    by_source.to_csv(staging / 'summary_by_layer_source.csv', index=False)
    controls.to_csv(staging / 'controls.csv', index=False)
    hg1.to_csv(staging / 'h_g1.csv', index=False)
    (staging / 'gates.json').write_text(json.dumps(gates, indent=2, default=str) + '\n', encoding='utf-8')
    output_sha256 = {rel(out_dir / f): ps.sha256_file(staging / f) for f in OUTPUTS}

    import sae_lens
    import transformers
    run_info = {
        'experiment': cfg.get('experiment'),
        'status': ('SMOKE (pipeline check, not a result)' if args.smoke else 'full pre-registered run')
                  + ('' if all_pass else ' -- INVALID: a validity gate failed (see gates.json)'),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_eval_docs': len(ev_docs), 'n_reference_docs': len(ref_docs), 'layers': layers,
        'model': {'hf_id': cfg['model']['hf_id'], 'revision': cfg['model']['revision'], 'dtype': 'bfloat16',
                  'attn_implementation': 'eager', 'loaded_commit_check': loaded_commits, 'logits_dtype': logits_dtype},
        'sae': {'hf_repo': cfg['sae']['hf_repo'], 'hf_revision': cfg['sae']['hf_revision'],
                'params_npz_sha256': sae_files},
        'code_provenance': provenance,
        'implementation_constants': {'sae_lens_crosscheck_rtol': SAE_LENS_CROSSCHECK_RTOL, 'v4_l0_factor': V4_L0_FACTOR,
                                     'batch_size': 1, 'tf32': False, 'use_cache': False,
                                     'deterministic_algorithms': 'True (warn_only)',
                                     'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'determinism_warnings': sorted(set(determinism_warnings)),
        'logit_precision_limitation_H4a': logit_precision,
        'hardware': hardware,
        'input_sha256': {'preregistration.json': ps.sha256_file(PREREG), 'README.md': ps.sha256_file(EXP / 'README.md'),
                         rel(Path(__file__)): ps.sha256_file(Path(__file__)),
                         'src/evaluation/pile_splicing.py': ps.sha256_file(REPO / 'src/evaluation/pile_splicing.py'),
                         rel(manifest_path): ps.sha256_file(manifest_path), rel(info_path): ps.sha256_file(info_path)},
        'output_sha256': output_sha256,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'transformers': transformers.__version__, 'sae_lens': sae_lens.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__},
        'runtime_s': round(time.time() - t_start, 1),
        'seconds_per_eval_document': round(sec_per_eval_doc, 2),
        'seconds_per_reference_document': round(sec_per_ref_doc, 2),
        'projected_full_run_hours': round(projected_h, 2),
        'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (staging / RUN_INFO).write_text(json.dumps(run_info, indent=2, default=str) + '\n', encoding='utf-8')
    for f in OUTPUTS + [RUN_INFO]:  # run_info.json last: its presence marks a complete output set
        os.replace(staging / f, out_dir / f)
    staging.rmdir()
    if not all_pass:
        (out_dir / 'gates_FAILED.json').write_text(json.dumps({'message': 'validity gate failed at the end of the full '
                                                                          'run; outputs kept for diagnosis only',
                                                               'gates': gates}, indent=2, default=str) + '\n',
                                                   encoding='utf-8')
    print(f"wrote {len(OUTPUTS) + 1} outputs to {rel(out_dir)}; status: {run_info['status']}")
    if not all_pass:  # A10
        print('\n' + '!' * 100 + '\nINVALID RUN: a pre-registered validity gate failed. The numbers below are for '
              'diagnosis only and must NOT be interpreted.\n' + '!' * 100)
    print(by_layer[['layer', 'explained_variance', 'l0', 'delta_ce_sae', 'delta_ce_random', 'kl_sae',
                    'loss_recovered']].to_string(index=False))
    if not all_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
