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

Run from repository root (CUDA required):
    python scripts/gemma/03_pile_gemma_scope_splicing.py --smoke   # first 8 eval + 4 reference docs -> smoke/ (SMOKE)
    python scripts/gemma/03_pile_gemma_scope_splicing.py           # full pre-registered run
Refuses to overwrite existing outputs.
"""
import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

EXP = REPO / 'results/gemma2_pile/gemma_scope_splicing'
PREREG = EXP / 'preregistration.json'
OUTPUTS = ['per_doc_metrics.csv', 'per_doc_sum_x.npz', 'summary_by_layer.csv', 'summary_by_layer_source.csv',
           'controls.csv', 'h_g1.csv', 'gates.json', 'run_info.json']
# Implementation constants that are NOT in the pre-registration (approved 2026-11-12; README section 2):
SAE_LENS_CROSSCHECK_RTOL = 1e-4  # smoke DIAGNOSTIC: max|x_hat_ours - x_hat_saelens| / max|x_hat_saelens|
V4_L0_FACTOR = 2.0               # pre-registered: measured L0 within a factor of 2 of the release's average_l0
SMOKE_N_EVAL, SMOKE_N_REF = 8, 4  # --smoke: first documents (by rank) of the committed sample; outputs are SMOKE


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).name


def fail(out_dir: Path, gates: dict, gate: str, msg: str):
    """Write what is known so far to gates_FAILED.json and stop (pre-registered: a failed gate stops the run)."""
    gates[gate] = {**gates.get(gate, {}), 'pass': False, 'message': msg}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'gates_FAILED.json').write_text(json.dumps(gates, indent=2, default=str) + '\n', encoding='utf-8')
    raise SystemExit(f'{gate} FAILED: {msg}  (details in {rel(out_dir / "gates_FAILED.json")})')


def stop_for_investigation(out_dir: Path, gates: dict, msg: str):
    """Smoke-only diagnostic stop (not a gate failure): record everything and stop so the cause can be investigated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'crosscheck_MISMATCH.json').write_text(json.dumps(gates, indent=2, default=str) + '\n',
                                                      encoding='utf-8')
    raise SystemExit(f'DIAGNOSTIC STOP: {msg}  (details in {rel(out_dir / "crosscheck_MISMATCH.json")})')


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
    args = ap.parse_args()
    t_start = time.time()

    cfg = ps.load_preregistration_eval(PREREG)
    layers = [int(x) for x in cfg['layers']]
    out_dir = EXP / 'smoke' if args.smoke else EXP
    clash = [rel(out_dir / f) for f in OUTPUTS + ['gates_FAILED.json', 'crosscheck_MISMATCH.json']
             if (out_dir / f).exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite existing outputs: {clash}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required (no CPU fallback for M6)')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=hidden_states)

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
        for repo in scan_cache_dir().repos:
            if repo.repo_id != cfg['sae']['hf_repo']:
                continue
            for rev in repo.revisions:
                files = {Path(f.file_path).relative_to(rev.snapshot_path).as_posix() for f in rev.files}
                cached.append({'revision': rev.commit_hash, 'refs': sorted(rev.refs),
                               'pinned_files_present': sorted(files & wanted)})
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
            sums[layer] += hooks[layer].captured[0, 1:].double().sum(0)
        n_ref_tok += len(token_ids[r.line_index]) - 1
    means = {layer: (sums[layer] / n_ref_tok).float() for layer in layers}
    sec_per_ref_doc = (time.time() - t_ref) / max(len(ref_docs), 1)

    # ---------------------------------------------------------------- evaluation loop
    gens = {layer: torch.Generator().manual_seed(ps.random_error_seed(cfg, layer)) for layer in layers}
    rows, v2_checks = [], 0
    sum_x = np.zeros((len(ev_docs), len(layers), d_model))
    sum_x_row = {int(r): i for i, r in enumerate(ev_docs['rank'])}
    t_loop = time.time()
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
        x_orig = {layer: hooks[layer].captured[smask].float() for layer in layers}
        del out
        rows.append({**base, 'layer': -1, 'condition': 'original', **ps.behaviour_sums(logits_o, logits_o, ids, pmask)})
        rows.append({**base, 'layer': -1, 'condition': 'original_repeat',
                     **ps.behaviour_sums(logits_o, run(ids).logits, ids, pmask)})

        for li, layer in enumerate(layers):
            sae = saes[layer]
            x_hat, l0 = sae(x_orig[layer])
            recon = ps.reconstruction_sums(x_orig[layer], x_hat, l0)
            sum_x[k, li] = recon.pop('sum_x')
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
                row = {**base, 'layer': layer, 'condition': cond, **ps.behaviour_sums(logits_o, logits_n, ids, pmask)}
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
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'per_doc_metrics.csv', index=False)
    np.savez_compressed(out_dir / 'per_doc_sum_x.npz', sum_x=sum_x, ranks=np.asarray(ranks_all), layers=np.asarray(layers))
    by_layer.to_csv(out_dir / 'summary_by_layer.csv', index=False)
    by_source.to_csv(out_dir / 'summary_by_layer_source.csv', index=False)
    controls.to_csv(out_dir / 'controls.csv', index=False)
    hg1.to_csv(out_dir / 'h_g1.csv', index=False)
    (out_dir / 'gates.json').write_text(json.dumps(gates, indent=2, default=str) + '\n', encoding='utf-8')
    if not all_pass:
        (out_dir / 'gates_FAILED.json').write_text(json.dumps(gates, indent=2, default=str) + '\n', encoding='utf-8')

    import sae_lens
    import transformers
    run_info = {
        'experiment': cfg.get('experiment'),
        'status': ('SMOKE (pipeline check, not a result)' if args.smoke else 'full pre-registered run')
                  + ('' if all_pass else ' -- INVALID: a validity gate failed (see gates.json)'),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_eval_docs': len(ev_docs), 'n_reference_docs': len(ref_docs), 'layers': layers,
        'model': {'hf_id': cfg['model']['hf_id'], 'revision': cfg['model']['revision'], 'dtype': 'bfloat16',
                  'attn_implementation': 'eager'},
        'sae': {'hf_repo': cfg['sae']['hf_repo'], 'hf_revision': cfg['sae']['hf_revision'],
                'params_npz_sha256': sae_files},
        'implementation_constants': {'sae_lens_crosscheck_rtol': SAE_LENS_CROSSCHECK_RTOL, 'v4_l0_factor': V4_L0_FACTOR,
                                     'batch_size': 1, 'tf32': False},
        'hashes': {'preregistration.json': ps.sha256_file(PREREG), 'README.md': ps.sha256_file(EXP / 'README.md'),
                   rel(Path(__file__)): ps.sha256_file(Path(__file__)),
                   'src/evaluation/pile_splicing.py': ps.sha256_file(REPO / 'src/evaluation/pile_splicing.py'),
                   rel(manifest_path): ps.sha256_file(manifest_path), rel(info_path): ps.sha256_file(info_path),
                   rel(out_dir / 'per_doc_sum_x.npz'): ps.sha256_file(out_dir / 'per_doc_sum_x.npz'),
                   rel(out_dir / 'per_doc_metrics.csv'): ps.sha256_file(out_dir / 'per_doc_metrics.csv')},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'transformers': transformers.__version__, 'sae_lens': sae_lens.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'gpu': torch.cuda.get_device_name(0)},
        'runtime_s': round(time.time() - t_start, 1),
        'seconds_per_eval_document': round(sec_per_eval_doc, 2),
        'seconds_per_reference_document': round(sec_per_ref_doc, 2),
        'projected_full_run_hours': round(projected_h, 2),
        'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (out_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2) + '\n', encoding='utf-8')
    print(f"wrote {len(OUTPUTS)} outputs to {rel(out_dir)}; status: {run_info['status']}")
    print(by_layer[['layer', 'explained_variance', 'l0', 'delta_ce_sae', 'delta_ce_random', 'kl_sae',
                    'loss_recovered']].to_string(index=False))
    if not all_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
