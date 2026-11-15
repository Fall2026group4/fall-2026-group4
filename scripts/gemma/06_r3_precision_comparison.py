#!/usr/bin/env python3
"""R3 Q0b: how far is T4 bf16 from fp32? (CPU only; reads committed per-document CSVs, never a GPU.)

Pre-registration: results/gemma2_pile/r3_gemma_scope_ladder/README.md, Amendment 1 (2), and preregistration.json
(questions.Q0b, margins). Q0 (T4 vs A10G) was dropped by Amendment 1 (1).

Cell: layer 12, SAE layer_12/width_16k/average_l0_82, the first 64 evaluation documents by rank.
  fp32    : fp32_reference/per_doc_metrics.csv + per_doc_sum_x.npz        (script 05, run K0)
  primary : M6's committed T4 bf16 rows (gemma_scope_splicing/per_doc_metrics.csv + per_doc_sum_x.npz)
  also    : the Kaggle T4 ladder rows for the same SAE (ladder/per_sae/layer_12_width_16k_l0_82/), if present

Per document i and metric m: e_i = |m_bf16,i - m_fp32,i| and b_i = m_bf16,i - m_fp32,i. Reported: mean e and mean b
over the 64 documents, each with a 95 % percentile bootstrap CI (1,000 document resamples, seed 0).
Verdict per metric on mean e against the fixed Q0 margins: WITHIN MARGIN if the CI upper bound < margin,
BEYOND MARGIN if the lower bound > margin, otherwise INCONCLUSIVE. Primary metric: delta_ce_sae (the headline).

Per-document metrics (each from that document's own rows):
  ce_original        ce_orig_sum / n_pred                       (unmodified forward pass)
  delta_ce_sae       (ce_new_sum - ce_orig_sum) / n_pred        sae_splice
  delta_ce_random    the same for norm_matched_random_error
  kl_sae             kl_sum / n_pred                            sae_splice
  top1_agreement     top1_agree / n_pred                        sae_splice
  explained_variance 1 - sse / (sum_sq - ||sum_x||^2 / n_tok)   the document's own EV (its own token mean)

Run from repository root (after K0, and after K3 if the ladder rows should be included):
    python scripts/gemma/06_r3_precision_comparison.py
    python scripts/gemma/06_r3_precision_comparison.py --r3-root /path/to/r3_out
Refuses to overwrite existing outputs.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

EXP = REPO / 'results/gemma2_pile/r3_gemma_scope_ladder'
PREREG = EXP / 'preregistration.json'
M6 = REPO / 'results/gemma2_pile/gemma_scope_splicing'
METRICS = ['delta_ce_sae', 'ce_original', 'delta_ce_random', 'kl_sae', 'top1_agreement', 'explained_variance']
OUT_FILES = ['q0b_t4_vs_fp32.csv', 'q0b_info.json']


# ================================================================================================ pure functions
def q0b_verdict(lo: float, hi: float, margin: float) -> str:
    """On the CI of mean |bf16 - fp32|: WITHIN MARGIN if hi < margin, BEYOND MARGIN if lo > margin."""
    if hi < margin:
        return 'WITHIN MARGIN'
    if lo > margin:
        return 'BEYOND MARGIN'
    return 'INCONCLUSIVE'


def q0b_metric(bf16: Sequence[float], fp32: Sequence[float], idx: np.ndarray) -> dict:
    """Mean absolute error and mean signed difference (bf16 - fp32) over documents, each with a bootstrap CI."""
    b = np.asarray(bf16, float) - np.asarray(fp32, float)
    e = np.abs(b)
    e_lo, e_hi = ps.bootstrap_ci(lambda r: float(e[r].mean()), idx, 0.95)
    b_lo, b_hi = ps.bootstrap_ci(lambda r: float(b[r].mean()), idx, 0.95)
    return {'n_docs': int(e.size), 'mean_abs_error': float(e.mean()), 'abs_error_ci95_lo': e_lo,
            'abs_error_ci95_hi': e_hi, 'mean_signed_diff': float(b.mean()), 'signed_ci95_lo': b_lo,
            'signed_ci95_hi': b_hi}


def per_doc_metrics(rows: pd.DataFrame, sum_x: Dict[int, np.ndarray], ranks: Sequence[int]) -> pd.DataFrame:
    """One row per document (in ``ranks`` order) with the six Q0b metrics, from one run's per-document rows.

    ``rows`` holds that run's sae_splice and norm_matched_random_error rows for the cell; ``sum_x`` maps rank to the
    document's sum of evaluated activations (needed for its own token mean in EV)."""
    sp = rows[rows.condition == 'sae_splice'].set_index('rank').loc[list(ranks)]
    rn = rows[rows.condition == 'norm_matched_random_error'].set_index('rank').loc[list(ranks)]
    if not (sp.n_pred.to_numpy() == rn.n_pred.to_numpy()).all():
        raise ValueError('sae_splice and random rows disagree on n_pred')
    n_pred = sp.n_pred.to_numpy(float)
    sx = np.stack([np.asarray(sum_x[int(r)], float) for r in ranks])
    total = sp.sum_sq.to_numpy(float) - (sx * sx).sum(1) / sp.n_tok.to_numpy(float)
    return pd.DataFrame({
        'rank': list(ranks),
        'n_pred': sp.n_pred.to_numpy(int), 'n_tok': sp.n_tok.to_numpy(int),
        'ce_original': sp.ce_orig_sum.to_numpy(float) / n_pred,
        'delta_ce_sae': (sp.ce_new_sum - sp.ce_orig_sum).to_numpy(float) / n_pred,
        'delta_ce_random': (rn.ce_new_sum - rn.ce_orig_sum).to_numpy(float) / n_pred,
        'kl_sae': sp.kl_sum.to_numpy(float) / n_pred,
        'top1_agreement': sp.top1_agree.to_numpy(float) / n_pred,
        'explained_variance': 1 - sp.sse.to_numpy(float) / total,
    })


def compare_runs(bf16: pd.DataFrame, fp32: pd.DataFrame, margins: dict, idx: np.ndarray, label: str,
                 primary: str) -> list:
    """Q0b rows for one bf16 run against fp32 (same documents, same token counts required)."""
    if list(bf16['rank']) != list(fp32['rank']):
        raise ValueError(f'{label}: documents differ from the fp32 reference')
    for col in ('n_pred', 'n_tok'):
        if not (bf16[col].to_numpy() == fp32[col].to_numpy()).all():
            raise ValueError(f'{label}: {col} differs from the fp32 reference (different tokens?)')
    out = []
    for m in METRICS:
        res = q0b_metric(bf16[m], fp32[m], idx)
        out.append({'bf16_run': label, 'metric': m, 'role': 'primary' if m == primary else 'secondary',
                    'margin': float(margins[m]), **res,
                    'verdict': q0b_verdict(res['abs_error_ci95_lo'], res['abs_error_ci95_hi'], float(margins[m])),
                    'mean_bf16': float(bf16[m].mean()), 'mean_fp32': float(fp32[m].mean())})
    return out


# ================================================================================================= file plumbing
def load_sum_x(npz_path: Path, layer: Optional[int] = None) -> Dict[int, np.ndarray]:
    with np.load(npz_path) as z:
        sx = z['sum_x']
        if sx.ndim == 3:  # M6 layout [doc, layer, d]
            sx = sx[:, [int(x) for x in z['layers']].index(int(layer))]
        return {int(r): sx[i] for i, r in enumerate(z['ranks'])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--r3-root', default=str(EXP), help='folder holding fp32_reference/ and ladder/ (script 05 --out)')
    ap.add_argument('--out', default=None, help='output folder (default: <r3-root>/precision_t4_vs_fp32)')
    args = ap.parse_args(argv)
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    q = cfg['questions']['Q0b']
    ref = cfg['fp32_reference']
    layer, sid, n_docs = int(ref['layer']), ref['sae_id'], int(ref['n_docs'])
    root = Path(args.r3_root)
    out_dir = Path(args.out) if args.out else root / 'precision_t4_vs_fp32'
    clash = [f for f in OUT_FILES if (out_dir / f).exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite {clash} in {out_dir}')

    fp_dir = root / 'fp32_reference'
    fp_rows = pd.read_csv(fp_dir / 'per_doc_metrics.csv')
    fp_rows = fp_rows[(fp_rows.layer == layer) & (fp_rows.sae_id == sid)]
    ranks = sorted(int(r) for r in fp_rows['rank'].unique())
    m6_all = pd.read_csv(M6 / 'per_doc_metrics.csv')
    m6_first = sorted(int(r) for r in m6_all[m6_all.condition == 'sae_splice']['rank'].unique())[:n_docs]
    if ranks != m6_first:
        raise SystemExit(f'fp32 documents are not the first {n_docs} evaluation documents by rank')
    fp32 = per_doc_metrics(fp_rows, load_sum_x(fp_dir / 'per_doc_sum_x.npz'), ranks)

    idx = ps.bootstrap_indices(len(ranks), int(cfg['bootstrap']['n_resamples']), int(cfg['seeds']['bootstrap']))
    margins, primary = cfg['margins'], q['primary_metric']
    m6_rows = m6_all[m6_all.layer == layer]
    m6 = per_doc_metrics(m6_rows, load_sum_x(M6 / 'per_doc_sum_x.npz', layer), ranks)
    results = compare_runs(m6, fp32, margins, idx, 'M6 T4 bf16 (primary)', primary)

    ladder_dir = root / 'ladder' / 'per_sae' / 'layer_12_width_16k_l0_82'
    inputs = [M6 / 'per_doc_metrics.csv', M6 / 'per_doc_sum_x.npz', fp_dir / 'per_doc_metrics.csv',
              fp_dir / 'per_doc_sum_x.npz']
    kaggle_status = 'not available (ladder run K3 not present)'
    if (ladder_dir / 'DONE.json').exists():
        k_rows = pd.read_csv(ladder_dir / 'per_doc_metrics.csv')
        kag = per_doc_metrics(k_rows, load_sum_x(ladder_dir / 'per_doc_sum_x.npz'), ranks)
        results += compare_runs(kag, fp32, margins, idx, 'Kaggle T4 bf16 (also reported)', primary)
        inputs += [ladder_dir / 'per_doc_metrics.csv', ladder_dir / 'per_doc_sum_x.npz']
        kaggle_status = 'included'

    table = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'q0b_t4_vs_fp32.csv', index=False, lineterminator='\n')
    head = table[(table.bf16_run.str.contains('primary')) & (table.metric == primary)].iloc[0]

    def rel(p):
        try:
            return Path(p).resolve().relative_to(REPO).as_posix()
        except ValueError:
            return Path(p).name

    info = {'question': 'Q0b: how far is T4 bf16 from fp32 (Amendment 1 (2))',
            'cell': {'layer': layer, 'sae_id': sid, 'n_docs': len(ranks)},
            'headline': {'metric': primary, 'verdict': head.verdict, 'mean_abs_error': head.mean_abs_error,
                         'ci95': [head.abs_error_ci95_lo, head.abs_error_ci95_hi], 'margin': head.margin,
                         'mean_signed_diff': head.mean_signed_diff},
            'kaggle_ladder_rows': kaggle_status,
            'limits': 'one layer, one SAE, 64 documents; fp32 is a reference with smaller rounding, not exact truth',
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'input_sha256': {rel(p): ps.sha256_file(p) for p in inputs + [PREREG, Path(__file__)]},
            'output_sha256': {'q0b_t4_vs_fp32.csv': ps.sha256_file(out_dir / 'q0b_t4_vs_fp32.csv')}}
    (out_dir / 'q0b_info.json').write_text(json.dumps(info, indent=2, default=str) + '\n', encoding='utf-8',
                                           newline='\n')
    print(table[['bf16_run', 'metric', 'mean_abs_error', 'abs_error_ci95_lo', 'abs_error_ci95_hi', 'margin',
                 'verdict']].to_string(index=False))
    print(f"headline ({primary}, M6 T4 bf16 vs fp32): {head.verdict}")


if __name__ == '__main__':
    main()
