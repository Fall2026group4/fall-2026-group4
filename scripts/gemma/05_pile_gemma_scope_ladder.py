#!/usr/bin/env python3
"""R3: Gemma Scope width x L0 ladder on The Pile, and the fp32 reference (GPU, inference only; Kaggle T4 x2).

Pre-registration: results/gemma2_pile/r3_gemma_scope_ladder/README.md (Section 1, Amendment 1, Clarifications 1-2)
and preregistration.json. Sample: M6's committed sample_manifest.csv / sample_info.json (never regenerated).

Runs (Amendment 1 (4), Clarification 2); one Kaggle session each, outputs under --out (default: the results folder):
  K0  V0 hardware/access, V1 sample, SMOKE ladder (8 eval + 4 reference documents; measures seconds per pass and
      memory; not a result). K1-K6 project their run time from this record.
  K0F the fp32 reference on its own (Part A3: layer 12, M6's 16k SAE, first 64 eval documents; Gemma split across
      both T4s), with CUDA_LAUNCH_BLOCKING=1 for this step only (Clarification 2). Can be retried at any time.
  K1  layer 5 Tier A        K2  layer 19 Tier A        K3  layer 12 Tier A
  K4  layer 12 width 131k   K5  layer 12 width 262k    K6  layer 12 width 524k (SAE on GPU 1)
  --aggregate (CPU): pooled tables, L0 bands, H1-H3, V6 against M6's committed rows, gates.json, run_info.json.
Every GPU step writes runs/<run id>/run_info.json even when it stops or crashes (status 'finished', 'stopped: ...' or
'crashed: ...'); only a 'finished' K0 record is used for the K1-K6 projection.

Per layer, shared controls (original, original_repeat, identity, mean_ablation) are written once to
ladder/per_layer/layer_{L}/. Per SAE, ladder/per_sae/layer_{L}_width_{W}_l0_{N}/ holds per_doc_metrics.csv,
per_doc_sum_x.npz, sae_info.json (deterministic content), run_record.json (run id, timings, peak memory) and, last,
DONE.json with the SHA-256 of the other four. A folder without a valid DONE.json is unfinished: it is moved (never
deleted) to ladder/incomplete/<run id>/ and run again. Finished SAEs are skipped, never rewritten.

Gates: V0 hardware, V1 sample, V2 hook equivalence, V3 identity/repeat KL and per-SAE original-logit identity,
V4 per-SAE sanity (exclude and report), V5 pinned IDs only, V6 consistency with M6 (aggregate step), V7 placement and
memory, V8 fp32 dtypes, V9 resume equals fresh (relative difference <= 1e-6; bit-identity recorded separately),
V10 secret scan (run by the Kaggle notebook through secret_scan()). A failed stopping gate writes
STOPPED_<run id>.json and exits.

Run from repository root:
    python scripts/gemma/05_pile_gemma_scope_ladder.py --run K0 --out /kaggle/working/r3_out
    python scripts/gemma/05_pile_gemma_scope_ladder.py --run K0F --out /kaggle/working/r3_out
    python scripts/gemma/05_pile_gemma_scope_ladder.py --run K1 --out /kaggle/working/r3_out \\
        --resume-from /kaggle/input/<previous run output>/r3_out --week-used-hours 1.2
    python scripts/gemma/05_pile_gemma_scope_ladder.py --aggregate --out results/gemma2_pile/r3_gemma_scope_ladder
The Hugging Face token is read by huggingface_hub from the HF_TOKEN environment variable of this process only
(set in memory by the notebook from Kaggle Secrets); this script never reads, prints or stores it.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # must be set before torch initialises cuBLAS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation import pile_splicing as ps  # noqa: E402

EXP = REPO / 'results/gemma2_pile/r3_gemma_scope_ladder'
PREREG = EXP / 'preregistration.json'
M6 = REPO / 'results/gemma2_pile/gemma_scope_splicing'

WIDTHS = {'16k': 16384, '32k': 32768, '65k': 65536, '131k': 131072, '262k': 262144, '524k': 524288,
          '1m': 1048576}
D_MODEL = 2304
GB = 1e9
BASE_GB = 7.7           # model + working memory on the T4, from M6 (8.58 GB peak minus three 16k SAEs)
GPU_LIMIT_GB = 14.0     # planning limit per 16 GB T4 (Amendment 1 (3))
CHUNK = 128
RUN_IDS = ['K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
STEP_IDS = ['K0', 'K0F'] + RUN_IDS[1:]   # K0F: the fp32 reference as its own Kaggle step (Clarification 2)
BUDGETED_RUNS = RUN_IDS[1:]               # the ladder runs that need --week-used-hours
SMOKE_N_EVAL, SMOKE_N_REF = 8, 4
TRACKED_INPUTS = ['scripts/gemma/05_pile_gemma_scope_ladder.py', 'src/evaluation/pile_splicing.py',
                  'results/gemma2_pile/r3_gemma_scope_ladder/preregistration.json',
                  'results/gemma2_pile/r3_gemma_scope_ladder/README.md',
                  'results/gemma2_pile/gemma_scope_splicing/sample_manifest.csv',
                  'results/gemma2_pile/gemma_scope_splicing/sample_info.json']
RESULT_COLUMNS = ['rank', 'line_index', 'pile_set_name', 'sae_id', 'layer', 'condition', 'n_pred', 'ce_orig_sum',
                  'ce_new_sum', 'kl_sum', 'top1_agree', 'n_tok', 'sse', 'sum_sq', 'cos_sum', 'l0_sum']
CONTROL_COLUMNS = ['rank', 'line_index', 'pile_set_name', 'layer', 'condition', 'n_pred', 'ce_orig_sum',
                   'ce_new_sum', 'kl_sum', 'top1_agree', 'logits_sha256']


# =================================================================================================== SAE identities
def parse_sae_id(sae_id: str) -> dict:
    """'layer_12/width_524k/average_l0_227' -> layer, width, d_sae, release L0. Unknown widths are an error."""
    m = re.fullmatch(r'layer_(\d+)/width_(\w+)/average_l0_(\d+)', sae_id)
    if not m or m.group(2) not in WIDTHS:
        raise ValueError(f'not a Gemma Scope residual SAE id with a known width: {sae_id!r}')
    return {'layer': int(m.group(1)), 'width': m.group(2), 'd_sae': WIDTHS[m.group(2)],
            'release_l0': float(m.group(3))}


def sae_folder_name(sae_id: str) -> str:
    m = re.fullmatch(r'layer_(\d+)/width_(\w+)/average_l0_(\d+)', sae_id)
    if not m:
        raise ValueError(f'bad SAE id {sae_id!r}')
    return f'layer_{m.group(1)}_width_{m.group(2)}_l0_{m.group(3)}'


def ladder_from_prereg(cfg: dict) -> List[dict]:
    """The 54 pinned SAEs in pre-registered order (Tier A and B; width 1M is listed as excluded, not here)."""
    out = []
    for cell in cfg['sae']['ladder']:
        for n in cell['average_l0']:
            sid = f"layer_{cell['layer']}/width_{cell['width']}/average_l0_{n}"
            out.append({'sae_id': sid, 'layer': int(cell['layer']), 'width': cell['width'], 'tier': cell['tier']})
    return out


# ============================================================================================ memory and run planning
def sae_gb(d_sae: int) -> float:
    """fp32 SAE weights + three 128-token encode buffers, in GB."""
    weights = (2 * D_MODEL * d_sae + 2 * d_sae + D_MODEL) * 4 / GB
    buffers = 3 * CHUNK * d_sae * 4 / GB
    return weights + buffers


def estimate_peak_gb(d_sae: int) -> float:
    """Amendment 1 (3): peak on GPU 0 when the model and the SAE share it."""
    return BASE_GB + sae_gb(d_sae)


def sae_placement(d_sae: int, limit_gb: float = GPU_LIMIT_GB) -> str:
    """'gpu0' if model + SAE fit one T4; 'gpu1' if the SAE alone fits the second T4; else 'excluded'."""
    if estimate_peak_gb(d_sae) <= limit_gb:
        return 'gpu0'
    if sae_gb(d_sae) <= limit_gb:
        return 'gpu1'
    return 'excluded'


def plan_runs(ladder: Sequence[dict]) -> Dict[str, List[str]]:
    """Amendment 1 (4): which SAEs each Kaggle run evaluates (pre-registered order inside a run)."""
    rules = {'K1': lambda r: r['tier'] == 'A' and r['layer'] == 5,
             'K2': lambda r: r['tier'] == 'A' and r['layer'] == 19,
             'K3': lambda r: r['tier'] == 'A' and r['layer'] == 12,
             'K4': lambda r: r['tier'] == 'B' and r['width'] == '131k',
             'K5': lambda r: r['tier'] == 'B' and r['width'] == '262k',
             'K6': lambda r: r['tier'] == 'B' and r['width'] == '524k'}
    plan = {'K0': []}
    for k, rule in rules.items():
        plan[k] = [r['sae_id'] for r in ladder if rule(r)]
    assigned = [s for v in plan.values() for s in v]
    missing = {r['sae_id'] for r in ladder} - set(assigned)
    if missing or len(assigned) != len(set(assigned)):
        raise ValueError(f'run plan does not cover every SAE exactly once (missing {sorted(missing)})')
    return plan


def project_run_hours(sae_ids: Sequence[str], sec_per_pass: float, *, n_docs: int = 512, n_ref: int = 64,
                      new_layers: int = 1, mean_tokens: float = 496.0, sae_tflops: float = 4.0,
                      download_mb_s: float = 50.0, v9_docs: int = 32) -> float:
    """Planning projection (hours) from the K0-measured seconds per forward pass."""
    sec = 0.0
    for sid in sae_ids:
        d = parse_sae_id(sid)['d_sae']
        sec += 3 * n_docs * sec_per_pass
        sec += 3 * n_docs * 4 * mean_tokens * D_MODEL * d / (sae_tflops * 1e12)
        sec += sae_gb(d) * 1e3 / download_mb_s
    sec += new_layers * (3 * n_docs + n_ref) * sec_per_pass          # shared per-layer controls
    sec += (3 * v9_docs + n_ref) * sec_per_pass                      # V9 recomputation if resumed
    return sec / 3600


def may_start(projected_h: float, week_used_h: float, max_run_h: float = 10.0, max_week_h: float = 30.0) -> dict:
    reasons = []
    if projected_h > max_run_h:
        reasons.append(f'projected {projected_h:.2f} h exceeds the {max_run_h} h per-run cap: split the run by width')
    if week_used_h + projected_h > max_week_h:
        reasons.append(f'{week_used_h:.2f} h used + {projected_h:.2f} h projected exceeds {max_week_h} h this week: '
                       'wait for the quota reset')
    return {'ok': not reasons, 'reasons': reasons}


# ================================================================================================ SAE arithmetic
def chunked_encode(sae, x, chunk: int = CHUNK):
    return torch.cat([sae.encode(x[s:s + chunk]) for s in range(0, x.shape[0], chunk)], 0)


def chunked_sae(sae, x, chunk: int = CHUNK):
    """(x_hat, L0 per token) with encode/decode done on ``chunk`` tokens at a time (Design: SAE forward)."""
    outs, l0s = [], []
    for s in range(0, x.shape[0], chunk):
        f = sae.encode(x[s:s + chunk])
        outs.append(sae.decode(f))
        l0s.append((f != 0).sum(-1))
    return torch.cat(outs, 0), torch.cat(l0s, 0)


def active_latent_agreement(f_a, f_b) -> float:
    """Fraction of (token, latent) active/inactive flags that agree (Clarification 1)."""
    return float(((f_a != 0) == (f_b != 0)).double().mean())


def random_generator(layer: int):
    """Generator for the norm-matched random error: seed 1000 + L, fresh at the start of every SAE's pass."""
    return torch.Generator().manual_seed(1000 + int(layer))


def tensor_sha256(t) -> str:
    """SHA-256 of a tensor's raw bytes (any dtype); used for the bit-for-bit original-logit check."""
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


# ================================================================================================ comparison rules
def relative_match(new, stored, rtol: float) -> dict:
    """Gate V9 rule: |new - stored| <= rtol * max(|new|, |stored|) element-wise (two exact zeros match)."""
    a, b = np.asarray(new, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if a.shape != b.shape:
        return {'pass': False, 'bit_identical': False, 'max_rel_diff': float('inf')}
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), 0.0)
    rel = np.where(both_nan, 0.0, rel)
    ok = both_nan | (a == b) | (diff <= rtol * scale)
    canon = lambda v: np.where(np.isnan(v), np.nan, v).tobytes()  # noqa: E731  one NaN bit pattern (empty CSV cells)
    return {'pass': bool(ok.all()), 'bit_identical': canon(a) == canon(b),
            'max_rel_diff': float(np.nan_to_num(rel, nan=np.inf).max()) if rel.size else 0.0}


def compare_vectors(new, stored, rtol: float) -> dict:
    """Gate V9 rule for the mean-ablation vector: ||new - stored|| / ||stored|| <= rtol."""
    a, b = np.asarray(new, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if a.shape != b.shape:
        return {'pass': False, 'bit_identical': False, 'rel_diff': float('inf')}
    nb = float(np.linalg.norm(b))
    d = float(np.linalg.norm(a - b))
    rel = d / nb if nb > 0 else (0.0 if d == 0 else float('inf'))
    return {'pass': bool(rel <= rtol), 'bit_identical': bool(a.tobytes() == b.tobytes()), 'rel_diff': float(rel)}


def compare_tables(new_df: pd.DataFrame, stored_df: pd.DataFrame, keys: Sequence[str], rtol: float) -> dict:
    """Same rows (by ``keys``) and columns; numbers within ``rtol`` (relative_match); text exactly equal."""
    problems = []
    if sorted(new_df.columns) != sorted(stored_df.columns):
        problems.append(f'columns differ: {sorted(set(new_df.columns) ^ set(stored_df.columns))}')
    k_new = set(map(tuple, new_df[list(keys)].astype(str).to_numpy()))
    k_old = set(map(tuple, stored_df[list(keys)].astype(str).to_numpy()))
    if k_new != k_old or len(new_df) != len(stored_df):
        problems.append(f'rows differ ({len(new_df)} new vs {len(stored_df)} stored)')
    if problems:
        return {'pass': False, 'bit_identical': False, 'max_rel_diff': float('inf'), 'problems': problems}
    a = new_df.sort_values(list(keys)).reset_index(drop=True)
    b = stored_df.sort_values(list(keys)).reset_index(drop=True)[a.columns]
    passed, bit, worst = True, True, 0.0
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            r = relative_match(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), rtol)
            passed &= r['pass']
            bit &= r['bit_identical']
            worst = max(worst, r['max_rel_diff'])
            if not r['pass']:
                problems.append(f'{col}: max relative difference {r["max_rel_diff"]:.3g}')
        elif not (a[col].astype(str).to_numpy() == b[col].astype(str).to_numpy()).all():
            passed, bit = False, False
            problems.append(f'{col}: text differs')
    return {'pass': bool(passed), 'bit_identical': bool(bit and passed), 'max_rel_diff': float(worst),
            'problems': problems}


# ============================================================================================ deterministic files
def save_npz_deterministic(path: Path, **arrays) -> None:
    """Like numpy.savez_compressed, but byte-identical across runs (fixed zip timestamps, sorted members)."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{name}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def write_json(path: Path, obj) -> None:
    # newline='\n': identical bytes on Windows and Linux (no CRLF translation)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8', newline='\n')


def write_done(folder: Path) -> None:
    """Write DONE.json (SHA-256 of every other file in ``folder``) last. Refuses to replace an existing marker."""
    folder = Path(folder)
    done = folder / 'DONE.json'
    if done.exists():
        raise FileExistsError(f'refusing to overwrite {done}')
    files = {p.name: ps.sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'DONE.json'}
    tmp = folder / 'DONE.json.tmp'
    write_json(tmp, {'files': files})
    os.replace(tmp, done)


def is_done(folder: Path) -> bool:
    """True if DONE.json exists and lists exactly the folder's other files with matching SHA-256."""
    folder = Path(folder)
    done = folder / 'DONE.json'
    if not done.is_file():
        return False
    try:
        listed = json.loads(done.read_text(encoding='utf-8'))['files']
    except (ValueError, KeyError):
        return False
    present = {p.name for p in folder.iterdir() if p.is_file() and p.name != 'DONE.json'}
    if set(listed) != present:
        return False
    return all(ps.sha256_file(folder / name) == sha for name, sha in listed.items())


def stop(ladder_dir: Path, run_id: str, msg: str, payload: dict):
    """A failed stopping gate: record it and exit. Stop files are never results."""
    Path(ladder_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(ladder_dir) / f'STOPPED_{run_id}.json', {'message': msg, **payload})
    raise SystemExit(msg)


# =========================================================================================== statistics helpers
def l0_bands(table: pd.DataFrame, tol: float = 0.10) -> List[dict]:
    """H2 bands: anchor = each 16k SAE (target = its measured L0); members = same layer, other widths, measured L0
    within ``tol`` of the target. Per width the member closest to the target is used (ties: lower L0); any other
    SAE of that width inside the band is listed under 'also_in_band'. Bands without members are not returned."""
    bands = []
    for layer, grp in table.groupby('layer', sort=True):
        anchors = grp[grp.width == '16k'].sort_values('measured_l0')
        others = grp[grp.width != '16k']
        for a in anchors.itertuples():
            t = float(a.measured_l0)
            inside = others[(others.measured_l0 - t).abs() <= tol * t].copy()
            if inside.empty:
                continue
            inside['dist'] = (inside.measured_l0 - t).abs()
            inside = inside.sort_values(['d_sae', 'dist', 'measured_l0'])
            chosen = inside.groupby('d_sae', sort=True).head(1)
            extra = inside.drop(chosen.index)
            bands.append({'layer': int(layer), 'anchor_sae_id': a.sae_id, 'target_l0': t,
                          'members': [a.sae_id] + chosen.sae_id.tolist(),
                          'also_in_band': extra.sae_id.tolist()})
    return bands


def iut_verdict(cis: Sequence) -> str:
    """Intersection-union rule (H1, H2): SUPPORTED if every lower bound > 0; CONTRADICTED if any upper bound < 0."""
    cis = list(cis)
    if not cis:
        raise ValueError('no pairs')
    if all(lo > 0 for lo, _ in cis):
        return 'SUPPORTED'
    if any(hi < 0 for _, hi in cis):
        return 'CONTRADICTED'
    return 'INCONCLUSIVE'


def pairwise_iut(ordered_ids: Sequence[str], doc_vals: Dict[str, np.ndarray], idx: np.ndarray) -> dict:
    """Adjacent pairs of ``ordered_ids``; d = value(lower in the order) - value(higher in the order) per document.
    H1 orders by measured L0 ascending, H2 by width ascending, so d > 0 is the predicted direction."""
    pairs = []
    for lo_id, hi_id in zip(ordered_ids[:-1], ordered_ids[1:]):
        d = np.asarray(doc_vals[lo_id], float) - np.asarray(doc_vals[hi_id], float)
        ci = ps.bootstrap_ci(lambda r: float(d[r].mean()), idx, 0.95)
        pairs.append({'lower': lo_id, 'higher': hi_id, 'mean_d': float(d.mean()), 'ci95_lo': ci[0], 'ci95_hi': ci[1]})
    return {'pairs': pairs, 'verdict': iut_verdict([(p['ci95_lo'], p['ci95_hi']) for p in pairs])}


def holm(pvals: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm step-down; returns rejection flags in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind='stable')
    reject = [False] * len(p)
    for i, j in enumerate(order):
        if p[j] <= alpha / (len(p) - i):
            reject[j] = True
        else:
            break
    return reject


def one_sided_p(mean: float, se: float) -> float:
    """One-sided p-value for H0: d <= 0, normal approximation z = mean / bootstrap SE."""
    if se <= 0:
        return 0.0 if mean > 0 else 1.0
    return 0.5 * math.erfc((mean / se) / math.sqrt(2))


def h3_verdict(significant: Sequence[bool]) -> str:
    sig = list(significant)
    n, k = len(sig), sum(bool(s) for s in sig)
    if n and k == n:
        return 'HOLDS ACROSS THE LADDER'
    if 3 * k >= 2 * n:
        return 'MOSTLY'
    return 'NOT GENERAL'


def equivalence_verdict(lo: float, hi: float, margin: float) -> str:
    """Q0-style rule: EQUIVALENT if the CI lies inside +-margin; DIFFERENT if wholly outside; else INCONCLUSIVE."""
    if lo >= -margin and hi <= margin:
        return 'EQUIVALENT'
    if lo > margin or hi < -margin:
        return 'DIFFERENT'
    return 'INCONCLUSIVE'


def v6_verdict(exact: bool, within_margin: bool) -> str:
    if exact:
        return 'M6 REPRODUCED EXACTLY'
    if within_margin:
        return 'M6 REPRODUCED WITHIN MARGIN'
    return 'FAILED'


# ================================================================================================= V0, fp32, V10
def v0_hardware_check(gpus: Sequence[dict], n_gpus: int = 2, name: str = 'T4', cc: str = '7.5') -> dict:
    problems = []
    if len(gpus) != n_gpus:
        problems.append(f'{len(gpus)} GPUs visible, expected {n_gpus}')
    for i, g in enumerate(gpus):
        if name not in str(g.get('name', '')):
            problems.append(f'GPU {i} is {g.get("name")!r}, expected a {name}')
        if str(g.get('compute_capability')) != cc:
            problems.append(f'GPU {i} compute capability {g.get("compute_capability")}, expected {cc}')
    return {'pass': not problems, 'problems': problems, 'gpus': list(gpus)}


def fp32_device_map(n_layers: int) -> Dict[str, int]:
    """Fixed split for the fp32 reference: embedding, output head (tied weight) and the first half of the layers on
    GPU 0; the second half and the final norm on GPU 1. For Gemma-2-2B (26 layers): 0-12 on GPU 0, 13-25 on GPU 1."""
    split = (n_layers + 1) // 2
    dmap = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'lm_head': 0, 'model.norm': 1}
    for i in range(n_layers):
        dmap[f'model.layers.{i}'] = 0 if i < split else 1
    return dmap


HF_PATTERN = re.compile(rb'hf_[A-Za-z0-9]{30,}')


def secret_scan(root: Path, token: Optional[str]) -> List[dict]:
    """Gate V10: files under ``root`` holding the exact token or anything shaped like a Hugging Face token.
    Returns file paths and the kind of hit only; the matched text is never returned or printed."""
    root = Path(root)
    needle = token.encode('utf-8') if token else None
    hits = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        data = p.read_bytes()
        rel_name = p.relative_to(root).as_posix()
        if needle and needle in data:
            hits.append({'file': rel_name, 'kind': 'exact_token'})
        if HF_PATTERN.search(data):
            hits.append({'file': rel_name, 'kind': 'hf_pattern'})
    return hits


# ===================================================================================================== the engine
@dataclass
class LadderContext:
    """Everything the ladder engine needs; real runs pass the Gemma model, tests pass a tiny one."""
    model: object
    eval_docs: List[dict]                      # {'rank', 'line_index', 'pile_set_name', 'ids'} in rank order
    ref_docs: List[dict]
    load_sae: Callable[[str], tuple]           # sae_id -> (params dict, params.npz SHA-256)
    device: str = 'cuda:0'
    sae_device: object = None                  # str, or callable(sae_id) -> str; default: same as device
    chunk: int = CHUNK
    v9_docs: int = 32
    order: List[str] = field(default_factory=list)   # pre-registered order of all ladder SAEs (V9 picks the first)
    v3_max_kl: float = 1e-6
    v4_factor: float = 2.0
    v9_rtol: float = 1e-6
    check_unchunked_ids: Sequence[str] = ()    # V6 / Clarification 1: active-latent agreement for these SAEs
    get_layer: Optional[Callable] = None
    n_passes: int = 0
    pass_seconds: float = 0.0

    def layer_module(self, layer: int):
        return self.get_layer(layer) if self.get_layer else self.model.model.layers[layer]

    def sae_device_for(self, sae_id: str) -> str:
        if callable(self.sae_device):
            return self.sae_device(sae_id)
        return self.sae_device or self.device


def _forward(ctx: LadderContext, ids, hidden_states: bool = False):
    t = time.time()
    with torch.inference_mode():
        out = ctx.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=hidden_states,
                        use_cache=False)
    ctx.n_passes += 1
    ctx.pass_seconds += time.time() - t
    return out


def _as_input(ctx: LadderContext, doc: dict):
    return torch.tensor([doc['ids']], device=ctx.device)


def _finite(name: str, *tensors):
    ps.assert_finite(name, *tensors)  # raises NumericsError: the run stops


def compute_layer_controls(ctx: LadderContext, layer: int, with_mean_ablation: bool = True):
    """Shared per-layer passes: original (V2, logit hash), original_repeat, identity and, unless disabled,
    mean_ablation. Returns (controls DataFrame, mean vector or None, number of reference tokens, gate dict)."""
    hook = ps.LayerSplice(ctx.layer_module(layer))
    try:
        mean, n_ref_tok = None, 0
        if with_mean_ablation:
            total = torch.zeros(ctx.model.config.hidden_size, dtype=torch.float64, device=ctx.device)
            for doc in ctx.ref_docs:
                _forward(ctx, _as_input(ctx, doc))
                _finite(f'reference activations, layer {layer}', hook.captured)
                total += hook.captured[0, 1:].double().sum(0).to(ctx.device)
                n_ref_tok += len(doc['ids']) - 1
            mean = (total / n_ref_tok).float()
            _finite('mean-ablation vector', mean)
        rows, v2 = [], 0
        for doc in ctx.eval_docs:
            ids = _as_input(ctx, doc)
            am = torch.ones_like(ids)
            smask, pmask = ps.splice_mask(am), ps.prediction_mask(am)
            base = {'rank': int(doc['rank']), 'line_index': int(doc['line_index']),
                    'pile_set_name': doc['pile_set_name'], 'layer': layer}
            hook.set(None)
            out = _forward(ctx, ids, hidden_states=True)
            if not torch.equal(hook.captured, out.hidden_states[layer + 1]):
                return None, None, None, {'V2': {'pass': False, 'rank': int(doc['rank'])}}
            v2 += 1
            logits_o = out.logits
            del out
            _finite(f'original logits (rank {doc["rank"]})', logits_o)
            rows.append({**base, 'condition': 'original', **ps.behaviour_sums(logits_o, logits_o, ids, pmask),
                         'logits_sha256': tensor_sha256(logits_o)})
            logits_r = _forward(ctx, ids).logits
            rows.append({**base, 'condition': 'original_repeat', **ps.behaviour_sums(logits_o, logits_r, ids, pmask)})
            conds = {'identity': lambda h: h}
            if with_mean_ablation:
                conds['mean_ablation'] = lambda h, m=mean: m.to(h.device).expand(h.shape[0], -1)
            for cond, fn in conds.items():
                hook.set(fn, smask)
                logits_n = _forward(ctx, ids).logits
                hook.set(None)
                _finite(f'{cond} logits (rank {doc["rank"]})', logits_n)
                rows.append({**base, 'condition': cond, **ps.behaviour_sums(logits_o, logits_n, ids, pmask)})
    finally:
        hook.remove()
    df = pd.DataFrame(rows).reindex(columns=CONTROL_COLUMNS)
    gates = {'V2': {'pass': True, 'n_checks': v2}}
    v3 = {}
    for cond in ('identity', 'original_repeat'):
        sub = df[df.condition == cond]
        v3[f'{cond}_mean_kl'] = float(sub.kl_sum.sum() / sub.n_pred.sum())
    v3['threshold'] = ctx.v3_max_kl
    v3['pass'] = all(v <= ctx.v3_max_kl for k, v in v3.items() if k.endswith('_mean_kl'))
    gates['V3'] = v3
    return df, (None if mean is None else mean.cpu().numpy()), n_ref_tok, gates


def evaluate_sae(ctx: LadderContext, layer: int, sae_id: str, docs: Sequence[dict], orig_hashes: Dict[int, str]):
    """Original (recomputed; must hash-match the layer's first pass), SAE splice and norm-matched random error for
    one SAE on ``docs``. Returns (rows DataFrame, sum_x [n_docs, d], deterministic info, volatile record)."""
    t0 = time.time()
    params, params_sha = ctx.load_sae(sae_id)
    d_model = ctx.model.config.hidden_size
    d_sae = int(np.asarray(params['W_enc']).shape[1])
    sdev = ctx.sae_device_for(sae_id)
    sae = ps.JumpReLUSAE(params, d_model, d_sae, device=sdev)
    del params
    gen = random_generator(layer)                # re-seeded for every SAE (Seeds; Amendment 1 (5))
    hook = ps.LayerSplice(ctx.layer_module(layer))
    rows, sum_x, logits_match = [], np.zeros((len(docs), d_model)), True
    agree_num = agree_den = 0.0
    check_unchunked = sae_id in set(ctx.check_unchunked_ids)
    try:
        for k, doc in enumerate(docs):
            ids = _as_input(ctx, doc)
            am = torch.ones_like(ids)
            smask, pmask = ps.splice_mask(am), ps.prediction_mask(am)
            hook.set(None)
            logits_o = _forward(ctx, ids).logits
            if tensor_sha256(logits_o) != orig_hashes.get(int(doc['rank'])):
                logits_match = False
            x = hook.captured[smask].float()
            _finite(f'activations, layer {layer} (rank {doc["rank"]})', x)
            with torch.inference_mode():
                xs = x.to(sdev)
                x_hat_s, l0 = chunked_sae(sae, xs, ctx.chunk)
                if check_unchunked:
                    n = xs.shape[0] * d_sae
                    agree_num += active_latent_agreement(chunked_encode(sae, xs, ctx.chunk), sae.encode(xs)) * n
                    agree_den += n
                x_hat = x_hat_s.to(x.device)
                pert = ps.norm_matched_random_error(x, x_hat, gen)
            _finite(f'SAE reconstruction {sae_id} (rank {doc["rank"]})', x_hat)
            recon = ps.reconstruction_sums(x, x_hat, l0)
            sum_x[k] = recon.pop('sum_x')
            base = {'rank': int(doc['rank']), 'line_index': int(doc['line_index']),
                    'pile_set_name': doc['pile_set_name'], 'sae_id': sae_id, 'layer': layer}
            for cond, value in (('sae_splice', x_hat), ('norm_matched_random_error', pert)):
                hook.set(lambda h, v=value: v, smask)
                logits_n = _forward(ctx, ids).logits
                hook.set(None)
                _finite(f'{cond} logits {sae_id} (rank {doc["rank"]})', logits_n)
                row = {**base, 'condition': cond, **ps.behaviour_sums(logits_o, logits_n, ids, pmask)}
                if cond == 'sae_splice':
                    row.update(recon)
                ps.assert_finite_values(f'{cond} sums', row)
                rows.append(row)
                del logits_n
            del logits_o
    finally:
        hook.remove()
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    sp = df[df.condition == 'sae_splice']
    measured_l0 = float(sp.l0_sum.sum() / sp.n_tok.sum())
    ev = ps.ev_from_sums(sp.sse.to_numpy(), sp.sum_sq.to_numpy(), sum_x, sp.n_tok.to_numpy())
    rel = parse_sae_id(sae_id)['release_l0']
    info = {'sae_id': sae_id, 'layer': layer, 'd_sae': d_sae, 'params_sha256': params_sha, 'sae_device': str(sdev),
            'n_docs': len(docs), 'measured_l0': measured_l0, 'explained_variance': ev,
            'V3_original_logits_match': bool(logits_match),
            'V4': {'pass': bool(rel / ctx.v4_factor <= measured_l0 <= rel * ctx.v4_factor and ev > 0),
                   'release_l0': rel, 'factor': ctx.v4_factor},
            'active_latent_agreement_chunked_vs_unchunked': (agree_num / agree_den) if agree_den else None}
    record = {'seconds': round(time.time() - t0, 2), 'peak_memory_gb': peak_memory_by_gpu()}
    return df, sum_x, info, record


def peak_memory_by_gpu() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {f'cuda:{i}': round(torch.cuda.max_memory_allocated(i) / GB, 3) for i in range(torch.cuda.device_count())}


def _csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(df.to_csv(index=False, lineterminator='\n')))


def _move_incomplete(parent: Path, ladder_dir: Path, run_id: str) -> List[str]:
    moved = []
    if not parent.is_dir():
        return moved
    for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
        if not is_done(folder):
            dest = ladder_dir / 'incomplete' / run_id / folder.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(f'refusing to overwrite {dest}')
            shutil.move(str(folder), str(dest))
            moved.append(folder.name)
    return moved


def _read_controls(layer_dir: Path):
    df = pd.read_csv(layer_dir / 'controls_per_doc.csv')
    with np.load(layer_dir / 'mean_ablation.npz') as z:
        mean = z['mean'] if 'mean' in z.files else None
    orig = df[df.condition == 'original']
    return df, mean, {int(r): h for r, h in zip(orig['rank'], orig['logits_sha256'])}


def run_v9(ctx: LadderContext, ladder_dir: Path, run_id: str, reused_layer: Optional[int]) -> Optional[dict]:
    """Gate V9 (Amendment 1 (5), as changed): recompute the first finished SAE (pre-registered order) on the first
    ``v9_docs`` evaluation documents, and the mean-ablation vector of a reused layer, and compare with the stored
    values within relative difference ``v9_rtol``. Bit-identity is recorded separately. Fails -> the run stops."""
    finished = [s for s in ctx.order if is_done(ladder_dir / 'per_sae' / sae_folder_name(s))]
    if not finished and reused_layer is None:
        return None
    result = {'rtol': ctx.v9_rtol, 'pass': True, 'bit_identical': True, 'max_rel_diff': 0.0}
    if finished:
        sid = finished[0]
        layer = parse_sae_id(sid)['layer']
        folder = ladder_dir / 'per_sae' / sae_folder_name(sid)
        layer_dir = ladder_dir / 'per_layer' / f'layer_{layer}'
        if not is_done(layer_dir):
            stop(ladder_dir, run_id, f'V9 FAILED: controls for layer {layer} of finished SAE {sid} are missing', {})
        _, _, hashes = _read_controls(layer_dir)
        docs = ctx.eval_docs[:ctx.v9_docs]
        new_df, new_sx, _, _ = evaluate_sae(ctx, layer, sid, docs, hashes)
        ranks = [int(d['rank']) for d in docs]
        stored = pd.read_csv(folder / 'per_doc_metrics.csv')
        stored = stored[stored['rank'].isin(ranks)]
        rows = compare_tables(_csv_roundtrip(new_df), stored, ['rank', 'condition'], ctx.v9_rtol)
        with np.load(folder / 'per_doc_sum_x.npz') as z:
            pos = {int(r): i for i, r in enumerate(z['ranks'])}
            stored_sx = z['sum_x'][[pos[r] for r in ranks]]
        sx = relative_match(new_sx, stored_sx, ctx.v9_rtol)
        result.update({'sae_id': sid, 'n_docs': len(docs), 'rows': rows, 'sum_x': sx})
        result['pass'] &= rows['pass'] and sx['pass']
        result['bit_identical'] &= rows['bit_identical'] and sx['bit_identical']
        result['max_rel_diff'] = max(rows['max_rel_diff'], sx['max_rel_diff'])
    if reused_layer is not None:
        layer_dir = ladder_dir / 'per_layer' / f'layer_{reused_layer}'
        _, stored_mean, _ = _read_controls(layer_dir)
        if stored_mean is not None:
            hook = ps.LayerSplice(ctx.layer_module(reused_layer))
            try:
                total = torch.zeros(ctx.model.config.hidden_size, dtype=torch.float64, device=ctx.device)
                n = 0
                for doc in ctx.ref_docs:
                    _forward(ctx, _as_input(ctx, doc))
                    total += hook.captured[0, 1:].double().sum(0).to(ctx.device)
                    n += len(doc['ids']) - 1
            finally:
                hook.remove()
            mv = compare_vectors((total / n).float().cpu().numpy(), stored_mean, ctx.v9_rtol)
            result['mean_ablation'] = {'layer': reused_layer, **mv}
            result['pass'] &= mv['pass']
            result['bit_identical'] &= mv['bit_identical']
    result['pass'], result['bit_identical'] = bool(result['pass']), bool(result['bit_identical'])
    if not result['pass']:
        stop(ladder_dir, run_id, f'V9 FAILED: resumed results differ from a fresh recomputation by more than '
                                 f'{ctx.v9_rtol} (relative)', {'V9': result})
    return result


def run_layer(ctx: LadderContext, layer: int, sae_ids: Sequence[str], ladder_dir: Path, run_id: str,
              max_saes: Optional[int] = None) -> dict:
    """One layer of one Kaggle run: move unfinished folders aside, V9 if resuming, shared controls (computed or
    reused), then every SAE in ``sae_ids`` that is not finished (at most ``max_saes`` new ones)."""
    ladder_dir = Path(ladder_dir)
    sae_root, layer_root = ladder_dir / 'per_sae', ladder_dir / 'per_layer'
    moved = _move_incomplete(sae_root, ladder_dir, run_id)
    moved_layers = _move_incomplete(layer_root, ladder_dir, run_id)
    layer_dir = layer_root / f'layer_{layer}'
    reused = is_done(layer_dir)
    v9 = run_v9(ctx, ladder_dir, run_id, layer if reused else None)

    if not reused:
        t0 = time.time()
        df, mean, n_ref_tok, gates = compute_layer_controls(ctx, layer)
        if not gates['V2']['pass']:
            stop(ladder_dir, run_id, f'V2 FAILED at layer {layer}', {'gates': gates})
        if not gates['V3']['pass']:
            stop(ladder_dir, run_id, f'V3 FAILED at layer {layer}', {'gates': gates})
        layer_dir.mkdir(parents=True, exist_ok=False)
        df.to_csv(layer_dir / 'controls_per_doc.csv', index=False, lineterminator='\n')
        save_npz_deterministic(layer_dir / 'mean_ablation.npz', mean=mean, n_ref_tokens=np.asarray(n_ref_tok))
        write_json(layer_dir / 'layer_info.json', {'layer': layer, 'n_eval_docs': len(ctx.eval_docs),
                                                    'n_reference_docs': len(ctx.ref_docs),
                                                    'n_reference_tokens': n_ref_tok, **gates})
        write_json(layer_dir / 'run_record.json', {'run_id': run_id, 'seconds': round(time.time() - t0, 2),
                                                    'peak_memory_gb': peak_memory_by_gpu()})
        write_done(layer_dir)
    _, _, hashes = _read_controls(layer_dir)

    ran, skipped = [], []
    for sid in sae_ids:
        folder = sae_root / sae_folder_name(sid)
        if is_done(folder):
            skipped.append(sid)
            continue
        if max_saes is not None and len(ran) >= max_saes:
            break
        df, sum_x, info, record = evaluate_sae(ctx, layer, sid, ctx.eval_docs, hashes)
        if not info['V3_original_logits_match']:
            stop(ladder_dir, run_id, f'V3 FAILED: recomputed original logits differ from the first pass ({sid})',
                 {'sae_info': info})
        folder.mkdir(parents=True, exist_ok=False)
        df.to_csv(folder / 'per_doc_metrics.csv', index=False, lineterminator='\n')
        save_npz_deterministic(folder / 'per_doc_sum_x.npz', sum_x=sum_x,
                               ranks=np.asarray([int(d['rank']) for d in ctx.eval_docs]))
        write_json(folder / 'sae_info.json', info)
        write_json(folder / 'run_record.json', {'run_id': run_id, **record})
        write_done(folder)
        ran.append(sid)
    return {'run_id': run_id, 'layer': layer, 'layer_controls_reused': reused, 'ran': ran, 'skipped': skipped,
            'moved_incomplete': moved, 'moved_incomplete_layers': moved_layers, 'v9': v9}


# ============================================================================================= real-run plumbing
def enable_strict_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).name


def gpu_facts() -> List[dict]:
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            native = torch.cuda.is_bf16_supported(including_emulation=False) if i == 0 else None
        except TypeError:
            native = 'unknown'
        props = torch.cuda.get_device_properties(i)
        out.append({'index': i, 'name': torch.cuda.get_device_name(i),
                    'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(i))),
                    'total_memory_gb': round(props.total_memory / GB, 2), 'bf16_native': native})
    return out


def host_facts(out_root: Path) -> dict:
    facts = {'disk_free_gb': round(shutil.disk_usage(out_root).free / GB, 1)}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemTotal', 'MemAvailable')):
                key, val = line.split(':')
                facts[key] = round(int(val.split()[0]) * 1024 / GB, 1)
    except OSError:
        facts['meminfo'] = 'unavailable'
    return facts


def software_versions() -> dict:
    import transformers
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'transformers': transformers.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
            'driver': _nvidia_driver()}


def _nvidia_driver():
    import subprocess
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return 'unknown'


def check_v0(cfg: dict, out_root: Path, run_id: str) -> dict:
    """V0: two T4s, bf16 recorded, software versions recorded against M6's, gated config.json at the pinned
    revision downloadable (internet on, token valid). Nothing else is downloaded if this fails."""
    gpus = gpu_facts()
    v0 = v0_hardware_check(gpus, cfg['hardware']['gpus'], cfg['hardware']['gpu_name_contains'],
                           cfg['hardware']['compute_capability'])
    sw = software_versions()
    pinned = cfg['hardware']['pinned_software']
    v0['software'] = sw
    v0['software_matches_m6'] = {k: sw.get(k) == v for k, v in pinned.items()}
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(cfg['model']['hf_id'], 'config.json', revision=cfg['model']['revision'])
        v0['gated_model_access'] = True
    except Exception as exc:  # noqa: BLE001
        v0['gated_model_access'] = False
        v0['problems'].append(f'cannot download the pinned config.json ({type(exc).__name__}); check internet, '
                              'licence acceptance and the HF_TOKEN secret')
    v0['pass'] = not v0['problems']
    if not v0['pass']:
        stop(out_root, run_id, f'V0 FAILED: {v0["problems"]}', {'V0': v0})
    return v0


def load_sample(cfg: dict, out_root: Path, run_id: str, n_eval: Optional[int] = None, n_ref: Optional[int] = None):
    """V1 (as M6): committed manifest, data and token hashes; returns eval and reference documents in rank order."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    b = cfg['builds_on']
    manifest_path, info_path = REPO / b['sample_manifest'], REPO / b['sample_info']
    info = json.loads(info_path.read_text(encoding='utf-8'))
    man = pd.read_csv(manifest_path)
    v1 = {'manifest_sha256_matches_prereg': ps.sha256_file(manifest_path) == b['sample_manifest_sha256'],
          'manifest_sha256_matches_sample_info': ps.sha256_file(manifest_path) == info['hashes'].get(
              rel(manifest_path)),
          'data_sha256_matches_prereg': info['data']['sha256'] == cfg['data']['sha256']}
    ev = man[man.set == 'eval'].sort_values('rank')
    rf = man[man.set == 'reference'].sort_values('rank')
    v1['counts'] = len(ev) == cfg['sampling']['n_eval'] and len(rf) == cfg['sampling']['n_reference']
    v1['disjoint'] = not set(ev.line_index) & set(rf.line_index)
    if not all(v1.values()):
        stop(out_root, run_id, f'V1 FAILED: {v1}', {'V1': v1})
    ev = ev.head(n_eval) if n_eval else ev
    rf = rf.head(n_ref) if n_ref else rf
    use = pd.concat([ev, rf])
    data = cfg['data']
    data_path = Path(hf_hub_download(repo_id=data['hf_dataset'], filename=data['file'], repo_type='dataset',
                                     revision=data['revision']))
    if ps.sha256_file(data_path) != data['sha256']:
        stop(out_root, run_id, 'V1 FAILED: data file SHA-256 differs from the pre-registration', {'V1': v1})
    objs = ps.load_lines(data_path, use.line_index.tolist())
    tok = AutoTokenizer.from_pretrained(cfg['tokenization']['tokenizer'], revision=cfg['tokenization']['revision'])
    ps.check_pinned_commit(getattr(tok, 'init_kwargs', {}).get('_commit_hash'), cfg['tokenization']['revision'],
                           'tokenizer')
    max_tok = int(cfg['tokenization']['max_tokens_including_bos'])

    def docs(frame):
        out = []
        for r in frame.itertuples():
            text = objs[r.line_index]['text']
            ids, _ = ps.truncate_with_bos(tok.encode(text, add_special_tokens=False), tok.bos_token_id, max_tok)
            if ps.sha256_text(text) != r.text_sha256 or ps.sha256_token_ids(ids) != r.token_ids_sha256:
                stop(out_root, run_id, f'V1 FAILED: text or token hash mismatch at line {r.line_index}', {'V1': v1})
            out.append({'rank': int(r.rank), 'line_index': int(r.line_index), 'pile_set_name': r.pile_set_name,
                        'ids': ids})
        return out

    eval_docs, ref_docs = docs(ev), docs(rf)
    v1.update({'retokenized_hashes_match': True, 'pass': True, 'n_eval': len(eval_docs), 'n_reference': len(ref_docs)})
    return eval_docs, ref_docs, v1


def make_sae_loader(cfg: dict, sae_hashes: dict, delete_after: bool = True):
    """V5: exactly the pinned ID at the pinned revision; SHA-256 recorded before use; file deleted after loading."""
    from huggingface_hub import hf_hub_download
    pinned = {r['sae_id'] for r in ladder_from_prereg(cfg)}

    def load(sae_id: str):
        if sae_id not in pinned:
            raise SystemExit(f'V5 FAILED: {sae_id} is not one of the pinned SAE IDs')
        try:
            p = Path(hf_hub_download(repo_id=cfg['sae']['hf_repo'], filename=f'{sae_id}/params.npz',
                                     revision=cfg['sae']['hf_revision']))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f'V5 FAILED: pinned SAE {sae_id} could not be downloaded ({type(exc).__name__}); '
                             'no substitution allowed')
        sha = ps.sha256_file(p)
        sae_hashes[sae_id] = sha
        with np.load(p) as z:
            params = {k: z[k] for k in z.files}
        ps.validate_sae_params(params, D_MODEL, parse_sae_id(sae_id)['d_sae'])
        if delete_after:
            for f in {p.resolve(), p}:
                try:
                    f.unlink()
                except OSError:
                    pass
        return params, sha

    return load


def load_model(cfg: dict, dtype, device_map):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg['model']['hf_id'], revision=cfg['model']['revision'],
                                                 dtype=dtype, attn_implementation='eager', device_map=device_map)
    model.eval()
    ps.check_pinned_commit(getattr(model.config, '_commit_hash', None), cfg['model']['revision'], 'model')
    return model


def copy_resume_input(src: Path, out_root: Path) -> List[str]:
    """Copy a previous run's output tree into ``out_root``; existing paths are never overwritten."""
    copied = []
    for p in sorted(Path(src).rglob('*')):
        dest = out_root / p.relative_to(src)
        if p.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied.append(dest.relative_to(out_root).as_posix())
    return copied


def latest_k0_sec_per_pass(out_root: Path) -> Optional[float]:
    """Seconds per forward pass from the newest K0 record that FINISHED (Clarification 2). A K0 that stopped or
    crashed does not count, so a projection is never based on a run that did not complete."""
    recs = sorted((out_root / 'runs').glob('K0_*/run_info.json'))
    for p in reversed(recs):
        rec = json.loads(p.read_text(encoding='utf-8'))
        val = rec.get('smoke', {}).get('seconds_per_pass')
        if rec.get('status') == 'finished' and val:
            return float(val)
    return None


def prepare_output_dir(final: Path, out_root: Path, run_id: str) -> Path:
    """Staging folder for an output that is declared finished only at the end (DONE.json written last).
    A finished ``final`` is never overwritten; an unfinished one (no valid DONE.json, e.g. left by a crash) and any
    leftover staging folder are moved, never deleted, to <out_root>/incomplete/<run id>/."""
    final, out_root = Path(final), Path(out_root)
    if is_done(final):
        raise SystemExit(f'refusing to overwrite the finished output {final}')
    staging = final.with_name(final.name + '.partial')
    for leftover in (final, staging):
        if leftover.exists():
            dest = out_root / 'incomplete' / run_id / leftover.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(f'refusing to overwrite {dest}')
            shutil.move(str(leftover), str(dest))
    staging.mkdir(parents=True)
    return staging


def finalize_output_dir(staging: Path, final: Path) -> None:
    """Move a complete staging folder into place, then write DONE.json (last)."""
    os.replace(staging, final)
    write_done(final)


def synchronize_all_gpus() -> None:
    """Wait for every queued kernel on every GPU, so an asynchronous CUDA error surfaces here, before any output is
    written or declared finished."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def fp32_reference(cfg: dict, eval_docs: List[dict], out_root: Path, run_id: str, sae_loader) -> dict:
    """Part A3 / Q0b, run as its own step K0F (Clarification 2): Gemma and the SAE in float32 across both T4s;
    layer 12, M6's 16k SAE, first 64 documents. Outputs are written to fp32_reference.partial/ and become
    fp32_reference/ (with DONE.json) only after the GPUs have finished all work and the model has been released
    cleanly. A crash at any point therefore leaves no folder that looks finished."""
    import gc
    ref = cfg['fp32_reference']
    final = out_root / 'fp32_reference'
    staging = prepare_output_dir(final, out_root, run_id)
    model = load_model(cfg, torch.float32, fp32_device_map(26))
    docs = eval_docs[:ref['n_docs']]
    ctx = LadderContext(model=model, eval_docs=docs, ref_docs=[], load_sae=sae_loader, device='cuda:0',
                        sae_device='cuda:0', order=[])
    layer, sid = int(ref['layer']), ref['sae_id']
    controls, _, _, gates = compute_layer_controls(ctx, layer, with_mean_ablation=False)
    hashes = {int(r): h for r, h in zip(controls[controls.condition == 'original']['rank'],
                                        controls[controls.condition == 'original']['logits_sha256'])}
    df, sum_x, info, record = evaluate_sae(ctx, layer, sid, docs, hashes)
    probe = _forward(ctx, _as_input(ctx, docs[0])).logits
    probe_dtype = str(probe.dtype)
    del probe
    synchronize_all_gpus()                     # every fp32 pass, including the probe, has completed without error
    v8 = {'param_dtypes': sorted({str(p.dtype) for p in model.parameters()}), 'logits_dtype': probe_dtype,
          'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'device_map': fp32_device_map(26),
          'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING'),
          'V2_pass': gates['V2']['pass'], 'V3_pass': gates['V3']['pass'] and info['V3_original_logits_match']}
    v8['pass'] = (v8['param_dtypes'] == ['torch.float32'] and v8['logits_dtype'] == 'torch.float32'
                  and not v8['tf32_matmul'] and v8['V2_pass'] and v8['V3_pass'])
    if not v8['pass']:
        stop(out_root, run_id, f'V8 FAILED: {v8}', {'V8': v8, 'gates': gates})
    pd.concat([controls.drop(columns=['logits_sha256']).assign(sae_id=None), df]).reindex(
        columns=RESULT_COLUMNS).to_csv(staging / 'per_doc_metrics.csv', index=False, lineterminator='\n')
    save_npz_deterministic(staging / 'per_doc_sum_x.npz', sum_x=sum_x, ranks=np.asarray([d['rank'] for d in docs]))
    write_json(staging / 'gates.json', {**gates, 'V8': v8, 'sae_info': info})
    passes, pass_seconds = ctx.n_passes, round(ctx.pass_seconds, 1)
    del ctx, model                             # release the two-GPU model, then check the GPUs are still healthy
    gc.collect()
    torch.cuda.empty_cache()
    synchronize_all_gpus()
    write_json(staging / 'run_info.json', {'run_id': run_id, 'status': 'fp32 reference (Part A3), step K0F',
                                           **record, 'passes': passes, 'pass_seconds': pass_seconds,
                                           'clean_gpu_release': True})
    finalize_output_dir(staging, final)
    return {'V8': v8, 'seconds': record['seconds']}


def gpu_run(args, cfg: dict) -> None:
    t_start = time.time()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.run}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = out_root / 'runs' / run_id
    if run_dir.exists():
        raise SystemExit(f'run folder {run_id} exists')
    copied = copy_resume_input(Path(args.resume_from), out_root) if args.resume_from else []
    provenance = ps.git_provenance(REPO, TRACKED_INPUTS)
    if not provenance['available'] or provenance['dirty_paths']:
        raise SystemExit(f'refused: code and inputs must be committed and unmodified ({provenance})')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    enable_strict_determinism()
    record = {'run_id': run_id, 'run': args.run, 'status': 'started',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'code_provenance': provenance,
              'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING'), 'resume_input_files_copied': len(copied),
              'input_sha256': {rel(p): ps.sha256_file(p) for p in (PREREG, EXP / 'README.md', Path(__file__),
                                                                   REPO / 'src/evaluation/pile_splicing.py')}}
    sae_hashes: Dict[str, str] = {}
    try:
        record['V0'] = check_v0(cfg, out_root, run_id)
        record['host'] = host_facts(out_root)
        ladder = ladder_from_prereg(cfg)
        plan = plan_runs(ladder)
        order = [r['sae_id'] for r in ladder]
        canonical = list(cfg['sae']['canonical_m6_ids'].values())
        loader = make_sae_loader(cfg, sae_hashes)

        def placement(sid):
            where = sae_placement(parse_sae_id(sid)['d_sae'], cfg['hardware']['planning_limit_gb_per_gpu'])
            if where == 'excluded':
                raise SystemExit(f'{sid} is excluded by the memory rule and must not be in a run')
            return 'cuda:0' if where == 'gpu0' else 'cuda:1'

        if args.run == 'K0':
            eval_docs, ref_docs, v1 = load_sample(cfg, out_root, run_id, SMOKE_N_EVAL, SMOKE_N_REF)
            record['V1'] = v1
            model = load_model(cfg, torch.bfloat16, 'cuda:0')
            smoke_ids = canonical + ['layer_12/width_65k/average_l0_72']
            ctx = LadderContext(model=model, eval_docs=eval_docs, ref_docs=ref_docs, load_sae=loader,
                                device='cuda:0', sae_device=placement, order=smoke_ids, check_unchunked_ids=canonical)
            smoke_dir = out_root / 'smoke' / run_id / 'ladder'
            recs = []
            for layer in sorted({parse_sae_id(s)['layer'] for s in smoke_ids}):
                recs.append(run_layer(ctx, layer, [s for s in smoke_ids if parse_sae_id(s)['layer'] == layer],
                                      smoke_dir, run_id))
            spp = ctx.pass_seconds / max(ctx.n_passes, 1)
            record['smoke'] = {'status': 'SMOKE (timing and memory; not a result)', 'layers': recs,
                               'seconds_per_pass': spp, 'passes': ctx.n_passes,
                               'peak_memory_gb': peak_memory_by_gpu(),
                               'projected_hours': {k: project_run_hours(plan[k], spp) for k in BUDGETED_RUNS}}
            print(f'smoke: {spp:.3f} s per pass', flush=True)
            for k, h in record['smoke']['projected_hours'].items():
                print(f'  projected {k}: {h:.2f} h')
        elif args.run == 'K0F':
            full_eval, _, v1 = load_sample(cfg, out_root, run_id, cfg['fp32_reference']['n_docs'], 1)
            record['V1'] = v1
            record['fp32_reference'] = fp32_reference(cfg, full_eval, out_root, run_id, loader)
        else:
            ids = plan[args.run]
            if args.widths:
                ids = [s for s in ids if parse_sae_id(s)['width'] in args.widths]
            spp = latest_k0_sec_per_pass(out_root)
            if spp is None:
                raise SystemExit('no finished K0 record found: run K0 first (or pass its output with --resume-from)')
            layers = sorted({parse_sae_id(s)['layer'] for s in ids})
            new_layers = sum(not is_done(out_root / 'ladder/per_layer' / f'layer_{L}') for L in layers)
            todo = [s for s in ids if not is_done(out_root / 'ladder/per_sae' / sae_folder_name(s))]
            projected = project_run_hours(todo, spp, new_layers=new_layers)
            budget = may_start(projected, args.week_used_hours)
            record['budget'] = {'seconds_per_pass_from_K0': spp, 'projected_hours': projected,
                                'week_used_hours': args.week_used_hours, **budget}
            if not budget['ok']:
                stop(out_root, run_id, f'run not started: {budget["reasons"]}', {'budget': record['budget']})
            eval_docs, ref_docs, v1 = load_sample(cfg, out_root, run_id)
            record['V1'] = v1
            model = load_model(cfg, torch.bfloat16, 'cuda:0')
            ctx = LadderContext(model=model, eval_docs=eval_docs, ref_docs=ref_docs, load_sae=loader,
                                device='cuda:0', sae_device=placement, order=order, check_unchunked_ids=canonical,
                                v9_docs=int(cfg['validity_gates']['V9_docs']),
                                v9_rtol=float(cfg['validity_gates']['V9_rel_tol']))
            record['layers'] = [run_layer(ctx, L, [s for s in ids if parse_sae_id(s)['layer'] == L],
                                          out_root / 'ladder', run_id) for L in layers]
            record['seconds_per_pass'] = ctx.pass_seconds / max(ctx.n_passes, 1)
        record['status'] = 'finished'
    except SystemExit as exc:
        record['status'] = f'stopped: {exc}'
        raise
    except BaseException as exc:
        record['status'] = f'crashed: {type(exc).__name__}: {exc}'
        raise
    finally:
        # Written in every case (Clarification 2), so what a step measured is never lost to a later failure.
        record['params_npz_sha256'] = sae_hashes
        try:
            record['peak_memory_gb'] = peak_memory_by_gpu()
        except Exception as exc:  # noqa: BLE001  the GPU may be unusable after a CUDA error
            record['peak_memory_gb'] = f'unavailable ({type(exc).__name__})'
        record['runtime_s'] = round(time.time() - t_start, 1)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / 'run_info.json', record)
        print(f'run {run_id}: {record["status"]} after {record["runtime_s"] / 3600:.2f} h; '
              f'record: {run_dir / "run_info.json"}', flush=True)


# ===================================================================================================== aggregate
def _load_s03():
    import importlib.util
    spec = importlib.util.spec_from_file_location('s03', REPO / 'scripts/gemma/03_pile_gemma_scope_splicing.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def doc_mean_dce(rows: pd.DataFrame, cond: str, ranks: Sequence[int]) -> np.ndarray:
    s = rows[rows.condition == cond].set_index('rank').loc[list(ranks)]
    return ((s.ce_new_sum - s.ce_orig_sum) / s.n_pred).to_numpy(dtype=float)


def aggregate(cfg: dict, out_root: Path) -> None:
    """CPU step after all runs: pooled tables, L0 bands, H1-H3 and V6, from the per-SAE and per-layer folders."""
    s03 = _load_s03()   # Simba's M6 aggregation (summarize), reused unchanged so metric definitions are identical
    ladder_dir = out_root / 'ladder'
    outputs = ['per_doc_metrics.csv', 'summary_by_sae.csv', 'summary_by_sae_source.csv', 'controls.csv',
               'l0_bands.csv', 'h1_l0_within_width.csv', 'h2_width_matched_l0.csv', 'h3_hg1_ladder.csv',
               'v6_m6_consistency.csv', 'gates.json', 'run_info.json']
    clash = [f for f in outputs if (ladder_dir / f).exists()]
    if clash:
        raise SystemExit(f'refusing to overwrite {clash}')
    n_boot, seed = int(cfg['bootstrap']['n_resamples']), int(cfg['seeds']['bootstrap'])
    low_n = int(cfg['low_n_source_threshold_docs'])
    margins = cfg['margins']
    ladder = ladder_from_prereg(cfg)

    layers_ctrl, infos, per_doc, sumx = {}, {}, [], {}
    for L in cfg['layers']:
        d = ladder_dir / 'per_layer' / f'layer_{L}'
        if is_done(d):
            layers_ctrl[L] = pd.read_csv(d / 'controls_per_doc.csv')
    not_run = []
    for r in ladder:
        d = ladder_dir / 'per_sae' / sae_folder_name(r['sae_id'])
        if not is_done(d):
            not_run.append(r['sae_id'])
            continue
        infos[r['sae_id']] = json.loads((d / 'sae_info.json').read_text(encoding='utf-8'))
        per_doc.append(pd.read_csv(d / 'per_doc_metrics.csv'))
        with np.load(d / 'per_doc_sum_x.npz') as z:
            sumx[r['sae_id']] = (z['sum_x'], [int(x) for x in z['ranks']])
    if not per_doc:
        raise SystemExit('no finished SAE folders to aggregate')
    rows = pd.concat(per_doc, ignore_index=True)
    ranks = sorted(int(x) for x in rows['rank'].unique())
    idx = ps.bootstrap_indices(len(ranks), n_boot, seed)

    def sae_frame(sid):
        L = infos[sid]['layer']
        ma = layers_ctrl[L][layers_ctrl[L].condition == 'mean_ablation'].drop(columns=['logits_sha256'])
        return pd.concat([rows[rows.sae_id == sid], ma.assign(sae_id=sid)], ignore_index=True)

    summaries, by_source = [], []
    evaluated = [s for s in infos if infos[s]['V4']['pass']]
    for sid in infos:
        f = sae_frame(sid)
        sx, sx_ranks = sumx[sid]
        row_of = {r: i for i, r in enumerate(sx_ranks)}
        meta = {'sae_id': sid, **parse_sae_id(sid), 'measured_l0_v4': infos[sid]['measured_l0'],
                'v4_pass': infos[sid]['V4']['pass']}
        L = meta['layer']
        summaries.append({**meta, **s03.summarize(f, sx[:, None, :], row_of, ranks, [L], n_boot, seed)[0]})
        src = f[f.condition == 'sae_splice'][['rank', 'pile_set_name']].drop_duplicates()
        for source, grp in src.groupby('pile_set_name'):
            rr = sorted(grp['rank'].astype(int))
            out = s03.summarize(f, sx[:, None, :], row_of, rr, [L], n_boot, seed)[0]
            by_source.append({**meta, 'pile_set_name': source, 'low_n': out['n_docs'] < low_n, **out})
    summary = pd.DataFrame(summaries)
    controls = []
    for L, c in layers_ctrl.items():
        for cond in ('identity', 'original_repeat'):
            sub = c[c.condition == cond]
            controls.append({'layer': L, 'condition': cond, 'mean_kl': float(sub.kl_sum.sum() / sub.n_pred.sum()),
                             'max_doc_mean_kl': float((sub.kl_sum / sub.n_pred).max())})

    # L0 bands and H1 / H2 on SAEs that passed V4
    table = summary[summary.v4_pass][['sae_id', 'layer', 'width', 'd_sae', 'l0']].rename(columns={'l0': 'measured_l0'})
    bands = l0_bands(table, tol=float(cfg['questions']['H2']['band_tolerance']))
    dce = {s: doc_mean_dce(rows[rows.sae_id == s], 'sae_splice', ranks) for s in evaluated}
    dce_rand = {s: doc_mean_dce(rows[rows.sae_id == s], 'norm_matched_random_error', ranks) for s in evaluated}
    h1 = []
    for (L, w), cell in table.groupby(['layer', 'width']):
        ordered = cell.sort_values('measured_l0').sae_id.tolist()
        if len(ordered) < 2:
            continue
        res = pairwise_iut(ordered, dce, idx)
        ev_pairs = []
        for a, b in zip(ordered[:-1], ordered[1:]):
            fa, fb = sae_frame(a), sae_frame(b)
            def ev_of(f, sid, r):
                sp = f[f.condition == 'sae_splice'].set_index('rank').loc[[ranks[i] for i in r]]
                sx, sr = sumx[sid]
                pos = {x: i for i, x in enumerate(sr)}
                return ps.ev_from_sums(sp.sse, sp.sum_sq, sx[[pos[ranks[i]] for i in r]], sp.n_tok)
            stat = lambda r: ev_of(fb, b, r) - ev_of(fa, a, r)  # noqa: E731  predicted > 0
            lo, hi = ps.bootstrap_ci(stat, idx, 0.95)
            ev_pairs.append((lo, hi))
        for p in res['pairs']:
            h1.append({'layer': L, 'width': w, **p, 'cell_verdict': res['verdict'],
                       'ev_cell_verdict': iut_verdict(ev_pairs)})
    h2 = []
    for b in bands:
        res = pairwise_iut(b['members'], dce, idx)
        for p in res['pairs']:
            l0a = float(table.set_index('sae_id').loc[p['lower'], 'measured_l0'])
            l0b = float(table.set_index('sae_id').loc[p['higher'], 'measured_l0'])
            h2.append({'layer': b['layer'], 'anchor': b['anchor_sae_id'], 'target_l0': b['target_l0'], **p,
                       'measured_l0_lower': l0a, 'measured_l0_higher': l0b, 'band_verdict': res['verdict'],
                       'also_in_band': ';'.join(b['also_in_band'])})
    # H3 with Holm
    h3 = []
    for s in evaluated:
        d = dce[s] - dce_rand[s]
        boots = np.array([d[r].mean() for r in idx])
        lo, hi = ps.bootstrap_ci(lambda r: float(d[r].mean()), idx, 0.95)
        se = float(boots.std(ddof=1))
        h3.append({'sae_id': s, 'mean_d': float(d.mean()), 'ci95_lo': lo, 'ci95_hi': hi, 'bootstrap_se': se,
                   'p_one_sided': one_sided_p(float(d.mean()), se), 'reversal': hi < 0,
                   'ratio_dce_sae_over_random': float(dce[s].mean() / dce_rand[s].mean())})
    sig = holm([r['p_one_sided'] for r in h3], float(cfg['questions']['H3']['alpha']))
    for r, flag in zip(h3, sig):
        r['holm_significant'] = bool(flag and r['mean_d'] > 0)
    h3_overall = h3_verdict([r['holm_significant'] for r in h3]) if h3 else 'not computed'

    # V6: canonical 16k SAEs vs M6's committed rows
    m6 = pd.read_csv(REPO / cfg['builds_on']['per_doc_metrics'])
    m6_hg1 = pd.read_csv(M6 / 'h_g1.csv').set_index('layer')
    exact_cols = ['n_pred', 'ce_orig_sum', 'ce_new_sum', 'kl_sum', 'top1_agree', 'n_tok', 'sse', 'sum_sq', 'cos_sum',
                  'l0_sum']
    v6 = []
    for L, sid in cfg['sae']['canonical_m6_ids'].items():
        L = int(L)
        if sid not in infos:
            v6.append({'layer': L, 'sae_id': sid, 'verdict': 'not run'})
            continue
        ours = rows[rows.sae_id == sid]
        theirs = m6[(m6.layer == L) & m6.condition.isin(['sae_splice', 'norm_matched_random_error'])]
        merged = ours.merge(theirs, on=['rank', 'condition'], suffixes=('', '_m6'))
        worst = 0.0
        for col in exact_cols:
            a, b = merged[col].to_numpy(float), merged[f'{col}_m6'].to_numpy(float)
            r = relative_match(a, b, 1.0)
            worst = max(worst, r['max_rel_diff'])
        agree = infos[sid]['active_latent_agreement_chunked_vs_unchunked']
        exact = (len(merged) == len(ours) == len(theirs) and worst <= cfg['validity_gates']['V6_exact_rel_tol']
                 and agree is not None and agree >= cfg['validity_gates']['V6_active_latent_agreement_min'])
        verdicts = {}
        m6sp = m6[(m6.layer == L)]
        for name, cond, col in (('delta_ce_sae', 'sae_splice', None), ('delta_ce_random', 'norm_matched_random_error',
                                                                      None)):
            dd = doc_mean_dce(ours, cond, ranks) - doc_mean_dce(m6sp, cond, ranks)
            lo, hi = ps.bootstrap_ci(lambda r: float(dd[r].mean()), idx, 0.95)
            verdicts[name] = equivalence_verdict(lo, hi, margins[name])
        for name, col in (('kl_sae', 'kl_sum'), ('top1_agreement', 'top1_agree')):
            a = ours[ours.condition == 'sae_splice'].set_index('rank').loc[ranks]
            b = m6sp[m6sp.condition == 'sae_splice'].set_index('rank').loc[ranks]
            dd = (a[col] / a.n_pred).to_numpy(float) - (b[col] / b.n_pred).to_numpy(float)
            lo, hi = ps.bootstrap_ci(lambda r: float(dd[r].mean()), idx, 0.95)
            verdicts[name] = equivalence_verdict(lo, hi, margins[name])
        # EV: pooled over the resampled documents, paired (same resample for both runs); margin +-0.001
        a = ours[ours.condition == 'sae_splice'].set_index('rank').loc[ranks]
        b = m6sp[m6sp.condition == 'sae_splice'].set_index('rank').loc[ranks]
        sx_ours, sr = sumx[sid]
        pos = {x: i for i, x in enumerate(sr)}
        sx_ours = sx_ours[[pos[r] for r in ranks]]
        with np.load(M6 / 'per_doc_sum_x.npz') as z:
            m6_pos = {int(x): i for i, x in enumerate(z['ranks'])}
            sx_m6 = z['sum_x'][[m6_pos[r] for r in ranks], list(z['layers']).index(L)]

        def ev_diff(r):
            return (ps.ev_from_sums(a.sse.to_numpy()[r], a.sum_sq.to_numpy()[r], sx_ours[r], a.n_tok.to_numpy()[r])
                    - ps.ev_from_sums(b.sse.to_numpy()[r], b.sum_sq.to_numpy()[r], sx_m6[r], b.n_tok.to_numpy()[r]))
        lo, hi = ps.bootstrap_ci(ev_diff, idx, 0.95)
        verdicts['explained_variance'] = equivalence_verdict(lo, hi, margins['explained_variance'])
        hg1_ours = ps.h_g1(doc_mean_dce(ours, 'sae_splice', ranks), doc_mean_dce(ours, 'norm_matched_random_error',
                                                                                  ranks), idx)['verdict']
        within = all(v == 'EQUIVALENT' for v in verdicts.values()) and hg1_ours == m6_hg1.loc[L, 'verdict']
        v6.append({'layer': L, 'sae_id': sid, 'max_rel_diff_sums': worst, 'active_latent_agreement': agree,
                   **{f'{k}_verdict': v for k, v in verdicts.items()}, 'hg1_ours': hg1_ours,
                   'hg1_m6': m6_hg1.loc[L, 'verdict'], 'verdict': v6_verdict(exact, within)})

    tables = {'per_doc_metrics.csv': rows, 'summary_by_sae.csv': summary,
              'summary_by_sae_source.csv': pd.DataFrame(by_source), 'controls.csv': pd.DataFrame(controls),
              'l0_bands.csv': pd.DataFrame([{**b, 'members': ';'.join(b['members']),
                                             'also_in_band': ';'.join(b['also_in_band'])} for b in bands]),
              'h1_l0_within_width.csv': pd.DataFrame(h1), 'h2_width_matched_l0.csv': pd.DataFrame(h2),
              'h3_hg1_ladder.csv': pd.DataFrame(h3), 'v6_m6_consistency.csv': pd.DataFrame(v6)}
    empty_columns = {  # an empty table still gets its header, so readers never see a header-less file
        'l0_bands.csv': ['layer', 'anchor_sae_id', 'target_l0', 'members', 'also_in_band'],
        'h1_l0_within_width.csv': ['layer', 'width', 'lower', 'higher', 'mean_d', 'ci95_lo', 'ci95_hi',
                                   'cell_verdict', 'ev_cell_verdict'],
        'h2_width_matched_l0.csv': ['layer', 'anchor', 'target_l0', 'lower', 'higher', 'mean_d', 'ci95_lo', 'ci95_hi',
                                    'measured_l0_lower', 'measured_l0_higher', 'band_verdict', 'also_in_band'],
        'h3_hg1_ladder.csv': ['sae_id', 'mean_d', 'ci95_lo', 'ci95_hi', 'bootstrap_se', 'p_one_sided', 'reversal',
                              'ratio_dce_sae_over_random', 'holm_significant']}
    for name, t in tables.items():
        if t.empty and name in empty_columns:
            t = pd.DataFrame(columns=empty_columns[name])
        t.to_csv(ladder_dir / name, index=False, lineterminator='\n')
    gates = {'V3_controls': controls, 'V4_excluded': [s for s in infos if not infos[s]['V4']['pass']],
             'V6': v6, 'H3_overall': h3_overall, 'not_run': not_run}
    write_json(ladder_dir / 'gates.json', gates)
    write_json(ladder_dir / 'run_info.json', {
        'status': 'aggregate of the R3 ladder runs', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'n_saes_aggregated': len(infos), 'not_run': not_run,
        'output_sha256': {f: ps.sha256_file(ladder_dir / f) for f in outputs if f != 'run_info.json'}})
    print(f'aggregated {len(infos)} SAEs; H3: {h3_overall}; V6: {[r["verdict"] for r in v6]}')


# ========================================================================================================= main
def enable_launch_blocking() -> None:
    """K0F only (Clarification 2): make CUDA report an error at the call that caused it. Must be set before CUDA is
    initialised in this process; it changes when errors are reported, not what is computed."""
    if torch.cuda.is_initialized():
        raise SystemExit('K0F must start in a fresh process: CUDA was already initialised, so '
                         'CUDA_LAUNCH_BLOCKING=1 would have no effect')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', choices=STEP_IDS,
                    help='which Kaggle step to execute (Amendment 1 (4); K0F: Clarification 2)')
    ap.add_argument('--aggregate', action='store_true', help='CPU step after all runs')
    ap.add_argument('--out', default=str(EXP), help='output root (on Kaggle: a folder under /kaggle/working)')
    ap.add_argument('--resume-from', default=None, help="previous run's output root, copied in first (no overwrite)")
    ap.add_argument('--week-used-hours', type=float, default=None,
                    help='GPU hours already used this week, from the Kaggle quota display (K1-K6)')
    ap.add_argument('--widths', nargs='+', default=None, help='restrict a run to these widths (to split a run)')
    args = ap.parse_args(argv)
    if bool(args.run) == bool(args.aggregate):
        raise SystemExit('give exactly one of --run or --aggregate')
    if args.run in BUDGETED_RUNS and args.week_used_hours is None:
        raise SystemExit('--week-used-hours is required for K1-K6 (budget rule)')
    if args.widths and (args.run not in BUDGETED_RUNS or any(w not in WIDTHS for w in args.widths)):
        raise SystemExit('--widths needs a run K1-K6 and known widths')
    cfg = json.loads(PREREG.read_text(encoding='utf-8'))
    if args.aggregate:
        aggregate(cfg, Path(args.out))
        return
    if args.run == 'K0F':
        enable_launch_blocking()
    gpu_run(args, cfg)


if __name__ == '__main__':
    main()
