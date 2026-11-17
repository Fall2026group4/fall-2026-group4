#!/usr/bin/env python3
"""R3 figures: Gemma Scope width x L0 ladder and Q0b, from saved tables only (no model, SAE, GPU or download).

Pre-registration: results/gemma2_pile/r3_gemma_scope_ladder/README.md (Section 1, Amendment 1, Clarification 1).
Reads <r3-root>/ladder/ (written by scripts/gemma/05_pile_gemma_scope_ladder.py --aggregate) and, if present,
<r3-root>/precision_t4_vs_fp32/ (scripts/gemma/06_r3_precision_comparison.py). Style and palette are reused from
M6's scripts/gemma/04_pile_gemma_scope_figures.py (loaded from its file, unchanged).

Before plotting it checks that
  * ladder/run_info.json exists and no STOPPED_*.json file is present in ladder/;
  * every table read here has the SHA-256 recorded in ladder/run_info.json (CRLF normalised to LF, as M6's script 04);
  * the Q0b table matches the SHA-256 in q0b_info.json.

Figures (descriptive; no layer ranking, no "best SAE"):
  fig1_reconstruction_vs_l0.png   Q1  EV vs measured L0, one line per width, one panel per layer
  fig2_splicing_vs_l0.png         Q2  delta CE_SAE and loss recovered vs measured L0 (loss recovered read with dCE)
  fig3_h3_sae_vs_random.png       H3  delta CE_SAE / delta CE_random vs measured L0; filled = Holm-significant
  fig4_h2_width_bands.png         H2  delta CE_SAE vs width inside each L0 band, with the band verdict
  fig5_source_variation.png       Q3  delta CE_SAE per Pile source (sources with >= 20 eval documents) vs measured L0
  fig6_q0b_t4_vs_fp32.png         Q0b mean |T4 bf16 - fp32| as a fraction of the fixed margin (only if Q0b exists)
SAEs that failed V4 are not drawn and are listed in the footer. Refuses to overwrite; use --out-dir for a copy.

Run from repository root:  python scripts/gemma/07_r3_figures.py [--r3-root DIR] [--out-dir DIR]
"""
import argparse
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'results/gemma2_pile/r3_gemma_scope_ladder'
LADDER_INPUTS = ['summary_by_sae.csv', 'summary_by_sae_source.csv', 'h3_hg1_ladder.csv', 'h2_width_matched_l0.csv',
                 'l0_bands.csv', 'gates.json']
FIGURES = ['fig1_reconstruction_vs_l0.png', 'fig2_splicing_vs_l0.png', 'fig3_h3_sae_vs_random.png',
           'fig4_h2_width_bands.png', 'fig5_source_variation.png', 'fig6_q0b_t4_vs_fp32.png']
INFO = 'figure_info.json'
WIDTH_ORDER = ['16k', '32k', '65k', '131k', '262k', '524k']
CANONICAL = {'layer_5/width_16k/average_l0_68', 'layer_12/width_16k/average_l0_82', 'layer_19/width_16k/average_l0_73'}


def load_m6_style():
    spec = importlib.util.spec_from_file_location('s04', REPO / 'scripts/gemma/04_pile_gemma_scope_figures.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines palette and helpers only; main() is not run
    return mod


S04 = load_m6_style()
INK, INK2, GRID, SURFACE, SAE_C, RAND_C = S04.INK, S04.INK2, S04.GRID, S04.SURFACE, S04.SAE_C, S04.RAND_C
SAVE_KW = S04.SAVE_KW
# width is ordered, so a light-to-dark ramp of M6's SAE blue (16k lightest ... 524k darkest)
WIDTH_C = dict(zip(WIDTH_ORDER, ['#9cc3ef', '#6fa6e6', '#2a78d6', '#1f5ca6', '#163f73', '#0c2645']))
SOURCE_C = ['#2a78d6', '#eb6834', '#3a9d5d', '#8a5cc2', '#c9a227', '#d6457a', '#52514e', '#2bb3b1']


# ------------------------------------------------------------------------------------------------------- checks
def check_inputs(root: Path) -> dict:
    lad = root / 'ladder'
    info_path = lad / 'run_info.json'
    if not info_path.exists():
        sys.exit(f'STOP: {info_path} missing - run script 05 --aggregate first.')
    stops = sorted(p.name for p in lad.glob('STOPPED_*.json'))
    if stops:
        sys.exit(f'STOP: stop files present in ladder/, results must not be plotted: {stops}')
    recorded = json.loads(info_path.read_text(encoding='utf-8'))['output_sha256']
    hashes = {}
    for name in LADDER_INPUTS:
        got = S04.sha256_lf(lad / name)
        if recorded.get(name) != got:
            sys.exit(f'STOP: ladder/{name} SHA-256 does not match ladder/run_info.json.')
        hashes[f'ladder/{name}'] = got
    q0b = root / 'precision_t4_vs_fp32'
    if (q0b / 'q0b_info.json').exists():
        rec = json.loads((q0b / 'q0b_info.json').read_text(encoding='utf-8'))['output_sha256']['q0b_t4_vs_fp32.csv']
        got = S04.sha256_lf(q0b / 'q0b_t4_vs_fp32.csv')
        if rec != got:
            sys.exit('STOP: precision_t4_vs_fp32/q0b_t4_vs_fp32.csv SHA-256 does not match q0b_info.json.')
        hashes['precision_t4_vs_fp32/q0b_t4_vs_fp32.csv'] = got
    return hashes


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.stat().st_size > 0 else pd.DataFrame()


# ------------------------------------------------------------------------------------------------------ helpers
def style(ax, title, ylabel=None, xlog=True):
    S04.style(ax, title, ylabel)
    if xlog:
        ax.set_xscale('log')
        ax.set_xlabel('measured L0 on our Pile sample (log scale)', color=INK2, fontsize=8)


def width_lines(ax, s, col, layer, canonical_marker=True):
    """One line per width at ``layer``: points with 95 % CI bars, joined in order of measured L0."""
    sub = s[s.layer == layer]
    for w in WIDTH_ORDER:
        g = sub[sub.width == w].sort_values('l0')
        if g.empty:
            continue
        v, lo, hi = (g[c].to_numpy(float) for c in (col, f'{col}_ci95_lo', f'{col}_ci95_hi'))
        ax.errorbar(g.l0.to_numpy(float), v, yerr=[v - lo, hi - v], fmt='-o',
                    color=WIDTH_C[w], ms=4, lw=1.3, elinewidth=1, capsize=2, label=w, zorder=2)
        if canonical_marker:
            c = g[g.sae_id.isin(CANONICAL)]
            ax.scatter(c.l0, c[col], s=110, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=3,
                       label='M6 SAE' if not c.empty else None)


def legend(ax, loc='best'):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    if seen:
        ax.legend(seen.values(), seen.keys(), fontsize=7, frameon=False, labelcolor=INK2, loc=loc)


def save(fig, out: Path, name: str, title: str, foot: str):
    """Footer wrapped to the figure width (about 1.7 characters per pixel-inch at 7 pt); more lines, more margin."""
    width_chars = int(fig.get_figwidth() * 17)
    lines = textwrap.wrap(foot, width_chars)
    S04.header(fig, title, '\n'.join(lines), bottom=0.03 + 0.11 * len(lines) / fig.get_figheight())
    fig.savefig(out / name, **SAVE_KW)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ figures
def fig1(s, layers, out, foot):
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, L in zip(axes[0], layers):
        width_lines(ax, s, 'explained_variance', L)
        style(ax, f'layer {L}', 'explained variance (higher is better)')
        legend(ax, 'lower right')
    save(fig, out, FIGURES[0], 'Q1  Reconstruction: explained variance vs sparsity, by width', foot)


def fig2(s, layers, out, foot):
    fig, axes = plt.subplots(2, len(layers), figsize=(5 * len(layers), 7.6), facecolor=SURFACE, squeeze=False)
    for j, L in enumerate(layers):
        width_lines(axes[0, j], s, 'delta_ce_sae', L)
        style(axes[0, j], f'layer {L}: delta CE when the SAE output is spliced in', 'nats per prediction (lower is better)')
        legend(axes[0, j], 'upper right')
        width_lines(axes[1, j], s, 'loss_recovered', L)
        style(axes[1, j], f'layer {L}: loss recovered (vs mean ablation)', 'fraction (higher is better)')
    save(fig, out, FIGURES[1], 'Q2  Behaviour when spliced, vs sparsity, by width (read loss recovered with delta CE)',
         foot)


def fig3(s, h3, layers, out, foot):
    sig = dict(zip(h3.sae_id, h3.holm_significant)) if not h3.empty else {}
    rev = dict(zip(h3.sae_id, h3.reversal)) if not h3.empty else {}
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, L in zip(axes[0], layers):
        sub = s[s.layer == L]
        for w in WIDTH_ORDER:
            g = sub[sub.width == w].sort_values('l0')
            if g.empty:
                continue
            ax.plot(g.l0, g.ratio_dce_sae_over_random, '-', color=WIDTH_C[w], lw=1.1, zorder=1, label=w)
            for r in g.itertuples():
                filled = bool(sig.get(r.sae_id, False))
                edge = RAND_C if rev.get(r.sae_id, False) else WIDTH_C[w]
                ax.scatter(r.l0, r.ratio_dce_sae_over_random, s=34, color=WIDTH_C[w] if filled else SURFACE,
                           edgecolors=edge, linewidths=1.4, zorder=2)
        ax.axhline(1, color=INK2, lw=1, ls='--', zorder=0)
        style(ax, f'layer {L}', 'delta CE (SAE) / delta CE (random error)')
        legend(ax, 'upper right')
    save(fig, out, FIGURES[2], 'H3  Is the SAE error more harmful than a random error of the same size? (> 1: yes)',
         foot + '  Filled: Holm-significant d > 0; hollow: not; orange edge: reversal (CI upper bound < 0).')


def fig4(s, h2, bands, out, foot):
    if bands.empty:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=SURFACE)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No L0 band has a member of another width (measured L0, +-10 %).', ha='center',
                color=INK2, fontsize=10)
        save(fig, out, FIGURES[3], 'H2  Width at matched sparsity', foot)
        return
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), facecolor=SURFACE, squeeze=False)
    by_id = s.set_index('sae_id')
    for ax, b in zip(axes[0], bands.itertuples()):
        members = str(b.members).split(';')
        g = by_id.loc[members]
        x = np.arange(len(members))
        v, lo, hi = (g[c].to_numpy(float) for c in ('delta_ce_sae', 'delta_ce_sae_ci95_lo', 'delta_ce_sae_ci95_hi'))
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt='-o', color=SAE_C, ms=5, lw=1.2, elinewidth=1, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}\nL0 {l0:.0f}" for w, l0 in zip(g.width, g.l0)], fontsize=7, color=INK2)
        verdict = h2[h2.anchor == b.anchor_sae_id].band_verdict.iloc[0] if not h2.empty else 'n/a'
        style(ax, f'layer {b.layer}, target L0 {b.target_l0:.0f}: {verdict}', 'delta CE_SAE (lower is better)',
              xlog=False)
    save(fig, out, FIGURES[3], 'H2  Width at matched sparsity: delta CE_SAE inside each L0 band', foot)


def fig5(src, layers, out, foot, low_n_note):
    src = src[~src.low_n.astype(bool)]
    sources = sorted(src.pile_set_name.unique())
    colors = dict(zip(sources, SOURCE_C * (1 + len(sources) // len(SOURCE_C))))
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, L in zip(axes[0], layers):
        sub = src[src.layer == L]
        for name in sources:
            g = sub[sub.pile_set_name == name].sort_values('l0')
            ax.scatter(g.l0, g.delta_ce_sae, s=16, color=colors[name], alpha=0.85, linewidths=0, label=name)
        style(ax, f'layer {L}', 'delta CE_SAE (lower is better)')
        legend(ax, 'upper right')
    save(fig, out, FIGURES[4], 'Q3  Variation across Pile sources (each point: one SAE on one source)',
         foot + f'  {low_n_note}')


def fig6(q, out, foot):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    metrics = list(dict.fromkeys(q.metric))
    runs = list(dict.fromkeys(q.bf16_run))
    for k, run in enumerate(runs):
        g = q[q.bf16_run == run].set_index('metric').loc[metrics]
        x = np.arange(len(metrics)) + (k - (len(runs) - 1) / 2) * 0.22
        v, lo, hi = g.mean_abs_error / g.margin, g.abs_error_ci95_lo / g.margin, g.abs_error_ci95_hi / g.margin
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt='o' if k == 0 else 's', color=SAE_C if k == 0 else RAND_C,
                    ms=5, elinewidth=1.2, capsize=3, label=run)
    ax.axhline(1, color=INK2, lw=1, ls='--')
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.replace('_', ' ') + ('\n(primary)' if m == 'delta_ce_sae' else '') for m in metrics],
                       fontsize=7, color=INK2)
    style(ax, 'mean |T4 bf16 - fp32| / pre-registered margin  (below 1: within margin)', 'fraction of margin',
          xlog=False)
    legend(ax, 'center right')  # away from the dashed margin line at 1
    save(fig, out, FIGURES[5], 'Q0b  How far are T4 bf16 numbers from fp32?  Layer 12, 16k SAE, first 64 documents',
         foot + '  One layer, one SAE, 64 documents; fp32 is a reference, not exact truth.')


# --------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--r3-root', default=str(EXP))
    ap.add_argument('--out-dir', default=None, help='default: <r3-root>/figures')
    args = ap.parse_args(argv)
    root = Path(args.r3_root)
    out = Path(args.out_dir) if args.out_dir else root / 'figures'
    hashes = check_inputs(root)
    have_q0b = 'precision_t4_vs_fp32/q0b_t4_vs_fp32.csv' in hashes
    todo = FIGURES if have_q0b else FIGURES[:5]
    clash = [f for f in todo + [INFO] if (out / f).exists()]
    if clash:
        sys.exit(f'STOP: refusing to overwrite {clash} in {out}')
    out.mkdir(parents=True, exist_ok=True)

    lad = root / 'ladder'
    s_all = read_csv(lad / 'summary_by_sae.csv')
    s = s_all[s_all.v4_pass.astype(bool)]
    excluded = s_all[~s_all.v4_pass.astype(bool)].sae_id.tolist()
    gates = json.loads((lad / 'gates.json').read_text(encoding='utf-8'))
    layers = sorted(int(L) for L in s.layer.unique())
    foot = (f'R3, Gemma-2-2B resid_post, Gemma Scope (google/gemma-scope-2b-pt-res), 512 Pile validation documents; '
            f'95 % document-bootstrap CIs. {len(s)} SAEs drawn.'
            + (f' Excluded by V4: {", ".join(excluded)}.' if excluded else '')
            + (f' Not run: {len(gates.get("not_run", []))}.' if gates.get('not_run') else ''))
    fig1(s, layers, out, foot + ' Circled: the M6 SAE.')
    fig2(s, layers, out, foot + ' Circled: the M6 SAE.')
    fig3(s, read_csv(lad / 'h3_hg1_ladder.csv'), layers, out, foot + f' H3 overall: {gates.get("H3_overall")}.')
    fig4(s, read_csv(lad / 'h2_width_matched_l0.csv'), read_csv(lad / 'l0_bands.csv'), out, foot)
    src = read_csv(lad / 'summary_by_sae_source.csv')
    src = src[src.sae_id.isin(set(s.sae_id))]
    low = sorted(src[src.low_n.astype(bool)].pile_set_name.unique())
    fig5(src, layers, out, foot, f'LOW-N sources not drawn: {", ".join(low) if low else "none"}.')
    if have_q0b:
        fig6(read_csv(root / 'precision_t4_vs_fp32/q0b_t4_vs_fp32.csv'), out, 'R3 Q0b (Amendment 1 (2)).')
    info = {'figures': todo, 'input_sha256_lf': hashes, 'v4_excluded': excluded,
            'q0b_figure': 'drawn' if have_q0b else 'not drawn (no precision_t4_vs_fp32/ yet)'}
    (out / INFO).write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'wrote {len(todo)} figures to {out}')


if __name__ == '__main__':
    main()
