#!/usr/bin/env python3
"""M6 figures: Gemma Scope SAE reconstruction and splicing on The Pile (Gemma-2-2B, layers 5/12/19).

Reads ONLY the saved outputs of scripts/gemma/03_pile_gemma_scope_splicing.py in
results/gemma2_pile/gemma_scope_splicing/ (no model, no SAE, no GPU, no downloads) and writes PNGs to its figures/.

Before plotting it checks that
  * run_info.json exists (a folder without it is not a result, safeguard A7) and no *_FAILED / *_MISMATCH file exists;
  * every validity gate in gates.json passed;
  * every CSV/JSON read here has the SHA-256 recorded in run_info.json (hashed after CRLF -> LF normalisation, so a
    Windows autocrlf checkout of the same commit verifies too);
  * the H-G1 mean differences recomputed from per_doc_metrics.csv equal h_g1.csv.

Figures (pre-registered questions in results/gemma2_pile/gemma_scope_splicing/README.md):
  fig1_reconstruction_by_layer.png   Q1  EV, cosine, relative squared error, L0
  fig2_splicing_by_layer.png         Q2  delta CE (SAE vs norm-matched random), KL, top-1 agreement, loss recovered
  fig3_h_g1_paired_difference.png    H-G1 per-document paired differences and mean d with 95 % / 98.33 % CIs
  fig4_source_variation.png          Q3  EV and delta CE_SAE per Pile source (sources with >= 20 eval documents only)

Layers are shown side by side for description only; each layer has its own SAE (different release L0), and no layer
ranking is drawn or implied. Refuses to overwrite existing figures; use --out-dir for a verification copy.

Run from the repository root:  python scripts/gemma/04_pile_gemma_scope_figures.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_REL = 'results/gemma2_pile/gemma_scope_splicing'
EXP = REPO / EXP_REL
INPUTS = ['summary_by_layer.csv', 'summary_by_layer_source.csv', 'h_g1.csv', 'controls.csv', 'gates.json',
          'per_doc_metrics.csv']
FIGURES = ['fig1_reconstruction_by_layer.png', 'fig2_splicing_by_layer.png', 'fig3_h_g1_paired_difference.png',
           'fig4_source_variation.png']
INFO = 'figure_info.json'

# repository figure palette (scripts/geometry_of_truth/1[369]_*.py)
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
SAE_C, RAND_C = '#2a78d6', '#eb6834'
JITTER_SEED = 0          # fig3 strip-plot jitter only; no effect on any number
H_G1_TOL = 1e-9
SAVE_KW = dict(dpi=160, facecolor=SURFACE, metadata={'Software': None})  # no version string: byte-stable output


# ------------------------------------------------------------------------------------------------------ checks
def sha256_lf(path):
    """SHA-256 of the file with CRLF normalised to LF (the run wrote LF on Linux)."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def check_inputs():
    run_info_path = EXP / 'run_info.json'
    if not run_info_path.exists():
        sys.exit(f'STOP: {EXP_REL}/run_info.json missing - the folder does not hold a completed run.')
    stops = sorted(p.name for p in EXP.glob('*') if p.name.endswith(('_FAILED.json', '_MISMATCH.json')))
    if stops:
        sys.exit(f'STOP: failure files present, results must not be plotted: {stops}')
    run_info = json.loads(run_info_path.read_text(encoding='utf-8'))
    if run_info.get('status') != 'full pre-registered run':
        sys.exit(f"STOP: run_info status is {run_info.get('status')!r}, not the full pre-registered run.")
    recorded = run_info['output_sha256']
    hashes = {}
    for name in INPUTS:
        key = f'{EXP_REL}/{name}'
        got = sha256_lf(EXP / name)
        if recorded.get(key) != got:
            sys.exit(f'STOP: {key} SHA-256 {got} does not match run_info.json ({recorded.get(key)}).')
        hashes[key] = got
    gates = json.loads((EXP / 'gates.json').read_text(encoding='utf-8'))
    failed = [g for g in ('V1', 'V2', 'V3', 'V4', 'V5') if not gates.get(g, {}).get('pass', False)]
    if failed:
        sys.exit(f'STOP: validity gate(s) {failed} did not pass; results are not plotted.')
    return run_info, hashes


def doc_level_d(per_doc):
    """Per-document d = doc-mean dCE_SAE - doc-mean dCE_random (the H-G1 statistic before averaging)."""
    p = per_doc[per_doc.condition.isin(['sae_splice', 'norm_matched_random_error'])].copy()
    p['dce'] = (p.ce_new_sum - p.ce_orig_sum) / p.n_pred
    w = p.pivot_table(index=['layer', 'rank'], columns='condition', values='dce')
    return (w['sae_splice'] - w['norm_matched_random_error']).rename('d').reset_index()


def check_h_g1(d, h):
    for r in h.itertuples():
        m = d.loc[d.layer == r.layer, 'd']
        if len(m) != r.n_docs or abs(m.mean() - r.mean_d) > H_G1_TOL:
            sys.exit(f'STOP: layer {r.layer}: mean d from per_doc_metrics.csv ({m.mean():.12f}, n={len(m)}) '
                     f'does not equal h_g1.csv ({r.mean_d:.12f}, n={r.n_docs}).')


# ------------------------------------------------------------------------------------------------------ styling
def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=9, loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)


def layer_axis(ax, layers):
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'layer {L}' for L in layers], color=INK2, fontsize=8)
    ax.set_xlim(-0.6, len(layers) - 0.4)


def point_ci(ax, x, v, lo, hi, color, label=None, marker='o', hollow=False):
    ax.errorbar(x, v, yerr=[np.asarray(v) - np.asarray(lo), np.asarray(hi) - np.asarray(v)], fmt='none',
                ecolor=color, elinewidth=1.4, capsize=3, zorder=2)
    ax.scatter(x, v, s=42, marker=marker, color=SURFACE if hollow else color, edgecolors=color, linewidths=1.5,
               zorder=3, label=label)


def zero_axis(ax, hi, bounded):
    """Zero-based y-axis so between-layer gaps are shown at their true scale (fig1, fig2).
    Bounded metrics (EV, cosine, top-1 agreement, loss recovered) use the full 0-1 range."""
    ax.set_ylim(0, 1.0 if bounded else float(np.max(hi)) * 1.25)


def value_labels(ax, x, v, fmt, dx=0.0):
    """Value next to each point; at a zero baseline the CI bars can be shorter than the marker."""
    for xi, vi in zip(x, v):
        ax.annotate(format(vi, fmt), (xi + dx, vi), xytext=(7, 0), textcoords='offset points', fontsize=7,
                    color=INK2, va='center')


def header(fig, title, foot, top=1.0, bottom=0.06):
    # tight_layout already reserves space for the suptitle; `top` < 1 would add a second margin
    fig.suptitle(title, x=0.01, y=0.99, ha='left', va='top', color=INK, fontsize=11)
    fig.text(0.01, 0.01, foot, fontsize=7, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, bottom, 1, top))


FOOT_BASE = ('Gemma-2-2B (bf16) resid_post; Gemma Scope width-16k JumpReLU SAEs, one per layer; 512 Pile validation documents '
             '(254,144 non-BOS tokens).\nToken-weighted means; bars = 95 % document-bootstrap CIs (1,000 resamples, seed 0). '
             'Each layer has a different SAE and release L0, so layers are described side by side, not ranked.')
FOOT_ZERO = ('\nAll y-axes start at zero (bounded metrics span 0-1), so between-layer gaps are shown at true scale; CIs can be '
             'narrower than the marker. Exact values and CIs: README tables.')


# ------------------------------------------------------------------------------------------------------ figures
def fig1(s, layers, out):
    x = np.arange(len(layers))
    panels = [('explained_variance', 'Explained variance (higher is better)', True, '.3f'),
              ('cosine', 'Cosine(x, x̂), mean over tokens (higher is better)', True, '.3f'),
              ('relative_sq_error', 'Relative squared error\nΣ‖x−x̂‖² / Σ‖x‖² (lower is better)', False, '.3f'),
              ('l0', 'L0: active latents per token\n(reported, not targeted)', False, '.1f')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=SURFACE)
    for ax, (col, title, bounded, fmt) in zip(axes, panels):
        style(ax, title)
        point_ci(ax, x, s[col], s[f'{col}_ci95_lo'], s[f'{col}_ci95_hi'], SAE_C,
                 label='measured on the Pile sample' if col == 'l0' else None)
        value_labels(ax, x, s[col], fmt)
        zero_axis(ax, s[f'{col}_ci95_hi'], bounded)
        layer_axis(ax, layers)
    axes[3].scatter(x - 0.22, s.release_average_l0, s=42, marker='D', color=SURFACE, edgecolors=INK2, linewidths=1.3,
                    zorder=3, label='release average_l0 (SAE name)')
    axes[3].legend(fontsize=7, frameon=False, labelcolor=INK2, loc='lower right')
    header(fig, 'Q1 - Reconstruction quality of the Gemma Scope SAEs on The Pile (descriptive)', FOOT_BASE + FOOT_ZERO,
           bottom=0.09)
    fig.savefig(out / FIGURES[0], **SAVE_KW)
    plt.close(fig)


def fig2(s, layers, out):
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=SURFACE)
    ax = axes[0]
    style(ax, 'ΔCE when spliced (nats / prediction; lower is better)')
    point_ci(ax, x - 0.12, s.delta_ce_sae, s.delta_ce_sae_ci95_lo, s.delta_ce_sae_ci95_hi, SAE_C, 'SAE reconstruction')
    point_ci(ax, x + 0.12, s.delta_ce_random, s.delta_ce_random_ci95_lo, s.delta_ce_random_ci95_hi, RAND_C,
             'norm-matched random error', marker='s')
    value_labels(ax, x - 0.12, s.delta_ce_sae, '.3f')
    value_labels(ax, x + 0.12, s.delta_ce_random, '.3f')
    zero_axis(ax, s.delta_ce_sae_ci95_hi, False)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='upper left')
    for ax, col, title, bounded in (
            (axes[1], 'kl_sae', 'KL(original ‖ SAE-spliced)\n(nats / prediction; lower is better)', False),
            (axes[2], 'top1_agreement_sae', 'Top-1 agreement with original\n(SAE splice; higher is better)', True),
            (axes[3], 'loss_recovered', 'Loss recovered vs mean ablation\n(SAE splice; higher is better)', True)):
        style(ax, title)
        point_ci(ax, x, s[col], s[f'{col}_ci95_lo'], s[f'{col}_ci95_hi'], SAE_C)
        value_labels(ax, x, s[col], '.3f')
        zero_axis(ax, s[f'{col}_ci95_hi'], bounded)
    for ax in axes:
        layer_axis(ax, layers)
    abl = ', '.join(f'L{L}: {v:.2f}' for L, v in zip(layers, s.delta_ce_mean_ablation))
    foot = (FOOT_BASE + FOOT_ZERO + f'\nOriginal CE {s.ce_original.iloc[0]:.3f} nats. Mean-ablation ΔCE (loss-recovered '
            f'denominator; off scale): {abl} nats. Behavioural splicing only; not a representation or causal claim.')
    header(fig, 'Q2 - Behavioural change when each SAE reconstruction is spliced into Gemma-2-2B (descriptive)', foot,
           bottom=0.12)
    fig.savefig(out / FIGURES[1], **SAVE_KW)
    plt.close(fig)


def fig3(d, h, layers, out):
    rng = np.random.default_rng(JITTER_SEED)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.35, 1]})
    style(a, 'Per-document paired difference d = ΔCE_SAE − ΔCE_random\n(each dot one document; box = quartiles)',
          'd (nats / prediction)')
    for i, L in enumerate(layers):
        v = d.loc[d.layer == L, 'd'].to_numpy()
        a.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=6, color=SAE_C, alpha=0.35, linewidths=0, zorder=2)
        a.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True, zorder=3,
                  boxprops=dict(facecolor='none', edgecolor=INK, lw=1.1), medianprops=dict(color=INK, lw=1.4),
                  whiskerprops=dict(color=INK, lw=1), capprops=dict(color=INK, lw=1))
    a.axhline(0, color=INK2, lw=1, ls='--', zorder=1)
    layer_axis(a, layers)
    a.set_xticklabels([f'layer {L}\n{(d.loc[d.layer == L, "d"] > 0).mean():.0%} of documents d > 0' for L in layers],
                      color=INK2, fontsize=8)

    style(b, 'H-G1: mean d with 95 % CI (thick) and\nBonferroni 98.33 % CI (thin); supported if 95 % lower bound > 0',
          'mean d (nats / prediction)')
    x = np.arange(len(layers))
    b.errorbar(x, h.mean_d, yerr=[h.mean_d - h['ci98.33_lo'], h['ci98.33_hi'] - h.mean_d], fmt='none', ecolor=SAE_C,
               elinewidth=1, capsize=6, zorder=2)
    b.errorbar(x, h.mean_d, yerr=[h.mean_d - h.ci95_lo, h.ci95_hi - h.mean_d], fmt='none', ecolor=SAE_C,
               elinewidth=3.5, capsize=0, zorder=3)
    b.scatter(x, h.mean_d, s=44, color=SURFACE, edgecolors=SAE_C, linewidths=1.6, zorder=4)
    for i, r in enumerate(h.itertuples()):
        b.annotate(f'{r.mean_d:.3f}\n[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]\n{r.verdict}', (i, r.mean_d),
                   xytext=(14, 0), textcoords='offset points', fontsize=7, color=INK2, va='center')
    b.axhline(0, color=INK2, lw=1, ls='--', zorder=1)
    b.set_ylim(bottom=min(0, float(h['ci98.33_lo'].min())) - 0.01)
    layer_axis(b, layers)
    foot = ('Paired over the same 512 documents per layer; d is computed per document from doc-mean ΔCE '
            '(so mean d differs slightly from the token-weighted ΔCE gap in fig2).\nH-G1 is a directional replication '
            'of the GPT-2 Phase 0 finding, tested at each layer separately. Values between layers are not compared.')
    header(fig, 'H-G1 - SAE reconstruction error raises next-token loss more than a random error of the same norm',
           foot)
    fig.savefig(out / FIGURES[2], **SAVE_KW)
    plt.close(fig)


def fig4(src, s, layers, low_n_threshold, out):
    keep = src[~src.low_n]
    order = (keep[keep.layer == layers[0]].sort_values(['n_docs', 'pile_set_name'], ascending=[True, False])
             [['pile_set_name', 'n_docs']])
    names = order.pile_set_name.tolist()
    ylabels = [f'{n} (n={k})' for n, k in zip(order.pile_set_name, order.n_docs)]
    dropped = src[src.low_n & (src.layer == layers[0])]
    fig, axes = plt.subplots(2, len(layers), figsize=(14, 6.6), facecolor=SURFACE, sharey=True)
    for row, (col, what) in enumerate((('explained_variance', 'explained variance (higher is better)'),
                                       ('delta_ce_sae', 'ΔCE_SAE, nats / prediction (lower is better)'))):
        for j, L in enumerate(layers):
            ax = axes[row, j]
            g = keep[keep.layer == L].set_index('pile_set_name').reindex(names)
            style(ax, f'layer {L}: {what}')
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color=GRID, lw=0.8)
            y = np.arange(len(names))
            ax.errorbar(g[col], y, xerr=[g[col] - g[f'{col}_ci95_lo'], g[f'{col}_ci95_hi'] - g[col]], fmt='none',
                        ecolor=SAE_C, elinewidth=1.4, capsize=3, zorder=2)
            ax.scatter(g[col], y, s=36, color=SAE_C, zorder=3)
            overall = float(s.loc[s.layer == L, col].iloc[0])
            ax.axvline(overall, color=INK2, lw=1, ls='--', zorder=1, label='all 512 documents')
            ax.set_yticks(y)
            ax.set_yticklabels(ylabels, fontsize=8, color=INK2)
            if row == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='upper left')
    foot = (f'Only the {len(names)} Pile sources with ≥ {low_n_threshold} evaluation documents ({int(keep[keep.layer == layers[0]].n_docs.sum())} '
            f'documents) are shown; {len(dropped)} LOW-N sources ({int(dropped.n_docs.sum())} documents) are not '
            'interpreted (pre-registered rule).\nToken-weighted per source; bars = 95 % document-bootstrap CIs. Natural '
            'source mixture, not stratified; sources differ in length, content and original CE. Descriptive (Q3), no test.\n'
            'EV centres each subset on its own token mean, so between-source variance enters only the all-document '
            'denominator; the all-document EV can therefore exceed most per-source EVs.\n'
            'x-axes are ZOOMED and differ between panels (to show source-level CIs); descriptive only, not a ranking of '
            'sources or layers.')
    header(fig, 'Q3 - Variation across Pile sources of reconstruction and splicing metrics (descriptive)', foot,
           bottom=0.09)
    fig.savefig(out / FIGURES[3], **SAVE_KW)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', type=Path, default=EXP / 'figures',
                    help='output folder (default: the experiment figures/ folder); must not already contain the figures')
    out = ap.parse_args().out_dir
    existing = [f for f in FIGURES + [INFO] if (out / f).exists()]
    if existing:
        sys.exit(f'STOP: refusing to overwrite {existing} in {out}. Use --out-dir for a new folder.')

    run_info, hashes = check_inputs()
    prereg = json.loads((EXP / 'preregistration.json').read_text(encoding='utf-8'))
    layers = list(prereg['layers'])
    s = pd.read_csv(EXP / 'summary_by_layer.csv').set_index('layer').loc[layers].reset_index()
    src = pd.read_csv(EXP / 'summary_by_layer_source.csv')
    h = pd.read_csv(EXP / 'h_g1.csv').set_index('layer').loc[layers].reset_index()
    d = doc_level_d(pd.read_csv(EXP / 'per_doc_metrics.csv'))
    check_h_g1(d, h)

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig1(s, layers, out)
    fig2(s, layers, out)
    fig3(d, h, layers, out)
    fig4(src, s, layers, prereg['low_n_source_threshold_docs'], out)

    info = {'script': 'scripts/gemma/04_pile_gemma_scope_figures.py',
            'inputs_sha256_lf_normalised': hashes,
            'inputs_verified_against': f'{EXP_REL}/run_info.json output_sha256',
            'run_code_commit': run_info['code_provenance']['commit'],
            'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
            'figures_sha256': {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in FIGURES}}
    (out / INFO).write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    for f in FIGURES + [INFO]:
        print(out / f)


if __name__ == '__main__':
    main()
