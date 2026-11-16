"""Dead-latent audit for the GPT-2 base (old) and R2 SAEs, built only from stored evaluation artifacts.

No model, SAE or activation is loaded. Definition (src/evaluation/faithfulness_metrics.py:reconstruct and
dead_fraction; scripts/shared/sae_ladder_evaluation.py line ~203): a latent is dead if its activation z is never > 0
on any token of the layer's OWT validation cache (non-BOS, 12,700 tokens); dead_fraction = dead / width.
Both source tables were produced by that same code path on that same corpus.

Run from the repository root:  python results/shared/dead_latent_audit/build_dead_latent_audit.py
Refuses to overwrite existing outputs.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/shared/dead_latent_audit'
BASE_SRC = 'results/openwebtext/spliced_evaluation/per_run.csv'
R2_SRC = 'results/openwebtext/r2_scaled_saes/ladder/ladder_layer{L}_r2_resjb.csv'
OLD_LADDER_SRC = 'results/openwebtext/r2_scaled_saes/ladder/ladder_layer{L}_old.csv'
MONITOR_SRC = 'results/openwebtext/r2_scaled_saes/training_monitor.csv'
APPENDIX_SRC = 'research_paper/Latex/sections/12_appendix.tex'
DEFINITION = 'zero activations (z>0 never) over OWT validation cache, non-BOS tokens; dead/width'
BASE_WIDTH, TARGET_L0 = 3072, 128
R2_LAYERS = (1, 8, 11)
R2_TRAINED = ('topk', 'batchtopk', 'relu', 'topk_w3072')
R2_ARCH = {'topk': 'topk', 'batchtopk': 'batchtopk', 'relu': 'relu', 'topk_w3072': 'topk',
           'topk_frozen_decoder': 'topk'}


def counts(frac, width):
    dead = frac * width
    d = int(round(dead))
    if abs(dead - d) > 1e-2:
        raise ValueError(f'dead_fraction {frac} x width {width} = {dead} is not an integer count')
    return d


def main():
    outs = [OUT / 'dead_latents_by_sae.csv', OUT / 'summary_by_family_layer.csv', OUT / 'checks.txt']
    if any(p.exists() for p in outs):
        sys.exit(f'refusing to overwrite existing outputs: {[str(p.relative_to(REPO)) for p in outs if p.exists()]}')
    rows, log = [], []

    # ---- base (old) SAEs: 12 layers x 3 architectures x 3 seeds, width 3072
    b = pd.read_csv(REPO / BASE_SRC)
    b = b[b['intervention'] == 'sae_reconstruction']
    for _, r in b.iterrows():
        d = counts(r['owt_val_recon__dead_fraction'], BASE_WIDTH)
        notes = [] if bool(r['matched_10pct']) else ['L0 outside 128 +/-10% on OWT validation (unmatched; excluded from rankings)']
        if isinstance(r['sparsity_flag'], str) and r['sparsity_flag']:
            notes.append(f"sparsity_flag={r['sparsity_flag']}")
        rows.append(dict(run_id=f"base|{r['architecture']}|L{r['layer']}|s{r['seed']}", phase='base', family=f"base_{r['family']}",
                         category='trained', architecture=r['architecture'], layer=int(r['layer']), seed=int(r['seed']),
                         width=BASE_WIDTH, target_l0=TARGET_L0, measured_l0=r['owt_val_recon__l0'],
                         evaluation_dataset='OWT validation cache (GPT-2 hook_resid_pre, non-BOS)', n_tokens=int(r['owt_val_recon__n']),
                         alive_latents=BASE_WIDTH - d, dead_latents=d, dead_fraction=d / BASE_WIDTH,
                         stored_dead_fraction=r['owt_val_recon__dead_fraction'], definition=DEFINITION,
                         source_file=BASE_SRC, evidence_status='validation', checkpoint_sha256=r['checkpoint_sha256'],
                         notes='; '.join(notes)))

    # ---- R2 SAEs (+ frozen-decoder random baseline, + public res-jb reference) at layers 1, 8, 11
    mon = pd.read_csv(REPO / MONITOR_SRC).dropna(subset=['dead'])
    mon_last = mon.sort_values('step').groupby(['layer', 'name', 'seed']).tail(1).set_index(['layer', 'name', 'seed'])
    for L in R2_LAYERS:
        src = R2_SRC.format(L=L)
        t = pd.read_csv(REPO / src)
        for _, r in t.iterrows():
            w = int(r['d_sae'])
            d = counts(r['owt_dead_fraction'], w)
            name, fam = r['name'], r['family']
            notes = []
            if fam == 'r2':
                phase = 'r2'
                category = 'trained' if name in R2_TRAINED else 'random_baseline'
                arch = R2_ARCH[name]
                if category == 'random_baseline':
                    notes.append('decoder frozen at random init; encoder trained (R2 comparison B); not a trained-family member')
                key = (L, name, int(r['seed']))
                if key in mon_last.index:
                    notes.append(f"training-monitor dead (shard 79, different corpus) at end = {mon_last.loc[key, 'dead']:.4f}")
                seed = int(r['seed'])
            else:
                phase, category, arch, seed = 'public_reference', 'public_reference', str(name), r['seed']
                notes.append('public gpt2-small-res-jb SAE (E2 reference); not trained in house')
            l0 = r['owt_l0']
            if not (0.9 * TARGET_L0 <= l0 <= 1.1 * TARGET_L0):
                notes.append('L0 outside 128 +/-10% on OWT validation')
            rows.append(dict(run_id=f"{phase}|{name}|L{L}|s{seed}", phase=phase, family=f"{fam}_{name}" if fam == 'r2' else str(fam),
                             category=category, architecture=arch, layer=L, seed=seed, width=w,
                             target_l0=TARGET_L0 if phase == 'r2' else np.nan, measured_l0=l0,
                             evaluation_dataset='OWT validation cache (GPT-2 hook_resid_pre, non-BOS)', n_tokens=int(r['owt_n']),
                             alive_latents=w - d, dead_latents=d, dead_fraction=d / w, stored_dead_fraction=r['owt_dead_fraction'],
                             definition=DEFINITION, source_file=src, evidence_status='validation',
                             checkpoint_sha256=r['checkpoint_sha256'], notes='; '.join(notes)))

    df = pd.DataFrame(rows)

    # ---- checks
    ok = True
    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        log.append(('PASS ' if cond else 'FAIL ') + msg)
    check((df['alive_latents'] + df['dead_latents'] == df['width']).all(), 'alive + dead = width for every row')
    check(np.allclose(df['dead_fraction'], df['dead_latents'] / df['width']), 'dead_fraction = dead/width for every row')
    check(np.allclose(df['dead_fraction'], df['stored_dead_fraction'], atol=1e-6), 'recomputed dead_fraction equals stored value (|diff|<=1e-6)')
    check(not df['run_id'].duplicated().any(), 'no duplicate run identifiers')
    check(not df['checkpoint_sha256'].duplicated().any(), 'no duplicate checkpoint hashes')
    check((df['n_tokens'] == 12700).all(), 'every row evaluated on 12,700 OWT validation tokens')
    base = df[df.phase == 'base']
    exp_base = {(L, a, s) for L in range(12) for a in ('relu', 'topk', 'jumprelu') for s in (42, 1042, 2042)}
    got_base = set(zip(base.layer, base.architecture, base.seed))
    check(got_base == exp_base, f'base: all 108 layer x architecture x seed combinations present (found {len(got_base)}; missing {sorted(exp_base - got_base)})')
    r2 = df[df.phase == 'r2']
    exp_r2 = {(L, n, s) for L in R2_LAYERS for n in R2_TRAINED for s in (42, 1042, 2042)} | {(L, 'topk_frozen_decoder', 42) for L in R2_LAYERS}
    got_r2 = set(zip(r2.layer, r2.family.str.replace('r2_', '', n=1), r2.seed))
    check(got_r2 == exp_r2, f'R2: all 39 layer x family x seed combinations present (found {len(got_r2)}; missing {sorted(exp_r2 - got_r2)})')
    # cross-check base vs the R2 ladder's re-evaluation of the old SAEs (same checkpoints, same harness family)
    for L in R2_LAYERS:
        o = pd.read_csv(REPO / OLD_LADDER_SRC.format(L=L))
        m = o.merge(base[base.layer == L], left_on=['name', 'seed'], right_on=['architecture', 'seed'], how='left')
        check(len(m) == 9 and m['dead_fraction'].notna().all(), f'L{L}: 9 old-ladder rows matched to base rows')
        check((m['checkpoint_sha256_x'] == m['checkpoint_sha256_y']).all(), f'L{L}: old-ladder checkpoint hashes equal base hashes')
        check(np.allclose(m['owt_dead_fraction'], m['dead_fraction'], atol=1e-6), f'L{L}: old-ladder dead fraction equals base dead fraction')
    # manuscript appendix ranges (parsed from 12_appendix.tex lines 24-25; from results/shared/m0_within_layer_reanalysis,
    # live_latents_mean in architecture_by_layer.csv = per-layer seed MEAN over L0-matched runs, rounded)
    matched = base[~base.notes.str.contains('unmatched')]
    tex = ' '.join((REPO / APPENDIX_SRC).read_text(encoding='utf-8').split()).replace('{,}', '')
    m = re.search(r'ReLU keeps (\d+)--(\d+) alive, TopK (\d+)--(\d+) and JumpReLU (\d+)--(\d+)', tex)
    check(m is not None, f'appendix alive-latent sentence found in {APPENDIX_SRC}')
    v = [int(x) for x in m.groups()] if m else [0] * 6
    for a, (lo, hi) in {'relu': (v[0], v[1]), 'topk': (v[2], v[3]), 'jumprelu': (v[4], v[5])}.items():
        s = base[base.architecture == a]
        lm = matched[matched.architecture == a].groupby('layer').alive_latents.mean()
        rlo, rhi = int(round(lm.min())), int(round(lm.max()))
        log.append(f'INFO appendix {a} alive {lo}-{hi}; per-run range (all runs) {s.alive_latents.min()}-{s.alive_latents.max()}; '
                   f'per-layer seed-mean range (L0-matched, M0 method) {lm.min():.1f}-{lm.max():.1f}')
        check((rlo, rhi) == (lo, hi), f'appendix {a} range {lo}-{hi} reproduced by per-layer seed means (got {rlo}-{rhi})')

    # ---- summary by phase x family x layer
    g = df.groupby(['phase', 'category', 'family', 'architecture', 'layer', 'width'], dropna=False)
    summ = g.agg(n_runs=('dead_fraction', 'size'), dead_fraction_mean=('dead_fraction', 'mean'),
                 dead_fraction_sd=('dead_fraction', lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                 dead_fraction_min=('dead_fraction', 'min'), dead_fraction_max=('dead_fraction', 'max'),
                 alive_latents_mean=('alive_latents', 'mean'), measured_l0_mean=('measured_l0', 'mean')).reset_index()

    cols = ['run_id', 'phase', 'family', 'category', 'architecture', 'layer', 'seed', 'width', 'target_l0', 'measured_l0',
            'evaluation_dataset', 'n_tokens', 'alive_latents', 'dead_latents', 'dead_fraction', 'definition', 'source_file',
            'evidence_status', 'checkpoint_sha256', 'notes']
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values(['phase', 'family', 'layer', 'seed']).to_csv(outs[0], index=False)
    summ.to_csv(outs[1], index=False)
    outs[2].write_text('\n'.join(log) + '\n', encoding='utf-8')
    print('\n'.join(log))
    if not ok:
        print('NOTE: one or more checks FAILED; see checks.txt (outputs are written so the discrepancy can be inspected)')


if __name__ == '__main__':
    main()
