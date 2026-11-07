#!/usr/bin/env python3
"""Tasks F and G: evaluate all nine layer-6 SAE variants on identical held-out data.

Variants: {relu, topk, jumprelu} x {pretrained (= OWT-trained base), reconstruction_only, truth_aware}.
Held-out sets (identical for every variant):
  OWT  : test split, 12 700 non-BOS tokens (reconstruction metrics)
  GoT  : grouped test split, final token (frozen LOGISTIC probe trained on original train activations
         with the Stage-3 validation-selected C - the probe the truth-aware loss was trained against;
         MLP-probe results are merged from results/geometry_of_truth/final_test_evaluation/test_summary.csv)
  RepE : validation AND test splits, final token (frozen original honesty direction; also a
         direction re-estimated on reconstructions; direction cosine; reconstruction metrics)
  Causal: E and empirical p of the reconstruction direction and top-1 feature from causal_summary.csv

Run from repository root: python scripts/shared/finetuning_heldout_eval.py [--out-root DIR]
(Formerly scripts/professor_review/finetuning_repe_got_eval.py.) Cross-dataset, so it lives in scripts/shared/;
every output goes to its own dataset directory:
  results/geometry_of_truth/got_faithfulness_frozen_probes.csv
  results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv
  results/repe/representation_preservation_val_test.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L10 = Path('results/geometry_of_truth/layer10_frozen_probe')
CAUSAL = Path('results/repe/steering/causal_random_controls')
GOT_FAITH = Path('results/geometry_of_truth/got_faithfulness_frozen_probes.csv')
FT_CMP = Path('results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv')
REPE_REP = Path('results/repe/representation_preservation_val_test.csv')
OUT_ROOT = Path('.')


def out(rel):
    p = OUT_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
ARCHS = ('relu', 'topk', 'jumprelu')
VARIANTS = ('pretrained', 'reconstruction_only', 'truth_aware')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ckpt(arch, variant):
    if variant == 'pretrained':
        return Path(f'results/openwebtext/convergence/layer_6/{arch}_l0_128_resume_5000_to_15000/checkpoint_step_15000.pt')
    return Path(f'results/geometry_of_truth/sae_finetuning/layer_6_{arch}/{variant}/finetuned_checkpoint.pt')


def recon_metrics(x, r, z):
    err = (x - r).square()
    return dict(mse=float(err.mean()), cosine=float(F.cosine_similarity(x, r, dim=-1).mean()),
                explained_variance=float(1 - err.sum() / (x - x.mean(0)).square().sum()),
                mean_l0=float((z > 0).sum(-1).float().mean()))


@torch.inference_mode()
def run_sae(sae, X, device):
    rs, zs = [], []
    for b in X.split(512):
        r, z, _ = sae(b.to(device))
        rs.append(r.cpu())
        zs.append(z.cpu())
    return torch.cat(rs), torch.cat(zs)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = load('scripts/openwebtext/03_training_sae.py', 'owt_train')
    owt_eval = load('scripts/openwebtext/10_reconstruction_evaluation.py', 'owt_eval')
    got9 = load('scripts/geometry_of_truth/09_final_test_evaluation.py', 'got9')

    saes = {}
    for arch in ARCHS:
        for v in VARIANTS:
            sae = base.SAE(768, 3072, arch, 128, temperature=0.1).to(device)
            sae.load_state_dict(torch.load(ckpt(arch, v), map_location='cpu', weights_only=False)['model_state_dict'])
            sae.eval()
            for p in sae.parameters():
                p.requires_grad_(False)
            saes[(arch, v)] = sae

    rows = {k: dict(architecture=k[0], variant=k[1]) for k in saes}

    # ---------------- OWT test
    for k, sae in saes.items():
        m = owt_eval.evaluate(sae, 6, device)
        rows[k].update({f'owt_test_{a}': b for a, b in m.items()})

    # ---------------- GoT test, frozen logistic probe (Stage-3 C), identical rows for all variants
    choices = pd.read_csv('results/geometry_of_truth/original_truth_probes/validation_best_C_by_layer.csv')
    got_rows = []
    for domain, names in {'geographic': ('cities', 'neg_cities'), 'numerical': ('larger_than', 'smaller_than')}.items():
        X, y, split = got9.load_domain(names)
        tr, te = split == 'train', split == 'test'
        C = float(choices[(choices.domain == domain) & (choices.layer == 6)].iloc[0]['C'])
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=42)).fit(X[tr], y[tr])
        auc_o = roc_auc_score(y[te], probe.decision_function(X[te]))
        got_rows.append(dict(dataset='GoT', domain=domain, layer=6, sae='none (original activations)',
                             variant='original', probe='frozen logistic (Stage-3 C)', n_test=int(te.sum()),
                             auroc=auc_o, auroc_gap=0.0))
        Xt = torch.tensor(X[te], dtype=torch.float32)
        for k, sae in saes.items():
            r, z = run_sae(sae, Xt, device)
            auc = roc_auc_score(y[te], probe.decision_function(r.numpy()))
            rm = recon_metrics(Xt, r, z)
            rows[k].update({f'got_{domain}_test_frozen_logistic_auroc': auc,
                            f'got_{domain}_test_frozen_logistic_gap': auc_o - auc,
                            f'got_{domain}_test_mse': rm['mse'], f'got_{domain}_test_l0': rm['mean_l0'],
                            f'got_{domain}_test_explained_variance': rm['explained_variance']})
            got_rows.append(dict(dataset='GoT', domain=domain, layer=6, sae=f'in-house {k[0]} (3072)', variant=k[1],
                                 probe='frozen logistic (Stage-3 C)', n_test=int(te.sum()), auroc=auc,
                                 auroc_gap=auc_o - auc, **rm))
    mlp = pd.read_csv('results/geometry_of_truth/final_test_evaluation/test_summary.csv')
    for _, r in mlp.iterrows():
        k = (r['architecture'], r['variant'])
        rows[k][f"got_{r['domain']}_test_mlp_auroc_mean5seeds"] = r['mean_auroc']
        rows[k][f"got_{r['domain']}_test_mlp_gap_mean5seeds"] = r['mean_gap']
        got_rows.append(dict(dataset='GoT', domain=r['domain'], layer=6, sae=f"in-house {r['architecture']} (3072)",
                             variant=r['variant'], probe='frozen MLP 768-64-1, mean of 5 seeds (existing result)',
                             n_test=r['test_examples'], auroc=r['mean_auroc'], auroc_gap=r['mean_gap'],
                             auroc_sd=r['std_auroc'], mean_l0=r['mean_l0']))
    l10 = pd.read_csv(L10 / 'got_layer10_probe_comparison.csv')
    for _, r in l10.iterrows():
        got_rows.append(dict(dataset='GoT cities (legacy split)', domain='cities', layer=10, sae=r['sae'],
                             variant='public pretrained (SAELens)', probe=r['probe'], split_seed=r['split_seed'],
                             n_test=r['n_test'], auroc=r['auroc'], accuracy=r['accuracy'],
                             auroc_gap=r['auroc_gap_vs_original'], gap_ci95_low=r.get('gap_ci95_low'),
                             gap_ci95_high=r.get('gap_ci95_high')))
    pd.DataFrame(got_rows).to_csv(out(GOT_FAITH), index=False)

    # ---------------- RepE
    acts = torch.load('data/repe/activations/gpt2_small/repe_final_token.pt', map_location='cpu', weights_only=True)
    X6 = acts['activations'][6].float()
    lab = acts['persona_labels'].numpy()
    spl = np.array(acts['splits'])
    ids = np.array(acts['statement_ids'])
    tr = spl == 'train'
    d = (X6[tr & (lab == 1)].mean(0) - X6[tr & (lab == 0)].mean(0)).numpy()
    d_hat = d / np.linalg.norm(d)
    cw = torch.load('data/repe/activations/cross_wording/repe_final_token.pt', map_location='cpu', weights_only=True)
    Xcw = cw['activations'][6].float()
    lab_cw = cw['persona_labels'].numpy()

    def pair_acc(scores, mask, labels=lab, idv=ids):
        p = pd.DataFrame(dict(i=idv[mask], l=labels[mask], s=scores[mask])).pivot(index='i', columns='l', values='s')
        return float((p[1] > p[0]).mean())

    repe = []
    for name, R, Z in [('original_gpt2', X6, None)] + [(f'{a}_{v}',) + run_sae(saes[(a, v)], X6, device)
                                                     for a in ARCHS for v in VARIANTS]:
        Rn = R.numpy()
        d_re = Rn[tr & (lab == 1)].mean(0) - Rn[tr & (lab == 0)].mean(0)
        row = dict(model=name, layer=6, direction_cosine_vs_original=float(d_re @ d / np.linalg.norm(d_re) / np.linalg.norm(d)),
                   direction_norm_ratio=float(np.linalg.norm(d_re) / np.linalg.norm(d)))
        for s in ('validation', 'test'):
            m = spl == s
            sc = Rn @ d_hat
            row[f'{s}_auroc_frozen_direction'] = roc_auc_score(lab[m], sc[m])
            row[f'{s}_pair_accuracy_frozen_direction'] = pair_acc(sc, m)
            sc2 = Rn @ (d_re / np.linalg.norm(d_re))
            row[f'{s}_auroc_reestimated_direction'] = roc_auc_score(lab[m], sc2[m])
            row[f'{s}_n_prompts'] = int(m.sum())
            if Z is not None:
                rm = recon_metrics(X6[m], R[m], Z[m])
                row.update({f'{s}_{k}': v for k, v in rm.items()})
        if Z is None:
            row['cross_wording_auroc_frozen_direction'] = roc_auc_score(lab_cw, Xcw.numpy() @ d_hat)
        else:
            a, v = name.split('_', 1)
            Rcw, _ = run_sae(saes[(a, v)], Xcw, device)
            row['cross_wording_auroc_frozen_direction'] = roc_auc_score(lab_cw, Rcw.numpy() @ d_hat)
        repe.append(row)
    rdf = pd.DataFrame(repe)
    orig = rdf.iloc[0]
    for s in ('validation', 'test', 'cross_wording'):
        col = f'{s}_auroc_frozen_direction'
        rdf[f'{s}_auroc_gap'] = orig[col] - rdf[col]
    rdf.to_csv(out(REPE_REP), index=False)
    old = pd.read_csv('results/repe/representation_preservation.csv').set_index('model')['validation_auroc']
    oldcw = pd.read_csv('results/repe/cross_wording/evaluation.csv').set_index('model')['auroc']
    oldcos = pd.read_csv('results/repe/direction_fidelity.csv').set_index('model')['direction_cosine']
    chk = rdf.set_index('model')
    print('max |recomputed - saved| validation AUROC:', float((chk['validation_auroc_frozen_direction'] - old).abs().max()))
    print('max |recomputed - saved| cross-wording AUROC:', float((chk['cross_wording_auroc_frozen_direction'] - oldcw).abs().max()))
    print('max |recomputed - saved| direction cosine:', float((chk.loc[oldcos.index, 'direction_cosine_vs_original'] - oldcos).abs().max()))
    for (a, v), row in rows.items():
        r = rdf[rdf.model == f'{a}_{v}'].iloc[0]
        row.update(repe_test_auroc=r['test_auroc_frozen_direction'], repe_test_auroc_gap=r['test_auroc_gap'],
                   repe_val_auroc=r['validation_auroc_frozen_direction'],
                   repe_cross_wording_auroc=r['cross_wording_auroc_frozen_direction'],
                   repe_direction_cosine=r['direction_cosine_vs_original'], repe_test_mse=r['test_mse'],
                   repe_test_l0=r['test_mean_l0'])

    # ---------------- causal
    cs = pd.read_csv(CAUSAL / 'causal_summary.csv')
    for (a, v), row in rows.items():
        for kind, tag in (('sae_reconstruction_direction', 'recon_dir'), ('sae_top1_feature', 'top1_feature'),
                          ('sae_top10_features', 'top10_features')):
            r = cs[(cs.sae_architecture == a) & (cs.sae_variant == v) & (cs.intervention == kind)].iloc[0]
            row[f'causal_{tag}_E'] = r['effect_E']
            row[f'causal_{tag}_p_vs_gaussian'] = r['vs_gaussian_p_empirical']
            if tag != 'recon_dir':
                row[f'causal_{tag}_p_vs_random_features'] = r['vs_random_features_p_empirical']
    fdf = pd.DataFrame(rows.values())
    fdf.to_csv(out(FT_CMP), index=False)
    print(fdf.round(4).T.to_string())


if __name__ == '__main__':
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--out-root', type=Path, default=Path('.'))
    OUT_ROOT = _ap.parse_args().out_root
    main()
