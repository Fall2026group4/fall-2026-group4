#!/usr/bin/env python3
"""Validation-only evaluation of the Method A multi-layer extension.

Evaluates newly-trained Layers 3 and 9, then combines them with the existing
committed Layer-6 Method A results to produce the Tuesday 3-layer table.

Run from repository root after script 22 --train:
    python scripts/geometry_of_truth/23_method_a_multilayer_evaluation.py

Outputs:
    results/geometry_of_truth/method_a_multilayer/
        per_sae_new_layers.csv
        per_sae_layers_3_6_9.csv
        summary_layers_3_6_9.csv
        comparisons_layers_3_6_9.csv
        criteria_layers_3_6_9.json
        improvement_table_layers_3_6_9.csv
        eval_info.json

No GoT/RepE test rows are scored.
"""

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402

MLP_SEEDS = (42, 123, 456, 789, 2026)
T_CRIT_DF2_975 = 4.302652729911275


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ci95_t(values):
    """Two-sided 95% Student-t CI for the mean; intended for n=3 paired seeds."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if len(x) == 3:
        tcrit = T_CRIT_DF2_975
    else:
        # This script's primary paired analyses are n=3.
        # Keep a conservative fallback rather than adding a scipy dependency.
        tcrit = 1.96
    half = tcrit * sd / math.sqrt(len(x))
    return mean - half, mean + half


def sae_list(s22, layer):
    out = []
    for arch in s22.ARCHS:
        for seed in s22.SEEDS:
            out.append(
                (
                    "base",
                    arch,
                    seed,
                    uninterrupted_checkpoint(
                        layer, arch, seed
                    ),
                )
            )
            out.append(
                (
                    "owt_only",
                    arch,
                    seed,
                    s22.CKPT
                    / f"layer{layer}_{arch}_owt_only_seed{seed}"
                    / "finetuned_checkpoint.pt",
                )
            )
            out.append(
                (
                    "method_a",
                    arch,
                    seed,
                    s22.CKPT
                    / f"layer{layer}_{arch}_got_replay_seed{seed}"
                    / "finetuned_checkpoint.pt",
                )
            )
    return out


def evaluate_layer(layer, s22, g07, sp, ev, model, S, shared):
    device = "cuda"

    got = {}
    for domain in s22.DOMAINS:
        X, m = s22.load_domain(domain, layer)
        y = m.label.to_numpy()
        B = (m.half == "B").to_numpy()
        va = (m.split == "val").to_numpy()

        _, CB, logB = s22.P11.select_C(
            X, y, B, va
        )

        dB = (
            X[B & (y == 1)]
            .astype(np.float64)
            .mean(0)
            - X[B & (y == 0)]
            .astype(np.float64)
            .mean(0)
        )

        mlps = []
        for seed in MLP_SEEDS:
            g07.SEED = seed
            torch.manual_seed(seed)
            mlps.append(
                g07.train_probe(X[B], y[B], device)
            )

        got[domain] = {
            "X": X,
            "y": y,
            "B": B,
            "va": va,
            "logB": logB,
            "CB": CB,
            "dB": dB,
            "mlps": mlps,
            "o_log": logB.decision_function(X[va]),
            "o_dm": X[va]
            @ (dB / np.linalg.norm(dB)),
            "o_mlp": [
                g07.probe_scores(p, s_, X[va], device)
                for p, s_ in mlps
            ],
        }

    rd = shared["rd"]
    repe_probes = ev.fit_original_probes(
        rd["X"][layer],
        rd["y"],
        rd["split"],
    )

    o3 = load(
        "scripts/openwebtext/03_training_sae.py",
        f"o3_layer{layer}",
    )
    owt_val = o3.load_split(
        REPO / "data/activations/gpt2_small",
        "validation",
        layer,
    ).to(device)

    rows = []

    for variant, arch, seed, path in sae_list(
        s22, layer
    ):
        sae, _ = fm.load_sae(
            path, device, architecture=arch
        )

        r = {
            "layer": layer,
            "variant": variant,
            "architecture": arch,
            "seed": seed,
            "checkpoint": Path(path)
            .relative_to(REPO)
            .as_posix(),
            "checkpoint_sha256": sha(path),
        }

        for domain, d in got.items():
            X, y = d["X"], d["y"]
            va, B = d["va"], d["B"]

            Xh, l0_all, _ = fm.reconstruct(
                sae, X, device
            )
            Xh = Xh.numpy()

            rec = fm.reconstruction_metrics(
                X[va],
                Xh[va],
                l0_all[torch.as_tensor(va)],
            )

            yv = y[va]
            s_log = d["logB"].decision_function(
                Xh[va]
            )

            mlp_gaps = [
                fm.safe_auroc(yv, o)
                - fm.safe_auroc(
                    yv,
                    g07.probe_scores(
                        p, s_, Xh[va], device
                    ),
                )
                for (p, s_), o in zip(
                    d["mlps"], d["o_mlp"]
                )
            ]

            refit = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=d["CB"],
                    max_iter=3000,
                    random_state=42,
                ),
            ).fit(Xh[B], y[B])

            dBh = (
                Xh[B & (y == 1)]
                .astype(np.float64)
                .mean(0)
                - Xh[B & (y == 0)]
                .astype(np.float64)
                .mean(0)
            )

            pre = f"got_{domain}_"
            r.update(
                {
                    pre
                    + "indep_logistic_gap": (
                        fm.safe_auroc(yv, d["o_log"])
                        - fm.safe_auroc(yv, s_log)
                    ),
                    pre
                    + "indep_mlp_gap_mean": float(
                        np.mean(mlp_gaps)
                    ),
                    pre
                    + "indep_mlp_gap_sd": float(
                        np.std(
                            mlp_gaps, ddof=1
                        )
                    ),
                    pre
                    + "indep_diffmean_gap": (
                        fm.safe_auroc(
                            yv, d["o_dm"]
                        )
                        - fm.safe_auroc(
                            yv,
                            Xh[va]
                            @ (
                                d["dB"]
                                / np.linalg.norm(
                                    d["dB"]
                                )
                            ),
                        )
                    ),
                    pre
                    + "indep_logistic_score_corr": float(
                        np.corrcoef(
                            d["o_log"], s_log
                        )[0, 1]
                    ),
                    pre
                    + "DIAG_refit_on_recon_val_auroc": (
                        fm.safe_auroc(
                            yv,
                            refit.decision_function(
                                Xh[va]
                            ),
                        )
                    ),
                    pre
                    + "direction_cosine_B": (
                        fm.direction_fidelity(
                            d["dB"], dBh
                        )["direction_cosine"]
                    ),
                    pre
                    + "val_relative_sq_error": rec[
                        "relative_sq_error"
                    ],
                    pre + "val_cosine": rec[
                        "cosine"
                    ],
                    pre + "val_l0": rec["l0"],
                }
            )

        Xr = rd["X"][layer]
        Xrh, l0r, _ = fm.reconstruct(
            sae, Xr, device
        )
        Xrh = Xrh.numpy()

        rv = rd["split"] == "validation"
        rtr = rd["split"] == "train"
        yr = rd["y"]

        so = ev.scores(repe_probes, Xr)
        sr = ev.scores(repe_probes, Xrh)

        dr = (
            Xrh[rtr & (yr == 1)]
            .astype(np.float64)
            .mean(0)
            - Xrh[rtr & (yr == 0)]
            .astype(np.float64)
            .mean(0)
        )

        rrec = fm.reconstruction_metrics(
            Xr[rv],
            Xrh[rv],
            l0r[torch.as_tensor(rv)],
        )

        r.update(
            {
                "repe_logistic_gap": (
                    fm.safe_auroc(
                        yr[rv], so["logistic"][rv]
                    )
                    - fm.safe_auroc(
                        yr[rv], sr["logistic"][rv]
                    )
                ),
                "repe_diffmean_gap": (
                    fm.safe_auroc(
                        yr[rv], so["diffmean"][rv]
                    )
                    - fm.safe_auroc(
                        yr[rv], sr["diffmean"][rv]
                    )
                ),
                "repe_direction_cosine": (
                    fm.direction_fidelity(
                        repe_probes["raw_direction"],
                        dr,
                    )["direction_cosine"]
                ),
                "repe_val_relative_sq_error": rrec[
                    "relative_sq_error"
                ],
                "repe_val_l0": rrec["l0"],
            }
        )

        xh, l0, act = fm.reconstruct(
            sae, owt_val, device
        )
        rec = fm.reconstruction_metrics(
            owt_val.cpu(), xh, l0
        )

        r.update(
            {
                "owt_val_ev": rec[
                    "explained_variance"
                ],
                "owt_val_mse": rec["mse"],
                "owt_val_cosine": rec["cosine"],
                "owt_val_l0": rec["l0"],
                "owt_val_dead_fraction": (
                    fm.dead_fraction(act)
                ),
            }
        )

        fn = sp.sae_fn(sae)

        o = sp.owt_eval(
            S,
            shared["owt_seqs"],
            layer,
            fn,
            shared["owt_orig"],
        )
        gg = sp.got_eval(
            S,
            shared["got_v"],
            layer,
            fn,
            shared["got_orig"],
        )
        rp = sp.repe_eval(
            S,
            shared["repe_v"],
            layer,
            fn,
            shared["repe_orig"],
            "final",
        )

        r.update(
            {
                "splice_owt_delta_ce": o[
                    "delta_ce"
                ],
                "splice_owt_kl": o["kl_mean"],
                "splice_got_statement_kl": gg[
                    "statement_kl_mean"
                ],
                "splice_got_cities_pref_change": (
                    gg[
                        "cities_truth_pair_accuracy"
                    ]
                    - shared["cities_orig"]
                ),
                "splice_repe_final_kl": rp[
                    "final_kl_mean"
                ],
                "splice_repe_persona_shift_cosine": (
                    rp[
                        "persona_shift_cosine"
                    ]
                ),
            }
        )

        rows.append(r)
        print(
            f"layer {layer} {variant} "
            f"{arch} seed {seed} done",
            flush=True,
        )

    return pd.DataFrame(rows)


def make_comparisons_and_criteria(df, s22):
    idcols = {
        "layer",
        "variant",
        "architecture",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
    }
    metrics = [
        c for c in df.columns if c not in idcols
    ]

    comps = []
    criteria = {}
    improvement_rows = []

    for layer in (3, 6, 9):
        ldf = df[df.layer == layer]

        for arch in s22.ARCHS:
            adf = ldf[
                ldf.architecture == arch
            ]

            piv = {
                v: adf[
                    adf.variant == v
                ].set_index("seed")[metrics]
                for v in (
                    "base",
                    "owt_only",
                    "method_a",
                )
            }

            if not all(
                set(s22.SEEDS).issubset(
                    set(x.index)
                )
                for x in piv.values()
            ):
                raise ValueError(
                    f"Missing paired seeds: "
                    f"layer={layer} arch={arch}"
                )

            for a_, b_ in [
                ("method_a", "base"),
                ("method_a", "owt_only"),
                ("owt_only", "base"),
            ]:
                diff = piv[a_] - piv[b_]

                for metric in metrics:
                    vals = diff[
                        metric
                    ].reindex(s22.SEEDS)
                    lo, hi = ci95_t(vals.values)

                    comps.append(
                        {
                            "layer": layer,
                            "architecture": arch,
                            "comparison": (
                                f"{a_} - {b_}"
                            ),
                            "metric": metric,
                            "mean": vals.mean(),
                            "sd": vals.std(),
                            "ci95_low": lo,
                            "ci95_high": hi,
                            **{
                                f"seed{s}": vals.get(s)
                                for s in s22.SEEDS
                            },
                        }
                    )

            tests = {}

            for dom in (
                "geographic",
                "numerical",
            ):
                for pr in (
                    "indep_mlp_gap_mean",
                    "indep_logistic_gap",
                ):
                    for other in (
                        "base",
                        "owt_only",
                    ):
                        col = f"got_{dom}_{pr}"
                        d = (
                            piv["method_a"][col]
                            - piv[other][col]
                        ).reindex(s22.SEEDS)

                        tests[
                            f"{dom}:{pr}:"
                            f"method_a<{other}"
                        ] = {
                            "pass": bool(
                                (d < 0).all()
                                and -d.mean()
                                > 2 * d.std()
                            ),
                            "paired_diffs_by_seed": (
                                d.round(4).tolist()
                            ),
                            "mean_diff": float(
                                d.mean()
                            ),
                            "sd_diff": float(
                                d.std()
                            ),
                            "ci95": list(
                                ci95_t(d.values)
                            ),
                        }

            ma = piv["method_a"]
            base = piv["base"]

            guards = {
                "l0_matched_owt_val_within_5pct": bool(
                    (
                        (
                            ma["owt_val_l0"]
                            - base["owt_val_l0"]
                        ).abs()
                        / base["owt_val_l0"]
                        <= 0.05
                    ).all()
                ),
                "owt_splice_delta_ce_le_1.10x_base": bool(
                    (
                        ma["splice_owt_delta_ce"]
                        <= 1.10
                        * base[
                            "splice_owt_delta_ce"
                        ]
                    ).all()
                ),
                "owt_val_ev_ge_base_minus_0.005": bool(
                    (
                        ma["owt_val_ev"]
                        >= base["owt_val_ev"]
                        - 0.005
                    ).all()
                ),
                "cities_pref_ge_base_minus_0.018": bool(
                    ma[
                        "splice_got_cities_pref_change"
                    ].mean()
                    >= base[
                        "splice_got_cities_pref_change"
                    ].mean()
                    - 0.018
                ),
                "repe_diffmean_gap_le_base_plus_0.02": bool(
                    ma[
                        "repe_diffmean_gap"
                    ].mean()
                    <= base[
                        "repe_diffmean_gap"
                    ].mean()
                    + 0.02
                ),
            }

            primary = (
                tests[
                    "geographic:"
                    "indep_mlp_gap_mean:"
                    "method_a<base"
                ]["pass"]
                and tests[
                    "geographic:"
                    "indep_mlp_gap_mean:"
                    "method_a<owt_only"
                ]["pass"]
            )

            success = bool(
                primary and all(guards.values())
            )

            criteria[
                f"layer{layer}_{arch}"
            ] = {
                "tests": tests,
                "primary_pass": primary,
                "guards": guards,
                "SUCCESS": success,
            }

            metric = (
                "got_geographic_"
                "indep_mlp_gap_mean"
            )
            base_vals = (
                piv["base"][metric]
                .reindex(s22.SEEDS)
            )
            ma_vals = (
                piv["method_a"][metric]
                .reindex(s22.SEEDS)
            )
            delta = ma_vals - base_vals
            lo, hi = ci95_t(delta.values)

            improvement_rows.append(
                {
                    "layer": layer,
                    "architecture": arch,
                    "seeds": "42/1042/2042",
                    "base_gap_mean": (
                        base_vals.mean()
                    ),
                    "base_gap_sd": (
                        base_vals.std()
                    ),
                    "method_a_gap_mean": (
                        ma_vals.mean()
                    ),
                    "method_a_gap_sd": (
                        ma_vals.std()
                    ),
                    "paired_delta_method_a_minus_base": (
                        delta.mean()
                    ),
                    "paired_delta_sd": (
                        delta.std()
                    ),
                    "paired_delta_ci95_low": lo,
                    "paired_delta_ci95_high": hi,
                    "primary_pass": primary,
                    "all_guards_pass": all(
                        guards.values()
                    ),
                    "SUCCESS": success,
                }
            )

    return (
        pd.DataFrame(comps),
        criteria,
        pd.DataFrame(improvement_rows),
    )


def main():
    s22 = load(
        "scripts/geometry_of_truth/"
        "22_method_a_multilayer_finetune.py",
        "s22",
    )
    OUT = s22.OUT

    final_table = (
        OUT
        / "improvement_table_layers_3_6_9.csv"
    )
    if final_table.exists():
        raise SystemExit(
            f"{final_table} exists; refusing "
            "to overwrite."
        )

    # Verify no post-preregistration changes.
    hashes = json.loads(
        (OUT / "pretraining_hashes.json")
        .read_text()
    )["sha256"]
    bad = [
        p
        for p, h in hashes.items()
        if sha(REPO / p) != h
    ]
    if bad:
        raise SystemExit(
            f"files changed since "
            f"pre-registration: {bad}"
        )

    g07 = load(
        "scripts/geometry_of_truth/"
        "07_independent_truth_probe.py",
        "g07",
    )
    sp = load(
        "scripts/shared/"
        "spliced_reconstruction_evaluation.py",
        "splice",
    )
    ev = load(
        "scripts/shared/"
        "all_layer_frozen_probe_evaluation.py",
        "ev",
    )

    device = "cuda"
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2-small", device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    S = sp.Splicer(model, device)

    owt_seqs, _ = sp.load_owt_validation()
    got_v = sp.load_got_validation(model)
    repe_v = sp.load_repe_validation(model)

    owt_orig = sp.owt_original(
        S, owt_seqs
    )
    got_orig = sp.got_original(
        S, got_v
    )

    toks, lens = sp.pad(
        list(repe_v.tokens), 50256
    )
    repe_orig = S.logprobs(
        toks, lens
    )[
        torch.arange(
            len(repe_v), device=device
        ),
        (lens - 1).to(device),
    ]

    cities_orig = sp.got_eval(
        S,
        got_v,
        0,
        lambda x: x.clone(),
        got_orig,
    )["cities_truth_pair_accuracy"]

    rd = ev.load_repe()[0]["honesty"]

    shared = {
        "rd": rd,
        "owt_seqs": owt_seqs,
        "got_v": got_v,
        "repe_v": repe_v,
        "owt_orig": owt_orig,
        "got_orig": got_orig,
        "repe_orig": repe_orig,
        "cities_orig": cities_orig,
    }

    new_frames = []
    for layer in s22.LAYERS:
        new_frames.append(
            evaluate_layer(
                layer,
                s22,
                g07,
                sp,
                ev,
                model,
                S,
                shared,
            )
        )

    new_df = pd.concat(
        new_frames, ignore_index=True
    )
    new_df.to_csv(
        OUT / "per_sae_new_layers.csv",
        index=False,
    )

    # Bring in the committed Layer-6 Method A results.
    layer6_path = (
        REPO
        / "results/geometry_of_truth/"
        "method_a_layer6/per_sae.csv"
    )
    if not layer6_path.exists():
        raise SystemExit(
            f"Missing committed layer-6 "
            f"evidence: {layer6_path}"
        )

    l6 = pd.read_csv(layer6_path)
    l6 = l6[
        l6["variant"].isin(
            ["base", "owt_only", "method_a"]
        )
        & l6["architecture"].isin(
            s22.ARCHS
        )
        & l6["seed"].isin(
            s22.SEEDS
        )
    ].copy()
    l6.insert(0, "layer", 6)

    all_df = pd.concat(
        [new_df, l6],
        ignore_index=True,
        sort=False,
    )

    all_df.to_csv(
        OUT / "per_sae_layers_3_6_9.csv",
        index=False,
    )

    idcols = {
        "layer",
        "variant",
        "architecture",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
    }
    metrics = [
        c
        for c in all_df.columns
        if c not in idcols
    ]

    summ = (
        all_df.groupby(
            ["layer", "architecture", "variant"]
        )[metrics]
        .agg(["mean", "std"])
    )
    summ.columns = [
        f"{a}_{'sd' if b == 'std' else b}"
        for a, b in summ.columns
    ]
    summ.reset_index().to_csv(
        OUT / "summary_layers_3_6_9.csv",
        index=False,
    )

    comps, crit, table = (
        make_comparisons_and_criteria(
            all_df, s22
        )
    )

    comps.to_csv(
        OUT
        / "comparisons_layers_3_6_9.csv",
        index=False,
    )
    (
        OUT
        / "criteria_layers_3_6_9.json"
    ).write_text(
        json.dumps(crit, indent=2)
    )
    table.to_csv(
        final_table, index=False
    )

    (OUT / "eval_info.json").write_text(
        json.dumps(
            {
                "script_sha256": sha(__file__),
                "data": "validation only",
                "test_rows_used": False,
                "new_layers": list(
                    s22.LAYERS
                ),
                "layer6_source": (
                    layer6_path
                    .relative_to(REPO)
                    .as_posix()
                ),
                "mlp_probe_seeds": list(
                    MLP_SEEDS
                ),
                "paired_ci": (
                    "two-sided 95% "
                    "Student-t CI across "
                    "3 paired training seeds; "
                    "df=2"
                ),
                "gpu_peak_allocated_mb": (
                    round(
                        torch.cuda
                        .max_memory_allocated()
                        / 2**20,
                        1,
                    )
                ),
                "runtime_seconds": round(
                    time.time() - t0, 1
                ),
            },
            indent=2,
        )
    )

    print(
        "\nTUESDAY IMPROVEMENT TABLE\n"
    )
    print(
        table.to_string(index=False)
    )


if __name__ == "__main__":
    main()
