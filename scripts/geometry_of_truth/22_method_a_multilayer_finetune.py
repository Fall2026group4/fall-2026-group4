#!/usr/bin/env python3
"""Method A multi-layer extension: layers 3 and 9.

This script extends the already-committed layer-6 Method A protocol without
modifying scripts 14/15 or their historical evidence.

Protocol held fixed from layer 6:
- GoT train groups A only for SAE fine-tuning.
- 64 geographic-A + 64 numerical-A + 128 OWT-train rows per Method-A step.
- Method A loss: 0.5*MSE_GoT + 0.5*MSE_OWT.
- OWT-only control: 256 OWT-train rows.
- Adam lr=1e-5, 500 steps, grad clip 10, decoder renormalisation.
- Seeds 42/1042/2042.
- ReLU/JumpReLU continue their base sparsity coefficient/controller.
- TopK has fixed L0 and no sparsity penalty.
- Validation/test are never used for training or checkpoint selection.

Run from repository root:
    python scripts/geometry_of_truth/22_method_a_multilayer_finetune.py --prepare
    python scripts/geometry_of_truth/22_method_a_multilayer_finetune.py --dry-run
    python scripts/geometry_of_truth/22_method_a_multilayer_finetune.py --train

New training matrix:
    layers {3,9} x architectures {relu,topk,jumprelu}
    x seeds {42,1042,2042} x variants {got_replay,owt_only}
    = 36 fine-tunes.

Layer 6 is NOT retrained here. Its committed Method A evidence remains untouched
and is incorporated later by script 23.
"""

import argparse
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
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation import faithfulness_metrics as fm  # noqa: E402
from src.sae.checkpoint_registry import uninterrupted_checkpoint  # noqa: E402


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P11 = load("scripts/geometry_of_truth/11_truth_aware_pilot.py", "p11")

LAYERS = (3, 9)
SEEDS = (42, 1042, 2042)
ARCHS = ("relu", "topk", "jumprelu")

LR = P11.LR
STEPS = P11.STEPS
CLIP = P11.CLIP
GOT_PER_DOMAIN = P11.GOT_PER_DOMAIN
OWT_BATCH = P11.OWT_BATCH
DOMAINS = P11.DOMAINS
SPLIT_CSV = P11.SPLIT_CSV

TARGET = 128
ADJUST_EVERY = 25
ADJUST_RATE = 0.5
TEMPERATURE = 0.1
COEF_MIN, COEF_MAX = 1e-7, 100.0

OUT = REPO / "results/geometry_of_truth/method_a_multilayer"
CKPT = REPO / "results/geometry_of_truth/sae_finetuning/method_a_multilayer"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_domain(domain, layer):
    """Load GoT original activations for one domain at the requested layer."""
    rows, Xs = [], []
    for name in DOMAINS[domain]:
        d = torch.load(
            REPO / f"data/got/activations/gpt2_small/{name}_final_token.pt",
            map_location="cpu",
            weights_only=False,
        )
        s = (
            pd.read_csv(REPO / f"data/got/splits/{name}_splits.csv")
            .set_index("row_id")
            .reindex(np.asarray(d["row_ids"]))
        )
        assert (s.label.to_numpy() == np.asarray(d["labels"])).all()
        rows.append(s.assign(dataset=name).reset_index())
        Xs.append(d["activations"][layer].numpy().astype(np.float32))

    meta = pd.concat(rows, ignore_index=True)
    meta["group_id"] = meta.group_id.astype(str)

    ab = pd.read_csv(SPLIT_CSV, dtype={"group_id": str})
    ab = ab[ab.domain == domain].set_index("group_id")["half"]
    meta["half"] = meta.group_id.map(ab).where(meta.split == "train", "")
    return np.concatenate(Xs), meta


def verify_split_and_layers():
    """Verify unchanged A/B split and probe sufficiency at each target layer."""
    pilot_pre = json.loads((P11.OUT / "preregistration.json").read_text())
    res = {
        "split_file": SPLIT_CSV.relative_to(REPO).as_posix(),
        "split_sha256": sha(SPLIT_CSV),
        "matches_layer6_pilot": sha(SPLIT_CSV) == pilot_pre["split_file_sha256"],
        "layers": {},
    }

    overall = res["matches_layer6_pilot"]

    for layer in LAYERS:
        lr = {}
        for domain in DOMAINS:
            X, m = load_domain(domain, layer)
            y = m.label.to_numpy()
            A = (m.half == "A").to_numpy()
            B = (m.half == "B").to_numpy()
            va = (m.split == "val").to_numpy()

            gA, gB = set(m.group_id[A]), set(m.group_id[B])
            other = set(m.group_id[m.split != "train"])

            aucA, CA, _ = P11.select_C(X, y, A, va)
            aucB, CB, _ = P11.select_C(X, y, B, va)

            d = {
                "rows_A": int(A.sum()),
                "rows_B": int(B.sum()),
                "A_B_overlap": len(gA & gB),
                "train_vs_val_test_overlap": len((gA | gB) & other),
                "test_rows_in_A_or_B": int(((m.split == "test") & (A | B)).sum()),
                "label_mean_A": float(y[A].mean()),
                "label_mean_B": float(y[B].mean()),
                "probe_A_val_auroc": float(aucA),
                "probe_B_val_auroc": float(aucB),
                "probe_A_C": float(CA),
                "probe_B_C": float(CB),
            }
            d["passes"] = bool(
                d["A_B_overlap"] == 0
                and d["train_vs_val_test_overlap"] == 0
                and d["test_rows_in_A_or_B"] == 0
                and abs(d["label_mean_A"] - 0.5) < 0.01
                and abs(d["label_mean_B"] - 0.5) < 0.01
                and d["rows_A"] >= 500
                and d["rows_B"] >= 500
                and d["probe_A_val_auroc"] >= P11.MIN_INFORMATIVE
                and d["probe_B_val_auroc"] >= P11.MIN_INFORMATIVE
            )
            lr[domain] = d
            overall &= d["passes"]
        res["layers"][str(layer)] = lr

    res["passes"] = bool(overall)
    return res


def plan():
    runs = []
    for layer in LAYERS:
        for arch in ARCHS:
            for seed in SEEDS:
                base = uninterrupted_checkpoint(layer, arch, seed)
                for variant in ("got_replay", "owt_only"):
                    name = f"layer{layer}_{arch}_{variant}_seed{seed}"
                    runs.append(
                        {
                            "name": name,
                            "layer": layer,
                            "arch": arch,
                            "seed": seed,
                            "variant": variant,
                            "base": base,
                            "path": CKPT / name / "finetuned_checkpoint.pt",
                        }
                    )
    return runs


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg_path = OUT / "preregistration.json"
    if prereg_path.exists():
        raise SystemExit("preregistration.json exists; refusing to overwrite.")

    checks = verify_split_and_layers()
    (OUT / "split_and_layer_verification.json").write_text(
        json.dumps(checks, indent=2)
    )
    if not checks["passes"]:
        raise SystemExit("split/layer verification failed")

    prereg = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Extend committed Method A from layer 6 to layers 3 and 9.",
        "historical_layer6_untouched": True,
        "layers_new": list(LAYERS),
        "seeds": list(SEEDS),
        "architectures": list(ARCHS),
        "split_file": SPLIT_CSV.relative_to(REPO).as_posix(),
        "split_file_sha256": checks["split_sha256"],
        "reference_protocol": (
            "scripts/geometry_of_truth/14_method_a_reconstruction_finetune.py; "
            "same GoT-A/OWT replay objective, batches, optimizer, steps, "
            "sparsity controller and guards."
        ),
        "training": {
            "method_a_loss": "0.5*MSE_GoT_A + 0.5*MSE_OWT",
            "owt_only_loss": "MSE_OWT on 256 OWT rows",
            "batch": "64 geographic-A + 64 numerical-A + 128 OWT train",
            "lr": LR,
            "steps": STEPS,
            "grad_clip": CLIP,
            "checkpoint_selection": "none; final step only",
            "new_training_runs": len(plan()),
        },
        "evaluation": {
            "data": "validation only; GoT/RepE test rows never scored",
            "primary_metric": "GoT geographic independent B-MLP gap",
            "paired_comparisons": "method_a vs base and method_a vs owt_only",
            "uncertainty": (
                "mean paired difference + SD + two-sided 95% Student-t CI "
                "across the 3 training seeds; CI is descriptive because n=3"
            ),
        },
        "success_criteria": {
            "primary": (
                "For each layer and architecture: Method A geographic "
                "B-MLP gap < paired base AND < paired OWT-only on all 3 seeds, "
                "and mean paired improvement > 2 x SD of paired improvements."
            ),
            "guards": [
                "OWT validation L0 within +/-5% of paired base on every seed",
                "OWT splice delta CE <= 1.10 x paired base on every seed",
                "OWT validation EV >= paired base - 0.005 on every seed",
                "GoT cities truth-preference change >= base - 0.018 (mean)",
                "RepE validation mass-mean gap <= base + 0.02 (mean)",
            ],
        },
    }
    prereg_path.write_text(json.dumps(prereg, indent=2))

    # Record hashes after both extension scripts are present.
    files = [
        "scripts/geometry_of_truth/22_method_a_multilayer_finetune.py",
        "scripts/geometry_of_truth/23_method_a_multilayer_evaluation.py",
        SPLIT_CSV.relative_to(REPO).as_posix(),
        "results/geometry_of_truth/method_a_multilayer/preregistration.json",
    ]
    hashes = {p: sha(REPO / p) for p in files}
    (OUT / "pretraining_hashes.json").write_text(
        json.dumps(
            {
                "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Recorded before any layer-3/layer-9 Method A run.",
                "sha256": hashes,
            },
            indent=2,
        )
    )
    print(json.dumps(checks, indent=2))
    print(f"PRE-REGISTERED: {len(plan())} new fine-tunes")


def finetune(run, X_A, owt_train, device):
    sae, ck = fm.load_sae(run["base"], device)
    sae.train()

    arch = run["arch"]
    seed = run["seed"]

    coef = (
        float(ck["validation_metrics"].get("sparsity_coefficient", 0.0))
        if arch != "topk"
        else 0.0
    )
    coef0 = coef

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)

    ema, trace = None, []

    for step in range(1, STEPS + 1):
        if run["variant"] == "got_replay":
            idx = {
                d: torch.randint(
                    len(X_A[d]), (GOT_PER_DOMAIN,), generator=gen
                )
                for d in DOMAINS
            }
            oidx = torch.randint(
                len(owt_train), (OWT_BATCH,), generator=gen
            )

            xg = torch.cat(
                [X_A[d][idx[d].to(device)] for d in DOMAINS]
            )
            xo = owt_train[oidx].to(device)

            rg, zg, pg = sae(xg)
            ro, zo, po = sae(xo)

            loss = (
                0.5 * F.mse_loss(rg, xg)
                + 0.5 * F.mse_loss(ro, xo)
            )
            z_all = torch.cat([zg, zo])
            p_all = torch.cat([pg, po])

        else:
            oidx = torch.randint(
                len(owt_train), (2 * OWT_BATCH,), generator=gen
            )
            xo = owt_train[oidx].to(device)
            ro, zo, po = sae(xo)

            loss = F.mse_loss(ro, xo)
            z_all, p_all = zo, po

        if arch == "relu":
            loss = loss + coef * z_all.sum(-1).mean()
        elif arch == "jumprelu":
            loss = loss + coef * torch.sigmoid(
                (p_all - sae.log_threshold.exp()) / TEMPERATURE
            ).sum(-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CLIP)
        opt.step()
        sae.normalize_decoder()

        if arch != "topk":
            with torch.no_grad():
                l0 = (zo > 0).sum(-1).float().mean().item()
            ema = l0 if ema is None else 0.9 * ema + 0.1 * l0

            if step % ADJUST_EVERY == 0:
                coef *= math.exp(
                    ADJUST_RATE
                    * max(
                        -2.0,
                        min(
                            2.0,
                            math.log(max(ema, 1.0) / TARGET),
                        ),
                    )
                )
                coef = max(COEF_MIN, min(COEF_MAX, coef))

        if step in (1, 100, 250, 500):
            trace.append(
                {
                    "layer": run["layer"],
                    "run": run["name"],
                    "step": step,
                    "loss": loss.item(),
                    "coef": coef,
                    "ema_owt_l0": (
                        ema if ema is not None else float("nan")
                    ),
                }
            )

    return sae, coef0, coef, trace


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--train", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        return prepare()

    hash_file = OUT / "pretraining_hashes.json"
    if not hash_file.exists():
        raise SystemExit("Run --prepare first.")

    hashes = json.loads(hash_file.read_text())["sha256"]
    bad = [p for p, h in hashes.items() if sha(REPO / p) != h]
    if bad:
        raise SystemExit(
            f"files changed since pre-registration: {bad}"
        )

    checks = verify_split_and_layers()
    if not checks["passes"]:
        raise SystemExit("split/layer verification failed")

    runs = plan()
    clash = [str(r["path"]) for r in runs if r["path"].exists()]
    if (OUT / "training_info.json").exists():
        clash.append(str(OUT / "training_info.json"))
    if clash:
        raise SystemExit(f"refusing to overwrite: {clash}")

    missing = [str(r["base"]) for r in runs if not Path(r["base"]).exists()]
    if missing:
        raise SystemExit(f"missing base checkpoints: {missing}")

    if args.dry_run:
        for r in runs:
            ck = torch.load(
                r["base"], map_location="cpu", weights_only=False
            )
            c = (
                ck["validation_metrics"].get("sparsity_coefficient")
                if r["arch"] != "topk"
                else None
            )
            print(
                f"{r['name']:42s} "
                f"base={Path(r['base']).relative_to(REPO)} "
                f"start_coef={c}"
            )
        print(
            f"{len(runs)} training runs planned; nothing trained."
        )
        return

    device = "cuda"
    o3 = load("scripts/openwebtext/03_training_sae.py", "o3")

    base_sha = {str(r["base"]): sha(r["base"]) for r in runs}
    info, traces = [], []
    t0 = time.time()

    torch.cuda.reset_peak_memory_stats()

    for layer in LAYERS:
        owt_train = o3.load_split(
            REPO / "data/activations/gpt2_small",
            "train",
            layer,
        )

        X_A = {}
        for d in DOMAINS:
            X, m = load_domain(d, layer)
            X_A[d] = torch.tensor(
                X[(m.half == "A").to_numpy()],
                device=device,
            )

        layer_runs = [r for r in runs if r["layer"] == layer]

        for r in layer_runs:
            sae, c0, c1, tr = finetune(
                r, X_A, owt_train, device
            )
            traces += tr

            r["path"].parent.mkdir(
                parents=True, exist_ok=False
            )

            torch.save(
                {
                    "model_state_dict": {
                        k: t.detach().cpu()
                        for k, t in sae.state_dict().items()
                    },
                    "config": {
                        "architecture": r["arch"],
                        "target_l0": TARGET,
                        "layer": layer,
                        "seed": r["seed"],
                        "variant": r["variant"],
                        "lr": LR,
                        "steps": STEPS,
                        "procedure": (
                            "method_a_multilayer_2026-11-08"
                        ),
                    },
                    "validation_metrics": {
                        "sparsity_coefficient": c1
                    },
                    "sparsity_coefficient_start": c0,
                    "source_checkpoint": Path(
                        r["base"]
                    ).relative_to(REPO).as_posix(),
                    "source_checkpoint_sha256": base_sha[
                        str(r["base"])
                    ],
                },
                r["path"],
            )

            info.append(
                {
                    "layer": layer,
                    "run": r["name"],
                    "arch": r["arch"],
                    "seed": r["seed"],
                    "variant": r["variant"],
                    "checkpoint": r["path"]
                    .relative_to(REPO)
                    .as_posix(),
                    "coef_start": c0,
                    "coef_end": c1,
                }
            )

            print(
                f"DONE {r['name']} "
                f"({time.time() - t0:.0f}s) "
                f"coef {c0:.4g} -> {c1:.4g}",
                flush=True,
            )

    assert all(
        sha(p) == h for p, h in base_sha.items()
    ), "a base checkpoint changed!"

    pd.DataFrame(traces).to_csv(
        OUT / "training_trace.csv", index=False
    )

    (OUT / "training_info.json").write_text(
        json.dumps(
            {
                "script_sha256": sha(__file__),
                "runs": info,
                "base_checkpoints_unchanged": True,
                "gpu_peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / 2**20,
                    1,
                ),
                "runtime_seconds": round(
                    time.time() - t0, 1
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
