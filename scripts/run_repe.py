"""RepE honesty analysis: baseline (stages 1-3), then SAE stages.

Run from the project root:
    python scripts/run_repe.py --stage 1

Inputs come from the EDA: data/repe/facts_split.csv and honesty_pairs_split.csv.
Tables go to results/tables/repe/, figures to results/figures/repe/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.repe import activations, controls, directions, evaluate, plots, prompts  # noqa: E402
from src.repe.io import (ACTIVATIONS_DIR, DATA_DIR, PAIRS_SPLIT_PATH, SPLIT_PATH,  # noqa: E402
                         save_table)

DIRECTIONS_DIR = DATA_DIR / "directions"


def stage_1_cache():
    """Stage 1: build the prompt sets and cache GPT-2 activations for each."""
    print("\n=== Stage 1: Cache activations ===")
    split_df = pd.read_csv(SPLIT_PATH)
    original_pairs = pd.read_csv(PAIRS_SPLIT_PATH)

    print("  loading GPT-2 (TransformerLens)...")
    model = activations.load_model()

    lengths = prompts.persona_lengths(lambda text: activations.count_tokens(model, text))
    print("\nLength-matched persona candidates:")
    print(lengths.to_string(index=False))
    save_table(lengths, "repe_s1_persona_candidates")
    if not lengths["length_matched"].any():
        print("  WARNING: no candidate pair is length-matched; add more to MATCHED_CANDIDATES.")

    sets = prompts.build_all_sets(split_df, original_pairs, lengths)
    summary = []
    for name, df in sets.items():
        print(f"\n  caching '{name}' ({len(df)} prompts)")
        last, mean, meta = activations.cache_prompt_set(model, df)
        activations.save_prompt_set(ACTIVATIONS_DIR, name, last, mean, meta)
        honest_len = meta.loc[meta["label"] == 1, "n_tokens"].mean()
        dishonest_len = meta.loc[meta["label"] == 0, "n_tokens"].mean()
        summary.append({
            "prompt_set": name, "n_prompts": len(meta), "n_pairs": meta["pair_id"].nunique(),
            "train_pairs": meta.loc[meta["split"] == "train", "pair_id"].nunique(),
            "test_pairs": meta.loc[meta["split"] == "test", "pair_id"].nunique(),
            "mean_tokens_honest": round(honest_len, 2),
            "mean_tokens_dishonest": round(dishonest_len, 2),
            "length_difference": round(dishonest_len - honest_len, 2),
            "activation_shape": str(last.shape),
        })

    summary = pd.DataFrame(summary)
    print("\nPrompt sets:")
    print(summary.to_string(index=False))
    save_table(summary, "repe_s1_prompt_sets")
    print(f"  activations saved in {ACTIVATIONS_DIR.relative_to(PROJECT_ROOT)}")
    

def split_rows(X, meta, split):
    """Rows of one split, with a fresh 'row' index that points into the returned X."""
    mask = (meta["split"] == split).values
    return X[mask], meta[mask].reset_index(drop=True).assign(row=lambda m: np.arange(len(m)))



def stage_2_baseline(prompt_set="original", n_random=200, n_boot=1000):
    """Stage 2: fit RepE PCA, mean-diff and logistic per layer; score on test pairs."""
    print("\n=== Stage 2: Baseline directions (no SAE) ===")
    last, mean, meta = activations.load_prompt_set(ACTIVATIONS_DIR, prompt_set)
    meta = meta.assign(row=np.arange(len(meta)))
    random_dirs = directions.random_directions(n_random)

    rows, band_rows = [], []
    saved = {}
    for read, acts in [("last", last), ("mean", mean)]:
        saved[read] = {m: np.zeros((acts.shape[1], acts.shape[2]), dtype=np.float32) for m in directions.METHODS}
        for p, position in enumerate(activations.POSITIONS):
            X = acts[:, p]
            X_tr, m_tr = split_rows(X, meta, "train")
            X_te, m_te = split_rows(X, meta, "test")

            for method in directions.METHODS:
                w, scorer = directions.fit(method, X_tr, m_tr)
                saved[read][method][p] = w
                scores = scorer(X_te)
                ci = evaluate.bootstrap_ci(scores, m_te, n_boot=n_boot)
                cv_mean, cv_std = evaluate.grouped_cv(lambda X_, m_: directions.fit(method, X_, m_),
                                                      X, meta)
                rows.append({
                    "prompt_set": prompt_set, "read": read, "position": position, "position_index": p,
                    "method": method,
                    "train_pair_accuracy": round(evaluate.pair_accuracy(scorer(X_tr), m_tr), 4),
                    "pair_accuracy": round(evaluate.pair_accuracy(scores, m_te), 4),
                    "pair_acc_low": round(ci["pair_acc_low"], 4), "pair_acc_high": round(ci["pair_acc_high"], 4),
                    "auroc": round(evaluate.auroc(scores, m_te), 4),
                    "auroc_low": round(ci["auroc_low"], 4), "auroc_high": round(ci["auroc_high"], 4),
                    "cv5_pair_accuracy_mean": round(cv_mean, 4), "cv5_pair_accuracy_std": round(cv_std, 4),
                })

            accs, aucs = evaluate.random_band(X_tr, m_tr, X_te, m_te, random_dirs)
            band_rows.append({"prompt_set": prompt_set, "read": read, "position": position,
                              "position_index": p,
                              "random_pair_accuracy_low": round(np.percentile(accs, 2.5), 4),
                              "random_pair_accuracy_high": round(np.percentile(accs, 97.5), 4),
                              "random_auroc_low": round(np.percentile(aucs, 2.5), 4),
                              "random_auroc_high": round(np.percentile(aucs, 97.5), 4)})
            print(f"  {read:4s} {position:5s} done")

    results, band = pd.DataFrame(rows), pd.DataFrame(band_rows)
    save_table(results, "repe_s2_baseline_by_layer")
    save_table(band, "repe_s2_random_band")

    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for read, by_method in saved.items():
        np.savez(DIRECTIONS_DIR / f"{prompt_set}_{read}.npz", **by_method)
    print(f"  directions saved in {DIRECTIONS_DIR.relative_to(PROJECT_ROOT)}")

    for metric in ["pair_accuracy", "auroc"]:
        view = results[results["read"] == "last"].pivot(index="position", columns="method", values=metric)
        view = view.reindex(activations.POSITIONS)[directions.METHODS]
        b = band[band["read"] == "last"].set_index("position")
        view["random_low"], view["random_high"] = b[f"random_{metric}_low"], b[f"random_{metric}_high"]
        print(f"\nTest {metric}, last token (0.5 = chance; random_low/high = 95% band of random directions):")
        print(view.to_string())

    n_test = meta.loc[meta["split"] == "test", "pair_id"].nunique()
    for metric in ["auroc", "pair_accuracy"]:
        plots.plot_baseline_by_layer(results, band, n_test, metric=metric)

def stage_3_controls(read="last"):
    """Stage 3: length-matched sets, position-only probe, wording transfer, false statements."""
    print("\n=== Stage 3: Controls ===")
    sets = controls.load_sets(ACTIVATIONS_DIR, read)
    print(f"  prompt sets: {', '.join(sets)}")

    position = controls.position_only(sets)
    print("\nCan token position ALONE predict the label?")
    print(position.to_string(index=False))
    save_table(position, "repe_s3_position_only")

    within, band = controls.within_set(sets)
    save_table(within, "repe_s3_within_set")
    save_table(band, "repe_s3_random_band")
    view = within[within["method"] == "mean_diff"].pivot(index="position", columns="prompt_set", values="auroc")
    print("\nWithin-set test AUROC, mean_diff (length-matched sets should be 0.5 at L0):")
    print(view.reindex(activations.POSITIONS)[list(sets)].to_string())

    tr = controls.transfer(sets)
    save_table(tr, "repe_s3_transfer")
    view = tr[(tr["source"] == "original") & (tr["method"] == "mean_diff")].pivot(
        index="position", columns="target", values="auroc")
    print("\nFitted on ORIGINAL wording, tested on each set (mean_diff AUROC):")
    print(view.reindex(activations.POSITIONS)[list(sets)].to_string())

    best = controls.best_layers(tr)
    print("\nLayers ranked by transfer to a DIFFERENT wording (mean_diff):")
    print(best.to_string(index=False))
    save_table(best, "repe_s3_best_layers")

    plots.plot_within_set(within, band)
    plots.plot_transfer_heatmap(tr, controls.SAE_POSITIONS)
    plots.plot_transfer_from_original(tr)
   
STAGES = {
    1: stage_1_cache,
    2: stage_2_baseline,
    3: stage_3_controls,
}


def main():
    parser = argparse.ArgumentParser(description="RepE honesty analysis")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="Run only this stage")
    args = parser.parse_args()

    for number in ([args.stage] if args.stage else sorted(STAGES)):
        STAGES[number]()
    print("\nDone.")


if __name__ == "__main__":
    main()