

"""Stage 0b: merge the per-batch token arrays produced by running
owt_00_prepare_data.py repeatedly with --skip/--count/--out (the
memory-ceiling workaround), then generate the shared neutral snippet set
and sample summary that batch mode skips.
 
Run: python scripts/owt_00b_merge_batches.py --config configs/owt.yaml \
       --batches data/owt/tokens/batch_0.npy data/owt/tokens/batch_1.npy ...
"""
 
from __future__ import annotations
 
import argparse
import sys
import time
from pathlib import Path
 
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
 
 
def main(config_path: str, batch_paths: list[str]) -> None:
    import numpy as np
    import pandas as pd
 
    from src.owt.data import load_config, load_token_array, save_neutral_snippets, save_token_array
 
    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
 
    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (data_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
 
    arrays = []
    for p in batch_paths:
        p = Path(p)
        if not p.is_absolute():
            p = REPO_ROOT / p
        arr = load_token_array(p)
        print(f"  loaded {p} -> shape {arr.shape}")
        arrays.append(arr)
 
    tokens = np.concatenate(arrays, axis=0)
    print(f"Merged {len(arrays)} batches into shape {tokens.shape}")
 
    from transformers import AutoTokenizer
 
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
 
    token_path = data_dir / "tokens" / "owt_20k.npy"
    save_token_array(tokens, token_path)
    print(f"Saved token array: {token_path} ({token_path.stat().st_size / 1e6:.1f} MB)")
 
    neutral_path = data_dir / "neutral_200.jsonl"
    save_neutral_snippets(
        tokens,
        tokenizer,
        n=config["sample"]["neutral_snippets"],
        path=neutral_path,
        seed=config["seed"],
    )
    print(f"Saved neutral snippets: {neutral_path} "
          f"(share with Simba and Aditi - first-priority deliverable)")
 
    summary = pd.DataFrame(
        [
            {
                "documents": tokens.shape[0],
                "context_length": tokens.shape[1],
                "tokens_total": tokens.shape[0] * tokens.shape[1],
                "mean_length": tokens.shape[1],
                "elapsed_seconds": None,
                "note": f"merged from {len(arrays)} batches",
            }
        ]
    )
    summary_path = results_dir / "tables" / "sample_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    parser.add_argument("--batches", nargs="+", required=True, help="Paths to the batch .npy files, in any order.")
    args = parser.parse_args()
    main(args.config, args.batches)
 
