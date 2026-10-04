"""Stage 0: Stream OpenWebText, tokenize, chunk to a fixed length, and
save the token-id array plus the shared neutral snippet set.

Run: python scripts/owt_00_prepare_data.py --config configs/owt.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config_path: str) -> None:
    import pandas as pd

    from src.owt.data import load_config, save_neutral_snippets, save_token_array, set_all_seeds, stream_and_tokenize

    if not Path(config_path).is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    set_all_seeds(config["seed"])

    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    results_dir = REPO_ROOT / config["paths"]["results_dir"]
    (data_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Streaming {config['sample']['n_documents']} documents from "
          f"{config['corpus']['owt_dataset']} ({config['corpus']['owt_split']})...")
    start = time.time()

    tokens = stream_and_tokenize(
        dataset_name=config["corpus"]["owt_dataset"],
        split=config["corpus"]["owt_split"],
        tokenizer=tokenizer,
        n_documents=config["sample"]["n_documents"],
        context_length=config["sample"]["context_length"],
        shuffle_buffer_size=config["sample"]["shuffle_buffer_size"],
        seed=config["seed"],
    )
    elapsed = time.time() - start

    print(f"Kept {tokens.shape[0]} documents of shape {tokens.shape} in {elapsed:.1f}s")

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
                "mean_length": tokens.shape[1],  # fixed length by construction
                "elapsed_seconds": round(elapsed, 1),
            }
        ]
    )
    summary_path = results_dir / "tables" / "sample_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/owt.yaml")
    args = parser.parse_args()
    main(args.config)
