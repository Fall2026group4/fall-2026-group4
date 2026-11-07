#!/usr/bin/env python3
"""Rebuild docs/checkpoint_registry.csv from every SAE checkpoint under results/ (read-only scan).

Run from repository root:  python scripts/shared/build_checkpoint_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sae.checkpoint_registry import REGISTRY_CSV, build  # noqa: E402

if __name__ == '__main__':
    df = build()
    print(df.groupby(['family', 'status']).size().to_string())
    print(f'{len(df)} SAE checkpoints -> {REGISTRY_CSV.relative_to(Path.cwd()) if REGISTRY_CSV.is_relative_to(Path.cwd()) else REGISTRY_CSV}')
