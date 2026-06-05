"""Train a model from a YAML config. Same scaffold runs every cell of the grid.

Usage (single GPU):
    python scripts/train.py configs/<run>.yaml [--max-steps N]
Usage (DDP across all visible GPUs):
    accelerate launch --multi_gpu --num_processes 4 scripts/train.py configs/<run>.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from babylm.config import BYTE_PREMIUM, RunConfig
from babylm.trainer import train


def _read_total_tokens(repo_card_path: Path) -> int:
    """Pull the README's 'Total Tokens: N' line. Cheap, deterministic."""
    import re
    txt = repo_card_path.read_text()
    m = re.search(r"Total Tokens:\*\*\s*([\d,]+)", txt)
    if not m:
        raise RuntimeError(f"could not find Total Tokens in {repo_card_path}")
    return int(m.group(1).replace(",", ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    p.add_argument("--data-card-dir", type=str, default="docs/audit",
                   help="dir containing per-lang dataset_README.md (used to read available tokens)")
    args = p.parse_args()

    cfg = RunConfig.load(args.config)

    # Compute available reference tokens per language from the dataset cards.
    avail = {}
    for lang in cfg.data.repos:
        card = Path(args.data_card_dir) / lang / "dataset_README.md"
        raw = _read_total_tokens(card)
        avail[lang] = int(raw / BYTE_PREMIUM[lang])
    print(f"[train] available reference tokens per lang: {avail}")
    print(f"[train] budget reference tokens: {cfg.data.budget_reference_tokens:,}")

    summary = train(cfg, available_reference_tokens=avail, max_steps=args.max_steps)
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
