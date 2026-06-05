"""Train (or re-train) the joint subword tokenizer.

Usage:
    python scripts/train_tokenizer.py configs/<run>.yaml [--token <hf_token>]
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow `python scripts/...` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from babylm.config import RunConfig
from babylm.tokenizer import train_joint_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    cfg = RunConfig.load(args.config)
    print(f"[tokenizer] training {cfg.tokenizer.algorithm} vocab={cfg.tokenizer.vocab_size}")
    print(f"[tokenizer] sampling {cfg.tokenizer.train_chars_per_lang:,} chars per language")
    path = train_joint_tokenizer(cfg, token=args.token)
    print(f"[tokenizer] saved -> {path}")


if __name__ == "__main__":
    main()
