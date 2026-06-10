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


def _read_category_tokens(repo_card_path: Path) -> dict[str, int]:
    """Parse the README's 'Tokens Per Category' table into {category: tokens}."""
    import re
    txt = repo_card_path.read_text()
    out: dict[str, int] = {}
    in_section = False
    for line in txt.splitlines():
        if "Tokens Per Category" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("###") or "Tokens Per Group" in line:
                break
            m = re.match(r"-\s*\*\*([^:*]+):\*\*\s*([\d,]+)\s*tokens", line.strip())
            if m:
                out[m.group(1).strip()] = int(m.group(2).replace(",", ""))
    if not out:
        raise RuntimeError(f"could not parse 'Tokens Per Category' in {repo_card_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    p.add_argument("--data-card-dir", type=str, default="docs/audit",
                   help="dir containing per-lang dataset_README.md (used to read available tokens)")
    args = p.parse_args()

    cfg = RunConfig.load(args.config)

    # Compute available reference tokens per language. If a category whitelist
    # is set for that language, sum only the whitelisted categories; otherwise
    # use the full corpus total. The result drives the per-language epoch cap.
    avail = {}
    for lang in cfg.data.repos:
        card = Path(args.data_card_dir) / lang / "dataset_README.md"
        whitelist = cfg.data.category_whitelist.get(lang)
        if whitelist:
            cat_tokens = _read_category_tokens(card)
            unknown = [c for c in whitelist if c not in cat_tokens]
            if unknown:
                raise RuntimeError(
                    f"[{lang}] category_whitelist contains unknown categories {unknown}. "
                    f"Known categories: {sorted(cat_tokens)}"
                )
            raw = sum(cat_tokens[c] for c in whitelist)
            print(f"[train] {lang} category-filter: {whitelist} -> {raw:,} raw tokens")
        else:
            raw = _read_total_tokens(card)
        avail[lang] = int(raw / BYTE_PREMIUM[lang])
    print(f"[train] available reference tokens per lang: {avail}")
    print(f"[train] total available across langs: {sum(avail.values()):,}")
    print(f"[train] budget reference tokens: {cfg.data.budget_reference_tokens:,}")
    print(f"[train] max epochs per lang: {cfg.data.max_epochs}")
    print(f"[train] effective compute cap: {cfg.data.budget_reference_tokens * cfg.data.max_epochs:,} ref tokens "
          f"(or whichever per-lang cap trips first)")

    summary = train(cfg, available_reference_tokens=avail, max_steps=args.max_steps)
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
