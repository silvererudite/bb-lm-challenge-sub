"""Run multilingual zero-shot eval on a HF model and log scores to W&B.

This is a thin wrapper around lm-eval-harness so our eval results land in the
same wandb project as training. One run per (model, revision) pair.

Usage:
    python scripts/eval_and_log.py <hf_repo> [--revision main]
                                              [--langs "eng nld zho"]
                                              [--name <wandb_run_name>]
                                              [--tags v3 ...]
                                              [--project babylm-2026-multilingual]

Result JSONs from lm-eval are written to
  /mnt/sagemaker-nvme/babylm/external/babylm-eval/results/<revision>/<org__model>/
and parsed back to extract group + per-task accuracies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

EVAL_REPO = Path("/mnt/sagemaker-nvme/babylm/external/babylm-eval")
RESULTS_ROOT = EVAL_REPO / "results"
TASK_DIR = EVAL_REPO / "multilingual" / "tasks"


def run_lm_eval(repo: str, revision: str, lang: str, log_path: Path) -> int:
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={repo},revision={revision},trust_remote_code=True",
        "--tasks", f"zeroshot_{lang}",
        "--device", "cuda:0",
        "--output_path", str(RESULTS_ROOT / revision),
        "--batch_size", "auto:4",
        "--num_fewshot", "0",
        "--include_path", str(TASK_DIR),
        "--log_samples",
    ]
    print(f"[eval] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n=== zeroshot_{lang} ({repo}@{revision}) ===\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(EVAL_REPO / "multilingual"),
                              stdout=f, stderr=subprocess.STDOUT, env=os.environ)
    return proc.returncode


def parse_latest(repo: str, revision: str) -> dict:
    org_model = repo.replace("/", "__")
    rev_dir = RESULTS_ROOT / revision / org_model
    if not rev_dir.is_dir():
        raise RuntimeError(f"no results dir at {rev_dir}")
    files = sorted(rev_dir.glob("results_*.json"))
    if not files:
        raise RuntimeError(f"no results_*.json under {rev_dir}")
    merged: dict[str, dict] = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for k, v in d.get("results", {}).items():
            merged[k] = v
    return merged


def extract_scores(merged: dict) -> dict:
    """Pull a flat dict of {task: acc} for the immediate-task indent (incl. groups)."""
    out: dict[str, float] = {}
    for task_name, task_data in merged.items():
        alias = task_data.get("alias", task_name)
        indent = len(alias) - len(alias.lstrip())
        if indent > 1:
            continue
        acc = task_data.get("acc,none")
        if acc is None:
            continue
        out[task_name.lstrip()] = float(acc)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--langs", default="eng nld zho")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default="babylm-2026-multilingual")
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    log_path = Path(f"logs/eval_{args.repo.replace('/', '_')}_{args.revision}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(
        project=args.project,
        name=args.name or f"eval/{args.repo.split('/')[-1]}@{args.revision}",
        tags=["eval"] + list(args.tags),
        config={"repo": args.repo, "revision": args.revision, "langs": args.langs},
    )
    print(f"[eval-and-log] wandb: {run.url}")

    t_total = time.time()
    for lang in args.langs.split():
        t0 = time.time()
        rc = run_lm_eval(args.repo, args.revision, lang, log_path)
        print(f"[eval] zeroshot_{lang} rc={rc} dt={time.time()-t0:.0f}s", flush=True)

    merged = parse_latest(args.repo, args.revision)
    scores = extract_scores(merged)

    # Group scores
    flat = {}
    groups = {}
    for k, v in scores.items():
        flat[f"score/{k}"] = v
        if k.startswith("zeroshot_"):
            groups[k] = v
    if groups:
        avg = sum(groups.values()) / len(groups)
        flat["score/zeroshot_avg"] = avg

    print("\n=== group scores ===")
    for k in ("zeroshot_eng", "zeroshot_nld", "zeroshot_zho"):
        if k in groups:
            print(f"  {k}: {groups[k]:.4f}")
    if "score/zeroshot_avg" in flat:
        print(f"  avg: {flat['score/zeroshot_avg']:.4f}")

    run.log(flat)
    for k, v in flat.items():
        run.summary[k] = v
    run.summary["eval_wallclock_s"] = time.time() - t_total
    run.finish()


if __name__ == "__main__":
    main()
