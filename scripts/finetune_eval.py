"""Run the BabyLM multilingual finetune-eval suite on a HF model.

Per the official rules, the per-task finetune step takes a fresh copy of our
PRETRAINED model, attaches a classification head, trains for at most
`max_epochs` on the task's small training set with early-stopping patience,
and reports validation accuracy. Predictions are written to predictions.txt.

This wrapper:
- enumerates only the leaderboard's EXPECTED_FINETUNE tasks per language
  (ignores xcopa, global_mmlu which are present locally but not scored),
- runs up to N_PARALLEL tasks at once, each pinned to its own GPU,
- logs every (model, lang, task) run to W&B as a separate run under one
  shared project, with `report_to=wandb` so per-step train/eval curves come
  for free,
- streams a stdout log line per task start/finish so the dashboard isn't the
  only place progress shows up.

Usage:
    python scripts/finetune_eval.py <hf_repo>
        [--revision main]
        [--langs eng nld zho]
        [--n-parallel 4]
        [--lr 5e-5] [--bsz 64] [--max-epochs 10] [--patience 3] [--seed 12]
        [--max-seq-length 128]
        [--project babylm-2026-multilingual]
        [--name-prefix baseline-uniform-v2]
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

EVAL_REPO = Path("/mnt/sagemaker-nvme/babylm/external/babylm-eval")
ML_DIR = EVAL_REPO / "multilingual"
DATA_ROOT = ML_DIR / "finetune" / "data" / "multilingual"
RESULTS_ROOT = ML_DIR / "finetune" / "results"

# Mirrors collate_results.EXPECTED_FINETUNE so the leaderboard validator is happy.
EXPECTED = {
    "en": ["arc", "belebele", "bmlama", "mnli", "sib200", "truthfulqa", "xnli"],
    "nl": ["arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa"],
    "zh": ["arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa", "xnli"],
}

# Map our 3-letter lang codes (used in CLI) to the 2-letter codes the on-disk
# data files use.
LANG_3_TO_2 = {"eng": "en", "nld": "nl", "zho": "zh"}


def build_cmd(
    repo: str,
    revision: str,
    lang2: str,
    task: str,
    out_dir: Path,
    args: argparse.Namespace,
    wandb_run_name: str,
) -> tuple[list[str], dict]:
    train_file = DATA_ROOT / lang2 / task / f"{task}_{lang2}.train.jsonl"
    valid_file = DATA_ROOT / lang2 / task / f"{task}_{lang2}.valid.jsonl"
    if not train_file.exists() or not valid_file.exists():
        raise FileNotFoundError(f"missing train/valid for {lang2}/{task}: {train_file}")
    cmd = [
        "python", "finetune/finetune_classification.py",
        "--model_name_or_path", repo,
        "--model_revision", revision,
        "--language", lang2,
        "--output_dir", str(out_dir),
        "--train_file", str(train_file),
        "--validation_file", str(valid_file),
        "--do_train", "True",
        "--do_eval",
        "--do_predict",
        "--max_seq_length", str(args.max_seq_length),
        "--per_device_train_batch_size", str(args.bsz),
        "--per_device_eval_batch_size", str(args.bsz),
        "--learning_rate", str(args.lr),
        "--num_train_epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--overwrite_output_dir",
        "--seed", str(args.seed),
        "--trust_remote_code",
        "--report_to", "wandb",
        "--run_name", wandb_run_name,
        # Suppress overly chatty model-card writes; we don't push.
        "--logging_steps", "20",
    ]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = args.project
    # Tag the run for easy dashboard filtering.
    env["WANDB_TAGS"] = ",".join(["finetune", args.name_prefix or "untagged", lang2, task])
    return cmd, env


def worker(gpu: int, jobs: queue.Queue, results: list, args, errlog: Path) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        repo, revision, lang2, task, out_dir, run_name = job
        cmd, env = build_cmd(repo, revision, lang2, task, out_dir, args, run_name)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_file = errlog.parent / f"finetune_{lang2}_{task}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[gpu{gpu}] {lang2}/{task} START -> {log_file}", flush=True)
        with open(log_file, "w") as f:
            f.write(f"# {' '.join(shlex.quote(c) for c in cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=str(ML_DIR), env=env,
                                   stdout=f, stderr=subprocess.STDOUT)
        rc = proc.returncode
        dt = time.time() - t0
        results.append((lang2, task, rc, dt, log_file))
        print(f"[gpu{gpu}] {lang2}/{task} END rc={rc} dt={dt:.0f}s", flush=True)
        jobs.task_done()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--langs", nargs="*", default=["eng", "nld", "zho"])
    p.add_argument("--n-parallel", type=int, default=4)
    p.add_argument("--lr", default="5e-5")
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--project", default="babylm-2026-multilingual")
    p.add_argument("--name-prefix", default=None)
    p.add_argument("--errlog", default="logs/finetune_errlog.log")
    p.add_argument("--only", nargs="*", default=None,
                   help="optional subset like 'eng/arc nld/mnli zho/sib200'")
    args = p.parse_args()

    name_prefix = args.name_prefix or args.repo.split("/")[-1]

    # Build job queue
    jobs: queue.Queue = queue.Queue()
    n_skip = 0
    only = set(args.only) if args.only else None
    for lang3 in args.langs:
        lang2 = LANG_3_TO_2[lang3]
        for task in EXPECTED[lang2]:
            tag = f"{lang3}/{task}"
            if only is not None and tag not in only:
                n_skip += 1
                continue
            out_dir = ML_DIR / "finetune" / "results" / args.repo.split("/")[-1] / lang2 / task
            run_name = f"finetune/{name_prefix}/{lang3}/{task}"
            jobs.put((args.repo, args.revision, lang2, task, out_dir, run_name))
    n_total = jobs.qsize()
    print(f"[orchestrator] {n_total} (lang,task) pairs queued, {n_skip} skipped, {args.n_parallel} parallel workers")

    Path(args.errlog).parent.mkdir(parents=True, exist_ok=True)
    results: list = []
    threads = []
    for gpu in range(args.n_parallel):
        t = threading.Thread(target=worker, args=(gpu, jobs, results, args, Path(args.errlog)), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Summary
    print("\n=== finetune summary ===")
    print(f"{'lang':>6} {'task':>14} {'rc':>4} {'dt':>8}  log")
    n_ok = 0
    for lang2, task, rc, dt, log in sorted(results):
        ok = "ok" if rc == 0 else f"FAIL({rc})"
        print(f"{lang2:>6} {task:>14} {rc:>4} {dt:>7.0f}s  {log}")
        if rc == 0:
            n_ok += 1
    print(f"\n{n_ok}/{len(results)} tasks succeeded")
    # Exit nonzero if anything failed
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
