"""Extract per-language scores from a fast-eval revision sweep into a tidy CSV.

After scripts/eval_revisions.sh has populated
  /mnt/sagemaker-nvme/babylm/external/babylm-eval/results/<rev>/<org__model>/results_*.json
this script walks every revision dir, picks out the zeroshot_eng/nld/zho
groups and their immediate subtask scores, and writes one CSV row per
(revision, language, task).

Usage: python scripts/collect_revision_curve.py <hf_repo> <out_csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# `org__model` directory naming convention used by lm-eval-harness.
RESULTS_ROOT = Path("/mnt/sagemaker-nvme/babylm/external/babylm-eval/results")


def revision_to_int(rev: str) -> int:
    m = re.match(r"chck_(\d+)M$", rev)
    return int(m.group(1)) if m else -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("hf_repo")
    p.add_argument("out_csv")
    args = p.parse_args()

    org_model = args.hf_repo.replace("/", "__")
    rows: list[dict] = []
    for rev_dir in sorted(RESULTS_ROOT.iterdir(), key=lambda d: revision_to_int(d.name)):
        if not rev_dir.is_dir() or not rev_dir.name.startswith("chck_"):
            continue
        model_dir = rev_dir / org_model
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("results_*.json")):
            with open(jf) as f:
                data = json.load(f)
            results = data.get("results", {})
            for task_name, task_data in results.items():
                alias = task_data.get("alias", task_name)
                indent = len(alias) - len(alias.lstrip())
                acc = task_data.get("acc,none")
                if acc is None:
                    continue
                # Keep group rows (zeroshot_eng/nld/zho, indent 0) and
                # immediate subtasks (indent 1). Skip indent-2 paradigm rows.
                if indent > 1:
                    continue
                # Group rows are zeroshot_eng / zeroshot_nld / zeroshot_zho;
                # subtasks are blimp/multiblimp/etc.
                lang = None
                for k, v in (("eng", "_eng"), ("nld", "_nld"), ("zho", "_zho")):
                    if k in task_name or v in task_name:
                        lang = k
                        break
                rows.append({
                    "revision": rev_dir.name,
                    "rev_M": revision_to_int(rev_dir.name),
                    "task": task_name.lstrip(),
                    "lang": lang,
                    "indent": indent,
                    "is_group": indent == 0,
                    "acc": round(acc, 4),
                    "stderr": round(task_data.get("acc_stderr,none") or 0, 4),
                })

    rows.sort(key=lambda r: (r["rev_M"], r["lang"] or "", r["task"]))
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        if not rows:
            print("no rows found")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    # Compact summary: avg per (revision, lang) at indent==0
    print()
    print(f"{'revision':>12}  {'eng':>6}  {'nld':>6}  {'zho':>6}")
    by_rev: dict[str, dict[str, float]] = {}
    for r in rows:
        if not r["is_group"]:
            continue
        by_rev.setdefault(r["revision"], {})[r["lang"]] = r["acc"]
    for rev in sorted(by_rev, key=revision_to_int):
        d = by_rev[rev]
        print(f"{rev:>12}  {d.get('eng', 0):>6.4f}  {d.get('nld', 0):>6.4f}  {d.get('zho', 0):>6.4f}")


if __name__ == "__main__":
    main()
