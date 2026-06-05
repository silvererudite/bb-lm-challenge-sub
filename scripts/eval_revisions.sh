#!/bin/bash
# Run zero-shot eval over every chck_NM revision present on the HF Hub for a model.
# Usage: scripts/eval_revisions.sh <hf_repo> [--langs "eng nld zho"]
#
# Skips revisions that don't exist (no hard failure). Results land under
# /mnt/sagemaker-nvme/babylm/external/babylm-eval/results/<revision>/<org__model>/.
# Use scripts/collect_revision_curve.py to extract a tidy CSV afterwards.
set -uo pipefail

HF_REPO=${1:?"hf repo, e.g. Shamima/babylm-2026-multilingual-uniform-100M"}
shift || true
LANGS="eng nld zho"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --langs) LANGS="$2"; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVAL_REPO=/mnt/sagemaker-nvme/babylm/external/babylm-eval

. $EVAL_REPO/.venv/bin/activate
export HF_HOME=/mnt/sagemaker-nvme/babylm/cache
export HF_DATASETS_CACHE=/mnt/sagemaker-nvme/babylm/cache
export HF_TOKEN=${HF_TOKEN:-}

cd $EVAL_REPO/multilingual

# Discover revisions on the Hub (cap at chck_500M to match our v2 schedule).
REVS=$(python -c "
import os
from huggingface_hub import HfApi
api = HfApi()
refs = api.list_repo_refs('${HF_REPO}')
names = sorted([b.name for b in refs.branches if b.name.startswith('chck_')], key=lambda n: int(n.removeprefix('chck_').removesuffix('M')))
print(' '.join(names))
")
echo "[sweep] revisions on hub: $REVS"

for rev in $REVS; do
    for lang in $LANGS; do
        echo "[sweep] ${rev} / zeroshot_${lang}"
        python -m lm_eval --model hf \
            --model_args pretrained=${HF_REPO},revision=${rev},trust_remote_code=True \
            --tasks zeroshot_${lang} \
            --device cuda:0 \
            --output_path ../results/${rev} \
            --batch_size auto:4 \
            --num_fewshot 0 \
            --log_samples \
            --include_path tasks/ 2>&1 | tail -3
    done
done
echo "[sweep] DONE"
