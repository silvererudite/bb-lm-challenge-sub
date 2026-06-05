#!/bin/bash
# End-to-end multilingual eval for a trained checkpoint.
# Usage: scripts/eval_model.sh <model_dir> <model_name> [--langs "eng nld zho"]
#
# Runs zero-shot via lm-eval-harness and writes a leaderboard-shaped
# submission JSON. Skips finetune by default (slow); pass --finetune to add.
set -euo pipefail

MODEL_DIR=${1:?"path to checkpoint dir, e.g. checkpoints/baseline_uniform/final"}
MODEL_NAME=${2:?"display name, e.g. baseline-uniform-100M"}
shift 2 || true

LANGS="eng nld zho"
DO_FINETUNE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --langs) LANGS="$2"; shift 2 ;;
        --finetune) DO_FINETUNE=1; shift ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVAL_REPO=/mnt/sagemaker-nvme/babylm/external/babylm-eval
RESULTS_DIR=$EVAL_REPO/results/main/${MODEL_NAME}

. $EVAL_REPO/.venv/bin/activate
export HF_HOME=/mnt/sagemaker-nvme/babylm/cache
export HF_DATASETS_CACHE=/mnt/sagemaker-nvme/babylm/cache
export HF_TOKEN=${HF_TOKEN:-}

cd $EVAL_REPO/multilingual

ABS_MODEL=$(cd "$REPO_ROOT" && readlink -f "$MODEL_DIR")
echo "[eval] model: $ABS_MODEL"
echo "[eval] langs: $LANGS"

for lang in $LANGS; do
    echo "[eval] zeroshot_${lang}"
    python -m lm_eval --model hf \
        --model_args pretrained=${ABS_MODEL},trust_remote_code=True \
        --tasks zeroshot_${lang} \
        --device cuda:0 \
        --output_path ../results/main \
        --batch_size auto:4 \
        --num_fewshot 0 \
        --include_path tasks/ \
        --log_samples
done

# Symlink the model dir under results/main using the display name so collate
# picks it up. lm-eval names the output dir using the path components, so we
# rename to MODEL_NAME for clean collate.
SRC=$(ls -d $EVAL_REPO/results/main/*${MODEL_DIR##*/}* 2>/dev/null | head -1 || true)
if [ -n "$SRC" ] && [ "$SRC" != "$RESULTS_DIR" ]; then
    mv "$SRC" "$RESULTS_DIR"
fi

if [ "$DO_FINETUNE" = "1" ]; then
    echo "[eval] finetune"
    bash scripts/finetune_model.sh --model_name "$ABS_MODEL" --langs "$(echo $LANGS)"
fi

# Collate
mkdir -p $REPO_ROOT/docs/submissions
python scripts/collate_results.py --model_name "$MODEL_NAME" \
    --output $REPO_ROOT/docs/submissions/${MODEL_NAME}_submission.json \
    --output_predictions $REPO_ROOT/docs/submissions/${MODEL_NAME}_predictions.json
echo "[eval] DONE -> docs/submissions/${MODEL_NAME}_submission.json"
