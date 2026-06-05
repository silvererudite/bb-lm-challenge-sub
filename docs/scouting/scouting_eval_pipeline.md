# BabyLM 2026 Eval Pipeline — Scouting Report

**Date:** 2026-06-05
**Scope:** MultiLingual track only (Strict / Strict-Small skimmed but not exercised).

## Repo state
- Cloned `https://github.com/babylm-org/babylm-eval` to `/mnt/sagemaker-nvme/babylm/external/babylm-eval/`
- Branch `main`, latest commit `3bf5142` ("Update the FT script") — recent and active.
- Already 2026-shaped: top-level dirs are `strict/` and `multilingual/`. `multilingual/` is fully populated with custom task yamls, a finetune script, and submission collation. The README references arxiv 2602.20092 (the 2026 call) and 2510.10159 (BabyBabelLM).
- Python venv created at `/mnt/sagemaker-nvme/babylm/external/babylm-eval/.venv` (Python 3.12, lm-eval 0.4.9, transformers 4.49.0, datasets 4.8.5, torch 2.8.0). Activates via `. .venv/bin/activate`.

## Eval scope (what gets scored)
**Zero-shot** (run via lm-eval-harness with custom tasks under `multilingual/tasks/`):
- **EN (5):** blimp_babylm_filtered, multiblimp_eng, winogrande_en_mubench, xstorycloze_en_mubench, hellaswag_en_mubench
- **NL (6):** blimp_nl, multiblimp_nld, winogrande_nl_mubench, xcomps_nl, hellaswag_nl_mubench, xstorycloze_nl_mubench
- **ZH (5):** zhoblimp, winogrande_zh_mubench, xcomps_zh, hellaswag_zh_mubench, xstorycloze_zh_mubench

These are aggregated into `zeroshot_eng / zeroshot_nld / zeroshot_zho` groups (each is a simple acc mean weighted by size). The headline aggregate is `zeroshot_babybabellm` = mean of the three.

There is also a *broader* `babybabellm` group (in `tasks/babybabellm/`) covering arc, belebele, copa, sib200, xnli, etc. — same family as the finetune list but evaluated zero-shot. It is **not** the scored leaderboard group; treat it as an analysis-only superset.

**Finetune** (run via `finetune/finetune_classification.py` — adapted from prior BabyLM editions):
- **EN (7):** arc, belebele, bmlama, mnli, sib200, truthfulqa, xnli
- **NL (7):** arc, belebele, bmlama, include, mnli, sib200, truthfulqa
- **ZH (8):** arc, belebele, bmlama, include, mnli, sib200, truthfulqa, xnli

Defaults: `lr=5e-5`, `bsz=64`, `max_epochs=10`, `patience=3`, `seed=12`. 128 max sequence length. Per-task data files expected at `finetune/data/multilingual/<lang>/<task>/<task>_<lang>.{train,valid}.jsonl`.

## Model interface contract
**Inputs:** any HuggingFace `AutoModelForCausalLM` checkpoint that lives on the Hub (`pretrained=ORG/MODEL`, optional `revision=...`). Public visibility is **required** for submission.

**Intermediate checkpoints:** the fast-eval flow expects revisions named `chck_1M, chck_2M, ..., chck_10M, chck_20M, ..., chck_1000M` on the Hub. Either name our checkpoints this way at upload time or patch `REVISIONS` in `collate_results.py` if we adopt a different scheme.

**Tokenizer:** must be HF-loadable (any `AutoTokenizer`). No constraints on vocab size.

## Submission artifacts
Two JSONs uploaded to the leaderboard (HF Spaces: `BabyLM-community/BabyLM-Leaderboard-2026`):
- `<model>_submission.json` — `{task_name: {key: acc}}`. Zeroshot uses `task_name` as both. Finetune uses `lang` (en/nl/zh) as inner key.
- `<model>_predictions.json` — `{ "zeroshot": <raw lm-eval results>, "finetune": <per-task predictions>, "fast_eval_results": [<scores at each revision>] }`.

Incomplete is allowed: missing tasks are scored as 0 in the average. Languages with zero results are silently skipped.

## Baseline numbers already in the README (multilingual track)

GPT-2 monolingual (NL only): zeroshot_nld avg **59.61** · finetune NL avg **32.96**.
GPT-2 monolingual (ZH only): zeroshot_zho avg **52.87** · finetune ZH avg **29.24**.
GPT-2 EN+NL equal: zeroshot_eng **56.81**, zeroshot_nld **58.56**, finetune EN **36.80**, finetune NL **35.18**.
GPT-2 EN+NL+ZH equal: zeroshot_eng **56.34**, zeroshot_nld **57.81**, zeroshot_zho **51.34**, finetune EN **28.34**, finetune NL **24.78**, finetune ZH **27.23**.

**Curse of multilinguality is visible.** Adding ZH to EN+NL drops Dutch zero-shot from 58.56 → 57.81 (−0.75pt) but more sharply hurts NL/ZH finetune (35.18 → 24.78 on NL; 35.02 → 27.23 on ZH). This is exactly the asymmetric-transfer phenomenon ATLAS predicts at scale; we now have a baseline at developmental scale to compare against.

## Verified end-to-end (smoke test)
Ran `lm_eval` with `sshleifer/tiny-gpt2` on `zeroshot_eng` (limit 4) — all 5 EN tasks executed, results landed at `babylm-eval/results/main/sshleifer__tiny-gpt2/results_<ts>.json`. Ran `collate_results.py` after a path fix (see bug below) — produced a valid `*_submission.json` with 5 tasks.

NL (`blimp_nl`) and ZH (`zhoblimp`) tasks also smoke-tested independently; HF datasets resolve and lm-eval scores them. Network and harness OK.

## Bugs / friction surfaced (to file upstream)

1. **Path inconsistency between `zeroshot_model.sh` and `collate_results.py`.** The shell script writes to `../results/${revision}` from inside `multilingual/`, which resolves to `babylm-eval/results/...`. The collator looks at `multilingual/results/main/...`. Workaround applied locally: `multilingual/results -> ../results` symlink. Real fix: pick one canonical location and update both.

2. **`EXPECTED_ZEROSHOT["eng"]` lists `"blimp"` but the EN group yaml emits `"blimp_babylm_filtered"`.** Result: the collate script will print a "missing blimp" warning, the produced submission JSON keys it as `blimp_babylm_filtered`, and the leaderboard validator (per the README baseline table, which calls it `blimp`) likely expects `blimp`. **This is the most important issue to resolve before submission** — either patch the expected set, rename the task back to `blimp`, or alias on output. We need to confirm what the leaderboard schema accepts.

3. `TRANSFORMERS_CACHE` env var emits a deprecation warning in transformers 4.49+. Cosmetic only — `HF_HOME` is honored.

4. The README and the actual eval script differ on whether the `--fast` flag is required for a "complete" multilingual submission. README says "required for a full BabyLM Challenge submission" but speaks about Strict; the multilingual collate script works fine without `--fast`. We should run fast-eval at intermediate checkpoints anyway (it's part of the contribution) but it's worth confirming with the organizers whether the multilingual leaderboard validates `fast_eval_results` strictly.

## What's next (actionable)

1. ✅ Eval pipeline reproducible from scratch — done.
2. Locate and audit BabyBabelLM training data on HF Hub (task #2). Likely org: `BabyLM-community` (the same org hosting baseline checkpoints).
3. Decide on task #2 whether to mirror the eval-yaml task names or use a renamed scaffold.
4. After scaffold is built, run **one full-track eval pass on the published GPT-2 trilingual baseline** — this gives us the ground-truth numbers and a known-good submission JSON to diff against ours during iteration.
5. File issues for bugs #1 and #2.

## Key paths to remember
- Repo: `/mnt/sagemaker-nvme/babylm/external/babylm-eval/`
- Venv: `/mnt/sagemaker-nvme/babylm/external/babylm-eval/.venv/`
- Eval results write to: `/mnt/sagemaker-nvme/babylm/external/babylm-eval/results/<revision>/<org__model>/`
- HF cache: `/mnt/sagemaker-nvme/babylm/cache` (set `HF_HOME` to this before any HF call)
- Smoke-test submission JSON: `/mnt/sagemaker-nvme/babylm/logs/smoke_eval/tiny-gpt2_submission.json`
