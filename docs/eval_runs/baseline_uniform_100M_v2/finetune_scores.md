# v2 finetune scores (full leaderboard breakdown)

Per-task results from running `multilingual/scripts/finetune_model.sh` (via
`scripts/finetune_eval.py` orchestrator) on
`Shamima/babylm-2026-multilingual-uniform-100M-v2@main`.

22/22 tasks succeeded.

## English (12 tasks total = 5 zero-shot + 7 finetune)

| Task | Acc | Type |
|---|---:|---|
| blimp_babylm_filtered | 58.98 | zero-shot |
| hellaswag_en_mubench | 26.69 | zero-shot |
| multiblimp_eng | 74.55 | zero-shot |
| winogrande_en_mubench | 50.70 | zero-shot |
| xstorycloze_en_mubench | 47.99 | zero-shot |
| arc | 24.55 | finetune |
| belebele | 20.31 | finetune |
| bmlama | 13.80 | finetune |
| mnli | 37.56 | finetune |
| sib200 | 47.92 | finetune |
| truthfulqa | 17.19 | finetune |
| xnli | 41.18 | finetune |
| **EN average** | **38.45** | |

## Dutch (13 tasks total = 6 zero-shot + 7 finetune)

| Task | Acc | Type |
|---|---:|---|
| blimp_nl | 71.13 | zero-shot |
| hellaswag_nl_mubench | 25.96 | zero-shot |
| multiblimp_nld | 85.50 | zero-shot |
| winogrande_nl_mubench | 50.95 | zero-shot |
| xcomps_nl | 51.26 | zero-shot |
| xstorycloze_nl_mubench | 47.60 | zero-shot |
| arc | 22.77 | finetune |
| belebele | 21.09 | finetune |
| bmlama | 12.50 | finetune |
| include | 29.69 | finetune |
| mnli | 38.02 | finetune |
| sib200 | 40.10 | finetune |
| truthfulqa | 25.00 | finetune |
| **NL average** | **40.05** | |

## Chinese (13 tasks total = 5 zero-shot + 8 finetune)

| Task | Acc | Type |
|---|---:|---|
| hellaswag_zh_mubench | 25.88 | zero-shot |
| winogrande_zh_mubench | 49.46 | zero-shot |
| xcomps_zh | 52.35 | zero-shot |
| xstorycloze_zh_mubench | 45.20 | zero-shot |
| zhoblimp | 65.97 | zero-shot |
| arc | 23.21 | finetune |
| belebele | 27.34 | finetune |
| bmlama | 14.32 | finetune |
| include | 26.56 | finetune |
| mnli | 41.67 | finetune |
| sib200 | 63.02 | finetune |
| truthfulqa | 25.00 | finetune |
| xnli | 41.23 | finetune |
| **ZH average** | **38.56** | |

## Aggregate

| Metric | Value |
|---|---:|
| Multilingual Average ((EN+NL+ZH)/3) | **39.04** |

## Wallclock

| Task class | Min wall (s) | Max wall (s) | Notes |
|---|---:|---:|---|
| sib200, truthfulqa, belebele | 21–62 | tiny train sets, finished in seconds | |
| arc, include | 27–87 | medium | |
| bmlama, mnli, xnli | 248–325 | the heavy hitters; MNLI/XNLI hit max_epochs (10) before patience kicked in | |

Total finetune wallclock: ~5.4 minutes per GPU on average; ~22 minutes for the full 22-task suite at 4-way parallelism. Pretty cheap relative to pretraining.

## Files

`docs/eval_runs/baseline_uniform_100M_v2/finetune/orchestrator.log` —
full orchestrator log + per-task wallclock summary.

`docs/eval_runs/baseline_uniform_100M_v2/finetune/{en,nl,zh}/<task>/eval_results.json` —
per-task `eval_accuracy`, `eval_loss`, `train_runtime`, etc., as written by HF Trainer.
