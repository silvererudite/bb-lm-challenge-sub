# v3 finetune scores

22/22 finetune tasks succeeded on `Shamima/babylm-2026-multilingual-v3-quality-filter@main`.
Total wallclock: ~22 min at 4-way GPU parallelism. Logged to wandb.

## English (12 tasks total = 5 zero-shot + 7 finetune)

| Task | v2 | v3 | Δ | Type |
|---|---:|---:|---:|---|
| blimp_babylm_filtered | 58.98 | **62.43** | +3.45 | zero-shot |
| hellaswag_en_mubench | 26.69 | 26.44 | −0.25 | zero-shot |
| multiblimp_eng | 74.55 | **80.91** | **+6.36** | zero-shot |
| winogrande_en_mubench | 50.70 | 49.38 | −1.32 | zero-shot |
| xstorycloze_en_mubench | 47.99 | 49.30 | +1.31 | zero-shot |
| arc | 24.55 | 23.44 | −1.11 | finetune |
| belebele | 20.31 | 20.31 | 0 | finetune |
| bmlama | 13.80 | 13.63 | −0.17 | finetune |
| mnli | 37.56 | **47.22** | **+9.66** | finetune |
| sib200 | 47.92 | **66.30** | **+18.38** | finetune |
| truthfulqa | 17.19 | 17.19 | 0 | finetune |
| xnli | 41.18 | **46.40** | **+5.22** | finetune |
| **EN avg** | **38.45** | **42.13** | **+3.68** | |

## Dutch (13 tasks total = 6 zero-shot + 7 finetune)

| Task | v2 | v3 | Δ | Type |
|---|---:|---:|---:|---|
| blimp_nl | 71.13 | **73.27** | +2.14 | zero-shot |
| hellaswag_nl_mubench | 25.96 | 26.44 | +0.48 | zero-shot |
| multiblimp_nld | 85.50 | **89.53** | **+4.03** | zero-shot |
| winogrande_nl_mubench | 50.95 | 48.31 | −2.64 | zero-shot |
| xcomps_nl | 51.26 | 51.78 | +0.52 | zero-shot |
| xstorycloze_nl_mubench | 47.60 | 47.91 | +0.31 | zero-shot |
| arc | 22.77 | 23.05 | +0.28 | finetune |
| belebele | 21.09 | 25.00 | +3.91 | finetune |
| bmlama | 12.50 | 11.92 | −0.58 | finetune |
| include | 29.69 | 30.46 | +0.77 | finetune |
| mnli | 38.02 | **52.69** | **+14.67** | finetune |
| sib200 | 40.10 | **55.73** | **+15.63** | finetune |
| truthfulqa | 25.00 | 23.21 | −1.79 | finetune |
| **NL avg** | **40.05** | **43.80** | **+3.75** | |

## Chinese (13 tasks total = 5 zero-shot + 8 finetune)

| Task | v2 | v3 | Δ | Type |
|---|---:|---:|---:|---|
| hellaswag_zh_mubench | 25.88 | 26.39 | +0.51 | zero-shot |
| winogrande_zh_mubench | 49.46 | 50.45 | +0.99 | zero-shot |
| xcomps_zh | 52.35 | 52.61 | +0.26 | zero-shot |
| xstorycloze_zh_mubench | 45.20 | 47.52 | +2.32 | zero-shot |
| zhoblimp | 65.97 | 64.93 | −1.04 | zero-shot |
| arc | 23.21 | 23.96 | +0.75 | finetune |
| belebele | 27.34 | 23.05 | −4.29 | finetune |
| bmlama | 14.32 | 13.65 | −0.67 | finetune |
| include | 26.56 | 30.46 | +3.90 | finetune |
| mnli | 41.67 | 45.55 | +3.88 | finetune |
| sib200 | 63.02 | **77.55** | **+14.53** | finetune |
| truthfulqa | 25.00 | 26.79 | +1.79 | finetune |
| xnli | 41.23 | 46.39 | +5.16 | finetune |
| **ZH avg** | **38.56** | **39.38** | +0.82 | |

## Aggregate

| Metric | v2 | v3 | Δ |
|---|---:|---:|---:|
| ML avg ((EN+NL+ZH)/3) | **39.04** | **41.77** | **+2.73** |

## What this tells us

- **The big finetune wins are on classification benchmarks**: MNLI EN +9.7, NL +14.7; SIB-200 EN +18.4, NL +15.6, ZH +14.5; XNLI EN/ZH +5.2 each. These benchmarks are bottlenecked by the *quality of pretrained representations*. The 5x compute on cleaner data made representations sharply better.
- **ZH belebele regressed −4.29.** Belebele is reading-comprehension over diverse text; we may have removed too much ZH variety by dropping WenetSpeech.
- **Knowledge-recall benchmarks are mostly flat**: ARC, BMLama, TruthfulQA. We threw out a lot of factual content with the padding-categories drop. Trade-off was net positive.

## Files

- `orchestrator.log` — the v3 finetune orchestrator stdout (22 task summary)
- `finetune/{en,nl,zh}/<task>/eval_results.json` — per-task `eval_accuracy` etc.
