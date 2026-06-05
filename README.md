# BabyLM 2026 — MultiLingual Track Submission (work in progress)

Submission to the BabyLM 2026 Challenge (4th edition), MultiLingual track (English / Dutch / Chinese, 100M-token byte-premium-adjusted budget, ≤10 epochs). Direct submission deadline: **2026-07-15**.

## Research direction

*A Mini-ATLAS for the developmental regime: cross-lingual transfer matrices and budget allocations at the BabyLM 100M scale, anchored in cognitively-plausible bilingual schedules.* See `CLAUDE.md` for the full thesis and methodological commitments.

## Status

| # | Status | Item |
|---|---|---|
| 1 | done | Eval pipeline cloned, installed, smoke-tested end-to-end on EN/NL/ZH; submission JSON shape verified — `docs/scouting/scouting_eval_pipeline.md`. |
| 2 | done | BabyBabelLM EN/NL/ZH per-language data card written from official cards + gated metadata access — `docs/data_card.md`. |
| 3 | next | Build training scaffold (decoder + tokenizer + byte-premium-aware data mixer). |
| 4 | next | Baseline: byte-premium-uniform trilingual run, full eval. |
| 5 | next | 3×3 EN/NL/ZH transfer matrix ablation. |
| 6 | next | Mixture-allocation Pareto sweep (uniform · loss-weighted · simultaneous-bilingual · CDS-weighted · typological-bridge curriculum). |
| 7 | next | Final submission run; OpenReview paper + leaderboard upload. |

## Layout

```
.
├── CLAUDE.md             # research direction, rules, methodological commitments
├── README.md             # this file
├── docs/
│   ├── data_card.md                    # per-language audit (EN/NL/ZH)
│   ├── audit/                          # per-language dataset README + metadata + sample
│   ├── scouting/scouting_eval_pipeline.md   # eval-pipeline scouting report
│   └── smoke_eval/                     # tiny-gpt2 smoke submission JSON
├── cache/ checkpoints/ data/ external/ logs/ models/   # gitignored — symlinks to /mnt/sagemaker-nvme
└── .gitignore
```

## Key findings so far (from scouting)

- The MultiLingual track scores 5 EN + 6 NL + 5 ZH zero-shot tasks (BLiMP, MultiBLiMP, HellaSwag, Winogrande, XStoryCloze, XCOMPS, ZhoBLiMP, BLiMP-NL) plus 7 EN + 7 NL + 8 ZH fine-tune tasks, aggregated by `lm-eval-harness` and a custom finetune script, with a single submission JSON per model.
- The published GPT-2 baselines already show the curse of multilinguality at developmental scale: adding ZH to EN+NL costs ~0.75pt zero-shot NL and ~10pt fine-tune NL/ZH. ATLAS-style transfer measurements at 10M–8B params have not been done at the BabyLM 100M / ≤10-epoch regime — this is the gap our work targets.
- The three corpora are NOT register-comparable: EN is ~80% genuine child-genre, NL is 63% OpenSubtitles padding, ZH is 66% subtitles. Cross-lingual transfer claims must control for register, not just language identity. This itself is a finding.
- Two real upstream bugs surfaced in the eval pipeline (path-resolution mismatch between zeroshot script and collator; `EXPECTED_ZEROSHOT["eng"]` lists `blimp` but the EN group emits `blimp_babylm_filtered`). Fixes/issues should be filed.

## How to reproduce

The scouting work uses a Python venv at `/mnt/sagemaker-nvme/babylm/external/babylm-eval/.venv` with `lm-eval==0.4.9`, `transformers==4.49.0`, `datasets>=4.8`, `torch==2.8.0`. Set `HF_HOME=/mnt/sagemaker-nvme/babylm/cache` before any Hugging Face call (the default `$HOME/.cache` is on a 5 GB partition).
