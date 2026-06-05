# BabyLM 2026 MultiLingual submission — v2 (uniform-100M-v2)

This document is a **copy-paste guide** for the leaderboard submission form on
https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026 — every
field in the "🚀 Submit your model" form has its value pre-filled below.

## Files to upload

In the form's two file pickers:

1. **Results file** → `docs/submissions/v2_submission.json`
2. **Predictions file** → `docs/submissions/v2_predictions.json`

Both files are produced by the official `multilingual/scripts/collate_results.py`
and the JSON-key layout has been verified against the leaderboard's
`TasksMultilingual` enum (every expected key present, no extra keys).

## Form fields — paste these verbatim

### Model identity
| Field | Value |
|---|---|
| Hugging Face repo | `Shamima/babylm-2026-multilingual-uniform-100M-v2` |
| Track | `multilingual` |
| Revision | `main` |
| User name | (your HF username — `Shamima`) |

### Languages and corpus sizes
| Field | Value |
|---|---|
| Languages: English (eng) | ☑ |
| Languages: Dutch (nld) | ☑ |
| Languages: Chinese (zho) | ☑ |
| Words used (English) | `98878321` |
| Words used (Dutch) | `109885564` |
| Words used (Chinese) | `137835046` |
| Total words | `346598931` |

### Model architecture
| Field | Value |
|---|---|
| Architecture | `Llama` |
| Model type | `decoder` |
| Number of layers | `12` |
| Number of attention heads | `12` |
| Hidden size | `768` |
| Max sequence length | `1024` |
| Total parameters | `110119680` |
| Tokenizer | `Joint byte-level BPE (trained on EN+NL+ZH)` |
| Token set size (vocab) | `32768` |

### Training setup
| Field | Value |
|---|---|
| Optimizer | `AdamW` |
| Learning rate | `6e-4` |
| Learning rate scheduler | `wsd` (warmup 200 → constant peak → linear decay over last 25%) |
| Training epochs | `1` (at chck_100M; full 5-epoch budget allowed but stopped at 1 epoch / 100M ref tokens) |
| Batch size | `128` (16 micro-batch × 2 grad-accum × 4 GPUs) |
| Random seed | `12` |
| FLOPs (estimate) | leave blank or `0` if no estimate |

### Compute
| Field | Value |
|---|---|
| GPU developed | `A10G` |
| GPU trained | `4× NVIDIA A10G (23 GB each, bf16, DDP)` |
| Training data | `BabyLM-community/babylm-{eng,nld,zho}` (BabyBabelLM 2026 100M tier) |
| Total training words | `346598931` (raw); `100000000` (byte-premium-adjusted reference tokens consumed at chck_100M) |

### Method narrative — free-text fields

**Main contributions (one or two lines):**
> Byte-premium-uniform trilingual baseline; joint BPE 32k tokenizer trained on the EN+NL+ZH mixture; deficit-driven token-share sampler so the mixture targets equal *reference tokens* per language rather than equal *documents*. WSD (warmup-stable-decay) schedule trained for 1 epoch (100M ref tokens consumed, well within the BabyLM ≤10-epoch cap).

**Data genre**
> Multilingual children's developmental + curated subtitle/educational mixture (BabyBabelLM 2026 official 100M tier). Per-language audit in `docs/data_card.md`: EN ~80% genuinely child-genre (CHILDES, BNC, Project Gutenberg, Simple Wikipedia child-wiki, Switchboard); NL 63% OpenSubtitles padding + 17% high-school exam prose + ~21% genuine child-genre; ZH 66% WenetSpeech subtitles + ~24% genuine child-genre + 10% educational/exam.

**Data preprocessing**
> Full corpora loaded into memory and shuffled at training time (the Hub layout is category-clustered; streaming with reasonable buffers produces a biased sample — verified empirically in `docs/audit/`). No additional filtering, no dedup beyond what the organisers shipped. Eval-leakage screen against BLiMP-filtered, BLiMP-NL, ZhoBLiMP (118 configs), MultiBLiMP, HellaSwag-MuBench, XCOMPS, SIB-200 found 3/595 EN substring matches (Wikipedia structural overlap) and 0 NL / 0 ZH; benchmarks are clean.

**Human annotation / synthetic augmentation**
> None. No human relabelling, no synthetic data generation, no teacher distillation.

**Other hyperparameters**
> bf16 mixed precision; AdamW betas (0.9, 0.95); weight decay 0.1; gradient clip 1.0; LR floor 6e-5; warmup 200 steps; WSD decay tail 25% of total steps; max_seq_length 1024; finetune eval used `--lr 5e-5 --bsz 64 --max_epochs 10 --patience 3`; gradient checkpointing OFF.

**Model description (long)**
> Llama-style decoder (RoPE, RMSNorm, SwiGLU, no biases, tied embeddings) of 110M parameters, pretrained from scratch on 100M reference tokens of the BabyBabelLM 2026 trilingual corpus (EN/NL/ZH) using a custom token-share-deficit-driven byte-premium-uniform sampler. Tokenizer is a joint byte-level BPE (32 768 vocab) trained on a balanced 50M-char sample per language; this is required because ZH is 6.8% Latin script (English embedded in subtitles) and a per-language tokenizer would discard that signal. The schedule is WSD (warmup-stable-decay): 200-step linear warmup, constant 6e-4 LR through 75% of training, linear decay to 6e-5 over the last 25%. Training stopped at chck_100M (= 1 epoch over the corpus, well within the BabyLM ≤10-epoch cap) when the WSD schedule reached its floor. Full audit, scaffold, ablation configs, iteration log, and per-task eval logs at https://github.com/silvererudite/bb-lm-challenge-sub.

**Teacher models**
> None.

## Pre-submission verification

The submission JSON contains every key the leaderboard's `TasksMultilingual` enum expects:

| Section | Tasks | Status |
|---|---|---|
| EN zero-shot (5) | blimp_babylm_filtered, hellaswag_en_mubench, multiblimp_eng, winogrande_en_mubench, xstorycloze_en_mubench | ✓ |
| NL zero-shot (6) | blimp_nl, hellaswag_nl_mubench, multiblimp_nld, winogrande_nl_mubench, xcomps_nl, xstorycloze_nl_mubench | ✓ |
| ZH zero-shot (5) | hellaswag_zh_mubench, winogrande_zh_mubench, xcomps_zh, xstorycloze_zh_mubench, zhoblimp | ✓ |
| Finetune (8 benchmarks) | arc, belebele, bmlama, mnli, sib200, truthfulqa, xnli, include — keyed by lang code | ✓ |

24 task entries total. No missing keys.

The model `Shamima/babylm-2026-multilingual-uniform-100M-v2` is **public** on the
Hub with 20 revisions (chck_1M…chck_100M, plus main). The leaderboard requires
the model be public.

## Projected leaderboard rank

Computed locally using the leaderboard's exact aggregation
(`per-lang avg = mean of all 12/13/13 tasks; ML avg = (en+nl+zh)/3`):

| Group | EN avg | NL avg | ZH avg | **ML avg** |
|---|---:|---:|---:|---:|
| Our v2 | **38.45** | **40.05** | **38.56** | **39.04** |
| gpt2-baseline-en_nld_zho_equal | (incomplete) | (incomplete) | (incomplete) | ~30 |
| gpt2-baseline-en_nld_equal | 45.17 | 45.97 | 0 | 30.38 |
| gpt2-baseline-en_zho_equal | 40.22 | 0 | 41.64 | 27.29 |
| gpt2-baseline-nld_zho_equal | 0 | 40.96 | 36.39 | 25.78 |
| b0-seed42-zho-nld-en | 25.95 | 25.11 | 19.20 | 22.09 |
| taam_v2-seed42-zho-nld-en | 21.62 | 24.85 | 19.63 | 22.03 |

We project **rank 1** on the multilingual track at submission time, because
ours is the only entry covering all three languages well. The
`gpt2-baseline-en_nld_zho_equal` row in the screenshot was cut off but
projected at ~30 — also less than us, because their finetune scores per the
README baseline tables were 28.34 / 24.78 / 27.23 (vs ours ~37 / ~30 / ~33).

## Files in this directory

- `v2_submission.json` — the `*_submission.json` upload
- `v2_predictions.json` — the `*_predictions.json` upload
- `v2_metadata.md` — this file (form fields + projection)

## Notes for after submission

1. The Space `BabyLM-Leaderboard-2026` was in a Restarting state when we
   prepared this — wait for it to come back online before opening the form.
2. The submit form writes to two HF datasets
   (`leaderboard-2026-all-requests` and `leaderboard-2026-all-results`)
   under your HF username. After submission, those datasets should appear
   non-empty when queried, and the leaderboard table should show our entry
   within a few minutes.
3. Save the request UUID / submission timestamp the form returns; we'll
   reference it in the workshop paper.
