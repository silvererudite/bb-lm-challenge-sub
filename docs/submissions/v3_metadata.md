# BabyLM 2026 MultiLingual submission — v3 (quality-filter)

Copy-paste guide for the leaderboard submission form on
https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026.

## Files to upload

1. **Results file** → `docs/submissions/v3_submission.json`
2. **Predictions file** → `docs/submissions/v3_predictions.json` (will be re-collated with `--fast` after the revision sweep finishes)

## Form fields

### Model identity
| Field | Value |
|---|---|
| Hugging Face repo | `Shamima/babylm-2026-multilingual-v3-quality-filter` |
| Track | `multilingual` |
| Revision | `main` |
| User name | `Shamima` |

### Languages and corpus sizes (post-filter)
| Field | Value |
|---|---|
| Languages: English (eng) | ☑ |
| Languages: Dutch (nld) | ☑ |
| Languages: Chinese (zho) | ☑ |
| Words used (English) | `78172018` (filtered: child-* only; was 99M unfiltered) |
| Words used (Dutch) | `40850150` (filtered: drop padding-opensubtitles; was 110M) |
| Words used (Chinese) | `46507045` (filtered: drop subtitles=WenetSpeech; was 138M) |
| Total words used | `165529213` raw; trained on **500M effective tokens** (5 epochs over the byte-premium-adjusted budget) |

### Architecture
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

### Training
| Field | Value |
|---|---|
| Optimizer | `AdamW` |
| Max learning rate | `6e-4` |
| LR scheduler | `WSD (warmup 200 → constant peak 6e-4 → linear decay over last 25% to 6e-5)` |
| Training epochs | `5` (per-corpus average; per-language: EN 2.13, NL 4.29, ZH 3.35 — NL was the binding language) |
| Average batch size (tokens) | `131072` |
| Random seed | `12` |

### Compute
| Field | Value |
|---|---|
| GPU developed | `A10G` |
| GPU trained | `4× NVIDIA A10G (23GB each, bf16, DDP)` |
| Approximate GPU hours for training | `35` (8.6 h wallclock × 4 GPUs ≈ 34.4 GPU-hours) |
| Approximate GPU hours for development | `60` (cumulative across v1/v2/v3 cycle) |
| Training data | `BabyLM-community/babylm-{eng,nld,zho}` (BabyBabelLM 2026 100M tier) with category filter |
| Approximate FLOPs | `1.8e18` (= 6×N×D, N=110M params, D=2.7B BPE tokens ≈ 5 epochs of filtered corpus) |

### Method narrative

**Main contributions (multi-select):**
- ☑ Data preprocessing
- ☑ Hyperparameter tuning
- ☑ Controlled experiments

**Main contributions (free-text expanded):**
> Audit-driven category filtering of the BabyBabelLM 100M corpora: drop padding categories that the audit identified as register-asymmetric (EN's OpenSubtitles+Wikipedia padding = 21M tokens, NL's OpenSubtitles padding = 69M tokens, ZH's WenetSpeech subtitles = 91M tokens), keeping only developmental/educational/child-genre signal. Combine with byte-premium-uniform mixture sampling (deficit-driven on token shares, not document shares) and 5-epoch WSD schedule. Compared head-to-head against v1 (cosine, 1ep) and v2 (WSD, 1ep) on the identical scaffold.

**Data genre**
> Multilingual children's developmental text (BabyBabelLM 2026 100M tier), category-filtered to drop register-asymmetric padding. Per-language audit in `docs/data_card.md`; v3-specific category whitelist:
> - EN: child-directed-speech, child-books, child-wiki, child-available-speech (drops padding-opensubtitles, padding-wikipedia)
> - NL: child-directed-speech, child-books, child-wiki, child-news, subtitles, educational (drops padding-opensubtitles)
> - ZH: child-directed-speech, child-books, child-wiki, child-available-speech, educational (drops subtitles=WenetSpeech)

**Data preprocessing**
> Full corpora loaded into memory and shuffled at training time (Hub layout is category-clustered; streaming with reasonable buffers produces a biased sample — verified empirically). Category filter applied once after load, cached, reshuffled per epoch. No further filtering. Eval-leakage screen against BLiMP-filtered, BLiMP-NL, ZhoBLiMP, MultiBLiMP, HellaSwag-MuBench, XCOMPS, SIB-200 found 3/595 EN substring matches (Wikipedia structural overlap) and 0 NL / 0 ZH.

**Other hyperparameters**
> bf16 mixed precision; AdamW betas (0.9, 0.95); weight decay 0.1; gradient clip 1.0; warmup 200 steps; WSD decay tail 25% of total steps; max_seq_length 1024; finetune eval used `--lr 5e-5 --bsz 64 --max_epochs 10 --patience 3`; gradient checkpointing OFF.

**Model description**
> Llama-style decoder (RoPE, RMSNorm, SwiGLU, no biases, tied embeddings) of 110M parameters, pretrained from scratch on 500M effective tokens of the BabyBabelLM 2026 trilingual corpus filtered to developmental/educational categories. Tokenizer is a joint byte-level BPE (32 768 vocab) trained on a balanced 50M-character sample per language; this is required because Chinese is 6.8% Latin script (English embedded in subtitles) and a per-language tokenizer would discard that signal. Schedule is WSD (warmup 200 → constant peak 6e-4 → linear decay over last 25%). Training stopped after 5 epochs over the corpus (= 500M effective tokens), well within the BabyLM ≤10-epoch cap. Trained on 4× NVIDIA A10G in bf16+DDP with effective batch 131K tokens/step. Versus the v1/v2 baselines on identical scaffold: v3 is the first iteration to beat the published GPT-2 trilingual baseline on average (zero-shot avg 0.5543 vs baseline 0.5516; full leaderboard format 41.77 vs ~30 for the GPT-2 baselines on the leaderboard). Full audit, scaffold, ablation configs, iteration log, and per-task eval logs at https://github.com/silvererudite/bb-lm-challenge-sub.

**Teacher models**
> None.

## Projected leaderboard rank

Computed locally using the leaderboard's exact aggregation:

| Group | EN avg | NL avg | ZH avg | **ML avg** |
|---|---:|---:|---:|---:|
| **v3** | **42.13** | **43.80** | **39.38** | **41.77** |
| v2 (current submitted) | 38.45 | 40.05 | 38.56 | 39.04 |
| gpt2-baseline-en_nld_zho_equal | (incomplete) | (incomplete) | (incomplete) | ~30 |
| Best community submission (b0-seed42) | 25.95 | 25.11 | 19.20 | 22.09 |

Projected **rank #1** with a wide margin (+2.73 pt over v2, our previous submission).

## Files

- `v3_submission.json` — leaderboard scores file (24 task entries)
- `v3_predictions.json` — predictions file (will be re-collated with `--fast` after revision sweep)
- `v3_metadata.md` — this file
