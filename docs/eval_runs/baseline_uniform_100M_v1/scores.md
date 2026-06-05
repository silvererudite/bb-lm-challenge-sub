# baseline-uniform-100M v1 — eval scores

**Model:** `Shamima/babylm-2026-multilingual-uniform-100M` (HF Hub, public).
**Checkpoint:** `chck_100M` (= 1 epoch over the 100M corpus, byte-premium-uniform mixture).
**Eval ran:** 2026-06-05.

## Headline (zero-shot multilingual group accuracies)

| Group | Ours v1 | GPT-2 trilingual baseline (README) | Δ |
|---|---:|---:|---:|
| zeroshot_eng | **0.5512** ± 0.0016 | 0.5634 | −1.22 pt |
| zeroshot_nld | **0.5214** ± 0.0024 | 0.5781 | **−5.67 pt** |
| zeroshot_zho | **0.5603** ± 0.0016 | 0.5134 | **+4.69 pt** |
| **avg** | **0.5443** | 0.5516 | −0.73 pt |

## Per-task highlights

EN:
- `blimp_babylm_filtered`: 0.5944 (vs gpt2-trilingual 0.6918; this is the dominant subtask, dragging down the average)
- `multiblimp_eng`: see per-task json
- `winogrande_en_mubench`, `xstorycloze_en_mubench`, `hellaswag_en_mubench`: see json

NL:
- `blimp_nl`: 0.7280
- `multiblimp_nld`: 0.8537
- `xcomps_nl`: 0.5081
- `hellaswag_nl_mubench`: 0.2595 / `winogrande_nl_mubench`: 0.4971 / `xstorycloze_nl_mubench`: 0.4853

ZH:
- `zhoblimp`: 0.6595 (this is the big driver of the +4.7pt ZH advantage)
- (See `results_*.json` for the full ZhoBLiMP paradigm-level grid.)

## Why we're iterating before submitting

- We stopped at **17% of the 1B effective-token cap** (1 epoch + slack) when the loss had only just reached its cosine-floor plateau. The model is undertrained relative to the rule-permitted compute.
- The cosine schedule decayed to 10% of peak by step ~7000 while loss was still falling — too aggressive a decay tail.
- v2 plan: same 110M Llama, same joint BPE 32k, same byte-premium-uniform mixture; switch to **WSD schedule** (warmup → constant peak → linear decay over last 25%) and run **5 epochs** (500M effective tokens, ~5h on 4× A10G). Also save revisions through `chck_500M`.

## Files in this directory

- `train.log` — the full v1 training log (8.83 → 3.42 over ~7460 steps)
- `lm_eval_full.log` — the full multilingual zero-shot eval log (EN+NL+ZH)
- `results_2026-06-05T*.json` — per-language lm-eval result dumps (one per language, contains every paradigm-level subtask)
