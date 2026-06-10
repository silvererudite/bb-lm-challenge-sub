# v3 — quality-filter — zero-shot scores

**Model:** `Shamima/babylm-2026-multilingual-v3-quality-filter` (HF Hub, public, 24 revisions).
**Checkpoint:** `main` = `chck_400M` (largest fast-eval milestone we saved during training).
**Trained:** 2026-06-09 → 2026-06-10. 23,295 steps · 8.6 hours wallclock · 500 M effective tokens consumed (the full 5-epoch budget).
**Eval ran:** 2026-06-10. wandb run `0jbm30gf`.

## Headline (zero-shot multilingual group accuracies)

| Group | v1 | v2 | **v3** | Δ v3-v2 | GPT-2 trilingual baseline |
|---|---:|---:|---:|---:|---:|
| zeroshot_eng | 0.5512 | 0.5477 | **0.5766** | **+2.89 pt** | 0.5634 |
| zeroshot_nld | 0.5214 | 0.5192 | **0.5293** | +1.01 pt | 0.5781 |
| zeroshot_zho | 0.5603 | 0.5610 | 0.5571 | −0.39 pt | 0.5134 |
| **avg** | 0.5443 | 0.5426 | **0.5543** | **+1.17 pt** | 0.5516 |

**v3 is now above the published GPT-2 trilingual baseline on average for the first time** (0.5543 vs 0.5516, +0.27 pt). And by a wide margin on EN (+1.32 pt over baseline, after being −1.22 pt below it in v1) and on ZH (+4.37 pt over baseline). NL is still −4.88 pt below — quality filtering only partially closed the NL gap.

## Per-task breakdown

### English

| Task | v2 | v3 | Δ |
|---|---:|---:|---:|
| blimp_babylm_filtered | 0.5898 | **0.6243** | **+3.45** |
| multiblimp_eng | 0.7455 | **0.8091** | **+6.36** |
| hellaswag_en_mubench | 0.2669 | 0.2644 | −0.25 |
| winogrande_en_mubench | 0.5070 | 0.4938 | −1.32 |
| xstorycloze_en_mubench | 0.4799 | 0.4930 | +1.31 |
| **avg (zs only)** | **0.5477** | **0.5766** | **+2.89** |

### Dutch

| Task | v2 | v3 | Δ |
|---|---:|---:|---:|
| blimp_nl | 0.7113 | **0.7327** | +2.14 |
| multiblimp_nld | 0.8550 | **0.8953** | **+4.03** |
| hellaswag_nl_mubench | 0.2596 | 0.2644 | +0.48 |
| winogrande_nl_mubench | 0.5095 | 0.4831 | −2.64 |
| xcomps_nl | 0.5126 | 0.5178 | +0.52 |
| xstorycloze_nl_mubench | 0.4760 | 0.4791 | +0.31 |
| **avg (zs only)** | **0.5192** | **0.5293** | **+1.01** |

### Chinese

| Task | v2 | v3 | Δ |
|---|---:|---:|---:|
| zhoblimp | 0.6597 | 0.6493 | −1.04 |
| hellaswag_zh_mubench | 0.2588 | 0.2639 | +0.51 |
| winogrande_zh_mubench | 0.4946 | 0.5045 | +0.99 |
| xcomps_zh | 0.5235 | 0.5261 | +0.26 |
| xstorycloze_zh_mubench | 0.4520 | 0.4752 | +2.32 |
| **avg (zs only)** | **0.5610** | **0.5571** | **−0.39** |

## Reading

The v3 hypothesis (data quality + 5× compute > schedule shape) was correct.
What's striking is *which* tasks moved:

- **Grammaticality benchmarks dominate the gain.** BLiMP-EN +3.5, MultiBLiMP-EN +6.4, BLiMP-NL +2.1, MultiBLiMP-NL +4.0. These tasks reward syntactic competence that compounds with more passes over clean developmental data. The category filter dropped 21 M EN padding (mostly OpenSubtitles + a tiny Wikipedia slice) and 69 M NL OpenSubtitles padding — exactly what the grammaticality tasks didn't need.
- **Commonsense / discourse benchmarks barely moved.** HellaSwag stayed at chance (0.26), Winogrande oscillated around 0.5. These need either much more data or much bigger models; the filter doesn't help.
- **ZH was the tricky case.** We dropped 91 M WenetSpeech subtitle tokens to match register with EN/NL. xstorycloze_zh +2.3 and the rest roughly flat says the subtitles weren't critical for ZH grammaticality, but **zhoblimp regressed −1.04 pt** — hinting that *some* ZH subtitle exposure was helping ZhoBLiMP specifically. A useful Pareto datapoint for v4.

## Process notes

- The original `eval_and_log.py` script silently lost the result JSONs because the `results/` parent dir was deleted by an earlier `rm -rf` and lm-eval's evaluation tracker logs "Could not save results" as a warning, not a hard error. The numbers in this report were **recovered by parsing lm-eval's stdout table**, not from a results JSON. For finetune + the revision sweep we should `mkdir -p` before the run and verify a results JSON lands.
- The v3 wandb training run is incomplete (the wandb history bug we hit and tried to fix). Curves were captured only in the train-log file; if needed I can backfill them to a fresh wandb run.

## Files

- `train.log` — full v3 training log (8.6 h, ~23k steps)
- `train_summary.json` — the trainer's summary.json
- `run.yaml` — the exact config used
- `zeroshot_eval.log` — the per-language lm-eval stdout (group + per-task tables)
- `zeroshot_recovered.json` — flat `{task: {key: acc}}` dict, the headline numbers above
