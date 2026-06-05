# baseline-uniform-100M v2 — eval scores (WSD schedule)

**Model:** `Shamima/babylm-2026-multilingual-uniform-100M-v2` (HF Hub, public).
**Checkpoint:** `chck_100M` — same comparison point as v1 (1 epoch over the corpus, 100M ref tokens consumed).
**Eval ran:** 2026-06-05.

## Headline (zero-shot multilingual group accuracies)

| Group | v1 (cosine) | v2 (WSD) | Δ v2-v1 | GPT-2 trilingual baseline | Δ v2-baseline |
|---|---:|---:|---:|---:|---:|
| zeroshot_eng | 0.5512 | **0.5477** | −0.35 | 0.5634 | −1.57 |
| zeroshot_nld | 0.5214 | **0.5192** | −0.22 | 0.5781 | −5.89 |
| zeroshot_zho | 0.5603 | **0.5610** | +0.07 | 0.5134 | +4.76 |
| **avg** | **0.5443** | **0.5426** | **−0.17** | 0.5516 | −0.90 |

## Honest reading

**The WSD schedule did NOT help at 1-epoch / 100M ref tokens.** v2 is
indistinguishable from v1 within stderr (~0.0016 per language). The hypothesis
was that v1 plateaued at loss 3.43 because cosine decayed to 10% of peak
prematurely. We now know that's false: v2 at the same checkpoint reaches loss
3.74 (higher than v1, because v2's LR was constant-peak through 75% of training
rather than decaying), and downstream performance is the same.

So the bottleneck at this scale is **NOT the schedule shape**; it's something
else. Two leading candidates:

1. **Data composition — specifically NL.** The audit showed NL is 63% OpenSubtitles
   padding and only ~19% genuinely developmental. We're losing 5.7pt on NL vs
   the GPT-2 baseline. The published baseline likely got more NL-developmental
   exposure indirectly via different mixture weights or category sampling.
2. **Training-token / parameter ratio.** 110M params × 100M ref tokens = ~0.9
   Chinchilla-equivalent. The published baselines are also at this ratio so
   this can't fully explain a 0.9pt gap. But it does explain why neither v1
   nor v2 saturates downstream tasks.

## Per-task numbers

EN (zeroshot_eng = 0.5477):
- `blimp_babylm_filtered`: ~0.591 (vs v1 0.594; vs baseline 0.692)
- `multiblimp_eng`, `winogrande_en_mubench`, `xstorycloze_en_mubench`, `hellaswag_en_mubench`: see results_*.json

NL (zeroshot_nld = 0.5192):
- `blimp_nl`: ~0.728 (same as v1)
- `multiblimp_nld`: ~0.85
- `xcomps_nl`: ~0.51
- `hellaswag_nl_mubench`, `winogrande_nl_mubench`, `xstorycloze_nl_mubench`: ~0.5 / chance level

ZH (zeroshot_zho = 0.5610):
- `zhoblimp`: ~0.66 (still beating the GPT-2 trilingual baseline 0.7544 → wait actually not, see note)
- HellaSwag/Winogrande/XCOMPS/XStoryCloze: see json

## What's in this directory

- `train.log` — full v2 training log (8.84 → 3.74 over ~4300 steps, WSD shape visible in LR column)
- `eval.log` — full multilingual zero-shot eval (3 languages)
- `results_2026-06-05T13-{33,36,41}-*.json` — per-language lm-eval result dumps

## Decision for v3

**Stop chasing schedule.** The right next bet, per the data card audit, is to
attack the NL gap directly:
- **v3a (NL category-filter)**: drop the 63% OpenSubtitles padding from NL,
  use only `child-*` + `educational` + `child-news` (~21M NL tokens);
  combine with EN/ZH at category-balanced shares; allow 5 epochs to use the
  rule-permitted compute on quality data.
- **v3b (NL-up-weight)**: keep all categories but shift the mixture to e.g.
  EN 0.30 / NL 0.45 / ZH 0.25 — give NL the data-time it needs.

These are exactly two cells in the C2 Pareto sweep (#8). Doing them now
gives the leaderboard a legitimate first submission and slots the same runs
into the paper's contribution.
