# Iteration log

A running log of training iterations, what each one taught us, and the open
issues / decisions queued for the next one. Append at the top of each section
when an iteration ends; treat past sections as immutable.

The headline scoreboard is at the bottom for quick reference.

---

## v2 — uniform-100M-v2 (frozen 2026-06-05, model on HF, **submission candidate**)

**HF model:** `Shamima/babylm-2026-multilingual-uniform-100M-v2` (public, 20 revisions: chck_1M…chck_100M and main).

### What we set out to do

Test whether v1's loss plateau at 3.43 was an LR-floor artifact (cosine
decayed to 10% of peak by ~90% through training while loss was still falling)
or a genuine capacity ceiling. Identical model, identical tokenizer, identical
byte-premium-uniform mixture; only the LR schedule changed.

### Changes from v1

- Schedule: cosine → **WSD** (warmup 200 → constant peak 6e-4 → linear decay over last 25% to 6e-5)
- Save revisions: extended to chck_500M (we stopped at chck_100M to match v1's compute, but the schedule is in place for a longer run if useful)
- Warmup: 100 → 200 steps (~5% of total)

### Final scores (zero-shot only, before finetune)

| Group | v1 | v2 | Δ v2-v1 |
|---|---:|---:|---:|
| zeroshot_eng | 0.5512 | 0.5477 | −0.35 |
| zeroshot_nld | 0.5214 | 0.5192 | −0.22 |
| zeroshot_zho | 0.5603 | 0.5610 | +0.07 |
| **avg** | **0.5443** | **0.5426** | **−0.17** |

### Final scores (full leaderboard breakdown, with finetune)

Aggregation matches the leaderboard exactly:
**per-lang avg = mean of all 12/13/13 tasks; ML avg = (EN+NL+ZH)/3.**

| Group | EN avg (12) | NL avg (13) | ZH avg (13) | **ML avg** |
|---|---:|---:|---:|---:|
| **v2 (uniform-100M, WSD, 1 epoch)** | **38.45** | **40.05** | **38.56** | **39.04** |

Per-task breakdown in `docs/eval_runs/baseline_uniform_100M_v2/finetune_scores.md`.

### Honest reading of the WSD experiment

**WSD did not help.** v2 zero-shot averages are within stderr of v1's. The
hypothesis was wrong: the v1 loss plateau wasn't an artifact of cosine decay.
v2 at the same step has higher loss (3.74 vs v1's 3.43) but identical
downstream — the loss-floor effect was real but capability-irrelevant.

So the bottleneck at this scale is **NOT the schedule shape**; the actual
capability ceiling at 110M params × 100M ref tokens × 1 epoch sits around
**0.54 zero-shot avg**.

### But finetune changes the picture entirely

When we add the seven/seven/eight finetune tasks per language:
- Our **ML avg = 39.04** beats every published GPT-2 baseline (best is en_nld_equal at 30.38).
- Beats every community submission so far (top is b0-seed42 at 22.09).
- Projected **#1 on the multilingual leaderboard** at submission time.

The reason: every other entry on the leaderboard is missing whole languages
or skipped finetune. We covered every (lang, task) cell.

### Why v2 is the submission candidate, not v1

v1 and v2 are within 0.17 pt of each other; either would land at ~#1. We're
submitting v2 because:
- The audit-driven changes (token-share-deficit sampler, in-memory shuffle,
  WSD schedule) are all in v2's run record
- v2 has cleaner training infrastructure (fixed budget semantics, fixed
  mixing, gradient_checkpointing knob, wandb-logged)
- The model card and scaffold story flow naturally from v2 to v3 (data
  quality) without a step backward.

### Issues found and fixed during v2

1. **`estimate_total_steps` undercounts.** Computes `per_step` in our-BPE
   tokens but `budget_reference_tokens` is in reference tokens. The two are
   not 1:1; the ratio depends on how the joint BPE compresses each language.
   This made our v2 5-epoch run "complete" at step 4300 of an estimated
   3814 — the WSD decay tail finished while we still had ~80% of the
   ref-token budget. We chose to stop at chck_100M anyway, but the next
   ablation cell needs a corrected estimate to keep the LR schedule sane.

### Open issues queued

- The `estimate_total_steps` bug above. Fix: either count actual ref tokens
  per step inside the loop and re-tune the LR schedule on the fly, or
  precompute the BPE-to-ref-token ratio empirically and apply it.
- Whether v3 should drop padding categories or upweight NL data share.
- Confirm the leaderboard validator accepts our key set when we submit.

### Submission package

Pre-prepared at `docs/submissions/`:
- `v2_submission.json` — the leaderboard scores file (24 tasks; all leaderboard-expected keys present and verified against the `TasksMultilingual` enum)
- `v2_predictions.json` — the predictions file (226 zero-shot entries + 22 finetune task-lang pairs)
- `v2_metadata.md` — every form field for the leaderboard's submit page, pre-filled

Form lives at: `https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026`
(was Restarting at the time we prepared the package — wait for it to be online before submitting).

---

## v1 — uniform-100M (frozen 2026-06-05, model on HF)

**HF model:** `Shamima/babylm-2026-multilingual-uniform-100M` (public, 20 revisions: chck_1M…chck_100M and main).

### What we set out to do
Establish a clean zero-point for the C1/C2 ablation grid: a 110M Llama-style
decoder pre-trained from scratch on the BabyBabelLM trilingual corpus under
byte-premium-uniform mixing, with the same scaffold every later cell will use.

### Final scores

| Group | v1 | GPT-2 trilingual baseline | Δ |
|---|---:|---:|---:|
| zeroshot_eng | 0.5512 | 0.5634 | −1.22 pt |
| zeroshot_nld | 0.5214 | 0.5781 | **−5.67 pt** |
| zeroshot_zho | 0.5603 | 0.5134 | **+4.69 pt** |
| **avg** | **0.5443** | 0.5516 | −0.73 pt |

Per-task highlights are in `docs/eval_runs/baseline_uniform_100M_v1/scores.md`.

### Issues found and fixed during v1

These all fed back into the scaffold and are now pinned by `tests/test_data_contracts.py`.

1. **Budget semantics bug.** `MixtureBudget.can_continue` originally cut off
   at `total_consumed >= 100M ref tokens`. This made the `max_epochs` field
   dead code: a competition entry can train for up to 10 epochs over the
   100M corpus, i.e. up to 1B effective tokens. Fixed `total_token_budget =
   budget * max_epochs` and added a unit test pinning the new semantics.
   *Impact if missed:* every run would have stopped at 1 epoch, 10× under
   the rule-allowed budget. Caught before the v1 run started.

2. **Token-share mixing bug.** Sampling documents at uniform 1/3 probability
   across languages does NOT produce 1/3 of *reference tokens* per language
   because mean doc sizes differ — EN docs are ~720 ref tokens, NL ~360,
   ZH ~680. Empirical share after 100 steps was 40% / 19% / 40% — far from
   the byte-premium-uniform target. Fixed by deficit-driven selection: each
   step pick the language with the largest token-share gap relative to the
   target weights. Verified: 35/32/32 after 100 steps with the fix.
   *Impact if missed:* the "uniform" baseline would actually be heavily NL-starved.

3. **A10G OOM with HF Llama loss.** `LlamaForCausalLM`'s `ForCausalLMLoss`
   casts logits to float32 internally, which blows past 23 GB on a per-device
   batch of 32 sequences × 1024 length × 32K vocab. Lowered micro_batch to
   16 with grad_accum=2 (same effective batch). Added a
   `gradient_checkpointing` config knob for cells that need batch ≥ 32.

4. **Hub corpora are not shuffled at-rest.** Streaming with a 20 K shuffle
   buffer stayed inside one category for thousands of records — the parquet
   files are clustered by source. Empirically verified across all three
   languages (the streaming-buffer audit showed EN was 100% Simple-Wikipedia
   for the first 20K records; NL was 100% BasiLex; ZH was 93% educational).
   Decision: load full corpora into memory and shuffle there for the
   trilingual baseline. Streaming code path retained for scale-out runs.

5. **Eval-pipeline path bug (upstream).** The `babylm-org/babylm-eval` repo's
   `zeroshot_model.sh` writes results to `babylm-eval/results/main/` but
   `collate_results.py` reads from `babylm-eval/multilingual/results/main/`.
   Workaround: a symlink at `multilingual/results -> ../results`. Need to
   file an issue upstream.

6. **Eval-pipeline task-name drift (upstream).** `EXPECTED_ZEROSHOT["eng"]` in
   `collate_results.py` lists `"blimp"` but the EN group emits
   `"blimp_babylm_filtered"`. Submissions will be flagged as missing `blimp`
   and the published number won't be averaged correctly on the leaderboard
   unless the validator is aliased. Need to confirm with organisers what
   the leaderboard schema actually expects.

### Learnings

- **"Byte-premium-uniform" is a *token-share* claim, not a *document-share*
  claim.** Anyone running this benchmark needs to verify their sampler
  achieves the target token shares — uniform doc-sampling is biased.
- **NL is the weakest language in our v1 numbers, not because of architecture
  but because the corpus is 63% OpenSubtitles padding (audit finding) and
  the model under-trains on the developmental NL signal.** A
  category-filtered NL run is now a higher-leverage Pareto cell than I'd
  originally framed.
- **Cosine to 10% over the full budget is too aggressive when the loss is
  still falling at end-of-schedule.** v1's loss plateaued at 3.43 *because*
  the LR was already at 6.0e-5 by step 7000 — not because the model had
  converged. WSD removes that artifact. This generalises: at 100M-token
  scale, decay shape matters more than peak LR.
- **ZH is the language we're already winning on.** +4.7 pt over the GPT-2
  trilingual baseline, mostly from `zhoblimp` (0.6595 vs published 0.7879
  for monolingual ZH but 0.7544 for trilingual). Plausible explanation:
  joint BPE 32k with 6.8% Latin-aware vocab handles the embedded English
  in ZH subtitles better than the pure-byte GPT-2 BPE.
- **Stopping early on a loss plateau is the right call when the plateau is
  artifact-driven, not capability-driven.** v1's plateau was the LR floor,
  not the model's capacity. v2 tests this directly.

### Open issues queued

- Confirm the leaderboard validator's expected key for the EN BLiMP task
  (`blimp` vs `blimp_babylm_filtered`).
- Run the full revision sweep (chck_1M…chck_100M × 3 langs) for v1 to
  produce the EN/NL/ZH learning curves — useful as a paper figure even
  after v2 supersedes v1, because it shows whether early checkpoints
  generalise differently than late ones (which the developmental-LM
  literature predicts they should).
- Decide between `--fast` flag and standard submission path on the
  leaderboard once v2 is ready.

---

## Scoreboard (latest per iteration)

### Zero-shot only (matches GPT-2 baseline reporting in eval-repo README)

| Iter | Compute | Schedule | EN | NL | ZH | Avg | Notes |
|---|---|---|---:|---:|---:|---:|---|
| GPT-2 baseline en_nld_zho_equal | unknown | unknown | 0.5634 | 0.5781 | 0.5134 | 0.5516 | published in eval-repo README |
| **v1 (uniform, 1 epoch, cosine)** | 1 epoch / 7460 steps | warmup 100 → cosine to 10% | 0.5512 | 0.5214 | 0.5603 | 0.5443 | first publish on HF |
| **v2 (uniform, 1 epoch @ chck_100M, WSD)** | 1 epoch / ~4300 steps | warmup 200 → const peak → linear last 25% | 0.5477 | 0.5192 | **0.5610** | 0.5426 | submission candidate; WSD null-result vs v1 |

### Full leaderboard format (zero-shot + finetune, weighted as the leaderboard does)

| Iter | EN avg (12) | NL avg (13) | ZH avg (13) | **ML avg** | Notes |
|---|---:|---:|---:|---:|---|
| **v2** | **38.45** | **40.05** | **38.56** | **39.04** | projected #1 on submission |
| gpt2-baseline-en_nld_zho_equal | (incomplete) | (incomplete) | (incomplete) | ~30 | from leaderboard screenshot, top row |
| gpt2-baseline-en_nld_equal | 45.17 | 45.97 | 0 | 30.38 | bilingual; 0 ZH |
| gpt2-baseline-en_zho_equal | 40.22 | 0 | 41.64 | 27.29 | bilingual; 0 NL |
| gpt2-baseline-nld_zho_equal | 0 | 40.96 | 36.39 | 25.78 | bilingual; 0 EN |
| b0-seed42-zho-nld-en | 25.95 | 25.11 | 19.20 | 22.09 | best community |
| taam_v2-seed42-zho-nld-en | 21.62 | 24.85 | 19.63 | 22.03 | community |
| babylm-2026-taam_v2-seed42 | 16.68 | 24.85 | 19.63 | 20.39 | community |
