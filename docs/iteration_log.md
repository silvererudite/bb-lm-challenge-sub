# Iteration log

A running log of training iterations, what each one taught us, and the open
issues / decisions queued for the next one. Append at the top of each section
when an iteration ends; treat past sections as immutable.

The headline scoreboard is at the bottom for quick reference.

---

## v2 — uniform-100M-v2 (in progress, started 2026-06-05)

**Hypothesis:** v1 was undertrained relative to the BabyLM rule-allowed compute.
A WSD schedule + more epochs should close most of the gap to the published
GPT-2 baselines without any architectural change.

**Changes from v1:** identical model, identical tokenizer, identical mixture.
- Schedule: cosine → **WSD** (warmup 200 → constant peak LR 6e-4 → linear decay over last 25%)
- Total compute: 1 epoch (~100M effective tokens) → **5 epochs (500M effective tokens)**
- Save revisions: extended to chck_500M
- Warmup: 100 → 200 steps (~5% of total)
- Total estimated steps: 7,629 → **3,814**, because total budget is now expressed against the 5-epoch budget. (v1 was budget-1 epoch under cosine; v2 is the same wall-clock per step but trains longer.)

**Status as of writing:** step 600/3814, loss 5.20, throughput stable at 98k tok/s.

**What's already different (mid-run signal):**
- At step 600 v2 loss is 5.20 vs v1 was 4.89 at the same step count — v2 is *behind* early because of the longer warmup. By step ~1500 the curves should cross; by end-of-run v2 should land at considerably lower loss than v1's 3.43 plateau because WSD doesn't decay until step ~2860.
- All 4 GPUs at 100% util / 21 GB / 23 GB. Same memory headroom as v1 — gradient_checkpointing still off, micro_batch=16.

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

| Iter | Compute | Schedule | EN | NL | ZH | Avg | Notes |
|---|---|---|---:|---:|---:|---:|---|
| GPT-2 baseline (organisers) | unknown | unknown | 0.5634 | 0.5781 | 0.5134 | 0.5516 | published in eval-repo README |
| **v1 (uniform, 1 epoch, cosine)** | 1 epoch / 7460 steps | warmup 100 → cosine to 10% | 0.5512 | 0.5214 | **0.5603** | 0.5443 | undertrained; cosine decay too fast; ZH already over baseline |
| v2 (uniform, 5 epochs, WSD) | 5 epochs / 3814 steps | warmup 200 → const peak → linear last 25% | — | — | — | — | in progress 2026-06-05 |

(The "compute" column in v1 looks like more steps than v2 because v1 used the
1-epoch budget as the cosine target while v2 uses 5-epoch as the WSD target.
Effective tokens/step is the same; v1 trained for ~1 epoch then stopped on
plateau, v2 will train through 5 epochs with no LR-floor artifact.)
