# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A submission to the **BabyLM 2026 Challenge (4th edition)**, **MultiLingual track**, co-located with EMNLP 2026 in Budapest.

**Quality bar (user-stated):** *genuinely good research contribution + competitive leaderboard rank.* Both must hold — not novelty-only, not benchmark-hacking. The best Pareto configuration from our ablation grid becomes the leaderboard entry; the science is the workshop paper.

## Hard rules (these bind every design decision)

- **Languages:** English, Dutch, Chinese. Evaluated on BabyBabelLM.
- **Data budget:** 100M tokens total, custom mixture, **byte-premium-adjusted** per language.
- **Compute cap:** ≤10 epochs over training data. This is the binding constraint, not GPU hours.
- **Submission deadline:** **2026-07-15** (direct, OpenReview). Camera-ready early September.
- Multimodal pairs and teacher-model feedback are explicitly allowed in Strict/Strict-Small. Verify the rule text before assuming the same for MultiLingual; it is not currently part of our plan but worth knowing.

## Research direction (decided)

**Working title:** *A Mini-ATLAS for the Developmental Regime: cross-lingual transfer matrices and budget allocations at the BabyLM 100M scale, anchored in cognitively-plausible bilingual schedules.*

Transpose ATLAS-style multilingual transfer measurement (Longpre et al., ICLR 2026, arxiv 2510.22037 — 1,444 language pairs, 10M–8B params) to the BabyLM regime, and contrast its purely-empirical recommendations with psycholinguistically-motivated bilingual exposure schedules.

Three contributions, in order of priority:

1. **3×3 EN/NL/ZH transfer matrix** at fixed compute — does the curse of multilinguality hold at the developmental scale ATLAS never measured? Does Dutch bridge English↔Chinese as typology predicts?
2. **Mixture-allocation Pareto sweep** at fixed 100M-token budget: byte-premium-uniform · loss-weighted (Upsample-or-Upweight, NAACL 2025) · simultaneous-bilingual ~60/40 · Zipf/CDS-weighted · typological-bridge curriculum (EN → +NL → +ZH).
3. **Per-language data card + provenance audit** of the composed mixture — Longpre's "audit before you train" signature.

## Methodological signatures to keep (Longpre-style)

- **Hold model and tokens constant; sweep data-side knobs.** Architecture ablations are not the contribution.
- **Report per-language deltas, not just aggregate scores.** Transfer-matrix style.
- **Frame results as Pareto trade-offs, not single winners.**
- **Treat the data card as a first-class artifact, not an appendix.**
- **One shared scaffold runs every cell of the grid.** No bespoke training scripts per ablation.
- **Release the ablation CSV, data card, training code, and checkpoints.** Half the contribution.

## Authoritative sources (re-fetch; the site updates as the challenge progresses)

- Challenge homepage: https://babylm.github.io/
- Eval pipeline: https://github.com/babylm-org/babylm-eval
- Training data: BabyLM Hugging Face community (BabyBabelLM)
- Leaderboard: HF Spaces, linked from the homepage

## Status (as of 2026-06-04)

Repo is empty. **Scouting phase:** (1) pull and verify the eval pipeline, (2) audit BabyBabelLM data per-language, (3) brief survey of BabyLM 2024/2025 winners and the BabyBabelLM paper, (4) confirm GPU model/count on this SageMaker node. The training scaffold is blocked on these. See `TaskList` for the live plan.

The April 2026 eval-pipeline release implies tooling may have rough edges; expect to file issues upstream and pin commits.

## Persistent memory

User profile, project context, and the full research-direction rationale live in `~/.claude/projects/-home-sagemaker-user-babylm-challenge-submission/memory/`. Read `MEMORY.md` for the index. Update `project_research_direction.md` and the task list as the plan evolves; do not re-derive decisions already recorded there.
