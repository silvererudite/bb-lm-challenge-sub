# BabyBabelLM 2026 — Per-Language Data Card (Audit)

**Date:** 2026-06-05
**Source datasets:** `BabyLM-community/babylm-eng` (sha `b78a9336…`), `babylm-nld` (sha `1aa063f7…`), `babylm-zho` (sha `600a6657…`).
**Method:** the *entire* corpus loaded into memory and tabulated; not a sample. Cross-validated against the README's published per-category counts.

## TL;DR — three numbers that should anchor every modelling decision

1. **EN is the only corpus that is genuinely developmental.** ~80% of EN tokens are child-directed speech, child books, child wiki, or child-available speech. NL is dominated by OpenSubtitles padding (63%) and ZH by WenetSpeech subtitles (66%). Calling the trilingual training "developmental" without controlling for register is misleading.
2. **ZH is 6.8% Latin script** (~9.3M tokens) — embedded English in subtitles. A Chinese-only tokenizer or one that under-allocates vocab to Latin will lose information; a joint tokenizer is the clean choice.
3. **Eval leakage is essentially nil**: 3 SIB-200 sentences (out of 595 EN probes) appear in EN child-wiki because both come from Wikipedia. NL: 0/399. ZH: 0/1,764. The benchmarks are honest.

## Schema (gated repos)

Each document carries: `text`, `doc-id`, `category`, `data-source`, `script` (ISO 15924), `age-estimate`, `license`, `misc`, `num-tokens` (whitespace for EN/NL, Qwen3-0.6B for ZH), `language` (ISO 639-3). The Beetle-Data mirrors are `text/language/source` only.

## English (`BabyLM-community/babylm-eng`)

- **137,710 docs · 98,878,321 raw tokens · 539 MB**
- **Byte premium: 1.000** (reference)
- **Document chars: median 396, p95 16,993, max 6,582,988** — extreme right tail (single Gutenberg books)
- **Script: 100% Latn**
- **Age estimate: 100% `n/a`**
- **Exact-prefix dup rate (first 300 chars): 0.04%**

| Category | Tokens | Token share |
|---|---:|---:|
| child-directed-speech | 27,712,538 | 28.0% |
| child-books | 26,748,028 | 27.1% |
| padding-opensubtitles | 19,699,367 | 19.9% |
| child-wiki | 14,609,286 | 14.8% |
| child-available-speech | 9,102,166 | 9.2% |
| padding-wikipedia | 1,006,936 | 1.0% |

| Top sources by tokens | Token share |
|---|---:|
| CHILDES | 28.0% |
| Project Gutenberg | 26.7% |
| OpenSubtitles | 19.9% |
| Simple Wikipedia | 13.9% |
| BNC | 7.9% |
| Switchboard | 1.4% |
| Wikipedia | 1.0% |
| vikidia | 0.9% |
| GlotStoryBook | 0.3% |
| Ririro | 0.1% |

## Dutch (`BabyLM-community/babylm-nld`)

- **304,611 docs · 109,885,564 raw tokens · 569 MB**
- **Byte premium: 1.0516** (NL needs ~5% more bytes per equivalent content)
- **Document chars: median 289, p95 5,942, max 2,049,911**
- **Script: 100% Latn**
- **Age estimate: 59% `n/a`, then `10-12` (6.5%) · `8` (5.6%) · `9` (4.9%) · `7-9` (4.5%)**
- **Exact-prefix dup rate: 0.65%** (highest of the three)

| Category | Tokens | Token share |
|---|---:|---:|
| **padding-opensubtitles** | 69,035,414 | **62.8%** |
| educational | 19,146,045 | 17.4% |
| child-wiki | 9,673,449 | 8.8% |
| child-books | 4,576,823 | 4.2% |
| child-directed-speech | 3,304,756 | 3.0% |
| child-news | 3,177,743 | 2.9% |
| subtitles | 971,334 | 0.9% |

| Top sources by tokens | Token share |
|---|---:|
| OpenSubtitles | 62.8% |
| BasiLex | 10.1% |
| wikikids | 8.8% |
| wikiwijs | 7.6% |
| examenblad.nl (high-school exams) | 5.6% |
| jeugdjournaal.nl (children's news) | 1.3% |
| scholieren.com | 0.7% |
| Ririro | 0.1% |

## Chinese (`BabyLM-community/babylm-zho`)

- **203,891 docs · 137,835,046 raw tokens · 519 MB**
- **Byte premium: 0.9360** (ZH packs more content per byte than EN)
- **Document chars: median 309, p95 4,825, max 80,765**
- **Script: 93.2% Hans, 6.8% Latn, 0.03% Hani** — note the substantial Latin-script content
- **Age estimate: 64% `n/a`, then `5` (11.6%) · `6-18` (8.6%) · `6-12` (5.9%) · `4` (5.2%) · `3` (2.4%)**
- **Exact-prefix dup rate: 0.81%**

| Category | Tokens | Token share |
|---|---:|---:|
| **subtitles** | 91,328,001 | **66.3%** |
| child-books | 15,976,688 | 11.6% |
| educational | 13,465,351 | 9.8% |
| child-directed-speech | 9,636,590 | 7.0% |
| child-available-speech | 7,403,441 | 5.4% |
| child-wiki | 24,975 | <0.1% |

| Top sources by tokens | Token share |
|---|---:|
| WenetSpeech (subtitles) | 66.3% |
| quangushi (children's stories) | 7.6% |
| ck12 (educational) | 5.4% |
| NaturalConv | 4.0% |
| CMRC | 3.6% |
| CompositionCorpus | 1.6% |
| CTS Conversational Telephone Speech | 1.4% |
| CSQ | 1.0% |
| GAOKAO | 1.0% |
| FCGEC | 0.4% |
| cft-zh | 0.3% |
| wikibook | 0.3% |
| ChildMandarin | 0.2% |

## Eval-leakage screen

For each language we pulled text fields from every relevant evaluation dataset (BLiMP-filtered, BLiMP-NL, ZhoBLiMP across 118 configs, MultiBLiMP, HellaSwag-MuBench, XCOMPS, SIB-200) and looked for substring containment in the full training corpora.

| Language | # probes | total hits | per set |
|---|---:|---:|---|
| eng | 595 | 3 | sib200_eng: 3 |
| nld | 399 | 0 | — |
| zho | 1,764 | 0 | — |

The 3 EN hits are SIB-200 sentences that appear in `child-wiki` (Simple Wikipedia / vikidia) — both eval and training pull from Wikipedia, so this is structural rather than benchmark contamination, and the 0.5% rate is too small to materially shift scores.

## Document-length distributions (chars)

| | EN | NL | ZH |
|---|---:|---:|---:|
| mean | 3,912 | 1,866 | 962 |
| median | 396 | 289 | 309 |
| p95 | 16,993 | 5,942 | 4,825 |
| max | 6,582,988 | 2,049,911 | 80,765 |

Heavy right tails for EN/NL (whole books). At seq_len = 1024, packing concatenates many docs per sequence; at seq_len = 4096 a single Gutenberg book may dominate one sequence. Default to **seq_len 1024** with `<eos>` between docs.

## Operational notes for training

- **The corpora are NOT shuffled at-rest on the Hub.** A 20k-buffer streaming shuffle stays inside one category for thousands of records (we observed it concretely for all three languages). Either pre-shuffle the entire dataset to disk, or use a multi-million-record shuffle buffer, or load fully into memory (~500 MB / lang × 3 = 1.5 GB — trivial on this node). Failing to do this means the *first* slice of training sees one category, and then the next, etc., which biases early-checkpoint evaluations and the entire `chck_1M…chck_10M` fast-eval curve. **Decision: load full corpora into memory and shuffle there. The streaming code path stays for scalability but is not used for the trilingual baseline.**
- **`num-tokens` is a per-language tokenizer's count.** EN/NL: whitespace. ZH: Qwen3-0.6B subwords. When we apply our own joint BPE, the actual token count on disk will differ. The "100M reference token" cap is enforced on the dataset's `num-tokens` field, byte-premium-adjusted — *not* on our tokenizer's output. This keeps our budget directly comparable to the README/baseline numbers.
- **NL is mostly subtitle text.** A schedule that wants "developmental NL" needs to category-filter NL down to `child-*` only — at most ~21M tokens — and accept the smaller budget for that cell.
- **ZH has 9.3M Latin-script tokens.** A joint tokenizer is the right choice; a per-language ZH tokenizer that ignores Latin will produce mostly `<unk>` for that 6.8%.
- **Age-estimate field is mostly `n/a`** for EN (100%) and ZH (64%) but rich for NL (40% labelled). If a follow-up wanted developmental-stage curricula, NL is the only language with usable age signals at scale.

## Cross-language comparability — the most important audit finding

| | EN | NL | ZH |
|---|---:|---:|---:|
| Genuine child-genre share | ~79% | ~19% | ~24% |
| Subtitle / "padding subtitles" share | 20% | 63% | 66% |
| Educational (exam prose / school books) | 0% | 17% | 10% |
| Reference tokenizer | whitespace | whitespace | Qwen3-0.6B |

Implications for the experimental design (these update the original `project_research_direction.md` claims):

1. **The 3×3 transfer matrix (C1) is fundamentally measuring the curse of multilinguality + register asymmetry, not language identity in isolation.** We should add a "register-controlled" cell to the Pareto sweep: train EN on a subtitle-heavy reweighting of `babylm-eng` (boost OpenSubtitles to ~63% to match NL) and see whether monolingual-EN scores fall toward trilingual numbers. If yes, register dominates language; if no, the curse is genuinely cross-lingual.
2. **The cognitively-plausible bilingual schedule (C2) needs a "developmental-only" sub-corpus.** Filter to `child-*` categories, accept the smaller budget (~80M EN, ~21M NL, ~24M ZH), make the simultaneous-bilingual story honest.
3. **Tokenizer ablation matters more than originally framed.** Joint BPE 32 768 is the default; a `Qwen3` tokenizer adapted with EN+NL merges, or a Unigram alternative, are reasonable ablation cells.
4. **Pre-shuffle the entire corpus to disk before training.** The Hub layout is category-clustered.

## Saved artifacts

- `docs/data_card.md` — this file
- `docs/audit/{eng,nld,zho}/dataset_README.md` — public dataset cards
- `docs/audit/{eng,nld,zho}/dataset_metadata.json` — gated metadata
- `docs/audit/full_audit.json` — full-corpus audit (numbers above)
- `docs/audit/empirical_audit.json` — early streaming sample, kept for reference
- `docs/audit/empirical_audit_shuffled.json` — illustrates the at-rest non-shuffle problem
- `docs/audit/eval_leakage.json` — EN/NL leakage screen
- `docs/audit/eval_leakage_zh.json` — ZH leakage screen across 118 ZhoBLiMP configs
- `docs/audit/{eng,nld,zho}/empirical_samples.json` — 2 sample documents per category
