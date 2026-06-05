# BabyBabelLM 2026 — Per-Language Data Card (Audit)

**Date:** 2026-06-05
**Compiled from:** public dataset cards (READMEs) on the HF Hub, plus a streaming sample of the ungated mirror `Beetle-Data/BabyBabel-eng-70M`.
**Status:** Official gated repos `BabyLM-community/babylm-{eng,nld,zho}` are accessible by token but require an additional manual "Access repository" click on each dataset page (gated 403 returned even with a valid write token). Once that one-time approval is granted the full schema (with `data-source`, `category`, `age-estimate`, `num-tokens` per document) becomes usable.

## Schema (full, gated repos)

Each row is one document with these fields:
- `text` — document content
- `doc-id` — unique id
- `category` — content type (e.g. `child-directed-speech`, `child-books`, `child-wiki`, `educational`, `padding-opensubtitles`, `padding-wikipedia`, `subtitles`)
- `data-source` — original corpus name
- `script` — ISO 15924 (e.g. `Latn`, `Hani`, `Hans`)
- `age-estimate` — target age range string
- `license` — per-document license
- `misc` — JSON-string of extras
- `num-tokens` — count by the language's reference tokenizer
- `language` — ISO 639-3

The Beetle-Data mirror is `text/language/source` only — adequate for training but **not** for category-filtered ablations.

## English (`BabyLM-community/babylm-eng`, sha `b78a9336…`)

- **Tier:** 100M
- **Byte premium:** 1.000000 (reference)
- **Total tokens (whitespace):** 98,878,321
- **Documents:** 137,710
- **Reference tokenizer:** whitespace
- **Size on disk:** 539 MB

**Tokens per category**
| Category | Tokens | Share |
|---|---:|---:|
| child-directed-speech | 27,712,538 | 28.0% |
| child-books | 26,748,028 | 27.0% |
| padding-opensubtitles | 19,699,367 | 19.9% |
| child-wiki | 14,609,286 | 14.8% |
| child-available-speech | 9,102,166 | 9.2% |
| padding-wikipedia | 1,006,936 | 1.0% |

**Group rollup**
- Transcription: 36.8M (37%) — the CHILDES-flavoured developmental core
- Books / Wiki / News: 41.4M (42%) — child books and child wiki
- Padding: 20.7M (21%) — added to hit the 100M tier
- Education: 0M
- Subtitles: 0M (subtitles routed under "padding-opensubtitles")

**Read:** Strongly developmental. ~80% is child-directed or child-genre material. Padding is acknowledged as such in the category labels (organisers were transparent about it).

## Dutch (`BabyLM-community/babylm-nld`, sha `1aa063f7…`)

- **Tier:** 100M
- **Byte premium:** 1.051606 (NL needs ~5% more bytes than EN per "equivalent" content)
- **Total tokens (whitespace):** 109,885,564
- **Documents:** 304,611
- **Reference tokenizer:** whitespace
- **Size on disk:** 569 MB

**Tokens per category**
| Category | Tokens | Share |
|---|---:|---:|
| **padding-opensubtitles** | 69,035,414 | **62.8%** |
| educational | 19,146,045 | 17.4% |
| child-wiki | 9,673,449 | 8.8% |
| child-books | 4,576,823 | 4.2% |
| child-directed-speech | 3,304,756 | 3.0% |
| child-news | 3,177,743 | 2.9% |
| subtitles | 971,334 | 0.9% |

**Group rollup**
- Padding: 69.0M (63%) — overwhelmingly OpenSubtitles
- Education: 19.1M (17%) — high-school exam texts (`alleexamens.nl`)
- Books / Wiki / News: 17.4M (16%)
- Transcription: 3.3M (3%) — only the CDS slice
- Subtitles: 1.0M (1%) — the non-padding subtitle slice

**Sources called out in the README:** Ririro children's stories, GlotStoryBooks, AlleExamens (high-school exams 1999–2024).

**Read:** Calling NL "developmental" is generous. The genuine child-genre share (CDS + child-books + child-news + child-wiki) is ~21M tokens — a fifth of the corpus. The remaining ~80% is OpenSubtitles padding plus Dutch high-school exams. Adult subtitle dialogue and exam prose dominate.

## Chinese (`BabyLM-community/babylm-zho`, sha `600a6657…`)

- **Tier:** > 100M (organisers flag this is over-tier)
- **Byte premium:** 0.935966 (ZH packs more content per byte than EN; adjusts the tier *downward*)
- **Total tokens (Qwen3-0.6B):** 137,835,046
- **Documents:** 203,891
- **Reference tokenizer:** **Qwen/Qwen3-0.6B** (NOT whitespace — Chinese needs subword tokenization to count tokens meaningfully)
- **Scripts:** Hani, Hans, Latn (mixed traditional/simplified plus Latin loanwords)
- **Size on disk:** 519 MB

**Tokens per category**
| Category | Tokens | Share |
|---|---:|---:|
| **subtitles** | 91,328,001 | **66.3%** |
| child-books | 15,976,688 | 11.6% |
| educational | 13,465,351 | 9.8% |
| child-directed-speech | 9,636,590 | 7.0% |
| child-available-speech | 7,403,441 | 5.4% |
| child-wiki | 24,975 | <0.1% |

**Group rollup**
- Subtitles: 91.3M (66%) — labelled `subtitles` directly, not `padding-`
- Books / Wiki / News: 16.0M (12%)
- Transcription: 17.0M (12%)
- Education: 13.5M (10%)
- Padding: 0M

**Sources called out in the README:** Quangushi (`quangushi.com`), GlotStoryBooks, GAOKAO 2010–2024 exams.

**Read:** Even more subtitle-dominated than NL — 66% of the corpus. Genuine child-genre share is ~24M tokens (CDS + child-books + child-available-speech + child-wiki). And critically, ZH is over-tier (137M vs the 100M target after byte-premium adjustment). When mixing for the 100M competition budget we need to **subsample ZH** rather than use it whole.

## Cross-language comparability — the most important audit finding

**The three corpora are NOT register-comparable, even though the byte-premium machinery makes them quantitatively comparable.** Concretely:

| | EN | NL | ZH |
|---|---:|---:|---:|
| Genuine child-genre share | ~80% | ~21% | ~17% |
| Subtitle / "padding subtitles" share | 20% | 63% | 66% |
| Educational (exam prose) share | 0% | 17% | 10% |
| Tokenizer used to count | whitespace | whitespace | Qwen3-0.6B |
| Token tier | exactly 100M | over-tier (110M) | over-tier (138M) |

**Implications for our experimental design:**

1. **Curse-of-multilinguality measurements are confounded by register asymmetry.** If trilingual training degrades EN BLiMP relative to monolingual EN, part of the cause may be that EN training tokens are shifted toward subtitle-style NL/ZH text, not "another language." We should try to control for this in a Pareto cell: **train an EN-only run on a subtitle-heavy reweighting of `babylm-eng` (boost the OpenSubtitles padding share to 60% to match NL/ZH register)** and see whether monolingual scores fall toward the trilingual numbers. If yes, register dominates language; if no, it's genuinely cross-lingual interference.

2. **The cognitively-plausible bilingual schedule (C2 in our research direction) needs a developmental-only sub-corpus.** Filter every language to its child-genre categories (drop `padding-*` and `subtitles` and `educational`). This shrinks the budget — ~80M EN, ~21M NL, ~24M ZH developmental — but makes the simultaneous-bilingual story honest. The Pareto sweep should include both "full corpora" and "developmental-only" rows.

3. **Tokenizer choice will couple to the corpus mix.** A whitespace-trained joint tokenizer fits EN and NL fine but tokenizes ZH catastrophically. Either use a joint subword tokenizer (Unigram or BPE trained on the joint corpus, with a vocab proportional to byte premium) or follow the organisers' lead and adopt Qwen3-0.6B's tokenizer for ZH and a separate one for EN+NL. The README's GPT-2 baselines almost certainly use a single GPT-2 BPE — which silently penalises ZH — so a tokenizer ablation is a high-leverage, well-grounded contribution.

4. **The "100M token" budget is mixture-defined.** Whitespace-counted EN at 99M is roughly comparable to Qwen-counted ZH at 138M *after byte premium*. In our scaffold, the tokenizer we use during training will produce a different token count than the README — so the rule we enforce is "≤ post-byte-premium adjusted total = 100M reference units", not literal subword tokens.

## What we still need to confirm or pull

- [ ] Click "Access repository" on `babylm-eng/nld/zho` so we can pull the full schema (one-time HF web step).
- [ ] Validate the Beetle-Data mirror is byte-identical to the gated repo (modulo schema). Streaming sample of `BabyBabel-eng-70M` showed `source = BabyLM-community/babylm-eng` per record, suggesting yes.
- [ ] Check eval-set leakage: does any document in `babylm-eng/nld/zho` overlap with BLiMP, ZhoBLiMP, MultiBLiMP, the various `_mubench` sets, ARC, Belebele, INCLUDE, etc.? The README does not state. We should run a dedup-vs-eval pass before the first training run.
- [ ] Confirm the Strict / Strict-Small (CHILDES + BNC + Switchboard + …) corpus is **not** to be combined with the multilingual corpora for the multilingual track. Our reading: the 100M budget refers to the multilingual corpora only, but the rule text on the homepage should be re-checked once we have full access.
- [ ] Inspect whether `dataset_metadata.json` (currently 403-gated) carries a deduplication manifest or a fixed train/val split.

## Saved artifacts

- This file: `/mnt/sagemaker-nvme/babylm/data/data_card.md`
- README copies (public): `/mnt/sagemaker-nvme/babylm/data/audit/{eng,nld,zho}/README.md`
- Beetle-Data EN sample (3 records, schema demo): in the streaming smoke-test log

## Next blocker

Granting "Access repository" approval on the three gated dataset pages is a UI action only the user can take. The token is fine; it's an account-level access list. After approval the `dataset_metadata.json` and full schema flow normally.
