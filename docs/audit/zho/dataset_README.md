---
task_categories:
- text-generation
language:
- zho
license: unknown
size_categories:
- 100K<n<1M
dataset_info:
  features:
    - name: text
      dtype: string
    - name: doc-id
      dtype: string
    - name: category
      dtype: string
    - name: data-source
      dtype: string
    - name: script
      dtype: string
    - name: age-estimate
      dtype: string
    - name: license
      dtype: string
    - name: misc
      dtype: string
    - name: num-tokens
      dtype: int64
    - name: language
      dtype: string
---

# BabyLM Dataset

## Dataset Description

This dataset is part of the BabyLM multilingual collection.   
More information at: [babylm.github.io/babybabellm](https://babylm.github.io/babybabellm/)

### Dataset Summary

- **Language:** zho
- **Script:** Hani, Hans, Latn
- **Tier:** > 100M
- **Byte Premium Factor:** 0.935966
- **Size (MB):** 518.85
- **Expected Size (MB):** 508.23
- **Number of Documents:** 203,891
- **Total Tokens:** 137,835,046
- **Tokenizer:** Qwen/Qwen3-0.6B

### Tokens Per Category

- **child-available-speech:** 7,403,441 tokens
- **child-books:** 15,976,688 tokens
- **child-directed-speech:** 9,636,590 tokens
- **child-wiki:** 24,975 tokens
- **educational:** 13,465,351 tokens
- **subtitles:** 91,328,001 tokens

### Tokens Per Group

- **Transcription:** 17,040,031 tokens
- **Education:** 13,465,351 tokens
- **Books, Wiki, News:** 16,001,663 tokens
- **Subtitles:** 91,328,001 tokens
- **Padding:** 0 tokens


### Data Fields

- `text`: The document text
- `doc-id`: Unique identifier for the document
- `category`: Type of content (e.g., child-directed-speech, educational, etc.)
- `data-source`: Original source of the data
- `script`: Writing system used (ISO 15924)
- `age-estimate`: Target age or age range
- `license`: Data license
- `misc`: Additional metadata (JSON string)
- `num-tokens`: Number of tokens per item (based on white-space split)
- `language`: Language code (ISO 639-3)

### Licensing Information

Please see license in individual documents

### Data Sources & Attribution

#### Books
- Quangushi (full stories)
	 - source: http://quangushi.com/
	 - description: Collection of children’s stories
- GlotStoryBooks Collection
	 - source: https://arxiv.org/abs/2310.16248
	 - description: Collection of children storybooks

#### Educational material (exam style)
- GAOKAO
	 - source: https://arxiv.org/abs/2305.12474
	 - description: Dataset including subjective and objective questions from exams from 2010 to 2024
- CK-12
	 - source: https://dl.acm.org/doi/10.1609/aaai.v38i17.29914
	 - description: Dataset covering most comprehensive knowledge points in Chinese K12 field
- CSQ
	 - source: https://figshare.com/articles/dataset/28667489
	 - description: Chinese Science Question dataset
- FCGEC
	 - source: https://aclanthology.org/2022.findings-emnlp.137/
	 - description: A human-annotated corpus based on multi-choice grammatical error problems
- Primary school students’ compositions
	 - source: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309717
	 - description: A hierarchical corpus of primary school students’ compositions

#### Reading comprehension
- CFT reading comprehension dataset
	 - source: https://aclanthology.org/C16-1167/
	 - description: Collection of children’s stories from a reading comprehension dataset
- CMRC-2019 reading comprehension dataset
	 - source: https://aclanthology.org/2020.coling-main.589/
	 - description: Collection of children’s stories from a reading comprehension dataset

#### Transcription
- ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5
	 - source: https://aclanthology.org/2025.acl-long.614/
	 - description: Transcriptions of children's conversations
- NaturalConv
	 - source: http://ojs.aaai.org/index.php/AAAI/article/view/17649
	 - description: A multi-turn topic-driven adult conversation dataset
- CHILDES
	 - source: https://talkbank.org/childes/
	 - description: Transcription of child directed language conversations

#### Wikis
- WikiJunior
	 - source: https://en.wikibooks.org/wiki/Wikijunior:Languages/Mandarin_Chinese
	 - description: Age-appropriate non-fiction books for children
- Wikibooks
	 - source: https://en.wikibooks.org/wiki/Chinese_(Mandarin)
	 - description: Age-appropriate non-fiction books for children

### Data Curators

* Hai Hu
* Linyang He
* Siyuan Song
* Ziyin Zhang



### Comments 

None