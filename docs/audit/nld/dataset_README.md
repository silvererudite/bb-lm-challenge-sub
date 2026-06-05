---
task_categories:
- text-generation
language:
- nld
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

- **Language:** nld
- **Script:** Latn
- **Tier:** 100M
- **Byte Premium Factor:** 1.051606
- **Size (MB):** 569.49
- **Expected Size (MB):** 571.02
- **Number of Documents:** 304,611
- **Total Tokens:** 109,885,564
- **Tokenizer:** separate by whitespace

### Tokens Per Category

- **child-books:** 4,576,823 tokens
- **child-directed-speech:** 3,304,756 tokens
- **child-news:** 3,177,743 tokens
- **child-wiki:** 9,673,449 tokens
- **educational:** 19,146,045 tokens
- **padding-opensubtitles:** 69,035,414 tokens
- **subtitles:** 971,334 tokens

### Tokens Per Group

- **Transcription:** 3,304,756 tokens
- **Education:** 19,146,045 tokens
- **Books, Wiki, News:** 17,428,015 tokens
- **Subtitles:** 971,334 tokens
- **Padding:** 69,035,414 tokens


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
- Ririro Story Collection
	 - source: https://ririro.com/
	 - description: Collection of children storybooks
- GlotStoryBooks Collection
	 - source: https://arxiv.org/abs/2310.16248
	 - description: Collection of children storybooks

#### Educational (school exams)
- AlleExamens
	 - source: https://www.alleexamens.nl/
	 - description: Texts of all high school exams from 1999 to 2024

#### Educational materials
- WikiWijs
	 - source: https://www.wikiwijs.nl/
	 - description: Educational materials for both primary and high school level
- KlasCement
	 - source: https://www.klascement.net/
	 - description: Flemish educational materials
- BasiLex
	 - source: https://www.clinjournal.org/clinj/article/view/50
	 - description: Collection of children’s media, children’s books and educational materials

#### Transcription
- CHILDES
	 - source: https://talkbank.org/childes/
	 - description: Transcription of child directed language conversations

### Data Curators

* Jaap Jumelet



### Comments 

None