---
task_categories:
- text-generation
language:
- eng
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

- **Language:** eng
- **Script:** Latn
- **Tier:** 100M
- **Byte Premium Factor:** 1.000000
- **Size (MB):** 539.18
- **Expected Size (MB):** 543.00
- **Number of Documents:** 137,710
- **Total Tokens:** 98,878,321
- **Tokenizer:** separate by whitespace

### Tokens Per Category

- **child-available-speech:** 9,102,166 tokens
- **child-books:** 26,748,028 tokens
- **child-directed-speech:** 27,712,538 tokens
- **child-wiki:** 14,609,286 tokens
- **padding-opensubtitles:** 19,699,367 tokens
- **padding-wikipedia:** 1,006,936 tokens

### Tokens Per Group

- **Transcription:** 36,814,704 tokens
- **Education:** 0 tokens
- **Books, Wiki, News:** 41,357,314 tokens
- **Subtitles:** 0 tokens
- **Padding:** 20,706,303 tokens


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

n/a

### Data Curators

n/a


### Comments 

None