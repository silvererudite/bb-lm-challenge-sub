"""Joint subword tokenizer training for the EN+NL+ZH mixture.

Per the data card, a single whitespace tokenizer would be catastrophic for
Chinese; a single GPT-2 BPE silently penalises ZH. The default scaffold uses
a joint BPE trained on a balanced sample from all three languages. Tokenizer
choice itself is an ablation; this module produces a single tokenizer dir
that downstream training/eval load by path.
"""
from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

from .config import LANGS, BYTE_PREMIUM, RunConfig


def _sample_text(
    repo: str,
    target_chars: int,
    seed: int,
    token: str | None = None,
) -> list[str]:
    """Pull `~target_chars` characters of text from a streaming dataset."""
    ds = load_dataset(repo, split="train", streaming=True, token=token)
    ds = ds.shuffle(buffer_size=10_000, seed=seed)
    out: list[str] = []
    total = 0
    for r in ds:
        t = r.get("text") or ""
        if not t:
            continue
        out.append(t)
        total += len(t)
        if total >= target_chars:
            break
    return out


def train_joint_tokenizer(cfg: RunConfig, token: str | None = None) -> Path:
    """Train a joint subword tokenizer and save it under cfg.tokenizer.tokenizer_dir.

    Returns the absolute path of the saved tokenizer.json.
    """
    out_dir = Path(cfg.tokenizer.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample text per language. Each language contributes train_chars_per_lang
    # so the tokenizer is jointly trained on a balanced (in chars) corpus.
    sources: list[str] = []
    for lang in LANGS:
        repo = cfg.data.repos[lang]
        sources.extend(_sample_text(repo, cfg.tokenizer.train_chars_per_lang, cfg.data.seed, token=token))

    if cfg.tokenizer.algorithm == "bpe":
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.tokenizer.vocab_size,
            special_tokens=cfg.tokenizer.specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
    elif cfg.tokenizer.algorithm == "unigram":
        tk = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=cfg.tokenizer.vocab_size,
            special_tokens=cfg.tokenizer.specials,
            unk_token="<unk>",
        )
        tk.normalizer = normalizers.NFKC()
        tk.pre_tokenizer = pre_tokenizers.Metaspace()
        tk.decoder = decoders.Metaspace()
    else:
        raise ValueError(f"unknown tokenizer algorithm {cfg.tokenizer.algorithm}")

    tk.train_from_iterator(sources, trainer=trainer)

    # Sanity: every special must be in vocab.
    for sp in cfg.tokenizer.specials:
        if tk.token_to_id(sp) is None:
            raise RuntimeError(f"special token {sp!r} did not survive training")

    tk_path = out_dir / "tokenizer.json"
    tk.save(str(tk_path))
    return tk_path


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    p = Path(tokenizer_dir) / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(f"no tokenizer at {p}")
    return Tokenizer.from_file(str(p))
