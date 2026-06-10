"""Streaming, byte-premium-aware multilingual data mixer.

Contract: yields `seq_len`-token sequences forever, drawing per-example from a
language i.i.d. according to the current mixture weights, while accounting for
per-language byte premiums and enforcing the BabyLM ≤10-epoch budget.

The same iterator drives every cell of the ablation grid; the only thing that
changes between cells is the YAML.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset

from .config import BYTE_PREMIUM, LANGS, DataConfig


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError(f"mixture weights must be positive, got {weights}")
    return {k: v / s for k, v in weights.items() if v > 0}


def schedule_weights(
    cfg: DataConfig, frac_done: float
) -> dict[str, float]:
    """Return the mixture weights at this point in training.

    `frac_done` ∈ [0, 1] is the fraction of the reference-token budget used.
    If `cfg.schedule` is empty, returns `cfg.mixture` (a constant). Otherwise
    piecewise-constant interpolation between schedule points.
    """
    if not cfg.schedule:
        return _normalize(cfg.mixture)
    pts = sorted(cfg.schedule)
    if frac_done <= pts[0][0]:
        return _normalize(pts[0][1])
    for (f0, w0), (f1, w1) in zip(pts, pts[1:]):
        if f0 <= frac_done < f1:
            return _normalize(w0)
    return _normalize(pts[-1][1])


@dataclass
class MixtureBudget:
    """Tracks reference-token consumption per language and total epochs."""

    budget_reference_tokens: int
    max_epochs: int
    available_reference_tokens: dict[str, int]  # static per-corpus totals (post-BP)
    consumed_reference_tokens: dict[str, int]

    @property
    def total_consumed(self) -> int:
        return sum(self.consumed_reference_tokens.values())

    @property
    def total_token_budget(self) -> int:
        """Effective token budget = corpus size × max_epochs (BabyLM rule)."""
        return self.budget_reference_tokens * self.max_epochs

    @property
    def frac_done(self) -> float:
        return self.total_consumed / max(1, self.total_token_budget)

    def epochs_seen(self, lang: str) -> float:
        return self.consumed_reference_tokens[lang] / max(
            1, self.available_reference_tokens[lang]
        )

    def can_continue(self) -> bool:
        # Total compute cap: ≤ 10 epochs over the corpus (BabyLM rule).
        if self.total_consumed >= self.total_token_budget:
            return False
        # Per-language epoch cap: also bound at max_epochs of that lang.
        for lang in self.available_reference_tokens:
            if self.epochs_seen(lang) > self.max_epochs:
                return False
        return True

    def add(self, lang: str, ref_tokens: int) -> None:
        self.consumed_reference_tokens[lang] += ref_tokens


class LanguageStream:
    """One language's document stream, re-shuffled at each epoch boundary.

    For streaming=False (the default for the trilingual baseline), the dataset
    is loaded ONCE and the category-filter is applied ONCE; only the per-epoch
    shuffle re-runs. This avoids re-filtering ~304k NL records every time the
    iterator exhausts, which dropped v3's smoke throughput from 98k to 64k tok/s.
    For streaming=True we have no choice but to re-open and re-filter.
    """

    def __init__(
        self,
        lang: str,
        repo: str,
        category_whitelist: list[str] | None,
        token_field: str,
        streaming: bool,
        shuffle_buffer: int,
        seed: int,
    ) -> None:
        self.lang = lang
        self.repo = repo
        self.category_whitelist = category_whitelist
        self.token_field = token_field
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0
        self._cached_ds = None  # in-memory dataset, post-filter; reused across epochs
        self._iter = self._open()

    def _open(self) -> Iterator[dict]:
        # Hub corpora are NOT shuffled at-rest; see docs/data_card.md. In-memory
        # shuffle (streaming=False) is required for unbiased sampling on the
        # 100M-scale trilingual run. Streaming mode is only honest if shuffle
        # buffer >> corpus, which is impractical at this scale.
        if self.streaming:
            ds = load_dataset(self.repo, split="train", streaming=True,
                              token=os.environ.get("HF_TOKEN"))
            if self.category_whitelist is not None:
                wl = set(self.category_whitelist)
                ds = ds.filter(lambda r: r.get("category") in wl)
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self._epoch)
            return iter(ds)

        # Non-streaming: load + filter ONCE; re-shuffle on each epoch.
        if self._cached_ds is None:
            ds = load_dataset(self.repo, split="train",
                              token=os.environ.get("HF_TOKEN"))
            if self.category_whitelist is not None:
                wl = set(self.category_whitelist)
                ds = ds.filter(lambda r: r.get("category") in wl)
            self._cached_ds = ds
        return iter(self._cached_ds.shuffle(seed=self.seed + self._epoch))

    def next_doc(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            self._iter = self._open()
            return next(self._iter)


def make_mixture_budget(
    cfg: DataConfig, available_reference_tokens: dict[str, int]
) -> MixtureBudget:
    return MixtureBudget(
        budget_reference_tokens=cfg.budget_reference_tokens,
        max_epochs=cfg.max_epochs,
        available_reference_tokens=dict(available_reference_tokens),
        consumed_reference_tokens={l: 0 for l in available_reference_tokens},
    )


def doc_iterator(
    cfg: DataConfig,
    budget: MixtureBudget,
) -> Iterator[tuple[str, dict]]:
    """Yield (lang, document) pairs sampled per-mixture, while updating budget.

    Stops when the budget is spent or any language exceeds max_epochs.
    """
    rng = random.Random(cfg.seed)
    streams = {
        lang: LanguageStream(
            lang=lang,
            repo=cfg.repos[lang],
            category_whitelist=cfg.category_whitelist.get(lang),
            token_field=cfg.token_field,
            streaming=cfg.streaming,
            shuffle_buffer=cfg.shuffle_buffer,
            seed=cfg.seed,
        )
        for lang in cfg.repos
    }
    while budget.can_continue():
        # Mixture weights are TOKEN shares, not document shares. Documents in
        # different languages have different mean ref-token sizes (EN ~720,
        # NL ~360, ZH ~680 per doc), so uniform doc-sampling produces a biased
        # token-share. Each step we pick the language with the largest
        # token-share deficit vs the target weights — this drives consumption
        # toward the configured mixture in expectation, regardless of doc-size
        # heterogeneity. Tie-broken stochastically to avoid lockstep.
        weights = schedule_weights(cfg, budget.frac_done)
        keys = list(weights.keys())
        # Current shares of consumed tokens (across the active subset).
        sub_total = max(1, sum(budget.consumed_reference_tokens[k] for k in keys))
        deficits = []
        for k in keys:
            target = weights[k]
            current = budget.consumed_reference_tokens[k] / sub_total
            deficits.append(target - current)
        # Pick the largest deficit; if all roughly equal, fall back to weighted RNG
        # so early-step ties resolve smoothly.
        d_max = max(deficits)
        contenders = [k for k, d in zip(keys, deficits) if d > d_max - 1e-6]
        if len(contenders) == 1:
            lang = contenders[0]
        else:
            sub_w = [weights[k] for k in contenders]
            lang = rng.choices(contenders, weights=sub_w, k=1)[0]
        doc = streams[lang].next_doc()
        # Reference tokens consumed by this document. The dataset's token field
        # is already in the per-language reference unit (whitespace for EN/NL,
        # Qwen3 for ZH). We byte-premium-adjust so that "1 reference token" is
        # comparable across languages.
        raw_tokens = int(doc.get(cfg.token_field) or 0)
        ref_tokens = int(raw_tokens / BYTE_PREMIUM[lang])
        budget.add(lang, ref_tokens)
        yield lang, doc


def packed_token_iterator(
    cfg: DataConfig,
    budget: MixtureBudget,
    tokenizer,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """Pack tokenized text into fixed-length sequences for autoregressive training.

    Documents are concatenated with `<eos>` between them and chunked into
    `seq_len`-length tensors. The packer never crosses budget boundaries — once
    `budget.can_continue()` flips to False, the inner loop drains the buffer
    and stops.
    """
    eos_id = tokenizer.token_to_id("<eos>") if hasattr(tokenizer, "token_to_id") else (
        getattr(tokenizer, "eos_token_id", None)
    )
    if eos_id is None:
        raise RuntimeError("tokenizer must define <eos>")
    buf: list[int] = []
    for _, doc in doc_iterator(cfg, budget):
        text = doc.get("text") or ""
        if not text:
            continue
        if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "encode_batch"):
            # transformers fast tokenizer
            ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            # tokenizers Tokenizer
            ids = tokenizer.encode(text).ids
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)
