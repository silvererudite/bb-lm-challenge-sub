"""Run configuration — one YAML per ablation cell.

The same scaffold is supposed to run every cell of the grid. The contract is:
the YAML fully specifies the run. No CLI flags toggle science-relevant
behavior (only operational concerns like resume / dry-run / output dir).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml

# Byte premiums from the official BabyBabelLM dataset cards.
# EN is the reference (1.000); NL needs ~5% more bytes per equivalent content;
# ZH needs ~6% fewer (denser script).
BYTE_PREMIUM = {"eng": 1.000000, "nld": 1.051606, "zho": 0.935966}

LANGS = ("eng", "nld", "zho")


@dataclass
class DataConfig:
    """Per-language data sourcing + mixture schedule."""

    # HF dataset id per language. Default is the official BabyBabelLM track repos.
    repos: dict[str, str] = field(
        default_factory=lambda: {
            "eng": "BabyLM-community/babylm-eng",
            "nld": "BabyLM-community/babylm-nld",
            "zho": "BabyLM-community/babylm-zho",
        }
    )
    # Optional category whitelist per language. None = use everything.
    # E.g. {"eng": ["child-directed-speech", "child-books", "child-wiki", "child-available-speech"]}
    # Drops the OpenSubtitles / Wikipedia padding to compare developmental-only mixtures.
    category_whitelist: dict[str, list[str] | None] = field(default_factory=dict)
    # Static mixture weights. Sum is renormalized; semantics: probability of
    # drawing the next document from each language. For curricula, use schedule.
    mixture: dict[str, float] = field(
        default_factory=lambda: {"eng": 1 / 3, "nld": 1 / 3, "zho": 1 / 3}
    )
    # Curriculum / cognitive schedule: list of (frac_through_training, weights).
    # Empty = use `mixture` throughout.
    # Example simultaneous-bilingual: [(0.0, {"eng":0.6,"nld":0.4,"zho":0.0}), (0.5,{"eng":0.5,"nld":0.3,"zho":0.2}),...]
    schedule: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    # Budget in REFERENCE TOKENS (byte-premium adjusted). 100M is the BabyLM cap.
    budget_reference_tokens: int = 100_000_000
    # ≤10 epochs cap. The data iterator will refuse more.
    max_epochs: int = 10
    # Per-document token field name in the dataset (whitespace count for EN/NL,
    # Qwen3-0.6B count for ZH). We treat this as the "reference token" count.
    token_field: str = "num-tokens"
    # Streaming or in-memory. Streaming is required at 100M scale to avoid
    # downloading the whole corpus eagerly; in-memory only for tests.
    streaming: bool = True
    # Shuffle buffer size (records) when streaming.
    shuffle_buffer: int = 10_000
    # Random seed for reproducible mixture sampling and shuffling.
    seed: int = 12


@dataclass
class TokenizerConfig:
    """Tokenizer training/loading.

    Contract: trained once, saved to disk, loaded by every downstream cell.
    Re-training a tokenizer is itself an ablation.
    """

    # Path under models/ where the trained tokenizer lives. Must exist for
    # training runs; trained on demand by `train_tokenizer.py`.
    tokenizer_dir: str = "models/tokenizer-joint-bpe-32k"
    vocab_size: int = 32_768
    algorithm: Literal["bpe", "unigram"] = "bpe"
    # Per-language sample sizes (chars) used to TRAIN the tokenizer. Should be
    # large enough to cover script coverage but not the full corpus.
    train_chars_per_lang: int = 50_000_000
    # Special tokens that always appear in vocab.
    specials: list[str] = field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )


@dataclass
class ModelConfig:
    """Llama-style decoder. Sized to ~125M for fair comparison vs GPT-2 baselines."""

    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12  # equal -> standard MHA; <n_heads -> GQA
    d_ff: int = 2048  # SwiGLU intermediate (~2.66x d_model with 2/3 split)
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 1024
    tie_embeddings: bool = True
    init_std: float = 0.02


@dataclass
class OptimConfig:
    optimizer: Literal["adamw"] = "adamw"
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_schedule: Literal["cosine", "linear", "constant", "wsd"] = "cosine"
    min_lr_ratio: float = 0.1
    # WSD-only: fraction of total steps spent decaying at the end. The rest
    # (after warmup) holds at the peak LR. Default 0.25 = 25% decay tail.
    wsd_decay_frac: float = 0.25


@dataclass
class TrainConfig:
    micro_batch_size: int = 32  # per-device
    grad_accum_steps: int = 1
    seq_len: int = 1024
    bf16: bool = True
    compile: bool = False  # torch.compile — disable until the loop is stable
    gradient_checkpointing: bool = False  # ~2x memory at ~1.4x compute; needed only for batch>=32 on A10G
    log_every: int = 25
    save_revisions: list[int] = field(
        default_factory=lambda: (
            [i * 1_000_000 for i in range(1, 10)]
            + [i * 10_000_000 for i in range(1, 10)]
            + [i * 100_000_000 for i in range(1, 11)]
        )
    )  # word-count milestones, matches `chck_{N}M` revision names from collate_results.py
    # WandB. Disabled iff project is None. The same project hosts every run;
    # set `wandb_run_name` per-cell so the dashboard's grouping is meaningful.
    wandb_project: str | None = "babylm-2026-multilingual"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None  # default = RunConfig.name
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


@dataclass
class RunConfig:
    name: str = "smoke"
    out_dir: str = "checkpoints/smoke"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        sub = {}
        for key, sub_cls in [
            ("data", DataConfig),
            ("tokenizer", TokenizerConfig),
            ("model", ModelConfig),
            ("optim", OptimConfig),
            ("train", TrainConfig),
        ]:
            if key in d:
                sub[key] = sub_cls(**d.pop(key))
        return cls(**d, **sub)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
