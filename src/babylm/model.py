"""Llama-style decoder, sized to ~125M for parity with the GPT-2 BabyLM baselines.

Why Llama-style over GPT-2: RoPE removes the absolute-position-embedding
budget, RMSNorm is fewer params and slightly more stable at small scale,
SwiGLU + tied embeddings are standard at this size in 2026, and writing a
clean reference is cheaper than fighting `transformers.GPT2Config`.

Persisted as a HuggingFace `LlamaForCausalLM` so the eval pipeline (which
uses lm-eval-harness via AutoModelForCausalLM) loads checkpoints directly.
"""
from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM

from .config import ModelConfig


def build_model(model_cfg: ModelConfig, vocab_size: int, pad_token_id: int = 0,
                gradient_checkpointing: bool = True) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg.d_model,
        intermediate_size=model_cfg.d_ff,
        num_hidden_layers=model_cfg.n_layers,
        num_attention_heads=model_cfg.n_heads,
        num_key_value_heads=model_cfg.n_kv_heads,
        max_position_embeddings=model_cfg.max_seq_len,
        rms_norm_eps=model_cfg.rms_norm_eps,
        rope_theta=model_cfg.rope_theta,
        tie_word_embeddings=model_cfg.tie_embeddings,
        initializer_range=model_cfg.init_std,
        pad_token_id=pad_token_id,
        use_cache=False,
        attention_bias=False,
    )
    m = LlamaForCausalLM(cfg)
    if gradient_checkpointing:
        m.gradient_checkpointing_enable()
    return m


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
