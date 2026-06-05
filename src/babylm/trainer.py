"""Single-process training loop. DDP is layered via accelerate at the entrypoint.

Loop responsibilities:
- pull packed sequences from the data iterator,
- forward / backward / step / log,
- emit a checkpoint at every revision in `cfg.train.save_revisions` (in
  reference-tokens). Checkpoints save as HF LlamaForCausalLM directories under
  `cfg.out_dir/chck_{N}M/` so the eval pipeline picks them up via
  `revision=chck_{N}M` (when uploaded to HF Hub) or via `pretrained=<local>`
  for offline eval.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast

from .config import RunConfig, BYTE_PREMIUM
from .data import (
    LanguageStream,  # imported for side-effects in tests; not used here directly
    MixtureBudget,
    doc_iterator,
    make_mixture_budget,
    packed_token_iterator,
)
from .model import build_model, num_params


def _cosine_lr(step: int, cfg: RunConfig, total_steps: int) -> float:
    """LR schedule with linear warmup. Schedule = {cosine, linear, constant, wsd}."""
    warmup = cfg.optim.warmup_steps
    base = cfg.optim.lr
    floor = base * cfg.optim.min_lr_ratio
    if step < warmup:
        return base * step / max(1, warmup)
    if cfg.optim.lr_schedule == "constant":
        return base
    if cfg.optim.lr_schedule == "linear":
        frac = (step - warmup) / max(1, total_steps - warmup)
        return base + (floor - base) * min(1.0, frac)
    if cfg.optim.lr_schedule == "wsd":
        # Warmup-Stable-Decay: hold at peak, then linear decay over the last
        # `wsd_decay_frac` of total steps. Loses less of the budget to a too-
        # aggressive cosine when the loss is still falling at end-of-schedule.
        decay_steps = int(total_steps * cfg.optim.wsd_decay_frac)
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return base
        d_frac = (step - decay_start) / max(1, decay_steps)
        d_frac = max(0.0, min(1.0, d_frac))
        return base + (floor - base) * d_frac
    # cosine
    frac = (step - warmup) / max(1, total_steps - warmup)
    frac = max(0.0, min(1.0, frac))
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * frac))


def _wrap_tokenizer_for_hf(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    """Wrap a tokenizers.Tokenizer in a transformers PreTrainedTokenizerFast so
    checkpoints save to disk in a way HF AutoModel can load."""
    tk_path = Path(tokenizer_dir) / "tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tk_path),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _save_checkpoint(model: LlamaForCausalLM, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def estimate_total_steps(cfg: RunConfig, world_size: int) -> int:
    """Rough estimate: total effective token budget / (per-step tokens)."""
    per_step = cfg.train.micro_batch_size * cfg.train.grad_accum_steps * cfg.train.seq_len * world_size
    total_budget = cfg.data.budget_reference_tokens * cfg.data.max_epochs
    return max(1, total_budget // per_step)


def train(
    cfg: RunConfig,
    available_reference_tokens: dict[str, int],
    max_steps: int | None = None,
) -> dict:
    """Run training. Returns a small summary dict.

    `available_reference_tokens` is per-language total budget (used for epoch
    accounting). `max_steps`, if set, caps the loop early — used in smoke tests.
    """
    accelerator = Accelerator(mixed_precision="bf16" if cfg.train.bf16 else "no")
    if accelerator.is_main_process:
        print(f"[trainer] world_size={accelerator.num_processes} device={accelerator.device}")

    tokenizer = _wrap_tokenizer_for_hf(cfg.tokenizer.tokenizer_dir)
    pad_id = tokenizer.pad_token_id

    model = build_model(cfg.model, vocab_size=tokenizer.vocab_size, pad_token_id=pad_id,
                        gradient_checkpointing=cfg.train.gradient_checkpointing)
    if accelerator.is_main_process:
        print(f"[trainer] model params: {num_params(model):,}")

    optim = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    model, optim = accelerator.prepare(model, optim)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        cfg.save(out / "run.yaml")
        with open(out / "available_reference_tokens.json", "w") as f:
            json.dump(available_reference_tokens, f)

    budget = make_mixture_budget(cfg.data, available_reference_tokens)
    seqs = packed_token_iterator(
        cfg.data, budget, tokenizer.backend_tokenizer, cfg.train.seq_len
    )

    # Estimate steps for the LR schedule.
    total_steps = estimate_total_steps(cfg, accelerator.num_processes)

    # Iterate microbatches.
    step = 0
    accum = 0
    micro_batch: list[torch.Tensor] = []
    next_revision_idx = 0
    revisions = sorted(set(cfg.train.save_revisions))
    t0 = time.time()
    last_log = t0
    running_loss = 0.0
    running_steps = 0

    for seq in seqs:
        micro_batch.append(seq)
        if len(micro_batch) < cfg.train.micro_batch_size:
            continue
        ids = torch.stack(micro_batch).to(accelerator.device)
        micro_batch = []

        # Standard causal LM: shift inside model
        out_obj = model(input_ids=ids, labels=ids)
        loss = out_obj.loss / cfg.train.grad_accum_steps
        accelerator.backward(loss)
        accum += 1

        if accum >= cfg.train.grad_accum_steps:
            accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            lr = _cosine_lr(step, cfg, total_steps)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            running_loss += loss.item() * cfg.train.grad_accum_steps
            running_steps += 1

            if accelerator.is_main_process and step % cfg.train.log_every == 0:
                now = time.time()
                tps = (
                    cfg.train.micro_batch_size
                    * cfg.train.grad_accum_steps
                    * cfg.train.seq_len
                    * accelerator.num_processes
                    * cfg.train.log_every
                ) / max(1e-6, now - last_log)
                print(
                    f"[step {step:>6}/{total_steps}] "
                    f"loss={running_loss / max(1, running_steps):.4f}  "
                    f"lr={lr:.2e}  "
                    f"ref_tok={budget.total_consumed:,}  "
                    f"frac={budget.frac_done:.3f}  "
                    f"tok/s={tps:,.0f}"
                )
                running_loss = 0.0
                running_steps = 0
                last_log = now

            # Save revisions when total reference tokens crosses a threshold.
            while (
                next_revision_idx < len(revisions)
                and budget.total_consumed >= revisions[next_revision_idx]
            ):
                rev_tokens = revisions[next_revision_idx]
                if accelerator.is_main_process:
                    # Match the eval pipeline's revision schema: chck_{N}M where N is
                    # the milestone in millions. ceil so smoke runs at 10k get a
                    # distinct name from real 1M revisions.
                    rev_m = max(1, math.ceil(rev_tokens / 1_000_000))
                    rev_name = f"chck_{rev_m}M"
                    print(f"[checkpoint] saving {rev_name} at {budget.total_consumed:,} ref tokens")
                    unwrapped = accelerator.unwrap_model(model)
                    _save_checkpoint(unwrapped, tokenizer, out / rev_name)
                next_revision_idx += 1

            if max_steps is not None and step >= max_steps:
                break
            if not budget.can_continue():
                break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        _save_checkpoint(unwrapped, tokenizer, out / "final")
        summary = {
            "steps": step,
            "ref_tokens": budget.total_consumed,
            "consumed_per_lang": dict(budget.consumed_reference_tokens),
            "epochs_per_lang": {
                l: budget.epochs_seen(l) for l in budget.available_reference_tokens
            },
            "wallclock_seconds": time.time() - t0,
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    return {}
