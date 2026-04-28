"""LoRA fine-tuning for any HuggingFace-compatible edge model.

PLE-awareness:
  Gemma 4's Per-Layer Embeddings are pre-computed for each vocabulary token
  and are not typically part of the set of modules fine-tuned by LoRA.
  Fine-tuning the PLE table directly is (a) expensive — it's the largest
  component — and (b) currently under-explored. This module targets only
  the decoder projection matrices (q/k/v/o, gate/up/down), leaving PLE
  frozen. If downstream quality needs PLE adaptation, that's a separate
  ADR.

Quantization:
  Base model is loaded in 4-bit (NF4) via bitsandbytes, then LoRA adapters
  are applied in bf16. This is the QLoRA recipe — fits E2B comfortably on
  a single 16 GB consumer GPU during training.

Shared KV Cache:
  Layers in the shared-KV tail do not have independent K/V projection
  matrices to adapt. The `target_modules` list automatically skips these
  because they don't exist in the model's named_modules() output.

This script assumes:
  * transformers >= 4.57.0 (for Gemma 4 support)
  * peft >= 0.14.0
  * bitsandbytes >= 0.45.0
  * trl >= 0.13.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA hyperparameters. Defaults taken from Google's recommended
    Gemma 4 fine-tuning recipe."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] = Field(default_factory=list)  # e.g. ["lm_head"] if needed


class TrainingConfig(BaseModel):
    """Training hyperparameters. Tuned for a single L4/A10G-class GPU."""

    output_dir: str = "./outputs/edge-model-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    bf16: bool = True
    fp16: bool = False
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "tensorboard"  # switch to "wandb" or "none" as needed
    seed: int = 42


class QuantConfig(BaseModel):
    """Base-model quantization config for QLoRA."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class FinetuneJobConfig(BaseModel):
    """Full job configuration.

    Load from YAML:
        cfg = FinetuneJobConfig.from_yaml("configs/my_model_lora.yaml")
    """

    base_model: str = ""              # required: HF model ID or local path
    dataset_name: str = ""              # HF dataset id or local path
    dataset_split_train: str = "train"
    dataset_split_eval: str = "validation"
    chat_template: str = "chatml"        # "gemma4", "chatml", "custom", etc.
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    quantization: QuantConfig = Field(default_factory=QuantConfig)
    hf_token: str | None = None          # defaults to env HF_TOKEN

    # Default: strip the ~300 MB audio encoder since ASR is a
    # standalone component, not in-model. Override to False only if you
    # specifically need to evaluate audio capability.
    strip_audio_encoder: bool = False   # True only if model has an audio encoder

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneJobConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            base_model=raw.get("base_model", ""),  # required field in YAML
            dataset_name=raw.get("dataset_name", ""),
            dataset_split_train=raw.get("dataset_split_train", "train"),
            dataset_split_eval=raw.get("dataset_split_eval", "validation"),
            chat_template=raw.get("chat_template", "chatml"),
            lora=LoRAConfig(**raw.get("lora", {})),
            training=TrainingConfig(**raw.get("training", {})),
            quantization=QuantConfig(**raw.get("quantization", {})),
            hf_token=raw.get("hf_token") or os.environ.get("HF_TOKEN"),
            strip_audio_encoder=raw.get("strip_audio_encoder", True),
        )


# ---------------------------------------------------------------------------
# Model & trainer construction
# ---------------------------------------------------------------------------
# These functions import heavy deps (torch / transformers / peft / trl) lazily
# so that `ple_analysis.py` and `turboquant.py` can be used without GPU tooling.


def strip_audio_encoder(model) -> tuple[object, dict]:
    """Remove the Gemma 4 audio encoder module from a loaded model in-place.

    The audio encoder is stripped — ASR is handled by a standalone pipeline,
    so the ~300 MB audio encoder is pure deadweight. Stripping it:
      * reduces the on-disk checkpoint by ~300 MB (save the stripped model)
      * reduces resident RAM by ~300 MB (counts against the 2 GB / 4 GB budget)
      * eliminates audio-token handling from the forward pass

    Per the HF Gemma 4 docs, PLE uses the pad-token ID for multimodal positions,
    so removing the audio encoder does NOT leave orphaned per-layer state.
    The tokenizer and chat template still work unchanged — we just lose the
    ability to pass audio inputs, which is by design.

    Implementation notes:
      * Gemma 4 multimodal checkpoints expose the audio stack under one of
        several attribute names depending on transformers version:
        `audio_tower`, `audio_encoder`, or under `model.audio_tower`. We
        check all common locations and strip whichever is present.
      * The audio projector/adapter (if present) is also removed; it's useless
        without the encoder.
      * Returns a diagnostic dict so callers can log what actually happened.

    Args:
        model: a loaded Gemma 4 model (text-only CausalLM or Multimodal).
    Returns:
        (model, diagnostic_dict) — model is the SAME object, modified in place.
        Diagnostic contains: 'stripped' (list of attr paths), 'params_removed',
        'bytes_removed' (approximate, at model's dtype), 'skipped' (reason
        if nothing was stripped).
    """
    import torch

    stripped_paths = []
    params_removed = 0

    # Candidate locations for the audio stack across transformers versions
    # and between text-only and multimodal checkpoints.
    candidates = [
        # (parent_path, attr_name)
        ("", "audio_tower"),
        ("", "audio_encoder"),
        ("model", "audio_tower"),
        ("model", "audio_encoder"),
        ("", "audio_projector"),       # projector adapter
        ("", "audio_adapter"),
        ("model", "audio_projector"),
        ("model", "audio_adapter"),
    ]

    for parent_path, attr_name in candidates:
        parent = model
        if parent_path:
            parent = getattr(model, parent_path, None)
            if parent is None:
                continue

        submodule = getattr(parent, attr_name, None)
        if submodule is None:
            continue

        # Count parameters before deletion
        for p in submodule.parameters():
            params_removed += p.numel()

        # Replace with a no-op so forward-pass checks don't break.
        # Setting to None can trip `hasattr` checks downstream; an empty
        # Identity module is safer and idempotent.
        setattr(parent, attr_name, torch.nn.Identity())
        stripped_paths.append(f"{parent_path}.{attr_name}" if parent_path else attr_name)

    # Also clear any config flags so the model doesn't try to dispatch
    # audio inputs through the Identity placeholder.
    for cfg_attr in ("has_audio_encoder", "audio_config"):
        if hasattr(model, "config") and hasattr(model.config, cfg_attr):
            setattr(model.config, cfg_attr, None if cfg_attr == "audio_config" else False)

    # Best-effort garbage collection so the freed tensors are released
    import gc

    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Approximate bytes removed — use model's param dtype for estimate
    bytes_per_param = 2  # bf16/fp16 default; NF4 would be 0.5 but post-load we see bf16
    try:
        # Pick any remaining parameter to infer dtype
        any_param = next(iter(model.parameters()), None)
        if any_param is not None:
            bytes_per_param = any_param.element_size()
    except StopIteration:
        pass

    return model, {
        "stripped": stripped_paths,
        "params_removed": params_removed,
        "bytes_removed": params_removed * bytes_per_param,
        "skipped": None if stripped_paths else "No audio encoder found — "
                   "likely a text-only checkpoint or already stripped.",
    }


def build_model_and_tokenizer(cfg: FinetuneJobConfig) -> tuple[Any, Any]:
    """Load the base Gemma 4 model in 4-bit with LoRA adapters attached.

    Returns:
        (model, tokenizer) — model is a PeftModel wrapping a 4-bit base.
    """
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    # For multimodal models, use AutoModelForImageTextToText instead.
    # For text-only fine-tuning, AutoModelForCausalLM is the correct class.
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    compute_dtype = getattr(torch, cfg.quantization.bnb_4bit_compute_dtype)

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map="auto",
        dtype=compute_dtype,
        token=cfg.hf_token,
        trust_remote_code=False,
    )

    # Strip audio encoder BEFORE prepare_model_for_kbit_training.
    # Doing it after kbit-prep would leave the placeholder Identity in a
    # partially-initialized state; stripping first keeps the model graph clean.
    if cfg.strip_audio_encoder:
        model, strip_diag = strip_audio_encoder(model)
        if strip_diag["stripped"]:
            # Log what was stripped so the training run has a paper trail
            print(
                f"[dhurandhar] Stripped audio encoder: "
                f"{str(strip_diag['stripped'])} "
                f"(~{strip_diag['bytes_removed'] / 1024 / 1024:.0f} MB, "
                f"{strip_diag['params_removed']:,} params)"
            )
        else:
            print(f"[dhurandhar] Audio strip skipped: {strip_diag['skipped']}")

    # Prepare the 4-bit model for k-bit training (casts norms to fp32, etc.)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
    )

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        target_modules=cfg.lora.target_modules,
        modules_to_save=cfg.lora.modules_to_save or None,
    )
    model = get_peft_model(model, peft_config)

    # For multimodal training you'd use AutoProcessor here instead.
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_trainer(cfg: FinetuneJobConfig, model, tokenizer, train_ds, eval_ds=None):
    """Construct a TRL SFTTrainer for LoRA fine-tuning."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy if eval_ds is not None else "no",
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        report_to=cfg.training.report_to,
        seed=cfg.training.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Sanity: ensure LoRA parameters actually attached
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError(
            "No trainable LoRA parameters attached. Check target_modules — "
            "names must match the model's named_modules() output."
        )
    return trainer


def count_parameters(model) -> dict[str, int]:
    """Return parameter counts for reporting."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / max(total, 1), 3),
    }
