"""
tinymind.core.config
====================
Central configuration dataclasses for a TinyMind distillation run.

Usage::

    cfg = DistillConfig.from_yaml("configs/default.yaml")
    cfg.to_yaml("checkpoints/run_01/config.yaml")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────


@dataclass
class LoRAConfig:
    """PEFT / LoRA adapter settings applied to the student model."""

    r: int = 16
    """Rank of the low-rank update matrices.  Higher = more capacity, more VRAM."""

    lora_alpha: int = 32
    """Scaling factor.  Effective LR scale = lora_alpha / r."""

    lora_dropout: float = 0.05
    """Dropout probability on LoRA activations (regularisation)."""

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    """Transformer projection names to attach adapters to."""

    bias: Literal["none", "all", "lora_only"] = "none"
    """Which bias tensors to make trainable alongside the adapters."""

    task_type: str = "CAUSAL_LM"
    """PEFT task type string passed to LoraConfig."""


@dataclass
class WandbConfig:
    """Weights & Biases experiment-tracking settings."""

    enabled: bool = False
    project: str = "tinymind"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_interval: int = 10
    """Emit metrics to W&B every N optimiser steps."""


@dataclass
class ExportConfig:
    """Mobile / edge deployment export settings."""

    backend: Literal["phone_llm", "onnx", "coreml"] = "phone_llm"
    quantization: Literal["int4", "int8", "fp16", "none"] = "int4"
    target_device: Literal["android", "ios", "generic"] = "android"
    output_dir: str = "exports/"


@dataclass
class ReasonDistillConfig:
    """Chain-of-thought / reasoning-signal alignment settings."""

    enabled: bool = True

    cot_injection_layer: int = -1
    """Layer index at which CoT signals are injected.  -1 = last layer."""

    reasoning_weight: float = 0.3
    """Weight of ReasoningLoss relative to DistillationLoss in the total objective."""

    align_hidden_states: bool = True
    """Activate layer-wise hidden-state MSE / cosine alignment."""

    hidden_state_layers: List[int] = field(default_factory=lambda: [-1, -4, -8])
    """Layer indices whose hidden states are aligned.  Negative = from the end."""

    contrastive_weight: float = 0.1
    """Weight of ContrastiveLoss in the total objective.  Set 0 to disable."""


# ─── Main config ──────────────────────────────────────────────────────────────


@dataclass
class DistillConfig:
    """
    Central configuration for a TinyMind distillation run.

    All nested sub-configs (``lora``, ``wandb``, ``export``, ``reason_distill``)
    are accessible as typed attributes and round-trip cleanly through YAML::

        cfg = DistillConfig.from_yaml("configs/default.yaml")
        # ... modify ...
        cfg.to_yaml("checkpoints/run_01/config.yaml")

    Attributes:
        teacher_model:               HuggingFace model ID or local path for the teacher.
        student_model:               HuggingFace model ID or local path for the student.
        temperature:                 KD softmax temperature (T > 1 = softer distributions).
        alpha:                       Blend weight: ``alpha * L_KD + (1 - alpha) * L_CE``.
        epochs:                      Number of full passes over the training data.
        batch_size:                  Per-device micro-batch size.
        gradient_accumulation_steps: Number of micro-steps before an optimiser update.
        learning_rate:               Peak learning rate for AdamW.
        weight_decay:                L2 regularisation on non-bias / non-norm parameters.
        max_grad_norm:               Gradient clipping threshold.
        warmup_steps:                Linear warmup duration (in optimiser steps).
        lr_scheduler:                LR schedule type (cosine | linear | constant | …).
        fp16:                        Enable float16 mixed precision via Accelerate.
        bf16:                        Enable bfloat16 mixed precision (preferred on Ampere+).
        max_seq_len:                 Maximum sequence length in tokens.
        output_dir:                  Directory for checkpoints.
        seed:                        Global random seed for reproducibility.
        logging_steps:               Log training metrics every N optimiser steps.
        eval_steps:                  Run evaluation every N optimiser steps.
        save_steps:                  Save a checkpoint every N optimiser steps.
    """

    # ── Models ────────────────────────────────────────────────────────────────
    teacher_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    student_model: str = "meta-llama/Llama-3.2-1B"

    # ── Distillation hyper-parameters ─────────────────────────────────────────
    temperature: float = 2.0
    alpha: float = 0.5

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    lr_scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = "cosine"
    fp16: bool = True
    bf16: bool = False
    max_seq_len: int = 2048
    output_dir: str = "checkpoints/"
    seed: int = 42

    # ── Logging / checkpoint cadence ──────────────────────────────────────────
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500

    # ── Sub-configs ───────────────────────────────────────────────────────────
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    reason_distill: ReasonDistillConfig = field(default_factory=ReasonDistillConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ─── Validation ───────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16 simultaneously.")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.export.output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Serialisation ────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DistillConfig":
        """Load a :class:`DistillConfig` from a YAML file.

        Any key present in the YAML overrides the dataclass default.
        Missing keys fall back silently to defaults, so partial configs are safe.

        Args:
            path: Path to a YAML file produced by :meth:`to_yaml` or written
                  by hand following the ``configs/default.yaml`` layout.

        Returns:
            A fully validated :class:`DistillConfig` instance.
        """
        with open(path, encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise this config to a YAML file.

        Args:
            path: Destination file path.  Parent directories are created
                  automatically.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` representation (fully nested, JSON-safe)."""
        return asdict(self)

    def __repr__(self) -> str:  # pragma: no cover
        return json.dumps(self.to_dict(), indent=2, default=str)

    # ─── Internal ─────────────────────────────────────────────────────────────

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "DistillConfig":
        """Construct from a (possibly partial) flat-or-nested dict."""

        def _extract(sub_cls, key: str) -> Any:
            raw = d.pop(key, {}) or {}
            valid = {k: v for k, v in raw.items() if k in sub_cls.__dataclass_fields__}
            return sub_cls(**valid)

        lora = _extract(LoRAConfig, "lora")
        reason_distill = _extract(ReasonDistillConfig, "reason_distill")
        wandb_cfg = _extract(WandbConfig, "wandb")
        export = _extract(ExportConfig, "export")

        top_level = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(
            lora=lora,
            reason_distill=reason_distill,
            wandb=wandb_cfg,
            export=export,
            **top_level,
        )
