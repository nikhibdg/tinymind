"""
tinymind.core.distiller
=======================
Main orchestrator for the TinyMind distillation pipeline.

Responsibilities:

* Applies LoRA adapters to the student (PEFT).
* Runs training with gradient accumulation and Accelerate mixed precision.
* Computes the combined :class:`~tinymind.core.losses.DistillationLoss`,
  :class:`~tinymind.core.losses.ReasoningLoss`, and
  :class:`~tinymind.core.losses.ContrastiveLoss`.
* Evaluates on a held-out dataloader and tracks the best checkpoint.
* Saves / restores checkpoints; delegates export to the configured backend.

Typical usage::

    from tinymind.core.config import DistillConfig
    from tinymind.core.distiller import Distiller

    cfg   = DistillConfig.from_yaml("configs/default.yaml")
    distiller = Distiller(cfg, teacher_model, student_model, tokenizer)
    history   = distiller.fit(train_dl, eval_dl)
    distiller.export("exports/student_Q4_K_M")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tinymind.core.config import DistillConfig
from tinymind.core.losses import (
    ContrastiveLoss,
    DistillationLoss,
    ReasoningLoss,
    mean_pool,
)

logger = logging.getLogger(__name__)


# ─── Optional heavy dependencies ──────────────────────────────────────────────

try:
    from accelerate import Accelerator

    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False
    logger.warning("accelerate not installed; mixed precision and multi-GPU unavailable.")

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False
    logger.warning("peft not installed; LoRA will be skipped.")

try:
    from transformers import get_scheduler

    _HAS_TRANSFORMERS_SCHED = True
except ImportError:
    _HAS_TRANSFORMERS_SCHED = False

try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ─── Mutable training state ───────────────────────────────────────────────────


@dataclass
class _TrainingState:
    """Mutable counters and history updated throughout :meth:`Distiller.fit`."""

    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float("inf")
    history: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, metrics: Dict[str, float]) -> None:
        """Append scalar metrics to per-key history lists."""
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)


# ─── Distiller ────────────────────────────────────────────────────────────────


class Distiller:
    """
    Orchestrates the full TinyMind knowledge-distillation pipeline.

    The constructor freezes the teacher, optionally wraps the student in LoRA
    adapters, builds all loss modules (eagerly), and sets up the AdamW
    optimiser over student + auxiliary-head parameters.

    Args:
        config:    Fully-populated :class:`~tinymind.core.config.DistillConfig`.
        teacher:   A *frozen* teacher model.  The constructor sets
                   ``requires_grad=False`` on all its parameters and switches
                   it to ``eval`` mode permanently.
        student:   The student model to distil into.  LoRA adapters are applied
                   in-place during ``__init__``.
        tokenizer: Shared tokeniser for both models (used during export and
                   checkpoint saving).
    """

    def __init__(
        self,
        config: DistillConfig,
        teacher: "PreTrainedModel",
        student: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        torch.manual_seed(config.seed)

        # 1. Freeze teacher ────────────────────────────────────────────────────
        self.teacher: "PreTrainedModel" = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # 2. Apply LoRA to student ─────────────────────────────────────────────
        self.student: "PreTrainedModel" = self._apply_lora(student)

        # 3. Build loss modules ────────────────────────────────────────────────
        (
            self._distill_loss,
            self._reasoning_loss,
            self._contrastive_loss,
        ) = self._build_losses()

        # 4. Optimiser (after LoRA so only adapter + aux params are included) ──
        self.optimizer = self._build_optimizer()

        # 5. Accelerator ───────────────────────────────────────────────────────
        if not _HAS_ACCELERATE:
            raise ImportError(
                "accelerate is required: pip install accelerate"
            )
        self.accelerator: "Accelerator" = self._build_accelerator()

        # 6. Weights & Biases ──────────────────────────────────────────────────
        self._wandb_run = None
        if config.wandb.enabled:
            self._init_wandb()

        # 7. Mutable state (reset on each call to fit()) ───────────────────────
        self._state = _TrainingState()

        logger.info(
            "Distiller ready.  Trainable parameters: %s",
            self._count_trainable_params(),
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full distillation training loop.

        The loop supports:

        * Gradient accumulation (configured via :attr:`~DistillConfig.gradient_accumulation_steps`).
        * Mixed-precision forward/backward via Accelerate.
        * Periodic evaluation, best-checkpoint tracking, and WandB logging.

        Args:
            train_dataloader: Yields batches with keys ``input_ids``,
                              ``attention_mask``, and ``labels``
                              (``-100`` to mask ignored positions).
            eval_dataloader:  Optional held-out set.  Evaluated every
                              :attr:`~DistillConfig.eval_steps` optimiser steps;
                              best checkpoint is saved automatically.

        Returns:
            History dict mapping metric names → list of recorded float values.
        """
        cfg = self.config
        self._state = _TrainingState()

        steps_per_epoch = math.ceil(
            len(train_dataloader) / cfg.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = self._build_scheduler(total_steps)

        # Prepare all objects for distributed / mixed-precision training
        prepared = self.accelerator.prepare(
            self.student,
            self.teacher,
            self.optimizer,
            train_dataloader,
        )
        self.student, self.teacher, self.optimizer, train_dataloader = prepared

        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Move auxiliary loss modules to the active device
        device = self.accelerator.device
        for module in filter(None, [self._reasoning_loss, self._contrastive_loss]):
            module.to(device)

        logger.info(
            "Starting training: %d epochs, ~%d optimiser steps.",
            cfg.epochs,
            total_steps,
        )

        for epoch in range(cfg.epochs):
            self._state.epoch = epoch
            self.student.train()

            for _micro_step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.student):
                    step_metrics = self._train_step(batch)
                    self.accelerator.backward(step_metrics["loss"])

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.student.parameters(),
                            cfg.max_grad_norm,
                        )

                    self.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.optimizer.zero_grad()

                # Post-sync bookkeeping
                if self.accelerator.sync_gradients:
                    self._state.global_step += 1
                    scalars = _to_scalars(step_metrics)
                    self._state.record(scalars)

                    if self._state.global_step % cfg.logging_steps == 0:
                        self._log(
                            {f"train/{k}": v for k, v in scalars.items()},
                            self._state.global_step,
                        )
                        self._log_lr(scheduler, self._state.global_step)

                    if (
                        eval_dataloader is not None
                        and self._state.global_step % cfg.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate(eval_dataloader)
                        self._log(
                            {f"eval/{k}": v for k, v in eval_metrics.items()},
                            self._state.global_step,
                        )
                        if eval_metrics["loss"] < self._state.best_eval_loss:
                            self._state.best_eval_loss = eval_metrics["loss"]
                            self.save(Path(cfg.output_dir) / "best")
                            logger.info(
                                "New best eval loss %.4f — checkpoint saved.",
                                self._state.best_eval_loss,
                            )

                    if self._state.global_step % cfg.save_steps == 0:
                        self.save(
                            Path(cfg.output_dir) / f"step_{self._state.global_step}"
                        )

            logger.info(
                "Epoch %d/%d complete — global step %d.",
                epoch + 1,
                cfg.epochs,
                self._state.global_step,
            )

        self.save(Path(cfg.output_dir) / "final")
        logger.info("Training complete.  Final checkpoint saved.")
        return self._state.history

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run a full evaluation pass over *dataloader*.

        Computes the same losses as the training step but without gradients.
        Perplexity is derived from the hard cross-entropy component.

        Args:
            dataloader: Evaluation DataLoader (same batch format as training).

        Returns:
            Dict with ``loss``, ``kd_loss``, ``ce_loss``, ``perplexity``, and
            any additional loss keys active in the training configuration.
        """
        self.student.eval()
        totals: Dict[str, float] = {}
        n = 0

        with torch.no_grad():
            for batch in dataloader:
                metrics = self._train_step(batch)
                for k, v in _to_scalars(metrics).items():
                    totals[k] = totals.get(k, 0.0) + v
                n += 1

        avgs = {k: v / max(n, 1) for k, v in totals.items()}
        avgs["perplexity"] = math.exp(min(avgs.get("ce_loss", float("inf")), 20.0))

        self.student.train()
        return avgs

    def export(
        self,
        output_path: Union[str, Path],
        backend: Optional[str] = None,
    ) -> Path:
        """
        Export the distilled student model to a mobile-ready format.

        Delegates to the backend specified in the config (or the *backend*
        override).  Currently supported backends:

        * ``phone_llm`` — export via PhoneLLM (Android / iOS).
        * ``onnx``      — ONNX export.
        * ``coreml``    — CoreML export (macOS / iOS).

        Args:
            output_path: Directory where the exported artefact is written.
            backend:     Override config export backend.

        Returns:
            :class:`~pathlib.Path` pointing to the exported artefact.

        Raises:
            ImportError: if the PhoneLLM exporter is not yet implemented.
            ValueError:  for unknown backend identifiers.
        """
        backend = backend or self.config.export.backend
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting student via backend=%s → %s", backend, output_path)

        if backend == "phone_llm":
            try:
                from tinymind.export.phone_llm import PhoneLLMExporter
            except ImportError as exc:
                raise ImportError(
                    "PhoneLLMExporter is not yet implemented. "
                    "See src/tinymind/export/phone_llm.py."
                ) from exc
            exporter = PhoneLLMExporter(
                config=self.config.export,
                model=self.accelerator.unwrap_model(self.student),
                tokenizer=self.tokenizer,
            )
            return exporter.export(output_path)

        if backend == "onnx":
            try:
                from tinymind.export.onnx_export import ONNXExporter
            except ImportError as exc:
                raise ImportError(
                    "ONNXExporter is not yet implemented. "
                    "See src/tinymind/export/onnx_export.py."
                ) from exc
            return ONNXExporter(self.config.export).export(
                self.accelerator.unwrap_model(self.student),
                output_path,
            )

        raise ValueError(
            f"Unknown export backend '{backend}'. "
            "Choose from: phone_llm, onnx, coreml."
        )

    def save(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Persist the student model, optimiser, and training metadata.

        Checkpoint layout::

            checkpoint_path/
            ├── student_model/        ← save_pretrained (weights + tokeniser)
            ├── optimizer.pt          ← optimiser state dict
            ├── training_state.pt     ← step, epoch, best_eval_loss
            └── config.yaml           ← full DistillConfig

        Args:
            checkpoint_path: Root directory for this checkpoint.
        """
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.student)
        unwrapped.save_pretrained(path / "student_model")
        self.tokenizer.save_pretrained(path / "student_model")

        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        torch.save(
            {
                "global_step": self._state.global_step,
                "epoch": self._state.epoch,
                "best_eval_loss": self._state.best_eval_loss,
            },
            path / "training_state.pt",
        )
        self.config.to_yaml(path / "config.yaml")
        logger.info("Checkpoint → %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        teacher: "PreTrainedModel",
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ) -> "Distiller":
        """
        Restore a :class:`Distiller` from a previously saved checkpoint.

        Reloads student weights, optimiser state, and training counters so
        training can resume seamlessly from an interrupted run.

        Args:
            checkpoint_path: Root directory written by :meth:`save`.
            teacher:         Teacher model (not stored in the checkpoint).
            tokenizer:       Optional override; loaded from the checkpoint
                             directory when omitted.

        Returns:
            Fully initialised :class:`Distiller` ready to resume or evaluate.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(checkpoint_path)
        cfg = DistillConfig.from_yaml(path / "config.yaml")

        student = AutoModelForCausalLM.from_pretrained(path / "student_model")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(path / "student_model")

        instance = cls(cfg, teacher, student, tokenizer)

        opt_state = torch.load(path / "optimizer.pt", map_location="cpu")
        instance.optimizer.load_state_dict(opt_state)

        state = torch.load(path / "training_state.pt", map_location="cpu")
        instance._state.global_step = state["global_step"]
        instance._state.epoch = state["epoch"]
        instance._state.best_eval_loss = state["best_eval_loss"]

        logger.info(
            "Resumed from %s (epoch %d, step %d)",
            path,
            instance._state.epoch,
            instance._state.global_step,
        )
        return instance

    # ─── Private: setup ───────────────────────────────────────────────────────

    def _apply_lora(self, student: "PreTrainedModel") -> "PreTrainedModel":
        """Wrap *student* in PEFT LoRA adapters, or return it unchanged."""
        if not _HAS_PEFT:
            logger.warning("peft not installed — training the full student without LoRA.")
            return student

        lc = self.config.lora
        peft_cfg = LoraConfig(
            r=lc.r,
            lora_alpha=lc.lora_alpha,
            lora_dropout=lc.lora_dropout,
            target_modules=lc.target_modules,
            bias=lc.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        student = get_peft_model(student, peft_cfg)
        student.print_trainable_parameters()
        return student

    def _build_losses(
        self,
    ) -> Tuple[
        DistillationLoss,
        Optional[ReasoningLoss],
        Optional[ContrastiveLoss],
    ]:
        """Construct all loss modules, using model configs to infer hidden dims."""
        cfg = self.config
        distill_loss = DistillationLoss(
            temperature=cfg.temperature,
            alpha=cfg.alpha,
        )

        reasoning_loss: Optional[ReasoningLoss] = None
        contrastive_loss: Optional[ContrastiveLoss] = None

        # Infer hidden sizes from the model's own config object
        try:
            s_dim: int = self.student.config.hidden_size
            t_dim: int = self.teacher.config.hidden_size
        except AttributeError:
            logger.warning(
                "Cannot determine model hidden sizes — "
                "ReasoningLoss and ContrastiveLoss disabled."
            )
            return distill_loss, None, None

        rd = cfg.reason_distill
        if rd.enabled and rd.align_hidden_states:
            reasoning_loss = ReasoningLoss(
                student_dim=s_dim,
                teacher_dim=t_dim,
                num_layers=len(rd.hidden_state_layers),
            )
            logger.info(
                "ReasoningLoss: %d layer pairs, student_dim=%d, teacher_dim=%d",
                len(rd.hidden_state_layers),
                s_dim,
                t_dim,
            )

        if rd.enabled and rd.contrastive_weight > 0:
            contrastive_loss = ContrastiveLoss(
                student_dim=s_dim,
                teacher_dim=t_dim,
            )
            logger.info("ContrastiveLoss enabled.")

        return distill_loss, reasoning_loss, contrastive_loss

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        AdamW optimiser with weight-decay separation.

        Bias terms and normalisation weights are excluded from weight decay
        (standard practice for transformer fine-tuning).  Auxiliary loss
        module projections are included in the decayed group.
        """
        decay, no_decay = [], []

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "LayerNorm.weight", "layer_norm.weight")):
                no_decay.append(param)
            else:
                decay.append(param)

        for module in filter(None, [self._reasoning_loss, self._contrastive_loss]):
            for param in module.parameters():
                if param.requires_grad:
                    decay.append(param)

        param_groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.config.learning_rate)

    def _build_accelerator(self) -> "Accelerator":
        mixed = "no"
        if self.config.fp16:
            mixed = "fp16"
        elif self.config.bf16:
            mixed = "bf16"

        return Accelerator(
            mixed_precision=mixed,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb" if (self.config.wandb.enabled and _HAS_WANDB) else None,
        )

    def _build_scheduler(self, num_training_steps: int):
        if not _HAS_TRANSFORMERS_SCHED:
            logger.warning("transformers not installed — no LR scheduler.")
            return None
        return get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _init_wandb(self) -> None:
        if not _HAS_WANDB:
            logger.warning("wandb not installed — experiment tracking disabled.")
            return
        wc = self.config.wandb
        self._wandb_run = _wandb.init(
            project=wc.project,
            entity=wc.entity,
            name=wc.run_name,
            tags=wc.tags,
            config=self.config.to_dict(),
        )

    # ─── Private: training step ───────────────────────────────────────────────

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a single (micro-)batch.

        Used identically during training (inside ``accumulate`` context) and
        evaluation (inside ``torch.no_grad``).  The caller is responsible for
        calling ``.backward()`` when in training mode.

        Args:
            batch: Dict with ``input_ids`` ``[B, T]``, ``attention_mask``
                   ``[B, T]`` (optional), and ``labels`` ``[B, T]``.

        Returns:
            Dict of named tensors.  ``"loss"`` is the total differentiable
            scalar; other keys are detached diagnostics.
        """
        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        labels: torch.Tensor = batch.get("labels", input_ids)

        needs_hidden = (
            self.config.reason_distill.enabled
            and self.config.reason_distill.align_hidden_states
        )
        needs_contrastive = (
            self._contrastive_loss is not None
            and self.config.reason_distill.contrastive_weight > 0
        )
        output_hidden = needs_hidden or needs_contrastive

        # ── Teacher forward (frozen; no gradients) ────────────────────────────
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden,
            )

        # ── Student forward (gradients enabled) ───────────────────────────────
        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden,
        )

        # ── Distillation loss (always active) ─────────────────────────────────
        distill_out = self._distill_loss(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            labels=labels,
            attention_mask=attention_mask,
        )
        total_loss: torch.Tensor = distill_out["loss"]
        out: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "kd_loss": distill_out["kd_loss"],
            "ce_loss": distill_out["ce_loss"],
        }

        # ── Reasoning / hidden-state alignment loss ───────────────────────────
        if (
            needs_hidden
            and self._reasoning_loss is not None
            and student_out.hidden_states
            and teacher_out.hidden_states
        ):
            s_layers = _select_layers(
                student_out.hidden_states,
                self.config.reason_distill.hidden_state_layers,
            )
            t_layers = _select_layers(
                teacher_out.hidden_states,
                self.config.reason_distill.hidden_state_layers,
            )
            reason_out = self._reasoning_loss(s_layers, t_layers)
            r_loss = reason_out["loss"] * self.config.reason_distill.reasoning_weight
            total_loss = total_loss + r_loss
            out["reasoning_loss"] = r_loss.detach()
            out["loss"] = total_loss

        # ── Contrastive representation loss ───────────────────────────────────
        if (
            needs_contrastive
            and self._contrastive_loss is not None
            and student_out.hidden_states
            and teacher_out.hidden_states
        ):
            s_repr = mean_pool(student_out.hidden_states[-1], attention_mask)
            t_repr = mean_pool(teacher_out.hidden_states[-1], attention_mask)
            contra_out = self._contrastive_loss(s_repr, t_repr)
            c_loss = contra_out["loss"] * self.config.reason_distill.contrastive_weight
            total_loss = total_loss + c_loss
            out["contrastive_loss"] = c_loss.detach()
            out["mean_positive_sim"] = contra_out["mean_positive_similarity"]
            out["loss"] = total_loss

        return out

    # ─── Private: logging ─────────────────────────────────────────────────────

    def _log(self, metrics: Dict[str, float], step: int) -> None:
        parts = [f"step={step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
        logger.info("  ".join(parts))
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def _log_lr(self, scheduler, step: int) -> None:
        if scheduler is None:
            return
        self._log({"train/lr": scheduler.get_last_lr()[0]}, step)

    def _count_trainable_params(self) -> str:
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        pct = 100.0 * trainable / max(total, 1)
        return f"{trainable:,} / {total:,} ({pct:.2f}%)"


# ─── Module-level utilities ───────────────────────────────────────────────────


def _select_layers(
    hidden_states: Tuple[torch.Tensor, ...],
    indices: List[int],
) -> List[torch.Tensor]:
    """
    Select hidden states by index, supporting negative (from-end) indexing.

    Args:
        hidden_states: Tuple of ``[B, T, D]`` tensors from a HuggingFace model.
        indices:       List of layer indices (negative OK).

    Returns:
        List of selected ``[B, T, D]`` tensors.
    """
    n = len(hidden_states)
    return [hidden_states[i % n] for i in indices]


def _to_scalars(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert a dict of tensors to plain Python floats for logging."""
    return {
        k: v.item()
        for k, v in metrics.items()
        if isinstance(v, torch.Tensor)
    }
