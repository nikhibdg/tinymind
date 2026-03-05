"""
tinymind.core.losses
====================
Composable PyTorch loss modules for knowledge distillation.

Three building blocks:

* :class:`DistillationLoss`  — soft KL divergence (temperature-scaled) + hard CE.
* :class:`ReasoningLoss`     — layer-wise hidden-state alignment (cosine / MSE).
* :class:`ContrastiveLoss`   — symmetric InfoNCE for representation alignment.

All modules return a ``dict`` so callers can log individual components without
adding boilerplate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Shared utilities ─────────────────────────────────────────────────────────


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over positions where *mask* is truthy, numerically safe.

    Args:
        values: Arbitrary shape ``[...]``.
        mask:   Same shape as *values*; non-zero positions are included.

    Returns:
        Scalar tensor.
    """
    mask = mask.bool()
    denom = mask.sum().clamp(min=1)
    return (values * mask).sum() / denom


# ─── DistillationLoss ─────────────────────────────────────────────────────────


class DistillationLoss(nn.Module):
    """
    Standard knowledge-distillation objective for causal language models.

    Combines a **soft KL loss** (distribution matching under temperature *T*)
    with a **hard cross-entropy loss** (ground-truth next-token prediction)::

        L = alpha * L_KD + (1 - alpha) * L_CE

    The soft loss is scaled by ``T²`` to preserve gradient magnitude as *T*
    grows (Hinton et al., 2015).

    Args:
        temperature: Softmax temperature *T* > 0.  Higher values produce
                     softer probability distributions.
        alpha:       Weight on the soft KD loss ∈ [0, 1].
                     ``1 − alpha`` is allocated to the hard CE loss.

    Example::

        criterion = DistillationLoss(temperature=2.0, alpha=0.5)
        out = criterion(student_logits, teacher_logits, labels)
        out["loss"].backward()
        print(out["kd_loss"], out["ce_loss"])
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined distillation loss.

        The logits are shifted by one position so the model is trained to
        predict token *t* from context *[0 … t-1]*, matching the standard
        causal-LM convention.

        Args:
            student_logits: ``[B, T, V]`` — raw student output logits.
            teacher_logits: ``[B, T, V]`` — raw teacher output logits (detached).
            labels:         ``[B, T]``    — target token ids; ``-100`` marks
                            positions excluded from the loss.
            attention_mask: ``[B, T]``    — optional; 0-entries are excluded
                            from the soft loss.

        Returns:
            Dict with:

            * ``loss``     — total scalar to backpropagate.
            * ``kd_loss``  — detached soft KL component.
            * ``ce_loss``  — detached hard CE component.
        """
        B, T, V = student_logits.shape

        # ── Shift: predict token t using tokens [0..t-1] ──────────────────────
        s_logits = student_logits[:, :-1].contiguous()   # [B, T-1, V]
        t_logits = teacher_logits[:, :-1].contiguous()   # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()         # [B, T-1]

        # ── Hard cross-entropy on ground-truth labels ──────────────────────────
        ce_loss = F.cross_entropy(
            s_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        # ── Soft KL divergence (temperature-scaled) ────────────────────────────
        # Mask out padding and ignored positions
        valid = shift_labels != -100  # [B, T-1]
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:].bool()

        s_log_p = F.log_softmax(s_logits / self.temperature, dim=-1)  # [B, T-1, V]
        t_p = F.softmax(t_logits / self.temperature, dim=-1)           # [B, T-1, V]

        # KL per token (sum over vocab), then mask-average over positions
        kl_per_token = F.kl_div(
            s_log_p.reshape(-1, V),
            t_p.reshape(-1, V),
            reduction="none",
        ).sum(dim=-1).reshape(B, T - 1)                                # [B, T-1]

        kd_loss = _masked_mean(kl_per_token, valid) * (self.temperature ** 2)

        # ── Combine ────────────────────────────────────────────────────────────
        loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        return {
            "loss": loss,
            "kd_loss": kd_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, alpha={self.alpha}"


# ─── ReasoningLoss ────────────────────────────────────────────────────────────


class ReasoningLoss(nn.Module):
    """
    Layer-wise hidden-state alignment between teacher and student.

    For each aligned layer pair the student hidden states are optionally
    projected into the teacher's representation space (when dimensions differ)
    and the **cosine distance** (default) or **MSE** is minimised::

        dist_i = 1 - cos_sim(proj(s_i), t_i)          [normalize=True]
        dist_i = MSE(proj(s_i), t_i)                   [normalize=False]
        L_reason = mean_i(mean_tokens(dist_i))

    A *reasoning mask* can up-weight positions that correspond to chain-of-
    thought tokens, focusing alignment on the reasoning trace rather than
    routine next-token prediction.

    Args:
        student_dim: Hidden size of the student model.
        teacher_dim: Hidden size of the teacher model.
        num_layers:  Number of ``(student_layer, teacher_layer)`` pairs to align.
        normalize:   Use cosine distance (True) or MSE (False).

    Example::

        r_loss = ReasoningLoss(student_dim=2048, teacher_dim=4096, num_layers=3)
        out = r_loss(student_hiddens, teacher_hiddens, cot_mask)
        print(out["loss"], out["layer_losses"])
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_layers: int = 3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.normalize = normalize

        # One linear projection per layer pair (identity when dims match)
        if student_dim != teacher_dim:
            self.projections: Optional[nn.ModuleList] = nn.ModuleList(
                [nn.Linear(student_dim, teacher_dim, bias=False) for _ in range(num_layers)]
            )
        else:
            self.projections = None

    def forward(
        self,
        student_hidden: List[torch.Tensor],
        teacher_hidden: List[torch.Tensor],
        reasoning_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the hidden-state alignment loss.

        Args:
            student_hidden: List of ``[B, T, D_s]`` tensors, one per aligned layer.
            teacher_hidden: List of ``[B, T, D_t]`` tensors, same length.
                            These should be detached (no gradients through teacher).
            reasoning_mask: Optional ``[B, T]`` float / bool tensor.  When given,
                            the per-token loss is averaged only over marked positions,
                            amplifying the signal from CoT token spans.

        Returns:
            Dict with:

            * ``loss``         — mean loss across all layer pairs.
            * ``layer_losses`` — list of per-layer scalar tensors (detached).

        Raises:
            ValueError: if ``student_hidden`` and ``teacher_hidden`` have
                        different lengths.
        """
        if len(student_hidden) != len(teacher_hidden):
            raise ValueError(
                f"student_hidden and teacher_hidden must be the same length; "
                f"got {len(student_hidden)} vs {len(teacher_hidden)}"
            )

        layer_losses: List[torch.Tensor] = []

        for i, (s_h, t_h) in enumerate(zip(student_hidden, teacher_hidden)):
            t_h = t_h.detach()

            # Optional dimension projection
            if self.projections is not None and i < len(self.projections):
                s_h = self.projections[i](s_h)

            if self.normalize:
                s_h = F.normalize(s_h, dim=-1)
                t_h = F.normalize(t_h, dim=-1)
                dist = 1.0 - (s_h * t_h).sum(dim=-1)   # [B, T], ∈ [0, 2]
            else:
                dist = F.mse_loss(s_h, t_h, reduction="none").mean(dim=-1)  # [B, T]

            if reasoning_mask is not None:
                layer_loss = _masked_mean(dist, reasoning_mask)
            else:
                layer_loss = dist.mean()

            layer_losses.append(layer_loss)

        total = torch.stack(layer_losses).mean()
        return {
            "loss": total,
            "layer_losses": [ll.detach() for ll in layer_losses],
        }

    def extra_repr(self) -> str:
        return (
            f"student_dim={self.student_dim}, teacher_dim={self.teacher_dim}, "
            f"normalize={self.normalize}"
        )


# ─── ContrastiveLoss ──────────────────────────────────────────────────────────


class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) contrastive loss for representation alignment.

    Pulls the pooled student representation toward the corresponding pooled
    teacher representation while pushing all other in-batch pairs apart.
    Both streams are independently projected to a shared ``embed_dim``-
    dimensional space by two-layer MLP heads (following SimCLR)::

        z_s = MLP_s(student_repr) / ‖…‖           [B, E]
        z_t = MLP_t(teacher_repr) / ‖…‖           [B, E]

        logits_st = z_s @ z_t.T / tau             [B, B]
        L = (CE(logits_st, diagonal) + CE(logits_ts, diagonal)) / 2

    The temperature here is a *sharpness* parameter (smaller = harder
    negatives), distinct from the distillation temperature in
    :class:`DistillationLoss`.

    Args:
        student_dim: Hidden size of the student model (projection input dim).
        teacher_dim: Hidden size of the teacher model (projection input dim).
        embed_dim:   Shared projection output dimension.
        temperature: InfoNCE temperature τ > 0.

    Example::

        c_loss = ContrastiveLoss(student_dim=2048, teacher_dim=4096, embed_dim=256)
        # student_repr, teacher_repr: mean-pooled last hidden state [B, D]
        out = c_loss(student_repr, teacher_repr)
        print(out["loss"], out["mean_positive_similarity"])
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(inplace=True),
            nn.Linear(student_dim, embed_dim),
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim, embed_dim),
        )

    def forward(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the symmetric InfoNCE loss.

        Args:
            student_repr: Mean-pooled student hidden state, ``[B, D_s]``.
            teacher_repr: Mean-pooled teacher hidden state, ``[B, D_t]``.
                          Detached internally; no gradients flow to the teacher.

        Returns:
            Dict with:

            * ``loss``                      — symmetric InfoNCE scalar.
            * ``student_to_teacher_loss``   — detached CE(s→t) component.
            * ``teacher_to_student_loss``   — detached CE(t→s) component.
            * ``mean_positive_similarity``  — detached mean cosine sim of
              correct pairs (diagnostic; higher = better alignment).
        """
        B = student_repr.size(0)

        z_s = F.normalize(self.student_proj(student_repr), dim=-1)          # [B, E]
        z_t = F.normalize(self.teacher_proj(teacher_repr.detach()), dim=-1) # [B, E]

        # All-pairs cosine similarity matrix divided by temperature
        logits_st = torch.matmul(z_s, z_t.T) / self.temperature  # [B, B]
        logits_ts = logits_st.T

        targets = torch.arange(B, device=student_repr.device)
        loss_st = F.cross_entropy(logits_st, targets)
        loss_ts = F.cross_entropy(logits_ts, targets)
        loss = (loss_st + loss_ts) * 0.5

        # Diagnostic: cosine sim of positive pairs (before /tau scaling)
        mean_pos_sim = (logits_st.diagonal() * self.temperature).mean().detach()

        return {
            "loss": loss,
            "student_to_teacher_loss": loss_st.detach(),
            "teacher_to_student_loss": loss_ts.detach(),
            "mean_positive_similarity": mean_pos_sim,
        }

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


# ─── Pooling helper (used by Distiller) ───────────────────────────────────────


def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention-mask-weighted mean pooling of token hidden states.

    Args:
        hidden_state:   ``[B, T, D]`` last (or any) layer hidden states.
        attention_mask: ``[B, T]`` float or bool mask; 1 = real token.
                        If None, all positions are averaged uniformly.

    Returns:
        ``[B, D]`` pooled representation.
    """
    if attention_mask is None:
        return hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
    summed = (hidden_state * mask).sum(dim=1)            # [B, D]
    lengths = mask.sum(dim=1).clamp(min=1e-9)            # [B, 1]
    return summed / lengths
