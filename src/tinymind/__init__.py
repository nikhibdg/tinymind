"""
TinyMind: End-to-end pipeline for LLM distillation, chain-of-thought injection,
and offline mobile deployment.
"""

__version__ = "0.1.0"
__author__ = "TinyMind Contributors"

from tinymind.core.config import DistillConfig
from tinymind.core.distiller import Distiller
from tinymind.core.losses import DistillationLoss, ReasoningLoss, ContrastiveLoss

__all__ = ["DistillConfig", "Distiller", "DistillationLoss", "ReasoningLoss", "ContrastiveLoss"]
