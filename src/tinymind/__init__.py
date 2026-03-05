"""
TinyMind: End-to-end pipeline for LLM distillation, chain-of-thought injection,
and offline mobile deployment.
"""

__version__ = "0.1.0"
__author__ = "TinyMind Contributors"

from tinymind.core import distiller, trainer
from tinymind.methods import reason_distill
from tinymind.export import phone_llm

__all__ = ["distiller", "trainer", "reason_distill", "phone_llm"]
