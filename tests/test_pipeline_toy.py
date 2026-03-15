#!/usr/bin/env python3
"""
tests/test_pipeline_toy.py
End-to-end CPU smoke test for the TinyMind distillation pipeline.

Run:
    python tests/test_pipeline_toy.py
"""
from __future__ import annotations

import atexit
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
BATCH_SIZE = 1
N_TRAIN = 3    # 3 batches at batch_size=1 → 3 training steps
N_EVAL = 10

# Temp directory persists for the whole run, cleaned up on exit
TMPDIR = Path(tempfile.mkdtemp(prefix="tinymind_toy_"))
atexit.register(shutil.rmtree, TMPDIR, ignore_errors=True)

results: list[tuple[str, bool]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    symbol = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {symbol}: {name}" + (f"\n         {detail}" if detail else ""))
    results.append((name, passed))


def _scalar(v) -> float:
    return v.item() if isinstance(v, torch.Tensor) else float(v)


def _is_finite(v) -> bool:
    return math.isfinite(_scalar(v))


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_examples(n: int = 50) -> list[dict]:
    """Generate n arithmetic Q&A examples with chain-of-thought reasoning."""
    examples = []
    ops = ["+", "-", "*"]
    for i in range(n):
        a = (i * 7 + 3) % 20 + 1
        b = (i * 3 + 5) % 15 + 1
        op = ops[i % 3]
        if op == "+":
            ans = a + b
            thought = f"Add {a} and {b}. {a} + {b} = {ans}."
        elif op == "-":
            ans = a - b
            thought = f"Subtract {b} from {a}. {a} - {b} = {ans}."
        else:
            ans = a * b
            thought = f"Multiply {a} by {b}. {a} * {b} = {ans}."
        examples.append({
            "prompt": f"What is {a} {op} {b}?",
            "reasoning": f"<think>{thought}</think>",
            "answer": str(ans),
        })
    return examples


def build_dataloader(
    examples: list[dict],
    tokenizer,
    max_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Tokenize examples and return a DataLoader with input_ids/attention_mask/labels."""
    texts = [f"{e['prompt']}\n{e['reasoning']}\n{e['answer']}" for e in examples]
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"]           # [N, T]
    masks = enc["attention_mask"]    # [N, T]
    labels = ids.clone()
    labels[masks == 0] = -100        # ignore padding positions

    ds = TensorDataset(ids, masks, labels)

    def collate(batch):
        i_, m_, l_ = zip(*batch)
        return {
            "input_ids": torch.stack(i_),
            "attention_mask": torch.stack(m_),
            "labels": torch.stack(l_),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False)


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  TinyMind toy pipeline smoke test")
print(f"  model={MODEL_ID}")
print(f"  device=cpu  seq_len={SEQ_LEN}  batch={BATCH_SIZE}  train_steps={N_TRAIN}")
print("=" * 62)


# ── Check 1: Config instantiation ─────────────────────────────────────────────
print("\n[1/9] Config instantiation...")
try:
    from tinymind.core.config import (
        DistillConfig, LoRAConfig, WandbConfig,
        ReasonDistillConfig, ExportConfig,
    )

    cfg = DistillConfig(
        teacher_model=MODEL_ID,
        student_model=MODEL_ID,
        fp16=False,
        bf16=False,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        epochs=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lora=LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]),
        reason_distill=ReasonDistillConfig(
            enabled=True,
            align_hidden_states=True,
            contrastive_weight=0.1,
        ),
        wandb=WandbConfig(enabled=False),
        output_dir=str(TMPDIR / "checkpoints"),
        export=ExportConfig(output_dir=str(TMPDIR / "exports")),
    )
    assert cfg.lora.r == 4, f"Expected lora.r=4, got {cfg.lora.r}"
    assert not cfg.fp16, "fp16 should be False for CPU test"
    assert cfg.max_seq_len == SEQ_LEN
    record("Config instantiation", True, f"lora_r={cfg.lora.r}, max_seq_len={cfg.max_seq_len}")
except Exception as exc:
    record("Config instantiation", False, str(exc))
    print("\nFatal: cannot continue without config.")
    sys.exit(1)


# ── Check 2: Model loading ─────────────────────────────────────────────────────
print("\n[2/9] Loading models (first run downloads ~1.2 GB)...")
t0 = time.time()
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    )
    elapsed = time.time() - t0
    hidden = teacher_model.config.hidden_size
    record("Model loading", True, f"hidden_size={hidden}, loaded in {elapsed:.1f}s")
except Exception as exc:
    record("Model loading", False, str(exc))
    print("\nFatal: cannot continue without models.")
    sys.exit(1)


# ── Check 3: LoRA attachment (Distiller init) ──────────────────────────────────
print("\n[3/9] Creating Distiller (applies LoRA to student)...")
try:
    from tinymind.core.distiller import Distiller

    distiller = Distiller(cfg, teacher_model, student_model, tokenizer)

    # Teacher must be fully frozen
    teacher_trainable = sum(1 for p in distiller.teacher.parameters() if p.requires_grad)
    assert teacher_trainable == 0, f"Teacher has {teacher_trainable} trainable params (expected 0)"

    # Student must have trainable parameters
    student_trainable = [
        (n, p) for n, p in distiller.student.named_parameters() if p.requires_grad
    ]
    assert student_trainable, "Student has no trainable parameters after LoRA"

    lora_params = [n for n, _ in student_trainable if "lora" in n.lower()]
    assert lora_params, "No LoRA-named parameters found in student"

    record(
        "LoRA attachment", True,
        f"teacher=frozen, student={len(student_trainable)} trainable tensors "
        f"({len(lora_params)} LoRA)",
    )
except Exception as exc:
    record("LoRA attachment", False, str(exc))
    print("\nFatal: cannot continue without Distiller.")
    sys.exit(1)


# ── Check 4: Data tokenization ─────────────────────────────────────────────────
print("\n[4/9] Tokenizing synthetic data...")
try:
    all_examples = make_examples(50)
    assert len(all_examples) == 50

    # Spot-check structure and reasoning format
    ex = all_examples[0]
    assert {"prompt", "reasoning", "answer"} <= ex.keys()
    assert ex["reasoning"].startswith("<think>") and ex["reasoning"].endswith("</think>")

    train_dl = build_dataloader(all_examples[:N_TRAIN], tokenizer, SEQ_LEN, BATCH_SIZE)
    eval_dl  = build_dataloader(all_examples[40:40 + N_EVAL], tokenizer, SEQ_LEN, BATCH_SIZE)

    sample = next(iter(train_dl))
    assert sample["input_ids"].shape == (BATCH_SIZE, SEQ_LEN)
    assert sample["attention_mask"].shape == (BATCH_SIZE, SEQ_LEN)
    assert sample["labels"].shape == (BATCH_SIZE, SEQ_LEN)
    assert (sample["labels"] != -100).any(), "All label positions are masked"

    record(
        "Data tokenization", True,
        f"50 examples, train={N_TRAIN} batches, eval={N_EVAL} batches, "
        f"shape={tuple(sample['input_ids'].shape)}",
    )
except Exception as exc:
    record("Data tokenization", False, str(exc))
    sys.exit(1)


# ── Check 5: Forward pass ──────────────────────────────────────────────────────
print("\n[5/9] Forward pass (CPU inference)...")
try:
    batch = next(iter(train_dl))
    inp = batch["input_ids"]
    msk = batch["attention_mask"]

    distiller.teacher.eval()
    distiller.student.eval()

    with torch.no_grad():
        t_out = distiller.teacher(input_ids=inp, attention_mask=msk)
        s_out = distiller.student(input_ids=inp, attention_mask=msk)

    assert s_out.logits.shape == t_out.logits.shape, (
        f"Shape mismatch: student={s_out.logits.shape}, teacher={t_out.logits.shape}"
    )
    B, T, V = s_out.logits.shape
    assert B == BATCH_SIZE and T == SEQ_LEN, f"Unexpected batch/seq dims: B={B}, T={T}"

    record("Forward pass", True, f"logits={tuple(s_out.logits.shape)}  [B, T, vocab_size={V}]")
    del t_out, s_out
except Exception as exc:
    record("Forward pass", False, str(exc))


# ── Check 6: Loss computation ──────────────────────────────────────────────────
print("\n[6/9] Loss computation (DistillationLoss + ReasoningLoss + ContrastiveLoss)...")
try:
    distiller.student.train()
    metrics6 = distiller._train_step(batch)

    # Base losses always present
    for key in ("loss", "kd_loss", "ce_loss"):
        assert key in metrics6, f"Missing key '{key}' in _train_step output"
        assert _is_finite(metrics6[key]), f"{key}={_scalar(metrics6[key])} is not finite"

    # Optional auxiliary losses
    for key in ("reasoning_loss", "contrastive_loss"):
        if key in metrics6:
            assert _is_finite(metrics6[key]), f"{key}={_scalar(metrics6[key])} is not finite"

    scalar_keys = [k for k, v in metrics6.items() if isinstance(v, torch.Tensor) and v.ndim == 0]
    summary = "  ".join(f"{k}={_scalar(metrics6[k]):.4f}" for k in scalar_keys)
    record("Loss computation", True, summary)
    del metrics6
except Exception as exc:
    record("Loss computation", False, str(exc))


# ── Check 7: Backward pass ─────────────────────────────────────────────────────
print("\n[7/9] Backward pass...")
try:
    distiller.optimizer.zero_grad()
    distiller.student.train()
    metrics7 = distiller._train_step(batch)
    loss7 = metrics7["loss"]
    loss7.backward()

    # Every trainable student parameter must have a gradient
    no_grad = [
        n for n, p in distiller.student.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, (
        f"{len(no_grad)} trainable params have None grad: {no_grad[:3]}"
    )

    # Teacher parameters must have no gradient
    teacher_with_grad = [
        n for n, p in distiller.teacher.named_parameters() if p.grad is not None
    ]
    assert not teacher_with_grad, f"Teacher has grads on: {teacher_with_grad[:3]}"

    n_trainable = sum(1 for _, p in distiller.student.named_parameters() if p.requires_grad)
    record(
        "Backward pass", True,
        f"{n_trainable} student params have grads, teacher grads=None",
    )
    del metrics7, loss7
    distiller.optimizer.zero_grad()
except Exception as exc:
    record("Backward pass", False, str(exc))


# ── Check 8: Checkpoint save ───────────────────────────────────────────────────
print("\n[8/9] Saving checkpoint...")
try:
    ckpt_path = TMPDIR / "checkpoint_step0"
    distiller.save(ckpt_path)

    required = [
        ckpt_path / "student_model",    # directory (PEFT adapter weights)
        ckpt_path / "optimizer.pt",
        ckpt_path / "training_state.pt",
        ckpt_path / "config.yaml",
    ]
    for p in required:
        assert p.exists(), f"Missing checkpoint artifact: {p}"

    adapter_files = list((ckpt_path / "student_model").iterdir())
    assert adapter_files, "student_model/ directory is empty"

    record(
        "Checkpoint save", True,
        f"→ {ckpt_path.name}/  ({len(adapter_files)} files in student_model/)",
    )
except Exception as exc:
    record("Checkpoint save", False, str(exc))


# ── Check 9: Evaluate ──────────────────────────────────────────────────────────
print("\n[9/9] Evaluating on held-out examples...")
try:
    eval_metrics = distiller.evaluate(eval_dl)

    for key in ("loss", "perplexity"):
        assert key in eval_metrics, f"Missing key '{key}' in eval output"
        assert isinstance(eval_metrics[key], float), f"'{key}' is not a float"
        assert math.isfinite(eval_metrics[key]), f"'{key}'={eval_metrics[key]} is not finite"
        assert eval_metrics[key] > 0, f"'{key}'={eval_metrics[key]} should be positive"

    record(
        "Evaluation", True,
        f"loss={eval_metrics['loss']:.4f}, perplexity={eval_metrics['perplexity']:.2f}",
    )
except Exception as exc:
    record("Evaluation", False, str(exc))


# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
passed = sum(ok for _, ok in results)
total = len(results)
print(f"  {passed}/{total} checks passed\n")
for name, ok in results:
    print(f"  {'✓' if ok else '✗'} {name}")
print("=" * 62)

sys.exit(0 if passed == total else 1)
