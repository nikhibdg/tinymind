# TinyMind

End-to-end pipeline for distilling large language models, injecting chain-of-thought
reasoning via **ReasonDistill**, and deploying offline to mobile via **PhoneLLM**.

Default model pair: **DeepSeek-R1** (teacher) → **Qwen3-4B** (student), exported as **Q4_K_M GGUF**.

## Architecture

```
deepseek-ai/DeepSeek-R1  (teacher — frozen)
    │
    ▼  ReasonDistill (CoT injection)
Qwen/Qwen3-4B  (student — LoRA fine-tuned)
    │
    ▼  PhoneLLM export (Q4_K_M quantization)
Android / iOS  (offline inference)
```

## Subpackages

| Package | Purpose |
|---|---|
| `tinymind.core` | Base distiller, training loop, model manager |
| `tinymind.methods` | ReasonDistill, KD losses, CoT injection strategies |
| `tinymind.export` | PhoneLLM, ONNX, quantization, CoreML |
| `tinymind.benchmarks` | Evaluation harness, metrics, reporting |

## Quickstart

```bash
# 1. Create conda environment
conda create -n tinymind python=3.11 -y
conda activate tinymind

# 2. Install in editable mode
pip install -e ".[dev]"

# 3. Run training
python scripts/train.py --config configs/default.yaml

# 4. Export to mobile
python scripts/export.py --checkpoint checkpoints/best --config configs/default.yaml
```

## Project Structure

```
tinymind/
├── src/tinymind/
│   ├── core/          # distiller, trainer, model_manager
│   ├── methods/       # reason_distill, kd_loss, cot_injection
│   ├── export/        # phone_llm, quantizer, onnx_export
│   └── benchmarks/    # evaluator, metrics, reporter
├── tests/
├── examples/
├── docs/
├── configs/
│   └── default.yaml
└── scripts/
    ├── train.py
    └── export.py
```

## License

MIT
