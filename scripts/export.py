#!/usr/bin/env python3
"""
Export a trained TinyMind checkpoint to a mobile-ready format via PhoneLLM.

Usage:
    python scripts/export.py --checkpoint checkpoints/best --config configs/default.yaml
"""

import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="TinyMind export script")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backend = cfg["export"]["backend"]
    quant = cfg["export"]["quantization"]
    print(f"Exporting {args.checkpoint} via {backend} with {quant} quantization.")
    print("Export pipeline not yet implemented — wire up tinymind.export here.")


if __name__ == "__main__":
    main()
