#!/usr/bin/env python3
"""
Entry point for running a TinyMind distillation training job.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="TinyMind training script")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config: {args.config}")
    print(f"Teacher: {cfg['distillation']['teacher_model']}")
    print(f"Student: {cfg['distillation']['student_model']}")
    print("Training pipeline not yet implemented — wire up tinymind.core.trainer here.")


if __name__ == "__main__":
    main()
