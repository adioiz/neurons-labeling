#!/usr/bin/env python3
"""
Top-level entry point for SAM 2 LoRA fine-tuning on neuron data.

Usage:
    # Step 1: Prepare training data from TIF stacks
    python train.py prepare --manifest data_manifest.json --output-dir training_data

    # Step 2: Train the model
    python train.py train

    # Step 2 (with custom settings):
    python train.py train --epochs 80 --lora-rank 16 --data-dir training_data

    # Step 2 (use ImageNet normalization instead of dataset stats):
    python train.py train --use-imagenet-stats

Example manifest.json:
    [
        {
            "image_tif": "/Volumes/Adi-SSD/EfratData/.../A31 C2.tif",
            "mask_tif": "labeled_masks/A31_C2_masks.tif",
            "name": "A31_C2"
        },
        ...
    ]
"""

import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cmd_prepare(args):
    """Prepare training data from TIF stacks."""
    from src.training.prepare_data import prepare_dataset

    with open(args.manifest) as f:
        manifest = json.load(f)

    prepare_dataset(
        manifest=manifest,
        output_dir=args.output_dir,
        label_id=args.label_id,
    )


def cmd_train(args):
    """Run the training loop."""
    from src.training.config import (
        TrainingConfig, LoRAConfig, DataConfig, OptimizerConfig, AugmentationConfig,
    )
    from src.training.train import train

    # If --use-dataset-stats is passed, leave normalize_mean/std as None
    # so that train.py auto-loads from manifest.json.
    # If --imagenet-stats is passed, use ImageNet stats explicitly.
    aug_config = AugmentationConfig()
    if args.use_imagenet_stats:
        aug_config.normalize_mean = [0.485, 0.456, 0.406]
        aug_config.normalize_std = [0.229, 0.224, 0.225]
    # else: normalize_mean/std remain None → auto-loaded from manifest

    config = TrainingConfig(
        num_epochs=args.epochs,
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        data=DataConfig(
            training_data_dir=args.data_dir,
            num_frames=args.num_frames,
            label_id=args.label_id,
            clip_stride=args.clip_stride,
            cache_in_memory=not args.no_cache,
            precompute_features=not args.no_precompute,
        ),
        optimizer=OptimizerConfig(
            lora_lr=args.lora_lr,
            base_lr=args.base_lr,
        ),
        augmentation=aug_config,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        val_every_n_epochs=args.val_every,
    )

    train(config)


def main():
    parser = argparse.ArgumentParser(
        description="SAM 2 LoRA Fine-Tuning for Neuron Segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- prepare command ---
    p_prepare = subparsers.add_parser(
        "prepare", help="Convert TIF stacks to training format"
    )
    p_prepare.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSON manifest listing image/mask TIF pairs",
    )
    p_prepare.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory",
    )
    p_prepare.add_argument(
        "--label-id",
        type=int,
        default=0,
        help="Label ID to extract (0 = all non-zero)",
    )

    # --- train command ---
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--lora-rank", type=int, default=8)
    p_train.add_argument("--lora-alpha", type=float, default=16.0)
    p_train.add_argument("--lora-lr", type=float, default=1e-4)
    p_train.add_argument("--base-lr", type=float, default=5e-6)
    p_train.add_argument("--num-frames", type=int, default=8)
    p_train.add_argument("--label-id", type=int, default=0)
    p_train.add_argument("--data-dir", type=str, default="training_data")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_train.add_argument("--val-every", type=int, default=5)
    p_train.add_argument(
        "--use-imagenet-stats",
        action="store_true",
        help="Use ImageNet normalization instead of dataset-specific stats (default: auto-load from manifest)",
    )
    # Performance optimization arguments
    p_train.add_argument(
        "--clip-stride",
        type=int,
        default=4,
        help="Stride between clip start frames to reduce overlap (default: 4, use 1 for all clips)",
    )
    p_train.add_argument(
        "--no-precompute",
        action="store_true",
        help="Disable image encoder feature precomputation (slower but keeps full augmentation)",
    )
    p_train.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory frame caching (saves RAM but slower on network storage)",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
