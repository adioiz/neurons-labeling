"""
Standalone training script for SAM 2 LoRA fine-tuning on neuron data.

This bypasses Meta's complex distributed Trainer and uses a simple PyTorch
training loop suitable for single-GPU training.

Key optimization: when `config.data.precompute_features` is True (default),
the frozen image encoder's output is precomputed once for all frames before
training begins. This eliminates ~60-70% of per-step computation since the
encoder doesn't need to re-process images every step.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2, _hf_download

from src.training.config import TrainingConfig
from src.training.lora import (
    apply_lora,
    get_trainable_parameters,
    save_lora_weights,
    SAM2_LORA_TARGETS,
)
from src.training.neuron_dataset import (
    NeuronVideoDataset,
    neuron_collate_fn,
    get_video_names,
    train_val_split,
)
from src.training.augmentations import VideoAugmentation, VideoValidationTransform
from src.training.evaluate import validate_epoch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_training_model(config: TrainingConfig, device: str):
    """
    Build SAM2Train model with LoRA adapters.

    Returns:
        model: SAM2Train model with LoRA applied
        loss_fn: MultiStepMultiMasksAndIous loss function
    """
    from training.model.sam2 import SAM2Train
    from training.loss_fns import MultiStepMultiMasksAndIous

    # Download checkpoint
    config_name, ckpt_path = _hf_download(config.sam2_model_id)

    # Build the model via hydra config in training mode
    # We need to override the target class to SAM2Train
    model = build_sam2(
        config_file=config_name,
        ckpt_path=ckpt_path,
        device="cpu",  # Load on CPU first
        mode="train",
        hydra_overrides_extra=[
            "++model._target_=training.model.sam2.SAM2Train",
            f"++model.freeze_image_encoder={config.freeze_image_encoder}",
            f"++model.num_init_cond_frames_for_train={config.num_init_cond_frames}",
            f"++model.prob_to_use_pt_input_for_train={config.prob_to_use_pt_input}",
        ],
        apply_postprocessing=False,
    )

    # Freeze image encoder if configured
    if config.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False

    # Apply LoRA to target modules
    print("\nApplying LoRA adapters...")
    wrapped = apply_lora(
        model,
        target_patterns=SAM2_LORA_TARGETS,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    print(f"  Wrapped {len(wrapped)} Linear layers with LoRA (rank={config.lora.rank})")
    for name in sorted(wrapped):
        print(f"    - {name}")

    # Unfreeze small learnable embeddings
    unfrozen_embeddings = []
    for name, param in model.named_parameters():
        if any(
            key in name
            for key in [
                "maskmem_tpos_enc",
                "no_mem_embed",
                "no_mem_pos_enc",
                "no_obj_ptr",
                "obj_ptr_proj",
                "obj_ptr_tpos_proj",
            ]
        ):
            param.requires_grad = True
            unfrozen_embeddings.append(name)

    if unfrozen_embeddings:
        print(f"\n  Also unfroze {len(unfrozen_embeddings)} embedding parameters:")
        for name in unfrozen_embeddings:
            print(f"    - {name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Build loss function
    loss_fn = MultiStepMultiMasksAndIous(
        weight_dict=config.loss_weight_dict,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        pred_obj_scores=True,
    )

    model = model.to(device)
    return model, loss_fn


def build_optimizer(model: nn.Module, config: TrainingConfig):
    """Build AdamW optimizer with separate LR for LoRA and other params."""
    param_groups = get_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": param_groups["lora"],
                "lr": config.optimizer.lora_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": param_groups["other"],
                "lr": config.optimizer.base_lr,
                "weight_decay": 0.0,  # No weight decay for embeddings
            },
        ],
    )
    return optimizer


def build_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """Build LR scheduler with warmup + cosine annealing."""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.optimizer.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if config.optimizer.use_cosine_schedule:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_normalization_stats(config: TrainingConfig):
    """
    Resolve normalization stats for training.

    If config.augmentation.normalize_mean/std are already set, use them.
    Otherwise, try to load dataset-specific stats from manifest.json.
    Falls back to ImageNet stats if manifest is not found.
    """
    mean = config.augmentation.normalize_mean
    std = config.augmentation.normalize_std

    if mean is not None and std is not None:
        return mean, std

    # Try loading from manifest
    manifest_path = Path(config.data.training_data_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        norm = manifest.get("normalization", {})
        ds_mean = norm.get("dataset_mean")
        ds_std = norm.get("dataset_std")
        if ds_mean and ds_std:
            print(f"  Using dataset normalization stats from {manifest_path}")
            print(f"    mean={ds_mean}, std={ds_std}")
            return ds_mean, ds_std

    # Fall back to ImageNet
    fallback_mean = [0.485, 0.456, 0.406]
    fallback_std = [0.229, 0.224, 0.225]
    print(f"  No dataset stats found, falling back to ImageNet normalization")
    print(f"    mean={fallback_mean}, std={fallback_std}")
    return fallback_mean, fallback_std


# ---------------------------------------------------------------------------
# Feature precomputation
# ---------------------------------------------------------------------------

def precompute_all_features(
    model: nn.Module,
    datasets: List[NeuronVideoDataset],
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
    norm_mean: List[float],
    norm_std: List[float],
    image_size: int = 1024,
) -> Tuple[Dict, List[torch.Tensor]]:
    """
    Precompute frozen image encoder features for every unique frame.

    Since the image encoder is frozen (no LoRA on Conv2d layers), its output
    is a deterministic function of each input image. We compute it once and
    cache in CPU memory, avoiding the expensive encoder forward pass during
    training (~60-70% of per-step compute).

    Args:
        model: SAM2Train model with frozen image encoder.
        datasets: List of datasets (train + val) whose videos to precompute.
        device: Device for computation.
        amp_dtype: AMP dtype (bfloat16 on CUDA).
        use_amp: Whether AMP is enabled.
        norm_mean, norm_std: Normalization stats for images.
        image_size: Target image size (1024 for SAM 2).

    Returns:
        feature_cache: Dict mapping (video_name, frame_idx) -> list of 3 FPN
                       feature tensors in float16 on CPU.
        pos_enc: List of 3 positional encoding tensors (same for all frames,
                 depends only on spatial resolution).
    """
    # Collect all unique (video_info) across datasets
    all_videos = {}
    for ds in datasets:
        for vid in ds.videos:
            if vid["name"] not in all_videos:
                all_videos[vid["name"]] = vid

    total_frames = sum(v["num_frames"] for v in all_videos.values())
    print(f"\n=== Precomputing Image Encoder Features ===")
    print(f"  {len(all_videos)} videos, {total_frames} frames total")

    feature_cache = {}
    pos_enc = None

    model.eval()
    precomp_start = time.time()

    with torch.no_grad():
        done = 0
        for vid_name, vid in all_videos.items():
            for idx in range(vid["num_frames"]):
                # Load and preprocess frame (same as validation transform:
                # resize to 1024x1024 + normalize, NO spatial augmentation)
                frame = NeuronVideoDataset._load_frame_from_disk(vid["img_dir"], idx)
                frame = TF.resize(
                    frame,
                    [image_size, image_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                frame = TF.normalize(frame, norm_mean, norm_std)
                frame_batch = frame.unsqueeze(0).to(device)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    backbone_out = model.forward_image(frame_batch)

                # Store backbone_fpn features as float16 on CPU (saves ~50% memory)
                feature_cache[(vid_name, idx)] = [
                    feat.squeeze(0).cpu().half()
                    for feat in backbone_out["backbone_fpn"]
                ]

                # PE depends only on spatial resolution — same for all frames
                if pos_enc is None:
                    pos_enc = [pe.cpu() for pe in backbone_out["vision_pos_enc"]]

                done += 1
                if done % 100 == 0 or done == total_frames:
                    print(f"  Processed {done}/{total_frames} frames...")

    precomp_time = time.time() - precomp_start
    cache_mb = sum(
        sum(f.nelement() * f.element_size() for f in feats)
        for feats in feature_cache.values()
    ) / 1e6
    print(f"  Feature precomputation done in {precomp_time:.1f}s ({cache_mb:.0f} MB cached)")

    model.train()
    return feature_cache, pos_enc


def assemble_cached_backbone(
    feature_cache: Dict,
    pos_enc: List[torch.Tensor],
    frame_id_lists: List[List[Tuple[str, int]]],
    device: str,
    do_hflip: bool = False,
    do_vflip: bool = False,
) -> Dict:
    """
    Assemble a backbone_out dict from cached per-frame features.

    Builds tensors matching the format that forward_image() would return:
    backbone_fpn[level] has shape [B*T, C, H, W] where items are ordered as
    [vid0_frame0, vid0_frame1, ..., vid0_frameT-1, vid1_frame0, ...].

    Args:
        feature_cache: Precomputed features dict.
        pos_enc: Shared positional encodings (list of 3 tensors).
        frame_id_lists: Per-video frame IDs from the collate function.
                        frame_id_lists[video_idx] = [(vid_name, frame_idx), ...].
        device: Target device.
        do_hflip: Apply horizontal flip to features (feature-level augmentation).
        do_vflip: Apply vertical flip to features (feature-level augmentation).

    Returns:
        backbone_out dict with "backbone_fpn" and "vision_pos_enc".
    """
    num_levels = len(pos_enc)
    # Collect features in flat_img_batch order: for each video, all T frames
    level_feats = [[] for _ in range(num_levels)]

    for video_frame_ids in frame_id_lists:
        for vid_name, frame_idx in video_frame_ids:
            cached = feature_cache[(vid_name, frame_idx)]
            for level in range(num_levels):
                feat = cached[level].float()  # float16 -> float32
                # Apply feature-level flip augmentation
                if do_hflip:
                    feat = feat.flip(-1)
                if do_vflip:
                    feat = feat.flip(-2)
                level_feats[level].append(feat)

    backbone_fpn = [
        torch.stack(level_feats[level], dim=0).to(device)
        for level in range(num_levels)
    ]

    # PE is spatial-only — expand to match batch size and move to device
    batch_t = backbone_fpn[0].shape[0]  # B*T
    vision_pos_enc = [
        pe.expand(batch_t, -1, -1, -1).to(device)
        for pe in pos_enc
    ]

    return {
        "backbone_fpn": backbone_fpn,
        "vision_pos_enc": vision_pos_enc,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: TrainingConfig = None):
    """Main training function."""
    if config is None:
        config = TrainingConfig()

    device = detect_device()
    print(f"Using device: {device}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # --- Data ---
    print("\n=== Data Setup ===")
    all_videos = get_video_names(config.data.training_data_dir)
    if not all_videos:
        raise RuntimeError(
            f"No training videos found in {config.data.training_data_dir}/images/. "
            f"Run prepare_data.py first."
        )

    train_videos, val_videos = train_val_split(
        all_videos, val_fraction=config.data.val_fraction, seed=config.seed
    )
    print(f"Training videos ({len(train_videos)}): {train_videos}")
    print(f"Validation videos ({len(val_videos)}): {val_videos}")

    # Resolve normalization stats (dataset-specific or ImageNet fallback)
    norm_mean, norm_std = _load_normalization_stats(config)

    # Augmentations
    # When using precomputed features, spatial augmentation (rotation/shear)
    # cannot be applied at the feature level. We use flip-only augmentation
    # and apply it during backbone assembly instead.
    use_precomputed = config.data.precompute_features and config.freeze_image_encoder
    if use_precomputed:
        # With precomputed features, images are already resized + normalized.
        # Spatial augmentation is handled at the feature level (flips only).
        # Use the validation transform (resize + normalize only) for image
        # preprocessing, since augmentation happens on features instead.
        train_transform = VideoValidationTransform(
            image_size=config.data.image_size,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )
        print("  Using precomputed features: spatial aug will be feature-level flips")
    else:
        train_transform = VideoAugmentation(
            image_size=config.data.image_size,
            horizontal_flip_p=config.augmentation.horizontal_flip_p,
            vertical_flip_p=config.augmentation.vertical_flip_p,
            rotation_degrees=config.augmentation.rotation_degrees,
            shear=config.augmentation.shear,
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
            normalize_mean=norm_mean,
            normalize_std=norm_std,
        )

    val_transform = VideoValidationTransform(
        image_size=config.data.image_size,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    # Datasets
    train_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=train_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=train_transform,
        label_id=config.data.label_id,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )
    val_dataset = NeuronVideoDataset(
        data_dir=config.data.training_data_dir,
        video_names=val_videos,
        num_frames=config.data.num_frames,
        max_objects=config.data.max_objects,
        transform=val_transform,
        label_id=config.data.label_id,
        clip_stride=config.data.clip_stride,
        cache_in_memory=config.data.cache_in_memory,
    )

    # When data is cached in memory, use num_workers=0 to avoid
    # pickling/forking overhead of sending cached tensors to workers
    effective_num_workers = 0 if config.data.cache_in_memory else config.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=(effective_num_workers > 0),
        drop_last=True,
        persistent_workers=(effective_num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        collate_fn=neuron_collate_fn,
        pin_memory=(effective_num_workers > 0),
    )

    # --- Model ---
    print("\n=== Model Setup ===")
    model, loss_fn = build_training_model(config, device)

    # --- AMP ---
    use_amp = config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # --- Feature precomputation ---
    feature_cache = None
    pos_enc = None
    if use_precomputed:
        feature_cache, pos_enc = precompute_all_features(
            model=model,
            datasets=[train_dataset, val_dataset],
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            norm_mean=norm_mean,
            norm_std=norm_std,
            image_size=config.data.image_size,
        )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # --- Training loop ---
    print(f"\n=== Training ({config.num_epochs} epochs, {len(train_loader)} steps/epoch) ===")
    if feature_cache is not None:
        print(f"  Using precomputed features (encoder skipped during training)")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    training_log = []

    # Flip augmentation probabilities (used with precomputed features)
    hflip_p = config.augmentation.horizontal_flip_p
    vflip_p = config.augmentation.vertical_flip_p

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for step, (batch, frame_id_lists) in enumerate(train_loader):
            batch = batch.to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                if feature_cache is not None:
                    # --- Precomputed features path (fast) ---
                    # Apply feature-level flip augmentation
                    do_hflip = random.random() < hflip_p
                    do_vflip = random.random() < vflip_p

                    backbone_out = assemble_cached_backbone(
                        feature_cache, pos_enc, frame_id_lists,
                        device, do_hflip, do_vflip,
                    )

                    # Apply matching flip to masks so they stay aligned
                    if do_hflip:
                        batch.masks = batch.masks.flip(-1)
                    if do_vflip:
                        batch.masks = batch.masks.flip(-2)

                    # Run tracking (skipping image encoder entirely)
                    backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
                    outputs = model.forward_tracking(backbone_out, batch)
                else:
                    # --- Original path (with full encoder) ---
                    outputs = model(batch)

                # Compute loss
                targets = batch.masks.float()
                losses = loss_fn(outputs, targets)
                loss = losses["core_loss"]

            # Backward
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.max_grad_norm
                )
                optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.item())

            # Logging
            if (step + 1) % config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_losses[-config.log_every_n_steps :])
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                sec_per_step = elapsed / (step + 1)
                print(
                    f"  Epoch {epoch+1}/{config.num_epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, "
                    f"{sec_per_step:.1f}s/step"
                )

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "time_seconds": epoch_time,
        }

        print(
            f"Epoch {epoch+1}/{config.num_epochs}: "
            f"train_loss={avg_epoch_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # --- Validation ---
        if (epoch + 1) % config.val_every_n_epochs == 0 and len(val_dataset) > 0:
            print("  Validating...")
            val_metrics = validate_epoch(
                model, val_loader, loss_fn, device,
                feature_cache=feature_cache, pos_enc=pos_enc,
            )
            log_entry.update(val_metrics)
            print(
                f"  val_loss={val_metrics['val_loss']:.4f}, "
                f"val_iou={val_metrics['val_iou']:.4f}, "
                f"val_dice={val_metrics['val_dice']:.4f}"
            )

            # Save best checkpoint
            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                save_lora_weights(model, checkpoint_dir / "lora_weights.pt")
                print(f"  New best IoU: {best_val_iou:.4f} - saved checkpoint")

        training_log.append(log_entry)

        # Save training log
        with open(checkpoint_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=str)

    # Save final checkpoint if not saving best only
    if not config.save_best_only:
        save_lora_weights(model, checkpoint_dir / "lora_weights_final.pt")

    # Save config
    import dataclasses

    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"\nTraining complete! Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'lora_weights.pt'}")

    return best_val_iou
