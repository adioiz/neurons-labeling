"""Evaluation utilities for SAM 2 fine-tuning on neuron data."""

from typing import Dict, List

import torch
import numpy as np


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        pred: Predicted mask (any shape, will be flattened).
        target: Ground truth mask (same shape as pred).

    Returns:
        IoU score (float). Returns 0.0 if both masks are empty.
    """
    pred = pred.bool().flatten()
    target = target.bool().flatten()

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Predicted mask.
        target: Ground truth mask.

    Returns:
        Dice score (float).
    """
    pred = pred.bool().flatten().float()
    target = target.bool().flatten().float()

    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection / denominator).item()


def evaluate_batch(
    outputs: List[Dict],
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate a batch of SAM2Train outputs against ground truth.

    Args:
        outputs: List of frame output dicts from SAM2Train.forward().
                Each dict has 'pred_masks_high_res' [B, 1, H, W].
        targets: Ground truth masks [T, B, H, W] (bool).

    Returns:
        Dict with 'mean_iou' and 'mean_dice' averaged over all frames and objects.
    """
    all_ious = []
    all_dices = []

    num_frames = len(outputs)
    for t in range(num_frames):
        if "pred_masks_high_res" not in outputs[t]:
            continue

        pred_masks = outputs[t]["pred_masks_high_res"]  # [B, 1, H, W]
        pred_binary = (pred_masks > 0).squeeze(1)  # [B, H, W]

        gt = targets[t]  # [B, H, W]

        # Resize GT to match pred if needed
        if pred_binary.shape[-2:] != gt.shape[-2:]:
            gt = torch.nn.functional.interpolate(
                gt.float().unsqueeze(1),
                size=pred_binary.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        for b in range(pred_binary.shape[0]):
            iou = compute_iou(pred_binary[b], gt[b])
            dice = compute_dice(pred_binary[b], gt[b])
            all_ious.append(iou)
            all_dices.append(dice)

    if not all_ious:
        return {"mean_iou": 0.0, "mean_dice": 0.0}

    return {
        "mean_iou": np.mean(all_ious),
        "mean_dice": np.mean(all_dices),
    }


@torch.no_grad()
def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device: str,
    feature_cache=None,
    pos_enc=None,
) -> Dict[str, float]:
    """
    Run validation on the entire validation set.

    Args:
        model: SAM2Train model.
        val_loader: Validation DataLoader.
        loss_fn: MultiStepMultiMasksAndIous loss function.
        device: Device string.
        feature_cache: Optional precomputed features dict.
        pos_enc: Optional positional encodings (required if feature_cache is set).

    Returns:
        Dict with 'val_loss', 'val_iou', 'val_dice'.
    """
    model.eval()

    total_loss = 0.0
    all_ious = []
    all_dices = []
    num_batches = 0

    if feature_cache is not None:
        from src.training.train import assemble_cached_backbone

    for batch, frame_id_lists in val_loader:
        batch = batch.to(device)

        if feature_cache is not None:
            # Use precomputed features (no augmentation for validation)
            backbone_out = assemble_cached_backbone(
                feature_cache, pos_enc, frame_id_lists, device,
            )
            backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
            outputs = model.forward_tracking(backbone_out, batch)
        else:
            outputs = model(batch)

        # Compute loss
        targets = batch.masks.float()  # [T, O, H, W]
        losses = loss_fn(outputs, targets)
        total_loss += losses["core_loss"].item()

        # Compute metrics
        metrics = evaluate_batch(outputs, batch.masks)
        all_ious.append(metrics["mean_iou"])
        all_dices.append(metrics["mean_dice"])
        num_batches += 1

    model.train()

    if num_batches == 0:
        return {"val_loss": float("inf"), "val_iou": 0.0, "val_dice": 0.0}

    return {
        "val_loss": total_loss / num_batches,
        "val_iou": np.mean(all_ious),
        "val_dice": np.mean(all_dices),
    }
