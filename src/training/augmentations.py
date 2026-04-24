"""
Microscopy-specific augmentations for neuron time-lapse training data.

All spatial transforms are applied consistently across frames in a clip.
Intensity transforms can optionally vary slightly per frame.
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF


class VideoAugmentation:
    """
    Augmentation pipeline for video clips (list of PIL images + masks).

    All spatial transforms use the same random parameters across all frames.
    Intensity transforms use the same parameters (microscopy images are grayscale
    duplicated to RGB, so per-frame color jitter is not meaningful).
    """

    def __init__(
        self,
        image_size: int = 1024,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_degrees: float = 25.0,
        shear: float = 20.0,
        brightness: float = 0.15,
        contrast: float = 0.15,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
    ):
        self.image_size = image_size
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_degrees = rotation_degrees
        self.shear = shear
        self.brightness = brightness
        self.contrast = contrast
        # Use dataset-specific stats if provided, else fall back to ImageNet
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply augmentations to a clip.

        Args:
            frames: List of [C, H, W] float tensors (0-1 range).
            masks: List of [H, W] int/bool tensors.

        Returns:
            Augmented (frames, masks) lists.
        """
        # Sample random parameters once for the whole clip
        do_hflip = random.random() < self.horizontal_flip_p
        do_vflip = random.random() < self.vertical_flip_p
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)

        aug_frames = []
        aug_masks = []

        for frame, mask in zip(frames, masks):
            # Ensure mask has channel dim for transforms
            mask_3d = mask.unsqueeze(0).float()

            # --- Spatial transforms (consistent) ---
            if do_hflip:
                frame = TF.hflip(frame)
                mask_3d = TF.hflip(mask_3d)

            if do_vflip:
                frame = TF.vflip(frame)
                mask_3d = TF.vflip(mask_3d)

            # Affine (rotation + shear)
            frame = TF.affine(
                frame,
                angle=angle,
                translate=[0, 0],
                scale=1.0,
                shear=[shear_x, shear_y],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_3d = TF.affine(
                mask_3d,
                angle=angle,
                translate=[0, 0],
                scale=1.0,
                shear=[shear_x, shear_y],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            )

            # Resize to target size
            frame = TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_3d = TF.resize(
                mask_3d,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

            # --- Intensity transforms (consistent across frames) ---
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            frame = torch.clamp(frame, 0.0, 1.0)

            # --- Normalize (dataset-specific stats or ImageNet fallback) ---
            frame = TF.normalize(frame, self.normalize_mean, self.normalize_std)

            # Remove channel dim from mask
            mask_out = mask_3d.squeeze(0).long()

            aug_frames.append(frame)
            aug_masks.append(mask_out)

        return aug_frames, aug_masks


class VideoValidationTransform:
    """Validation-only transform: resize + normalize, no augmentation."""

    def __init__(
        self,
        image_size: int = 1024,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
    ):
        self.image_size = image_size
        # Use dataset-specific stats if provided, else fall back to ImageNet
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        aug_frames = []
        aug_masks = []

        for frame, mask in zip(frames, masks):
            mask_3d = mask.unsqueeze(0).float()

            frame = TF.resize(
                frame,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_3d = TF.resize(
                mask_3d,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

            frame = TF.normalize(frame, self.normalize_mean, self.normalize_std)
            mask_out = mask_3d.squeeze(0).long()

            aug_frames.append(frame)
            aug_masks.append(mask_out)

        return aug_frames, aug_masks
