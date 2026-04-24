"""
PyTorch Dataset for neuron time-lapse video segmentation training.

Loads pre-converted PNG frames and PNG masks, samples random subsequences,
and returns data in the format expected by SAM2Train.forward().
"""

import json
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.utils.data_utils import (
    BatchedVideoDatapoint,
    BatchedVideoMetaData,
    Frame,
    Object,
    VideoDatapoint,
    collate_fn,
)


class NeuronVideoDataset(Dataset):
    """
    Dataset that yields video clips for SAM 2 training.

    Each item is a random subsequence of `num_frames` consecutive frames
    from a labeled neuron time-lapse video.
    """

    def __init__(
        self,
        data_dir: str,
        video_names: List[str],
        num_frames: int = 8,
        max_objects: int = 3,
        transform: Optional[Callable] = None,
        label_id: int = 0,
        clip_stride: int = 1,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing images/ and masks/ subdirs.
            video_names: List of video names (subdirectory names) to include.
            num_frames: Number of frames per training clip.
            max_objects: Maximum number of objects per clip.
            transform: Augmentation transform (receives lists of tensors).
            label_id: If 0, use all non-zero labels. If > 0, use only that label.
            clip_stride: Stride between clip start frames (1 = all clips, 4 = every 4th).
            cache_in_memory: If True, load all frames/masks into RAM at init.
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.max_objects = max_objects
        self.transform = transform
        self.label_id = label_id
        self.clip_stride = clip_stride

        # Build index of all videos and their frame counts
        self.videos = []
        for name in video_names:
            img_dir = self.data_dir / "images" / name
            mask_dir = self.data_dir / "masks" / name
            if not img_dir.exists() or not mask_dir.exists():
                print(f"Warning: Skipping {name} (directory not found)")
                continue

            frame_files = sorted(img_dir.glob("*.png"))
            n_frames = len(frame_files)
            if n_frames < num_frames:
                print(
                    f"Warning: {name} has {n_frames} frames, need {num_frames}. Skipping."
                )
                continue

            self.videos.append(
                {
                    "name": name,
                    "num_frames": n_frames,
                    "img_dir": img_dir,
                    "mask_dir": mask_dir,
                }
            )

        if not self.videos:
            raise ValueError(f"No valid videos found in {data_dir}")

        # Compute total number of possible clips (with stride to reduce overlap)
        self.clips = []
        for vid_idx, vid in enumerate(self.videos):
            n_clips = vid["num_frames"] - num_frames + 1
            for start in range(0, n_clips, clip_stride):
                self.clips.append((vid_idx, start))

        # Optionally cache all frames and masks in RAM
        self._frame_cache = {}
        self._mask_cache = {}
        if cache_in_memory:
            print("  Caching all frames and masks in memory...")
            for vid in self.videos:
                for idx in range(vid["num_frames"]):
                    key = (vid["name"], idx)
                    self._frame_cache[key] = self._load_frame_from_disk(
                        vid["img_dir"], idx
                    )
                    self._mask_cache[key] = self._load_mask_from_disk(
                        vid["mask_dir"], idx
                    )
            total_mb = sum(f.nelement() * f.element_size() for f in self._frame_cache.values()) / 1e6
            print(f"  Cached {len(self._frame_cache)} frames ({total_mb:.0f} MB)")

        print(
            f"NeuronVideoDataset: {len(self.videos)} videos, "
            f"{len(self.clips)} clips of {num_frames} frames "
            f"(stride={clip_stride})"
        )

    def __len__(self) -> int:
        return len(self.clips)

    @staticmethod
    def _load_frame_from_disk(img_dir: Path, idx: int) -> torch.Tensor:
        """Load a PNG frame as [C, H, W] float tensor in [0, 1]."""
        path = img_dir / f"{idx:05d}.png"
        img = Image.open(path).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _load_mask_from_disk(mask_dir: Path, idx: int) -> torch.Tensor:
        """Load a PNG mask as [H, W] integer tensor."""
        path = mask_dir / f"{idx:05d}.png"
        mask = Image.open(path)
        return torch.from_numpy(np.array(mask)).long()

    def _load_frame(self, vid_name: str, img_dir: Path, idx: int) -> torch.Tensor:
        """Load frame from cache or disk."""
        key = (vid_name, idx)
        if key in self._frame_cache:
            return self._frame_cache[key]
        return self._load_frame_from_disk(img_dir, idx)

    def _load_mask(self, vid_name: str, mask_dir: Path, idx: int) -> torch.Tensor:
        """Load mask from cache or disk."""
        key = (vid_name, idx)
        if key in self._mask_cache:
            return self._mask_cache[key]
        return self._load_mask_from_disk(mask_dir, idx)

    def __getitem__(self, index: int) -> VideoDatapoint:
        vid_idx, start_frame = self.clips[index]
        vid = self.videos[vid_idx]

        # Load frames and masks for this clip
        frames_list = []
        masks_list = []
        frame_ids = []  # Track (video_name, frame_idx) for feature cache lookup
        for t in range(self.num_frames):
            frame_idx = start_frame + t
            frame = self._load_frame(vid["name"], vid["img_dir"], frame_idx)
            mask = self._load_mask(vid["name"], vid["mask_dir"], frame_idx)
            frames_list.append(frame)
            masks_list.append(mask)
            frame_ids.append((vid["name"], frame_idx))

        # Apply augmentations
        if self.transform is not None:
            frames_list, masks_list = self.transform(frames_list, masks_list)

        # Determine object IDs present in this clip
        all_labels = set()
        for mask in masks_list:
            labels = torch.unique(mask)
            for l in labels:
                l_val = l.item()
                if l_val > 0:  # Skip background
                    if self.label_id == 0 or l_val == self.label_id:
                        all_labels.add(l_val)

        # Limit to max_objects
        obj_ids = sorted(all_labels)[: self.max_objects]

        # If no objects found in this clip, create a dummy object with empty masks.
        # This teaches the model that not every frame contains a neuron, preventing
        # hallucination on background-only regions (pred_obj_scores handles this).
        if not obj_ids:
            obj_ids = [1]

        # Build per-frame objects
        video_frames = []
        for t in range(self.num_frames):
            frame_tensor = frames_list[t]  # [C, H, W]
            mask_tensor = masks_list[t]  # [H, W]

            objects = []
            for obj_id in obj_ids:
                if self.label_id > 0:
                    binary_mask = (mask_tensor == self.label_id)
                else:
                    binary_mask = (mask_tensor == obj_id)

                objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=t,
                        segment=binary_mask,  # [H, W] bool
                    )
                )

            video_frames.append(Frame(data=frame_tensor, objects=objects))

        H, W = masks_list[0].shape[-2:]
        return (
            VideoDatapoint(
                frames=video_frames,
                video_id=vid_idx,
                size=(H, W),
            ),
            frame_ids,  # List of (video_name, frame_idx) for feature cache lookup
        )


def neuron_collate_fn(batch):
    """Collate function wrapping Meta's collate_fn.

    Each item in batch is a tuple of (VideoDatapoint, frame_ids).
    Returns (BatchedVideoDatapoint, frame_id_lists) where frame_id_lists
    is a list (one per video in batch) of lists of (video_name, frame_idx) tuples.
    """
    video_datapoints = [item[0] for item in batch]
    frame_id_lists = [item[1] for item in batch]
    batched = collate_fn(video_datapoints, dict_key="neuron_train")
    return batched, frame_id_lists


def get_video_names(data_dir: str) -> List[str]:
    """List all available video names in the data directory."""
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    if not img_dir.exists():
        return []
    return sorted([d.name for d in img_dir.iterdir() if d.is_dir()])


def train_val_split(
    video_names: List[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split video names into train and validation sets."""
    rng = random.Random(seed)
    names = list(video_names)
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_fraction))
    val_names = names[:n_val]
    train_names = names[n_val:]

    return train_names, val_names
