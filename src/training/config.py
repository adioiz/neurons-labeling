"""Training configuration for SAM 2 LoRA fine-tuning on neuron data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05


@dataclass
class DataConfig:
    # Directory containing converted training data (PNG images + PNG masks)
    training_data_dir: str = "training_data"
    # Number of frames per training clip
    num_frames: int = 8
    # Image size for SAM 2 (must match pretrained config)
    image_size: int = 1024
    # Maximum objects per clip
    max_objects: int = 3
    # Validation split (fraction of videos held out)
    val_fraction: float = 0.2
    # Label ID used for neurons in mask TIFs (0 = all non-zero)
    label_id: int = 0
    # Clip sampling stride (1 = every possible clip, 4 = every 4th, reduces overlap)
    clip_stride: int = 4
    # Cache all frames/masks in memory at init (eliminates disk I/O during training)
    cache_in_memory: bool = True
    # Precompute frozen image encoder features before training (biggest speedup)
    precompute_features: bool = True


@dataclass
class AugmentationConfig:
    # Spatial augmentations (applied consistently across frames)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_degrees: float = 25.0
    shear: float = 20.0
    # Intensity augmentations
    brightness: float = 0.15
    contrast: float = 0.15
    # Normalization stats — set from manifest.json (dataset-specific) by default.
    # Falls back to ImageNet stats if not overridden.
    # Use --use-dataset-stats to auto-load from manifest.json (recommended).
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None


@dataclass
class OptimizerConfig:
    # Learning rates
    base_lr: float = 5e-6  # For non-LoRA trainable params
    lora_lr: float = 1e-4  # For LoRA adapter params
    weight_decay: float = 0.1
    # Gradient clipping
    max_grad_norm: float = 0.1
    # Scheduler
    warmup_epochs: int = 3
    # Use cosine annealing after warmup
    use_cosine_schedule: bool = True


@dataclass
class LossConfig:
    # Loss weights matching Meta's MOSE fine-tuning config
    loss_mask_weight: float = 20.0  # Focal loss
    loss_dice_weight: float = 1.0  # Dice loss
    loss_iou_weight: float = 1.0  # IoU prediction loss
    loss_class_weight: float = 1.0  # Object score loss
    # Focal loss params
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    # Model
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    freeze_image_encoder: bool = True

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Augmentations
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Loss
    loss: LossConfig = field(default_factory=LossConfig)

    # Training
    batch_size: int = 1
    num_epochs: int = 60
    num_workers: int = 2
    seed: int = 42
    # Use mixed precision (bfloat16 on CUDA, float32 on MPS/CPU)
    use_amp: bool = True

    # Validation
    val_every_n_epochs: int = 5

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # SAM2Train settings
    prob_to_use_pt_input: float = 0.0  # Use mask input (not point input)
    num_init_cond_frames: int = 1  # Condition on 1 frame

    # Logging
    log_every_n_steps: int = 10

    @property
    def loss_weight_dict(self) -> Dict[str, float]:
        return {
            "loss_mask": self.loss.loss_mask_weight,
            "loss_dice": self.loss.loss_dice_weight,
            "loss_iou": self.loss.loss_iou_weight,
            "loss_class": self.loss.loss_class_weight,
        }
