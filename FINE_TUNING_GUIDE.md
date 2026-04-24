# Fine-Tuning Guide: SAM2 LoRA for Neuron Segmentation

This guide explains how to fine-tune SAM2 on your own neuron microscopy data using LoRA (Low-Rank Adaptation) for efficient training.

## Overview

Fine-tuning adapts SAM2 to your specific microscopy domain, improving segmentation accuracy for neurons. We use:

- **LoRA**: Efficient fine-tuning that only trains small adapter layers (~0.5% of parameters)
- **Feature Precomputation**: 10x faster training by caching frozen image encoder output
- **In-Memory Caching**: All frames cached in RAM, eliminating disk I/O
- **Augmentations**: Flip, rotation, shear, brightness/contrast (feature-level flips when precomputed)

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

## Step 1: Prepare Your Data

### Create a Manifest File

Create `data_manifest.json` listing your TIF stacks:

```json
[
    {
        "image_tif": "/path/to/neuron_video_1.tif",
        "mask_tif": "/path/to/labeled_masks_1.tif",
        "name": "video_1"
    },
    {
        "image_tif": "/path/to/neuron_video_2.tif",
        "mask_tif": "/path/to/labeled_masks_2.tif",
        "name": "video_2"
    }
]
```

**Requirements:**
- `image_tif`: Grayscale microscopy TIF stack (T, H, W)
- `mask_tif`: Integer-labeled mask TIF stack (T, H, W) where each neuron has a unique ID
- At least 5-10 videos recommended for good generalization

### Convert to Training Format

```bash
python train.py prepare --manifest data_manifest.json --output-dir training_data
```

This converts TIF stacks to PNG images and masks, computing dataset statistics for normalization.

## Step 2: Train the Model

### Basic Training

```bash
python train.py train
```

### Custom Training

```bash
python train.py train \
    --epochs 30 \
    --batch-size 1 \
    --lora-rank 8 \
    --lora-lr 1e-4 \
    --clip-stride 2 \
    --val-every 5
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 60 | Maximum training epochs |
| `--batch-size` | 1 | Batch size (1 recommended for most GPUs) |
| `--lora-rank` | 8 | LoRA rank (8-16 typical) |
| `--lora-alpha` | 16.0 | LoRA alpha (usually 2x rank) |
| `--lora-lr` | 1e-4 | LoRA learning rate |
| `--base-lr` | 5e-6 | Learning rate for non-LoRA params |
| `--num-frames` | 8 | Frames per training clip |
| `--val-every` | 5 | Validation frequency (epochs) |
| `--use-imagenet-stats` | — | Use ImageNet normalization instead of dataset stats |

**Performance parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clip-stride` | 4 | Stride between clip start frames (reduces overlap, 1=all clips) |
| `--no-precompute` | — | Disable feature precomputation (slower but keeps full augmentation) |
| `--no-cache` | — | Disable in-memory frame caching (saves RAM but slower) |

**Note:** The learning rate scheduler uses linear warmup (3 epochs) followed by cosine annealing.

### Hardware Requirements

With feature precomputation enabled (default):

| Hardware | Batch Size | Training Time (5 epochs) | Training Time (60 epochs) |
|----------|------------|--------------------------|---------------------------|
| NVIDIA GPU (8GB+) | 1 | ~55 min | ~11 hours |
| Apple M1/M2 (MPS) | 1 | ~2-3 hours | ~24 hours |
| CPU | 1 | Not recommended | Not recommended |

Without feature precomputation (`--no-precompute`), training is ~10x slower.

## Step 3: Monitor Training

Training outputs progress to console:

```
=== Precomputing Image Encoder Features ===
  12 videos, 876 frames total
  Processed 876/876 frames...
  Feature precomputation done in 240.3s (6912 MB cached)

=== Training (60 epochs, 82 steps/epoch) ===
  Using precomputed features (encoder skipped during training)
  Epoch 1/60, Step 10/82, Loss: 1.2340, LR: 3.33e-05, 7.8s/step
Epoch 1/60: train_loss=1.1200, time=640.2s
  Validating...
  val_loss=0.987, val_iou=0.543, val_dice=0.654
  New best IoU: 0.543 - saved checkpoint
```

### Key Metrics to Watch

| Metric | Good Range | What It Means |
|--------|------------|---------------|
| `val_iou` | 0.65-0.85 | Intersection over Union (higher = better) |
| `val_dice` | 0.75-0.90 | Dice coefficient (higher = better) |
| `sec/step` | 5-10 (precomputed) | Training speed per step |
| Train/Val gap | < 0.1 | Low gap = not overfitting |

### Signs of Problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| Val loss increasing | Overfitting | Reduce `--lora-rank`, add more data |
| Train loss stuck | LR too low | Increase `--lora-lr` |
| NaN losses | LR too high | Decrease `--lora-lr` |
| Loss plateau | Not enough epochs | Increase `--epochs` |

## Step 4: Use Fine-Tuned Model

After training, weights are saved to `checkpoints/lora_weights.pt`.

### In the GUI

1. Launch: `python main.py`
2. Load your TIF video
3. Paint mask on frame 0
4. Check **"Use fine-tuned model"** checkbox (enabled by default)
5. Click "SAM2: Propagate from frame 0"

### Programmatically

```python
from src.training.lora import apply_lora, load_lora_weights, SAM2_LORA_TARGETS
from sam2.build_sam import build_sam2_video_predictor

# Load base model
predictor = build_sam2_video_predictor("facebook/sam2.1-hiera-tiny")

# Apply LoRA and load weights
apply_lora(predictor, target_patterns=SAM2_LORA_TARGETS)
load_lora_weights(predictor, "checkpoints/lora_weights.pt")
predictor.eval()

# Use normally
# ...
```

## Loss Functions Explained

Training uses SAM2's built-in `MultiStepMultiMasksAndIous` loss (from Meta's training package), with weights matching the MOSE fine-tuning configuration:

### Focal Loss (weight: 20.0)

Handles class imbalance between background (many pixels) and neurons (few pixels).

```
Focal(p) = -α(1-p)^γ log(p)
```

- `α=0.25`: Down-weights easy background pixels
- `γ=2.0`: Focuses on hard examples

### Dice Loss (weight: 1.0)

Optimizes for overall shape overlap, similar to IoU.

```
Dice = 2|A ∩ B| / (|A| + |B|)
```

### IoU Loss (weight: 1.0)

Trains the model to predict its own accuracy (IoU prediction head).

### Object Score Loss (weight: 1.0)

Trains the model to predict whether an object is present.

## Data Augmentations

Augmentations prevent overfitting on small datasets. All spatial transforms are applied consistently across all frames in a clip.

### With Precomputed Features (default)

When feature precomputation is enabled, only flip augmentations are applied at the feature level:

| Augmentation | Parameters | How |
|--------------|------------|-----|
| Horizontal flip | 50% prob | `feat.flip(-1)` on cached feature maps |
| Vertical flip | 50% prob | `feat.flip(-2)` on cached feature maps |

Masks are flipped to match: `batch.masks = batch.masks.flip(-1)`.

### Without Precomputed Features (`--no-precompute`)

Full augmentation pipeline with additional spatial and intensity transforms:

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Horizontal flip | 50% prob | Basic invariance |
| Vertical flip | 50% prob | Basic invariance |
| Rotation | ±25° | Orientation invariance |
| Shear | ±20° | Perspective invariance |
| Brightness | ±15% | Lighting invariance |
| Contrast | ±15% | Lighting invariance |

**Trade-off**: Full augmentations provide more diversity but training is ~10x slower since the image encoder must run every step.

**Important**: All spatial augmentations are applied consistently across all frames in a clip.

## Training Architecture

### Feature Precomputation

The biggest performance optimization. SAM2's image encoder (Hiera backbone + FPN + conv_s0/conv_s1) is frozen during training (no LoRA on Conv2d layers), so its output is deterministic for each input image.

**How it works:**
1. Before training, run the encoder on every frame once (~3-5 min for 876 frames)
2. Cache the 3-level FPN features in float16 on CPU (~8 MB/frame, ~7 GB total)
3. Store positional encodings once (they're purely spatial, independent of content)
4. During training, assemble cached features into `backbone_out` and skip the encoder entirely

**Impact:** ~80 sec/step → ~8 sec/step (10x speedup). 5 epochs in ~55 min instead of ~36 hours.

### Mask Prompt Conditioning

The training wrapper (`SAM2TrainingWrapper`) enables temporal learning by conditioning each frame on the previous mask:

1. **Frame 0**: Uses ground truth mask as prompt
2. **Frame 1+**: Uses previous predicted mask as prompt

```
Frame 0: GT Mask → Encode → Mask Decoder → Pred 0
Frame 1: Pred 0  → Encode → Mask Decoder → Pred 1
Frame 2: Pred 1  → Encode → Mask Decoder → Pred 2
...
```

This teaches the model to **propagate masks** rather than segment from scratch. The mask encoding uses SAM2's built-in `mask_downscaling` network:

```python
# Resize mask to SAM2's expected 256x256 resolution
mask_for_encoder = F.interpolate(mask_input, size=(256, 256))

# Encode through convolutional downscaling
dense_prompt = prompt_encoder.mask_downscaling(mask_for_encoder)

# Pass to mask decoder (conditioned on previous mask)
output = mask_decoder(image_features, dense_prompt_embeddings=dense_prompt)
```

## LoRA Architecture

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into the model:

```
W' = W + BA
```

Where:
- `W`: Original frozen weights
- `B`: Trainable matrix (d × r)
- `A`: Trainable matrix (r × d)
- `r`: Rank (default 8)

We apply LoRA to attention layers in SAM2's:
- Memory attention
- Mask decoder

This trains only ~0.5% of parameters while achieving similar performance to full fine-tuning.

### LoRA Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `rank` | 8 | Capacity (higher = more expressive) |
| `alpha` | 16.0 | Scaling factor (usually 2x rank) |
| `dropout` | 0.05 | Regularization |

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size (already default=1)
python train.py train --batch-size 1

# Reduce frames per clip
python train.py train --num-frames 4

# Disable in-memory frame caching (saves ~2 GB RAM)
python train.py train --no-cache
```

### Training Too Slow

Feature precomputation is enabled by default and gives a **10x speedup**. If training is still slow:

```bash
# 1. Ensure precomputation is enabled (default)
python train.py train
# Look for: "Using precomputed features (encoder skipped during training)"

# 2. Increase clip stride to reduce steps per epoch
python train.py train --clip-stride 8

# 3. Reduce frames per clip
python train.py train --num-frames 4
```

| Optimization | Effect |
|-------------|--------|
| Feature precomputation (default) | 10x faster per step |
| `--clip-stride 4` (default) | 4x fewer steps/epoch |
| `--clip-stride 8` | 8x fewer steps/epoch |
| `--num-frames 4` | ~2x faster per step |

### Poor Generalization

1. Add more training data (5-10 videos minimum)
2. Reduce LoRA rank: `--lora-rank 4`
3. Use full augmentation: `--no-precompute` (slower but more augmentation diversity)
4. Check for data leakage (same video in train and val)

### Broken Neuron Branches

If predictions break thin branches:
1. Add more training data with well-labeled branch connections
2. Increase training epochs
3. Use `--no-precompute` for full spatial augmentations

## Tips for Best Results

1. **Quality over quantity**: Well-labeled masks are more important than many masks
2. **Diverse data**: Include different neuron morphologies, lighting conditions
3. **Consistent labeling**: Use same brush size, label all visible neurons
4. **Validate manually**: Spot-check predictions on held-out videos
5. **Iterate**: Fine-tune on your own predictions after manual correction

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
