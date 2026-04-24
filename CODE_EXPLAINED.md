# Code Architecture Explained

This document explains the codebase structure and how the components work together.

## Project Structure

```
neurons-labeling/
├── main.py                      # Entry point for GUI
├── train.py                     # Entry point for training CLI
├── src/
│   ├── gui.py                   # Napari GUI with SAM2 integration
│   ├── io.py                    # CSV I/O utilities
│   ├── utils.py                 # Helper functions
│   └── training/                # Fine-tuning module
│       ├── train.py             # Training loop + feature precomputation
│       ├── config.py            # Configuration dataclasses
│       ├── augmentations.py     # Data augmentations
│       ├── neuron_dataset.py    # Video dataset loader with caching
│       ├── evaluate.py          # Validation metrics (IoU, Dice)
│       ├── lora.py              # LoRA implementation
│       └── prepare_data.py      # TIF to PNG converter
├── checkpoints/                 # Saved model weights
└── training_data/               # Converted training data
```

## GUI Module (`src/gui.py`)

### Class: `NeuronsLabelingGUI`

Main application class that manages the napari viewer and SAM2 integration.

#### Key Methods

```python
def launch(self):
    """Start the napari viewer with all widgets."""
```

```python
def _load_sam2_predictor(self, use_finetuned: bool = True, device: str = "cpu"):
    """
    Load or return cached SAM2 video predictor.

    If use_finetuned=True and checkpoints/lora_weights.pt exists,
    loads the base model with LoRA adapters applied.
    """
```

```python
def _run_sam2(self, source_frame_mask, start_frame, label_ids, use_finetuned):
    """
    Run SAM2 video propagation from specified frame.

    1. Exports frames to temp directory as JPGs
    2. Loads SAM2 predictor (with optional LoRA)
    3. Adds mask prompts for each label
    4. Propagates through video
    5. Returns combined mask array
    """
```

### GUI Widgets

The GUI uses `magicgui` decorators for automatic widget generation:

```python
@magicgui(call_button="SAM2: Propagate from frame 0")
def sam2_propagate(label_id: int = 0, from_current: bool = False, use_finetuned: bool = True):
    """Main propagation button with options."""
```

## Training Module (`src/training/`)

### Configuration (`config.py`)

Dataclasses define all training hyperparameters:

```python
@dataclass
class LoRAConfig:
    rank: int = 8           # Low-rank dimension
    alpha: float = 16.0     # Scaling factor
    dropout: float = 0.05   # Regularization

@dataclass
class DataConfig:
    training_data_dir: str = "training_data"
    num_frames: int = 8          # Frames per training clip
    clip_stride: int = 4         # Stride between clip starts (reduces overlap)
    cache_in_memory: bool = True # Cache all frames in RAM
    precompute_features: bool = True  # Precompute frozen encoder output

@dataclass
class OptimizerConfig:
    base_lr: float = 5e-6   # For non-LoRA trainable params
    lora_lr: float = 1e-4   # LoRA adapter learning rate
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    warmup_epochs: int = 3
    use_cosine_schedule: bool = True

@dataclass
class LossConfig:
    loss_mask_weight: float = 20.0  # Focal loss (Meta's config)
    loss_dice_weight: float = 1.0   # Dice loss
    loss_iou_weight: float = 1.0    # IoU prediction loss
    loss_class_weight: float = 1.0  # Object score loss

@dataclass
class TrainingConfig:
    # Combines all sub-configs
    lora: LoRAConfig
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    # ... other settings
```

### Loss Functions

Training uses SAM2's built-in `MultiStepMultiMasksAndIous` loss function from the `training.loss_fns` module in the SAM2 package. This combines multiple losses with configurable weights:

```python
class MultiStepMultiMasksAndIous(nn.Module):
    def forward(self, outputs, targets):
        loss_mask = self.focal(pred_masks, targets)   # Focal loss (class imbalance)
        loss_dice = self.dice(pred_masks, targets)     # Dice loss (shape overlap)

        core_loss = (
            20.0 * loss_mask +     # Heavy emphasis on focal
            1.0 * loss_dice +
            1.0 * loss_iou +       # IoU prediction accuracy
            1.0 * loss_class       # Object presence prediction
        )
        return {"core_loss": core_loss, ...}
```

The weights follow Meta's MOSE fine-tuning configuration.

### Data Augmentations (`augmentations.py`)

#### VideoAugmentation

Applies consistent transforms across all frames in a clip:

```python
class VideoAugmentation:
    def __call__(self, frames, masks):
        # Sample random parameters ONCE
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angle = random.uniform(-25, 25)
        shear_x = random.uniform(-20, 20)
        brightness_factor = 1.0 + random.uniform(-0.15, 0.15)

        for frame, mask in zip(frames, masks):
            # Apply SAME transforms to each frame
            if do_hflip:
                frame = hflip(frame)
                mask = hflip(mask)
            frame = affine(frame, angle=angle, shear=[shear_x, shear_y])
            mask = affine(mask, angle=angle, shear=[shear_x, shear_y])
            frame = adjust_brightness(frame, brightness_factor)
            frame = normalize(frame, mean, std)
```

#### VideoValidationTransform

Resize + normalize only, no augmentation. Used for validation and for training when precomputed features are enabled (since spatial augmentation can't be applied at the feature level).

#### Feature-Level Augmentation

When using precomputed features (default), spatial augmentation is limited to **flips only** — applied directly on cached feature maps during training:

```python
# In assemble_cached_backbone():
if do_hflip:
    feat = feat.flip(-1)   # Flip feature maps horizontally
if do_vflip:
    feat = feat.flip(-2)   # Flip feature maps vertically

# Matching flip applied to masks:
if do_hflip:
    batch.masks = batch.masks.flip(-1)
```

Rotation, shear, and intensity augmentations are not available with precomputed features. Use `--no-precompute` to enable the full augmentation pipeline if needed.

**Why consistency matters:**

- Temporal consistency is crucial for video segmentation
- If frames have different transforms, the model learns spurious correlations
- Same random seed/parameters across all frames in a clip

### LoRA Implementation (`lora.py`)

#### LoRA Layer

Injects trainable low-rank matrices:

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16.0, dropout=0.1):
        self.linear = original_linear  # Frozen
        self.lora_A = nn.Linear(in_features, rank, bias=False)  # Trainable
        self.lora_B = nn.Linear(rank, out_features, bias=False)  # Trainable
        self.scaling = alpha / rank

    def forward(self, x):
        # Original path (frozen)
        result = self.linear(x)
        # LoRA path (trainable)
        result = result + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result
```

#### Target Patterns

LoRA is applied to attention layers. Patterns support both training (`model.xxx`) and inference (`xxx`) prefixes:

```python
SAM2_LORA_TARGETS = [
    # Mask decoder transformer attention (with optional model. prefix)
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_image_to_token\.(q_proj|k_proj|v_proj|out_proj)",
    # Mask decoder final attention
    r"(model\.)?sam_mask_decoder\.transformer\.final_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    # Memory attention layers
    r"(model\.)?memory_attention\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.cross_attn_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.(linear1|linear2)",
]
```

**Key compatibility feature**: The `(model\.)?` prefix makes patterns work for both:
- **Training**: SAM2Train wraps model as `model.xxx`
- **Inference**: Video predictor uses `xxx` directly

#### Weight Management

```python
def save_lora_weights(model, path):
    """Save only LoRA parameters (small file ~2MB)."""
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path)

def load_lora_weights(model, path, device="cpu"):
    """Load LoRA weights into model with LoRA applied."""
    lora_state = torch.load(path, map_location=device)
    model.load_state_dict(lora_state, strict=False)
```

### Training Wrapper (`train.py`)

#### SAM2TrainingWrapper

The `SAM2TrainingWrapper` class wraps SAM2 for video training. A key feature is **mask prompt conditioning**:

```python
class SAM2TrainingWrapper(nn.Module):
    """Wrapper that adapts SAM2Base for training with video clips."""

    def forward(self, batch):
        """
        For each frame:
        1. Frame 0: Use ground truth mask as prompt
        2. Frame 1+: Use previous predicted mask as prompt
        """
        for t in range(T):
            if t == 0:
                # First frame: condition on GT mask
                mask_input = F.interpolate(gt_masks[:, 0], size=(H_feat, W_feat))
            else:
                # Subsequent frames: condition on previous prediction
                mask_input = F.interpolate(prev_pred, size=(H_feat, W_feat))

            pred_mask = self._decode_mask(frame_feat, mask_input)
```

#### Mask Prompt Encoding

The `_decode_mask` method encodes mask prompts through SAM2's `mask_downscaling` network:

```python
def _decode_mask(self, image_features, mask_input):
    """
    Decode mask prediction using image features + mask prompt.

    CRITICAL: The mask_input MUST be encoded through SAM2's mask_downscaling
    network, not set to zeros. This enables temporal conditioning.
    """
    # Resize mask to SAM2's expected resolution (256x256 for 64x64 embeddings)
    mask_for_encoder = F.interpolate(
        mask_input,
        size=(256, 256),  # SAM2 expects 4x the embedding resolution
        mode="bilinear",
        align_corners=False,
    )

    # Encode through the prompt encoder's convolutional downscaling network
    # This converts [B, 1, 256, 256] -> [B, embed_dim, 64, 64]
    dense_prompt = self.model.sam_prompt_encoder.mask_downscaling(mask_for_encoder)

    # Pass to mask decoder
    decoder_output = self.model.sam_mask_decoder(
        image_embeddings=image_features,
        dense_prompt_embeddings=dense_prompt,  # Conditioned on previous mask!
        ...
    )
```

**Why mask conditioning matters:**
- Without it, each frame is processed independently (no temporal learning)
- With it, the model learns to propagate masks through time
- This is the core task for video segmentation

### Training Loop (`train.py`)

#### Feature Precomputation (Key Optimization)

The frozen image encoder (Hiera + FPN + conv_s0/conv_s1) accounts for ~60-70% of per-step compute. Since its output is deterministic (no LoRA on Conv2d layers), we precompute it once before training:

```python
def precompute_all_features(model, datasets, device, ...):
    """Precompute frozen encoder output for all frames. ~3-5 min one-time cost."""
    feature_cache = {}  # (video_name, frame_idx) -> [3 FPN tensors in float16]
    
    with torch.no_grad():
        for vid in all_videos:
            for idx in range(vid["num_frames"]):
                frame = load_and_preprocess(vid, idx)  # resize 1024, normalize
                backbone_out = model.forward_image(frame.unsqueeze(0))
                feature_cache[(vid_name, idx)] = [
                    feat.squeeze(0).cpu().half()  # float16 on CPU
                    for feat in backbone_out["backbone_fpn"]
                ]
    
    # PE is spatial-only (same for all frames) — store once
    pos_enc = [pe.cpu() for pe in backbone_out["vision_pos_enc"]]
    return feature_cache, pos_enc
```

During training, cached features are assembled into the expected backbone_out format:

```python
def assemble_cached_backbone(feature_cache, pos_enc, frame_id_lists, device,
                             do_hflip=False, do_vflip=False):
    """Build backbone_out from cached features in flat_img_batch order."""
    # Order: [vid0_f0, vid0_f1, ..., vid0_fT-1, vid1_f0, ...]
    for video_frame_ids in frame_id_lists:
        for vid_name, frame_idx in video_frame_ids:
            feat = feature_cache[(vid_name, frame_idx)]
            if do_hflip: feat = feat.flip(-1)  # Feature-level augmentation
            level_feats[level].append(feat)
    return {"backbone_fpn": stacked_feats, "vision_pos_enc": expanded_pe}
```

**Result:** ~80 sec/step → ~8 sec/step (10x speedup).

#### Learning Rate Scheduler

Warmup + cosine annealing:

```python
def build_scheduler(optimizer, config, steps_per_epoch):
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.optimizer.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Linear warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine decay

    return LambdaLR(optimizer, lr_lambda)
```

#### Main Training Function

```python
def train(config: TrainingConfig):
    # 1. Setup
    device = detect_device()  # CUDA > MPS > CPU
    model = build_training_model(config)  # SAM2Train + LoRA
    
    # 2. Data (with in-memory caching and clip stride)
    train_loader, val_loader = create_dataloaders(config)
    
    # 3. Feature precomputation (one-time cost, ~3-5 min)
    if config.data.precompute_features:
        feature_cache, pos_enc = precompute_all_features(model, ...)
    
    # 4. Optimizer with separate LR groups
    optimizer = AdamW([
        {"params": lora_params, "lr": 1e-4},   # LoRA adapters
        {"params": other_params, "lr": 5e-6},   # Unfrozen embeddings
    ])

    # 5. Training loop
    for epoch in range(config.num_epochs):
        for batch, frame_id_lists in train_loader:
            if feature_cache is not None:
                # Fast path: skip encoder, use cached features
                do_hflip = random.random() < 0.5
                backbone_out = assemble_cached_backbone(
                    feature_cache, pos_enc, frame_id_lists, device, do_hflip
                )
                if do_hflip:
                    batch.masks = batch.masks.flip(-1)  # Match flip
                backbone_out = model.prepare_prompt_inputs(backbone_out, batch)
                outputs = model.forward_tracking(backbone_out, batch)
            else:
                # Original path: full encoder forward
                outputs = model(batch)

            losses = loss_fn(outputs, batch.masks.float())
            losses["core_loss"].backward()
            optimizer.step()
            scheduler.step()

        # Validate & save best checkpoint
        if epoch % val_every == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device,
                                         feature_cache, pos_enc)
            if val_metrics["val_iou"] > best_val_iou:
                save_lora_weights(model, "checkpoints/lora_weights.pt")
```

### Dataset (`neuron_dataset.py`)

#### NeuronVideoDataset

Loads video clips for training with performance optimizations:

```python
class NeuronVideoDataset(Dataset):
    def __init__(self, data_dir, video_names, num_frames=8, transform=None,
                 clip_stride=4, cache_in_memory=True):
        # Build index: pre-enumerate all valid clips with stride
        for vid_idx, vid in enumerate(self.videos):
            n_clips = vid["num_frames"] - num_frames + 1
            for start in range(0, n_clips, clip_stride):  # stride reduces overlap
                self.clips.append((vid_idx, start))
        
        # Optionally cache all frames/masks in RAM at init
        if cache_in_memory:
            for vid in self.videos:
                for idx in range(vid["num_frames"]):
                    self._frame_cache[(vid["name"], idx)] = load_frame(vid, idx)
                    self._mask_cache[(vid["name"], idx)] = load_mask(vid, idx)

    def __getitem__(self, idx):
        vid_idx, start_frame = self.clips[idx]
        # Load from cache or disk, apply augmentations
        # Returns (VideoDatapoint, frame_ids) — frame_ids for feature cache lookup
        return video_datapoint, [(vid_name, frame_idx) for each frame]
```

**Key optimizations:**
- **Clip stride** (default 4): Reduces 329→~82 clips/epoch by skipping overlapping clips
- **In-memory caching**: All frames loaded once at init, eliminating per-step disk I/O
- **Frame ID passthrough**: Returns `(video_name, frame_idx)` tuples alongside data, enabling feature cache lookups during training

#### Collate Function

Custom collate that preserves frame IDs for feature cache lookup:

```python
def neuron_collate_fn(batch):
    video_datapoints = [item[0] for item in batch]
    frame_id_lists = [item[1] for item in batch]  # For feature cache lookup
    batched = collate_fn(video_datapoints, dict_key="neuron_train")
    return batched, frame_id_lists
```

## Data Flow

### Training Data Flow

```
TIF Stacks
    ↓
prepare_data.py (convert to PNG)
    ↓
training_data/
├── images/video_1/00000.png, 00001.png, ...
├── masks/video_1/00000.png, 00001.png, ...
└── manifest.json (stats)
    ↓
NeuronVideoDataset (load clips, in-memory cache)
    ↓
precompute_all_features() [one-time, ~3-5 min]
    ↓  backbone_fpn cached in float16 per frame
    ↓  vision_pos_enc stored once (spatial-only)
    ↓
Training loop:
    assemble_cached_backbone() → prepare_prompt_inputs() → forward_tracking()
    ↓  (skips image encoder entirely)
MultiStepMultiMasksAndIous (compute loss)
    ↓
Backprop + Update LoRA weights
    ↓
checkpoints/lora_weights.pt
```

### Inference Data Flow

```
User paints mask on frame 0
    ↓
GUI extracts mask as numpy array
    ↓
Export frames to temp JPGs
    ↓
Load SAM2 predictor
    ↓
(Optional) Apply LoRA + load weights
    ↓
Add mask prompts via add_new_mask()
    ↓
propagate_in_video()
    ↓
Combine per-object masks
    ↓
Update napari Labels layer
```

## Key Design Decisions

### Why LoRA?

1. **Efficiency**: Only trains ~0.5% of parameters
2. **Memory**: Smaller memory footprint than full fine-tuning
3. **Portability**: Small checkpoint files (~2MB vs ~200MB)
4. **Composability**: Can swap LoRA weights without reloading base model

### Why Feature Precomputation?

1. **Frozen encoder**: Image encoder is fully frozen (no LoRA on Conv2d), so output is deterministic
2. **Dominant cost**: Encoder forward pass is ~60-70% of per-step compute
3. **10x speedup**: ~80 sec/step → ~8 sec/step
4. **Minimal trade-off**: Only loses rotation/shear/intensity augmentations (flips preserved at feature level)

### Why Mask Prompt Conditioning?

1. **Temporal learning**: Model learns to propagate masks, not segment from scratch
2. **Previous mask as context**: Each frame is conditioned on its predecessor
3. **Video coherence**: Ensures predictions are consistent across time
4. **SAM2's design**: The mask decoder is designed to accept dense prompt embeddings

### Why Augmentations?

1. **Small datasets**: Typical neuron datasets have 5-20 videos
2. **Overfitting risk**: SAM2 is large, can memorize training data
3. **Domain gap**: Microscopy images differ from natural images
4. **Feature-level flips**: With precomputed features, flips are applied on cached feature maps — an acceptable approximation for fine-tuning

## Extending the Code

### Adding New Augmentations

Edit `src/training/augmentations.py`:

```python
class VideoAugmentation:
    def __init__(self, ..., my_new_aug_p: float = 0.2):
        self.my_new_aug_p = my_new_aug_p

    def __call__(self, frames, masks):
        do_my_aug = random.random() < self.my_new_aug_p
        # ... apply consistently to all frames
```

### Adding New Loss Functions

Edit `src/training/loss_fns.py`:

```python
class MyNewLoss(nn.Module):
    def forward(self, pred, target):
        # Compute your loss
        return loss

# Add to MultiStepMultiMasksAndIous
class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(self, ..., use_my_loss: bool = True):
        if use_my_loss:
            self.my_loss = MyNewLoss()

    def forward(self, outputs, targets):
        loss_my = self.my_loss(pred_masks, targets) if self.use_my_loss else 0
        # Add to core_loss
```

### Adding LoRA to New Layers

Edit `src/training/lora.py`:

```python
SAM2_LORA_TARGETS = [
    # ... existing patterns
    "my_new_layer.*",  # Add new pattern
]
```

## Performance Considerations

### Speed Optimization (most impactful)

1. **Feature precomputation** (default, `--no-precompute` to disable): Caches frozen image encoder output once. **10x speedup** (~80→~8 sec/step).
2. **Clip stride** (default 4, `--clip-stride`): Reduces overlapping clips. stride=4 gives 4x fewer steps per epoch.
3. **In-memory caching** (default, `--no-cache` to disable): Eliminates disk I/O by loading all frames into RAM at init.
4. **DataLoader workers**: Auto-set to 0 when caching (avoids pickle overhead), otherwise uses `num_workers` from config.
5. **Mixed precision**: Enabled by default on CUDA (bfloat16).

| Configuration | sec/step | steps/epoch | 5 epochs |
|---------------|----------|-------------|----------|
| No optimizations | ~80 | ~329 | ~36 hours |
| Stride=4 only | ~80 | ~82 | ~9 hours |
| **All defaults** | **~8** | **~82** | **~55 min** |

### Memory Optimization

- Use `--batch-size 1` if OOM (already the default)
- Reduce `--num-frames` to 4
- Use `--no-cache` if RAM is limited (trades I/O speed for memory)

### Quality Optimization

- More training data > more epochs
- Validate frequently (`--val-every 5`)
- Use dataset-specific normalization stats (auto-loaded from manifest)
- Use `--no-precompute` for full augmentation pipeline if overfitting is a concern
