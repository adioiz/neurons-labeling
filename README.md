# Neurons Labeling

Napari-based tool for neuron segmentation with **SAM2 integration** for automatic video propagation, plus **LoRA fine-tuning** for domain adaptation.

## Features

- **Interactive labeling** with napari brush/eraser tools
- **SAM2 video propagation** - paint one frame, auto-propagate to all
- **Fine-tuning support** - adapt SAM2 to your specific neuron data using LoRA
- **Feature precomputation** - 10x faster training by caching frozen encoder output

## Workflow

1. **Load** your neuron TIFF video
2. **Paint** mask on frame 0 using the Labels layer (brush tool)
3. **Click** "SAM2: Propagate from frame 0" button
4. **Review/Edit** any frame with brush/eraser
5. **Save** with "Masks → Coordinates (ALL pixels)" button

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git
```

## Usage

```bash
source venv/bin/activate
python main.py
```

## GUI Buttons

| Button | Description |
|--------|-------------|
| **SAM2: Propagate from frame 0** | Propagates your painted mask to all frames |
| **Use fine-tuned model** | Checkbox to use LoRA fine-tuned weights (if available) |
| **Masks → Coordinates (ALL pixels)** | Save masks as CSV coordinate table |
| **Carry forward labels** | Copy mask from previous frame |
| **Clear labels (current frame)** | Erase current frame mask |
| **Save labels stack** | Save as TIFF mask file |

## Fine-Tuning SAM2

You can fine-tune SAM2 on your own neuron data for better domain-specific performance.

### Step 1: Prepare Training Data

Create a `data_manifest.json` file listing your TIF stacks:

```json
[
    {
        "image_tif": "/path/to/neuron_video.tif",
        "mask_tif": "/path/to/labeled_masks.tif",
        "name": "video_name"
    }
]
```

Then convert to training format:

```bash
python train.py prepare --manifest data_manifest.json --output-dir training_data
```

### Step 2: Train

```bash
# Default training (recommended settings — precomputed features + in-memory caching)
python train.py train

# Custom settings
python train.py train --epochs 60 --lora-rank 8 --clip-stride 2

# Disable optimizations if needed
python train.py train --no-precompute --no-cache
```

### Training Features

| Feature | Description |
|---------|-------------|
| **LoRA** | Efficient fine-tuning with low-rank adapters (rank=8, alpha=16) |
| **Feature Precomputation** | 10x faster training by caching frozen encoder output |
| **In-Memory Caching** | All frames cached in RAM, eliminating disk I/O |
| **Strong Augmentations** | Rotation, shear, flip, brightness/contrast (feature-level flips when precomputed) |

### Using Fine-Tuned Model

After training, weights are saved to `checkpoints/lora_weights.pt`. In the GUI:

1. Check **"Use fine-tuned model"** checkbox
2. Click "SAM2: Propagate from frame 0"
3. The fine-tuned model will be loaded automatically

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 60 | Number of training epochs |
| `--batch-size` | 1 | Batch size |
| `--lora-rank` | 8 | LoRA rank (higher = more capacity) |
| `--lora-alpha` | 16.0 | LoRA alpha scaling factor |
| `--lora-lr` | 1e-4 | LoRA learning rate |
| `--base-lr` | 5e-6 | Base learning rate for non-LoRA params |
| `--num-frames` | 8 | Frames per training clip |
| `--val-every` | 5 | Validation frequency (epochs) |
| `--clip-stride` | 4 | Stride between clip start frames (reduces overlap) |
| `--no-precompute` | — | Disable feature precomputation (slower but keeps full augmentation) |
| `--no-cache` | — | Disable in-memory frame caching (saves RAM) |
| `--use-imagenet-stats` | — | Use ImageNet normalization instead of dataset stats |

## CSV Output Format

```csv
frame,object_id,vertex_id,x,y
0,1,0,206.0,86.0
0,1,1,207.0,86.0
...
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `f` | Carry forward labels |
| `n` | Next frame |
| `p` | Previous frame |

## Project Structure

```
neurons-labeling/
├── main.py                      # Entry point
├── train.py                     # Training CLI
├── src/
│   ├── gui.py                   # Napari GUI with SAM2 integration
│   ├── io.py                    # CSV I/O utilities
│   ├── utils.py                 # Helper functions
│   └── training/                # Fine-tuning module
│       ├── train.py             # Training loop + feature precomputation
│       ├── config.py            # Training configuration
│       ├── augmentations.py     # Data augmentations
│       ├── neuron_dataset.py    # Video dataset loader with caching
│       ├── evaluate.py          # Validation metrics (IoU, Dice)
│       ├── lora.py              # LoRA implementation
│       └── prepare_data.py      # TIF to training data converter
├── checkpoints/                 # Saved model weights
│   └── lora_weights.pt          # Fine-tuned LoRA weights
├── training_data/               # Converted training data
├── labeled_masks/               # Your labeled mask TIFFs (source of truth)
├── labeled_coordinates/         # Exported CSV coordinates
├── requirements.txt
└── README.md
```

## Hardware & Performance

- **GPU recommended**: CUDA (NVIDIA) or MPS (Apple Silicon)
- **CPU**: Works but slower (training not recommended on CPU)

### Training Performance

With feature precomputation enabled (default), training is **~10x faster** since the frozen image encoder output is cached before training begins:

| Configuration | sec/step | 5 epochs |
|---------------|----------|----------|
| Without precomputation | ~80 | ~36 hours |
| **With precomputation (default)** | **~8** | **~55 min** |

## Loss Functions

The training uses SAM2's built-in `MultiStepMultiMasksAndIous` loss:

| Loss | Weight | Purpose |
|------|--------|---------|
| **Focal** | 20.0 | Handles class imbalance |
| **Dice** | 1.0 | Overall shape overlap |
| **IoU** | 1.0 | IoU prediction accuracy |
| **Object Score** | 1.0 | Object presence prediction |

## Data Augmentations

Augmentations are applied consistently across all frames in a clip:

| Augmentation | Parameters | With Precomputed Features |
|--------------|------------|---------------------------|
| Horizontal flip | 50% probability | Applied on feature maps |
| Vertical flip | 50% probability | Applied on feature maps |
| Rotation | ±25 degrees | Not available (skipped) |
| Shear | ±20 degrees | Not available (skipped) |
| Brightness | ±15% | Not available (skipped) |
| Contrast | ±15% | Not available (skipped) |

When using precomputed features (default), only flip augmentations are applied at the feature level. Use `--no-precompute` to enable the full augmentation pipeline (at the cost of ~10x slower training).

## References

- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
