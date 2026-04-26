# Neurons Labeling

Napari-based tool for neuron segmentation with **SAM2 integration** for automatic video propagation.

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

## Using Fine-Tuned Models

The GUI resolves the LoRA checkpoint by checking these locations in order:

1. `$SAM2_LORA_WEIGHTS` (environment variable, if set)
2. `/Users/adioiz/Documents/Learnings/Thesis/scripts/Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints/lora_weights.pt` (absolute fallback)
3. `../Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints/lora_weights.pt` (sibling-repo)
4. `checkpoints/lora_weights.pt` (in this repo, e.g. via symlink)

After training in `Fine-tune-SAM2-model-for-Neurons-segmentation`:
1. Check the **"Use fine-tuned model"** checkbox in the GUI
2. Click "SAM2: Propagate labels"

If the fine-tuned checkpoint can't be found or loaded, the GUI shows a
warning explaining why and falls back to the base SAM 2 model. The
"SAM2 complete!" message reports which model was actually used, so a
silent fallback is no longer possible.

To override the lookup explicitly:

```bash
export SAM2_LORA_WEIGHTS=/path/to/lora_weights.pt
python main.py
```

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
├── main.py                  # Entry point
├── coordinets_visualization.py
├── requirements.txt
├── labeled_masks/           # Your labeled mask TIFFs (source of truth)
├── labeled_coordinates/     # Exported CSV coordinates
├── checkpoints/             # Optional symlink -> ../Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints (the GUI also accepts SAM2_LORA_WEIGHTS / sibling-repo paths)
└── src/
    ├── gui.py               # Napari GUI with SAM2 + LoRA support
    ├── io.py                # CSV I/O utilities
    ├── utils.py             # Helper functions
    └── lora_utils.py        # LoRA loading for inference
```

## Repository Connection

This repo is connected to `Fine-tune-SAM2-model-for-Neurons-segmentation` via symlinks:

| This Repo | Links To |
|-----------|----------|
| `checkpoints/` | `../Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints/` |

The training repo also symlinks back here:

| Training Repo | Links To |
|---------------|----------|
| `labeled_masks/` | `../neurons-labeling/labeled_masks/` |
| `labeled_coordinates/` | `../neurons-labeling/labeled_coordinates/` |

## Hardware

- **GPU recommended**: CUDA (NVIDIA) or MPS (Apple Silicon)
- **CPU**: Works but slower

## Training Your Own Model

To fine-tune SAM2 on your labeled data, use the companion repository:
[Fine-tune-SAM2-model-for-Neurons-segmentation](../Fine-tune-SAM2-model-for-Neurons-segmentation/)
