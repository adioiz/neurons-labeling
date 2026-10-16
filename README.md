# Neurons Labeling — annotation and inference for NeuroVestigo

Napari plug-in for annotating single neurons in label-free phase-contrast
time-lapse microscopy, and for running **NeuroVestigo** — the fine-tuned SAM 2
model described in the accompanying paper — to propagate one annotation across
a whole recording.

Annotate a neuron on frame 0, propagate, correct anything wrong, export. Because
a corrected frame is a mask in exactly the format the model was trained on,
ordinary use of the tool produces training data without a separate annotation
effort.

**Model weights ship with this repository** —
`checkpoints-16-frames/lora_weights.pt` (22 MB). These are the 5.56 M parameters
NeuroVestigo trains (LoRA adapters plus the fine-tuned prompt encoder, memory
encoder and decoder heads), not a full SAM 2 checkpoint: they are applied on top
of the public SAM 2.1 Hiera-tiny weights, which are downloaded separately and
remain under their own license.

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

Nothing to install or download — tick **"Use fine-tuned model"** in the GUI and
click "SAM2: Propagate labels". The weights are in this repository.

The GUI resolves the LoRA checkpoint by checking these locations in order:

1. `$SAM2_LORA_WEIGHTS` (environment variable, if set)
2. `../NeuroVestigo/checkpoints/lora_weights.pt` (sibling training repo — where training writes, so this is the most recent run)
3. `checkpoints/lora_weights.pt`
4. **`checkpoints-16-frames/lora_weights.pt` (bundled here)**
5. the same two paths under `checkpoints-new/` (legacy layout)

`checkpoints-16-frames/` holds the released NeuroVestigo model: run 1 of the
paper's three-split evaluation, at its topology tie-break checkpoint (epoch 110
of 110; validation IoU 0.8364, Betti-0 error 0.605). The recipe is the paper's:
16-frame training clips, `topology_resolution` 512, loss weights focal 3.0 /
Dice 2.0 / cbDice 0.3 / supervoxel 0.3, 23 training and 5 validation
recordings. Its four held-out test recordings are A71_C1, A81_C1, A12_C3_100k
and A12_C1_100k_2.
`training_config.json` sits beside the weights and records the exact
configuration that produced them.

If you have NeuroVestigo checked out alongside this repo, entry 2 wins and you
get whatever that checkout's `checkpoints/` holds (training writes there; it is
not in git). Set `SAM2_LORA_WEIGHTS` to pin a specific checkpoint instead.

Leaving the checkbox unticked runs base SAM 2 instead, which is how the
zero-shot baseline in the paper is reproduced.

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
├── checkpoints-16-frames/   # NeuroVestigo weights (bundled fallback)
│   ├── lora_weights.pt      #   LoRA adapters, 22 MB
│   └── training_config.json #   the exact config that produced them
│                            # ../NeuroVestigo/checkpoints/ takes priority
│                            # when that repo is checked out alongside
└── src/
    ├── gui.py               # Napari GUI with SAM2 + LoRA support
    ├── io.py                # CSV I/O utilities
    ├── utils.py             # Helper functions
    └── lora_utils.py        # LoRA loading for inference
```

## Repository Connection

This repo is self-contained: the weights it needs are bundled, so it can be
cloned and used on its own.

If you also check out the training repo as a sibling directory, that repo
symlinks back here for the annotation data:

| Training Repo | Links To |
|---------------|----------|
| `labeled_masks/` | `../neurons-labeling/labeled_masks/` |
| `labeled_coordinates/` | `../neurons-labeling/labeled_coordinates/` |

## Hardware

- **GPU recommended**: CUDA (NVIDIA) or MPS (Apple Silicon)
- **CPU**: Works but slower

## Training Your Own Model

To fine-tune SAM2 on your labeled data, use the companion repository:
[NeuroVestigo](../NeuroVestigo/)
