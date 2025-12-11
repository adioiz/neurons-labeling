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
| **Masks → Coordinates (ALL pixels)** | Save masks as CSV coordinate table |
| **Carry forward labels** | Copy mask from previous frame |
| **Clear labels (current frame)** | Erase current frame mask |
| **Save labels stack** | Save as TIFF mask file |

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
├── main.py              # Entry point
├── src/
│   ├── gui.py           # Napari GUI with SAM2 integration
│   ├── io.py            # CSV I/O utilities
│   └── utils.py         # Helper functions
├── requirements.txt
└── README.md
```

## Hardware

- **GPU recommended**: CUDA (NVIDIA) or MPS (Apple Silicon)
- **CPU**: Works but slower
