# Neurons Labeling
labeling neuron as "objects" in images of neural network using napari  

Napari app to manually create **coordinate tables** from TIFF stacks.

## Features
- Draw **polygons** (contours) → `contour_vertices.csv` (frame, object_id, vertex_id, x, y)
- Add **landmarks** (keypoints) → `landmarks.csv` (frame, object_id, keypoint_name, x, y)
- **Duplicate** previous frame landmarks (fast updates)
- Save CSV anytime

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt