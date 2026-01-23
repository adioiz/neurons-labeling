# src/io.py
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

# check if needed ?

def save_landmarks(rows: List[Dict], basepath: Path) -> Optional[Path]:
    """
    Save landmark rows to <base>_landmarks.csv.
    Each row must be: {frame, object_id, keypoint_name, x, y}
    """
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["frame", "object_id", "keypoint_name"])
    out = basepath.with_name(basepath.name + "_landmarks.csv")
    df.to_csv(out, index=False)
    return out

def save_contours(rows: List[Dict], basepath: Path) -> Optional[Path]:
    """
    Save contour vertices to <base>_contour_vertices.csv.
    Each row must be: {frame, object_id, vertex_id, x, y}
    """
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["frame", "object_id", "vertex_id"])
    out = basepath.with_name(basepath.name + ".csv")
    df.to_csv(out, index=False)
    return out