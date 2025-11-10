from pathlib import Path
import numpy as np
from skimage.measure import find_contours

from tifffile import imread, imwrite
import napari
from magicgui import magicgui

from .utils import current_frame, info, warn
from .io import save_landmarks, save_contours


class VectorLabelerApp:
    """
    Napari-based manual vector & mask labeler:
    - Draw polygons (contours) per frame → export ordered vertices as CSV.
    - Add landmarks (keypoints) per frame → export as CSV.
    - Duplicate previous frame's landmarks to current frame.
    - Paint masks with a Labels layer, carry previous frame forward, and save labels stack.
    """
    def __init__(self, tiff_path: Path):
        self.tiff_path = Path(tiff_path)

        # In-memory tables mirroring what we save:
        self.landmark_rows = []  # dicts: {frame, object_id, keypoint_name, x, y}
        self.contour_rows  = []  # dicts: {frame, object_id, vertex_id, x, y}

        # Napari handles – set during launch():
        self.viewer = None
        self.imgs = None
        self.points_layer = None
        self.shapes_layer = None

        # Labels (mask) stack
        self.labels_data = None
        self.labels_layer = None

    # ---------- Entry point ----------
    def launch(self):
        """Load TIFF, create layers, dock widgets, hotkeys, and start Napari."""
        # Read TIFF stack. Normalize to (T, H, W).
        self.imgs = imread(str(self.tiff_path))
        if self.imgs.ndim == 2:
            self.imgs = self.imgs[np.newaxis, ...]

        # Viewer + image
        self.viewer = napari.Viewer(title=f"Manual Vector Labeler: {self.tiff_path.name}")
        self.viewer.add_image(self.imgs, name="stack", rgb=False)

        # Labels layer for brush-based masks
        self.labels_data = np.zeros_like(self.imgs, dtype=np.int32)
        self.labels_layer = self.viewer.add_labels(self.labels_data, name="Labels")

        # Landmarks (POINTS), explicit ndim=3 so points store time t
        self.points_layer = self.viewer.add_points(
            ndim=3,
            name="landmarks",
            size=6,
            properties={
                "object_id": np.array([], int),
                "keypoint_name": np.array([], str),
            },
            text={"string": "{object_id}:{keypoint_name}", "size": 10, "anchor": "upper_left"},
        )
        self.points_layer.face_color = "white"
        self.points_layer.edge_color = "black"

        # Contours (SHAPES)
        self.shapes_layer = self.viewer.add_shapes(
            name="contours",
            shape_type="polygon",
            edge_width=2,
        )
        # Robust color setup across napari versions
        self.shapes_layer.edge_color = "cyan"
        self.shapes_layer.face_color = "cyan"
        self.shapes_layer.opacity = 0.25
        self.shapes_layer.blending = "translucent"
        self.shapes_layer.current_edge_color = "cyan"
        self.shapes_layer.current_face_color = "cyan"

        # Panels & hotkeys
        self._build_panels()
        self._bind_hotkeys()

        napari.run()

    # ---------- Right-side control panels ----------
    def _build_panels(self):
        @magicgui(
            call_button="Save CSV",
            layout="vertical",
            csv_base={"label": "CSV base name (no extension)", "value": "annotations"},
        )
        def saver(csv_base: str = "annotations"):
            """
            Write CSVs:
              - <base>_landmarks.csv
              - <base>_contour_vertices.csv
            Only files with accumulated rows are written.
            """
            base = Path(csv_base).with_suffix("")
            out_lm = save_landmarks(self.landmark_rows, base)
            out_ct = save_contours(self.contour_rows, base)
            if not out_lm and not out_ct:
                warn("Nothing to save yet. Add landmarks or commit polygons first.")
                return
            msg = []
            if out_lm: msg.append(f"Landmarks → {out_lm.name}")
            if out_ct: msg.append(f"Contours  → {out_ct.name}")
            info("Saved:\n" + "\n".join(msg))

        # ---------- Mask workflow helpers (FIXED by reassigning .data) ----------
        @magicgui(call_button="Carry forward labels (prev → current)")
        def carry_forward():
            """Copy Labels of frame f-1 into frame f, using .data reassignment to trigger update."""
            f = current_frame(self.viewer)
            if f == 0:
                warn("At frame 0: nothing to carry forward.")
                return
            data = np.asarray(self.labels_layer.data)  # snapshot of live data
            new_data = data.copy()
            new_data[f] = data[f - 1]
            self.labels_layer.data = new_data  # <-- reassign to fire napari update
            info(f"Copied labels from frame {f-1} → {f}. Now refine with the brush/eraser.")

            # keep self.labels_data in sync so Save labels stack uses newest array
            self.labels_data = new_data

        @magicgui(call_button="Clear labels (current frame)")
        def clear_frame():
            """Clear current frame labels to zeros, using .data reassignment to trigger update."""
            f = current_frame(self.viewer)
            data = np.asarray(self.labels_layer.data)
            new_data = data.copy()
            new_data[f] = 0
            self.labels_layer.data = new_data  # <-- reassign to fire napari update
            info(f"Cleared labels at frame {f}.")
            self.labels_data = new_data

        @magicgui(
            call_button="Masks → Coordinates (ALL frames)",
            layout="vertical",
            label_id={"label": "Use label id (0 = any nonzero)", "value": 4, "min": 0},
            object_id={"label": "object_id", "value": 1, "min": 1},
            csv_path={"label": "Output CSV filename", "value": "annotations.csv"},
            also_draw={"label": "Also draw polygons in viewer", "value": False},
        )
        def masks_to_coordinates_all(
            label_id: int = 4,
            object_id: int = 1,
            csv_path: str = "annotations.csv",
            also_draw: bool = False
        ):
            """
            Convert the painted Labels layer into a single polygon (largest contour) per frame,
            then save a coordinate table CSV via save_contours(...):
            columns: frame, object_id, vertex_id, x, y
            """
            lab = np.asarray(self.labels_layer.data)
            T = lab.shape[0] if lab.ndim == 3 else 1

            rows = []
            num_frames_with_contour = 0

            def _canonicalize_contour(c: np.ndarray) -> np.ndarray:
                c = np.asarray(c)  # (N, 2) (y, x)
                i0 = np.lexsort((c[:, 1], c[:, 0]))[0]
                if i0 > 0:
                    c = np.concatenate([c[i0:], c[:i0]], axis=0)
                return c

            for f in range(T):
                m = lab[f] if lab.ndim == 3 else lab
                mask = (m > 0) if label_id == 0 else (m == int(label_id))
                if not mask.any():
                    continue
                contours = find_contours(mask, 0.5)
                if not contours:
                    continue
                contour = _canonicalize_contour(max(contours, key=lambda c: c.shape[0]))
                if also_draw:
                    self.shapes_layer.add(contour, shape_type='polygon')
                for vid, (y, x) in enumerate(contour):
                    rows.append({
                        "frame": int(f),
                        "object_id": int(object_id),
                        "vertex_id": int(vid),
                        "x": float(x),
                        "y": float(y),
                    })
                num_frames_with_contour += 1

            if not rows:
                warn("No contours found. Check label_id or make sure masks exist.")
                return

            base = Path(csv_path).with_suffix("")
            out = save_contours(rows, base)
            info(
                f"Saved coordinates:\n"
                f"  {out.name}\n"
                f"Frames with contours: {num_frames_with_contour}/{T}\n"
                f"Total vertices: {len(rows)}"
            )

        @magicgui(
            call_button="Save labels stack",
            layout="vertical",
            out_path={"label": "Output TIFF filename", "value": "labels.tif"},
            as_npy={"label": "Also save .npy", "value": False},
        )
        def save_labels(out_path: str = "labels.tif", as_npy: bool = False):
            """
            Save the full labels volume (T,H,W) from the live layer.
            """
            self.labels_data = np.asarray(self.labels_layer.data).astype(np.int32)
            p = Path(out_path).expanduser()
            imwrite(str(p), self.labels_data)
            msg = [f"Saved TIFF → {p} (shape {self.labels_data.shape})"]
            if as_npy:
                np.save(str(p.with_suffix(".npy")), self.labels_data)
                msg.append(f"Saved NPY  → {p.with_suffix('.npy')}")
            info("\n".join(msg))
                
        @magicgui(
            call_button="Load labels from file → active layer",
            layout="vertical",
            in_path={"label": "Existing labels file", "value": "labels.tif"},
        )
        def load_labels_from_file(in_path: str = "labels.tif"):
            """Read labels from disk and replace the current Labels layer data."""
            p = Path(in_path).expanduser()
            if not p.exists():
                warn(f"File not found: {p}")
                return
            loaded = imread(str(p))
            # normalize shape & dtype
            if loaded.ndim == 2:
                loaded = loaded[np.newaxis, ...]
            loaded = loaded.astype(np.int32)

            # basic shape check against the image stack
            if loaded.shape[1:] != self.imgs.shape[1:]:
                warn(f"Shape mismatch: file {loaded.shape} vs image {self.imgs.shape}.")
                return
            if loaded.shape[0] != self.imgs.shape[0]:
                warn(f"Time mismatch: file has T={loaded.shape[0]} frames, image has T={self.imgs.shape[0]}.")
                return

            # replace data of our existing Labels layer (fires a redraw)
            self.labels_layer.data = loaded
            self.labels_data = loaded
            info(f"Loaded labels from {p.name} into active 'Labels' layer.")


        
        # Add panels to the right dock
        self.viewer.window.add_dock_widget(saver, area="right")

        self.viewer.window.add_dock_widget(carry_forward, area="right")
        self.viewer.window.add_dock_widget(clear_frame, area="right")
        self.viewer.window.add_dock_widget(save_labels, area="right")

        self.viewer.window.add_dock_widget(masks_to_coordinates_all, area="right")
        self.viewer.window.add_dock_widget(load_labels_from_file, area="right")


        self._masks_to_coordinates_all = masks_to_coordinates_all

        # Keep references for hotkeys
        self._saver = saver
        self._carry_forward = carry_forward
        self._clear_frame = clear_frame
        self._save_labels = save_labels

    # ---------- Hotkeys ----------
    def _bind_hotkeys(self):
        @self.viewer.bind_key("a")
        def _add_point(v):
            self._point_tools()

        @self.viewer.bind_key("f")  # quick carry-forward for masks
        def _carry_mask(v):
            self._carry_forward()

        @self.viewer.bind_key("n")
        def _next_frame(v):
            f = current_frame(v)
            v.dims.current_step = (min(f + 1, self.imgs.shape[0] - 1),) + v.dims.current_step[1:]

        @self.viewer.bind_key("p")
        def _prev_frame(v):
            f = current_frame(v)
            v.dims.current_step = (max(f - 1, 0),) + v.dims.current_step[1:]