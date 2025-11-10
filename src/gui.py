# src/gui.py
from pathlib import Path
import numpy as np

from tifffile import imread
import napari
from magicgui import magicgui

from .utils import current_frame, info, warn
from .io import save_landmarks, save_contours

class VectorLabelerApp:
    """
    Napari-based manual vector labeler:
    - Draw polygons (contours) per frame → export ordered vertices as CSV.
    - Add landmarks (keypoints) per frame → export as CSV.
    - Duplicate previous frame's landmarks to current frame.
    """
    def __init__(self, tiff_path: Path):
        self.tiff_path = Path(tiff_path)

        # In-memory tables that mirror the CSVs we write:
        self.landmark_rows = []  # list of dicts: {frame, object_id, keypoint_name, x, y}
        self.contour_rows  = []  # list of dicts: {frame, object_id, vertex_id, x, y}

        # Napari handles – set during launch():
        self.viewer = None
        self.imgs = None
        self.points_layer = None
        self.shapes_layer = None

    # ---------- Entry point ----------
    def launch(self):
        """Load TIFF, create layers, dock widgets, hotkeys, and start Napari."""
        # Read TIFF stack. tifffile returns ndarray. We normalize to (T, H, W).
        self.imgs = imread(str(self.tiff_path))
        if self.imgs.ndim == 2:
            # If single frame (H, W), promote to (T=1, H, W) for consistent time handling.
            self.imgs = self.imgs[np.newaxis, ...]

        # Create the viewer and show the image stack
        self.viewer = napari.Viewer(title=f"Manual Vector Labeler: {self.tiff_path.name}")
        self.viewer.add_image(self.imgs, name="stack", rgb=False)

        # Landmarks layer (POINTS): explicitly 3D (t, y, x) so each point stores its frame.
        self.points_layer = self.viewer.add_points(
            ndim=3,                 # (t, y, x)
            name="landmarks",
            size=6,
            properties={
                "object_id": np.array([], int),
                "keypoint_name": np.array([], str),
            },
            text={
                "string": "{object_id}:{keypoint_name}",
                "size": 10,
                "anchor": "upper_left",
            },
        )
        # Set visual properties AFTER creation for broader version compatibility
        self.points_layer.face_color = "white"
        self.points_layer.edge_color = "black"

        # Contours layer (SHAPES): polygons are naturally 2D (y, x).
        # We stamp the *current frame* only when you click "Commit" (so the table is clean).
        self.shapes_layer = self.viewer.add_shapes(
            name="contours",
            shape_type="polygon",
            edge_width=2,
            edge_color="cyan",
            face_color=[0, 1, 1, 0.1],  # transparent cyan fill (RGBA)
        )

        # Build dock widgets (buttons & inputs) and hotkeys
        self._build_panels()
        self._bind_hotkeys()

        # Start the Napari Qt event loop (blocking until window closes)
        napari.run()

    # ---------- Right-side control panels ----------
    def _build_panels(self):
        @magicgui(
            call_button="Add point at cursor (A)",
            layout="vertical",
            object_id={"label": "object_id", "value": 1, "min": 1},
            keypoint_name={"label": "keypoint_name", "value": "kp1"},
        )
        def point_tools(object_id: int = 1, keypoint_name: str = "kp1"):
            """
            Add a landmark at mouse cursor location, tagged with the current frame.
            Napari uses world coords for cursor; for our 2D image that is (x, y).
            Points layer expects (t, y, x) ordering.
            """
            if self.viewer.cursor.position is None:
                warn("Move the cursor over the image first.")
                return

            f = current_frame(self.viewer)      # which frame are we on?
            x, y = self.viewer.cursor.position  # cursor in (x, y)

            # VISUAL: add the point to the layer so you see it immediately.
            self.points_layer.add(
                np.array([[f, y, x]]),  # note (t, y, x)
                properties={"object_id": [object_id], "keypoint_name": [keypoint_name]},
            )

            # TABLE: append to in-memory list (what will be saved to CSV).
            self.landmark_rows.append({
                "frame": f,
                "object_id": int(object_id),
                "keypoint_name": str(keypoint_name),
                "x": float(x),
                "y": float(y),
            })

        @magicgui(
            call_button="Commit selected polygon(s)",
            layout="vertical",
            object_id={"label": "object_id", "value": 1, "min": 1},
        )
        def polygon_tools(object_id: int = 1):
            """
            Take the currently selected polygon(s) from the shapes layer,
            stamp them with the current frame, and append all vertices in order.
            """
            selected = list(self.shapes_layer.selected_data)
            if not selected:
                warn("Select a polygon in 'contours' (click it) before committing.")
                return

            f = current_frame(self.viewer)
            committed = 0

            for idx in selected:
                # Each shape's .data is an (N, 2) array of (y, x) vertices in order
                data = np.asarray(self.shapes_layer.data[idx])
                for vid, (y, x) in enumerate(data, start=0):
                    self.contour_rows.append({
                        "frame": f,
                        "object_id": int(object_id),
                        "vertex_id": int(vid),
                        "x": float(x),
                        "y": float(y),
                    })
                committed += 1

            info(
                f"Committed {committed} polygon(s) at frame {f}.\n"
                f"Tip: Draw → Select → Commit per frame to build a clean coordinate table."
            )

        @magicgui(call_button="Duplicate previous frame → current (D)")
        def duplicate_prev():
            """
            Copy landmarks from frame-1 to frame (fast manual tracking).
            We clone both the coordinates and per-point properties.
            """
            f = current_frame(self.viewer)
            if f == 0:
                warn("No previous frame (you are at f=0).")
                return

            data = self.points_layer.data
            if data is None or len(data) == 0:
                warn("No landmarks exist to duplicate.")
                return

            prev_idxs = np.where(data[:, 0] == (f - 1))[0]
            if prev_idxs.size == 0:
                warn(f"No landmarks found at frame {f-1}.")
                return

            # VISUAL: clone those points, change their time to current frame
            cloned = data[prev_idxs].copy()
            cloned[:, 0] = f

            # Clone point properties for the subset being duplicated
            props = {k: v[prev_idxs] for k, v in self.points_layer.properties.items()}

            self.points_layer.add(cloned, properties=props)

            # TABLE: also append to our CSV buffer
            for i in prev_idxs:
                _, y, x = self.points_layer.data[i]  # original coords
                self.landmark_rows.append({
                    "frame": f,
                    "object_id": int(self.points_layer.properties["object_id"][i]),
                    "keypoint_name": str(self.points_layer.properties["keypoint_name"][i]),
                    "x": float(x),
                    "y": float(y),
                })

            info(f"Duplicated {prev_idxs.size} landmark(s) from frame {f-1} → {f}.\n"
                 f"Now tweak their positions if needed, and add any new points.")

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

            msg_lines = []
            if out_lm: msg_lines.append(f"Landmarks → {out_lm.name}")
            if out_ct: msg_lines.append(f"Contours  → {out_ct.name}")
            info("Saved:\n" + "\n".join(msg_lines))

        # Add the panels to the right side of the Napari window
        self.viewer.window.add_dock_widget(point_tools, area="right")
        self.viewer.window.add_dock_widget(polygon_tools, area="right")
        self.viewer.window.add_dock_widget(duplicate_prev, area="right")
        self.viewer.window.add_dock_widget(saver, area="right")

        # Keep references to functions so hotkeys can call them
        self._point_tools = point_tools
        self._polygon_tools = polygon_tools
        self._duplicate_prev = duplicate_prev
        self._saver = saver

    # ---------- Hotkeys ----------
    def _bind_hotkeys(self):
        @self.viewer.bind_key("a")
        def _add_point(v):
            # Trigger the same logic as clicking the "Add point" button
            self._point_tools()

        @self.viewer.bind_key("d")
        def _dup_prev(v):
            self._duplicate_prev()

        @self.viewer.bind_key("n")
        def _next_frame(v):
            # Move forward in time axis (0), keep other dims unchanged
            f = current_frame(v)
            v.dims.current_step = (min(f + 1, self.imgs.shape[0] - 1),) + v.dims.current_step[1:]

        @self.viewer.bind_key("p")
        def _prev_frame(v):
            f = current_frame(v)
            v.dims.current_step = (max(f - 1, 0),) + v.dims.current_step[1:]