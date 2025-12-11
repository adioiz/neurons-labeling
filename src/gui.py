import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

from pathlib import Path
import numpy as np
from skimage.measure import find_contours

from tifffile import imread, imwrite
import napari
from magicgui import magicgui

from .utils import current_frame, info, warn
from .io import save_landmarks, save_contours

# Check if SAM2 is available
try:
    from sam2.build_sam import build_sam2_video_predictor_hf
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class VectorLabelerApp:
    """
    Napari-based manual vector & mask labeler:
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
            call_button="Masks → Coordinates (ALL pixels)",
            layout="vertical",
            label_id={"label": "Use label id (0 = any nonzero)", "value": 4, "min": 0},
            object_id={"label": "object_id", "value": 1, "min": 1},
            csv_path={"label": "Output CSV filename", "value": "annotations.csv"},
        )
        def masks_to_coordinates_all(
            label_id: int = 4,
            object_id: int = 1,
            csv_path: str = "annotations.csv"
        ):
            """
            Convert the painted Labels layer into a coordinate table of ALL pixels in the mask.
            columns: frame, object_id, vertex_id, x, y
            """
            lab = np.asarray(self.labels_layer.data)
            T = lab.shape[0] if lab.ndim == 3 else 1

            rows = []
            num_frames_with_mask = 0

            for f in range(T):
                m = lab[f] if lab.ndim == 3 else lab
                
                # Create a boolean mask for the specified label_id
                if label_id == 0:
                    mask = m > 0
                else:
                    mask = m == int(label_id)
                
                if not mask.any():
                    continue
                
                num_frames_with_mask += 1

                # Get the coordinates of all pixels in the mask
                # np.where returns (y_coords, x_coords)
                y_coords, x_coords = np.where(mask)
                
                # Iterate over all pixels and add them to the rows
                for i in range(len(y_coords)):
                    rows.append({
                        "frame": int(f),
                        "object_id": int(object_id),
                        # Use 'i' as a sequential vertex_id for now, as it's not a contour
                        "vertex_id": int(i), 
                        "x": float(x_coords[i]),
                        "y": float(y_coords[i]),
                    })

            if not rows:
                warn("No masks found. Check label_id or make sure masks exist.")
                return

            base = Path(csv_path).with_suffix("")
            # We reuse save_contours to save the rows, as it just saves a CSV from a list of dicts
            out = save_contours(rows, base)
            
            if out:
                info(
                    f"Saved coordinates (all mask pixels):\n"
                    f"  {out.name}\n"
                    f"Frames with masks: {num_frames_with_mask}/{T}\n"
                    f"Total pixels: {len(rows)}"
                )
            else:
                warn("Could not save coordinates.")

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

        # ---------- SAM2 Propagation ----------
        @magicgui(
            call_button="SAM2: Propagate from frame 0",
            layout="vertical",
            label_id={"label": "Label ID to propagate", "value": 4, "min": 1},
        )
        def sam2_propagate(label_id: int = 4):
            """
            Use SAM2 to propagate mask from frame 0 to ALL frames.

            Steps:
            1. Paint your mask on frame 0 using the Labels layer
            2. Click this button
            3. SAM2 propagates to all frames
            4. Review/edit with brush/eraser
            5. Save with "Masks → Coordinates"
            """
            if not SAM2_AVAILABLE:
                warn(
                    "SAM2 not installed!\n\n"
                    "Install with:\n"
                    "  pip install huggingface_hub\n"
                    "  pip install git+https://github.com/facebookresearch/sam2.git"
                )
                return

            # Get labels data
            labels_data = np.asarray(self.labels_layer.data)

            # Get mask from frame 0
            if labels_data.ndim == 3:
                frame0_mask = (labels_data[0] == label_id).astype(np.uint8)
            else:
                frame0_mask = (labels_data == label_id).astype(np.uint8)

            if frame0_mask.sum() == 0:
                warn(f"No mask found on frame 0 with label {label_id}.\n\n"
                     f"Please paint a mask on frame 0 first.")
                return

            info(f"Starting SAM2 propagation...\n"
                 f"Frame 0 mask: {frame0_mask.sum()} pixels\n"
                 f"This may take a moment...")

            try:
                output_masks = self._run_sam2(frame0_mask, label_id)

                # Update labels layer
                self.labels_layer.data = output_masks
                self.labels_data = output_masks

                frames_with_mask = np.sum(output_masks.max(axis=(1, 2)) > 0)
                total_frames = output_masks.shape[0]

                info(f"SAM2 complete!\n"
                     f"Frames with mask: {frames_with_mask}/{total_frames}\n\n"
                     f"You can now:\n"
                     f"- Browse frames to review\n"
                     f"- Edit with brush/eraser\n"
                     f"- Save with 'Masks → Coordinates'")

            except Exception as e:
                warn(f"SAM2 error: {str(e)}")
                import traceback
                traceback.print_exc()


        # Add panels to the right dock
        self.viewer.window.add_dock_widget(saver, area="right")

        self.viewer.window.add_dock_widget(carry_forward, area="right")
        self.viewer.window.add_dock_widget(clear_frame, area="right")
        self.viewer.window.add_dock_widget(save_labels, area="right")

        self.viewer.window.add_dock_widget(masks_to_coordinates_all, area="right")
        self.viewer.window.add_dock_widget(load_labels_from_file, area="right")

        # SAM2 button
        self.viewer.window.add_dock_widget(sam2_propagate, area="right")

        self._masks_to_coordinates_all = masks_to_coordinates_all
        self._sam2_propagate = sam2_propagate

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

    # ---------- SAM2 Implementation ----------
    def _run_sam2(self, frame0_mask: np.ndarray, label_id: int) -> np.ndarray:
        """
        Run SAM2 video propagation from frame 0 mask.

        Args:
            frame0_mask: Binary mask (H, W) for frame 0
            label_id: Label ID to assign to output masks

        Returns:
            output_masks: (T, H, W) array with propagated masks
        """
        import tempfile
        import shutil
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2_video_predictor_hf

        # Detect device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        print(f"SAM2: Using device: {device}")

        # Create temp directory for video frames
        temp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
        print(f"SAM2: Temp dir: {temp_dir}")

        try:
            # Save video frames as JPEG images
            T, H, W = self.imgs.shape[:3]
            print(f"SAM2: Saving {T} frames...")

            for i in range(T):
                frame = self.imgs[i].copy().astype(np.float32)

                # Normalize to 0-255
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max > frame_min:
                    frame = (frame - frame_min) / (frame_max - frame_min) * 255
                frame = frame.astype(np.uint8)

                # Convert grayscale to RGB
                frame_rgb = np.stack([frame, frame, frame], axis=-1)

                # Save as JPEG
                img = Image.fromarray(frame_rgb)
                img.save(temp_dir / f'{i:05d}.jpg')

            # Load SAM2 model
            print("SAM2: Loading model (tiny)...")
            predictor = build_sam2_video_predictor_hf(
                "facebook/sam2.1-hiera-tiny",
                device=device
            )

            # Initialize video state
            print("SAM2: Initializing video state...")
            inference_state = predictor.init_state(video_path=str(temp_dir))

            # Add frame 0 mask as prompt
            print(f"SAM2: Adding mask prompt ({frame0_mask.sum()} pixels)...")
            _, obj_ids, mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=frame0_mask
            )

            # Propagate through video
            print("SAM2: Propagating through video...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                video_segments[out_frame_idx] = mask
                print(f"  Frame {out_frame_idx}: {mask.sum()} pixels")

            # Assemble output mask stack
            output_masks = np.zeros((T, H, W), dtype=np.int32)
            for frame_idx, mask in video_segments.items():
                output_masks[frame_idx][mask > 0] = label_id

            print(f"SAM2: Done! Propagated to {len(video_segments)} frames")
            return output_masks

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)