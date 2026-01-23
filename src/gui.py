import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

from pathlib import Path
import numpy as np
from skimage.measure import find_contours

from tifffile import imread, imwrite
import napari
from magicgui import magicgui

from .utils import current_frame, info, warn
from .io import save_contours

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

        # Panels & hotkeys
        self._build_panels()
        self._bind_hotkeys()

        napari.run()

    # ---------- Right-side control panels ----------
    def _build_panels(self):
        # --- 1. SETUP PATHS & NAMES ---
        # Get filename without extension (stem), replace spaces with underscores
        clean_stem = self.tiff_path.stem.replace(" ", "_")
        
        # Define output directories
        mask_dir = Path("labeled_masks")
        coord_dir = Path("labeled_coordinates")
        
        # Create directories if they don't exist
        mask_dir.mkdir(exist_ok=True, parents=True)
        coord_dir.mkdir(exist_ok=True, parents=True)

        # Define default filenames
        default_csv_name = f"{clean_stem}_labels.csv"
        default_tif_name = f"{clean_stem}_masks.tif"

        # --- 2. BUILD WIDGETS ---

        # ---------- Mask workflow helpers ----------
        @magicgui(call_button="Carry forward labels (prev → current)")
        def carry_forward():
            """Copy Labels of frame f-1 into frame f."""
            f = current_frame(self.viewer)
            if f == 0:
                warn("At frame 0: nothing to carry forward.")
                return
            data = np.asarray(self.labels_layer.data) 
            new_data = data.copy()
            new_data[f] = data[f - 1]
            self.labels_layer.data = new_data
            info(f"Copied labels from frame {f-1} → {f}.")
            self.labels_data = new_data

        @magicgui(call_button="Clear labels (current frame)")
        def clear_frame():
            """Clear current frame labels to zeros."""
            f = current_frame(self.viewer)
            data = np.asarray(self.labels_layer.data)
            new_data = data.copy()
            new_data[f] = 0
            self.labels_layer.data = new_data 
            info(f"Cleared labels at frame {f}.")
            self.labels_data = new_data

        @magicgui(
            call_button="Masks → Coordinates",
            layout="vertical",
            label_id={"label": "Use label id:", "value": 4, "min": 0},
            object_id={"label": "object_id", "value": 1, "min": 1},
            csv_filename={"label": "Output CSV name", "value": default_csv_name},
        )
        def masks_to_coordinates_all(
            label_id: int = 4,
            object_id: int = 1,
            csv_filename: str = default_csv_name
        ):
            """
            Convert painted Labels to coordinates.
            Saves to: labeled_coordinates/<csv_filename>
            """
            lab = np.asarray(self.labels_layer.data)
            T = lab.shape[0] if lab.ndim == 3 else 1

            rows = []
            num_frames_with_mask = 0

            for f in range(T):
                m = lab[f] if lab.ndim == 3 else lab
                
                if label_id == 0:
                    mask = m > 0
                else:
                    mask = m == int(label_id)
                
                if not mask.any():
                    continue
                
                num_frames_with_mask += 1
                y_coords, x_coords = np.where(mask)
                
                for i in range(len(y_coords)):
                    rows.append({
                        "frame": int(f),
                        "object_id": int(object_id),
                        "vertex_id": int(i), 
                        "x": float(x_coords[i]),
                        "y": float(y_coords[i]),
                    })

            if not rows:
                warn("No masks found. Check label_id or make sure masks exist.")
                return

            # Construct full path into labeled_coordinates folder
            full_path = coord_dir / csv_filename
            base = full_path.with_suffix("") # save_contours adds .csv automatically
            
            # We reuse save_contours to save the rows
            out = save_contours(rows, base)
            
            if out:
                info(
                    f"Saved coordinates:\n"
                    f"  {out}\n" # Prints full path
                    f"Frames with masks: {num_frames_with_mask}/{T}\n"
                    f"Total pixels: {len(rows)}"
                )
            else:
                warn("Could not save coordinates.")

        @magicgui(
            call_button="Save masks labels stack",
            layout="vertical",
            out_filename={"label": "Output TIFF name", "value": default_tif_name},
        )
        def save_labels(out_filename: str = default_tif_name):
            """
            Save the full labels volume.
            Saves to: labeled_masks/<out_filename>
            """
            self.labels_data = np.asarray(self.labels_layer.data).astype(np.int32)
            
            # Construct full path into labeled_masks folder
            p = mask_dir / out_filename
            
            imwrite(str(p), self.labels_data)
            msg = [f"Saved TIFF → {p} (shape {self.labels_data.shape})"]
            info("\n".join(msg))
                
        @magicgui(
            call_button="Load labels from file → active layer",
            layout="vertical",
            in_path={"label": "Existing labels file", "value": str(mask_dir / default_tif_name)},
        )
        def load_labels_from_file(in_path: str = str(mask_dir / default_tif_name)):
            """Read labels from disk and replace the current Labels layer data."""
            p = Path(in_path).expanduser()
            if not p.exists():
                warn(f"File not found: {p}")
                return
            loaded = imread(str(p))
            if loaded.ndim == 2:
                loaded = loaded[np.newaxis, ...]
            loaded = loaded.astype(np.int32)

            if loaded.shape[1:] != self.imgs.shape[1:]:
                warn(f"Shape mismatch: file {loaded.shape} vs image {self.imgs.shape}.")
                return
            if loaded.shape[0] != self.imgs.shape[0]:
                warn(f"Time mismatch: file has T={loaded.shape[0]} frames, image has T={self.imgs.shape[0]}.")
                return

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
            # (Keep existing SAM2 logic exactly as is)
            if not SAM2_AVAILABLE:
                warn("SAM2 not installed!")
                return
            labels_data = np.asarray(self.labels_layer.data)
            if labels_data.ndim == 3:
                frame0_mask = (labels_data[0] == label_id).astype(np.uint8)
            else:
                frame0_mask = (labels_data == label_id).astype(np.uint8)

            if frame0_mask.sum() == 0:
                warn(f"No mask found on frame 0 with label {label_id}.\nPlease paint a mask on frame 0 first.")
                return

            info(f"Starting SAM2 propagation...")

            try:
                output_masks = self._run_sam2(frame0_mask, label_id)
                self.labels_layer.data = output_masks
                self.labels_data = output_masks
                info(f"SAM2 complete!")
            except Exception as e:
                warn(f"SAM2 error: {str(e)}")
                import traceback
                traceback.print_exc()

        # Add panels to the right dock
        self.viewer.window.add_dock_widget(carry_forward, area="right")
        self.viewer.window.add_dock_widget(clear_frame, area="right")
        self.viewer.window.add_dock_widget(save_labels, area="right")
        self.viewer.window.add_dock_widget(masks_to_coordinates_all, area="right")
        self.viewer.window.add_dock_widget(load_labels_from_file, area="right")
        self.viewer.window.add_dock_widget(sam2_propagate, area="right")

        self._masks_to_coordinates_all = masks_to_coordinates_all
        self._sam2_propagate = sam2_propagate

        # Keep references for hotkeys
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