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


# Candidate locations for fine-tuned LoRA weights, tried in order.
# Override with the SAM2_LORA_WEIGHTS environment variable.
#
# checkpoints-16-frames/ holds the SHIPPED model -- the sixteen-frame run that
# the paper reports. Every path here previously named checkpoints-new/ or
# checkpoints/, neither of which contains weights on any machine, so the
# checkbox fell back to base SAM 2 and the tool never ran the published model.
# The legacy names are kept last so older local setups still resolve.
#
# The absolute path that used to head this list pointed into one developer's
# home directory; it cannot resolve for anyone else and is removed. Use
# SAM2_LORA_WEIGHTS for an out-of-tree checkpoint.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_REPO = "Fine-tune-SAM2-model-for-Neurons-segmentation"
LORA_PATH_CANDIDATES = [
    _REPO_ROOT.parent / _TRAIN_REPO / "checkpoints-16-frames" / "lora_weights.pt",
    _REPO_ROOT / "checkpoints-16-frames" / "lora_weights.pt",
    _REPO_ROOT.parent / _TRAIN_REPO / "checkpoints-new" / "lora_weights.pt",
    _REPO_ROOT / "checkpoints-new" / "lora_weights.pt",
    _REPO_ROOT.parent / _TRAIN_REPO / "checkpoints" / "lora_weights.pt",
    _REPO_ROOT / "checkpoints" / "lora_weights.pt",
]


def _resolve_lora_path():
    """Return the first existing LoRA checkpoint path, or None."""
    env = os.environ.get("SAM2_LORA_WEIGHTS")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for p in LORA_PATH_CANDIDATES:
        if p.exists():
            return p
    return None


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

        # SAM2 model cache
        self._sam2_predictor = None
        self._sam2_is_finetuned = None  # Track which variant is loaded

        # SAM2 propagation history: list of dicts, one per run, persisted for
        # the lifetime of this VectorLabelerApp so a base run and a fine-tuned
        # run done one after the other can be compared side-by-side.
        # Each entry: {"model": str, "start_frame": int, "label_ids": list[int],
        #              "per_frame": {frame_idx: {label_id: pixel_count}}}
        self._sam2_run_history = []

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

        @magicgui(
            call_button="SAM2: Propagate labels",
            layout="vertical",
            label_id={"label": "Label ID (0=all non-zero)", "value": 0, "min": 0, "max": 255},
            from_current={"label": "From current frame", "value": False},
            use_finetuned={"label": "Use fine-tuned model", "value": True},
        )
        def sam2_propagate(label_id: int = 0, from_current: bool = False, use_finetuned: bool = True):
            """
            Use SAM2 to propagate mask(s) forward through video.

            Steps:
            1. Paint your mask(s) on the source frame using the Labels layer
            2. (Optional) Check "From current frame" to start from current frame instead of frame 0
            3. Set label_id to 0 to propagate ALL labels, or specify a single label ID
            4. Toggle "Use fine-tuned model" to use LoRA fine-tuned weights (if available)
            5. Click this button
            6. SAM2 propagates forward to remaining frames
            7. Review/edit with brush/eraser
            8. Save with "Masks → Coordinates"
            """
            if not SAM2_AVAILABLE:
                warn(
                    "SAM2 not installed!\n\n"
                    "Install with:\n"
                    "  pip install huggingface_hub\n"
                    "  pip install git+https://github.com/facebookresearch/sam2.git"
                )
                return

            # Determine start frame
            start_frame = current_frame(self.viewer) if from_current else 0

            # Get labels data
            labels_data = np.asarray(self.labels_layer.data)

            # Get mask from source frame
            if labels_data.ndim == 3:
                source_mask = labels_data[start_frame]
            else:
                source_mask = labels_data

            # Determine which label IDs to propagate
            if label_id == 0:
                # Auto-detect all non-zero labels
                label_ids = np.unique(source_mask[source_mask > 0]).tolist()
                if len(label_ids) == 0:
                    warn(f"No masks found on frame {start_frame}.\n\n"
                        f"Please paint mask(s) on frame {start_frame} first.")
                    return
                label_str = f"{len(label_ids)} label(s): {label_ids}"
            else:
                # Single label ID specified
                if (source_mask == label_id).sum() == 0:
                    warn(f"No mask found on frame {start_frame} with label {label_id}.\n\n"
                        f"Please paint a mask with label {label_id} on frame {start_frame} first.")
                    return
                label_ids = [label_id]
                label_str = f"label {label_id}"

            requested_str = "fine-tuned" if use_finetuned else "base"
            info(f"Starting SAM2 propagation ({requested_str})...\n"
                f"Source frame: {start_frame}\n"
                f"Propagating: {label_str}\n"
                f"This may take a moment...")

            try:
                output_masks, actually_finetuned = self._run_sam2(
                    source_frame_mask=source_mask,
                    start_frame=start_frame,
                    label_ids=label_ids,
                    use_finetuned=use_finetuned,
                )

                # Update labels layer
                self.labels_layer.data = output_masks
                self.labels_data = output_masks

                frames_with_mask = np.sum(output_masks.max(axis=(1, 2)) > 0)
                total_frames = output_masks.shape[0]

                actual_str = "fine-tuned" if actually_finetuned else "base"
                info(f"SAM2 complete! (model used: {actual_str})\n"
                    f"Propagated from frame {start_frame} forward\n"
                    f"Labels: {label_str}\n"
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

    def _print_sam2_session_comparison(self):
        """
        Print every SAM2 propagation run from this session in a single side-by-side
        table. One column per run, rows are frame indices. Each cell is the total
        pixel count for that run's propagated labels in that frame. Lets the user
        run base, then fine-tuned (or vice versa) and visually compare without
        having to scroll back through individual run reports.
        """
        history = self._sam2_run_history
        if not history:
            return

        # Build a compact tag for each run: "#1 base L4" / "#2 ft L4,7"
        col_tags = []
        for i, run in enumerate(history, start=1):
            label_part = ",".join(str(l) for l in run["label_ids"])
            short_model = "ft" if run["model"] == "fine-tuned" else "base"
            col_tags.append(f"#{i} {short_model} L{label_part}")

        col_w = max(10, *(len(t) for t in col_tags))

        # Union of all frames seen across all runs
        all_frames = sorted({f for run in history for f in run["per_frame"].keys()})

        header = f"{'frame':>6} | " + " | ".join(f"{t:>{col_w}}" for t in col_tags)
        sep = "=" * len(header)
        thin = "-" * len(header)

        print()
        print(sep)
        print(f"SAM2 session comparison  |  {len(history)} run(s)")
        for i, run in enumerate(history, start=1):
            print(f"  #{i}: model={run['model']:<10}  start_frame={run['start_frame']}  labels={run['label_ids']}")
        print(thin)
        print(header)
        print(thin)

        col_totals = [0 for _ in history]
        for f in all_frames:
            cells = []
            for i, run in enumerate(history):
                per_label = run["per_frame"].get(f)
                if per_label is None:
                    cells.append(f"{'-':>{col_w}}")
                else:
                    n = sum(per_label.values())
                    col_totals[i] += n
                    cells.append(f"{n:>{col_w},}")
            print(f"{f:>6} | " + " | ".join(cells))

        print(thin)
        totals_cells = " | ".join(f"{n:>{col_w},}" for n in col_totals)
        print(f"{'TOTAL':>6} | {totals_cells}")

        # Pairwise diffs are handy when comparing exactly two runs (the
        # canonical base vs fine-tuned case)
        if len(history) >= 2:
            base_totals = col_totals
            print(thin)
            print("Pairwise totals delta (col_j - col_i, positive = j has more pixels):")
            for i in range(len(history)):
                for j in range(i + 1, len(history)):
                    delta = base_totals[j] - base_totals[i]
                    pct = (delta / base_totals[i] * 100.0) if base_totals[i] else float('nan')
                    print(f"  #{j+1} vs #{i+1}: {delta:+,} px  ({pct:+.2f}%)")

        print(sep)

    def _load_sam2_predictor(self, use_finetuned: bool = True, device: str = "cpu"):
        """
        Load or return cached SAM2 video predictor.

        When use_finetuned is True, loads the base SAM 2 model with LoRA adapters
        from a fine-tuned checkpoint resolved via _resolve_lora_path(). If the
        checkpoint can't be found or loaded, raises — callers must decide whether
        to fall back to the base model. Never silently swaps in the base model
        while the caller still believes the fine-tuned weights are active.

        Returns the predictor; the caller should read self._sam2_is_finetuned
        to know which variant is active.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        # Return cached predictor if variant matches
        if (
            self._sam2_predictor is not None
            and self._sam2_is_finetuned == use_finetuned
        ):
            return self._sam2_predictor

        if use_finetuned:
            lora_path = _resolve_lora_path()
            if lora_path is None:
                searched = "\n  ".join(str(p) for p in LORA_PATH_CANDIDATES)
                raise FileNotFoundError(
                    "Fine-tuned LoRA checkpoint not found. Searched:\n  "
                    + searched
                    + "\nSet SAM2_LORA_WEIGHTS to override."
                )

            print(f"SAM2: Loading base model + LoRA fine-tuned weights from {lora_path}...")
            from src.lora_utils import (
                apply_lora,
                load_lora_weights,
                detect_lora_config_from_checkpoint,
                SAM2_LORA_TARGETS,
                HIERA_LAST_STAGE_LORA_TARGETS,
            )
            from sam2.build_sam import build_sam2_video_predictor, _hf_download

            config_name, ckpt_path = _hf_download("facebook/sam2.1-hiera-tiny")
            predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=ckpt_path,
                device=device,
                mode="eval",
            )

            # Auto-detect ranks from the checkpoint so the in-memory LoRA
            # layers match the saved tensor shapes. Run 6 introduced a
            # SECOND LoRA pass on the Hiera encoder's last stage at a
            # smaller rank than the main LoRA — so we need to detect and
            # apply each group separately. Without this split, the
            # single-rank detector would pick whichever group's tensors
            # came first and fail to load the other.
            lora_cfg = detect_lora_config_from_checkpoint(str(lora_path))
            main_cfg = lora_cfg["main"]
            hiera_cfg = lora_cfg["hiera"]
            print(
                f"SAM2: detected main LoRA rank={main_cfg['rank']}, "
                f"alpha={main_cfg['alpha']} from checkpoint"
            )
            apply_lora(
                predictor,
                target_patterns=SAM2_LORA_TARGETS,
                rank=main_cfg["rank"],
                alpha=main_cfg["alpha"],
            )
            if hiera_cfg is not None:
                print(
                    f"SAM2: detected Hiera last-stage LoRA rank="
                    f"{hiera_cfg['rank']}, alpha={hiera_cfg['alpha']} "
                    f"— applying second LoRA pass on Hiera trunk"
                )
                apply_lora(
                    predictor,
                    target_patterns=HIERA_LAST_STAGE_LORA_TARGETS,
                    rank=hiera_cfg["rank"],
                    alpha=hiera_cfg["alpha"],
                )
            load_lora_weights(predictor, str(lora_path), device=device)
            predictor.eval()
            print("SAM2: Fine-tuned model loaded successfully")

            self._sam2_predictor = predictor
            self._sam2_is_finetuned = True
            return predictor

        print("SAM2: Loading base model (tiny)...")
        predictor = build_sam2_video_predictor_hf(
            "facebook/sam2.1-hiera-tiny",
            device=device,
        )
        self._sam2_predictor = predictor
        self._sam2_is_finetuned = False
        return predictor

    def _run_sam2(
        self,
        source_frame_mask: np.ndarray,
        start_frame: int = 0,
        label_ids: list = None,
        use_finetuned: bool = True,
    ):
        """
        Run SAM2 video propagation from specified frame with support for multiple labels.

        Args:
            source_frame_mask: Label mask (H, W) from source frame (can contain multiple label IDs)
            start_frame: Frame index to start propagation from
            label_ids: List of label IDs to propagate. If None, auto-detect all non-zero labels
            use_finetuned: If True, attempt to load LoRA fine-tuned model. If the
                fine-tuned checkpoint can't be loaded, the user is warned and
                propagation falls back to the base model.

        Returns:
            (output_masks, actually_finetuned):
                output_masks: (T, H, W) array with propagated masks, preserving label IDs
                actually_finetuned: True iff the LoRA fine-tuned weights were used
        """
        import tempfile
        import shutil
        import torch
        from PIL import Image

        # Auto-detect label IDs if not provided
        if label_ids is None:
            label_ids = np.unique(source_frame_mask[source_frame_mask > 0]).tolist()

        if len(label_ids) == 0:
            raise ValueError("No labels found in source frame mask")

        # Detect device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        print(f"SAM2: Using device: {device}")
        print(f"SAM2: Propagating {len(label_ids)} label(s): {label_ids}")
        print(f"SAM2: Start frame: {start_frame}")

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

            # Load SAM2 model (cached, with optional LoRA). If the fine-tuned
            # load fails, surface the reason to the user and fall back to base —
            # but only after telling them, so the success message can't lie.
            try:
                predictor = self._load_sam2_predictor(
                    use_finetuned=use_finetuned, device=device
                )
            except Exception as e:
                if use_finetuned:
                    import traceback
                    traceback.print_exc()
                    warn(
                        "Failed to load fine-tuned model — falling back to base SAM 2.\n\n"
                        f"Reason: {e}"
                    )
                    predictor = self._load_sam2_predictor(
                        use_finetuned=False, device=device
                    )
                else:
                    raise

            # Initialize video state
            print("SAM2: Initializing video state...")
            inference_state = predictor.init_state(video_path=str(temp_dir))

            # Add masks for all label IDs from source frame
            for label_id in label_ids:
                binary_mask = (source_frame_mask == label_id).astype(np.uint8)
                pixel_count = binary_mask.sum()
                print(f"SAM2: Adding label {label_id} ({pixel_count} pixels)...")

                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=start_frame,
                    obj_id=label_id,  # Use actual label value as object ID
                    mask=binary_mask
                )

            # Propagate through video
            print("SAM2: Propagating through video...")
            video_segments = {}  # {frame_idx: {label_id: mask}}

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {}

                # Process each object's mask
                for obj_idx, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[obj_idx] > 0.0).cpu().numpy().squeeze()
                    video_segments[out_frame_idx][obj_id] = mask
                    print(f"  Frame {out_frame_idx}, Label {obj_id}: {mask.sum()} pixels")

            # Get current labels data to preserve labels before start_frame
            current_labels = np.asarray(self.labels_layer.data).copy()

            # Assemble output mask stack
            output_masks = current_labels.copy()  # Start with existing labels

            # Only update frames from start_frame onwards
            for frame_idx in range(start_frame, T):
                if frame_idx in video_segments:
                    # Clear only the label IDs we're propagating in this frame
                    for label_id in label_ids:
                        output_masks[frame_idx][output_masks[frame_idx] == label_id] = 0

                    # Apply new masks for each label
                    for label_id, mask in video_segments[frame_idx].items():
                        output_masks[frame_idx][mask > 0] = label_id

            propagated_frames = len([f for f in video_segments.keys() if f >= start_frame])
            print(f"SAM2: Done! Propagated to {propagated_frames} frames (from frame {start_frame})")

            # Comparison-friendly per-frame coverage summary. Print this as a
            # self-contained block so it can be eyeballed side-by-side across
            # base vs fine-tuned runs in the same terminal session.
            model_tag = "fine-tuned" if self._sam2_is_finetuned else "base"
            sorted_label_ids = sorted(label_ids)
            label_headers = [f"L{lid}" for lid in sorted_label_ids]
            col_w = max(8, *(len(h) + 2 for h in label_headers))

            print("=" * 72)
            print(f"SAM2 coverage report  |  model: {model_tag}  |  start_frame: {start_frame}")
            print(f"Label IDs propagated: {sorted_label_ids}")
            header = f"{'frame':>6} | " + " | ".join(f"{h:>{col_w}}" for h in label_headers) + f" | {'total':>10}"
            print(header)
            print("-" * len(header))

            running_totals = {lid: 0 for lid in sorted_label_ids}
            grand_total = 0
            per_frame_counts = {}  # {frame_idx: {label_id: pixel_count}} for history
            for f in sorted(video_segments.keys()):
                per_label_counts = []
                frame_total = 0
                per_frame_counts[f] = {}
                for lid in sorted_label_ids:
                    mask = video_segments[f].get(lid)
                    n = int(mask.sum()) if mask is not None else 0
                    per_label_counts.append(n)
                    per_frame_counts[f][lid] = n
                    running_totals[lid] += n
                    frame_total += n
                grand_total += frame_total
                cells = " | ".join(f"{n:>{col_w},}" for n in per_label_counts)
                print(f"{f:>6} | {cells} | {frame_total:>10,}")

            print("-" * len(header))
            totals_cells = " | ".join(f"{running_totals[lid]:>{col_w},}" for lid in sorted_label_ids)
            print(f"{'TOTAL':>6} | {totals_cells} | {grand_total:>10,}")
            print("=" * 72)

            # Append this run to the session history and re-print the
            # cross-run comparison so the user can compare base vs fine-tuned
            # (or any sequence of runs) without re-running anything.
            self._sam2_run_history.append({
                "model": model_tag,
                "start_frame": start_frame,
                "label_ids": sorted_label_ids,
                "per_frame": per_frame_counts,
            })
            self._print_sam2_session_comparison()

            return output_masks, bool(self._sam2_is_finetuned)

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
