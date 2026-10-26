"""Live, per-frame L2CS-Net gaze estimation. An earlier offline comparison
found that swapping the hand-crafted 4-dim eye-in-head vector for L2CS-Net's
pretrained, Gaze360-trained (yaw, pitch) estimate gives a real within-person
accuracy win on higher-resolution test screens, at worst a wash on lower-
resolution ones (see the README's Gaze feature backends section). This
module is what lets scripts/calibrate.py and scripts/phase2_gaze_scroll.py
actually use that backend live (via their --features l2cs flag) instead of
only as an offline research comparison.

Both calibrate.py and phase2_gaze_scroll.py already run MediaPipe once per
live frame for head pose and blink detection -- so
L2CSGazeEstimator.extract_from_frame() takes the already-detected
`points_px` landmarks for face cropping instead of detecting them again,
which keeps this to one landmark detection per frame rather than two, and
keeps crop geometry consistent with whatever head-pose estimation already
saw.

Heavier dependency, deliberately isolated: torch/torchvision/safetensors and
the vendored ml/third_party/L2CS-Net repo are NOT in this project's default
dependencies (see pyproject.toml's "l2cs" extra) -- importing this module at
all requires them, so every caller (calibrate.py, phase2_gaze_scroll.py) only
imports it lazily, inside the branch where --features l2cs was actually
chosen. The default --features handcrafted path never touches this file or
needs torch installed at all.

Live-performance caveat (can't be verified from Claude's sandbox): a ResNet50
forward pass per frame is real work. _pick_device() prefers Apple Silicon's
"mps" backend over "cpu" when available, which should be far faster than a
CPU-only run -- but actual live framerate with this in the capture loop needs
checking on a real Mac with a real camera, which Claude cannot do itself.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
L2CS_NET_PATH = REPO_ROOT / "ml" / "third_party" / "L2CS-Net"
WEIGHTS_PATH = REPO_ROOT / "models" / "l2cs_gaze360_resnet50.safetensors"

NUM_BINS = 90
FACE_CROP_PAD_FRAC = 0.25  # extra margin around the landmark bounding box, as a fraction of its width/height
L2CS_FEATURE_DIM = 2  # (yaw, pitch) -- see calibration.py's FEATURE_DIM_L2CS (this + 6-dim head pose = 8)


def face_crop_from_points(frame_bgr: np.ndarray, points_px: np.ndarray, pad_frac: float = FACE_CROP_PAD_FRAC) -> np.ndarray | None:
    """Crops frame_bgr to a padded bounding box around points_px (already-
    detected MediaPipe landmarks for this frame). Returns None if the
    landmarks don't yield a usable box (shouldn't normally happen given a
    non-None detection result, but a degenerate/clamped-to-nothing box is
    cheap to guard against explicitly rather than let a 0-size crop reach
    cv2.resize()). Pure numpy/cv2 -- no torch, so this is testable without
    the L2CS-Net model or weights.
    """
    x_min, y_min = points_px.min(axis=0)
    x_max, y_max = points_px.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    x_min -= w * pad_frac
    x_max += w * pad_frac
    y_min -= h * pad_frac
    y_max += h * pad_frac
    H, W = frame_bgr.shape[:2]
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(W, int(x_max)), min(H, int(y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return frame_bgr[y_min:y_max, x_min:x_max]


def _pick_device(torch_module):
    """Prefers Apple Silicon's MPS backend (torch 2.x on arm64 macOS) over
    CPU -- a ResNet50 forward pass per live frame is squarely the kind of
    workload MPS exists for, and CPU-only is a real bottleneck even for
    offline, non-real-time use. Falls back to CUDA then CPU on other
    platforms."""
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


class L2CSGazeEstimator:
    """Loads L2CS-Net (ResNet50, Gaze360-pretrained) once, then predicts
    (yaw, pitch) in radians per face crop, following the exact resize and
    normalization that the offline comparison motivating this backend (see
    the README's Gaze feature backends section) validated against -- any
    drift here would silently invalidate that validation for the live
    path.

    Constructing this instantiates torch and loads the model immediately
    (it's meant to be built once, up front, not per-frame) -- so construct
    it only when --features l2cs is actually selected, never unconditionally
    at module import time.
    """

    def __init__(self, weights_path: Path | str = WEIGHTS_PATH, l2cs_net_path: Path | str = L2CS_NET_PATH):
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "L2CSGazeEstimator needs torch, torchvision and safetensors installed -- "
                "run `uv sync --extra l2cs` (or `pip install .[l2cs]`) first. See pyproject.toml."
            ) from exc

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Expected L2CS-Net (Gaze360-pretrained ResNet50) weights, as a safetensors file, "
                f"at {weights_path}."
            )
        l2cs_net_path = Path(l2cs_net_path)
        if not l2cs_net_path.exists():
            raise FileNotFoundError(
                f"Expected the vendored L2CS-Net repo (https://github.com/Ahmednull/L2CS-Net) at "
                f"{l2cs_net_path}."
            )

        if str(l2cs_net_path) not in sys.path:
            sys.path.insert(0, str(l2cs_net_path))

        # l2cs/__init__.py unconditionally
        # imports its own Pipeline class, which needs the separate
        # `face_detection` (RetinaFace) package -- not a dependency here, and
        # deliberately not wanted (this module does its own MediaPipe-based
        # cropping, see face_crop_from_points() above, and RetinaFace would
        # additionally try to download its own pretrained weights over the
        # network at runtime). Stub the module out so `import l2cs` succeeds
        # without it.
        if "face_detection" not in sys.modules:
            _face_detection_stub = types.ModuleType("face_detection")

            class _UnusedRetinaFace:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "face_detection.RetinaFace is a stub in l2cs_backend.py -- it is never "
                        "meant to be instantiated (MediaPipe is used for face cropping instead). "
                        "If you see this, something imported l2cs.Pipeline unexpectedly."
                    )

            _face_detection_stub.RetinaFace = _UnusedRetinaFace
            sys.modules["face_detection"] = _face_detection_stub

        from l2cs import getArch
        from l2cs.utils import transformations

        self._torch = torch
        self._transformations = transformations
        self._device = _pick_device(torch)

        model = getArch("ResNet50", NUM_BINS)
        state_dict = load_file(str(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"L2CS-Net weights at {weights_path} don't match the expected ResNet50/{NUM_BINS}-bin "
                f"architecture (missing={missing}, unexpected={unexpected}) -- stopping rather than "
                "predicting from a partially-loaded model."
            )
        model.eval()
        model.to(self._device)
        self._model = model
        self._idx_tensor = torch.arange(NUM_BINS, dtype=torch.float32, device=self._device)

    @property
    def device(self) -> str:
        return str(self._device)

    def predict(self, face_bgr: np.ndarray) -> tuple[float, float]:
        """face_bgr: an already-cropped face image (e.g. from
        face_crop_from_points()). Returns (yaw, pitch) in RADIANS -- see
        calibration.py's feature_backend tag, which keeps profiles from
        different sessions/cameras from being silently pooled together
        regardless of unit convention."""
        torch = self._torch
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        with torch.no_grad():
            img = self._transformations(face_rgb).unsqueeze(0).to(self._device)
            yaw_logits, pitch_logits = self._model(img)
            yaw_bins = torch.softmax(yaw_logits, dim=1)
            pitch_bins = torch.softmax(pitch_logits, dim=1)
            yaw_deg = float((yaw_bins[0] * self._idx_tensor).sum()) * 4 - 180
            pitch_deg = float((pitch_bins[0] * self._idx_tensor).sum()) * 4 - 180
        return float(np.deg2rad(yaw_deg)), float(np.deg2rad(pitch_deg))

    def extract_from_frame(self, frame_bgr: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
        """One-call convenience: crop + predict. Returns a (2,) array
        [yaw, pitch] in radians, or None if no usable crop could be made
        (treated as a "no face" candidate rather than raising)."""
        crop = face_crop_from_points(frame_bgr, points_px)
        if crop is None or crop.size == 0:
            return None
        yaw, pitch = self.predict(crop)
        return np.array([yaw, pitch], dtype=np.float64)
