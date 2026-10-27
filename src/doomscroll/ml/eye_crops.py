"""THE single definition of "what pixels the CNN sees".

Like ml/features.py for the landmark models: dataset building (offline, from
recorded video) and live inference both call `extract_crops`, so training and
runtime cannot drift apart. If you change any constant here, rebuild the crop
datasets (scripts/build_crops.py --force) and retrain; saved CNN checkpoints
store CROP_SPEC and refuse to load if it no longer matches.

Crops are "normalized" with the MediaPipe landmarks before the CNN sees them:
  - each eye is cut out along the line through its two corners, so in-plane
    head roll is removed and the eye always fills the same part of the crop
    (the iris position INSIDE the crop is then mostly gaze);
  - its size follows the eye width, so leaning closer/further doesn't change
    the crop scale;
  - histogram-equalized, which removes most lighting differences.
Grayscale only: the live loop already has a gray frame, and colour adds
nothing for gaze but differences between rooms.
"""
from __future__ import annotations

import cv2
import numpy as np

# Landmark indices (MediaPipe 478-point mesh), ordered IMAGE-left -> IMAGE-right
# so that "left corner" always means the same thing in the crop.
EYE_CORNERS = (
    (33, 133),   # eye on the left of the image (subject's right eye): outer, inner
    (362, 263),  # eye on the right of the image (subject's left eye): inner, outer
)

EYE_H, EYE_W = 36, 60       # MPIIGaze-style eye patch
EYE_WIDTH_SCALE = 1.7        # crop width = this x corner-to-corner distance
FACE_SIZE = 64               # low-res face patch (head pose / position context)
FACE_SCALE = 1.15            # face crop side = this x landmark bbox size

CROP_SPEC = {
    "eye_hw": [EYE_H, EYE_W], "eye_width_scale": EYE_WIDTH_SCALE, "eye_corners": [list(c) for c in EYE_CORNERS],
    "face_size": FACE_SIZE, "face_scale": FACE_SCALE, "equalize": True, "version": 1,
}


def _warp(gray: np.ndarray, center: np.ndarray, angle_deg: float, scale: float, out_wh: tuple[int, int]) -> np.ndarray:
    """Rotate by angle around `center`, scale, and put `center` in the middle of the output."""
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, scale)
    M[0, 2] += out_wh[0] / 2.0 - center[0]
    M[1, 2] += out_wh[1] / 2.0 - center[1]
    return cv2.warpAffine(gray, M, out_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_eye(gray: np.ndarray, p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    """-> (EYE_H, EYE_W) uint8 patch of one eye, deskewed and scale-normalized."""
    d = p_right - p_left
    width = float(np.hypot(*d))
    if width < 1e-3:
        return np.zeros((EYE_H, EYE_W), np.uint8)
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    center = (p_left + p_right) / 2.0
    patch = _warp(gray, center, angle, EYE_W / (width * EYE_WIDTH_SCALE), (EYE_W, EYE_H))
    return cv2.equalizeHist(patch)


def crop_face(gray: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """-> (FACE_SIZE, FACE_SIZE) uint8 patch of the whole face, roll-corrected."""
    lo, hi = points_px.min(axis=0), points_px.max(axis=0)
    side = float(max(hi - lo)) * FACE_SCALE
    if side < 1e-3:
        return np.zeros((FACE_SIZE, FACE_SIZE), np.uint8)
    (a, _), (_, b) = EYE_CORNERS
    d = points_px[b] - points_px[a]
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    patch = _warp(gray, (lo + hi) / 2.0, angle, FACE_SIZE / side, (FACE_SIZE, FACE_SIZE))
    return cv2.equalizeHist(patch)


def extract_crops(gray: np.ndarray, points_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gray: full grayscale frame; points_px: (478, 2) landmarks in pixels.
    -> eyes (2, EYE_H, EYE_W) uint8 [image-left eye, image-right eye], face (FACE_SIZE, FACE_SIZE) uint8."""
    eyes = np.stack([crop_eye(gray, points_px[a], points_px[b]) for a, b in EYE_CORNERS])
    return eyes, crop_face(gray, points_px)
