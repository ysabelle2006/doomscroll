# Crops and normalizes eye/face image patches for the CNN model's inputs.

from __future__ import annotations

import cv2
import numpy as np

EYE_CORNERS = (
    (33, 133),
    (362, 263),
)

EYE_H, EYE_W = 36, 60
EYE_WIDTH_SCALE = 1.7
FACE_SIZE = 64
FACE_SCALE = 1.15

# Describes the crop geometry, saved alongside the CNN model so inference matches training.
CROP_SPEC = {
    "eye_hw": [EYE_H, EYE_W], "eye_width_scale": EYE_WIDTH_SCALE, "eye_corners": [list(c) for c in EYE_CORNERS],
    "face_size": FACE_SIZE, "face_scale": FACE_SCALE, "equalize": True, "version": 1,
}


# Rotates/scales/translates a grayscale image into a fixed-size output patch.
def _warp(gray: np.ndarray, center: np.ndarray, angle_deg: float, scale: float, out_wh: tuple[int, int]) -> np.ndarray:
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, scale)
    M[0, 2] += out_wh[0] / 2.0 - center[0]
    M[1, 2] += out_wh[1] / 2.0 - center[1]
    return cv2.warpAffine(gray, M, out_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# Crops and aligns one eye, leveled by the corner-to-corner angle.
def crop_eye(gray: np.ndarray, p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    d = p_right - p_left
    width = float(np.hypot(*d))
    if width < 1e-3:
        return np.zeros((EYE_H, EYE_W), np.uint8)
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    center = (p_left + p_right) / 2.0
    patch = _warp(gray, center, angle, EYE_W / (width * EYE_WIDTH_SCALE), (EYE_W, EYE_H))
    return cv2.equalizeHist(patch)


# Crops and aligns the face, leveled by the eye-corner angle.
def crop_face(gray: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    lo, hi = points_px.min(axis=0), points_px.max(axis=0)
    side = float(max(hi - lo)) * FACE_SCALE
    if side < 1e-3:
        return np.zeros((FACE_SIZE, FACE_SIZE), np.uint8)
    (a, _), (_, b) = EYE_CORNERS
    d = points_px[b] - points_px[a]
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    patch = _warp(gray, (lo + hi) / 2.0, angle, FACE_SIZE / side, (FACE_SIZE, FACE_SIZE))
    return cv2.equalizeHist(patch)


# Extracts both eye crops and the face crop for one frame.
def extract_crops(gray: np.ndarray, points_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eyes = np.stack([crop_eye(gray, points_px[a], points_px[b]) for a, b in EYE_CORNERS])
    return eyes, crop_face(gray, points_px)
