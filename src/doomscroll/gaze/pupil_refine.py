"""Sub-pixel iris-center refinement via classical image processing, applied
on top of MediaPipe's landmark-based iris center.

Idea (not code) borrowed from how hybrid webcam gaze trackers commonly
refine landmark-level pupil localization with a dedicated per-eye image
analysis step, rather than trusting a general face-mesh model's iris
landmark alone -- publicly-documented trackers like EyeGestures take this
kind of layered approach, though this implementation is entirely our own
(classical thresholding/contours via OpenCV, no external gaze library).

MediaPipe's iris landmarks are already good, but they're one general-purpose
model's prediction with no per-frame local sanity check. A lightweight
threshold-based blob centroid computed directly from the current eye crop
is an independent error source -- averaging it with the landmark position
tends to cancel out some of each method's independent noise, rather than
compounding it.

This never fully replaces the MediaPipe landmark: if the local refinement
disagrees too much (likely an eyelash/shadow/blink artifact, or a keratin
highlight from the "catch light" reflection) or finds no plausible dark
blob at all, it falls back to the landmark position unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

# How far a refined center may disagree with MediaPipe's landmark (in units
# of eye width) before it's treated as noise and discarded rather than
# blended in.
MAX_DISAGREEMENT_EYE_WIDTHS = 0.35

# How much of the refined position to blend in when it passes the sanity
# check -- kept well under 1.0 so a single bad frame's refinement can only
# nudge the result, never fully override the landmark-based estimate.
BLEND_WEIGHT = 0.4

# Crop half-height as a fraction of half-width -- eyes are wider than tall.
CROP_ASPECT = 0.7


def _crop_eye_region(
    gray: np.ndarray, eye_center_px: np.ndarray, eye_width: float
) -> tuple[np.ndarray, np.ndarray] | None:
    half_w = max(int(eye_width * 0.6), 6)
    half_h = max(int(half_w * CROP_ASPECT), 5)
    x0, y0 = int(eye_center_px[0] - half_w), int(eye_center_px[1] - half_h)
    x1, y1 = int(eye_center_px[0] + half_w), int(eye_center_px[1] + half_h)
    h, w = gray.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return gray[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float64)


def refine_iris_center(
    gray: np.ndarray,
    mediapipe_iris_px: np.ndarray,
    eye_outer_px: np.ndarray,
    eye_inner_px: np.ndarray,
) -> np.ndarray:
    """Returns a possibly-refined iris center in the same pixel coordinate
    frame as mediapipe_iris_px. Falls back to mediapipe_iris_px unchanged
    whenever refinement isn't confidently better -- see module docstring."""
    eye_width = float(np.linalg.norm(eye_outer_px - eye_inner_px))
    if eye_width < 4:
        return mediapipe_iris_px

    crop_info = _crop_eye_region(gray, mediapipe_iris_px, eye_width)
    if crop_info is None:
        return mediapipe_iris_px
    crop, origin = crop_info

    crop_eq = cv2.equalizeHist(crop)
    _thresh_val, binary = cv2.threshold(crop_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mediapipe_iris_px

    # Prefer the contour whose centroid is closest to MediaPipe's own
    # landmark position -- treats the landmark as a strong prior on *where*
    # the pupil should be, rather than picking "the single biggest dark
    # blob" (often an eyebrow shadow or eyelash cluster instead).
    prior_local = mediapipe_iris_px - origin
    best_centroid = None
    best_dist = None
    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] < 1e-3:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dist = float(np.hypot(cx - prior_local[0], cy - prior_local[1]))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_centroid = np.array([cx, cy], dtype=np.float64)

    if best_centroid is None:
        return mediapipe_iris_px

    refined_px = best_centroid + origin
    disagreement = float(np.linalg.norm(refined_px - mediapipe_iris_px)) / eye_width
    if disagreement > MAX_DISAGREEMENT_EYE_WIDTHS:
        return mediapipe_iris_px  # likely noise -- distrust and fall back

    return (1 - BLEND_WEIGHT) * mediapipe_iris_px + BLEND_WEIGHT * refined_px
