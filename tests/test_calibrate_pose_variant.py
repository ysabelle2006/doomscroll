"""Tests for scripts/calibrate.py's self-paced pose-variant readout
(_pose_variant_readout / PoseVariant / wait_for_pose_ready) -- the
replacement for a fixed-duration show_transition_message() during the
head-pose variation pass. See calibrate.py's module docstring and
wait_for_pose_ready()'s docstring for the rationale.

These test the pure readout/range-check logic directly (no camera, no cv2
GUI window needed -- scripts/calibrate.py is importable on its own since
nothing opens a camera or window at import time, only inside main()).
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.head_pose import HeadPose
from scripts.calibrate import FRONT_YAW_PITCH_TOLERANCE_DEG, PoseVariant, _front_readout, _pose_variant_readout


def _make_head_pose(yaw_deg: float = 0.0, pitch_deg: float = 0.0, z_mm: float = 400.0) -> HeadPose:
    return HeadPose(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
        rotation_vector=np.zeros((3, 1)),
        translation_vector=np.array([[0.0], [0.0], [z_mm]]),
    )


def test_face_not_detected_reports_out_of_range_regardless_of_variant():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, baseline_z = _pose_variant_readout(None, variant, baseline_z=None)
    assert readout == "face not detected"
    assert in_range is False
    assert baseline_z is None


def test_yaw_variant_reports_in_range_within_bounds():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(yaw_deg=-12.0), variant, baseline_z=None)
    assert in_range is True
    assert "yaw=-12.0" in readout


def test_yaw_variant_reports_out_of_range_outside_bounds():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(yaw_deg=0.0), variant, baseline_z=None)
    assert in_range is False
    assert "yaw=" in readout


def test_pitch_variant_reports_in_range_within_bounds():
    variant = PoseVariant("tilt UP", "tilt up", pitch_range=(6.0, 20.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(pitch_deg=12.0), variant, baseline_z=None)
    assert in_range is True
    assert "pitch=+12.0" in readout


def test_pitch_variant_reports_out_of_range_outside_bounds():
    variant = PoseVariant("tilt UP", "tilt up", pitch_range=(6.0, 20.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(pitch_deg=-5.0), variant, baseline_z=None)
    assert in_range is False


def test_z_delta_variant_captures_baseline_on_first_reading():
    variant = PoseVariant("lean CLOSER", "lean closer", z_delta_range=(-100.0, -25.0))
    readout, in_range, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=420.0), variant, baseline_z=None)
    # First reading establishes the baseline -- delta is 0, so not yet "in range"
    assert baseline_z == 420.0
    assert in_range is False
    assert "distance change=+0mm" in readout


def test_z_delta_variant_reports_in_range_after_moving_closer():
    variant = PoseVariant("lean CLOSER", "lean closer", z_delta_range=(-100.0, -25.0))
    _, _, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=420.0), variant, baseline_z=None)
    readout, in_range, baseline_z2 = _pose_variant_readout(_make_head_pose(z_mm=370.0), variant, baseline_z=baseline_z)
    assert baseline_z2 == baseline_z  # baseline doesn't drift on subsequent calls
    assert in_range is True
    assert "distance change=-50mm" in readout


def test_z_delta_variant_sit_back_uses_positive_delta():
    variant = PoseVariant("sit BACK", "sit back", z_delta_range=(25.0, 100.0))
    _, _, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=400.0), variant, baseline_z=None)
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(z_mm=460.0), variant, baseline_z=baseline_z)
    assert in_range is True
    assert "distance change=+60mm" in readout


def test_variant_with_no_range_set_is_always_in_range():
    variant = PoseVariant("no-op", "no-op")
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(), variant, baseline_z=None)
    assert readout == ""
    assert in_range is True


@pytest.mark.parametrize("label", [v for v in ["lean LEFT", "lean RIGHT", "lean CLOSER", "sit BACK", "tilt UP", "tilt DOWN"]])
def test_default_pose_variants_each_have_exactly_one_range_set(label: str):
    """Sanity check on the actual POSE_VARIANTS list shipped in
    calibrate.py: every variant should define exactly one of
    yaw_range/pitch_range/z_delta_range, so _pose_variant_readout always has
    an unambiguous signal to show."""
    from scripts.calibrate import POSE_VARIANTS

    variant = next(v for v in POSE_VARIANTS if v.label == label)
    ranges_set = sum(
        r is not None for r in (variant.yaw_range, variant.pitch_range, variant.z_delta_range)
    )
    assert ranges_set == 1


def test_front_readout_face_not_detected():
    readout, straight = _front_readout(None)
    assert readout == "face not detected"
    assert straight is False


def test_front_readout_straight_when_yaw_and_pitch_both_near_zero():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=2.0, pitch_deg=-3.0))
    assert straight is True
    assert "(straight)" in readout
    assert "yaw=+2.0" in readout and "pitch=-3.0" in readout


def test_front_readout_not_straight_when_yaw_exceeds_tolerance():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=FRONT_YAW_PITCH_TOLERANCE_DEG + 1.0, pitch_deg=0.0))
    assert straight is False
    assert "(not quite straight)" in readout


def test_front_readout_not_straight_when_pitch_exceeds_tolerance():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=0.0, pitch_deg=-(FRONT_YAW_PITCH_TOLERANCE_DEG + 1.0)))
    assert straight is False


def test_front_readout_boundary_tolerance_counts_as_straight():
    """Exactly at the tolerance boundary should still count as straight --
    <= not <, matching _pose_variant_readout's own inclusive range checks."""
    readout, straight = _front_readout(_make_head_pose(yaw_deg=FRONT_YAW_PITCH_TOLERANCE_DEG, pitch_deg=FRONT_YAW_PITCH_TOLERANCE_DEG))
    assert straight is True


def test_front_readout_reports_distance_in_mm():
    readout, _ = _front_readout(_make_head_pose(z_mm=512.0))
    assert "distance=512mm" in readout
