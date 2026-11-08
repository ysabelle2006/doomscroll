# Tests for calibrate.py's pose-variant and front-facing readout helpers.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.head_pose import HeadPose
from scripts.calibrate import FRONT_YAW_PITCH_TOLERANCE_DEG, PoseVariant, _front_readout, _pose_variant_readout


# Builds a HeadPose with the given yaw/pitch/distance for testing.
def _make_head_pose(yaw_deg: float = 0.0, pitch_deg: float = 0.0, z_mm: float = 400.0) -> HeadPose:
    return HeadPose(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
        rotation_vector=np.zeros((3, 1)),
        translation_vector=np.array([[0.0], [0.0], [z_mm]]),
    )


# Checks a missing face reports out of range regardless of variant.
def test_face_not_detected_reports_out_of_range_regardless_of_variant():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, baseline_z = _pose_variant_readout(None, variant, baseline_z=None)
    assert readout == "face not detected"
    assert in_range is False
    assert baseline_z is None


# Checks a yaw within its variant's range reports in-range.
def test_yaw_variant_reports_in_range_within_bounds():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(yaw_deg=-12.0), variant, baseline_z=None)
    assert in_range is True
    assert "yaw=-12.0" in readout


# Checks a yaw outside its variant's range reports out-of-range.
def test_yaw_variant_reports_out_of_range_outside_bounds():
    variant = PoseVariant("lean LEFT", "lean left", yaw_range=(-20.0, -6.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(yaw_deg=0.0), variant, baseline_z=None)
    assert in_range is False
    assert "yaw=" in readout


# Checks a pitch within its variant's range reports in-range.
def test_pitch_variant_reports_in_range_within_bounds():
    variant = PoseVariant("tilt UP", "tilt up", pitch_range=(6.0, 20.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(pitch_deg=12.0), variant, baseline_z=None)
    assert in_range is True
    assert "pitch=+12.0" in readout


# Checks a pitch outside its variant's range reports out-of-range.
def test_pitch_variant_reports_out_of_range_outside_bounds():
    variant = PoseVariant("tilt UP", "tilt up", pitch_range=(6.0, 20.0))
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(pitch_deg=-5.0), variant, baseline_z=None)
    assert in_range is False


# Checks the first reading for a z-delta variant captures the baseline distance.
def test_z_delta_variant_captures_baseline_on_first_reading():
    variant = PoseVariant("lean CLOSER", "lean closer", z_delta_range=(-100.0, -25.0))
    readout, in_range, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=420.0), variant, baseline_z=None)
    assert baseline_z == 420.0
    assert in_range is False
    assert "distance change=+0mm" in readout


# Checks moving closer after the baseline reports in-range for a "closer" variant.
def test_z_delta_variant_reports_in_range_after_moving_closer():
    variant = PoseVariant("lean CLOSER", "lean closer", z_delta_range=(-100.0, -25.0))
    _, _, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=420.0), variant, baseline_z=None)
    readout, in_range, baseline_z2 = _pose_variant_readout(_make_head_pose(z_mm=370.0), variant, baseline_z=baseline_z)
    assert baseline_z2 == baseline_z
    assert in_range is True
    assert "distance change=-50mm" in readout


# Checks the "sit back" variant uses a positive distance delta.
def test_z_delta_variant_sit_back_uses_positive_delta():
    variant = PoseVariant("sit BACK", "sit back", z_delta_range=(25.0, 100.0))
    _, _, baseline_z = _pose_variant_readout(_make_head_pose(z_mm=400.0), variant, baseline_z=None)
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(z_mm=460.0), variant, baseline_z=baseline_z)
    assert in_range is True
    assert "distance change=+60mm" in readout


# Checks a variant with no range configured is always in range.
def test_variant_with_no_range_set_is_always_in_range():
    variant = PoseVariant("no-op", "no-op")
    readout, in_range, _ = _pose_variant_readout(_make_head_pose(), variant, baseline_z=None)
    assert readout == ""
    assert in_range is True


# Checks every default pose variant has exactly one range type set.
@pytest.mark.parametrize("label", [v for v in ["lean LEFT", "lean RIGHT", "lean CLOSER", "sit BACK", "tilt UP", "tilt DOWN"]])
def test_default_pose_variants_each_have_exactly_one_range_set(label: str):
    from scripts.calibrate import POSE_VARIANTS

    variant = next(v for v in POSE_VARIANTS if v.label == label)
    ranges_set = sum(
        r is not None for r in (variant.yaw_range, variant.pitch_range, variant.z_delta_range)
    )
    assert ranges_set == 1


# Checks a missing face reports "face not detected" for the front readout.
def test_front_readout_face_not_detected():
    readout, straight = _front_readout(None)
    assert readout == "face not detected"
    assert straight is False


# Checks near-zero yaw/pitch reports straight.
def test_front_readout_straight_when_yaw_and_pitch_both_near_zero():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=2.0, pitch_deg=-3.0))
    assert straight is True
    assert "(straight)" in readout
    assert "yaw=+2.0" in readout and "pitch=-3.0" in readout


# Checks yaw beyond tolerance reports not straight.
def test_front_readout_not_straight_when_yaw_exceeds_tolerance():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=FRONT_YAW_PITCH_TOLERANCE_DEG + 1.0, pitch_deg=0.0))
    assert straight is False
    assert "(not quite straight)" in readout


# Checks pitch beyond tolerance reports not straight.
def test_front_readout_not_straight_when_pitch_exceeds_tolerance():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=0.0, pitch_deg=-(FRONT_YAW_PITCH_TOLERANCE_DEG + 1.0)))
    assert straight is False


# Checks yaw/pitch exactly at the tolerance boundary still counts as straight.
def test_front_readout_boundary_tolerance_counts_as_straight():
    readout, straight = _front_readout(_make_head_pose(yaw_deg=FRONT_YAW_PITCH_TOLERANCE_DEG, pitch_deg=FRONT_YAW_PITCH_TOLERANCE_DEG))
    assert straight is True


# Checks the front readout reports distance in millimeters.
def test_front_readout_reports_distance_in_mm():
    readout, _ = _front_readout(_make_head_pose(z_mm=512.0))
    assert "distance=512mm" in readout
