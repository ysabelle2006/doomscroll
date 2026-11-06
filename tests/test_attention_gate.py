"""Tests for doomscroll.fixation.attention_gate.AttentionGate -- the
look-away pause gate: coarse yaw/pitch thresholds with an asymmetric
grace period (pausing requires the out-of-range condition to persist for
pause_grace_ms; resuming is immediate). No camera/screen/threading
involved, same style as test_scroll_trigger.py.

Covers the six independent paths (P1-P6) from lab4/test_cases.pdf
section 3.2 (TC-AG-01 through TC-AG-06), plus a few supplementary cases
(pitch-only trigger, a missing head pose, and an intermittent in-range
frame resetting the grace timer) that the manual's six rows don't
individually spell out but the same state machine needs to get right.
"""
from __future__ import annotations

from dataclasses import dataclass

from doomscroll.fixation.attention_gate import AttentionGate, AttentionGateConfig


@dataclass
class _FakeHeadPose:
    """Stand-in for doomscroll.gaze.head_pose.HeadPose -- AttentionGate.update()
    only ever reads yaw_deg/pitch_deg, so a minimal object keeps this test
    free of the real HeadPose's cv2/mediapipe import chain, same reasoning
    as test_scroll_trigger.py's _FakeTracker."""
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


def _config(**overrides) -> AttentionGateConfig:
    base = dict(max_yaw_deg=35.0, max_pitch_deg=30.0, pause_grace_ms=500)
    base.update(overrides)
    return AttentionGateConfig(**base)


# TC-AG-01 (P1): within limits -> stays active.
def test_within_limits_stays_active():
    gate = AttentionGate(_config())
    paused = gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=0)
    assert paused is False


# TC-AG-02 (P2): first frame out of range -> grace timer starts, still active.
def test_first_out_of_range_frame_starts_grace_still_active():
    gate = AttentionGate(_config())
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    assert paused is False


# TC-AG-03 (P3): out of range for less than pause_grace_ms -> still active.
def test_out_of_range_under_grace_period_stays_active():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=300)
    assert paused is False


# TC-AG-04 (P4): out of range for >= pause_grace_ms -> transitions to paused.
def test_out_of_range_for_full_grace_period_pauses():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)
    assert paused is True


# TC-AG-05 (P5): already paused, still out of range -> remains paused.
def test_already_paused_and_still_out_of_range_remains_paused():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)  # now paused
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=1000)
    assert paused is True


# TC-AG-06 (P6): already paused, pose returns within limits -> resumes
# immediately, no grace period on the way back.
def test_already_paused_resumes_immediately_once_back_in_range():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)  # now paused
    paused = gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=510)
    assert paused is False


# --- Supplementary cases beyond the manual's six rows ---------------------

def test_pitch_alone_exceeding_its_limit_also_counts_as_looking_away():
    gate = AttentionGate(_config(max_yaw_deg=35.0, max_pitch_deg=30.0, pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=0.0, pitch_deg=45.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=0.0, pitch_deg=45.0), timestamp_ms=500)
    assert paused is True


def test_missing_head_pose_counts_as_looking_away():
    """No face detected (head_pose=None) should behave exactly like an
    out-of-range pose -- grace period, then pause."""
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(None, timestamp_ms=0)
    paused = gate.update(None, timestamp_ms=500)
    assert paused is True


def test_intermittent_in_range_frame_resets_the_grace_timer():
    """A single back-in-range frame in the middle of an otherwise
    out-of-range stretch must reset _look_away_since_ms, not just leave
    the pause flag alone -- so the grace period has to restart from that
    frame, not from the original look-away."""
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=300)  # one in-range frame
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=350)  # look-away restarts here
    # Only 450ms since the restart at t=350 -- short of the 500ms grace.
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=800)
    assert paused is False
