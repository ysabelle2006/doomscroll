# Tests for AttentionGate's look-away detection and pause/resume grace period.
from __future__ import annotations

from dataclasses import dataclass

from doomscroll.fixation.attention_gate import AttentionGate, AttentionGateConfig


# Minimal stand-in for HeadPose carrying just yaw/pitch.
@dataclass
class _FakeHeadPose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


# Builds an AttentionGateConfig with sensible defaults, allowing overrides.
def _config(**overrides) -> AttentionGateConfig:
    base = dict(max_yaw_deg=35.0, max_pitch_deg=30.0, pause_grace_ms=500)
    base.update(overrides)
    return AttentionGateConfig(**base)


# Checks a pose within limits stays active (not paused).
def test_within_limits_stays_active():
    gate = AttentionGate(_config())
    paused = gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=0)
    assert paused is False


# Checks the first out-of-range frame starts the grace period but stays active.
def test_first_out_of_range_frame_starts_grace_still_active():
    gate = AttentionGate(_config())
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    assert paused is False


# Checks staying out of range under the grace period keeps it active.
def test_out_of_range_under_grace_period_stays_active():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=300)
    assert paused is False


# Checks staying out of range for the full grace period triggers a pause.
def test_out_of_range_for_full_grace_period_pauses():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)
    assert paused is True


# Checks it remains paused while still out of range after pausing.
def test_already_paused_and_still_out_of_range_remains_paused():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=1000)
    assert paused is True


# Checks it resumes immediately once back in range after being paused.
def test_already_paused_resumes_immediately_once_back_in_range():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=500)
    paused = gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=510)
    assert paused is False



# Checks pitch alone exceeding its limit also counts as looking away.
def test_pitch_alone_exceeding_its_limit_also_counts_as_looking_away():
    gate = AttentionGate(_config(max_yaw_deg=35.0, max_pitch_deg=30.0, pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=0.0, pitch_deg=45.0), timestamp_ms=0)
    paused = gate.update(_FakeHeadPose(yaw_deg=0.0, pitch_deg=45.0), timestamp_ms=500)
    assert paused is True


# Checks a missing head pose counts as looking away.
def test_missing_head_pose_counts_as_looking_away():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(None, timestamp_ms=0)
    paused = gate.update(None, timestamp_ms=500)
    assert paused is True


# Checks a brief in-range frame resets the grace timer.
def test_intermittent_in_range_frame_resets_the_grace_timer():
    gate = AttentionGate(_config(pause_grace_ms=500))
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=0)
    gate.update(_FakeHeadPose(yaw_deg=10.0, pitch_deg=5.0), timestamp_ms=300)
    gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=350)
    paused = gate.update(_FakeHeadPose(yaw_deg=40.0, pitch_deg=0.0), timestamp_ms=800)
    assert paused is False
