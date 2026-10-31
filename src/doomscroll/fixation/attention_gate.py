"""Off-screen / turned-away pause gate.

Deliberately a coarse, high head-pose-angle threshold (not a measure of
gaze-position jitter, which gaze/smoothing.py's Kalman filter already
handles separately), so ordinary reading head movement never approaches
it while an actual "user got up / turned away" event reliably does. A
short grace period avoids pausing on a single missed-detection frame or a
quick glance aside.
"""
from __future__ import annotations

from dataclasses import dataclass

from doomscroll.gaze.head_pose import HeadPose


@dataclass
class AttentionGateConfig:
    max_yaw_deg: float = 35.0
    max_pitch_deg: float = 30.0
    pause_grace_ms: int = 500


class AttentionGate:
    def __init__(self, config: AttentionGateConfig | None = None):
        self.config = config or AttentionGateConfig()
        self._look_away_since_ms: int | None = None
        self._paused = False

    def update(self, head_pose: HeadPose | None, timestamp_ms: int) -> bool:
        """Returns True if the system should PAUSE (no scrolling) this tick."""
        looking_away = (
            head_pose is None
            or abs(head_pose.yaw_deg) > self.config.max_yaw_deg
            or abs(head_pose.pitch_deg) > self.config.max_pitch_deg
        )

        if looking_away:
            if self._look_away_since_ms is None:
                self._look_away_since_ms = timestamp_ms
            elif timestamp_ms - self._look_away_since_ms >= self.config.pause_grace_ms:
                self._paused = True
        else:
            self._look_away_since_ms = None
            self._paused = False

        return self._paused
