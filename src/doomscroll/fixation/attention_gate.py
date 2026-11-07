# Pauses scrolling when the user's head pose shows they've looked away.
from __future__ import annotations

from dataclasses import dataclass

from doomscroll.gaze.head_pose import HeadPose


# Thresholds for what counts as looking away, and how long before pausing.
@dataclass
class AttentionGateConfig:
    max_yaw_deg: float = 35.0
    max_pitch_deg: float = 30.0
    pause_grace_ms: int = 500


# Tracks head pose over time to decide whether scrolling should be paused.
class AttentionGate:
    # Stores the gate config and resets look-away tracking state.
    def __init__(self, config: AttentionGateConfig | None = None):
        self.config = config or AttentionGateConfig()
        self._look_away_since_ms: int | None = None
        self._paused = False

    # Updates look-away tracking from the latest head pose; returns whether paused.
    def update(self, head_pose: HeadPose | None, timestamp_ms: int) -> bool:
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
