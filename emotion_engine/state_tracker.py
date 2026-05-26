from __future__ import annotations

import time
from collections import deque

from .config import EmotionEngineConfig
from .schemas import FusedAffectState, TARGET_AFFECT_LABELS, TrackedAffectState


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class AffectStateTracker:
    def __init__(self, config: EmotionEngineConfig) -> None:
        self.config = config
        self.history: deque[tuple[float, dict[str, float]]] = deque(maxlen=24)
        self.ema_scores = {label: 0.0 for label in TARGET_AFFECT_LABELS}
        self.previous_scores = {label: 0.0 for label in TARGET_AFFECT_LABELS}
        self.active_since = {label: None for label in TARGET_AFFECT_LABELS}
        self.turn_count = 0
        self.last_timestamp: float | None = None

    def update(self, fused: FusedAffectState, timestamp: float | None = None) -> TrackedAffectState:
        now = timestamp if timestamp is not None else time.time()
        self.turn_count += 1
        if self.turn_count == 1:
            alpha = 1.0
        elif self.turn_count < 4:
            alpha = 0.55
        else:
            alpha = 0.40
        for label in TARGET_AFFECT_LABELS:
            current = float(fused.scores.get(label, 0.0))
            self.ema_scores[label] = (alpha * current) + ((1.0 - alpha) * self.ema_scores[label])
            if self.ema_scores[label] >= self.config.moderation_threshold:
                if self.active_since[label] is None:
                    self.active_since[label] = now
            else:
                self.active_since[label] = None

        durations = {
            label: 0.0 if self.active_since[label] is None else max(0.0, now - float(self.active_since[label]))
            for label in TARGET_AFFECT_LABELS
        }

        if durations["confusion"] >= self.config.prolonged_confusion_seconds:
            self.ema_scores["prolonged_confusion"] = max(
                self.ema_scores["prolonged_confusion"],
                _clamp(0.55 + 0.35 * min(1.0, durations["confusion"] / (self.config.prolonged_confusion_seconds * 2.0))),
            )
        else:
            self.ema_scores["prolonged_confusion"] = max(
                self.ema_scores["prolonged_confusion"] * 0.92,
                fused.scores.get("prolonged_confusion", 0.0),
            )

        trends = {
            label: float(self.ema_scores[label] - self.previous_scores[label])
            for label in TARGET_AFFECT_LABELS
        }
        self.previous_scores = dict(self.ema_scores)
        self.history.append((now, dict(self.ema_scores)))

        productive_confusion = bool(
            self.ema_scores["confusion"] >= self.config.productive_confusion_threshold
            and self.ema_scores["prolonged_confusion"] < 0.52
            and durations["confusion"] < (self.config.prolonged_confusion_seconds * 0.45)
            and self.ema_scores["frustration"] < 0.45
            and self.ema_scores["anxiety_self_doubt"] < 0.45
            and self.ema_scores["overload"] < 0.45
            and self.ema_scores["engagement"] >= 0.42
        )

        if len(self.history) >= 2:
            negative_now = (
                self.ema_scores["frustration"]
                + self.ema_scores["anxiety_self_doubt"]
                + self.ema_scores["overload"]
            ) / 3.0
            negative_then = (
                self.history[0][1]["frustration"]
                + self.history[0][1]["anxiety_self_doubt"]
                + self.history[0][1]["overload"]
            ) / 3.0
            recovery_score = _clamp(negative_then - negative_now + (self.ema_scores["confidence"] * 0.25))
        else:
            recovery_score = _clamp(self.ema_scores["confidence"] * 0.25)

        dominant_label, dominant_score = max(self.ema_scores.items(), key=lambda item: item[1])
        self.last_timestamp = now
        return TrackedAffectState(
            scores=dict(self.ema_scores),
            durations_sec=durations,
            trends=trends,
            dominant_label=dominant_label,
            dominant_score=float(dominant_score),
            productive_confusion=productive_confusion,
            recovery_score=float(recovery_score),
            turn_count=self.turn_count,
        )
