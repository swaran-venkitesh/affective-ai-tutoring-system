from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from emotion_engine.landmarks import AttentionState
from emotion_engine.utils import SmoothedEmotionState, TutoringStateResult, clamp


@dataclass
class EngagementState:
    score: float
    label: str
    rationale: str


@dataclass
class AlertEvent:
    timestamp: float
    source: str
    key: str
    severity: str
    message: str


class EngagementScorer:
    def compute(
        self,
        attention_state: Optional[AttentionState],
        emotion_state: Optional[SmoothedEmotionState],
        tutoring_state: Optional[TutoringStateResult],
    ) -> EngagementState:
        score = 0.58
        reasons: list[str] = []

        if attention_state is None:
            score -= 0.06
            reasons.append("camera_off")
        elif not attention_state.face_present:
            score -= 0.18
            reasons.append("no_face")
        else:
            head_direction = str(attention_state.head_direction or "unknown")
            note_taking_pose = head_direction == "down"
            score += 0.10 * attention_state.face_presence_ratio
            score += 0.10 * attention_state.look_straight_ratio
            score -= min(0.26, attention_state.current_eye_closure_duration * 0.22)
            score -= max(0.0, attention_state.blink_rate_per_min - 24.0) * 0.005
            score -= 0.10 if attention_state.yawn_active else 0.0

            if attention_state.prolonged_eye_closure:
                score -= 0.20
                reasons.append("eyes_closed")

            if attention_state.looking_away:
                if note_taking_pose and attention_state.current_away_duration < 4.0:
                    score -= 0.04
                    reasons.append("note_taking")
                else:
                    score -= min(0.24, attention_state.current_away_duration * 0.10)
                    reasons.append("looking_away")
            elif note_taking_pose:
                score -= 0.02
                reasons.append("note_taking")

        if emotion_state is not None:
            score += 0.08 * max(emotion_state.valence, 0.0)
            if emotion_state.arousal < -0.25:
                score -= 0.08
                reasons.append("low_arousal")

        if tutoring_state is not None:
            label = tutoring_state.label
            if label in {"engaged", "happy"}:
                score += 0.15
                reasons.append(label)
            elif label in {"neutral"}:
                score += 0.04
            elif label in {"bored", "anxious", "frustrated", "sad", "fear", "confused"}:
                score -= 0.14
                reasons.append(label)
            elif label == "no_face":
                score -= 0.12

        score = clamp(score, 0.0, 1.0)
        if score >= 0.68:
            label = "engaged"
        elif score >= 0.40:
            label = "monitor"
        else:
            label = "disengaged"

        rationale = ", ".join(reasons[:4]) if reasons else "stable"
        return EngagementState(score=score, label=label, rationale=rationale)


class AlertManager:
    def __init__(self) -> None:
        self.cooldowns = {
            "eyes_closed": 18.0,
            "looking_away": 15.0,
            "looking_away_long": 12.0,
            "blink_burst": 30.0,
            "yawn": 30.0,
            "no_face": 15.0,
            "phone": 25.0,
            "phone_call": 15.0,
            "phone_active_use": 20.0,
            "multi_person": 25.0,
        }
        self.last_alert_times: dict[str, float] = {}

    def _emit(self, key: str, source: str, severity: str, message: str, now: float) -> Optional[AlertEvent]:
        cooldown = self.cooldowns.get(key, 15.0)
        if now - self.last_alert_times.get(key, 0.0) < cooldown:
            return None
        self.last_alert_times[key] = now
        return AlertEvent(timestamp=now, source=source, key=key, severity=severity, message=message)

    def evaluate_fast(
        self,
        attention_state: Optional[AttentionState],
        engagement_state: Optional[EngagementState],
        tutoring_state: Optional[TutoringStateResult],
        timestamp: Optional[float] = None,
    ) -> list[AlertEvent]:
        now = timestamp if timestamp is not None else time.time()
        events: list[AlertEvent] = []
        if attention_state is None:
            return events

        if not attention_state.face_present and attention_state.face_absent_duration >= 3.0:
            event = self._emit(
                "no_face",
                "landmarks",
                "medium",
                "Face not visible for a while. A quick camera-facing check may help.",
                now,
            )
            if event:
                events.append(event)

        if attention_state.prolonged_eye_closure and attention_state.current_eye_closure_duration >= 1.5:
            event = self._emit(
                "eyes_closed",
                "landmarks",
                "high",
                "Eyes appear closed for an extended moment. Consider a quick focus check.",
                now,
            )
            if event:
                events.append(event)

        note_taking_pose = str(attention_state.head_direction or "unknown") == "down"
        if attention_state.current_away_duration >= 2.5 and not note_taking_pose:
            event = self._emit(
                "looking_away",
                "landmarks",
                "medium",
                "Attention seems away from the screen. A gentle re-engagement prompt may help.",
                now,
            )
            if event:
                events.append(event)

        if attention_state.current_away_duration >= 5.5 and not note_taking_pose:
            event = self._emit(
                "looking_away_long",
                "landmarks",
                "high",
                "Attention has stayed away from the screen for several seconds. A stronger focus alert is justified.",
                now,
            )
            if event:
                events.append(event)

        if attention_state.blink_rate_per_min >= 34.0:
            event = self._emit(
                "blink_burst",
                "landmarks",
                "low",
                "Blink rate is elevated. Marking a lower-engagement risk for this period.",
                now,
            )
            if event:
                events.append(event)

        if attention_state.yawn_active:
            event = self._emit(
                "yawn",
                "landmarks",
                "low",
                "A yawn-like mouth pattern is visible. Energy may be dropping.",
                now,
            )
            if event:
                events.append(event)

        if engagement_state is not None and engagement_state.label == "disengaged":
            if tutoring_state is not None and tutoring_state.label in {"bored", "confused", "anxious", "frustrated"}:
                event = self._emit(
                    f"emotion_{tutoring_state.label}",
                    "affect",
                    "low",
                    f"The affective state trends toward {tutoring_state.label}. A short check-in may help.",
                    now,
                )
                if event:
                    events.append(event)

        return events

    def evaluate_vlm(self, observation, timestamp: Optional[float] = None) -> list[AlertEvent]:
        now = timestamp if timestamp is not None else time.time()
        events: list[AlertEvent] = []

        if observation is None:
            return events

        if observation.phone_present:
            phone_message = "Phone detected in the scene. Logging a possible distraction event."
            phone_key = "phone"
            phone_severity = "medium"
            if getattr(observation, "phone_activity", "") == "on_call":
                phone_key = "phone_call"
                phone_severity = "high"
                phone_message = "Phone appears to be at the ear or in an active call posture."
            elif getattr(observation, "phone_activity", "") in {"typing", "scrolling", "browsing", "video"}:
                phone_key = "phone_active_use"
                phone_message = f"Phone appears to be in active use ({observation.phone_activity})."
            event = self._emit(
                phone_key,
                "vlm",
                phone_severity,
                phone_message,
                now,
            )
            if event:
                events.append(event)

        if observation.person_count > 1:
            event = self._emit(
                "multi_person",
                "vlm",
                "medium",
                "More than one person appears in view. Logging a shared-scene event.",
                now,
            )
            if event:
                events.append(event)

        return events
