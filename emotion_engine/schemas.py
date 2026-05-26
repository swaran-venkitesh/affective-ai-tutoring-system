from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TARGET_AFFECT_LABELS = [
    "confusion",
    "prolonged_confusion",
    "frustration",
    "boredom",
    "anxiety_self_doubt",
    "engagement",
    "confidence",
    "overload",
]


def _default_scores() -> dict[str, float]:
    return {label: 0.0 for label in TARGET_AFFECT_LABELS}


@dataclass(slots=True)
class EmotionEngineSettings:
    enabled: bool = True
    show_monitor: bool = False
    face_mesh_overlay: bool = False
    allow_client_toggle: bool = False
    show_monitor_ui: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "enabled": bool(self.enabled),
            "show_monitor": bool(self.show_monitor),
            "face_mesh_overlay": bool(self.face_mesh_overlay),
            "allow_client_toggle": bool(self.allow_client_toggle),
            "show_monitor_ui": bool(self.show_monitor_ui),
        }


@dataclass(slots=True)
class ModalityEstimate:
    scores: dict[str, float] = field(default_factory=_default_scores)
    confidence: float = 0.0
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "confidence": float(self.confidence),
            "source": self.source,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class StateConfidence:
    text: float = 0.0
    performance: float = 0.0
    camera: float = 0.0
    face: float = 0.0
    speech: float = 0.0
    fused: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "text": float(self.text),
            "performance": float(self.performance),
            "camera": float(self.camera),
            "face": float(self.face),
            "speech": float(self.speech),
            "fused": float(self.fused),
        }


@dataclass(slots=True)
class FusedAffectState:
    scores: dict[str, float] = field(default_factory=_default_scores)
    dominant_label: str = "engagement"
    dominant_score: float = 0.0
    confidence: StateConfidence = field(default_factory=StateConfidence)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "dominant_label": self.dominant_label,
            "dominant_score": float(self.dominant_score),
            "confidence": self.confidence.to_dict(),
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class TrackedAffectState:
    scores: dict[str, float] = field(default_factory=_default_scores)
    durations_sec: dict[str, float] = field(default_factory=_default_scores)
    trends: dict[str, float] = field(default_factory=_default_scores)
    dominant_label: str = "engagement"
    dominant_score: float = 0.0
    productive_confusion: bool = False
    recovery_score: float = 0.0
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "durations_sec": dict(self.durations_sec),
            "trends": dict(self.trends),
            "dominant_label": self.dominant_label,
            "dominant_score": float(self.dominant_score),
            "productive_confusion": bool(self.productive_confusion),
            "recovery_score": float(self.recovery_score),
            "turn_count": int(self.turn_count),
        }


@dataclass(slots=True)
class EmpathyDecision:
    empathy_needed: bool = False
    empathy_type: str = "none"
    pedagogical_action: str = "normal_explain"
    tone_guidance: str = "clear, calm, concise"
    response_rules: list[str] = field(default_factory=list)
    justification: str = ""
    suppressed_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "empathy_needed": bool(self.empathy_needed),
            "empathy_type": self.empathy_type,
            "pedagogical_action": self.pedagogical_action,
            "tone_guidance": self.tone_guidance,
            "response_rules": list(self.response_rules),
            "justification": self.justification,
            "suppressed_reason": self.suppressed_reason,
        }


@dataclass(slots=True)
class LearnerStateSnapshot:
    confusion: str = "low"
    frustration: str = "low"
    engagement: str = "low"
    attention: str = "medium"
    self_doubt: bool = False
    detected_events: list[str] = field(default_factory=list)
    productive_confusion: bool = False
    recommended_strategy: str = "normal_explain"
    attention_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion": self.confusion,
            "frustration": self.frustration,
            "engagement": self.engagement,
            "attention": self.attention,
            "self_doubt": bool(self.self_doubt),
            "detected_events": list(self.detected_events),
            "productive_confusion": bool(self.productive_confusion),
            "recommended_strategy": self.recommended_strategy,
            "attention_status": self.attention_status,
        }


@dataclass(slots=True)
class EmotionControlPacket:
    emotion_engine_enabled: bool = False
    affect_state: dict[str, float] = field(default_factory=_default_scores)
    state_confidence: StateConfidence = field(default_factory=StateConfidence)
    empathy_needed: bool = False
    empathy_type: str = "none"
    pedagogical_action: str = "normal_explain"
    tone_guidance: str = "clear, calm, concise"
    response_rules: list[str] = field(default_factory=list)
    policy_notes: str = ""
    productive_confusion: bool = False
    learner_state: dict[str, Any] = field(default_factory=dict)
    monitor_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_engine_enabled": bool(self.emotion_engine_enabled),
            "affect_state": dict(self.affect_state),
            "state_confidence": self.state_confidence.to_dict(),
            "empathy_needed": bool(self.empathy_needed),
            "empathy_type": self.empathy_type,
            "pedagogical_action": self.pedagogical_action,
            "tone_guidance": self.tone_guidance,
            "response_rules": list(self.response_rules),
            "policy_notes": self.policy_notes,
            "productive_confusion": bool(self.productive_confusion),
            "learner_state": dict(self.learner_state),
            "monitor_flags": dict(self.monitor_flags),
        }


@dataclass(slots=True)
class MonitorSnapshot:
    settings: EmotionEngineSettings = field(default_factory=EmotionEngineSettings)
    text: ModalityEstimate = field(default_factory=ModalityEstimate)
    performance: ModalityEstimate = field(default_factory=ModalityEstimate)
    camera: ModalityEstimate = field(default_factory=ModalityEstimate)
    face: ModalityEstimate = field(default_factory=ModalityEstimate)
    speech: ModalityEstimate = field(default_factory=ModalityEstimate)
    fused: FusedAffectState = field(default_factory=FusedAffectState)
    tracked: TrackedAffectState = field(default_factory=TrackedAffectState)
    policy: EmpathyDecision = field(default_factory=EmpathyDecision)
    raw_face_emotion: str = "Not available"
    raw_face_confidence: float = 0.0
    raw_face_probabilities: dict[str, float] = field(default_factory=dict)
    tutoring_face_label: str = "Not available"
    tutoring_face_confidence: float = 0.0
    valence: float | None = None
    arousal: float | None = None
    engagement_score: float | None = None
    engagement_label: str = "Not available"
    attention_status: str = "Camera off"
    blink_count: int | None = None
    blink_rate: float | None = None
    yawn_count: int | None = None
    face_present: bool = False
    vlm_state: str = "Not available"
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "text": self.text.to_dict(),
            "performance": self.performance.to_dict(),
            "camera": self.camera.to_dict(),
            "face": self.face.to_dict(),
            "speech": self.speech.to_dict(),
            "fused": self.fused.to_dict(),
            "tracked": self.tracked.to_dict(),
            "policy": self.policy.to_dict(),
            "raw_face_emotion": self.raw_face_emotion,
            "raw_face_confidence": float(self.raw_face_confidence),
            "raw_face_probabilities": dict(self.raw_face_probabilities),
            "tutoring_face_label": self.tutoring_face_label,
            "tutoring_face_confidence": float(self.tutoring_face_confidence),
            "valence": None if self.valence is None else float(self.valence),
            "arousal": None if self.arousal is None else float(self.arousal),
            "engagement_score": None if self.engagement_score is None else float(self.engagement_score),
            "engagement_label": self.engagement_label,
            "attention_status": self.attention_status,
            "blink_count": self.blink_count,
            "blink_rate": None if self.blink_rate is None else float(self.blink_rate),
            "yawn_count": self.yawn_count,
            "face_present": bool(self.face_present),
            "vlm_state": self.vlm_state,
            "landmarks": [[float(x), float(y)] for x, y in self.landmarks],
            "extras": dict(self.extras),
        }


def packet_from_dict(data: dict[str, Any]) -> EmotionControlPacket:
    return EmotionControlPacket(
        emotion_engine_enabled=bool(data.get("emotion_engine_enabled", False)),
        affect_state=dict(data.get("affect_state") or _default_scores()),
        state_confidence=StateConfidence(**(data.get("state_confidence") or {})),
        empathy_needed=bool(data.get("empathy_needed", False)),
        empathy_type=str(data.get("empathy_type", "none")),
        pedagogical_action=str(data.get("pedagogical_action", "normal_explain")),
        tone_guidance=str(data.get("tone_guidance", "clear, calm, concise")),
        response_rules=list(data.get("response_rules") or []),
        policy_notes=str(data.get("policy_notes", "")),
        productive_confusion=bool(data.get("productive_confusion", False)),
        learner_state=dict(data.get("learner_state") or {}),
        monitor_flags=dict(data.get("monitor_flags") or {}),
    )
