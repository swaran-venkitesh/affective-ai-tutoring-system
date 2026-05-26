from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any

from .camera_affect import CameraAffectEstimator
from .config import EmotionEngineConfig
from .empathy_policy import EmpathyPolicyEngine
from .fusion import AffectFusionEngine
from .llm_conditioning import LLMConditioner
from .performance_affect import PerformanceAffectEstimator
from .schemas import (
    EmotionControlPacket,
    EmotionEngineSettings,
    EmpathyDecision,
    FusedAffectState,
    LearnerStateSnapshot,
    ModalityEstimate,
    MonitorSnapshot,
    StateConfidence,
    TrackedAffectState,
)
from .settings_store import SettingsStore
from .speech_affect import SpeechAffectEstimator
from .state_tracker import AffectStateTracker
from .text_affect import TextAffectClassifier


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class EmotionEngine:
    def __init__(self, config: EmotionEngineConfig | None = None, socketio=None, log_callback=None) -> None:
        self.config = config or EmotionEngineConfig()
        self.socketio = socketio
        self.log_callback = log_callback or (lambda _msg: None)
        self.settings_store = SettingsStore(self.config.settings_path, self.config)
        self.text = None
        self.performance = PerformanceAffectEstimator()
        self.camera = None
        self.speech = SpeechAffectEstimator()
        self.fusion = AffectFusionEngine(self.config)
        self.state_tracker = AffectStateTracker(self.config)
        self.policy_engine = EmpathyPolicyEngine(self.config)
        self.conditioner = LLMConditioner()
        self.user_history: deque[str] = deque(maxlen=6)
        self.recent_events: deque[tuple[float, str]] = deque(maxlen=16)
        self._camera_lock = threading.Lock()

        self.latest_text = ModalityEstimate(source="text")
        self.latest_performance = ModalityEstimate(source="performance")
        self.latest_speech = self.speech.analyze()
        self.latest_fused = FusedAffectState()
        self.latest_tracked = TrackedAffectState()
        self.latest_policy = EmpathyDecision()
        self.latest_packet = EmotionControlPacket()
        self.latest_learner_state = LearnerStateSnapshot()
        self.latest_monitor = MonitorSnapshot(settings=self.settings_store.get())
        initial_settings = self.get_settings()
        if initial_settings.enabled:
            self._ensure_text_runtime()
        if initial_settings.show_monitor or initial_settings.face_mesh_overlay:
            self._ensure_camera_runtime()

    def _ensure_text_runtime(self) -> None:
        if self.text is not None:
            return
        self.text = TextAffectClassifier(self.config)

    def _ensure_camera_runtime(self) -> None:
        if self.camera is not None:
            return
        self.camera = CameraAffectEstimator(self.config, log_callback=self.log_callback)

    def get_settings(self) -> EmotionEngineSettings:
        return self.settings_store.get()

    def update_settings(self, **kwargs) -> EmotionEngineSettings:
        settings = self.settings_store.update(**kwargs)
        if settings.enabled:
            self._ensure_text_runtime()
        self.latest_monitor.settings = settings
        self.emit_status()
        self.emit_monitor_update()
        return settings

    def _disabled_packet(self) -> EmotionControlPacket:
        settings = self.get_settings()
        return EmotionControlPacket(
            emotion_engine_enabled=False,
            affect_state=dict(self.latest_tracked.scores),
            state_confidence=StateConfidence(),
            empathy_needed=False,
            empathy_type="none",
            pedagogical_action="normal_explain",
            tone_guidance="clear, calm, concise",
            response_rules=["preserve existing tutor behavior"],
            policy_notes="Emotion Engine disabled.",
            productive_confusion=False,
            learner_state=self.latest_learner_state.to_dict(),
            monitor_flags=settings.to_dict(),
        )

    def _score_level(self, score: float) -> str:
        score = float(score or 0.0)
        if score >= 0.60:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def _current_attention_status(self) -> str:
        status = str((self.latest_monitor.attention_status if self.latest_monitor else "") or "")
        if status and status != "Camera off":
            return status
        if self.camera is not None:
            details = getattr(self.camera.latest_camera_estimate, "details", {}) or {}
            fallback = str(details.get("effective_attention") or "")
            if fallback:
                return fallback
        return "unknown"

    def _attention_level(self, status: str) -> str:
        low_states = {"away", "sleepy", "phone"}
        medium_states = {"distracted_side", "multiple_people", "unknown", "camera off"}
        status = str(status or "unknown").strip().lower()
        if status in low_states:
            return "low"
        if status in medium_states:
            return "medium"
        return "high"

    def _active_events(self, now: float | None = None) -> list[str]:
        ref = now if now is not None else time.time()
        active: list[str] = []
        while self.recent_events and (ref - float(self.recent_events[0][0])) > 120.0:
            self.recent_events.popleft()
        for _ts, name in self.recent_events:
            if name not in active:
                active.append(name)

        attention_status = self._current_attention_status().strip().lower()
        attention_map = {
            "away": "looking_away",
            "distracted_side": "looking_away",
            "sleepy": "sleepy",
            "phone": "phone_detected",
        }
        mapped = attention_map.get(attention_status)
        if mapped and mapped not in active:
            active.append(mapped)
        return active

    def handle_support_event(self, event: str, timestamp: float | None = None, details: dict[str, Any] | None = None) -> None:
        name = str(event or "").strip().lower()
        if not name:
            return
        now = timestamp if timestamp is not None else time.time()
        self.recent_events.append((now, name))
        if details and self.config.validation_log_enabled:
            self.log_callback(f"[EmotionValidation] support_event={name} details={details}")

    def _build_learner_state(self, decision: EmpathyDecision, timestamp: float | None = None) -> LearnerStateSnapshot:
        perf_details = self.latest_performance.details if self.latest_performance else {}
        attention_status = self._current_attention_status()
        events = self._active_events(timestamp)

        if int(perf_details.get("clarification_streak", 0) or 0) >= 2 and "repeated_confusion" not in events:
            events.append("repeated_confusion")
        if self.latest_tracked.productive_confusion and "productive_confusion" not in events:
            events.append("productive_confusion")
        if self.latest_tracked.scores.get("frustration", 0.0) >= 0.55 and "frustration_rising" not in events:
            events.append("frustration_rising")

        self_doubt = bool(self.latest_tracked.scores.get("anxiety_self_doubt", 0.0) >= 0.56)
        if self_doubt and "self_doubt" not in events:
            events.append("self_doubt")

        return LearnerStateSnapshot(
            confusion=self._score_level(self.latest_tracked.scores.get("confusion", 0.0)),
            frustration=self._score_level(self.latest_tracked.scores.get("frustration", 0.0)),
            engagement=self._score_level(self.latest_tracked.scores.get("engagement", 0.0)),
            attention=self._attention_level(attention_status),
            self_doubt=self_doubt,
            detected_events=events,
            productive_confusion=bool(self.latest_tracked.productive_confusion),
            recommended_strategy=decision.pedagogical_action,
            attention_status=attention_status,
        )

    def current_validation_snapshot(self, adaptive_prompt_injected: bool | None = None) -> dict[str, Any]:
        snapshot = {
            "emotion_engine_enabled": bool(self.get_settings().enabled),
            "learner_state": self.latest_learner_state.to_dict(),
            "detected_events": list(self.latest_learner_state.detected_events),
            "productive_confusion": bool(self.latest_tracked.productive_confusion),
            "selected_strategy": self.latest_policy.pedagogical_action,
        }
        if adaptive_prompt_injected is not None:
            snapshot["adaptive_prompt_injected"] = bool(adaptive_prompt_injected)
        return snapshot

    def log_validation_snapshot(self, adaptive_prompt_injected: bool | None = None, label: str = "") -> None:
        if not self.config.validation_log_enabled:
            return
        payload = self.current_validation_snapshot(adaptive_prompt_injected=adaptive_prompt_injected)
        prefix = f"[EmotionValidation] {label.strip()} " if label else "[EmotionValidation] "
        self.log_callback(prefix + str(payload))

    def _max_score(self, label: str) -> float:
        values: list[float] = []
        for source in (
            self.latest_text,
            self.latest_performance,
            self.latest_speech,
            self.latest_fused,
            self.latest_tracked,
            self.camera.latest_camera_estimate if self.camera is not None else None,
            self.camera.latest_face_estimate if self.camera is not None else None,
        ):
            if source is None:
                continue
            scores = getattr(source, "scores", None)
            if isinstance(scores, dict):
                values.append(float(scores.get(label, 0.0) or 0.0))
        return max(values or [0.0])

    def _derive_proxy_valence_arousal(self, camera_fields: dict[str, Any]) -> tuple[float | None, float | None]:
        if camera_fields.get("valence") is not None and camera_fields.get("arousal") is not None:
            return float(camera_fields.get("valence")), float(camera_fields.get("arousal"))

        engagement = self._max_score("engagement")
        confidence = self._max_score("confidence")
        confusion = self._max_score("confusion")
        frustration = self._max_score("frustration")
        boredom = self._max_score("boredom")
        self_doubt = self._max_score("anxiety_self_doubt")
        overload = self._max_score("overload")

        strongest = max(engagement, confidence, confusion, frustration, boredom, self_doubt, overload)
        if strongest <= 0.02:
            return 0.0, 0.0

        positive = (0.55 * confidence) + (0.45 * engagement)
        negative = (0.30 * confusion) + (0.30 * frustration) + (0.22 * boredom) + (0.18 * self_doubt)
        valence = _clamp(positive - negative)
        arousal = _clamp((0.55 * frustration) + (0.40 * overload) + (0.22 * engagement) + (0.14 * confusion) - (0.40 * boredom))
        return round(valence, 4), round(arousal, 4)

    def _derive_monitor_engagement(self, camera_fields: dict[str, Any]) -> tuple[float | None, str]:
        score = camera_fields.get("engagement_score")
        if score is None:
            score = max(
                float(self.latest_fused.scores.get("engagement", 0.0) or 0.0),
                float(self.latest_tracked.scores.get("engagement", 0.0) or 0.0),
                float(self.latest_text.scores.get("engagement", 0.0) or 0.0),
                float(self.latest_performance.scores.get("engagement", 0.0) or 0.0),
            )
        score = None if score is None else float(score)
        if score is None:
            return None, "Not available"
        if score >= 0.68:
            label = "engaged"
        elif score >= 0.40:
            label = "monitor"
        else:
            label = "disengaged"
        return score, label

    def _derive_attention_status(self, camera_fields: dict[str, Any], engagement_score: float | None) -> str:
        status = str(camera_fields.get("attention_status") or "Camera off")
        if status and status != "Camera off":
            return status
        if engagement_score is not None and engagement_score >= 0.58:
            return "text_active"
        if max(
            float(self.latest_text.scores.get("confusion", 0.0) or 0.0),
            float(self.latest_performance.scores.get("confusion", 0.0) or 0.0),
        ) >= 0.40:
            return "reflecting"
        return "Camera off"

    def _rebuild_monitor(self) -> None:
        camera_fields = dict(self.camera.latest_monitor_fields) if self.camera is not None else {}
        valence, arousal = self._derive_proxy_valence_arousal(camera_fields)
        engagement_score, engagement_label = self._derive_monitor_engagement(camera_fields)
        attention_status = self._derive_attention_status(camera_fields, engagement_score)
        extras = dict(camera_fields.get("extras") or {})
        extras.setdefault("monitoring_mode", "multimodal")
        extras["monitor_engagement_source"] = "camera" if camera_fields.get("engagement_score") is not None else "multimodal"
        extras.setdefault("attention_source", "multimodal" if attention_status == "Camera off" else extras.get("attention_source", "pose"))
        self.latest_monitor = MonitorSnapshot(
            settings=self.get_settings(),
            text=self.latest_text,
            performance=self.latest_performance,
            camera=self.camera.latest_camera_estimate if self.camera is not None else ModalityEstimate(source="camera"),
            face=self.camera.latest_face_estimate if self.camera is not None else ModalityEstimate(source="face"),
            speech=self.latest_speech,
            fused=self.latest_fused,
            tracked=self.latest_tracked,
            policy=self.latest_policy,
            raw_face_emotion=str(camera_fields.get("raw_face_emotion", "Not available")),
            raw_face_confidence=float(camera_fields.get("raw_face_confidence", 0.0) or 0.0),
            raw_face_probabilities=dict(camera_fields.get("raw_face_probabilities") or {}),
            tutoring_face_label=str(camera_fields.get("tutoring_face_label", "Not available")),
            tutoring_face_confidence=float(camera_fields.get("tutoring_face_confidence", 0.0) or 0.0),
            valence=valence,
            arousal=arousal,
            engagement_score=engagement_score,
            engagement_label=engagement_label,
            attention_status=attention_status,
            blink_count=camera_fields.get("blink_count"),
            blink_rate=camera_fields.get("blink_rate"),
            yawn_count=camera_fields.get("yawn_count"),
            face_present=bool(camera_fields.get("face_present", False)),
            vlm_state=str(camera_fields.get("vlm_state", "Not available")),
            landmarks=list(camera_fields.get("landmarks") or []),
            extras=extras,
        )

    def emit_status(self) -> None:
        if self.socketio is None:
            return
        self.socketio.emit(
            "emotion_engine_status",
            {
                "settings": self.get_settings().to_dict(),
                "packet": self.latest_packet.to_dict(),
            },
        )

    def emit_monitor_update(self) -> None:
        self._rebuild_monitor()
        if self.socketio is None:
            return
        self.socketio.emit("emotion_monitor_update", self.latest_monitor.to_dict())
        if self.camera is not None:
            for alert in self.camera.latest_alerts:
                self.socketio.emit("emotion_alert", alert)

    def handle_camera_status(
        self,
        attention: str,
        gender: str,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_camera_runtime()
        with self._camera_lock:
            self.camera.update_vlm_attention(attention, gender, confidence, details=details)
            self.emit_monitor_update()

    def handle_camera_frame(self, jpeg_b64: str, timestamp: float | None = None) -> None:
        self._ensure_camera_runtime()
        with self._camera_lock:
            self.camera.process_frame_b64(jpeg_b64, timestamp=timestamp)
            self.emit_monitor_update()

    def handle_user_turn(self, text: str, intent: str, phase: str, timestamp: float | None = None) -> EmotionControlPacket:
        settings = self.get_settings()
        self.user_history.append(text)

        now = timestamp if timestamp is not None else time.time()
        self._ensure_text_runtime()
        self.latest_text = self.text.analyze(text, history=list(self.user_history))
        self.latest_performance = self.performance.observe_turn(text, intent, phase=phase, timestamp=now)
        self.latest_speech = self.speech.analyze()
        self.latest_fused = self.fusion.fuse(
            self.latest_text,
            self.latest_performance,
            self.camera.latest_camera_estimate if self.camera is not None else None,
            self.camera.latest_face_estimate if self.camera is not None else None,
            self.latest_speech,
        )
        self.latest_tracked = self.state_tracker.update(self.latest_fused, timestamp=now)
        self.latest_learner_state = LearnerStateSnapshot(
            confusion=self._score_level(self.latest_tracked.scores.get("confusion", 0.0)),
            frustration=self._score_level(self.latest_tracked.scores.get("frustration", 0.0)),
            engagement=self._score_level(self.latest_tracked.scores.get("engagement", 0.0)),
            attention=self._attention_level(self._current_attention_status()),
            self_doubt=bool(self.latest_tracked.scores.get("anxiety_self_doubt", 0.0) >= 0.56),
            detected_events=self._active_events(now),
            productive_confusion=bool(self.latest_tracked.productive_confusion),
            recommended_strategy="normal_explain",
            attention_status=self._current_attention_status(),
        )

        if not settings.enabled:
            self.latest_policy = EmpathyDecision()
            self.latest_packet = self._disabled_packet()
            self.emit_status()
            self.emit_monitor_update()
            self.log_validation_snapshot(adaptive_prompt_injected=False, label="user_turn")
            return self.latest_packet

        self.latest_policy = self.policy_engine.decide(
            self.latest_tracked,
            self.latest_performance,
            active_events=self._active_events(now),
        )
        self.latest_learner_state = self._build_learner_state(self.latest_policy, timestamp=now)
        self.latest_packet = self.conditioner.build_packet(
            enabled=True,
            tracked=self.latest_tracked,
            fused=self.latest_fused,
            decision=self.latest_policy,
            state_confidence=self.latest_fused.confidence,
            learner_state=self.latest_learner_state.to_dict(),
            monitor_flags=settings.to_dict(),
        )
        self.emit_status()
        self.emit_monitor_update()
        self.log_validation_snapshot(adaptive_prompt_injected=None, label="user_turn")
        return self.latest_packet

    def record_qa_result(self, correct: bool, partial: bool = False, timestamp: float | None = None) -> None:
        self.performance.record_qa_result(correct, partial=partial, timestamp=timestamp)

    def current_prompt_block(self) -> str:
        packet = self.latest_packet if self.get_settings().enabled else self._disabled_packet()
        return self.conditioner.render_prompt_block(packet)
