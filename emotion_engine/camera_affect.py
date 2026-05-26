from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np

from .config import EmotionEngineConfig
from .detector import YuNetFaceDetector
from .emotion_model import EmotionRecognizer
from .engagement import AlertManager, EngagementScorer
from .landmarks import AttentionState, FACE_MESH_CONTOURS, FACE_MESH_IRISES, FACE_MESH_TESSELATION, FaceMeshMonitor
from .mapper import TutoringStateMapper
from .schemas import ModalityEstimate
from .smoother import TemporalSmoother
from .utils import FaceDetection, crop_face


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class CameraAffectEstimator:
    def __init__(self, config: EmotionEngineConfig, log_callback=None) -> None:
        self.config = config
        self.log_callback = log_callback or (lambda _msg: None)
        self.frame_index = 0
        self.latest_vlm_attention = {
            "attention": "unknown",
            "gender": "unknown",
            "confidence": 0.0,
            "person_count": 0,
            "phone_present": False,
            "phone_activity": "none",
            "object_label": "",
            "observed_action": "",
            "visible_text": "",
            "attention_comment": "",
            "multiple_people": False,
            "vlm_runtime": "",
        }
        self.latest_monitor_fields: dict[str, Any] = {
            "attention_status": "Camera off",
            "vlm_state": "Not available",
            "face_present": False,
            "landmarks": [],
            "engagement_label": "Not available",
            "engagement_score": None,
            "raw_face_emotion": "Not available",
            "raw_face_confidence": 0.0,
            "raw_face_probabilities": {},
            "tutoring_face_label": "Not available",
            "tutoring_face_confidence": 0.0,
            "valence": None,
            "arousal": None,
            "blink_count": None,
            "blink_rate": None,
            "yawn_count": None,
            "extras": {},
        }
        self.latest_face_estimate = ModalityEstimate(source="face")
        self.latest_camera_estimate = ModalityEstimate(source="camera")
        self.latest_alerts: list[dict[str, Any]] = []

        self.detector = None
        self.face_mesh = None
        self.emotion_model = None
        self.smoother = None
        self.mapper = TutoringStateMapper()
        self.engagement = EngagementScorer()
        self.alerts = AlertManager()

        self._init_models()

    def _init_models(self) -> None:
        try:
            self.detector = YuNetFaceDetector(
                self.config.yu_net_model_path,
                score_threshold=self.config.yu_net_score_threshold,
            )
        except Exception as exc:
            self.log_callback(f"Emotion detector unavailable: {exc}")
            self.detector = None

        try:
            self.face_mesh = FaceMeshMonitor()
        except Exception as exc:
            self.log_callback(f"Face mesh unavailable: {exc}")
            self.face_mesh = None

        try:
            self.emotion_model = EmotionRecognizer(
                model_name=self.config.face_model_name,
                engine=self.config.face_engine,
                device=self.config.face_device,
            )
            self.smoother = TemporalSmoother(self.emotion_model.labels)
        except Exception as exc:
            self.log_callback(f"Face emotion backend unavailable: {exc}")
            self.emotion_model = None
            self.smoother = None

    def update_vlm_attention(
        self,
        attention: str,
        gender: str,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        self.latest_vlm_attention = {
            "attention": attention or "unknown",
            "gender": gender or "unknown",
            "confidence": float(confidence or 0.0),
            "person_count": int(details.get("person_count", 0) or 0),
            "phone_present": bool(details.get("phone_present", False)),
            "phone_activity": str(details.get("phone_activity") or "none"),
            "object_label": str(details.get("object_label") or ""),
            "observed_action": str(details.get("observed_action") or ""),
            "visible_text": str(details.get("visible_text") or ""),
            "attention_comment": str(details.get("attention_comment") or ""),
            "multiple_people": bool(details.get("multiple_people", False)),
            "vlm_runtime": str(details.get("vlm_runtime") or ""),
        }

    def _decode_frame(self, jpeg_b64: str):
        try:
            raw = base64.b64decode(jpeg_b64)
            array = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _fallback_detection(self, frame_bgr) -> FaceDetection | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        height, width = frame_bgr.shape[:2]
        side = min(height, width)
        x = max(0, (width - side) // 2)
        y = max(0, (height - side) // 2)
        return FaceDetection(
            bbox=(x, y, side, side),
            score=0.15,
            landmarks=[],
            area=float(side * side),
        )

    def _normalize_landmarks(self, landmarks: list[tuple[int, int]] | None, width: int, height: int) -> list[tuple[float, float]]:
        if not landmarks or width <= 0 or height <= 0:
            return []
        return [(float(x) / float(width), float(y) / float(height)) for x, y in landmarks]

    def _build_face_estimate(self, emotion_state, tutoring_state) -> ModalityEstimate:
        scores = {
            "confusion": 0.0,
            "prolonged_confusion": 0.0,
            "frustration": 0.0,
            "boredom": 0.0,
            "anxiety_self_doubt": 0.0,
            "engagement": 0.0,
            "confidence": 0.0,
            "overload": 0.0,
        }
        if emotion_state is None or tutoring_state is None:
            return ModalityEstimate(scores=scores, confidence=0.0, source="face")

        label = tutoring_state.label
        conf = float(tutoring_state.confidence)
        valence = float(emotion_state.valence)
        arousal = float(emotion_state.arousal)

        if label == "confused":
            scores["confusion"] = conf
        elif label == "frustrated":
            scores["frustration"] = conf
            scores["overload"] = max(scores["overload"], conf * 0.45)
        elif label == "bored":
            scores["boredom"] = conf
        elif label in {"anxious", "fear", "sad"}:
            scores["anxiety_self_doubt"] = conf
        elif label == "engaged":
            scores["engagement"] = conf
            scores["confidence"] = max(scores["confidence"], conf * 0.45)
        elif label == "happy":
            scores["confidence"] = conf * 0.75
            scores["engagement"] = max(scores["engagement"], conf * 0.55)
        else:
            scores["engagement"] = max(scores["engagement"], 0.22 if valence >= -0.05 else 0.0)

        if valence > 0.18:
            scores["confidence"] = max(scores["confidence"], _clamp(0.45 + (0.35 * valence)))
            scores["engagement"] = max(scores["engagement"], _clamp(0.36 + (0.25 * valence)))
        if valence < -0.22 and arousal > 0.05:
            scores["frustration"] = max(scores["frustration"], _clamp(0.35 + (0.30 * -valence)))
        if valence < -0.18 and arousal < -0.05:
            scores["boredom"] = max(scores["boredom"], _clamp(0.28 + (0.20 * -valence)))
        scores["overload"] = max(
            scores["overload"],
            _clamp(
                0.35 * scores["confusion"]
                + 0.35 * scores["frustration"]
                + 0.30 * scores["anxiety_self_doubt"]
            ),
        )

        return ModalityEstimate(
            scores=scores,
            confidence=float(emotion_state.confidence),
            source="face",
            details={
                "raw_emotion": emotion_state.emotion,
                "raw_confidence": emotion_state.confidence,
                "valence": valence,
                "arousal": arousal,
                "tutoring_label": label,
                "tutoring_confidence": conf,
            },
        )

    def _derive_pose_attention(self, attention_state: AttentionState | None) -> str:
        if attention_state is None:
            return "Camera off"
        if not attention_state.face_present:
            return "away" if attention_state.face_absent_duration >= 3.0 else "Camera off"
        if attention_state.prolonged_eye_closure and attention_state.current_eye_closure_duration >= 3.4:
            return "sleepy"
        if attention_state.current_away_duration >= 6.0:
            return "away"

        head_direction = str(attention_state.head_direction or "unknown")
        gaze_direction = str(attention_state.gaze_direction or "unknown")
        if head_direction == "down" or gaze_direction == "down":
            return "looking_down"
        if (
            attention_state.current_away_duration >= 1.2
            and (
                head_direction in {"left", "right", "up"}
                or gaze_direction in {"left", "right"}
                or attention_state.looking_away
            )
        ):
            return "distracted_side"
        return "focused"

    def _resolve_attention_status(self, attention_state: AttentionState | None, vlm_attention: str) -> tuple[str, str, str, str]:
        scene_attention = str(vlm_attention or "unknown")
        pose_attention = self._derive_pose_attention(attention_state)

        if scene_attention == "phone" and bool(self.latest_vlm_attention.get("phone_present")):
            return "phone", pose_attention, scene_attention, "scene"
        if scene_attention == "multiple_people" and bool(self.latest_vlm_attention.get("multiple_people")):
            return "multiple_people", pose_attention, scene_attention, "scene"
        if pose_attention != "Camera off":
            return pose_attention, pose_attention, scene_attention, "pose"
        if scene_attention == "away":
            return "away", pose_attention, scene_attention, "scene"
        return "Camera off", pose_attention, scene_attention, "none"

    def _build_camera_estimate(
        self,
        attention_state: AttentionState | None,
        engagement_state,
        effective_attention: str,
        vlm_attention: str,
    ) -> ModalityEstimate:
        scores = {
            "confusion": 0.0,
            "prolonged_confusion": 0.0,
            "frustration": 0.0,
            "boredom": 0.0,
            "anxiety_self_doubt": 0.0,
            "engagement": 0.0,
            "confidence": 0.0,
            "overload": 0.0,
        }
        details: dict[str, Any] = {
            "vlm_attention": vlm_attention,
            "effective_attention": effective_attention,
        }

        if attention_state is None:
            confidence = _clamp(0.18 + 0.45 * float(self.latest_vlm_attention.get("confidence") or 0.0))
            if effective_attention == "phone":
                scores["boredom"] = 0.55
                scores["engagement"] = 0.16
            elif effective_attention == "away":
                scores["boredom"] = 0.44
                scores["engagement"] = 0.12
            elif effective_attention == "multiple_people":
                confidence = max(confidence, 0.40)
            return ModalityEstimate(scores=scores, confidence=confidence, source="camera", details=details)

        confidence = 0.25 + (0.40 * attention_state.face_presence_ratio)
        scores["engagement"] = _clamp(
            (0.35 * attention_state.face_presence_ratio)
            + (0.35 * attention_state.look_straight_ratio)
            + (0.20 if not attention_state.looking_away else 0.0)
        )
        scores["confidence"] = _clamp(0.20 + (0.30 * attention_state.look_straight_ratio))

        if effective_attention == "away":
            scores["boredom"] = max(scores["boredom"], 0.46)
            scores["engagement"] = min(scores["engagement"], 0.30)
        elif effective_attention == "distracted_side":
            scores["boredom"] = max(scores["boredom"], 0.38)
            scores["engagement"] = min(scores["engagement"], 0.36)
        elif effective_attention == "looking_down":
            scores["engagement"] = max(
                scores["engagement"],
                _clamp(0.36 + (0.14 * attention_state.face_presence_ratio)),
            )
        elif effective_attention == "sleepy":
            scores["boredom"] = max(scores["boredom"], 0.52)
            scores["overload"] = max(scores["overload"], 0.28)

        if attention_state.prolonged_eye_closure:
            scores["boredom"] = max(scores["boredom"], 0.52)
            scores["overload"] = max(scores["overload"], 0.28)
        if attention_state.yawn_active:
            scores["boredom"] = max(scores["boredom"], 0.48)
        if effective_attention == "phone":
            scores["boredom"] = max(scores["boredom"], 0.55)
            scores["engagement"] = min(scores["engagement"], 0.24)
        if engagement_state is not None:
            scores["engagement"] = max(scores["engagement"], float(engagement_state.score))

        details.update(
            {
                "blink_count": attention_state.blink_count,
                "blink_rate_per_min": attention_state.blink_rate_per_min,
                "yawn_count": attention_state.yawn_count,
                "look_straight_ratio": attention_state.look_straight_ratio,
                "looking_away": attention_state.looking_away,
                "head_direction": attention_state.head_direction,
                "gaze_direction": attention_state.gaze_direction,
            }
        )
        return ModalityEstimate(scores=scores, confidence=_clamp(confidence), source="camera", details=details)

    def process_frame(self, frame_bgr, timestamp: float | None = None) -> dict[str, Any]:
        now = timestamp if timestamp is not None else time.time()
        self.frame_index += 1
        if frame_bgr is None or frame_bgr.size == 0:
            return dict(self.latest_monitor_fields)

        frame_h, frame_w = frame_bgr.shape[:2]
        detection = self.detector.detect_primary(frame_bgr) if self.detector else None
        if detection is None:
            detection = self._fallback_detection(frame_bgr)

        attention_state = None
        if self.face_mesh is not None and detection is not None:
            try:
                attention_state = self.face_mesh.process(frame_bgr, detection.bbox, now)
            except Exception:
                attention_state = None

        emotion_state = self.smoother.last_state if self.smoother else None
        if detection is None:
            if self.smoother is not None:
                self.smoother.mark_no_face()
                emotion_state = self.smoother.last_state
        elif self.emotion_model is not None and self.smoother is not None:
            if self.frame_index % max(1, self.config.emotion_infer_every) == 0 or emotion_state is None:
                face_crop, _ = crop_face(frame_bgr, detection.bbox, padding=0.18, square=True)
                if face_crop.size != 0:
                    try:
                        prediction = self.emotion_model.predict(face_crop)
                        emotion_state = self.smoother.update(prediction)
                    except Exception as exc:
                        self.log_callback(f"Face emotion inference failed: {exc}")
            else:
                emotion_state = self.smoother.last_state

        tutoring_state = self.mapper.update(emotion_state, detection) if detection is not None else None
        engagement_state = self.engagement.compute(attention_state, emotion_state, tutoring_state)
        vlm_attention = str(self.latest_vlm_attention.get("attention") or "unknown")
        effective_attention, pose_attention, scene_attention, attention_source = self._resolve_attention_status(attention_state, vlm_attention)
        fast_alerts = self.alerts.evaluate_fast(attention_state, engagement_state, tutoring_state, timestamp=now)
        self.latest_alerts = [event.__dict__ for event in fast_alerts]

        face_estimate = self._build_face_estimate(emotion_state, tutoring_state)
        camera_estimate = self._build_camera_estimate(attention_state, engagement_state, effective_attention, scene_attention)
        self.latest_face_estimate = face_estimate
        self.latest_camera_estimate = camera_estimate

        landmarks = self._normalize_landmarks(
            attention_state.landmarks if attention_state else None,
            frame_w,
            frame_h,
        )
        self.latest_monitor_fields = {
            "attention_status": effective_attention,
            "vlm_state": scene_attention,
            "face_present": bool(attention_state.face_present) if attention_state else False,
            "landmarks": landmarks,
            "engagement_label": getattr(engagement_state, "label", "Not available"),
            "engagement_score": getattr(engagement_state, "score", None),
            "raw_face_emotion": emotion_state.emotion if emotion_state else "Not available",
            "raw_face_confidence": float(emotion_state.confidence) if emotion_state else 0.0,
            "raw_face_probabilities": dict(emotion_state.probabilities) if emotion_state else {},
            "tutoring_face_label": tutoring_state.label if tutoring_state else "Not available",
            "tutoring_face_confidence": float(tutoring_state.confidence) if tutoring_state else 0.0,
            "valence": None if emotion_state is None else float(emotion_state.valence),
            "arousal": None if emotion_state is None else float(emotion_state.arousal),
            "blink_count": None if attention_state is None else int(attention_state.blink_count),
            "blink_rate": None if attention_state is None else float(attention_state.blink_rate_per_min),
            "yawn_count": None if attention_state is None else int(attention_state.yawn_count),
            "extras": {
                "eyes": {
                    "state": (
                        "unknown"
                        if attention_state is None
                        else ("closed" if attention_state.eyes_closed else "open")
                    ),
                    "ear": None if attention_state is None else float(attention_state.ear_avg),
                },
                "mouth": {
                    "ratio": None if attention_state is None else float(attention_state.mouth_open_ratio),
                    "presence_ratio": None if attention_state is None else float(attention_state.face_presence_ratio),
                },
                "attention_line": (
                    "Camera off"
                    if attention_state is None
                    else (
                        f"attention {effective_attention} | gaze {attention_state.gaze_direction} | head {attention_state.head_direction} | "
                        f"away {attention_state.current_away_duration:.1f}s | straight {attention_state.look_straight_ratio:.2f}"
                    )
                ),
                "eyes_line": (
                    "No face detected"
                    if attention_state is None
                    else (
                        f"{'closed' if attention_state.eyes_closed else 'open'} | EAR {attention_state.ear_avg:.3f} | "
                        f"blinks {attention_state.blink_count} | blink rate {attention_state.blink_rate_per_min:.0f}/m"
                    )
                ),
                "mouth_line": (
                    "Not available"
                    if attention_state is None
                    else (
                        f"{attention_state.mouth_open_ratio:.2f} | yawns {attention_state.yawn_count} | "
                        f"presence {attention_state.face_presence_ratio:.2f}"
                    )
                ),
                "detector_info": "yunet",
                "emotion_runtime": self.emotion_model.runtime_label if self.emotion_model is not None else "disabled",
                "landmarks_enabled": bool(self.face_mesh is not None),
                "vlm": dict(self.latest_vlm_attention),
                "attention_source": attention_source,
                "pose_attention_status": pose_attention,
                "scene_attention_status": scene_attention,
                "head_direction": attention_state.head_direction if attention_state else "unknown",
                "gaze_direction": attention_state.gaze_direction if attention_state else "unknown",
                "look_straight_ratio": None if attention_state is None else float(attention_state.look_straight_ratio),
                "current_away_duration": None if attention_state is None else float(attention_state.current_away_duration),
                "mesh_connections": {
                    "tesselation": [list(pair) for pair in FACE_MESH_TESSELATION],
                    "contours": [list(pair) for pair in FACE_MESH_CONTOURS],
                    "irises": [list(pair) for pair in FACE_MESH_IRISES],
                },
            },
        }
        return dict(self.latest_monitor_fields)

    def process_frame_b64(self, jpeg_b64: str, timestamp: float | None = None) -> dict[str, Any]:
        frame = self._decode_frame(jpeg_b64)
        return self.process_frame(frame, timestamp=timestamp)
