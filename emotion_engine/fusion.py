from __future__ import annotations

from .config import EmotionEngineConfig
from .schemas import FusedAffectState, ModalityEstimate, StateConfidence, TARGET_AFFECT_LABELS


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class AffectFusionEngine:
    def __init__(self, config: EmotionEngineConfig) -> None:
        self.config = config

    def _weight_map(self) -> dict[str, float]:
        return {
            "text": self.config.text_weight,
            "performance": self.config.performance_weight,
            "camera": self.config.camera_weight,
            "face": self.config.face_weight,
            "speech": self.config.speech_weight,
        }

    def fuse(
        self,
        text: ModalityEstimate | None,
        performance: ModalityEstimate | None,
        camera: ModalityEstimate | None,
        face: ModalityEstimate | None,
        speech: ModalityEstimate | None,
    ) -> FusedAffectState:
        estimates = {
            "text": text,
            "performance": performance,
            "camera": camera,
            "face": face,
            "speech": speech,
        }
        base_weights = self._weight_map()
        weighted_scores = {label: 0.0 for label in TARGET_AFFECT_LABELS}
        total_weight = 0.0
        evidence: list[str] = []

        state_confidence = StateConfidence(
            text=float(text.confidence if text else 0.0),
            performance=float(performance.confidence if performance else 0.0),
            camera=float(camera.confidence if camera else 0.0),
            face=float(face.confidence if face else 0.0),
            speech=float(speech.confidence if speech else 0.0),
        )

        for name, estimate in estimates.items():
            if estimate is None or estimate.confidence <= 0.0:
                continue
            weight = base_weights[name] * estimate.confidence
            if weight <= 0.0:
                continue
            total_weight += weight
            dominant = max(estimate.scores.items(), key=lambda item: item[1])[0]
            evidence.append(f"{name}:{dominant}")
            for label in TARGET_AFFECT_LABELS:
                weighted_scores[label] += weight * float(estimate.scores.get(label, 0.0))

        if total_weight > 0.0:
            fused_scores = {
                label: _clamp(weighted_scores[label] / total_weight)
                for label in TARGET_AFFECT_LABELS
            }
        else:
            fused_scores = {label: 0.0 for label in TARGET_AFFECT_LABELS}
            fused_scores["engagement"] = 0.35

        text_perf_available = any(
            estimate is not None and estimate.confidence >= 0.25
            for estimate in (text, performance)
        )
        if not text_perf_available:
            for label in ("confusion", "frustration", "boredom", "anxiety_self_doubt", "overload"):
                fused_scores[label] = min(fused_scores[label], 0.55)

        perf_evidence = {str(item) for item in ((performance.details or {}).get("evidence") or [])} if performance else set()
        if performance is not None and "self_doubt_text" in perf_evidence:
            fused_scores["anxiety_self_doubt"] = max(
                fused_scores["anxiety_self_doubt"],
                _clamp(float(performance.scores.get("anxiety_self_doubt", 0.0) or 0.0)),
            )
            fused_scores["frustration"] = max(
                fused_scores["frustration"],
                _clamp(float(performance.scores.get("frustration", 0.0) or 0.0)),
            )

        dominant_label, dominant_score = max(fused_scores.items(), key=lambda item: item[1])
        state_confidence.fused = _clamp(
            0.30
            + 0.25 * state_confidence.text
            + 0.25 * state_confidence.performance
            + 0.10 * state_confidence.camera
            + 0.10 * state_confidence.face
        )

        return FusedAffectState(
            scores=fused_scores,
            dominant_label=dominant_label,
            dominant_score=float(dominant_score),
            confidence=state_confidence,
            evidence=evidence[:6],
        )
