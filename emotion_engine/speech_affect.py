from __future__ import annotations

from .schemas import ModalityEstimate


class SpeechAffectEstimator:
    def __init__(self) -> None:
        self.available = False

    def analyze(self, *_args, **_kwargs) -> ModalityEstimate:
        return ModalityEstimate(
            confidence=0.0,
            source="speech_unavailable",
            details={"status": "Speech affect is not enabled in this build."},
        )
