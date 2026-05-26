from __future__ import annotations

from collections import deque

import numpy as np

from emotion_engine.utils import EmotionPrediction, SmoothedEmotionState, clamp


class TemporalSmoother:
    def __init__(
        self,
        labels: list[str],
        ema_alpha: float = 0.35,
        signal_alpha: float = 0.25,
        history_size: int = 45,
        clear_after_missed_faces: int = 12,
    ) -> None:
        self.labels = labels
        self.ema_alpha = clamp(ema_alpha, 0.05, 1.0)
        self.signal_alpha = clamp(signal_alpha, 0.05, 1.0)
        self.history_size = history_size
        self.clear_after_missed_faces = clear_after_missed_faces

        self.probabilities_ema: np.ndarray | None = None
        self.valence_ema: float = 0.0
        self.arousal_ema: float = 0.0
        self.confidence_ema: float = 0.0
        self.label_history: deque[str] = deque(maxlen=history_size)
        self.missed_face_frames = 0
        self.last_state: SmoothedEmotionState | None = None

    def update(self, prediction: EmotionPrediction) -> SmoothedEmotionState:
        vector = np.array([prediction.probabilities[label] for label in self.labels], dtype=np.float32)
        if self.probabilities_ema is None:
            self.probabilities_ema = vector.copy()
            self.valence_ema = float(prediction.valence or 0.0)
            self.arousal_ema = float(prediction.arousal or 0.0)
            self.confidence_ema = prediction.confidence
        else:
            self.probabilities_ema = (self.ema_alpha * vector) + (
                (1.0 - self.ema_alpha) * self.probabilities_ema
            )
            self.valence_ema = (self.signal_alpha * float(prediction.valence or 0.0)) + (
                (1.0 - self.signal_alpha) * self.valence_ema
            )
            self.arousal_ema = (self.signal_alpha * float(prediction.arousal or 0.0)) + (
                (1.0 - self.signal_alpha) * self.arousal_ema
            )
            self.confidence_ema = (self.signal_alpha * prediction.confidence) + (
                (1.0 - self.signal_alpha) * self.confidence_ema
            )

        total = float(self.probabilities_ema.sum())
        if total > 0.0:
            self.probabilities_ema /= total

        top_index = int(np.argmax(self.probabilities_ema))
        top_probability = float(self.probabilities_ema[top_index])
        top_label = self.labels[top_index]

        sorted_probs = np.sort(self.probabilities_ema)[::-1]
        top_gap = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        entropy = float(
            -np.sum(self.probabilities_ema * np.log(np.clip(self.probabilities_ema, 1e-6, 1.0)))
            / np.log(len(self.labels))
        )

        self.label_history.append(top_label)
        change_rate = 0.0
        if len(self.label_history) > 1:
            transitions = sum(
                1
                for index in range(1, len(self.label_history))
                if self.label_history[index] != self.label_history[index - 1]
            )
            change_rate = transitions / float(len(self.label_history) - 1)

        instability = clamp((0.5 * entropy) + (0.25 * (1.0 - top_gap)) + (0.25 * change_rate), 0.0, 1.0)
        self.missed_face_frames = 0

        self.last_state = SmoothedEmotionState(
            emotion=top_label,
            confidence=top_probability,
            probabilities={self.labels[index]: float(self.probabilities_ema[index]) for index in range(len(self.labels))},
            valence=self.valence_ema,
            arousal=self.arousal_ema,
            va_source=prediction.va_source,
            instability=instability,
            raw_emotion=prediction.emotion,
            raw_confidence=prediction.confidence,
        )
        return self.last_state

    def mark_no_face(self) -> None:
        self.missed_face_frames += 1
        if self.missed_face_frames >= self.clear_after_missed_faces:
            self.clear()

    def clear(self) -> None:
        self.probabilities_ema = None
        self.valence_ema = 0.0
        self.arousal_ema = 0.0
        self.confidence_ema = 0.0
        self.label_history.clear()
        self.missed_face_frames = 0
        self.last_state = None
