from __future__ import annotations

from collections import deque

import numpy as np

from emotion_engine.utils import FaceDetection, SmoothedEmotionState, TutoringStateResult, clamp, compute_bbox_motion


class TutoringStateMapper:
    def __init__(self, history_size: int = 60) -> None:
        self.history_size = history_size
        self.motion_history: deque[float] = deque(maxlen=history_size)
        self.presence_history: deque[float] = deque(maxlen=history_size)
        self.label_history: deque[str] = deque(maxlen=history_size)
        self.last_bbox: tuple[int, int, int, int] | None = None

    def update(
        self,
        emotion_state: SmoothedEmotionState | None,
        detection: FaceDetection | None,
    ) -> TutoringStateResult:
        if emotion_state is None or detection is None:
            self.presence_history.append(0.0)
            self.motion_history.append(0.0)
            self.last_bbox = None
            return TutoringStateResult(
                label="no_face",
                confidence=0.0,
                source="system",
                rationale="No face detected.",
            )

        motion = compute_bbox_motion(self.last_bbox, detection.bbox)
        self.last_bbox = detection.bbox
        self.motion_history.append(motion)
        self.presence_history.append(1.0)
        self.label_history.append(emotion_state.emotion)

        presence_ratio = float(np.mean(self.presence_history)) if self.presence_history else 0.0
        motion_mean = float(np.mean(self.motion_history)) if self.motion_history else 0.0
        instability = emotion_state.instability

        probabilities = emotion_state.probabilities
        anger = probabilities.get("anger", 0.0)
        contempt = probabilities.get("contempt", 0.0)
        disgust = probabilities.get("disgust", 0.0)
        fear = probabilities.get("fear", 0.0)
        happiness = probabilities.get("happiness", 0.0)
        neutral = probabilities.get("neutral", 0.0)
        sadness = probabilities.get("sadness", 0.0)
        surprise = probabilities.get("surprise", 0.0)
        sorted_scores = sorted(probabilities.values(), reverse=True)
        top_margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0]

        valence = float(emotion_state.valence)
        arousal = float(emotion_state.arousal)
        positive_valence = clamp(max(valence, 0.0), 0.0, 1.0)
        negative_valence = clamp(max(-valence, 0.0), 0.0, 1.0)
        positive_arousal = clamp(max(arousal, 0.0), 0.0, 1.0)
        low_arousal = clamp(max(-arousal, 0.0), 0.0, 1.0)
        inactivity = clamp(1.0 - min(motion_mean * 15.0, 1.0), 0.0, 1.0)
        engagement_band = clamp(1.0 - min(abs(arousal - 0.15) / 0.65, 1.0), 0.0, 1.0)

        happy_score = max(happiness, 0.65 * positive_valence)
        sad_score = max(sadness, (0.7 * negative_valence) + (0.15 * low_arousal))
        fear_score = max(fear, (0.55 * negative_valence) + (0.35 * positive_arousal))
        frustrated_score = clamp(
            (0.45 * max(anger, contempt, sadness))
            + (0.35 * negative_valence)
            + (0.20 * positive_arousal),
            0.0,
            1.0,
        )
        anxious_score = clamp(
            (0.45 * fear)
            + (0.20 * surprise)
            + (0.20 * negative_valence)
            + (0.20 * positive_arousal)
            + (0.15 * instability),
            0.0,
            1.0,
        )
        bored_score = clamp(
            (0.35 * neutral)
            + (0.20 * sadness)
            + (0.30 * low_arousal)
            + (0.10 * inactivity)
            + (0.10 * presence_ratio)
            + (0.05 * negative_valence),
            0.0,
            1.0,
        )
        confused_score = clamp(
            (0.30 * surprise)
            + (0.25 * neutral)
            + (0.25 * instability)
            + (0.20 * negative_valence),
            0.0,
            1.0,
        )
        engaged_score = clamp(
            (0.30 * presence_ratio)
            + (0.25 * engagement_band)
            + (0.20 * positive_valence)
            + (0.15 * (neutral + happiness))
            + (0.10 * (1.0 - negative_valence)),
            0.0,
            1.0,
        )
        uncertain_affect = emotion_state.confidence < 0.34 or top_margin < 0.07

        if happy_score > 0.60 and (happiness > 0.30 or valence > 0.30):
            return TutoringStateResult("happy", happy_score, "direct", "Positive valence with happiness trend.")

        if sad_score > 0.62 and (sadness > 0.28 or valence < -0.38):
            return TutoringStateResult("sad", sad_score, "direct", "Low valence with sadness trend.")

        if fear_score > 0.66 and (fear > 0.28 or (valence < -0.42 and arousal > 0.18)):
            return TutoringStateResult("fear", fear_score, "direct", "Fear signal or strongly negative high-arousal state.")

        if uncertain_affect and neutral > 0.22 and abs(valence) < 0.28 and abs(arousal) < 0.30:
            return TutoringStateResult(
                "neutral",
                max(neutral, 0.55),
                "direct",
                "Uncertain low-intensity affect. Prefering a neutral reading over a stronger emotional claim.",
            )

        if frustrated_score > 0.70 and valence < -0.34 and max(anger, contempt, sadness) > 0.24:
            return TutoringStateResult(
                "frustrated",
                frustrated_score,
                "heuristic",
                "Negative valence plus anger/sadness-like high-effort expression.",
            )

        if anxious_score > 0.64 and valence < -0.18 and arousal > 0.10 and max(fear, surprise) > 0.18:
            return TutoringStateResult(
                "anxious",
                anxious_score,
                "heuristic",
                "Negative valence plus elevated arousal with fear/surprise trend.",
            )

        if bored_score > 0.64 and arousal < -0.06 and motion_mean < 0.05 and presence_ratio > 0.55:
            return TutoringStateResult(
                "bored",
                bored_score,
                "heuristic",
                "Low arousal and low motion over time.",
            )

        if confused_score > 0.60 and -0.40 < valence < 0.16 and instability > 0.24:
            return TutoringStateResult(
                "confused",
                confused_score,
                "heuristic",
                "Unstable expression with mild negative or uncertain affect.",
            )

        if engaged_score > 0.63 and valence > -0.06 and arousal > -0.12 and anxious_score < 0.55 and bored_score < 0.55:
            return TutoringStateResult(
                "engaged",
                engaged_score,
                "heuristic",
                "Sustained face presence with balanced arousal and neutral/positive affect.",
            )

        if neutral > 0.40 or (abs(valence) < 0.18 and abs(arousal) < 0.22):
            return TutoringStateResult("neutral", max(neutral, 0.55), "direct", "Neutral affect.")

        fallback_map = {
            "happiness": ("happy", "direct"),
            "sadness": ("sad", "direct"),
            "fear": ("fear", "direct"),
            "anger": ("frustrated", "heuristic"),
            "contempt": ("frustrated", "heuristic"),
            "disgust": ("frustrated", "heuristic"),
            "surprise": ("confused", "heuristic"),
            "neutral": ("neutral", "direct"),
        }
        label, source = fallback_map.get(emotion_state.emotion, ("neutral", "heuristic"))
        if label in {"frustrated", "confused"} and emotion_state.confidence < 0.45 and neutral > 0.20:
            return TutoringStateResult(
                "neutral",
                max(neutral, 0.52),
                "direct",
                "Low-confidence negative expression. Prefering a neutral fallback over a stronger tutoring label.",
            )
        return TutoringStateResult(
            label=label,
            confidence=max(0.40, emotion_state.confidence),
            source=source,
            rationale=f"Fallback mapping from raw emotion '{emotion_state.emotion}'.",
        )
