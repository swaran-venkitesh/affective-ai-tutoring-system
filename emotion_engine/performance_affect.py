from __future__ import annotations

import re
import time
from collections import deque

from .schemas import ModalityEstimate, TARGET_AFFECT_LABELS


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class PerformanceAffectEstimator:
    CLARIFY_RE = re.compile(
        r"\b(i don't understand|i do not understand|still don't get it|still do not get it|again|repeat|confused|unclear|"
        r"can you explain|what does that mean|make it simple|simpler|break it down|slow down)\b",
        re.I,
    )
    HINT_RE = re.compile(r"\b(hint|clue|small step|smaller step|help me)\b", re.I)
    SELF_DOUBT_RE = re.compile(
        r"\b(i can't|i cannot|i am not good|i'm not good|i will fail|i'm bad at|"
        r"i am dumb|i'm dumb|i am stupid|i'm stupid|i will never get this|i'll never get this|"
        r"i will never understand|i'll never understand)\b",
        re.I,
    )
    REENGAGE_RE = re.compile(r"\b(next|continue|go on|another one|more example|let's continue|harder question)\b", re.I)

    def __init__(self) -> None:
        self.incorrect_streak = 0
        self.correct_streak = 0
        self.clarification_streak = 0
        self.hint_requests: deque[float] = deque(maxlen=12)
        self.failure_times: deque[float] = deque(maxlen=12)
        self.turn_timestamps: deque[float] = deque(maxlen=20)
        self.last_user_turn_ts: float | None = None
        self.last_intent: str = ""

    def _empty_scores(self) -> dict[str, float]:
        return {label: 0.0 for label in TARGET_AFFECT_LABELS}

    def observe_turn(
        self,
        text: str,
        intent: str,
        phase: str = "",
        timestamp: float | None = None,
    ) -> ModalityEstimate:
        now = timestamp if timestamp is not None else time.time()
        self.turn_timestamps.append(now)
        seconds_since_turn = 0.0 if self.last_user_turn_ts is None else max(0.0, now - self.last_user_turn_ts)
        self.last_user_turn_ts = now
        self.last_intent = intent

        low = text.lower()
        if intent == "not_understood" or self.CLARIFY_RE.search(low):
            self.clarification_streak += 1
        else:
            self.clarification_streak = max(0, self.clarification_streak - 1)

        if self.HINT_RE.search(low):
            self.hint_requests.append(now)

        scores = self._empty_scores()
        evidence: list[str] = []

        scores["confusion"] = _clamp(0.18 * self.clarification_streak)
        scores["frustration"] = _clamp((0.16 * self.incorrect_streak) + (0.10 * max(0, self.clarification_streak - 1)))
        scores["anxiety_self_doubt"] = _clamp(0.20 * len(self.failure_times) / 4.0)
        scores["confidence"] = _clamp(0.20 * self.correct_streak)
        scores["engagement"] = 0.28 if phase else 0.0

        if self.SELF_DOUBT_RE.search(low):
            scores["anxiety_self_doubt"] = max(scores["anxiety_self_doubt"], 0.78)
            scores["frustration"] = max(scores["frustration"], 0.44)
            evidence.append("self_doubt_text")
        if self.CLARIFY_RE.search(low):
            scores["confusion"] = max(scores["confusion"], 0.58)
            scores["engagement"] = max(scores["engagement"], 0.48)
            evidence.append("clarify_request")
        if self.HINT_RE.search(low):
            scores["confusion"] = max(scores["confusion"], 0.44)
            scores["engagement"] = max(scores["engagement"], 0.52)
            evidence.append("hint_request")
        if self.REENGAGE_RE.search(low):
            scores["engagement"] = max(scores["engagement"], 0.55)
            evidence.append("continue_signal")

        if self.incorrect_streak >= 2:
            scores["frustration"] = max(scores["frustration"], 0.62)
            evidence.append("incorrect_streak")
        if self.incorrect_streak >= 3:
            scores["overload"] = max(scores["overload"], 0.58)
            evidence.append("failure_overload")
        if self.clarification_streak >= 3:
            scores["prolonged_confusion"] = max(scores["prolonged_confusion"], 0.60)
            evidence.append("prolonged_clarification")
        if seconds_since_turn >= 12.0 and phase not in {"IDLE", ""}:
            scores["overload"] = max(scores["overload"], 0.34)
            evidence.append("long_response_latency")

        if self.correct_streak >= 2:
            scores["confidence"] = max(scores["confidence"], 0.66)
            scores["engagement"] = max(scores["engagement"], 0.56)
            evidence.append("correct_streak")

        confidence = _clamp(0.25 + 0.08 * len(evidence) + 0.04 * min(self.clarification_streak, 3))
        return ModalityEstimate(
            scores=scores,
            confidence=confidence,
            source="performance",
            details={
                "incorrect_streak": self.incorrect_streak,
                "correct_streak": self.correct_streak,
                "clarification_streak": self.clarification_streak,
                "recent_hint_requests": len(self.hint_requests),
                "seconds_since_turn": round(seconds_since_turn, 2),
                "evidence": evidence,
            },
        )

    def record_qa_result(self, correct: bool, partial: bool = False, timestamp: float | None = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        if correct:
            self.correct_streak += 1
            self.incorrect_streak = 0
        else:
            self.incorrect_streak += 1
            self.correct_streak = 0
            self.failure_times.append(now)
        if partial:
            self.clarification_streak = max(self.clarification_streak, 1)
