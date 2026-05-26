from __future__ import annotations

import contextlib
import io
import math
import re
import threading
from typing import Any

import numpy as np

from .config import EmotionEngineConfig
from .schemas import ModalityEstimate, TARGET_AFFECT_LABELS


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class TextAffectClassifier:
    _MODEL_CACHE: dict[str, tuple[Any, Any, dict[int, str], str]] = {}
    _CACHE_LOCK = threading.Lock()

    SELF_DOUBT_RE = re.compile(
        r"\b(i can't|i cannot|i'm bad|i am bad|i feel dumb|i am dumb|not good at|"
        r"i suck|i'm stupid|i am stupid|i will fail|i'm going to fail|"
        r"i don't think i can|i am not smart enough)\b",
        re.I,
    )
    CONFUSION_RE = re.compile(
        r"\b(i don't understand|i do not understand|dont understand|do not understand|confused|unclear|what do you mean|how does this work|"
        r"i'm lost|still lost|doesn't make sense|why is this happening|make it simple|make this simple|simpler please|break it down|slow down)\b",
        re.I,
    )
    FRUSTRATION_RE = re.compile(
        r"\b(frustrat|annoy|stuck|this is hard|this is impossible|hate this|"
        r"fed up|why isn't it working)\b",
        re.I,
    )
    BOREDOM_RE = re.compile(
        r"\b(bored|boring|sleepy|tired of this|not interested|zoned out)\b",
        re.I,
    )
    OVERLOAD_RE = re.compile(
        r"\b(too much|overwhelming|overloaded|so many steps|too many things|brain full)\b",
        re.I,
    )
    CONFIDENCE_RE = re.compile(
        r"\b(i got it|that makes sense|i understand|easy|i know this|i can do it|"
        r"confident|ready)\b",
        re.I,
    )
    ENGAGEMENT_RE = re.compile(
        r"\b(show me|let's do it|next question|another example|i want to learn|"
        r"can we continue|go deeper|teach me|let's continue)\b",
        re.I,
    )
    HINT_RE = re.compile(r"\b(hint|small hint|clue|nudge)\b", re.I)

    LABEL_MAP = {
        "confusion": "confusion",
        "nervousness": "anxiety_self_doubt",
        "fear": "anxiety_self_doubt",
        "sadness": "anxiety_self_doubt",
        "disappointment": "frustration",
        "annoyance": "frustration",
        "anger": "frustration",
        "disapproval": "frustration",
        "embarrassment": "anxiety_self_doubt",
        "remorse": "anxiety_self_doubt",
        "grief": "anxiety_self_doubt",
        "joy": "confidence",
        "optimism": "confidence",
        "pride": "confidence",
        "approval": "confidence",
        "relief": "confidence",
        "curiosity": "engagement",
        "desire": "engagement",
        "excitement": "engagement",
        "realization": "engagement",
        "neutral": "engagement",
    }

    def __init__(self, config: EmotionEngineConfig) -> None:
        self.config = config
        self.backend = "heuristic"
        self.model = None
        self.tokenizer = None
        self.id2label: dict[int, str] = {}
        self._load_classifier()

    def _load_classifier(self) -> None:
        with self._CACHE_LOCK:
            cached = self._MODEL_CACHE.get(self.config.text_model_id)
            if cached is not None:
                self.tokenizer, self.model, self.id2label, self.backend = cached
                return

            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_id, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.text_model_id,
                        local_files_only=True,
                    )
                id2label = {
                    int(index): str(label).lower()
                    for index, label in (model.config.id2label or {}).items()
                }
                backend = "roberta_go_emotions"
            except Exception:
                tokenizer = None
                model = None
                id2label = {}
                backend = "heuristic"

            self._MODEL_CACHE[self.config.text_model_id] = (tokenizer, model, id2label, backend)
            self.tokenizer = tokenizer
            self.model = model
            self.id2label = id2label
            self.backend = backend

    def _empty_scores(self) -> dict[str, float]:
        return {label: 0.0 for label in TARGET_AFFECT_LABELS}

    def _classifier_probs(self, text: str) -> dict[str, float]:
        if self.model is None or self.tokenizer is None:
            return {}
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        return {
            self.id2label.get(index, f"label_{index}"): float(prob)
            for index, prob in enumerate(probs)
        }

    def _map_classifier_scores(self, label_probs: dict[str, float]) -> dict[str, float]:
        scores = self._empty_scores()
        for label, prob in label_probs.items():
            target = self.LABEL_MAP.get(label)
            if not target:
                continue
            if target == "confidence":
                scores[target] = max(scores[target], prob)
                scores["engagement"] = max(scores["engagement"], prob * 0.5)
            elif target == "engagement":
                scores[target] = max(scores[target], prob)
            else:
                scores[target] = max(scores[target], prob)

        scores["overload"] = _clamp(
            0.45 * scores["confusion"]
            + 0.35 * scores["anxiety_self_doubt"]
            + 0.20 * scores["frustration"]
        )
        return scores

    def _heuristic_scores(self, text: str) -> tuple[dict[str, float], list[str]]:
        lowered = text.strip().lower()
        scores = self._empty_scores()
        evidence: list[str] = []

        if self.CONFUSION_RE.search(lowered):
            scores["confusion"] = max(scores["confusion"], 0.72)
            scores["engagement"] = max(scores["engagement"], 0.52)
            evidence.append("confusion_phrase")
        if self.FRUSTRATION_RE.search(lowered):
            scores["frustration"] = max(scores["frustration"], 0.74)
            evidence.append("frustration_phrase")
        if self.BOREDOM_RE.search(lowered):
            scores["boredom"] = max(scores["boredom"], 0.68)
            evidence.append("boredom_phrase")
        if self.OVERLOAD_RE.search(lowered):
            scores["overload"] = max(scores["overload"], 0.76)
            evidence.append("overload_phrase")
        if self.SELF_DOUBT_RE.search(lowered):
            scores["anxiety_self_doubt"] = max(scores["anxiety_self_doubt"], 0.82)
            evidence.append("self_doubt_phrase")
        if self.CONFIDENCE_RE.search(lowered):
            scores["confidence"] = max(scores["confidence"], 0.72)
            scores["engagement"] = max(scores["engagement"], 0.55)
            evidence.append("confidence_phrase")
        if self.ENGAGEMENT_RE.search(lowered):
            scores["engagement"] = max(scores["engagement"], 0.68)
            evidence.append("engagement_phrase")
        if self.HINT_RE.search(lowered):
            scores["confusion"] = max(scores["confusion"], 0.42)
            scores["engagement"] = max(scores["engagement"], 0.46)
            evidence.append("hint_request")

        if not evidence and len(lowered.split()) >= 5 and lowered.endswith("?"):
            scores["engagement"] = max(scores["engagement"], 0.42)
            evidence.append("active_question")

        return scores, evidence

    def analyze(self, text: str, history: list[str] | None = None) -> ModalityEstimate:
        history = history or []
        text = (text or "").strip()
        classifier_probs = self._classifier_probs(text)
        classifier_scores = self._map_classifier_scores(classifier_probs)
        heuristic_scores, evidence = self._heuristic_scores(text)
        scores = self._empty_scores()

        for label in TARGET_AFFECT_LABELS:
            scores[label] = _clamp(max(classifier_scores[label], heuristic_scores[label]))

        if len(history) >= 2:
            recent = " ".join(history[-2:]) + " " + text
            if self.CONFUSION_RE.search(recent.lower()) and scores["confusion"] >= 0.45:
                scores["prolonged_confusion"] = max(scores["prolonged_confusion"], 0.48)
            if self.FRUSTRATION_RE.search(recent.lower()) and scores["frustration"] >= 0.45:
                scores["frustration"] = _clamp(scores["frustration"] + 0.08)
            if self.SELF_DOUBT_RE.search(recent.lower()):
                scores["anxiety_self_doubt"] = _clamp(scores["anxiety_self_doubt"] + 0.08)

        sorted_raw = sorted(classifier_probs.items(), key=lambda item: item[1], reverse=True)
        top_raw = sorted_raw[:6]
        top_score = max(scores.values()) if scores else 0.0
        if self.backend == "heuristic":
            confidence = max(0.25, top_score)
        else:
            confidence = _clamp(0.30 + (0.70 * max([prob for _, prob in top_raw] or [0.0])))

        return ModalityEstimate(
            scores=scores,
            confidence=confidence,
            source=self.backend,
            details={
                "backend": self.backend,
                "raw_labels": top_raw,
                "evidence": evidence,
                "input_text": text[:600],
            },
        )
