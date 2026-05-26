from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from emotion_engine.utils import EmotionPrediction, clamp


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message="You are using a Python version", category=FutureWarning)


class EmotionRecognizer:
    LABEL_CANONICAL_MAP = {
        "happy": "happiness",
        "sad": "sadness",
    }

    def __init__(
        self,
        model_name: str = "auto_best",
        engine: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_name = self._resolve_model_name(model_name, device)
        self.engine, self.device = self._resolve_runtime(engine, device)
        self.recognizer = None
        self.backend = "emotiefflib"

        try:
            self.recognizer = self._create_recognizer(self.engine, self.device)
        except Exception as exc:
            if engine == "auto" and self.engine == "torch":
                self.engine, self.device = "onnx", "cpu"
                self.recognizer = self._create_recognizer(self.engine, self.device)
            else:
                raise RuntimeError(self._format_error(exc)) from exc

        self.has_direct_va = self.backend == "emonet" or "_mtl" in self.model_name
        if self.backend == "emotiefflib":
            self.labels_display = [
                self.recognizer.idx_to_emotion_class[index]
                for index in sorted(self.recognizer.idx_to_emotion_class)
            ]
            self.labels = [self._canonicalize_label(label.lower()) for label in self.labels_display]
        else:
            self.labels_display = self.recognizer["labels_display"]
            self.labels = [self._canonicalize_label(label.lower()) for label in self.labels_display]
        self.neutral_index = self.labels.index("neutral") if "neutral" in self.labels else -1
        self.runtime_label = f"{self.engine}/{self.device}:{self.model_name}"

    def _resolve_model_name(self, model_name: str, device: str) -> str:
        if model_name not in {"auto", "auto_best"}:
            return model_name
        if self._emonet_available() and (device == "cuda" or (device == "auto" and self._cuda_available())):
            return "emonet_8"
        return "enet_b0_8_va_mtl"

    def _emonet_available(self) -> bool:
        repo_root = Path(__file__).resolve().parents[2] / "external" / "emonet_official" / "emonet-master"
        weights_path = repo_root / "pretrained" / "emonet_8.pth"
        return repo_root.exists() and weights_path.exists()

    def _canonicalize_label(self, label: str) -> str:
        return self.LABEL_CANONICAL_MAP.get(label, label)

    def _resolve_runtime(self, engine: str, device: str) -> tuple[str, str]:
        if engine not in {"auto", "onnx", "torch"}:
            raise ValueError(f"Unsupported engine: {engine}")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported device: {device}")

        if self.model_name.startswith("emonet_"):
            if device == "auto":
                device = "cuda" if self._cuda_available() else "cpu"
            return "torch", device

        if engine == "onnx":
            return "onnx", "cpu"

        if engine == "torch":
            if device == "auto":
                device = "cuda" if self._cuda_available() else "cpu"
            return "torch", device

        if device == "cuda" and self._cuda_available():
            return "torch", "cuda"
        return "onnx", "cpu"

    def _cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _create_recognizer(self, engine: str, device: str):
        if self.model_name.startswith("emonet_"):
            self.backend = "emonet"
            return self._create_emonet_recognizer(device)
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer

        return EmotiEffLibRecognizer(engine=engine, model_name=self.model_name, device=device)

    def _create_emonet_recognizer(self, device: str):
        import torch

        repo_root = Path(__file__).resolve().parents[2] / "external" / "emonet_official" / "emonet-master"
        if not repo_root.exists():
            raise RuntimeError(
                "Official EmoNet repo not found under external/emonet_official/emonet-master. "
                "Download it into the workspace before using the EmoNet backend."
            )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from emonet.models import EmoNet

        n_expression = 8 if self.model_name.endswith("_8") else 5
        state_dict_path = repo_root / "pretrained" / f"emonet_{n_expression}.pth"
        if not state_dict_path.exists():
            raise RuntimeError(f"Missing EmoNet weights: {state_dict_path}")

        torch_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        state_dict = torch.load(str(state_dict_path), map_location="cpu")
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        net = EmoNet(n_expression=n_expression).to(torch_device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        labels_display = (
            ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
            if n_expression == 8
            else ["Neutral", "Happy", "Sad", "Surprise", "Fear"]
        )
        return {
            "net": net,
            "torch": torch,
            "device": torch_device,
            "n_expression": n_expression,
            "labels_display": labels_display,
        }

    def _format_error(self, exc: Exception) -> str:
        if self.model_name.startswith("emonet_"):
            return (
                f"Failed to initialize the emotion model '{self.model_name}'. "
                "The official EmoNet repo should be available under external/emonet_official/emonet-master "
                "with pretrained weights under its pretrained/ directory. "
                f"Original error: {exc}"
            )
        model_url = (
            "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/onnx/"
            f"{self.model_name}.onnx?raw=true"
        )
        return (
            f"Failed to initialize the emotion model '{self.model_name}'. "
            f"EmotiEffLib can auto-download the model on first run; if that fails, "
            f"download it manually from {model_url} and place it in ~/.emotiefflib/. "
            f"Original error: {exc}"
        )

    def _estimate_valence_arousal(self, probabilities: np.ndarray) -> tuple[float, float]:
        valence_weights = np.array([-0.75, -0.55, -0.60, -0.72, 0.95, 0.05, -0.85, 0.18], dtype=np.float32)
        arousal_weights = np.array([0.70, 0.18, 0.25, 0.82, 0.58, -0.18, -0.40, 0.92], dtype=np.float32)
        valence = float(np.dot(probabilities, valence_weights))
        arousal = float(np.dot(probabilities, arousal_weights))
        return clamp(valence, -1.0, 1.0), clamp(arousal, -1.0, 1.0)

    def _select_top_label(self, probabilities: np.ndarray) -> tuple[int, float]:
        top_index = int(np.argmax(probabilities))
        top_confidence = float(probabilities[top_index])
        neutral_confidence = float(probabilities[self.neutral_index]) if self.neutral_index >= 0 else 0.0
        if (
            self.neutral_index >= 0
            and top_index != self.neutral_index
            and top_confidence < 0.38
            and neutral_confidence >= (top_confidence - 0.05)
        ):
            return self.neutral_index, neutral_confidence
        return top_index, top_confidence

    def _calibrate_affect(
        self,
        probabilities: np.ndarray,
        top_index: int,
        top_confidence: float,
        valence: float,
        arousal: float,
    ) -> tuple[float, float]:
        sorted_scores = np.sort(probabilities)[::-1]
        second_confidence = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        margin = max(0.0, top_confidence - second_confidence)
        confidence_scale = clamp(0.30 + (1.10 * margin) + (0.50 * top_confidence), 0.30, 1.0)
        neutral_confidence = float(probabilities[self.neutral_index]) if self.neutral_index >= 0 else 0.0

        if self.neutral_index >= 0 and top_index == self.neutral_index:
            confidence_scale *= 0.68
        elif neutral_confidence >= 0.25 and top_confidence < 0.42:
            confidence_scale *= 0.72
        if top_confidence < 0.30:
            confidence_scale *= 0.75

        valence *= confidence_scale
        arousal *= 0.45 + (0.55 * confidence_scale)
        return clamp(valence, -1.0, 1.0), clamp(arousal, -1.0, 1.0)

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Emotion inference received an empty face crop.")

        if self.backend == "emonet":
            return self._predict_emonet(face_bgr)

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        start = perf_counter()
        features = np.asarray(self.recognizer.extract_features(face_rgb), dtype=np.float32)
        _, scores = self.recognizer.classify_emotions(features, logits=False)
        latency_ms = (perf_counter() - start) * 1000.0

        scores = np.asarray(scores, dtype=np.float32)[0]
        if self.has_direct_va and len(scores) >= len(self.labels) + 2:
            emotion_scores = scores[: len(self.labels)]
            valence = clamp(float(scores[len(self.labels)]), -1.0, 1.0)
            arousal = clamp(float(scores[len(self.labels) + 1]), -1.0, 1.0)
            va_source = "direct"
        else:
            emotion_scores = scores[: len(self.labels)]
            valence, arousal = self._estimate_valence_arousal(emotion_scores)
            va_source = "estimated"

        emotion_scores = emotion_scores.astype(np.float32)
        emotion_scores /= max(1e-6, float(emotion_scores.sum()))
        top_index, top_confidence = self._select_top_label(emotion_scores)
        valence, arousal = self._calibrate_affect(emotion_scores, top_index, top_confidence, valence, arousal)
        top_label = self.labels[top_index]
        probabilities = {
            self.labels[index]: float(emotion_scores[index]) for index in range(len(self.labels))
        }

        return EmotionPrediction(
            emotion=top_label,
            confidence=top_confidence,
            probabilities=probabilities,
            valence=valence,
            arousal=arousal,
            va_source=va_source,
            latency_ms=latency_ms,
            feature_vector=features[0].copy(),
        )

    def _predict_emonet(self, face_bgr: np.ndarray) -> EmotionPrediction:
        torch = self.recognizer["torch"]
        net = self.recognizer["net"]
        device = self.recognizer["device"]

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        image_tensor = torch.tensor(face_rgb, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0

        start = perf_counter()
        with torch.no_grad():
            output = net(image_tensor.unsqueeze(0))
            logits = output["expression"][0]
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32)
            valence = clamp(float(output["valence"].clamp(-1.0, 1.0).detach().cpu().item()), -1.0, 1.0)
            arousal = clamp(float(output["arousal"].clamp(-1.0, 1.0).detach().cpu().item()), -1.0, 1.0)
        latency_ms = (perf_counter() - start) * 1000.0

        probs /= max(1e-6, float(probs.sum()))
        top_index, top_confidence = self._select_top_label(probs)
        valence, arousal = self._calibrate_affect(probs, top_index, top_confidence, valence, arousal)
        top_label = self.labels[top_index]
        probabilities = {self.labels[index]: float(probs[index]) for index in range(len(self.labels))}

        return EmotionPrediction(
            emotion=top_label,
            confidence=top_confidence,
            probabilities=probabilities,
            valence=valence,
            arousal=arousal,
            va_source="direct",
            latency_ms=latency_ms,
            feature_vector=probs.copy(),
        )
