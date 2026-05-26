from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    score: float
    landmarks: list[tuple[int, int]]
    area: float


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    valence: Optional[float]
    arousal: Optional[float]
    va_source: str
    latency_ms: float
    feature_vector: Optional[np.ndarray] = None


@dataclass
class SmoothedEmotionState:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    valence: float
    arousal: float
    va_source: str
    instability: float
    raw_emotion: str
    raw_confidence: float


@dataclass
class TutoringStateResult:
    label: str
    confidence: float
    source: str
    rationale: str


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def crop_face(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 0.18,
    square: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = bbox
    if square:
        side = max(w, h) * (1.0 + (2.0 * padding))
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        x1 = int(round(cx - (side / 2.0)))
        y1 = int(round(cy - (side / 2.0)))
        x2 = int(round(cx + (side / 2.0)))
        y2 = int(round(cy + (side / 2.0)))
    else:
        x1 = int(round(x - (w * padding)))
        y1 = int(round(y - (h * padding)))
        x2 = int(round(x + w + (w * padding)))
        y2 = int(round(y + h + (h * padding)))

    frame_h, frame_w = frame_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame_bgr.dtype), (0, 0, 0, 0)

    crop = frame_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)


def center_crop(frame_bgr: np.ndarray, ratio: float = 0.45) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame_bgr.shape[:2]
    side = int(min(height, width) * ratio)
    side = max(32, side)
    x = max(0, (width - side) // 2)
    y = max(0, (height - side) // 2)
    crop = frame_bgr[y : y + side, x : x + side].copy()
    return crop, (x, y, side, side)


def compute_bbox_motion(
    previous_bbox: Optional[tuple[int, int, int, int]],
    current_bbox: tuple[int, int, int, int],
) -> float:
    if previous_bbox is None:
        return 0.0

    prev_x, prev_y, prev_w, prev_h = previous_bbox
    curr_x, curr_y, curr_w, curr_h = current_bbox

    prev_cx = prev_x + (prev_w / 2.0)
    prev_cy = prev_y + (prev_h / 2.0)
    curr_cx = curr_x + (curr_w / 2.0)
    curr_cy = curr_y + (curr_h / 2.0)

    center_shift = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
    normalizer = max(1.0, (prev_w + prev_h + curr_w + curr_h) / 4.0)
    return clamp(center_shift / normalizer, 0.0, 1.0)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    capture = None
    for backend in backends:
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()

    if capture is None or not capture.isOpened():
        raise RuntimeError(
            f"Unable to open webcam index {camera_index}. "
            "Check camera permissions and whether another app is already using the webcam."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


def save_frame(frame_bgr: np.ndarray, output_dir: str | Path, prefix: str = "frame") -> Path:
    directory = ensure_dir(output_dir)
    output_path = directory / f"{prefix}_{timestamp_string()}.jpg"
    success = cv2.imwrite(str(output_path), frame_bgr)
    if not success:
        raise RuntimeError(f"Failed to write image to {output_path}")
    return output_path


class FPSMeter:
    def __init__(self, ema_alpha: float = 0.2) -> None:
        self.ema_alpha = ema_alpha
        self.last_time: Optional[float] = None
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self.last_time is not None:
            delta = max(1e-6, now - self.last_time)
            instant_fps = 1.0 / delta
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (self.ema_alpha * instant_fps) + ((1.0 - self.ema_alpha) * self.fps)
        self.last_time = now
        return self.fps


class CsvLogger:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        ensure_dir(self.csv_path.parent)
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp",
                "frame_index",
                "raw_emotion",
                "emotion_confidence",
                "valence",
                "arousal",
                "tutoring_label",
                "tutoring_confidence",
                "label_source",
            ],
        )
        self.writer.writeheader()

    def log(
        self,
        frame_index: int,
        emotion_state: Optional[SmoothedEmotionState],
        tutoring_state: Optional[TutoringStateResult],
    ) -> None:
        self.writer.writerow(
            {
                "timestamp": iso_timestamp(),
                "frame_index": frame_index,
                "raw_emotion": emotion_state.emotion if emotion_state else "",
                "emotion_confidence": f"{emotion_state.confidence:.4f}" if emotion_state else "",
                "valence": f"{emotion_state.valence:.4f}" if emotion_state else "",
                "arousal": f"{emotion_state.arousal:.4f}" if emotion_state else "",
                "tutoring_label": tutoring_state.label if tutoring_state else "",
                "tutoring_confidence": f"{tutoring_state.confidence:.4f}" if tutoring_state else "",
                "label_source": tutoring_state.source if tutoring_state else "",
            }
        )
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


class AlertSounder:
    def __init__(self, enabled: bool = True, cooldown_seconds: float = 0.9) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_beep_time = 0.0

    def eye_closure_beep(self) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_beep_time < self.cooldown_seconds:
            return
        self.last_beep_time = now
        if winsound is not None:
            winsound.Beep(1200, 120)
            winsound.Beep(900, 120)
        else:
            print("\a", end="", flush=True)

    def away_attention_beep(self) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_beep_time < max(1.3, self.cooldown_seconds):
            return
        self.last_beep_time = now
        if winsound is not None:
            winsound.Beep(850, 180)
            winsound.Beep(850, 180)
        else:
            print("\a", end="", flush=True)
