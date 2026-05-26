from __future__ import annotations

from pathlib import Path

import cv2
import requests

from emotion_engine.utils import FaceDetection, ensure_dir


YUNET_MODEL_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        max_input_size: int = 640,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 20,
        auto_download: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_input_size = max(320, max_input_size)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.auto_download = auto_download
        self.backend_name = "yunet"

        self._ensure_model()
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            if not self._is_lfs_pointer():
                return
            self.model_path.unlink(missing_ok=True)
        if not self.auto_download:
            raise FileNotFoundError(
                f"YuNet model not found at {self.model_path}. "
                f"Download it from {YUNET_MODEL_URLS[0]} and place it there."
            )

        ensure_dir(self.model_path.parent)
        last_error: Exception | None = None
        for url in YUNET_MODEL_URLS:
            try:
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                with self.model_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                if self._is_lfs_pointer():
                    raise RuntimeError(f"Downloaded a Git LFS pointer instead of the ONNX file from {url}")
                return
            except Exception as exc:
                last_error = exc
                try:
                    self.model_path.unlink(missing_ok=True)
                except Exception:
                    pass

        raise RuntimeError(
            "Failed to download the YuNet detector model automatically. "
            f"Place the ONNX file at {self.model_path}. Last error: {last_error}"
        )

    def _is_lfs_pointer(self) -> bool:
        if not self.model_path.exists() or self.model_path.stat().st_size < 256:
            try:
                with self.model_path.open("rb") as handle:
                    return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
            except Exception:
                return False
        return False

    def detect(self, frame_bgr) -> list[FaceDetection]:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = min(1.0, float(self.max_input_size) / float(max(frame_h, frame_w)))

        if scale < 1.0:
            resized = cv2.resize(
                frame_bgr,
                (int(frame_w * scale), int(frame_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = frame_bgr

        self.detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self.detector.detect(resized)

        if faces is None or len(faces) == 0:
            return []

        scale_back = 1.0 / scale
        detections: list[FaceDetection] = []
        for row in faces:
            x, y, w, h = row[:4]
            score = float(row[14])
            if score < self.score_threshold:
                continue

            bbox = (
                int(round(x * scale_back)),
                int(round(y * scale_back)),
                int(round(w * scale_back)),
                int(round(h * scale_back)),
            )
            landmarks = [
                (int(round(row[index] * scale_back)), int(round(row[index + 1] * scale_back)))
                for index in range(4, 14, 2)
            ]
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    score=score,
                    landmarks=landmarks,
                    area=float(max(1, bbox[2]) * max(1, bbox[3])),
                )
            )

        detections.sort(key=lambda item: item.area, reverse=True)
        return detections

    def detect_primary(self, frame_bgr) -> FaceDetection | None:
        detections = self.detect(frame_bgr)
        return detections[0] if detections else None
