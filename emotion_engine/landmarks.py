from __future__ import annotations

import contextlib
import math
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from google.protobuf import message_factory

from emotion_engine.utils import clamp, crop_face


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOGGING_MIN_SEVERITY", "3")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
)

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity("error")
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass


if not hasattr(message_factory, "GetMessageClass"):
    _proto_factory = message_factory.MessageFactory()

    def _get_message_class(descriptor):
        return _proto_factory.GetPrototype(descriptor)

    message_factory.GetMessageClass = _get_message_class


import mediapipe as mp


try:
    mp_solutions = mp.solutions
except AttributeError:
    from mediapipe.python import solutions as mp_solutions

mp_face_mesh = mp_solutions.face_mesh
FACE_MESH_CONTOURS = tuple(mp_face_mesh.FACEMESH_CONTOURS)
FACE_MESH_TESSELATION = tuple(mp_face_mesh.FACEMESH_TESSELATION)
FACE_MESH_IRISES = tuple(mp_face_mesh.FACEMESH_IRISES)

L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

POSE_LANDMARK_INDICES = {
    "nose": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass
class AttentionState:
    face_present: bool
    face_presence_ratio: float
    face_present_duration: float
    face_absent_duration: float
    ear_left: float
    ear_right: float
    ear_avg: float
    eyes_closed: bool
    blink_count: int
    blink_rate_per_min: float
    last_blink_duration: float
    current_eye_closure_duration: float
    prolonged_eye_closure: bool
    prolonged_eye_closure_count: int
    mouth_open_ratio: float
    yawn_active: bool
    yawn_count: int
    gaze_horizontal: str
    gaze_vertical: str
    gaze_direction: str
    head_direction: str
    head_yaw: float
    head_pitch: float
    looking_away: bool
    look_straight_ratio: float
    current_away_duration: float
    total_away_duration: float
    landmarks: Optional[list[tuple[int, int]]] = None


@contextlib.contextmanager
def suppress_native_stderr():
    original_stderr_fd = None
    saved_stderr_fd = None
    devnull_fd = None
    try:
        original_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, original_stderr_fd)
        yield
    finally:
        if saved_stderr_fd is not None and original_stderr_fd is not None:
            os.dup2(saved_stderr_fd, original_stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
        if saved_stderr_fd is not None:
            os.close(saved_stderr_fd)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eye_ear(points: list[int], landmarks: list[tuple[int, int]]) -> float:
    p1 = landmarks[points[0]]
    p2 = landmarks[points[1]]
    p3 = landmarks[points[2]]
    p4 = landmarks[points[3]]
    p5 = landmarks[points[4]]
    p6 = landmarks[points[5]]
    return (float(_dist(p2, p6)) + float(_dist(p3, p5))) / (2.0 * (float(_dist(p1, p4)) + 1e-6))


def _point_mean(indices: list[int], landmarks: list[tuple[int, int]]) -> tuple[float, float]:
    xs = [landmarks[index][0] for index in indices]
    ys = [landmarks[index][1] for index in indices]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


class FaceMeshMonitor:
    def __init__(
        self,
        ear_closed_threshold: float = 0.20,
        blink_min_duration: float = 0.06,
        blink_max_duration: float = 0.80,
        prolonged_closure_seconds: float = 3.40,
        yawn_ratio_threshold: float = 0.38,
        yawn_min_duration: float = 1.0,
        face_presence_window: int = 180,
        look_window: int = 90,
        roi_padding: float = 0.22,
    ) -> None:
        self.ear_closed_threshold = ear_closed_threshold
        self.blink_min_duration = blink_min_duration
        self.blink_max_duration = blink_max_duration
        self.prolonged_closure_seconds = prolonged_closure_seconds
        self.yawn_ratio_threshold = yawn_ratio_threshold
        self.yawn_min_duration = yawn_min_duration
        self.roi_padding = roi_padding

        self.blink_count = 0
        self.last_blink_duration = 0.0
        self.prolonged_eye_closure_count = 0
        self.yawn_count = 0
        self.closed_since: Optional[float] = None
        self.prolonged_closure_active = False
        self.yawn_since: Optional[float] = None
        self.yawn_active = False
        self.face_present_since: Optional[float] = None
        self.face_absent_since: Optional[float] = None
        self.looking_away_since: Optional[float] = None
        self.total_away_duration = 0.0
        self.head_yaw_baseline = 0.0
        self.head_pitch_baseline = 0.0
        self.head_yaw_smoothed = 0.0
        self.head_pitch_smoothed = 0.0
        self.blink_timestamps: deque[float] = deque()
        self.face_presence_history: deque[int] = deque(maxlen=face_presence_window)
        self.look_straight_history: deque[int] = deque(maxlen=look_window)

        with suppress_native_stderr():
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def close(self) -> None:
        try:
            self.face_mesh.close()
        except Exception:
            pass

    def _build_no_face_state(self, now: float) -> AttentionState:
        self.face_presence_history.append(0)
        self.look_straight_history.append(0)

        if self.face_absent_since is None:
            self.face_absent_since = now
        self.face_present_since = None

        if self.closed_since is not None:
            self.closed_since = None
            self.prolonged_closure_active = False
        self.yawn_since = None
        self.yawn_active = False

        presence_ratio = float(sum(self.face_presence_history) / max(1, len(self.face_presence_history)))
        absent_duration = now - self.face_absent_since
        return AttentionState(
            face_present=False,
            face_presence_ratio=presence_ratio,
            face_present_duration=0.0,
            face_absent_duration=absent_duration,
            ear_left=0.0,
            ear_right=0.0,
            ear_avg=0.0,
            eyes_closed=False,
            blink_count=self.blink_count,
            blink_rate_per_min=self._blink_rate(now),
            last_blink_duration=self.last_blink_duration,
            current_eye_closure_duration=0.0,
            prolonged_eye_closure=False,
            prolonged_eye_closure_count=self.prolonged_eye_closure_count,
            mouth_open_ratio=0.0,
            yawn_active=False,
            yawn_count=self.yawn_count,
            gaze_horizontal="unknown",
            gaze_vertical="unknown",
            gaze_direction="unknown",
            head_direction="unknown",
            head_yaw=0.0,
            head_pitch=0.0,
            looking_away=False,
            look_straight_ratio=float(sum(self.look_straight_history) / max(1, len(self.look_straight_history))),
            current_away_duration=0.0,
            total_away_duration=self.total_away_duration,
            landmarks=None,
        )

    def _blink_rate(self, now: float) -> float:
        while self.blink_timestamps and (now - self.blink_timestamps[0] > 60.0):
            self.blink_timestamps.popleft()
        return float(len(self.blink_timestamps))

    def _eye_gaze_ratio(
        self,
        iris_indices: list[int],
        corner_a: int,
        corner_b: int,
        upper_lid: int,
        lower_lid: int,
        landmarks: list[tuple[int, int]],
    ) -> tuple[float, float]:
        iris_x, iris_y = _point_mean(iris_indices, landmarks)
        x1, y1 = landmarks[corner_a]
        x2, y2 = landmarks[corner_b]
        horizontal = (iris_x - min(x1, x2)) / (abs(x2 - x1) + 1e-6)
        vertical = (iris_y - min(y1, y2, landmarks[upper_lid][1], landmarks[lower_lid][1])) / (
            abs(landmarks[lower_lid][1] - landmarks[upper_lid][1]) + 1e-6
        )
        return float(horizontal), float(vertical)

    def _classify_axis(self, value: float, low: float = 0.35, high: float = 0.65) -> str:
        if value < low:
            return "left_or_up"
        if value > high:
            return "right_or_down"
        return "center"

    def _estimate_head_pose(
        self,
        landmarks: list[tuple[int, int]],
        frame_shape: tuple[int, ...],
        yaw_hint: float,
        pitch_hint: float,
    ) -> tuple[float, float] | None:
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return None

        image_points = np.array(
            [
                landmarks[POSE_LANDMARK_INDICES["nose"]],
                landmarks[POSE_LANDMARK_INDICES["chin"]],
                landmarks[POSE_LANDMARK_INDICES["left_eye_outer"]],
                landmarks[POSE_LANDMARK_INDICES["right_eye_outer"]],
                landmarks[POSE_LANDMARK_INDICES["mouth_left"]],
                landmarks[POSE_LANDMARK_INDICES["mouth_right"]],
            ],
            dtype=np.float64,
        )

        focal_length = float(max(frame_w, frame_h))
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_w / 2.0],
                [0.0, focal_length, frame_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                POSE_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            projection_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        except Exception:
            return None

        pose_pitch_deg = float(euler_angles[0, 0])
        pose_yaw_deg = float(euler_angles[1, 0])

        if abs(yaw_hint) >= 0.08 and pose_yaw_deg != 0.0:
            if math.copysign(1.0, pose_yaw_deg) != math.copysign(1.0, yaw_hint):
                pose_yaw_deg *= -1.0
        if abs(pitch_hint) >= 0.08 and pose_pitch_deg != 0.0:
            if math.copysign(1.0, pose_pitch_deg) != math.copysign(1.0, pitch_hint):
                pose_pitch_deg *= -1.0

        pose_yaw = clamp(pose_yaw_deg / 30.0, -1.0, 1.0)
        pose_pitch = clamp(pose_pitch_deg / 24.0, -1.0, 1.0)
        return float(pose_yaw), float(pose_pitch)

    def process(
        self,
        frame_bgr,
        detection_bbox: Optional[tuple[int, int, int, int]],
        timestamp: Optional[float] = None,
    ) -> AttentionState:
        now = timestamp if timestamp is not None else time.time()
        if detection_bbox is None:
            return self._build_no_face_state(now)

        face_crop, crop_bbox = crop_face(frame_bgr, detection_bbox, padding=self.roi_padding, square=True)
        if face_crop.size == 0:
            return self._build_no_face_state(now)

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with suppress_native_stderr():
            result = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_face_landmarks:
            return self._build_no_face_state(now)

        crop_x, crop_y, crop_w, crop_h = crop_bbox
        face = result.multi_face_landmarks[0]
        landmarks = [
            (
                int(crop_x + (point.x * crop_w)),
                int(crop_y + (point.y * crop_h)),
            )
            for point in face.landmark
        ]

        self.face_presence_history.append(1)
        if self.face_present_since is None:
            self.face_present_since = now
        self.face_absent_since = None
        face_present_duration = now - self.face_present_since
        presence_ratio = float(sum(self.face_presence_history) / max(1, len(self.face_presence_history)))

        ear_left = _eye_ear(L_EYE, landmarks)
        ear_right = _eye_ear(R_EYE, landmarks)
        ear_avg = (ear_left + ear_right) / 2.0
        eyes_closed = ear_avg < self.ear_closed_threshold

        current_closure_duration = 0.0
        prolonged_eye_closure = False
        if eyes_closed:
            if self.closed_since is None:
                self.closed_since = now
                self.prolonged_closure_active = False
            current_closure_duration = now - self.closed_since
            if current_closure_duration >= self.prolonged_closure_seconds:
                prolonged_eye_closure = True
                if not self.prolonged_closure_active:
                    self.prolonged_eye_closure_count += 1
                    self.prolonged_closure_active = True
        elif self.closed_since is not None:
            duration = now - self.closed_since
            if self.blink_min_duration <= duration <= self.blink_max_duration:
                self.blink_count += 1
                self.last_blink_duration = duration
                self.blink_timestamps.append(now)
            self.closed_since = None
            self.prolonged_closure_active = False

        blink_rate = self._blink_rate(now)

        mouth_open_ratio = _dist(landmarks[13], landmarks[14]) / (_dist(landmarks[78], landmarks[308]) + 1e-6)
        if mouth_open_ratio >= self.yawn_ratio_threshold:
            if self.yawn_since is None:
                self.yawn_since = now
            if (now - self.yawn_since) >= self.yawn_min_duration:
                if not self.yawn_active:
                    self.yawn_count += 1
                self.yawn_active = True
        else:
            self.yawn_since = None
            self.yawn_active = False

        gaze_left_h, gaze_left_v = self._eye_gaze_ratio(LEFT_IRIS, 362, 263, 386, 374, landmarks)
        gaze_right_h, gaze_right_v = self._eye_gaze_ratio(RIGHT_IRIS, 33, 133, 159, 145, landmarks)
        gaze_h = (gaze_left_h + gaze_right_h) / 2.0
        gaze_v = (gaze_left_v + gaze_right_v) / 2.0
        gaze_horizontal = self._classify_axis(gaze_h)
        gaze_vertical = self._classify_axis(gaze_v)
        if gaze_horizontal == "center" and gaze_vertical == "center":
            gaze_direction = "center"
        elif gaze_horizontal != "center":
            gaze_direction = "left" if gaze_horizontal == "left_or_up" else "right"
        else:
            gaze_direction = "up" if gaze_vertical == "left_or_up" else "down"

        left_face = landmarks[234]
        right_face = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]
        nose = landmarks[1]

        yaw_ratio = (nose[0] - left_face[0]) / (max(1e-6, right_face[0] - left_face[0]))
        yaw_hint = clamp((yaw_ratio - 0.5) * 2.0, -1.0, 1.0)
        pitch_ratio = (nose[1] - forehead[1]) / (max(1e-6, chin[1] - forehead[1]))
        pitch_hint = clamp((pitch_ratio - 0.52) * 2.0, -1.0, 1.0)
        pose_estimate = self._estimate_head_pose(landmarks, frame_bgr.shape, yaw_hint, pitch_hint)
        if pose_estimate is None:
            raw_head_yaw = yaw_hint
            raw_head_pitch = pitch_hint
        else:
            pose_yaw, pose_pitch = pose_estimate
            raw_head_yaw = (0.82 * pose_yaw) + (0.18 * yaw_hint)
            raw_head_pitch = (0.82 * pose_pitch) + (0.18 * pitch_hint)

        if gaze_direction == "center" and abs(raw_head_yaw - self.head_yaw_baseline) < 0.42:
            self.head_yaw_baseline = (0.965 * self.head_yaw_baseline) + (0.035 * raw_head_yaw)
        if gaze_direction == "center" and abs(raw_head_pitch - self.head_pitch_baseline) < 0.46:
            self.head_pitch_baseline = (0.97 * self.head_pitch_baseline) + (0.03 * raw_head_pitch)

        head_yaw = clamp(raw_head_yaw - self.head_yaw_baseline, -1.0, 1.0)
        head_pitch = clamp(raw_head_pitch - self.head_pitch_baseline, -1.0, 1.0)
        self.head_yaw_smoothed = (0.80 * self.head_yaw_smoothed) + (0.20 * head_yaw)
        self.head_pitch_smoothed = (0.84 * self.head_pitch_smoothed) + (0.16 * head_pitch)
        head_yaw = float(self.head_yaw_smoothed)
        head_pitch = float(self.head_pitch_smoothed)

        abs_yaw = abs(head_yaw)
        abs_pitch = abs(head_pitch)
        if abs_yaw < 0.38 and abs_pitch < 0.46:
            head_direction = "forward"
        elif abs_pitch >= (abs_yaw + 0.16) and abs_pitch >= 0.58:
            head_direction = "up" if head_pitch < 0 else "down"
        elif abs_yaw >= 0.42:
            head_direction = "left" if head_yaw < 0 else "right"
        else:
            head_direction = "forward"

        strong_head_away = abs_yaw >= 0.58 or head_pitch <= -0.70
        gaze_drift = (
            (gaze_direction in {"left", "right"} and abs_yaw >= 0.34)
            or (gaze_direction == "up" and abs_pitch >= 0.52)
        )
        looking_away = strong_head_away or (gaze_drift and head_direction != "forward")
        if looking_away:
            if self.looking_away_since is None:
                self.looking_away_since = now
            current_away_duration = now - self.looking_away_since
        else:
            if self.looking_away_since is not None:
                self.total_away_duration += now - self.looking_away_since
            self.looking_away_since = None
            current_away_duration = 0.0

        look_straight = 1 if not looking_away else 0
        self.look_straight_history.append(look_straight)
        look_straight_ratio = float(sum(self.look_straight_history) / max(1, len(self.look_straight_history)))

        return AttentionState(
            face_present=True,
            face_presence_ratio=presence_ratio,
            face_present_duration=face_present_duration,
            face_absent_duration=0.0,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            eyes_closed=eyes_closed,
            blink_count=self.blink_count,
            blink_rate_per_min=blink_rate,
            last_blink_duration=self.last_blink_duration,
            current_eye_closure_duration=current_closure_duration,
            prolonged_eye_closure=prolonged_eye_closure,
            prolonged_eye_closure_count=self.prolonged_eye_closure_count,
            mouth_open_ratio=mouth_open_ratio,
            yawn_active=self.yawn_active,
            yawn_count=self.yawn_count,
            gaze_horizontal="left" if gaze_horizontal == "left_or_up" else "right" if gaze_horizontal == "right_or_down" else "center",
            gaze_vertical="up" if gaze_vertical == "left_or_up" else "down" if gaze_vertical == "right_or_down" else "center",
            gaze_direction=gaze_direction,
            head_direction=head_direction,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            looking_away=looking_away,
            look_straight_ratio=look_straight_ratio,
            current_away_duration=current_away_duration,
            total_away_duration=self.total_away_duration
            + (now - self.looking_away_since if self.looking_away_since is not None else 0.0),
            landmarks=landmarks,
        )
