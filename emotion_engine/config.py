from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(slots=True)
class EmotionEngineConfig:
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    enabled_default: bool = field(default_factory=lambda: _env_flag("EMOTION_ENGINE_ENABLED", True))
    allow_client_toggle: bool = field(default_factory=lambda: _env_flag("EMOTION_ENGINE_ALLOW_CLIENT_TOGGLE", False))
    show_monitor_ui: bool = field(default_factory=lambda: _env_flag("EMOTION_MONITOR_UI_ENABLED", False))
    validation_log_enabled: bool = field(default_factory=lambda: _env_flag("EMOTION_ENGINE_VALIDATION_LOGS", True))
    text_model_id: str = "SamLowe/roberta-base-go_emotions"
    face_model_name: str = "auto_best"
    face_engine: str = "auto"
    face_device: str = "auto"
    camera_frame_interval_ms: int = 1200
    emotion_infer_every: int = 2
    yu_net_score_threshold: float = 0.7
    settings_filename: str = "emotion_engine_settings.json"
    output_dir_name: str = "emotion_output"
    text_weight: float = 0.45
    performance_weight: float = 0.33
    camera_weight: float = 0.12
    face_weight: float = 0.10
    speech_weight: float = 0.0
    productive_confusion_threshold: float = 0.50
    productive_confusion_max_turns: int = 2
    prolonged_confusion_seconds: float = 70.0
    strong_state_threshold: float = 0.62
    moderation_threshold: float = 0.42

    @property
    def output_dir(self) -> Path:
        return self.repo_root / self.output_dir_name

    @property
    def settings_path(self) -> Path:
        return self.output_dir / self.settings_filename

    @property
    def yu_net_model_path(self) -> Path:
        return self.repo_root / "emotion_engine" / "models" / "face_detection_yunet_2023mar.onnx"
