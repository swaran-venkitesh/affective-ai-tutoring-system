from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import EmotionEngineConfig
from .schemas import EmotionEngineSettings


class SettingsStore:
    def __init__(self, path: Path, config: EmotionEngineConfig) -> None:
        self.path = path
        self.config = config
        self._lock = threading.Lock()
        self._settings = self._coerce_settings({})
        self._load()

    def _coerce_settings(self, data: dict) -> EmotionEngineSettings:
        enabled = self.config.enabled_default
        if self.config.allow_client_toggle:
            enabled = bool(data.get("enabled", enabled))

        show_monitor = bool(data.get("show_monitor", False)) if self.config.show_monitor_ui else False
        face_mesh_overlay = bool(data.get("face_mesh_overlay", False)) if self.config.show_monitor_ui else False

        return EmotionEngineSettings(
            enabled=bool(enabled),
            show_monitor=show_monitor,
            face_mesh_overlay=face_mesh_overlay,
            allow_client_toggle=bool(self.config.allow_client_toggle),
            show_monitor_ui=bool(self.config.show_monitor_ui),
        )

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._settings = self._coerce_settings(data)
        except Exception:
            self._settings = self._coerce_settings({})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": bool(self._settings.enabled),
            "show_monitor": bool(self._settings.show_monitor),
            "face_mesh_overlay": bool(self._settings.face_mesh_overlay),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self) -> EmotionEngineSettings:
        with self._lock:
            return self._coerce_settings(self._settings.to_dict())

    def update(self, **kwargs) -> EmotionEngineSettings:
        with self._lock:
            current = self._settings.to_dict()
            allowed = {
                "show_monitor": bool(kwargs.get("show_monitor", current["show_monitor"])),
                "face_mesh_overlay": bool(kwargs.get("face_mesh_overlay", current["face_mesh_overlay"])),
            }
            if self.config.allow_client_toggle and "enabled" in kwargs:
                allowed["enabled"] = bool(kwargs["enabled"])
            self._settings = self._coerce_settings(allowed)
            self._save()
            return self._coerce_settings(self._settings.to_dict())
