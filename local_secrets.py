from __future__ import annotations

import os
from pathlib import Path

_LOADED = False
_CANDIDATES = ("local_secrets.env", ".env.local", ".env")


def load_local_env(base_dir: str | Path | None = None, override: bool = False) -> None:
    global _LOADED
    if _LOADED and not override:
        return
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    for name in _CANDIDATES:
        env_path = root / name
        if not env_path.exists():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or os.getenv(key) in (None, ""):
                os.environ[key] = value
    _LOADED = True
