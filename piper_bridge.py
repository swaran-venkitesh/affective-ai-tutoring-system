import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import soundfile as sf


PLAY_SHARED_ROOT = Path(os.environ.get("PLAY_SHARED_ROOT", r"E:\Tutor\all_in_one\PLAY"))
LOCAL_TTS_PRETTIFY_PATH = Path(__file__).with_name("tts_prettify.py")
SHARED_TTS_PRETTIFY_PATH = PLAY_SHARED_ROOT / "tts_prettify.py"
DEFAULT_PIPER_MODEL = Path(
    os.environ.get("PIPER_MODEL", r"E:\piper\voices\en_US-lessac-medium.onnx")
)
DEFAULT_PIPER_SPEAKER = "Lessac"
PIPER_TIMEOUT_SEC = float(os.environ.get("PIPER_TIMEOUT_SEC", "30"))
PIPER_SPEAKER_MODELS = {
    DEFAULT_PIPER_SPEAKER.lower(): DEFAULT_PIPER_MODEL,
    "piper": DEFAULT_PIPER_MODEL,
}

_tts_friendly_func: Optional[Callable[..., str]] = None
_sample_rate_cache: Dict[str, int] = {}


def _load_tts_friendly() -> Callable[..., str]:
    global _tts_friendly_func
    if _tts_friendly_func is not None:
        return _tts_friendly_func

    def _identity(text: str, **_kwargs) -> str:
        return text

    funcs: list[Callable[..., str]] = []
    for idx, path in enumerate((LOCAL_TTS_PRETTIFY_PATH, SHARED_TTS_PRETTIFY_PATH)):
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"tts_prettify_{idx}", str(path)
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load tts_prettify module.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, "tts_friendly", None)
            if callable(func):
                funcs.append(func)
        except Exception:
            continue

    if not funcs:
        _tts_friendly_func = _identity
        return _tts_friendly_func

    def _composed(text: str, **kwargs) -> str:
        out = text
        for func in funcs:
            out = func(out, **kwargs)
        return out

    _tts_friendly_func = _composed
    return _tts_friendly_func


def prettify_bot_tts_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return clean
    try:
        spoken = _load_tts_friendly()(clean)
        if re.search(r"less than less than less than P\d+ greater than greater than greater than", spoken):
            return clean
        return spoken
    except Exception:
        return clean


def _model_path_for_speaker(speaker_name: str) -> Path:
    key = (speaker_name or DEFAULT_PIPER_SPEAKER).strip().lower()
    return PIPER_SPEAKER_MODELS.get(key, DEFAULT_PIPER_MODEL)


def _sample_rate_for_model(model_path: Path) -> int:
    key = str(model_path)
    if key in _sample_rate_cache:
        return _sample_rate_cache[key]

    config_path = Path(str(model_path) + ".json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Piper config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sample_rate = int(data["audio"]["sample_rate"])
    _sample_rate_cache[key] = sample_rate
    return sample_rate


def piper_tts_to_wav_bytes(speaker_name: str, text: str) -> bytes:
    model_path = _model_path_for_speaker(speaker_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Piper model not found: {model_path}")

    piper_exe = shutil.which("piper")
    if not piper_exe:
        raise RuntimeError("Could not find 'piper' in PATH.")

    sample_rate = _sample_rate_for_model(model_path)
    spoken = prettify_bot_tts_text(text)
    cmd = [piper_exe, "--model", str(model_path), "--output-raw"]
    proc = subprocess.run(
        cmd,
        input=((spoken or "").strip() + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=PIPER_TIMEOUT_SEC,
        check=False,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or f"Piper exited with code {proc.returncode}.")

    raw_pcm = proc.stdout
    if not raw_pcm:
        raise RuntimeError("Piper returned no audio.")

    samples = np.frombuffer(raw_pcm, dtype=np.int16)
    if samples.size == 0:
        raise RuntimeError("Piper returned empty PCM.")

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
