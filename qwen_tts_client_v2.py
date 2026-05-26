# qwen_tts_client_v2.py
import base64
import os
import socket
import subprocess
import time
from urllib.parse import urlparse

import requests

DEFAULT_TTS_BASE = "http://172.16.13.91:8017"
DEFAULT_MODE = "custom_voice"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"

# ✅ simple tutor instruction (spoken-friendly)
DEFAULT_INSTRUCT = "Friendly tutor voice. Clear, medium pace. Warm tone."

TIMEOUT_SEC = float(os.getenv("QWEN_TTS_TIMEOUT_SEC", "5"))
HEALTH_TIMEOUT_SEC = float(os.getenv("QWEN_TTS_HEALTH_TIMEOUT_SEC", "1.5"))
HEALTH_CACHE_SEC = float(os.getenv("QWEN_TTS_HEALTH_CACHE_SEC", "15"))

_sess = requests.Session()
_health_cache = {"ok": None, "checked_at": 0.0}


def tts_backend_available(
    base: str = DEFAULT_TTS_BASE,
    timeout_sec: float = HEALTH_TIMEOUT_SEC,
    ttl_sec: float = HEALTH_CACHE_SEC,
    force: bool = False,
) -> bool:
    now = time.time()
    cached_ok = _health_cache.get("ok")
    checked_at = float(_health_cache.get("checked_at") or 0.0)
    if not force and cached_ok is not None and (now - checked_at) <= ttl_sec:
        return bool(cached_ok)

    parsed = urlparse(base)
    host = parsed.hostname
    if not host:
        _health_cache.update({"ok": False, "checked_at": now})
        return False

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_sec)):
            ok = True
    except OSError:
        ok = False

    _health_cache.update({"ok": ok, "checked_at": now})
    return ok

def tts_request_bytes(
    text: str,
    base: str = DEFAULT_TTS_BASE,
    mode: str = DEFAULT_MODE,
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
    instruct: str = DEFAULT_INSTRUCT,
    model_size: str = "1.7B",
):
    """
    Returns (wav_bytes, sample_rate).
    Uses /tts endpoint returning audio_b64 WAV.
    """
    url = base.rstrip("/") + "/tts"
    payload = {
        "mode": mode,
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
        "text": text,
        "model_size": model_size,
    }
    r = _sess.post(url, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok", False):
        raise RuntimeError(f"TTS error: {data.get('error')}")
    wav_b64 = data.get("audio_b64", "")
    if not wav_b64:
        raise RuntimeError("TTS ok=true but missing audio_b64")
    wav_bytes = base64.b64decode(wav_b64)
    sr = int(data.get("sample_rate", 24000))
    return wav_bytes, sr


# (optional) keep your old style
def tts_request(text: str, **kwargs):
    wav_bytes, _sr = tts_request_bytes(text, **kwargs)
    # save to temp and play (fallback)
    out = "outputs/auto.wav"
    os.makedirs("outputs", exist_ok=True)
    with open(out, "wb") as f:
        f.write(wav_bytes)
    # Windows quick play
    subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(out)], shell=False)
