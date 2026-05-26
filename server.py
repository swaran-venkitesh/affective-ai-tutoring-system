# =========================
# server.py v3.0 â€” DEEP TUTOR ENGINE
# Teaching: Subtopic planning â†’ Deep coverage â†’ LLM confidence check â†’ Smart Q&A (MCQ + simple)
# Prior knowledge aware. Score-based review. Board+TTS synchronized. Parallel TTS.
# ALL existing infrastructure (voice, FX, board, audio, sockets) UNCHANGED.
# =========================

import atexit
import time
import queue
import threading
import contextvars
import io
import socket
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os
import json
import datetime
import uuid
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO

from stt_model_v2 import STTClient, STTConfig
from LLM_QEWN_v2 import QwenChat, QwenConfig
from qwen_tts_client_v2 import tts_request_bytes, DEFAULT_INSTRUCT, tts_backend_available
from file_processor import process_upload
from camera_monitor import CameraMonitor
from emotion_engine import EmotionEngine, EmotionEngineConfig
from piper_bridge import (
    DEFAULT_PIPER_SPEAKER,
    piper_tts_to_wav_bytes,
    prettify_bot_tts_text,
)
from trusted_web_search import build_trusted_web_context
from session_artifacts import build_material_text, generate_session_artifacts, send_session_email, send_welcome_email
from local_secrets import load_local_env
from student_accounts import (
    clear_last_active_student,
    create_student,
    get_last_active_student,
    get_student,
    list_students,
    save_profile,
    student_paths,
    student_runtime_state_path,
    student_session_dir,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_local_env(Path(__file__).resolve().parent)

# â”€â”€ Empathy-specific TTS tones â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMPATHY_INSTRUCT_LATE    = "Speak with a warm, concerned professor tone. Gentle and caring, slightly quiet, like a mentor who genuinely worries about you staying up too late. Soft and unhurried."
EMPATHY_INSTRUCT_BREAK   = "Speak in a calm, encouraging professor tone. Warm and caring â€” like suggesting a coffee break to a student you care about. Easy-going, not urgent."
EMPATHY_INSTRUCT_WELCOME = "Speak in a warm, welcoming professor tone. Upbeat but gentle â€” delighted the student is back, ready to continue, like greeting them after a short walk."
EMPATHY_INSTRUCT_CAP     = "Speak in a gentle, concerned professor tone. Caring and empathetic â€” like a professor who truly wants the student to rest. Soft, slow, sincere."


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Voice selection globals
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
VOICE_ENGINE  = "humanised"
VOICE_SPEAKER = "Ryan"
_voice_lock   = threading.Lock()

MIC_MUTED = False
SPK_MUTED = False

BOT_SPEAKER_MAP = {
    "david": "david",
    "zira":  "zira",
    "mark":  "mark",
}
VOICE_DEFAULTS = {
    "humanised": "Ryan",
    "bot": "David",
    "piper": DEFAULT_PIPER_SPEAKER,
}


def _voice_ack_payload(engine: str, speaker: str, *, fallback: bool = False, reason: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"engine": engine, "speaker": speaker}
    if fallback:
        payload["fallback"] = True
    if reason:
        payload["reason"] = reason
    return payload


def _auto_fallback_voice(reason: str, *, force_check: bool = False, emit: bool = True) -> Tuple[str, str]:
    global VOICE_ENGINE, VOICE_SPEAKER
    with _voice_lock:
        engine = VOICE_ENGINE
        speaker = VOICE_SPEAKER

    if engine != "humanised":
        return engine, speaker

    try:
        available = tts_backend_available(force=force_check)
    except Exception:
        available = False

    if available:
        return engine, speaker

    switched = False
    with _voice_lock:
        if VOICE_ENGINE != "piper" or VOICE_SPEAKER != DEFAULT_PIPER_SPEAKER:
            VOICE_ENGINE = "piper"
            VOICE_SPEAKER = DEFAULT_PIPER_SPEAKER
            switched = True
        engine = VOICE_ENGINE
        speaker = VOICE_SPEAKER

    if switched:
        log_cli(f"⚠️ Humanised TTS unavailable ({reason}). Switching to local Piper.")
    if emit and (switched or force_check):
        socketio.emit("voice_ack", _voice_ack_payload(engine, speaker, fallback=True, reason=reason))
    return engine, speaker

LLM_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen3.5-9b": {
        "label": "Qwen 3.5 9B",
        "model_id": "qwen3.5-9b",
        "base_url": "http://172.16.13.91:8092/v1",
        "request_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen2.5-32b": {
        "label": "Qwen 2.5 32B",
        "model_id": "qwen2.5-32b",
        "base_url": "http://172.16.13.91:8091/v1",
        "request_kwargs": {},
    },
}
DEFAULT_LLM_MODEL_KEY = os.getenv("TUTOR_LLM_MODEL", "qwen3.5-9b").strip() or "qwen3.5-9b"
current_llm_model_key = DEFAULT_LLM_MODEL_KEY if DEFAULT_LLM_MODEL_KEY in LLM_MODEL_CATALOG else "qwen3.5-9b"


def bot_tts_to_wav_bytes(speaker_name: str, text: str) -> bytes:
    import pyttsx3
    import tempfile
    tmp_path = tempfile.mktemp(suffix=".wav")
    try:
        spoken = prettify_bot_tts_text(text)
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        key = speaker_name.lower().strip()
        match_str = BOT_SPEAKER_MAP.get(key, key)
        voices = engine.getProperty("voices")
        for v in voices:
            if match_str in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.save_to_file(spoken, tmp_path)
        engine.runAndWait()
        engine.stop()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Flask / SocketIO
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-only-local-secret"

# We emit status events from the connect handler, so the namespace must be
# accepted before those emits are sent to the client.
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    always_connect=True,
    ping_interval=25,
    ping_timeout=60,
)

CHUNK_MARK = "<CHUNK>"
text_q = queue.Queue(maxsize=50)

interrupt_event = threading.Event()
tutor_busy = False
_last_interrupt_ts: float = 0.0   # debounce: ignore repeated hand_raise within 3s
cam_monitor: Optional[CameraMonitor] = None
emotion_engine: Optional[EmotionEngine] = None
_stt_ref: Optional[Any] = None          # Fix 2: empathy threads can pause STT
CAMERA_INTERVENTION_COOLDOWN_SEC = 8.0
CAMERA_ACTION_CONTEXT_SEC = 14.0
camera_runtime_lock = threading.Lock()
camera_runtime_state: Dict[str, Any] = {
    "last_attention": "unknown",
    "last_gender": "unknown",
    "last_details": {},
    "last_update_ts": 0.0,
    "last_alert_attention": "",
    "last_alert_ts": 0.0,
    "last_return_ts": 0.0,
    "paused_by_monitor": False,
}
CAMERA_DRINK_RE = re.compile(r"\b(drink|drinking|sip|sipping|water|bottle|glass|cup|mug)\b", re.I)
CAMERA_WAVE_RE = re.compile(r"\b(wave|waving|raised hand|hand wave)\b", re.I)
CAMERA_THUMBS_RE = re.compile(r"\bthumbs up|thumb up\b", re.I)
CAMERA_TWO_RE = re.compile(r"\b(two fingers|2 fingers|peace sign|victory sign)\b", re.I)
CAMERA_THREE_RE = re.compile(r"\b(three fingers|3 fingers)\b", re.I)


LOG_ICON_MAP = {
    "✅": "[OK]",
    "âœ…": "[OK]",
    "🧵": "[Worker]",
    "ðŸ§µ": "[Worker]",
    "🧠": "[Analyze]",
    "ðŸ§ ": "[Analyze]",
    "🗣️": "Tutor:",
    "🗣": "Tutor:",
    "ðŸ—£ï¸": "Tutor:",
    "📋": "BOARD:",
    "ðŸ“‹": "BOARD:",
    "🎤": "[Listening]",
    "ðŸŽ¤": "[Listening]",
    "🔊": "Voice:",
    "ðŸ”Š": "Voice:",
    "🔌": "[Disconnect]",
    "📊": "[Report]",
    "⚡": "[Route]",
    "↩️": "[Resume]",
    "⏸️": "[Pause]",
    "🌙": "[Night]",
    "❓": "[Question]",
    "📌": "[Context]",
    "🐍": "[Python]",
}


def _normalize_log_text(text: str) -> str:
    fixed = repair_mojibake_text(text)
    for src, dst in LOG_ICON_MAP.items():
        fixed = fixed.replace(src, dst)
    fixed = fixed.replace("→", "->").replace("â†’", "->").replace("—", "-")
    return fixed


def log_cli(msg: str):
    msg = _normalize_log_text(msg)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe_msg = str(msg).encode("ascii", "replace").decode("ascii")
        print(safe_msg, flush=True)
    socketio.emit("cli_log", {"msg": msg})
    _file_logger.write(msg)


def enqueue_student_text(text: str, *, source: str = "typed", web_search: bool = False, extra: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "text": str(text or ""),
        "source": str(source or "typed"),
        "web_search": bool(web_search),
    }
    if extra:
        payload.update(dict(extra))
    try:
        text_q.put_nowait(payload)
        return True
    except queue.Full:
        return False


def _camera_details_text(details: Optional[Dict[str, Any]]) -> str:
    info = dict(details or {})
    parts = [
        str(info.get("observed_action") or ""),
        str(info.get("object_label") or ""),
        str(info.get("attention_comment") or ""),
        str(info.get("visible_text") or ""),
    ]
    return " ".join(part.strip() for part in parts if str(part or "").strip()).lower()


def _extract_camera_action_label(details: Optional[Dict[str, Any]]) -> str:
    blob = _camera_details_text(details)
    if not blob:
        return ""
    if CAMERA_WAVE_RE.search(blob):
        return "wave"
    if CAMERA_THUMBS_RE.search(blob):
        return "thumbs_up"
    if CAMERA_THREE_RE.search(blob):
        return "three_fingers"
    if CAMERA_TWO_RE.search(blob):
        return "two_fingers"
    if CAMERA_DRINK_RE.search(blob):
        return "drinking_water"
    return ""


def _remember_camera_runtime(attention: str, gender: str, details: Optional[Dict[str, Any]] = None) -> None:
    info = dict(details or {})
    action = _extract_camera_action_label(info)
    with camera_runtime_lock:
        camera_runtime_state["last_attention"] = str(attention or "unknown")
        camera_runtime_state["last_gender"] = str(gender or "unknown")
        camera_runtime_state["last_details"] = info
        camera_runtime_state["last_update_ts"] = time.time()
        if action:
            camera_runtime_state["last_action"] = action
            camera_runtime_state["last_action_ts"] = time.time()


def _camera_context_block() -> str:
    with camera_runtime_lock:
        last_ts = float(camera_runtime_state.get("last_update_ts") or 0.0)
        details = dict(camera_runtime_state.get("last_details") or {})
        attention = str(camera_runtime_state.get("last_attention") or "unknown")
        action = str(camera_runtime_state.get("last_action") or "")
        action_ts = float(camera_runtime_state.get("last_action_ts") or 0.0)
    now = time.time()
    bits: List[str] = []
    if last_ts and (now - last_ts) <= CAMERA_ACTION_CONTEXT_SEC:
        if attention and attention not in {"unknown", "focused"}:
            bits.append(f"Recent camera state: {attention.replace('_', ' ')}.")
        comment = str(details.get("attention_comment") or "").strip()
        if comment:
            bits.append(f"Camera note: {comment}")
    if action and action_ts and (now - action_ts) <= CAMERA_ACTION_CONTEXT_SEC:
        if action == "wave":
            bits.append("Recent gesture: the student waved. If it fits the moment, greet them naturally once.")
        elif action == "thumbs_up":
            bits.append("Recent gesture: thumbs up. If you just explained something, you may acknowledge understanding briefly.")
        elif action == "two_fingers":
            bits.append("Recent gesture: two fingers. If a question is active, treat this as a possible answer of 2 or option B.")
        elif action == "three_fingers":
            bits.append("Recent gesture: three fingers. If a question is active, treat this as a possible answer of 3 or option C.")
        elif action == "drinking_water":
            bits.append("Recent action: the student appears to be drinking water. Do not treat that as disengagement by itself.")
    return "\n".join(bits).strip()


def _queue_monitor_intervention(event: str, attention: str, message: str, details: Optional[Dict[str, Any]] = None, *, instruct: str = "") -> None:
    now = time.time()
    with camera_runtime_lock:
        if event == "alert":
            if (
                str(camera_runtime_state.get("last_alert_attention") or "") == str(attention or "")
                and (now - float(camera_runtime_state.get("last_alert_ts") or 0.0)) < CAMERA_INTERVENTION_COOLDOWN_SEC
            ):
                return
            camera_runtime_state["last_alert_attention"] = str(attention or "")
            camera_runtime_state["last_alert_ts"] = now
            camera_runtime_state["paused_by_monitor"] = attention in {"phone", "sleepy"}
        elif event == "return":
            if (now - float(camera_runtime_state.get("last_return_ts") or 0.0)) < 4.0:
                return
            camera_runtime_state["last_return_ts"] = now
            camera_runtime_state["paused_by_monitor"] = False
    enqueue_student_text(
        message,
        source="camera_monitor",
        extra={
            "camera_event": event,
            "camera_attention": str(attention or ""),
            "camera_details": dict(details or {}),
            "camera_instruct": str(instruct or ""),
        },
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… FILE LOGGER â€” fresh log.txt each run
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class _FileLogger:
    """
    Writes every log_cli message to log.txt (same folder as server.py).
    File is TRUNCATED fresh on each server start â€” no stale lines.
    Each line: [HH:MM:SS.mmm] <message>
    """
    def __init__(self):
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
        self._lock = threading.Lock()
        # Truncate / create fresh on import (= server start)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(f"# TUTOR SESSION LOG â€” started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# " + "â”€" * 60 + "\n")
        except Exception:
            pass

    def write(self, msg: str):
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            line = f"[{ts}] {msg}\n"
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass


_file_logger = _FileLogger()


def _startup_urls(port: int = 5000) -> List[str]:
    urls = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        host_ip = ""
    if host_ip and not host_ip.startswith("127.") and host_ip not in {"0.0.0.0", "::1"}:
        lan_url = f"http://{host_ip}:{port}"
        if lan_url not in urls:
            urls.append(lan_url)
    return urls


def emit_timing(event_type: str, **kwargs):
    """Emit a structured timing event to the frontend sidebar log."""
    payload = {"type": event_type, "ts": time.time()}
    payload.update(kwargs)
    try:
        socketio.start_background_task(socketio.emit, "timing_event", payload)
    except Exception:
        try:
            socketio.emit("timing_event", payload)
        except Exception:
            pass


def emit_layer_update(layer: str, status: str, input_text: str = "", output_text: str = "", meta: dict = None):
    """
    Emit a real-time pipeline layer update to the frontend architecture flow panel.
    layer:  "emotion_input" | "state_tracker" | "empathy_policy" |
            "pedagogy" | "llm_gen" | "observe"
    status: "idle" | "active" | "done" | "error"
    """
    try:
        payload = {
            "layer": layer,
            "status": status,
            "input": (input_text or "")[:200],
            "output": (output_text or "")[:200],
            "meta": meta or {},
            "ts": time.time(),
        }
        socketio.start_background_task(socketio.emit, "emotion_layer_update", payload)
    except Exception:
        pass


def timed_llm(system_prompt: str, user_prompt: str, label: str = "", **kwargs) -> str:
    """Wrapper around complete_once that emits LLM timing + architecture layer updates."""
    t0 = time.time()
    log_cli(f"[LLM] start: {label or 'LLM call'}")
    emit_timing("llm_start", label=label or "LLM call")
    # â”€â”€ Architecture layer: LLM Gen active â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _is_main_response = label not in ("Router", "Agenda plan", "Subtopic plan",
                                       "Confidence check", "QA eval")
    if _is_main_response:
        emit_layer_update("pedagogy", "done",
            input_text=label or "LLM call",
            output_text="routing to LLM generator")
        emit_layer_update("llm_gen", "active",
            input_text=(user_prompt[:120] if user_prompt else ""),
            meta={"label": label})
    use_emotion_engine = bool(kwargs.pop("use_emotion_engine", False))
    if use_emotion_engine and emotion_engine is not None:
        try:
            if emotion_engine.get_settings().enabled:
                block = emotion_engine.current_prompt_block()
                emit_layer_update("empathy_policy", "done",
                    input_text="emotion state evaluated",
                    output_text=block[:120] if block else "no conditioning")
                user_prompt = block + "\n\n" + user_prompt
        except Exception as exc:
            log_cli(f"Emotion prompt conditioning skipped: {exc}")
    if _is_main_response and emotion_engine is not None:
        try:
            emotion_engine.log_validation_snapshot(
                adaptive_prompt_injected=bool(use_emotion_engine and emotion_engine.get_settings().enabled),
                label=label or "LLM call",
            )
        except Exception:
            pass
    if _is_main_response:
        user_prompt = _append_turn_web_context(user_prompt)
    try:
        result = global_llm.complete_once(system_prompt, user_prompt, **kwargs)
    except Exception as exc:
        log_cli(f"[LLM] error: {label or 'LLM call'} -> {exc}")
        raise
    dur_ms = round((time.time() - t0) * 1000)
    log_cli(f"[LLM] done: {label or 'LLM call'} in {dur_ms}ms")
    emit_timing("llm_done", duration_ms=dur_ms, label=label or "LLM call",
                tokens=len((result or "").split()))
    if _is_main_response:
        emit_layer_update("llm_gen", "done",
            input_text=label or "LLM call",
            output_text=(result or "")[:120],
            meta={"duration_ms": dur_ms, "tokens": len((result or "").split())})
        emit_layer_update("observe", "active",
            input_text="tutor reply generated",
            output_text=f"{dur_ms}ms | {len((result or '').split())} tokens")
    return result


def emotion_complete_once(system_prompt: str, user_prompt: str, **kwargs) -> str:
    conditioned_prompt = user_prompt
    if emotion_engine is not None:
        try:
            if emotion_engine.get_settings().enabled:
                conditioned_prompt = emotion_engine.current_prompt_block() + "\n\n" + user_prompt
        except Exception as exc:
            log_cli(f"Emotion complete_once skipped: {exc}")
    conditioned_prompt = _append_turn_web_context(conditioned_prompt)
    return global_llm.complete_once(system_prompt, conditioned_prompt, **kwargs)


def update_board_sync(
    text: str,
    append: bool = True,
    mode: str = "instant",
    cps: int = 35,
    scroll: bool = True,
    timeout: float = 8.0,
):
    done = threading.Event()

    def _ack(_=None):
        done.set()

    socketio.emit(
        "board_text",
        {"text": text, "append": append, "mode": mode, "cps": cps, "scroll": scroll},
        callback=_ack,
    )
    done.wait(timeout=timeout)
    set_visual_render_text(text, append=append)


def clear_board():
    reset_board_tracker()
    socketio.emit("clear_board")


def show_user_speech(text: str):
    socketio.emit("user_speech", {"text": text})


def set_status(status: str):
    socketio.emit("tutor_status", {"status": status})


def emit_fx(actions: List[dict]):
    if actions:
        socketio.emit("board_fx", {"actions": actions})


def emit_board_highlight(search_text: str):
    """Ask browser to scroll board to a piece of previously written text."""
    socketio.emit("board_highlight", {"text": search_text})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Audio helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def play_wav_bytes_interruptible(
    wav_bytes: bytes,
    stop_event: threading.Event,
    on_audio_start=None,
    volume_gain: float = 1.0,
):
    if SPK_MUTED:
        if callable(on_audio_start):
            try:
                on_audio_start()
            except Exception:
                pass
        return
    try:
        wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        if volume_gain != 1.0:
            wav = np.clip(wav * volume_gain, -1.0, 1.0)
        silence_samples = int(sr * 0.20)
        silence = np.zeros((silence_samples, wav.shape[1]), dtype="float32")
        wav = np.concatenate((wav, silence))
        block_size = 2048
        cur = 0
        started = False
        with sd.OutputStream(samplerate=sr, channels=wav.shape[1], blocksize=block_size) as stream:
            while cur < len(wav):
                if stop_event.is_set():
                    return
                chunk = wav[cur: cur + block_size]
                if not started:
                    started = True
                    if callable(on_audio_start):
                        try:
                            on_audio_start()
                        except Exception:
                            pass
                stream.write(chunk)
                cur += block_size
    except Exception as e:
        log_cli(f"❌ Audio Play Error: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Session logger
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SessionLogger:
    def __init__(self, base_dir: Optional[str] = None):
        self.log_content = []
        self.session_started = datetime.datetime.now()
        safe_date = self.session_started.strftime("%Y-%m-%d_%H-%M-%S")
        default_dir = Path(__file__).resolve().parent / f"Session_{safe_date}"
        self.base_dir = os.path.abspath(base_dir or str(default_dir))

    def log_board(self, text: str):
        if not (text or "").strip():
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_content.append(f"\n### [{ts}] Update\n{text}\n")
        # â”€â”€ Print board text + timestamp to CLI log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        preview = text[:120].replace("\n", " ").strip()
        log_cli(f"ðŸ“‹ [{ts}] BOARD: {preview}")
        self.save()

    def save(self):
        try:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            fp = os.path.join(self.base_dir, "notes.md")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(f"# Session Notes - {self.session_started.strftime('%Y-%m-%d')}\n")
                f.write("".join(self.log_content))
        except Exception as e:
            print(f"Logger Error: {e}", flush=True)


session_logger = SessionLogger()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Session Analytics Tracker
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_analytics_lock = threading.Lock()


def _new_session_id() -> str:
    return "sess_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


current_session_id = _new_session_id()
current_session_started_at = time.time()

active_student_lock = threading.Lock()
ACTIVE_STUDENT = get_last_active_student() or {
    "student_id": "guest_default",
    "name": "Student",
    "age": 20,
    "email": "",
    "folder_name": "guest_default",
}
RUNTIME_STATE_PATH = student_runtime_state_path(ACTIVE_STUDENT)
last_session_artifacts: Dict[str, Any] = {}
last_material_text: str = ""
_last_profile_log_signature: Optional[Tuple[Any, ...]] = None


def get_active_student_profile() -> Dict[str, Any]:
    with active_student_lock:
        return dict(ACTIVE_STUDENT)


def _sync_student_to_empathy(profile: Dict[str, Any]) -> None:
    with emp_lock:
        EMP.student_name = str(profile.get("name") or "Student")
        try:
            EMP.student_age = int(profile.get("age") or 20)
        except Exception:
            EMP.student_age = 20


def _student_session_dir(session_id: str = "") -> Path:
    profile = get_active_student_profile()
    return student_session_dir(profile, session_id or current_session_id)


def _build_student_state_payload(profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    current = dict(profile or get_active_student_profile())
    current_paths = student_paths(current)
    students_payload = []
    for item in list_students():
        item_paths = student_paths(item)
        students_payload.append({
            "student_id": item.get("student_id"),
            "name": item.get("name"),
            "age": item.get("age"),
            "email": item.get("email", ""),
            "folder": str(item_paths["root"]),
            "last_seen_at": item.get("last_seen_at"),
        })
    return {
        "active": {
            "student_id": current.get("student_id"),
            "name": current.get("name"),
            "age": current.get("age"),
            "email": current.get("email", ""),
            "folder": str(current_paths["root"]),
        },
        "students": students_payload,
    }


def emit_student_state(profile: Optional[Dict[str, Any]] = None) -> None:
    socketio.emit("student_state", _build_student_state_payload(profile))


def _set_active_student(profile: Dict[str, Any], restore_runtime: bool = True, fresh_session: bool = False) -> Dict[str, Any]:
    global ACTIVE_STUDENT, RUNTIME_STATE_PATH, session_logger, last_session_artifacts, last_material_text

    try:
        save_runtime_state()
    except Exception:
        pass

    normalized = save_profile(profile, make_last_active=True)
    with active_student_lock:
        ACTIVE_STUDENT = normalized
    RUNTIME_STATE_PATH = student_runtime_state_path(normalized)
    _sync_student_to_empathy(normalized)
    last_session_artifacts = {}
    last_material_text = ""

    if fresh_session:
        reset_session_runtime(get_active_learning_mode())
    elif restore_runtime and RUNTIME_STATE_PATH.exists():
        load_runtime_state()
        restore_visual_board(get_active_learning_mode(), force_clear_if_empty=True)
        emit_learning_mode_state()
        schedule_runtime_state_save()
    else:
        reset_session_runtime(get_active_learning_mode())

    session_logger = SessionLogger(str(_student_session_dir() / "live_notes"))
    emit_student_state(normalized)
    log_cli(f"Active student: {normalized.get('name')} [{normalized.get('student_id')}]")
    return normalized




def _reset_analytics_only() -> None:
    with _analytics_lock:
        _session_analytics["start_time"] = None
        _session_analytics["turns"] = []
        _session_analytics["attention_log"] = []
        _session_analytics["emotion_log"] = []
        _session_analytics["topics"] = []
        _session_analytics["qa_results"] = []
        _session_analytics["tutor_tone_log"] = []
        _session_analytics["empathy_events"] = []


def _reset_visual_state_obj(state) -> None:
    state.live_board_text = ""
    state.current_board_text = ""
    state.shown_board_lines = set()
    state.board_heading_buffer = ""
    state.last_emitted_heading = ""
    state.last_board_hashes.clear()
    state.render_kind = "empty"
    state.render_payload = {}


def _reset_mode_memory_obj(memory) -> None:
    memory.board_history.clear()
    memory.speech_history.clear()
    memory.user_history.clear()


_session_analytics: Dict[str, Any] = {
    "start_time": None,
    "turns":        [],   # [{ts, user_text, intent, llm_ms, topic}]
    "attention_log":[],   # [{ts, state, confidence}]
    "emotion_log":  [],   # [{ts, valence, arousal, dominant, engagement}]
    "topics":       [],
    "qa_results":   [],
    "tutor_tone_log": [],
    "empathy_events": [],
}


def reset_session_runtime(mode: Optional[Any] = None) -> Dict[str, Any]:
    global S, ACTIVE_LEARNING_MODE, session_logger, current_session_id, current_session_started_at
    target_mode = normalize_learning_mode(mode) if mode is not None else get_active_learning_mode()

    with state_lock:
        S = TutorState()

    upload_history.clear()
    upload_documents.clear()
    web_research_history.clear()
    for state in visual_boards.values():
        _reset_visual_state_obj(state)
    for memory in mode_memories.values():
        _reset_mode_memory_obj(memory)

    _reset_analytics_only()

    with learning_mode_lock:
        ACTIVE_LEARNING_MODE = target_mode

    current_session_id = _new_session_id()
    current_session_started_at = time.time()
    session_logger = SessionLogger(str(_student_session_dir(current_session_id) / "live_notes"))

    with emp_lock:
        EMP.session_start_ts = 0.0
        EMP.break_active = False
        EMP.last_break_start = 0.0
        EMP.last_break_secs = 0.0
        EMP.break_pause_topic = ""
        EMP.break_pause_subtopic = ""

    restore_visual_board(target_mode, force_clear_if_empty=True)
    emit_learning_mode_state()
    emit_course_progress()
    schedule_runtime_state_save()

    if emotion_engine is not None:
        try:
            emotion_engine.user_history.clear()
            emotion_engine.emit_status()
            emotion_engine.emit_monitor_update()
        except Exception:
            pass

    payload = {"session_id": current_session_id, "mode": target_mode.value}
    socketio.emit("session_reset", payload)
    log_cli(f"New session ready [{target_mode.value}] id={current_session_id}")
    emit_session_meta(target_mode)
    return payload


def log_analytics_turn(user_text: str, intent: str, llm_ms: float, topic: str):
    with _analytics_lock:
        if _session_analytics["start_time"] is None:
            _session_analytics["start_time"] = time.time()
        _session_analytics["turns"].append({
            "ts": time.time(), "user_text": user_text[:100],
            "intent": intent, "llm_ms": llm_ms, "topic": topic,
        })
        if topic and topic not in _session_analytics["topics"]:
            _session_analytics["topics"].append(topic)


def log_analytics_attention(state: str, confidence: float, details: Optional[Dict[str, Any]] = None):
    info = dict(details or {})
    with _analytics_lock:
        _session_analytics["attention_log"].append({
            "ts": time.time(),
            "state": state,
            "confidence": confidence,
            "source": "camera" if info.get("monitoring_active") else "text",
            "attention_source": str(info.get("attention_source") or ""),
            "blink_count": info.get("blink_count"),
            "blink_rate": info.get("blink_rate"),
            "yawn_count": info.get("yawn_count"),
            "observed_action": str(info.get("observed_action") or ""),
        })


def log_analytics_emotion(valence: float, arousal: float, dominant: str, engagement: str):
    with _analytics_lock:
        _session_analytics["emotion_log"].append({
            "ts": time.time(), "valence": round(float(valence or 0), 3),
            "arousal": round(float(arousal or 0), 3),
            "dominant": dominant, "engagement": engagement,
        })


_TONE_KEYWORDS = [
    ("reassuring", ("no problem", "take your time", "it's okay", "you can do this", "welcome back")),
    ("encouraging", ("great job", "excellent", "well done", "outstanding", "nice work")),
    ("step_by_step", ("step by step", "first", "next", "then", "let's break")),
    ("clarifying", ("in simple terms", "that means", "notice", "the key idea", "for example")),
]


def log_analytics_tutor_speech(text: str, instruct: Optional[str] = None) -> None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return
    tone = "supportive"
    cue = "clear guidance"
    for label, cues in _TONE_KEYWORDS:
        match = next((item for item in cues if item in lowered), None)
        if match:
            tone = label
            cue = match
            break
    if instruct:
        ilow = str(instruct).lower()
        if "warm" in ilow or "caring" in ilow:
            tone = "empathetic"
            cue = cue if cue != "clear guidance" else "warm tone"
    with _analytics_lock:
        _session_analytics["tutor_tone_log"].append({
            "ts": time.time(),
            "tone": tone,
            "cue": cue,
            "preview": str(text or "")[:160],
        })


def log_analytics_empathy_event(kind: str, detail: str = "") -> None:
    with _analytics_lock:
        _session_analytics["empathy_events"].append({
            "ts": time.time(),
            "kind": str(kind or "event"),
            "detail": str(detail or "")[:160],
        })


def _session_manifest_path(session_id: str, profile: Optional[Dict[str, Any]] = None) -> Path:
    owner = profile or get_active_student_profile()
    return student_session_dir(owner, session_id) / "manifest.json"


def _save_session_manifest(session_id: str, manifest: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> None:
    path = _session_manifest_path(session_id, profile)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_session_manifest(session_id: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any] | None:
    path = _session_manifest_path(session_id, profile)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _runtime_snapshot_path(session_id: str = "", profile: Optional[Dict[str, Any]] = None) -> Path:
    owner = profile or get_active_student_profile()
    target_session_id = str(session_id or current_session_id or "session_unknown").strip() or "session_unknown"
    return student_session_dir(owner, target_session_id) / "runtime_state.json"


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _load_session_report_payload(session_id: str, profile: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    manifest = None
    if last_session_artifacts and last_session_artifacts.get("session_id") == session_id:
        manifest = dict(last_session_artifacts)
    if manifest is None:
        manifest = _load_session_manifest(session_id, profile)
    if not manifest:
        return None, None
    file_map = dict(manifest.get("files") or {})
    target = Path(str(file_map.get("json") or ""))
    if not target.exists():
        return manifest, None
    try:
        report = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return manifest, None
    report["downloads"] = _report_download_urls(session_id)
    return manifest, report


_TRIVIAL_SESSION_TEXTS = {
    "hi", "hello", "hey", "ok", "okay", "yes", "no", "start", "continue",
    "teach me", "help me", "next", "go on", "ready",
}


def _normalize_session_title(text: str, fallback: str = "Session") -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-:|")
    cleaned = re.sub(r"^[\"'`]+|[\"'`]+$", "", cleaned)
    cleaned = re.sub(r"^(teach me|can you teach me|help me with|i want to learn|please teach me)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" \t\r\n-:|")
    if not cleaned:
        return fallback
    return cleaned[:80]


def _first_meaningful_turn_text(turns: List[Dict[str, Any]]) -> str:
    for turn in turns:
        text = _normalize_session_title(str(turn.get("user_text") or ""), "")
        if not text:
            continue
        if text.lower() in _TRIVIAL_SESSION_TEXTS:
            continue
        return text
    return ""


def _latest_upload_title() -> str:
    docs = list(upload_documents)
    for doc in reversed(docs):
        filename = str(doc.get("filename") or "").strip()
        if filename:
            return _normalize_session_title(Path(filename).stem.replace("_", " "), "")
    return ""


def _infer_report_title(
    report_mode: str,
    state_title: str,
    last_topic: str,
    topics: List[str],
    turns: List[Dict[str, Any]],
) -> str:
    preferred = _normalize_session_title(state_title, "")
    if preferred:
        return preferred
    if str(report_mode or "shallow") == LearningMode.COURSE.value:
        course_choice = _normalize_session_title(last_topic or (topics[0] if topics else ""), "")
        return course_choice or "Course Session"
    shallow_choice = _first_meaningful_turn_text(turns) or _latest_upload_title()
    return _normalize_session_title(shallow_choice, "Session")


def _infer_live_session_title(mode: Optional[Any] = None, user_text: str = "") -> str:
    target_mode = normalize_learning_mode(mode) if mode is not None else get_active_learning_mode()
    with state_lock:
        state_title = _normalize_session_title(S.title, "")
        last_topic = str(S.last_topic or "")
    if target_mode == LearningMode.COURSE:
        return state_title or _normalize_session_title(last_topic, "") or "Course Session"
    focus = _infer_shallow_focus(user_text)
    if focus:
        return _normalize_session_title(focus, "Session")
    turns = []
    with _analytics_lock:
        turns = list(_session_analytics["turns"])
    candidate = _first_meaningful_turn_text(turns) or _normalize_session_title(user_text, "") or _latest_upload_title()
    return _normalize_session_title(candidate, "Session")


def emit_session_meta(mode: Optional[Any] = None, user_text: str = "", title_override: str = "") -> Dict[str, Any]:
    target_mode = normalize_learning_mode(mode) if mode is not None else get_active_learning_mode()
    title = _normalize_session_title(title_override, "") or _infer_live_session_title(target_mode, user_text)
    active_student = get_active_student_profile()
    payload = {
        "session_id": current_session_id,
        "mode": target_mode.value,
        "title": title,
        "student_name": str(active_student.get("name") or "Student"),
        "student_id": str(active_student.get("student_id") or ""),
        "updated_at": int(time.time() * 1000),
    }
    socketio.emit("session_meta", payload)
    return payload


def _session_has_meaningful_activity(
    turns: List[Dict[str, Any]],
    qa_res: List[Dict[str, Any]],
    topics: List[str],
    board_material: str,
    uploaded_materials: List[Dict[str, str]],
    duration_secs: float,
) -> bool:
    meaningful_turns = [
        text for text in (_normalize_session_title(str(turn.get("user_text") or ""), "") for turn in turns)
        if text and text.lower() not in _TRIVIAL_SESSION_TEXTS and len(text) >= 8
    ]
    non_generic_topics = [topic for topic in topics if str(topic or "").strip() and str(topic).strip().lower() != "general"]
    board_text = str(board_material or "").strip()
    useful_uploads = [
        item for item in (uploaded_materials or [])
        if str(item.get("content") or "").strip() and not extract_upload_issue(str(item.get("content") or ""))
    ]
    activity_score = 0
    if meaningful_turns:
        activity_score += 1
    if len(meaningful_turns) >= 2:
        activity_score += 1
    if qa_res:
        activity_score += 1
    if len(board_text) >= 120:
        activity_score += 1
    if useful_uploads:
        activity_score += 1
    if non_generic_topics:
        activity_score += 1
    if duration_secs >= 180:
        activity_score += 1
    return activity_score >= 2 or bool(qa_res) or len(meaningful_turns) >= 2 or (len(board_text) >= 120 and bool(non_generic_topics))


def _has_break_worthy_study_activity() -> bool:
    with _analytics_lock:
        turns = list(_session_analytics["turns"])
    meaningful_turns = [
        text for text in (_normalize_session_title(str(turn.get("user_text") or ""), "") for turn in turns)
        if text and text.lower() not in _TRIVIAL_SESSION_TEXTS and len(text) >= 8
    ]
    if meaningful_turns:
        return True
    with state_lock:
        phase = S.phase
        has_course_context = bool(S.title or S.last_topic or S.taught_points or S.subtopics_done)
    board_text = _current_board_material().strip()
    if phase not in (Phase.IDLE, Phase.AGENDA, Phase.WAIT_START) and (has_course_context or len(board_text) >= 120):
        return True
    return False


def _build_tone_timeline(start_ts: float, tone_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline = []
    for entry in tone_log[:18]:
        minute = round(max(0.0, float(entry.get("ts") or start_ts) - start_ts) / 60.0, 1)
        timeline.append({
            "t": minute,
            "tone": str(entry.get("tone") or "supportive"),
            "cue": str(entry.get("cue") or ""),
            "preview": str(entry.get("preview") or "")[:120],
        })
    return timeline


_TONE_GRAPH_SCORE = {
    "empathetic": 88,
    "reassuring": 84,
    "encouraging": 82,
    "supportive": 78,
    "clarifying": 74,
    "step_by_step": 72,
}
_CAMERA_GRAPH_SCORE = {
    "focused": 90,
    "looking_down": 74,
    "text_active": 72,
    "unknown": 55,
    "distracted_side": 34,
    "sleepy": 18,
    "away": 12,
    "phone": 8,
}


def _build_tone_graph_timeline(start_ts: float, tone_log: List[Dict[str, Any]], attn_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in tone_log[:24]:
        minute = round(max(0.0, float(entry.get("ts") or start_ts) - start_ts) / 60.0, 1)
        tone = str(entry.get("tone") or "supportive")
        rows.append({
            "t": minute,
            "text_score": _TONE_GRAPH_SCORE.get(tone, 76),
            "text_label": tone,
            "camera_score": None,
            "camera_label": "",
        })
    for entry in attn_log[:48]:
        if str(entry.get("source") or "camera") != "camera":
            continue
        minute = round(max(0.0, float(entry.get("ts") or start_ts) - start_ts) / 60.0, 1)
        state = str(entry.get("state") or "unknown")
        rows.append({
            "t": minute,
            "text_score": None,
            "text_label": "",
            "camera_score": _CAMERA_GRAPH_SCORE.get(state, 55),
            "camera_label": state,
        })
    rows.sort(key=lambda item: float(item.get("t") or 0.0))
    if not rows:
        return []
    merged: List[Dict[str, Any]] = []
    for row in rows:
        minute = float(row.get("t") or 0.0)
        if merged and abs(float(merged[-1].get("t") or 0.0) - minute) < 0.05:
            if row.get("text_score") is not None:
                merged[-1]["text_score"] = row.get("text_score")
                merged[-1]["text_label"] = row.get("text_label") or merged[-1].get("text_label")
            if row.get("camera_score") is not None:
                merged[-1]["camera_score"] = row.get("camera_score")
                merged[-1]["camera_label"] = row.get("camera_label") or merged[-1].get("camera_label")
            continue
        merged.append(dict(row))
    return merged[:28]


def _build_empathy_summary(engagement_timeline: List[Dict[str, Any]], tone_timeline: List[Dict[str, Any]], empathy_enabled: bool, attention_score: int) -> List[str]:
    usable = [point for point in engagement_timeline if point.get("score") is not None]
    if usable:
        first_score = round((usable[0].get("score") or 0) * 100)
        last_score = round((usable[-1].get("score") or 0) * 100)
        delta = last_score - first_score
    else:
        first_score = last_score = delta = 0
    confusion_now = 0
    if usable:
        confusion_now = round((usable[-1].get("confusion") or 0) * 100)
    tone_bits = [f"{item.get('tone')} ({item.get('cue')})".strip() for item in tone_timeline[:4] if item.get("tone")]
    summary = []
    if empathy_enabled:
        summary.append(f"Empathy support stayed active through the session, with engagement moving from {first_score}% to {last_score}% ({delta:+d} points).")
        summary.append(f"Attention stayed around {attention_score}%, and the latest confusion/stress estimate settled near {confusion_now}%.")
    else:
        summary.append(f"The emotion engine was off, but monitoring still tracked engagement and attention signals. Engagement moved from {first_score}% to {last_score}% ({delta:+d} points).")
        summary.append(f"Attention stayed around {attention_score}%, so the report still reflects monitoring-state changes even without empathy interventions enabled.")
    if tone_bits:
        summary.append("Tutor tone shifted across the session using cues like " + ", ".join(tone_bits) + ".")
    return summary


def _current_board_material() -> str:
    combined = []
    for mode in (LearningMode.COURSE, LearningMode.SHALLOW):
        state = get_visual_state(mode)
        if state.live_board_text.strip():
            combined.append(state.live_board_text.strip())
    return _trim_context_block("\n\n---\n\n".join(combined), 5000)


def _report_download_urls(session_id: str) -> Dict[str, str]:
    return {
        "json": f"/session-report/download?session_id={session_id}&format=json",
        "pdf": f"/session-report/download?session_id={session_id}&format=pdf",
        "docx": f"/session-report/download?session_id={session_id}&format=docx",
        "png": f"/session-report/download?session_id={session_id}&format=png",
        "material_pdf": f"/session-material/download?session_id={session_id}&format=pdf",
    }


def _course_progress_payload(completed_topic: str = "", mode: Optional[Any] = None) -> Dict[str, Any]:
    active_mode = normalize_learning_mode(mode) if mode is not None else get_active_learning_mode()
    if active_mode != LearningMode.COURSE:
        return {
            "mode": active_mode.value,
            "topics": [],
            "completed_topics": [],
            "current_topic": "",
            "completed_topic": "",
            "percent": 0,
        }
    with state_lock:
        topics = list(S.topics or [])
        completed_topics = [topic for topic in topics if topic in S.all_taught]
        current_topic = S.last_topic or (topics[S.topic_idx] if topics and S.topic_idx < len(topics) else "")
    total = len(topics)
    percent = round((len(completed_topics) / total) * 100) if total else 0
    return {
        "mode": active_mode.value,
        "topics": topics,
        "completed_topics": completed_topics,
        "current_topic": current_topic,
        "completed_topic": completed_topic,
        "percent": percent,
    }


def emit_course_progress(completed_topic: str = "", mode: Optional[Any] = None) -> None:
    payload = _course_progress_payload(completed_topic, mode=mode)
    socketio.emit("course_progress", payload)
    if completed_topic:
        socketio.emit("course_topic_completed", payload)



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Parsing helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def split_ready_chunks(buf: str):
    parts = buf.split(CHUNK_MARK)
    if len(parts) == 1:
        return [], buf
    chunks = [p.strip() for p in parts[:-1] if p.strip()]
    rem = parts[-1]
    return chunks, rem


def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}\b[^>]*>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    start = re.search(rf"<{tag}\b[^>]*>", text, re.IGNORECASE)
    if not start:
        return ""
    tail = text[start.end():]
    next_tag = re.search(rf"(?=\s*<(?:speech|board|fx|meta)\b)|(?={re.escape(CHUNK_MARK)})", tail, re.IGNORECASE)
    if next_tag:
        return tail[:next_tag.start()].strip()
    return tail.strip()


def _strip_structural_tags(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace(CHUNK_MARK, " ")
    cleaned = re.sub(r"</?(speech|board|fx|meta)\b[^>]*>", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_fx_block(fx_text: str):
    actions = []
    if not fx_text:
        return actions
    lines = [ln.strip() for ln in fx_text.splitlines() if ln.strip()]
    for ln in lines:
        if ":" not in ln:
            continue
        typ, tgt = ln.split(":", 1)
        typ = typ.strip().lower()
        tgt = tgt.strip()
        delay_ms = 0
        if "@" in tgt:
            tgt, d = tgt.rsplit("@", 1)
            tgt = tgt.strip()
            try:
                delay_ms = int(d.strip())
            except Exception:
                delay_ms = 0
        if typ == "popseq":
            items = [x.strip() for x in tgt.split("|") if x.strip()]
            for i, it in enumerate(items):
                actions.append({"type": "pop", "target": it, "duration_ms": 650, "delay_ms": delay_ms + i * 650})
            continue
        if typ not in ("glow", "pop"):
            continue
        targets = [x.strip() for x in tgt.split("|") if x.strip()]
        for t in targets:
            actions.append({"type": typ, "target": t, "duration_ms": 900, "delay_ms": delay_ms})
    return actions


def safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s2 = s.strip()
    s2 = re.sub(r"^```(?:json)?", "", s2, flags=re.IGNORECASE).strip()
    s2 = re.sub(r"```$", "", s2).strip()
    m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
    if m:
        s2 = m.group(0)
    try:
        return json.loads(s2)
    except Exception:
        return {}


def postprocess_board_text(board: str) -> str:
    board = repair_mojibake_text(board or "")
    if not board:
        return ""

    def repl(m):
        inner = m.group(1)
        out_lines = []
        for ln in inner.splitlines():
            ln = ln.rstrip()
            if ln.strip():
                out_lines.append(ln)
        return "\n".join(out_lines)

    b = re.sub(r"```(?:python)?\s*(.*?)```", repl, board, flags=re.DOTALL | re.IGNORECASE)
    b = re.sub(r"\\([+\-*/=])", r"\1", b)
    b = b.replace('\\"', '"').replace("\\'", "'")

    # Strip markdown horizontal rules (--- / === / ***) that LLM sometimes adds
    b = re.sub(r"^\s*[-=*]{2,}\s*$", "", b, flags=re.MULTILINE)

    # Fix 4: Remove intra-chunk heading duplication
    # Pattern: "**Title**\nTitle" or "Title\n**Title**" â†’ keep only the bold version
    b = re.sub(
        r"(\*\*([^*\n]+)\*\*)\n+\2\b",
        r"\1",
        b,
        flags=re.IGNORECASE
    )
    b = re.sub(
        r"^([^\n*]{3,60})\n+\*\*\1\*\*",
        r"**\1**",
        b,
        flags=re.MULTILINE | re.IGNORECASE
    )

    # Collapse 3+ consecutive blank lines â†’ max 1 blank line
    b = re.sub(r"\n{3,}", "\n\n", b)

    return b.strip()


def normalize_board_text(board: str) -> str:
    b = (board or "").replace("\r\n", "\n").replace("\r", "\n")
    if not b.strip():
        return ""
    lines = [ln.rstrip() for ln in b.splitlines()]
    cleaned = []
    empty_run = 0
    for ln in lines:
        if not ln.strip():
            empty_run += 1
            if empty_run <= 1:
                cleaned.append("")
            continue
        empty_run = 0
        cleaned.append(ln)
    dedup = []
    prev = None
    for ln in cleaned:
        if prev is not None and ln.strip() and ln.strip() == prev.strip():
            continue
        dedup.append(ln)
        prev = ln
    out = []
    for ln in dedup:
        s = ln.strip()
        is_headingish = bool(re.match(r"^\*\*.+\*\*$", s)) or (s.endswith(":") and len(s) <= 40)
        if is_headingish and out and out[-1].strip():
            out.append("")
        out.append(ln)
    b2 = "\n".join(out).strip()
    if len(b2) > 2500:
        b2 = b2[:2500].rstrip() + "\n..."
    return b2


def repair_mojibake_text(text: str) -> str:
    if not text:
        return ""
    fixed = str(text)
    if any(marker in fixed for marker in ("â", "ð", "Â")):
        try:
            fixed = fixed.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass
    return fixed


def _normalize_llm_model_key(raw: Any) -> str:
    candidate = str(raw or "").strip()
    if candidate in LLM_MODEL_CATALOG:
        return candidate
    return "qwen3.5-9b"


def _current_llm_model_payload() -> Dict[str, Any]:
    selected = _normalize_llm_model_key(current_llm_model_key)
    config = dict(LLM_MODEL_CATALOG[selected])
    return {
        "selected": selected,
        "label": str(config.get("label") or selected),
        "model_id": str(config.get("model_id") or selected),
        "base_url": str(config.get("base_url") or ""),
        "options": [
            {
                "key": key,
                "label": str(item.get("label") or key),
                "model_id": str(item.get("model_id") or key),
            }
            for key, item in LLM_MODEL_CATALOG.items()
        ],
    }


def _build_llm_config(model_key: str) -> QwenConfig:
    selected = _normalize_llm_model_key(model_key)
    config = dict(LLM_MODEL_CATALOG[selected])
    return QwenConfig(
        model_id=str(config.get("model_id") or selected),
        base_url=str(config.get("base_url") or ""),
        default_request_kwargs=dict(config.get("request_kwargs") or {}),
    )


def apply_llm_model(model_key: str, *, emit: bool = True, log_change: bool = True) -> Dict[str, Any]:
    global global_llm, current_llm_model_key
    selected = _normalize_llm_model_key(model_key)
    current_llm_model_key = selected
    global_llm = QwenChat(_build_llm_config(selected))
    payload = _current_llm_model_payload()
    if emit:
        socketio.emit("llm_model_status", payload)
    if log_change:
        log_cli(f"LLM model: {payload['label']} [{payload['model_id']}]")
    return payload
    fixed = fixed.replace("\ufeff", "").replace("\ufffd", "")
    fixed = fixed.encode("utf-8", "ignore").decode("utf-8", "ignore")
    fixed = re.sub(r"[\ud800-\udfff]", "", fixed)
    return fixed


def sanitize_for_speech(text: str) -> str:
    """Convert code/math symbols to spoken words so TTS sounds natural."""
    text = repair_mojibake_text(text or "")
    if not text:
        return ""

    # Strip chunk markers and HTML tags
    text = text.replace(CHUNK_MARK, " ")
    text = re.sub(r"<[^>]+>", " ", text)

    # Backtick code spans â†’ unwrap the content (just say the word)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Markdown bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)

    # Math / comparison operators â†’ spoken words (order matters!)
    text = text.replace("**", " to the power of ")
    text = text.replace("//", " floor divided by ")
    text = text.replace("!=", " not equal to ")
    text = text.replace(">=", " greater than or equal to ")
    text = text.replace("<=", " less than or equal to ")
    text = text.replace("==", " equals ")
    text = text.replace("->", " gives ")
    text = text.replace("=>", " results in ")
    text = text.replace(">", " greater than ")
    text = text.replace("<", " less than ")
    text = text.replace("%", " modulo ")
    text = text.replace("//", " floor division ")  # in case any remain

    # Arithmetic
    # Keep + - * / as-is in most contexts (TTS handles them ok)

    # Python-specific tokens that TTS mispronounces
    text = text.replace("print(", "print, ")
    text = text.replace("def ", "define function ")
    text = text.replace("True", "True")   # TTS usually fine
    text = text.replace("False", "False")

    # Remove brackets from math expressions: (5 > 3) â†’ 5 greater than 3
    text = re.sub(r"\(([^()]+)\)", r"\1", text)

    # Collapse whitespace and newlines
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = prettify_bot_tts_text(text)
    return text


def parse_chunk(chunk: str):
    speech = repair_mojibake_text(_extract_tag(chunk, "speech"))
    board = _extract_tag(chunk, "board")
    fx_raw = _extract_tag(chunk, "fx")
    meta_raw = _extract_tag(chunk, "meta")
    meta = safe_json_load(meta_raw)
    if speech and "<board" in speech.lower():
        speech = repair_mojibake_text(_strip_structural_tags(speech))
    if not speech:
        speech = repair_mojibake_text(_strip_structural_tags(re.sub(r"<board\b.*", "", chunk, flags=re.I | re.S)))
    if board and "<" in board:
        board = _strip_structural_tags(board)
    fx_actions = parse_fx_block(fx_raw)
    board = postprocess_board_text(board)
    board = normalize_board_text(board)
    return speech, board, fx_actions, meta


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… TUTOR STATE (v3 â€” deep teaching)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Mode(str, Enum):
    QUICK = "quick"
    TUTOR = "tutor"


class Phase(str, Enum):
    IDLE             = "IDLE"
    AGENDA           = "AGENDA"
    WAIT_START       = "WAIT_START"
    SUBTOPIC_PLAN    = "SUBTOPIC_PLAN"     # plan subtopics for current main topic
    TEACH_SUBTOPIC   = "TEACH_SUBTOPIC"    # deeply teach current subtopic
    SUBTOPIC_WRAP    = "SUBTOPIC_WRAP"     # between subtopics â€” any questions?
    CONFIDENCE_CHECK = "CONFIDENCE_CHECK"  # LLM decides: ready for Q&A?
    QA               = "QA"               # Q&A session
    QA_REVIEW        = "QA_REVIEW"        # re-teach wrong concepts + retry
    NEXT             = "NEXT"             # move to next main topic


@dataclass
class TutorState:
    mode:  Mode  = Mode.QUICK
    phase: Phase = Phase.IDLE

    # Course structure
    title:  str        = ""
    topics: List[str]  = field(default_factory=list)
    topic_idx: int     = 0
    prior_known_indices: List[int] = field(default_factory=list)  # topics student already knows

    # Subtopic tracking (within current main topic)
    subtopics:     List[str] = field(default_factory=list)
    subtopic_idx:  int       = 0
    subtopics_done: List[str] = field(default_factory=list)

    # Teaching memory
    taught_points: List[str] = field(default_factory=list)
    # Cross-topic memory: topic_name â†’ taught_points (for referencing prior topics)
    all_taught: Dict[str, List[str]] = field(default_factory=dict)

    # Q&A
    qa_questions:     List[Dict] = field(default_factory=list)  # full batch, each: {type,question,options,answer,concept}
    qa_idx:           int        = 0
    qa_correct:       int        = 0
    qa_wrong_items:   List[Dict] = field(default_factory=list)  # questions student got wrong
    qa_review_idx:    int        = 0
    qa_retry_questions: List[Dict] = field(default_factory=list)  # similar questions for review
    qa_retry_idx:     int        = 0
    qa_interrupted_idx: int      = -1   # question idx when hand-raise interrupted QA
    qa_review_phase:    str      = "RETEACH"  # "RETEACH" or "RETRY"
    qa_result_log:  List[Dict] = field(default_factory=list)  # [{q, correct, answer, student}]

    # Legacy / convenience
    last_question:         str = ""
    last_question_answer:  str = ""
    last_question_concept: str = ""
    last_question_type:    str = ""

    last_user_goal: str = ""
    last_topic:     str = ""
    topic_completed: bool = False
    interrupt_just_answered: bool = False  # True right after answering a hand-raise question
    subtopic_interrupted:   bool = False   # True when speak_chunks was cut by hand-raise mid-subtopic
    needs_example:   bool = False
    suggest_qa:      str  = "none"
    correct_streak:  int  = 0
    qa_round:        int  = 0
    qa_round_max:    int  = 3
    student_context: str  = ""  # Free-form notes from student (version, setup, background)
    prior_confirm_mode:    bool      = False   # Waiting for student to confirm/modify skip plan
    prior_confirm_indices: List[int] = field(default_factory=list)  # Topics detected as known
    prior_confirm_topic:   str       = ""      # Next topic we plan to jump to
    switch_confirm_mode:   bool      = False   # Waiting for user to confirm mid-session topic switch
    switch_confirm_goal:   str       = ""      # The new topic they asked to switch to


S = TutorState()
state_lock = threading.Lock()

global_llm: Optional[QwenChat] = None
shared_student_context: str = ""


class LearningMode(str, Enum):
    SHALLOW = "shallow"
    COURSE = "course"


@dataclass
class VisualBoardState:
    live_board_text: str = ""
    current_board_text: str = ""
    shown_board_lines: set = field(default_factory=set)
    board_heading_buffer: str = ""
    last_emitted_heading: str = ""
    last_board_hashes: deque = field(default_factory=lambda: deque(maxlen=25))
    render_kind: str = "empty"
    render_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeMemoryState:
    board_history: deque = field(default_factory=lambda: deque(maxlen=12))
    speech_history: deque = field(default_factory=lambda: deque(maxlen=10))
    user_history: deque = field(default_factory=lambda: deque(maxlen=10))


ACTIVE_LEARNING_MODE = LearningMode.SHALLOW
learning_mode_lock = threading.Lock()
TURN_LEARNING_MODE: contextvars.ContextVar[Optional[LearningMode]] = contextvars.ContextVar(
    "turn_learning_mode",
    default=None,
)
TURN_WEB_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "turn_web_context",
    default=None,
)
visual_boards = {
    LearningMode.COURSE: VisualBoardState(),
    LearningMode.SHALLOW: VisualBoardState(),
}
mode_memories = {
    LearningMode.COURSE: ModeMemoryState(),
    LearningMode.SHALLOW: ModeMemoryState(),
}
RUNTIME_STATE_PATH = student_runtime_state_path(ACTIVE_STUDENT)
runtime_state_lock = threading.Lock()
_runtime_state_timer: Optional[threading.Timer] = None
_runtime_restore_note = ""


def normalize_learning_mode(raw: Any) -> LearningMode:
    if isinstance(raw, LearningMode):
        return raw
    value = str(getattr(raw, "value", raw) or "").strip().lower()
    if value == LearningMode.SHALLOW.value:
        return LearningMode.SHALLOW
    return LearningMode.COURSE


def get_active_learning_mode() -> LearningMode:
    with learning_mode_lock:
        return ACTIVE_LEARNING_MODE


def get_effective_learning_mode(mode: Optional[Any] = None) -> LearningMode:
    if mode is not None:
        return normalize_learning_mode(mode)
    bound_mode = TURN_LEARNING_MODE.get()
    if bound_mode is not None:
        return normalize_learning_mode(bound_mode)
    return get_active_learning_mode()


def get_visual_state(mode: Optional[Any] = None) -> VisualBoardState:
    target = get_effective_learning_mode(mode)
    return visual_boards[target]


def get_mode_memory(mode: Optional[Any] = None) -> ModeMemoryState:
    target = get_effective_learning_mode(mode)
    return mode_memories[target]


def learning_mode_label(mode: Optional[Any] = None) -> str:
    target = get_effective_learning_mode(mode)
    return "Course Mode" if target == LearningMode.COURSE else "Shallow Mode"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… EMPATHY + SESSION TRACKING STATE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class EmpathyState:
    # Student profile (set from frontend)
    student_name:   str   = "Student"
    student_age:    int   = 20

    # Real GPS location (always from browser geolocation, NOT user-selected)
    real_lat:       float = 10.7905   # Tiruchirappalli default
    real_lon:       float = 78.7047
    real_city:      str   = "Tiruchirappalli"
    real_tz:        str   = "Asia/Kolkata"
    weather_temp:   float = 30.0
    weather_label:  str   = "Clear"

    # Session timing
    session_start_ts:   float = 0.0   # unix ts when tutor session started
    total_study_secs:   float = 0.0   # accumulated study time (excl. breaks)
    last_break_start:   float = 0.0   # unix ts when current/last break started
    last_break_secs:    float = 0.0   # duration of last completed break
    breaks_today:       int   = 0

    # Break state
    break_active:       bool  = False
    break_offered_at:   float = 0.0   # last time we offered a break (avoid spam)
    mins_since_break:   float = 0.0   # updated periodically

    # Where we paused for break (so we can resume)
    break_pause_topic:    str = ""
    break_pause_subtopic: str = ""

    # 50-min daily soft cap tracking
    session_warned_50:  bool  = False

EMP = EmpathyState()
emp_lock = threading.Lock()
_sync_student_to_empathy(ACTIVE_STUDENT)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Memory buffers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
upload_history = deque(maxlen=8)
upload_documents = deque(maxlen=4)
web_research_history = deque(maxlen=4)


# â”€â”€ Fix 2: line-level dedup â€” track every significant line already shown â”€â”€

# â”€â”€ Fix 3: lone-heading buffer â€” hold a heading until it has body content â”€â”€


def _board_hash(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return str(hash(s))


def _norm_line(ln: str) -> str:
    """Normalise a line for dedup comparison (collapse whitespace, lower)."""
    return re.sub(r"\s+", " ", (ln or "").strip()).lower()


def _serialize_visual_state(state: VisualBoardState) -> dict[str, Any]:
    return {
        "live_board_text": state.live_board_text,
        "current_board_text": state.current_board_text,
        "shown_board_lines": sorted(state.shown_board_lines),
        "board_heading_buffer": state.board_heading_buffer,
        "last_emitted_heading": state.last_emitted_heading,
        "last_board_hashes": list(state.last_board_hashes),
        "render_kind": state.render_kind,
        "render_payload": dict(state.render_payload or {}),
    }


def _restore_visual_state(state: VisualBoardState, data: dict[str, Any]) -> None:
    live_board_text = sanitize_board_output(str(data.get("live_board_text", "") or ""))
    current_board_text = sanitize_board_output(str(data.get("current_board_text", "") or ""))
    shown_lines = set(data.get("shown_board_lines") or [])
    shown_lines = {
        str(line).strip().lower()
        for line in shown_lines
        if str(line).strip() and not BOARD_STATUS_LINE_RE.match(str(line).strip())
    }
    state.live_board_text = live_board_text
    state.current_board_text = current_board_text
    state.shown_board_lines = shown_lines
    state.board_heading_buffer = str(data.get("board_heading_buffer", "") or "")
    state.last_emitted_heading = str(data.get("last_emitted_heading", "") or "")
    state.last_board_hashes = deque(list(data.get("last_board_hashes") or []), maxlen=25)
    state.render_kind = str(data.get("render_kind", "empty") or "empty")
    payload = dict(data.get("render_payload") or {})
    if "text" in payload:
        payload["text"] = sanitize_board_output(str(payload.get("text") or ""))
    state.render_payload = payload if payload.get("text") or payload else {}


def _serialize_mode_memory(memory: ModeMemoryState) -> dict[str, list[str]]:
    return {
        "board_history": list(memory.board_history),
        "speech_history": list(memory.speech_history),
        "user_history": list(memory.user_history),
    }


def _restore_mode_memory(memory: ModeMemoryState, data: dict[str, Any]) -> None:
    memory.board_history = deque(list(data.get("board_history") or []), maxlen=12)
    memory.speech_history = deque(list(data.get("speech_history") or []), maxlen=10)
    memory.user_history = deque(list(data.get("user_history") or []), maxlen=10)


def _serialize_tutor_state(state: TutorState) -> dict[str, Any]:
    payload = asdict(state)
    payload["mode"] = state.mode.value
    payload["phase"] = state.phase.value
    return payload


def _restore_tutor_state(data: dict[str, Any]) -> TutorState:
    restored = TutorState()
    for field_name in TutorState.__dataclass_fields__.keys():
        if field_name in {"mode", "phase"}:
            continue
        if field_name in data:
            setattr(restored, field_name, data[field_name])
    try:
        restored.mode = Mode(str(data.get("mode", Mode.QUICK.value)))
    except Exception:
        restored.mode = Mode.QUICK
    try:
        restored.phase = Phase(str(data.get("phase", Phase.IDLE.value)))
    except Exception:
        restored.phase = Phase.IDLE
    return restored


def _build_runtime_state_payload() -> Dict[str, Any]:
    with state_lock:
        tutor_state = _serialize_tutor_state(S)
        shared_context = shared_student_context
    with learning_mode_lock:
        active_mode = ACTIVE_LEARNING_MODE.value
    return {
        "saved_at": time.time(),
        "active_learning_mode": active_mode,
        "active_student": get_active_student_profile(),
        "current_session_id": current_session_id,
        "current_session_started_at": current_session_started_at,
        "tutor_state": tutor_state,
        "shared_student_context": shared_context,
        "upload_history": list(upload_history),
        "upload_documents": list(upload_documents),
        "web_research_history": list(web_research_history),
        "visual_boards": {
            mode.value: _serialize_visual_state(state)
            for mode, state in visual_boards.items()
        },
        "mode_memories": {
            mode.value: _serialize_mode_memory(memory)
            for mode, memory in mode_memories.items()
        },
    }


def _apply_runtime_state_payload(data: Dict[str, Any]) -> None:
    global S, ACTIVE_LEARNING_MODE, shared_student_context, current_session_id, current_session_started_at, session_logger
    with state_lock:
        S = _restore_tutor_state(dict(data.get("tutor_state") or {}))
        shared_student_context = str(data.get("shared_student_context", "") or "")
    with learning_mode_lock:
        ACTIVE_LEARNING_MODE = normalize_learning_mode(data.get("active_learning_mode"))
    current_session_id = str(data.get("current_session_id") or current_session_id)
    current_session_started_at = float(data.get("current_session_started_at") or time.time())
    upload_history.clear()
    upload_history.extend(list(data.get("upload_history") or []))
    upload_documents.clear()
    upload_documents.extend(list(data.get("upload_documents") or []))
    web_research_history.clear()
    web_research_history.extend(list(data.get("web_research_history") or []))
    for mode, state in visual_boards.items():
        _restore_visual_state(state, dict((data.get("visual_boards") or {}).get(mode.value) or {}))
    for mode, memory in mode_memories.items():
        _restore_mode_memory(memory, dict((data.get("mode_memories") or {}).get(mode.value) or {}))
    session_logger = SessionLogger(str(_student_session_dir(current_session_id) / "live_notes"))


def save_runtime_state() -> None:
    global _runtime_state_timer
    payload = _build_runtime_state_payload()
    with runtime_state_lock:
        _runtime_state_timer = None
        try:
            _write_json_atomic(RUNTIME_STATE_PATH, payload)
        except Exception:
            pass
        try:
            _write_json_atomic(_runtime_snapshot_path(str(payload.get("current_session_id") or "")), payload)
        except Exception:
            pass


def schedule_runtime_state_save(delay: float = 0.35) -> None:
    global _runtime_state_timer
    with runtime_state_lock:
        if _runtime_state_timer is not None:
            _runtime_state_timer.cancel()
        _runtime_state_timer = threading.Timer(delay, save_runtime_state)
        _runtime_state_timer.daemon = True
        _runtime_state_timer.start()


def load_runtime_state() -> None:
    global _runtime_restore_note
    if not RUNTIME_STATE_PATH.exists():
        return
    try:
        data = json.loads(RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
        _apply_runtime_state_payload(data)
        _runtime_restore_note = (
            f"Runtime memory restored [{ACTIVE_LEARNING_MODE.value}] "
            f"(course topic: {S.last_topic or 'none'})"
        )
    except Exception as exc:
        _runtime_restore_note = f"Runtime restore skipped: {exc}"


def remember_board(b: str):
    state = get_visual_state()
    memory = get_mode_memory()
    b = sanitize_board_output(b or "")
    if b:
        memory.board_history.append(b)
        state.live_board_text = (state.live_board_text + "\n\n" + b).strip() if state.live_board_text else b
        state.render_kind = "text"
        state.render_payload = {"text": state.live_board_text}
        schedule_runtime_state_save()


def set_visual_render_text(text: str, append: bool = False, mode: Optional[Any] = None):
    state = get_visual_state(mode)
    rendered = sanitize_board_output(text or "")
    if not append:
        state.live_board_text = rendered
    elif rendered:
        state.live_board_text = (state.live_board_text + "\n\n" + rendered).strip() if state.live_board_text else rendered

    if state.live_board_text:
        state.render_kind = "text"
        state.render_payload = {"text": state.live_board_text}
    else:
        state.render_kind = "empty"
        state.render_payload = {}
    schedule_runtime_state_save()


def set_visual_render_payload(kind: str, payload: Optional[Dict[str, Any]] = None, mode: Optional[Any] = None):
    state = get_visual_state(mode)
    state.render_kind = kind
    state.render_payload = dict(payload or {})
    schedule_runtime_state_save()


def reset_board_tracker():
    """Call this every time the board is cleared/reset."""
    state = get_visual_state()
    state.current_board_text = ""
    state.shown_board_lines = set()
    state.board_heading_buffer = ""
    state.live_board_text = ""
    state.last_emitted_heading = ""
    state.last_board_hashes.clear()
    state.render_kind = "empty"
    state.render_payload = {}
    schedule_runtime_state_save()


def _filter_duplicate_lines(text: str) -> str:
    """
    Fix 2: Remove individual lines from `text` that are already in
    _shown_board_lines (line-level dedup, catches partial repeats that
    the hash dedup misses).  Updates the set with new significant lines.
    Short lines (< 12 chars) and pure-markdown decorators are kept as-is.
    Fix 4: Also strip cross-chunk heading duplicates.
    """
    state = get_visual_state()
    out_lines = []
    for raw_ln in text.splitlines():
        key = _norm_line(raw_ln)
        stripped = raw_ln.strip()

        # Fix Issue 4: detect if this line is a heading (bold or ends with :)
        _is_heading = bool(re.match(r"^\*\*.+\*\*$", stripped)) or (stripped.endswith(":") and len(stripped) <= 60)

        # Always keep: blank lines, short decorators
        if not key or len(key) < 12:
            out_lines.append(raw_ln)
            continue

        if _is_heading:
            _norm_heading = re.sub(r"\*\*", "", key).strip()
            _norm_last = re.sub(r"\*\*", "", _norm_line(state.last_emitted_heading)).strip()
            if _norm_heading and _norm_heading == _norm_last:
                continue   # same heading already on board â€” drop it
            state.last_emitted_heading = stripped
            out_lines.append(raw_ln)
            continue

        if key in state.shown_board_lines:
            continue   # already on board â€” skip
        state.shown_board_lines.add(key)
        out_lines.append(raw_ln)
    return "\n".join(out_lines)


def _apply_heading_buffer(text: str) -> str:
    """
    Fix 3: If a previous lone-heading is buffered and the incoming text has
    real content (body lines), prepend the heading so it arrives with its body.
    If incoming text is ITSELF a lone heading, buffer it instead of emitting.
    Returns (final_text_to_emit_or_empty, heading_stored_for_next).
    """
    state = get_visual_state()

    stripped = (text or "").strip()
    if not stripped:
        return ""

    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    is_lone_heading = (
        len(lines) == 1
        and (lines[0].rstrip().endswith(":") or re.match(r"^\*\*.+\*\*$", lines[0].strip()))
        and len(lines[0].strip()) <= 60
    )

    if is_lone_heading:
        # Don't emit yet â€” buffer and wait for body
        state.board_heading_buffer = stripped
        return ""

    # Incoming text has body content
    if state.board_heading_buffer:
        combined = state.board_heading_buffer + "\n" + stripped
        state.board_heading_buffer = ""
        return combined

    return stripped


def extract_board_delta(new_bd: str) -> str:
    """
    LLM often outputs FULL cumulative board text each chunk
    (e.g. Chunk3 = Chunk1 + Chunk2 + new_line).
    This strips whatever is already on the board and returns only the NEW lines.
    """
    state = get_visual_state()
    new_bd = (new_bd or "").strip()
    if not new_bd:
        return ""

    existing = state.current_board_text.strip()

    if not existing:
        # Nothing on board yet â€” emit everything
        state.current_board_text = new_bd
        return new_bd

    # Normalise both for comparison (collapse whitespace)
    def _norm(s):
        import re as _r
        return _r.sub(r'\s+', ' ', s.strip())

    norm_existing = _norm(existing)
    norm_new      = _norm(new_bd)

    # If new text starts with existing text â†’ emit only the suffix
    if norm_new.startswith(norm_existing):
        # Find the character position in the ORIGINAL new_bd that corresponds
        # to the end of the existing content
        suffix = new_bd[len(existing):].strip()
        if suffix:
            state.current_board_text = new_bd   # update tracker to full text
            return suffix
        else:
            return ""   # nothing new

    # new text doesn't start with existing â†’ completely new section
    state.current_board_text = new_bd
    return new_bd


def remember_speech(s: str):
    memory = get_mode_memory()
    s = (s or "").strip()
    if s:
        memory.speech_history.append(s)
        schedule_runtime_state_save()


def remember_user(u: str):
    memory = get_mode_memory()
    u = (u or "").strip()
    if u:
        memory.user_history.append(u)
        schedule_runtime_state_save()


def remember_upload_context(content: str, label: str):
    content = (content or "").strip()
    label = (label or "Uploaded file").strip()
    if not content:
        return
    upload_history.append({
        "label": label,
        "content": content[:5000],
    })
    schedule_runtime_state_save()


_UPLOAD_PAGE_RE = re.compile(r"page(?:s)?\s+(\d+)(?:\s*[-?]\s*(\d+))?", re.I)
_UPLOAD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}", re.I)
_UPLOAD_QUERY_PAGE_RE = re.compile(r"\bpages?\s+(\d{1,4})(?:\s*(?:-|to)\s*(\d{1,4}))?\b", re.I)


def _extract_upload_pages(label: str) -> List[int]:
    match = _UPLOAD_PAGE_RE.search(str(label or ""))
    if not match:
        return []
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if end < start:
        start, end = end, start
    return list(range(start, min(end, start + 4) + 1))


def _upload_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _UPLOAD_TOKEN_RE.findall(str(text or "")) if len(tok) > 2}


def _extract_requested_pages(text: str) -> set[int]:
    pages: set[int] = set()
    for match in _UPLOAD_QUERY_PAGE_RE.finditer(str(text or "")):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        pages.update(range(start, min(end, start + 6) + 1))
    return pages


def remember_upload_document(chunks: List[Tuple[str, str]], filename: str, mime_type: str) -> Dict[str, Any] | None:
    stored_chunks = []
    upload_history.clear()
    for idx, (content, label) in enumerate(chunks):
        body = str(content or "").strip()
        if not body:
            continue
        record = {
            "idx": idx,
            "label": str(label or f"FROM FILE '{filename}'"),
            "content": body[:12000],
            "pages": _extract_upload_pages(label),
            "tokens": sorted(_upload_tokens(f"{label or ''}\n{body}"))[:320],
        }
        stored_chunks.append(record)
        snippet = body if extract_upload_issue(body) else _trim_context_block(body, 900)
        upload_history.append({"label": record["label"], "content": snippet})
    if not stored_chunks:
        return None
    document = {
        "doc_id": "upl_" + uuid.uuid4().hex[:8],
        "filename": str(filename or "upload"),
        "mime_type": str(mime_type or "application/octet-stream"),
        "uploaded_at": time.time(),
        "chunks": stored_chunks,
    }
    upload_documents.append(document)
    schedule_runtime_state_save()
    return document


def retrieve_upload_context(query: str = "", max_chars: int = 1400, max_chunks: int = 4) -> List[Dict[str, str]]:
    if not upload_documents:
        return list(upload_history)[-1:]

    query_text = str(query or "")
    query_tokens = _upload_tokens(query_text)
    requested_pages = _extract_requested_pages(query_text)
    scored = []
    page_matched = False
    docs = list(upload_documents)
    for doc_rank, doc in enumerate(reversed(docs)):
        recency_bonus = max(0, 4 - doc_rank)
        filename_tokens = _upload_tokens(str(doc.get("filename") or ""))
        filename_overlap = len(query_tokens & filename_tokens)
        for chunk in doc.get("chunks") or []:
            content = str(chunk.get("content") or "")
            if not content:
                continue
            chunk_tokens = set(chunk.get("tokens") or [])
            overlap = len(query_tokens & chunk_tokens)
            page_hits = len(requested_pages & set(chunk.get("pages") or []))
            if page_hits:
                page_matched = True
            score = recency_bonus + overlap * 3 + page_hits * 7 + filename_overlap * 2
            if not query_tokens and doc_rank == 0:
                score += max(0, 2 - int(chunk.get("idx") or 0))
            if query_text and query_text.lower() in content.lower():
                score += 4
            if query_text and query_text.lower() in str(chunk.get("label") or "").lower():
                score += 5
            scored.append((score, doc_rank, page_hits, chunk))
    if requested_pages and page_matched:
        scored = [item for item in scored if item[2] > 0]
    scored.sort(key=lambda item: (-item[0], item[1], int(item[3].get("idx") or 0)))
    picked_raw = []
    seen = set()
    for score, _, _, chunk in scored:
        chunk_key = (chunk.get("label"), chunk.get("idx"))
        if chunk_key in seen:
            continue
        if score <= 0 and query_tokens:
            continue
        seen.add(chunk_key)
        picked_raw.append(chunk)
        if len(picked_raw) >= max_chunks:
            break
    if not picked_raw:
        latest = docs[-1]
        for chunk in (latest.get("chunks") or [])[:max_chunks]:
            picked_raw.append(chunk)

    per_chunk_chars = max_chars // max(1, len(picked_raw))
    picked = []
    for chunk in picked_raw:
        picked.append({
            "label": str(chunk.get("label") or "Uploaded file"),
            "content": _query_focused_context_block(str(chunk.get("content") or ""), query_text, per_chunk_chars),
        })
    return picked


UPLOAD_ANALYSIS_UNAVAILABLE_RE = re.compile(
    r"^\[(?:image content unavailable|vision server unavailable):\s*(.+?)\]$",
    re.I | re.S,
)


def extract_upload_issue(text: str) -> str:
    match = UPLOAD_ANALYSIS_UNAVAILABLE_RE.match((text or "").strip())
    return match.group(1).strip() if match else ""


def build_upload_response_guidance() -> str:
    if not upload_history:
        return ""

    latest = dict(list(upload_history)[-1] or {})
    issue = extract_upload_issue(str(latest.get("content") or ""))
    if issue:
        return "\n".join([
            "- The student uploaded a file for this turn.",
            f"- The file was received, but automatic visual/file analysis failed: {issue}.",
            "- Do not say you cannot access uploaded images or PDFs in general.",
            "- Explain specifically that the vision/file-analysis server is currently unavailable, then ask the student to retry later or describe the relevant part.",
        ])

    return "\n".join([
        "- The student uploaded a file for this turn.",
        "- Use the uploaded file context from shared memory when it is relevant to the answer.",
        "- Do not claim that you cannot view or access the uploaded image/PDF when uploaded file context is present.",
    ])


UPLOAD_REFERENCE_RE = re.compile(
    r"\b(upload(?:ed)?|file|image|photo|picture|screenshot|screen ?shot|pdf|attachment|attached|"
    r"error|error message|traceback|stack trace|exception|issue|bug)\b",
    re.I,
)


def _latest_upload_age_secs() -> Optional[float]:
    if not upload_documents:
        return None
    latest = dict(list(upload_documents)[-1] or {})
    try:
        uploaded_at = float(latest.get("uploaded_at") or 0.0)
    except Exception:
        uploaded_at = 0.0
    if uploaded_at <= 0:
        return None
    return max(0.0, time.time() - uploaded_at)


def _should_use_upload_context(user_text: str, max_age_secs: float = 300.0) -> bool:
    if not upload_documents:
        return False
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    age = _latest_upload_age_secs()
    if age is not None and age > max_age_secs:
        return False
    if UPLOAD_REFERENCE_RE.search(text):
        return True
    if age is not None and age <= 90 and len(text.split()) <= 8:
        if any(token in text for token in ("this", "it", "why", "what", "help", "look", "check", "got")):
            return True
    return False


def _fallback_upload_board(upload_blocks: List[Dict[str, str]]) -> str:
    for item in upload_blocks:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        picked = lines[:4]
        if picked:
            return "Uploaded file note\n" + "\n".join(picked)
    return "Uploaded file received"


def answer_course_upload_question(
    user_text: str,
    title: str,
    topic: str,
    subtopic: str,
    phase_name: str = "",
) -> List[Tuple[str, str, List[dict]]]:
    upload_blocks = retrieve_upload_context(query=user_text, max_chars=1600, max_chunks=3)
    if not upload_blocks:
        return []
    upload_context = "\n\n".join(
        f"{item.get('label') or 'Uploaded file'}:\n{_trim_context_block(str(item.get('content') or ''), 700)}"
        for item in upload_blocks
        if str(item.get("content") or "").strip()
    ).strip()
    if not upload_context:
        return []

    emp_ctx = build_empathy_context()
    answer_prompt = (
        f"{emp_ctx}\n\n"
        f"COURSE TITLE: {title or 'Course'}\n"
        f"CURRENT TOPIC: {topic or 'General'}\n"
        f"CURRENT SUBTOPIC: {subtopic or topic or 'General'}\n"
        f"CURRENT PHASE: {phase_name or 'COURSE'}\n\n"
        f"UPLOADED FILE CONTEXT:\n{upload_context}\n\n"
        f"STUDENT MESSAGE: {user_text}\n\n"
        "Answer the student's uploaded-file question directly using the uploaded content.\n"
        "If the upload shows a Python or coding error, identify the exact error, the likely cause, and the next fix step.\n"
        "Do not ignore the uploaded file. Do not continue the lesson until you have answered the file question.\n"
        "Keep it concise, practical, and aligned with the current lesson when relevant.\n"
        "End with one short question asking whether they want to fix this first or continue the lesson."
    )
    speech = timed_llm(
        "You are a precise, practical tutor. Reply in 2-4 short sentences. No bullet list.",
        answer_prompt,
        label="Course upload answer",
        temperature=0.25,
        max_tokens=220,
        use_emotion_engine=True,
    )
    speech = _strip_latexish_math(_strip_structural_tags(str(speech or "")).strip())
    if not speech:
        speech = "I checked the uploaded file. Let me help you fix that issue first. Do you want to solve it now or continue the lesson after that?"
    if "?" not in speech:
        speech = speech.rstrip(". ") + ". Do you want to fix this first, or continue the lesson?"

    board = timed_llm(
        "Convert this into a concise chalkboard note. Plain text only. Use one short heading and 2-4 lines. Mention the exact error and the fix when present.",
        f"UPLOAD CONTEXT:\n{upload_context}\n\nANSWER:\n{speech}",
        label="Course upload board",
        temperature=0.1,
        max_tokens=100,
    )
    board = _strip_latexish_math(_strip_structural_tags(str(board or "")).strip())
    if not board:
        board = _fallback_upload_board(upload_blocks)
    return [(speech, board, [])]


def remember_web_research_context(query: str, payload: Dict[str, Any]) -> None:
    if not payload or not (payload.get("sources") or payload.get("context")):
        return
    web_research_history.append({
        "query": str(query or payload.get("query") or "").strip(),
        "sources": list(payload.get("sources") or []),
        "context": str(payload.get("context") or ""),
        "ts": time.time(),
    })


def retrieve_web_context(query: str = "", max_chars: int = 1200, max_sources: int = 3) -> Dict[str, Any]:
    turn_payload = TURN_WEB_CONTEXT.get()
    if turn_payload and (turn_payload.get("sources") or turn_payload.get("context")):
        context_text = _trim_context_block(str(turn_payload.get("context") or ""), max_chars)
        return {
            "query": str(turn_payload.get("query") or query or ""),
            "sources": list(turn_payload.get("sources") or [])[:max_sources],
            "context": context_text,
            "error": str(turn_payload.get("error") or ""),
        }
    if turn_payload and str(turn_payload.get("error") or "").strip():
        return {
            "query": str(turn_payload.get("query") or query or ""),
            "sources": [],
            "context": "",
            "error": str(turn_payload.get("error") or "").strip(),
        }
    if not web_research_history:
        return {}
    query_tokens = _upload_tokens(query or "")
    if not query_tokens:
        return {}
    best = None
    best_score = -1
    for item in reversed(web_research_history):
        item_text = f"{item.get('query', '')}\n{item.get('context', '')}"
        score = len(query_tokens & _upload_tokens(item_text))
        if score >= best_score:
            best_score = score
            best = item
    if not best or best_score <= 0:
        return {}
    return {
        "query": str(best.get("query") or ""),
        "sources": list(best.get("sources") or [])[:max_sources],
        "context": _trim_context_block(str(best.get("context") or ""), max_chars),
        "error": "",
    }


def build_web_response_guidance(query: str = "") -> str:
    payload = retrieve_web_context(query=query, max_chars=1000, max_sources=3)
    if not payload or not payload.get("sources"):
        return ""
    domains = [str(item.get("domain") or "").strip() for item in (payload.get("sources") or []) if item.get("domain")]
    domains = [domain for idx, domain in enumerate(domains) if domain and domain not in domains[:idx]]
    source_text = ", ".join(domains[:3]) or "trusted web sources"
    return "\n".join([
        "- Trusted web search was explicitly enabled for this turn.",
        "- Use the provided web snippets only for claims they support.",
        f"- Mention the source domains you used, such as: {source_text}.",
        "- If the snippets do not answer the question cleanly, say that the web context was limited instead of inventing details.",
    ])


def _append_turn_web_context(prompt: str) -> str:
    payload = retrieve_web_context(max_chars=1200, max_sources=3)
    if not payload:
        return prompt
    if not payload.get("context"):
        error_text = str(payload.get("error") or "").strip()
        if not error_text:
            return prompt
        return f"{prompt.strip()}\n\nWEB SEARCH STATUS:\nTrusted web search was enabled, but no usable source snippets were collected: {error_text}"
    guidance = build_web_response_guidance(str(payload.get("query") or ""))
    blocks = [prompt.strip()]
    if guidance:
        blocks.append("WEB GUIDANCE:\n" + guidance)
    blocks.append("TRUSTED WEB CONTEXT:\n" + str(payload.get("context") or "").strip())
    return "\n\n".join(block for block in blocks if block)


def current_web_source_tagline(max_sources: int = 3) -> str:
    payload = retrieve_web_context(max_chars=400, max_sources=max_sources)
    if not payload or not payload.get("sources"):
        return ""
    names = [str(item.get("domain") or item.get("title") or "").strip() for item in (payload.get("sources") or [])]
    names = [name for idx, name in enumerate(names) if name and name not in names[:idx]]
    if not names:
        return ""
    return "Sources checked: " + ", ".join(names[:max_sources]) + "."


def build_empathy_context() -> str:
    """Returns a compact empathy context block injected into every LLM prompt."""
    with emp_lock:
        name       = EMP.student_name
        age        = EMP.student_age
        city       = EMP.real_city
        tz         = EMP.real_tz
        temp       = EMP.weather_temp
        weather    = EMP.weather_label
        study_secs = EMP.total_study_secs
        last_brk   = EMP.last_break_secs
        brk_count  = EMP.breaks_today
        break_act  = EMP.break_active

    # Current local time in student's actual timezone
    try:
        import zoneinfo
        tz_obj = zoneinfo.ZoneInfo(tz)
        local_now = datetime.datetime.now(tz_obj)
        time_str  = local_now.strftime("%I:%M %p")
        day_str   = local_now.strftime("%A")
        hour      = local_now.hour
    except Exception:
        local_now = datetime.datetime.now()
        time_str  = local_now.strftime("%I:%M %p")
        day_str   = local_now.strftime("%A")
        hour      = local_now.hour

    # Human-readable study duration
    study_min = int(study_secs // 60)
    if study_min < 1:
        study_dur = "just started"
    elif study_min < 60:
        study_dur = f"{study_min} minutes today"
    else:
        h, m = divmod(study_min, 60)
        study_dur = f"{h}h {m}m today"

    last_brk_str = f"{int(last_brk//60)}m {int(last_brk%60)}s" if last_brk > 0 else "none yet"

    # Time-of-day sentiment for LLM awareness
    if hour < 6:
        time_note = "very late night / early morning â€” student likely exhausted"
    elif hour < 9:
        time_note = "early morning â€” student fresh"
    elif hour < 12:
        time_note = "morning â€” good energy"
    elif hour < 14:
        time_note = "post-lunch â€” may feel sleepy"
    elif hour < 17:
        time_note = "afternoon â€” moderate energy"
    elif hour < 20:
        time_note = "evening â€” winding down"
    elif hour < 22:
        time_note = "late evening"
    else:
        time_note = "night â€” student should rest soon"

    lines = [
        f"[STUDENT CONTEXT - use naturally, never read out loud]",
        "For academic/problem-solving turns, use this only to tune tone and pacing.",
        "Do not open content answers with greetings, time-of-day remarks, weather, or wellbeing check-ins unless the student asked for that.",
        f"Name: {name} | Age: {age} | City: {city}",
        f"Local time: {time_str} {day_str} ({time_note})",
        f"Weather: {weather}, {temp:.0f}C",
        f"Study today: {study_dur} | Breaks taken: {brk_count} | Last break: {last_brk_str}",
        f"Recommended: 20-min study -> 5-min break | 50 min/day soft cap",
    ]
    if break_act:
        lines.append("STATUS: Student is currently on a break - await their return.")
    return "\n".join(lines)


def _trim_context_block(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return text
    keep_head = int(max_chars * 0.35)
    keep_tail = max_chars - keep_head - 8
    return text[:keep_head].rstrip() + "\n...\n" + text[-keep_tail:].lstrip()


def _query_focused_context_block(text: str, query_text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if not content or len(content) <= max_chars:
        return content

    q_tokens = _upload_tokens(query_text)
    if not q_tokens:
        return _trim_context_block(content, max_chars)

    raw_lines = [line.rstrip() for line in content.splitlines()]
    lines = [line for line in raw_lines if line.strip()]
    if len(lines) < 3:
        return _trim_context_block(content, max_chars)

    window_size = min(10, max(4, len(lines)))
    best_score = -1
    best_window = ""
    best_start = 0
    best_end = 0
    for start in range(0, len(lines)):
        window_lines = lines[start:start + window_size]
        if not window_lines:
            continue
        window_text = "\n".join(window_lines).strip()
        if not window_text:
            continue
        window_tokens = _upload_tokens(window_text)
        overlap = len(q_tokens & window_tokens)
        phrase_bonus = 0
        lowered_window = window_text.lower()
        lowered_query = str(query_text or "").lower()
        if lowered_query and lowered_query in lowered_window:
            phrase_bonus += 5
        score = overlap * 4 + phrase_bonus
        if score > best_score:
            best_score = score
            best_window = window_text
            best_start = start
            best_end = min(len(lines), start + len(window_lines))
    if best_score <= 0 or not best_window:
        return _trim_context_block(content, max_chars)
    expand_start = best_start
    expand_end = best_end
    expanded = best_window
    while len(expanded) < max_chars:
        grew = False
        if expand_start > 0:
            candidate = "\n".join(lines[expand_start - 1:expand_end]).strip()
            if len(candidate) <= max_chars:
                expand_start -= 1
                expanded = candidate
                grew = True
        if expand_end < len(lines):
            candidate = "\n".join(lines[expand_start:expand_end + 1]).strip()
            if len(candidate) <= max_chars:
                expand_end += 1
                expanded = candidate
                grew = True
        if not grew:
            break
    best_window = expanded
    if len(best_window) > max_chars:
        return _trim_context_block(best_window, max_chars)
    return best_window


def build_resume_context(
    max_chars: int = 2200,
    board_chars: int = 900,
    speech_chars: int = 280,
    user_chars: int = 220,
    include_uploads: bool = True,
    mode: Optional[Any] = None,
    query: str = "",
) -> str:
    target_mode = normalize_learning_mode(mode) if mode is not None else get_effective_learning_mode()
    memory = get_mode_memory(target_mode)
    last_board  = _trim_context_block("\n---\n".join(list(memory.board_history)[-2:]), board_chars)
    last_speech = _trim_context_block(" | ".join(list(memory.speech_history)[-2:]), speech_chars)
    last_users  = _trim_context_block(" | ".join(list(memory.user_history)[-2:]), user_chars)
    last_uploads = retrieve_upload_context(query=query) if include_uploads else []
    last_web = retrieve_web_context(query=query, max_chars=900, max_sources=3)
    pieces = []
    if target_mode == LearningMode.COURSE:
        tp = ", ".join(S.taught_points[-8:])
        if S.last_topic:
            pieces.append(f"Current topic: {S.last_topic}")
        if S.subtopics and S.subtopic_idx < len(S.subtopics):
            pieces.append(f"Current subtopic: {S.subtopics[S.subtopic_idx]}")
        if S.subtopics_done:
            pieces.append(f"Subtopics already covered: {', '.join(S.subtopics_done)}")
        if tp:
            pieces.append(f"Already taught points: {tp}")
        for t_name, t_pts in list(S.all_taught.items())[-3:]:
            if t_pts:
                pieces.append(f"From topic '{t_name}': {', '.join(t_pts[:6])}")
        if S.prior_known_indices and S.topics:
            known = [S.topics[i] for i in S.prior_known_indices if i < len(S.topics)]
            if known:
                pieces.append(f"Student self-reported prior knowledge: {', '.join(known)}")
    effective_student_context = (shared_student_context or S.student_context).strip()
    if effective_student_context:
        pieces.append(f"Student setup/background: {effective_student_context}")
    camera_context = _camera_context_block()
    if camera_context:
        pieces.append("Recent camera/action context:\n" + camera_context)
    if last_uploads:
        upload_blocks = []
        for item in last_uploads:
            snippet = _trim_context_block(item["content"], 700)
            upload_blocks.append(f"{item['label']}:\n{snippet}")
        pieces.append("Uploaded file context:\n" + "\n\n".join(upload_blocks))
    if last_web and last_web.get("context"):
        pieces.append("Trusted web context:\n" + str(last_web.get("context") or ""))
    if last_board:
        pieces.append(f"Board recently:\n{last_board}")
    if last_speech:
        pieces.append(f"Tutor recently said: {last_speech}")
    if last_users:
        pieces.append(f"Student recently said: {last_users}")
    return _trim_context_block("\n".join(pieces).strip(), max_chars)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Fallback heuristics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FALLBACK_TUTOR_REQ = re.compile(
    r"\b(teach me|from scratch|step by step|syllabus|course|i want to learn|i wanna learn)\b", re.I
)

SESSION_END_RE = re.compile(
    r"\b(good night|goodnight|bye|goodbye|see you|see ya|take care|"
    r"i'?m (done|leaving|quitting|logging off)|"
    r"i'?m going (?:away|offline|to sleep|home)|"
    r"(close|end|quit|stop) (the )?(session|class|lesson)|"
    r"talk (to you )?later|catch you later)\b", re.I
)
SESSION_PAUSE_RE = re.compile(
    r"\b(i'?m (so |very |really |feeling )?(tired|exhausted|sleepy|drained)|"
    r"(continue|resume|study) later|need (a |some )?(break|rest)|"
    r"take a break|shall we (continue|stop|pause|resume) later|"
    r"can we (stop|pause|take a break)|not now|another (day|time))\b", re.I
)
QA_OPTION_RE = re.compile(r'^(option\s*)?[A-Da-d]\.?$', re.I)
QA_OPTION_PREFIX_RE = re.compile(r'^option\s+[A-Da-d]', re.I)


def _looks_like_qa_answer(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return (
        len(raw.split()) <= 3
        or bool(QA_OPTION_RE.match(raw))
        or bool(QA_OPTION_PREFIX_RE.search(raw))
    )


def _is_quiz_resume_signal(text: str, intent: str) -> bool:
    lowered = str(text or "").strip().lower()
    if intent in ("continue_lesson", "confirm_start"):
        return True
    return lowered in {
        "continue", "go on", "go ahead", "next", "proceed", "resume",
        "keep going", "keep goin", "carry on", "okay", "ok", "yes", "yep", "yeah",
    }


def is_tutor_request(txt: str) -> bool:
    return bool(FALLBACK_TUTOR_REQ.search(txt or ""))


def guess_subject_type(txt: str) -> str:
    t = (txt or "").lower()
    if any(w in t for w in ["python", "program", "code", "java", "c++", "loop", "function", "variable"]):
        return "programming"
    if any(w in t for w in ["math", "integral", "derivative", "algebra", "trigonometry"]):
        return "math"
    if any(w in t for w in ["grammar", "english", "tamil", "language", "vocab"]):
        return "language"
    return "general"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… SYSTEM PROMPTS â€” v3
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ROUTER_SYSTEM = r"""
You are the tutoring system's intent router.
Return ONLY JSON. No extra text.

Allowed intents:
- "smalltalk"            (greetings, audible checks, how are you)
- "tutor_request"        (wants a structured course / from scratch / teach me X)
- "prior_knowledge_claim" (student says "I already know X" or "I've covered X and Y")
- "continue_lesson"      (resume last lesson)
- "confirm_start"        (agrees to begin/continue: yes, okay, go ahead, sure, start)
- "reject_or_skip"       (skip, not needed, move on, next)
- "ask_question"         (student asks a doubt about the content)
- "answer_qa"            (student is answering a quiz question)
- "not_understood"       (student says confused, didn't get it, explain again)
- "meta_complaint"       (complaining about system behavior)
- "session_pause"        (student wants to pause, take a break, continue later, feeling tired, need rest)
- "session_end"          (student says bye, goodbye, good night, closing the session, I'm done, quit)
- "unknown"

CRITICAL ROUTING RULES:
- If student says anything like "I'm tired", "shall we continue later", "I need a break",
  "I'm done for today", "can we pause" -> ALWAYS return "session_pause"
- If student says "bye", "goodbye", "good night", "I'm leaving", "close the session",
  "shall I quit", "see you later", "that's it for today" -> ALWAYS return "session_end"
- These two take HIGHEST priority over all other intents when clearly present.

Also output:
- tone: "tired" | "frustrated" | "excited" | "normal"
- confidence: 0.0 to 1.0
- one_line_reply: short natural teacher reply suggestion (max 12 words)

JSON schema:
{
  "intent": "...",
  "tone": "...",
  "confidence": 0.0,
  "one_line_reply": "..."
}
"""


AGENDA_SYSTEM = """
You are a master curriculum designer.
Return ONLY JSON with: title (string), topics (list of 3â€“7 strings).
Topics must be the right granularity â€” not too broad, not too narrow.
Order from foundational to advanced.
No extra text.
"""


SUBTOPIC_PLAN_SYSTEM = """
You are an expert curriculum designer.
Given a COURSE with multiple main topics and ONE specific main topic to plan subtopics for,
generate 2â€“5 UNIQUE subtopics that break down ONLY that specific main topic internally.

CRITICAL RULES:
- Subtopics MUST be sub-components of the given main topic â€” NOT other main topics from the course.
- NEVER use any of the other course topics as a subtopic. They are already scheduled separately.
- Each subtopic is a distinct concept/skill that deserves its own teaching block.
- Think: what are the internal building blocks of this ONE topic?
- Order from most fundamental to most advanced.
- If STUDENT_PREFERENCE mentions "minimal", "simple", "quick" â€” use 2-3 subtopics only.

Return ONLY JSON: {"subtopics": ["subtopic1", "subtopic2", ...]}
No extra text.
"""


STEP_SYSTEM = r"""
You are a REAL expert human professor with THREE capabilities you must use together:
  VOICE - your <speech> is spoken aloud via TTS (the student HEARS you)
  BOARD - your <board> is written on the blackboard (the student SEES it)
  EARS  - the student's words reach you via speech recognition
Use all three. Your speech explains what is on the board. The board shows what you are saying.

==== OUTPUT FORMAT (STRICT) ====
Output ONLY multiple CHUNK blocks. Nothing outside CHUNKs.

Each CHUNK:
<speech>...</speech>
<board>...</board>
<fx>...</fx>
<meta>...</meta>
<CHUNK>

==== BOARD -> SPEECH SYNC (MOST CRITICAL - RULE #1) ====
- Speech MUST describe and explain exactly what appears on the board in the SAME chunk.
- Board shows the content -> Speech narrates/explains that content simultaneously.
- NEVER say something in speech that is not on the board in that chunk.
- NEVER put something on the board that speech does not explain in that chunk.
- CORRECT: board shows "name = Alice" -> speech says "Here, name is a string variable holding Alice."
- WRONG: board shows code but speech talks about something else.

==== BOARD FREQUENCY (MANDATORY - RULE #2) ====
- EVERY teaching chunk MUST have a non-empty <board> block. Empty board = FORBIDDEN.
- Maximum 2 speech sentences per chunk. After 2 sentences, start a new CHUNK with new board content.
- This keeps board and speech in constant sync - the student always sees what you're saying.
- WRONG: 6 speech sentences in one chunk with one board item.
- CORRECT: 6 speech sentences = 3 chunks of 2 sentences each, each with its own board item.

==== BOARD FORMATTING RULES ====
- Each <board> block MUST contain ONLY the NEW content for that chunk - NOT a repeat.
- NEVER output the full accumulated board. Output only what is NEW in this chunk.
- Section/sub-headings MUST use bold markdown: **Heading** (this renders in yellow on the board).
  Examples: **Variables:**, **Primitive Data Types:**, **Type Conversion:**
- NEVER write plain unbolded sub-headings. Always **bold** them.
- Write a section heading ONLY ONCE. Never repeat it in later chunks.
- New items use the format:  - ItemName: value_or_description
- Sub-items append naturally - no heading or prior bullet repetition ever.
- NEVER write "Subtopic:" or "Topic:" or "Section:" as a prefix.

==== DEPTH RULES ====
- Cover the subtopic COMPLETELY. Do not rush. Do not skip aspects.
- Use real examples, analogies, edge cases.
- If the topic has multiple components, each gets its own CHUNK.
- If student is tired/frustrated (from tone context), slow down, simplify.
- Ask rhetorical questions in speech to keep student engaged.
- ALWAYS use the student context (Python version, setup) to personalize examples.
  If student is on Python 3.10 -> say "since you're on 3.10..." when relevant.

==== STUDENT CONTEXT AWARENESS ====
- If student mentioned they already have something installed/configured -> skip that step.
- Reference their version, setup, or background naturally in speech.
- Never re-teach something the student explicitly said they already know.

==== SUMMARY RULES (CRITICAL) ====
- A summary ONLY recaps points ACTUALLY taught in THIS session's chunks.
- NEVER include data, examples, or concepts not yet covered in any chunk above.
- If only 2 points were taught, the summary has exactly 2 points - not 5.

==== COMPLETION RULES (MANDATORY - RULE #3) ====
- The FINAL chunk of a subtopic MUST have "subtopic_complete": true in meta.
- The FINAL chunk speech MUST end with a CLEAR, UNMISSABLE invitation. Examples:
  "That's everything on [topic]. Ready for the next part?"
  "We've finished [topic]. Any questions before I move on?"
  "Got all of that? Say 'continue' whenever you're ready."
- Make the closing line its OWN sentence - never buried at the end of a long paragraph.
- The student must clearly know it's their turn to respond.

==== FX RULES ====
- pop: for new terms, list items, outputs.
- glow: for headings, keywords, conclusions.
- Max 2 FX per CHUNK. Only when it truly helps learning.
- FX target must appear verbatim on the board in the same CHUNK.

==== META JSON ====
{
  "taught_points": ["point1", "point2"],  // new concepts covered in THIS chunk
  "subtopic_complete": true/false,        // true on the FINAL chunk of the subtopic - MANDATORY
  "needs_example": true/false
}
"""


CONFIDENCE_CHECK_SYSTEM = """
You are an experienced human tutor assessing student readiness for Q&A.
Based on the TAUGHT_CONTENT and STUDENT_INTERACTIONS, decide if the student is ready.

Rules:
- ready=true  if the subtopics were taught and student showed SOME engagement (yes/ok/understood = enough).
- ready=true  if student asked clarification questions â€” asking questions means they are ENGAGED and learning, NOT confused.
- ready=false ONLY if student explicitly said they did NOT understand, or most of the teaching was skipped.
- NEVER say ready=false just because the student asked one or two questions â€” that is normal learning.
- qa_count: 3 for short topics, 4-5 for detailed topics.

Return ONLY JSON:
{
  "ready": true/false,
  "reason": "one sentence",
  "qa_count": 3
}
"""


QA_GEN_SYSTEM = """
You are a quiz generator for a human tutor.
Generate exactly {count} questions from ONLY the provided TAUGHT_POINTS.

Rules:
- Mix: some MCQ (with A/B/C/D), some direct-answer questions.
- MCQ: 1 correct + 3 plausible wrong options. Wrong options should test common misconceptions.
- Simple: require a short direct answer. Must be answerable from TAUGHT_POINTS.
- NEVER ask about anything not in TAUGHT_POINTS.
- "concept" field: what specific point this question tests.

Return ONLY JSON:
{
  "questions": [
    {
      "type": "mcq",
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A",
      "concept": "..."
    },
    {
      "type": "simple",
      "question": "...",
      "answer": "expected answer keywords",
      "concept": "..."
    }
  ]
}
"""


QA_EVAL_SYSTEM = """
You are a kind, encouraging human tutor evaluating a student's answer.

For MCQ: correct if student says the right letter (A/B/C/D) OR the text of the right option.
For simple: correct if student's answer contains the key concepts (exact wording not required).

Return ONLY JSON:
{
  "correct": true/false,
  "feedback": "One warm, natural sentence. If wrong, give a subtle hint. Never robotic.",
  "partial": true/false
}

Rules:
- Never say robotic lines like "The student did not provide..."
- If student is trying to skip, set correct=false, feedback="Okay, let's revisit that one."
- Be encouraging even when wrong.
"""


QA_SIMILAR_SYSTEM = """
You are a quiz generator.
Generate ONE question similar in difficulty and concept to the WRONG_QUESTION, but different wording.
Tests the exact same CONCEPT. Not a repeat of the original question.

Return ONLY JSON:
{
  "type": "mcq" or "simple",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},  // only if mcq
  "answer": "...",
  "concept": "..."
}
"""


RETEACH_SYSTEM = r"""
You are a human tutor re-explaining a concept a student got wrong.
Be warm and clear. Use a different approach/angle than what you taught before.
Use a concrete example or analogy.

Same CHUNK format as always:
<speech>...</speech>
<board>...</board>
<fx>...</fx>
<meta>{"taught_points": [], "subtopic_complete": false, "needs_example": false}</meta>
<CHUNK>

Board -> Speech sync: speech must explain exactly what's on the board.
Keep it SHORT - 1-3 chunks max.
"""


PRIOR_KNOWLEDGE_SYSTEM = """
You are analyzing a student's statement to detect which topics they claim to already know.
TOPICS_LIST is the ORDERED list of topics â€” index 0 is the FIRST topic, index 1 is the SECOND, etc.

CRITICAL RULES:
- "topic 1" or "the first topic" ALWAYS means index 0 (the very first item in TOPICS_LIST).
- "topic 2" or "the second topic" ALWAYS means index 1.
- NEVER confuse topic NUMBER (1-based position) with topic CONTENT name.
- If student says "I know topic 1 and 2", return known_indices: [0, 1].
- Match by POSITION first, then by content name if no positional reference found.
- known_indices must be 0-based (subtract 1 from any topic number the student mentions).

Return ONLY JSON:
{
  "known_indices": [0, 1],  // 0-based indices of topics student claims to know
  "confidence": 0.0         // 0.0 to 1.0 â€” how confident the student actually claimed prior knowledge
}

If no prior knowledge claim, return {"known_indices": [], "confidence": 0.0}
"""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… LLM HELPER FUNCTIONS â€” v3
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SHALLOW_MODE_SYSTEM = r"""
You are running SHALLOW MODE, the tutor's free-mode teaching system.

This mode is NOT the course engine:
- Do not create a syllabus, agenda, quiz, or long lesson plan unless the student explicitly asks.
- Answer the student's current request directly.
- Keep the reasoning shallow-but-clear: start from the immediate idea, then one useful example, then one practical next point if needed.
- Reuse shared memory naturally when relevant, but do not repeat old board content.
- If the student is asking a concrete academic question, start with the answer or the first solving step, not a greeting.
- Do not mention time of day, weather, or the student's name unless the student's main intent is social/wellbeing talk.
- If the student says things like "step by step", "solve it", "show me", or similar follow-ups, continue the current problem from memory/board instead of restarting with small talk.
- If the student gives a brief acknowledgement like "okay", "yes", or "go on" while a shallow explanation is already in progress, continue from the next teaching step instead of restarting the topic.
- For math or algebra:
  - use plain text only, never LaTeX delimiters like \( \), \[ \], or $...$
  - show each transformation clearly on the board
  - explicitly state the final answer
  - if asked for step-by-step, split the explanation into small ordered chunks
- Avoid filler like "Good evening" or "Let's tackle that together" on direct problem-solving turns.
- When teaching rather than just answering a tiny factoid, finish with one short forward-moving line or a natural check-in so the response feels complete.

OUTPUT FORMAT:
Output ONLY CHUNK blocks. Nothing else.

Each chunk:
<speech>...</speech>
<board>...</board>
<fx>...</fx>
<meta>{"clear_before": false}</meta>
<CHUNK>

Rules:
- Casual chat or tiny acknowledgements: keep <board> empty.
- Teaching/explanation: keep <board> concise and supportive, not a full course board.
- If you mention code, formulas, syntax, or structured steps, show them on the board.
- Maximum 2 speech sentences per chunk.
- Board and speech must stay synchronized in the same chunk.
- If the student switches to a genuinely new topic and the previous shallow board would be misleading, set "clear_before": true in the FIRST chunk only.
- Use FX sparingly.
"""

SHALLOW_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b[\s!.?,]*$", re.I)
SHALLOW_MATH_RE = re.compile(
    r"(\bsolve\b|\bfind\b|\bwhat is\b|\bequation\b|\balgebra\b|\bstep by step\b|[a-zA-Z]\s*[+\-*/=]\s*[a-zA-Z0-9]|\d+\s*[+\-*/=]\s*\d+)",
    re.I,
)
SHALLOW_CONTINUE_RE = re.compile(
    r"\b(step by step|solve it|show me|continue|go on|next step|then show me|explain that|how)\b",
    re.I,
)
SHALLOW_ACK_CONTINUE_RE = re.compile(
    r"^\s*(ok|okay|yes|yeah|yep|right|go on|continue|next|hmm|hmmm|alright)\s*[!.?]*\s*$",
    re.I,
)
SHALLOW_BROAD_REQUEST_RE = re.compile(
    r"\b(i need to know|teach me|i want to learn|i wanna learn|i am new|i'm new|basics|beginner|from scratch|start with|what is|what's|structure|overview)\b",
    re.I,
)
SHALLOW_GENERIC_FOLLOWUP_RE = re.compile(
    r"\b(example question|one example|another example|basic example|real[- ]world example|real[- ]life example|practical example|use case|application|make it simple|simpler|break it down|explain again|quiz me|next one)\b",
    re.I,
)
SHALLOW_EXPLICIT_BRANCH_RE = re.compile(
    r"\b(start with|now|switch to|move to)\s+(verbal|quant|quantitative|writing|essay)\b",
    re.I,
)
SHALLOW_GRE_RE = re.compile(r"\bgre\b", re.I)
SHALLOW_GRE_STRUCTURE_RE = re.compile(r"\b(gre).*\b(structure|format|section|sections|pattern|overview|score|timing)\b|\b(structure|format|section|sections|pattern|overview|score|timing).*\b(gre)\b", re.I)
SHALLOW_LATEST_RE = re.compile(r"\b(current|latest|recent|today|updated|official)\b", re.I)
SHALLOW_TRIG_RE = re.compile(
    r"\b(trigonometry|trigonom[a-z]*|trig|sine|cosine|tangent|sin\b|cos\b|tan\b|hypotenuse|adjacent|opposite|right triangle|triangle)\b",
    re.I,
)
SHALLOW_EXAMPLE_RE = re.compile(
    r"\b(example|real[- ]world|real[- ]life|practical|application|use case)\b",
    re.I,
)
SHALLOW_FORMAT_CONFUSION_RE = re.compile(
    r"\b(that'?s it|that is it|is that|what kind of|how is that|why is that)\b.*\b(question|problem)\b|"
    r"\b(confus(?:ed|ing)?|don'?t get it|dont get it|not clear|hard to follow)\b",
    re.I,
)
SHALLOW_GRE_LEAK_RE = re.compile(
    r"\bgre\b|\bverbal reasoning\b|\bquantitative reasoning\b|\banalytical writing\b|\binference\b|\breading passage\b",
    re.I,
)


def _classify_shallow_focus(text: str, recent_has_gre: bool = False) -> str:
    low = (text or "").lower()
    if not low.strip():
        return ""
    if SHALLOW_TRIG_RE.search(low):
        return "Trigonometry"
    if "quant" in low or "quantitative" in low:
        return "GRE Quantitative Reasoning" if ("gre" in low or recent_has_gre) else "Quantitative Reasoning"
    if "verbal" in low:
        return "GRE Verbal Reasoning" if ("gre" in low or recent_has_gre) else "Verbal Reasoning"
    if any(word in low for word in ("writing", "essay", "issue task", "analytical writing")):
        return "GRE Analytical Writing" if ("gre" in low or recent_has_gre) else "Analytical Writing"
    if "gre" in low:
        return "GRE General Test"
    if "python" in low and "loop" in low:
        return "Python loops"
    if "python" in low and "variable" in low:
        return "Python variables"
    if "python" in low:
        return "Python"
    if any(word in low for word in ("math", "mathematics", "algebra", "geometry", "equation")):
        return "Math"
    return ""


def _focus_needs_followup_question(user_text: str, focus: str = "", broad_request: bool = False) -> bool:
    low = (user_text or "").strip().lower()
    if broad_request:
        return True
    if SHALLOW_GENERIC_FOLLOWUP_RE.search(low):
        return True
    if focus and len(low.split()) <= 8 and SHALLOW_EXAMPLE_RE.search(low):
        return True
    return False


def _interactive_followup_chunk(focus: str = "") -> Tuple[str, str, List[dict], Dict[str, Any]]:
    low_focus = (focus or "").lower()
    if "trigonometry" in low_focus:
        question = "Do you want one more example with shadows and heights, or a quick practice question?"
    elif "python" in low_focus:
        question = "Do you want one tiny code example next, or a quick practice question?"
    elif "gre verbal" in low_focus:
        question = "Do you want one more verbal example, or a quick mini question to try?"
    elif "gre quant" in low_focus:
        question = "Do you want one more quant example, or a short problem to solve?"
    elif "gre" in low_focus:
        question = "Do you want to continue with Verbal, Quant, or Writing next?"
    elif focus:
        question = f"Do you want one more example in {focus}, or a quick practice question next?"
    else:
        question = "Do you want one more example, or a quick practice question next?"
    return (question, "", [], {"clear_before": False})


def _deterministic_shallow_chunks(user_text: str, focus: str = "") -> List[Tuple[str, str, List[dict], Dict[str, Any]]]:
    low = (user_text or "").lower()
    low_focus = (focus or "").lower()
    if "trigonometry" in low_focus and SHALLOW_EXAMPLE_RE.search(low):
        speech = (
            "A simple real-world trigonometry example is finding the height of a building from its shadow. "
            "If the angle of elevation is 45 degrees and the shadow is 10 meters, then tan 45 equals height divided by 10, so the height is 10 meters."
        )
        board = (
            "Real-world trigonometry\n"
            "tan(theta) = opposite / adjacent\n"
            "Example:\n"
            "tan 45 = h / 10\n"
            "1 = h / 10\n"
            "h = 10 m"
        )
        return [
            (speech, board, [], {"clear_before": False}),
            _interactive_followup_chunk(focus),
        ]
    recent_memory = get_mode_memory(LearningMode.SHALLOW)
    recent_context = "\n".join(
        list(recent_memory.board_history)[-2:] + list(recent_memory.speech_history)[-2:] + list(recent_memory.user_history)[-2:]
    ).lower()
    if "gre quantitative reasoning" in low_focus and SHALLOW_ACK_CONTINUE_RE.fullmatch((user_text or "").strip()):
        if any(marker in recent_context for marker in ("average speed", "speed =", "distance / time", "arithmetic")):
            speech = (
                "Good. Let us stay with the same kind of arithmetic word problem first so the pattern becomes natural. "
                "Try this one yourself: a bus travels 180 kilometres in 3 hours. What is its average speed in kilometres per hour?"
            )
            board = (
                "Practice problem\n"
                "Speed = Distance / Time\n"
                "Distance = 180 km\n"
                "Time = 3 h\n"
                "Find speed"
            )
            follow = "Take a moment and tell me your answer, and I will check your reasoning."
            return [
                (speech, board, [], {"clear_before": False}),
                (follow, "", [], {"clear_before": False}),
            ]
    if "gre quantitative reasoning" in low_focus and SHALLOW_FORMAT_CONFUSION_RE.search(low):
        speech = (
            "Fair question. Yes, that is a real GRE Quantitative Comparison question, but it feels odd at first because you compare two columns instead of solving for one final number. "
            "Let me make it friendlier: we will start with a regular problem-solving question, and once that feels easy, I will show you how the comparison format works."
        )
        board = (
            "GRE Quant question styles\n"
            "1. Problem Solving\n"
            "2. Quantitative Comparison\n"
            "Start with Problem Solving first"
        )
        speech2 = "Here is a cleaner starter question: a notebook costs 40 rupees and you buy 3 notebooks. What is the total cost?"
        board2 = (
            "Starter problem\n"
            "Cost of 1 notebook = 40\n"
            "Number of notebooks = 3\n"
            "Total cost = ?"
        )
        follow = "Do you want to solve this yourself first, or should I walk through it step by step?"
        return [
            (speech, board, [], {"clear_before": False}),
            (speech2, board2, [], {"clear_before": False}),
            (follow, "", [], {"clear_before": False}),
        ]
    return []


def _needs_focus_repair(user_text: str, focus: str, raw_output: str) -> bool:
    low_focus = (focus or "").lower()
    if not low_focus:
        return False
    clean = _strip_structural_tags(_strip_latexish_math(str(raw_output or "")))
    if not clean.strip():
        return False
    low_user = (user_text or "").lower()
    if "gre" not in low_focus and "gre" not in low_user and SHALLOW_GRE_LEAK_RE.search(clean):
        return True
    if "trigonometry" in low_focus and SHALLOW_EXAMPLE_RE.search(low_user) and not SHALLOW_TRIG_RE.search(clean):
        return True
    return False


def _infer_shallow_focus(user_text: str = "") -> str:
    memory = get_mode_memory(LearningMode.SHALLOW)
    recent_user = list(memory.user_history)[-4:]
    recent_other = list(memory.board_history)[-2:] + list(memory.speech_history)[-2:]
    recent_has_gre = "gre" in " ".join(str(bit or "") for bit in (recent_user + recent_other)).lower()
    current_focus = _classify_shallow_focus(user_text, recent_has_gre=recent_has_gre)
    if current_focus:
        return current_focus
    for text in reversed(recent_user):
        focus = _classify_shallow_focus(str(text or ""), recent_has_gre=recent_has_gre)
        if focus:
            return focus
    for text in reversed(recent_other):
        focus = _classify_shallow_focus(str(text or ""), recent_has_gre=recent_has_gre)
        if focus:
            return focus
    return ""


def _is_shallow_broad_request(user_text: str, has_existing_context: bool = False) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    if SHALLOW_MATH_RE.search(text):
        return False
    if SHALLOW_EXPLICIT_BRANCH_RE.search(text):
        return False
    if has_existing_context and (SHALLOW_CONTINUE_RE.search(text) or SHALLOW_GENERIC_FOLLOWUP_RE.search(text)):
        return False
    words = text.split()
    return bool(SHALLOW_BROAD_REQUEST_RE.search(text)) and len(words) <= 10


def _should_auto_ground_shallow_query(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    if SHALLOW_GRE_STRUCTURE_RE.search(text):
        return True
    if SHALLOW_GRE_RE.search(text) and SHALLOW_LATEST_RE.search(text):
        return True
    return False


def _source_tagline_from_payload(payload: Dict[str, Any] | None, max_sources: int = 2) -> str:
    if not payload or not payload.get("sources"):
        return ""
    names = [str(item.get("domain") or item.get("title") or "").strip() for item in (payload.get("sources") or [])]
    names = [name for idx, name in enumerate(names) if name and name not in names[:idx]]
    if not names:
        return ""
    return "Sources checked: " + ", ".join(names[:max_sources]) + "."


def _interactive_intro_chunk(user_text: str, focus: str = "") -> Tuple[str, str, List[dict], Dict[str, Any]]:
    low = (user_text or "").lower()
    if "python" in low:
        question = "Have you written any code before, or is this your first time with Python?"
    elif "gre" in low:
        question = "Do you want the GRE format first, or should we start with Verbal, Quant, or Writing?"
    elif "math" in low or "algebra" in low or "trigonometry" in low:
        question = "Do you want the concept first, or one worked example right away?"
    elif focus:
        question = f"Do you want the basics of {focus} first, or should I start with one simple example?"
    else:
        question = "Do you want the basics first, or one simple example to start?"
    return (question, "", [], {"clear_before": False})


def _current_gre_structure_chunks(payload: Dict[str, Any] | None = None) -> List[Tuple[str, str, List[dict], Dict[str, Any]]]:
    tagline = _source_tagline_from_payload(payload, max_sources=1)
    speech = (
        "Based on the current ETS GRE General Test structure, the exam has one Analytical Writing task, "
        "two Verbal Reasoning sections, and two Quantitative Reasoning sections. "
        "The total test time is about 1 hour and 58 minutes."
    )
    if tagline:
        speech += f" {tagline}"
    board = (
        "Current ETS GRE Structure\n"
        "1. Analytical Writing: 1 'Analyze an Issue' task - 30 min\n"
        "2. Verbal Reasoning: Section 1 = 12 questions / 18 min; Section 2 = 15 questions / 23 min\n"
        "3. Quantitative Reasoning: Section 1 = 12 questions / 21 min; Section 2 = 15 questions / 26 min\n"
        "Order: Writing first; then Verbal and Quant in either order.\n"
        "Adaptive design: Verbal and Quant are section-level adaptive."
    )
    follow = "Do you want to start with Verbal, Quant, or the Writing task next?"
    return [
        (speech, board, [], {"clear_before": False}),
        (follow, "", [], {"clear_before": False}),
    ]


def _strip_latexish_math(text: str) -> str:
    if not text:
        return ""
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\times", " times ")
    text = text.replace(r"\cdot", " times ")
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", r"\1", text)
    return text


def build_shallow_mode_guidance(user_text: str) -> str:
    low = (user_text or "").strip().lower()
    shallow_memory = get_mode_memory(LearningMode.SHALLOW)
    lines = []
    is_greeting_only = bool(SHALLOW_GREETING_RE.fullmatch((user_text or "").strip()))
    is_math_turn = bool(SHALLOW_MATH_RE.search(low))
    is_follow_up = bool(SHALLOW_CONTINUE_RE.search(low))
    is_generic_followup = bool(SHALLOW_GENERIC_FOLLOWUP_RE.search(low))
    is_ack_continue = bool(SHALLOW_ACK_CONTINUE_RE.fullmatch((user_text or "").strip()))
    has_existing_context = bool(
        shallow_memory.board_history
        or shallow_memory.speech_history
        or get_visual_state(LearningMode.SHALLOW).live_board_text.strip()
    )
    focus = _infer_shallow_focus(user_text)
    is_broad_request = _is_shallow_broad_request(user_text, has_existing_context=has_existing_context)

    if not is_greeting_only:
        lines.append("- This is not a social opener. Do not greet or mention time/weather/name.")
    if is_math_turn:
        lines.extend([
            "- This is a direct problem-solving turn. Start with the first solving step immediately.",
            "- Use plain text math only. Never use LaTeX delimiters.",
            "- Put the algebra or arithmetic steps on the board.",
            "- Always include the final answer explicitly.",
        ])
    if is_follow_up:
        lines.extend([
            "- The student is continuing the current problem from shared memory/board.",
            "- Continue the existing solution instead of restarting the conversation socially.",
        ])
    if (is_generic_followup or is_ack_continue or is_follow_up) and focus:
        lines.extend([
            f"- The active shallow subject from memory is: {focus}.",
            f"- Stay on {focus}. Do not switch to a different domain or random example.",
            "- If the student asks for an example or a simpler version, keep it inside the active subject.",
        ])
    if focus and "gre" not in (focus or "").lower():
        lines.extend([
            "- Do not drift into GRE, verbal reasoning, quant sections, essays, inference drills, or news-headline examples unless the student explicitly asks for GRE.",
            "- Ignore any stale exam-style context that does not match the current active subject.",
        ])
    if "verbal" in low and "gre" in (focus or "").lower():
        lines.extend([
            "- The student explicitly chose GRE Verbal Reasoning.",
            "- Do not drift to GRE Quantitative or Analytical Writing in this answer.",
        ])
    if ("quant" in low or "quantitative" in low) and "gre" in (focus or "").lower():
        lines.extend([
            "- The student explicitly chose GRE Quantitative Reasoning.",
            "- Do not drift back to GRE Verbal or Analytical Writing in this answer.",
        ])
    if any(word in low for word in ("writing", "essay")) and "gre" in (focus or "").lower():
        lines.extend([
            "- The student explicitly chose GRE Analytical Writing.",
            "- Stay on the writing task and do not drift to verbal or quant.",
        ])
    if is_ack_continue and has_existing_context:
        lines.extend([
            "- The student's brief acknowledgement means: continue the current shallow explanation.",
            "- Do not restart from the beginning. Take the next concrete teaching step from shared memory/board.",
        ])
    if is_broad_request and not has_existing_context:
        lines.extend([
            "- This is a broad beginner request.",
            "- Be interactive first: ask one short calibration question before giving more than one concept.",
            "- After that question, give only a compact orientation or first concept, not a lecture dump.",
            "- Keep the first response beginner-friendly and leave room for the student's reply.",
        ])
    if "step by step" in low:
        lines.extend([
            "- Give the solution in small ordered steps.",
            "- Prefer one transformation per chunk when possible.",
        ])
    if "show me" in low or "solve it" in low:
        lines.append("- Show the actual worked solution, not just a hint.")
    if SHALLOW_GRE_STRUCTURE_RE.search(low):
        lines.extend([
            "- This is a factual GRE structure question. Stay aligned with the current official ETS GRE structure.",
            "- Do not mention outdated default GRE content like Analyze an Argument or an always-present unscored section.",
            "- If trusted source context is present, use it directly and mention the source briefly.",
        ])
    if focus and not (is_follow_up or is_generic_followup or is_ack_continue):
        lines.append(f"- If you choose an example, keep it relevant to {focus}.")
    upload_guidance = build_upload_response_guidance()
    if upload_guidance:
        lines.append(upload_guidance)
    return "\n".join(lines).strip()


def should_include_shallow_resume(user_text: str) -> bool:
    text = (user_text or "").strip()
    lowered = text.lower()
    if not text:
        return False
    if SHALLOW_ACK_CONTINUE_RE.fullmatch(text):
        return True
    if SHALLOW_CONTINUE_RE.search(lowered):
        return True
    if SHALLOW_GENERIC_FOLLOWUP_RE.search(lowered):
        return True
    if re.search(r"\b(this|that|it|same|again|continue|next|more|why|how so)\b", lowered):
        return True
    if len(text.split()) <= 4 and not is_tutor_request(text):
        return True
    return False


def shallow_mode_chunks(user_text: str) -> List[Tuple[str, str, List[dict], Dict[str, Any]]]:
    focus = _infer_shallow_focus(user_text)
    has_existing_context = bool(
        get_mode_memory(LearningMode.SHALLOW).board_history
        or get_mode_memory(LearningMode.SHALLOW).speech_history
        or get_visual_state(LearningMode.SHALLOW).live_board_text.strip()
    )
    broad_request = _is_shallow_broad_request(user_text, has_existing_context=has_existing_context)
    deterministic = _deterministic_shallow_chunks(user_text, focus)
    if deterministic:
        return deterministic
    resume = ""
    generic_followup = bool(SHALLOW_GENERIC_FOLLOWUP_RE.search((user_text or "").lower()))
    if should_include_shallow_resume(user_text):
        resume = build_resume_context(
            max_chars=900,
            board_chars=360,
            speech_chars=160,
            user_chars=140,
            include_uploads=True,
            mode=LearningMode.SHALLOW,
            query=user_text,
        )
    local_web_payload: Dict[str, Any] = retrieve_web_context(query=user_text, max_chars=1100, max_sources=3)
    if (not local_web_payload or not local_web_payload.get("sources")) and _should_auto_ground_shallow_query(user_text):
        try:
            auto_payload = build_trusted_web_context(user_text, max_sources=2)
            if auto_payload.get("sources"):
                remember_web_research_context(user_text, auto_payload)
                local_web_payload = auto_payload
        except Exception as exc:
            local_web_payload = {"query": user_text, "sources": [], "context": "", "error": str(exc)}
    if SHALLOW_GRE_STRUCTURE_RE.search(user_text):
        return _current_gre_structure_chunks(local_web_payload)
    emp_ctx = build_empathy_context()
    shallow_guidance = build_shallow_mode_guidance(user_text)
    web_guidance = build_web_response_guidance(user_text)
    local_web_context = str(local_web_payload.get("context") or "").strip()
    focus_line = f"CURRENT ACTIVE SUBJECT: {focus}\n" if focus else ""
    web_context_block = f"TRUSTED WEB CONTEXT:\n{local_web_context}\n" if local_web_context else ""
    prompt = (
        f"{emp_ctx}\n\n"
        f"SHARED MEMORY:\n{resume or 'No prior memory yet.'}\n\n"
        f"TURN-SPECIFIC GUIDANCE:\n{shallow_guidance or '- Answer directly and keep it concise.'}\n"
        f"{web_guidance + chr(10) if web_guidance else ''}\n"
        f"{focus_line}"
        f"{web_context_block}"
        f"ACTIVE MODE: Shallow free mode\n"
        f"STUDENT_MESSAGE: {user_text}\n"
        f"Answer in shallow mode."
    )
    out = timed_llm(
        SHALLOW_MODE_SYSTEM,
        prompt,
        label="Shallow mode",
        temperature=0.45,
        max_tokens=650,
        use_emotion_engine=True,
    )
    out = (out or "").strip()
    out = _strip_latexish_math(out)
    if _needs_focus_repair(user_text, focus, out):
        repair_prompt = (
            f"{prompt}\n\n"
            "CRITICAL REWRITE:\n"
            f"- The current active subject is {focus}.\n"
            "- The previous draft drifted to the wrong domain.\n"
            f"- Rewrite the answer only about {focus}.\n"
            "- Do not mention GRE, inference drills, news headlines, verbal reasoning, or quant sections.\n"
            "- End with one short follow-up question."
        )
        repaired = timed_llm(
            SHALLOW_MODE_SYSTEM,
            repair_prompt,
            label="Shallow mode repair",
            temperature=0.3,
            max_tokens=520,
            use_emotion_engine=True,
        )
        if repaired and repaired.strip():
            out = _strip_latexish_math(repaired.strip())
    source_tagline = _source_tagline_from_payload(local_web_payload) or current_web_source_tagline()
    if source_tagline and source_tagline.lower() not in out.lower():
        out = f"{out}\n\n<speech>{source_tagline}</speech>\n<board></board>\n<fx></fx>\n<meta>{{\"clear_before\": false}}</meta>\n{CHUNK_MARK}".strip()
    if not out:
        return [("Let's take it one step at a time.", "", [], {"clear_before": False})]

    if CHUNK_MARK not in out:
        fallback = _strip_latexish_math(_strip_structural_tags(out))
        intro = [_interactive_intro_chunk(user_text, focus)] if broad_request and "?" not in fallback else []
        follow = [_interactive_followup_chunk(focus)] if _focus_needs_followup_question(user_text, focus, broad_request) and "?" not in fallback else []
        return intro + [(fallback, "", [], {"clear_before": False})] + follow

    chunks_raw, _ = split_ready_chunks(out + "\n" + CHUNK_MARK)
    results: List[Tuple[str, str, List[dict], Dict[str, Any]]] = []
    for chunk in chunks_raw:
        sp, bd, fx, meta = parse_chunk(chunk)
        sp = _strip_latexish_math(sp)
        bd = _strip_latexish_math(bd)
        if not (sp or bd or fx):
            continue
        meta = meta if isinstance(meta, dict) else {}
        results.append((sp, bd, fx, {
            "clear_before": bool(meta.get("clear_before")),
        }))

    if results:
        joined_speech = " ".join(str(item[0] or "") for item in results[:2])
        if broad_request and "?" not in joined_speech:
            results.insert(0, _interactive_intro_chunk(user_text, focus))
        if _focus_needs_followup_question(user_text, focus, broad_request) and "?" not in joined_speech:
            results.append(_interactive_followup_chunk(focus))
        return results

    fallback = _strip_latexish_math(_strip_structural_tags(out))
    intro = [_interactive_intro_chunk(user_text, focus)] if broad_request and "?" not in fallback else []
    follow = [_interactive_followup_chunk(focus)] if _focus_needs_followup_question(user_text, focus, broad_request) and "?" not in fallback else []
    return intro + [(fallback, "", [], {"clear_before": False})] + follow


def llm_route(user_text: str) -> Dict[str, Any]:
    with state_lock:
        phase = S.phase
        mode  = S.mode
        last_topic = S.last_topic
        last_goal  = S.last_user_goal
        last_q     = S.last_question

    ctx = f"mode={mode}, phase={phase}, last_topic={last_topic}, last_goal={last_goal}, has_last_question={bool(last_q)}"
    emp_ctx = build_empathy_context()
    prompt = f"{emp_ctx}\n\nCONTEXT: {ctx}\nSTUDENT_MESSAGE: {user_text}\nReturn routing JSON."

    try:
        out = timed_llm(ROUTER_SYSTEM, prompt, label="Router", temperature=0.15, max_tokens=140)
        obj = safe_json_load(out)
    except Exception:
        obj = {}

    if not obj:
        intent = "tutor_request" if is_tutor_request(user_text) else "unknown"
        return {"intent": intent, "tone": "normal", "confidence": 0.0, "one_line_reply": ""}

    obj["intent"]       = str(obj.get("intent", "unknown")).strip().lower()
    obj["tone"]         = str(obj.get("tone", "normal")).strip().lower()
    obj["one_line_reply"] = str(obj.get("one_line_reply", "")).strip()
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    return obj


def agenda_json(user_text: str) -> Dict[str, Any]:
    subj = guess_subject_type(user_text)
    prompt = f'User wants structured tutoring on: "{user_text}"\nSubject type: {subj}\nCreate a practical tutoring agenda.'
    out = timed_llm(AGENDA_SYSTEM, prompt, label="Agenda plan", temperature=0.4, max_tokens=300)
    obj = safe_json_load(out)
    title  = (obj.get("title") or "").strip() or "Lesson"
    topics = obj.get("topics") or []
    topics = [str(x).strip() for x in topics if str(x).strip()]
    if len(topics) < 3:
        topics = ["Basics", "Examples", "Practice"]
    return {"title": title, "topics": topics[:7]}


def plan_subtopics(title: str, topic: str, subject_type: str,
                   all_topics: List[str] = None, student_pref: str = "") -> List[str]:
    other_topics = ""
    if all_topics:
        others = [t for t in all_topics if t.strip() != topic.strip()]
        if others:
            other_topics = "\nOTHER COURSE TOPICS (DO NOT use these as subtopics, they are taught separately): " + ", ".join(others)
    pref_line = f"\nSTUDENT_PREFERENCE: {student_pref}" if student_pref else ""
    prompt = (
        f"COURSE TITLE: {title}\n"
        f"MAIN TOPIC TO BREAK DOWN: {topic}\n"
        f"SUBJECT TYPE: {subject_type}"
        f"{other_topics}"
        f"{pref_line}\n"
        f"Plan 2â€“5 internal subtopics for ONLY the above main topic."
    )
    try:
        out = timed_llm(SUBTOPIC_PLAN_SYSTEM, prompt, label="Subtopic plan", temperature=0.3, max_tokens=200)
        obj = safe_json_load(out)
        subs = obj.get("subtopics") or []
        subs = [str(x).strip() for x in subs if str(x).strip()]
        if subs:
            return subs[:5]
    except Exception:
        pass
    return [topic]  # fallback


def detect_prior_knowledge(user_text: str, topics: List[str]) -> List[int]:
    """
    Detect which topic INDICES (0-based) the student claims to already know.
    Ordinal references like 'topic 1', 'topic 2', 'first topic', 'second topic'
    are resolved LOCALLY (no LLM) to avoid position-vs-content confusion.
    """
    if not topics:
        return []

    import re as _re

    txt_low = user_text.lower().strip()
    ordinal_indices: List[int] = []

    # â”€â”€ Heuristic: positional/ordinal references â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # "topic 1", "topic 2", "topics 1 and 2", "topics 1, 2, 3"
    for m in _re.finditer(r'\btopics?\s+(\d+)\b', txt_low):
        idx = int(m.group(1)) - 1        # 1-based â†’ 0-based
        if 0 <= idx < len(topics):
            ordinal_indices.append(idx)

    # Also catch bare numbers right after "know" or "covered" near a topic context
    # e.g. "I know 1 and 2", "I've done 1"
    if not ordinal_indices:
        if any(kw in txt_low for kw in ('know', 'covered', 'done', 'learned', 'familiar')):
            for m in _re.finditer(r'\b(\d+)\b', txt_low):
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(topics):
                    ordinal_indices.append(idx)

    # "first topic", "second topic", "the first one", "the first"
    _word_map = {
        'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
        'sixth': 5, 'seventh': 6, 'eighth': 7, 'ninth': 8, 'tenth': 9,
    }
    if not ordinal_indices:
        for word, idx in _word_map.items():
            if _re.search(rf'\b{word}\b', txt_low) and idx < len(topics):
                ordinal_indices.append(idx)

    if ordinal_indices:
        result = sorted(set(ordinal_indices))
        log_cli(f"ðŸ” Prior knowledge (ordinal match): indices={result} â†’ {[topics[i] for i in result]}")
        return result

    # â”€â”€ Fallback: LLM semantic matching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Include numbered list so LLM can see positional context
    numbered = {str(i): f"{i+1}. {t}" for i, t in enumerate(topics)}
    prompt = (
        f'STUDENT_STATEMENT: "{user_text}"\n'
        f'TOPICS_LIST (0-based index -> topic name):\n'
        + "\n".join([f'  index {i}: {t}' for i, t in enumerate(topics)])
        + '\n\nRemember: "topic 1" = index 0, "topic 2" = index 1, etc.'
    )
    try:
        out = global_llm.complete_once(PRIOR_KNOWLEDGE_SYSTEM, prompt, temperature=0.1, max_tokens=100)
        obj = safe_json_load(out)
        conf = float(obj.get("confidence", 0.0))
        if conf >= 0.6:
            raw = obj.get("known_indices") or []
            indices = [int(i) for i in raw if 0 <= int(i) < len(topics)]
            log_cli(f"ðŸ” Prior knowledge (LLM match, conf={conf:.2f}): indices={indices} â†’ {[topics[i] for i in indices]}")
            return indices
    except Exception:
        pass
    return []


def teach_subtopic_chunks(
    title: str, topic: str, subtopic: str, subject_type: str, user_text: str
) -> List[Tuple[str, str, List[dict], dict]]:
    resume = build_resume_context(mode=LearningMode.COURSE, query=user_text)
    prompt = f"""COURSE TITLE: {title}
SUBJECT TYPE: {subject_type}
MAIN TOPIC: {topic}
SUBTOPIC TO TEACH NOW: {subtopic}

CONTEXT (do NOT repeat already-covered content):
{resume}

Student said: "{user_text}"

Teach this subtopic COMPLETELY and DEEPLY. Cover all its aspects with examples.
Use short CHUNKs for sync. Board -> Speech must be perfectly synced."""

    emp_ctx = build_empathy_context()
    prompt  = emp_ctx + "\n\n" + prompt
    out = timed_llm(
        STEP_SYSTEM,
        prompt,
        label="Teach subtopic",
        temperature=0.55,
        max_tokens=1800,
        use_emotion_engine=True,
    )
    out = out + "\n" + CHUNK_MARK
    chunks_raw, _ = split_ready_chunks(out)
    results = []
    for c in chunks_raw:
        sp, bd, fx, meta = parse_chunk(c)
        results.append((sp, bd, fx, meta))
    return results


def check_confidence_ready(
    title: str, topic: str, subtopics_done: List[str], taught_points: List[str],
    user_interactions: List[str]
) -> Dict[str, Any]:
    prompt = f"""COURSE: {title}
TOPIC: {topic}
SUBTOPICS TAUGHT: {', '.join(subtopics_done)}
TAUGHT_POINTS: {', '.join(taught_points[:20])}
RECENT STUDENT INTERACTIONS: {' | '.join(user_interactions[-5:])}"""
    try:
        out = timed_llm(CONFIDENCE_CHECK_SYSTEM, prompt, label="Confidence check", temperature=0.2, max_tokens=120)
        obj = safe_json_load(out)
        return {
            "ready":    bool(obj.get("ready", True)),
            "reason":   str(obj.get("reason", "")),
            "qa_count": min(5, max(3, int(obj.get("qa_count", 3)))),
        }
    except Exception:
        return {"ready": True, "reason": "", "qa_count": 3}


def generate_qa_questions(
    title: str, topic: str, taught_points: List[str], count: int
) -> List[Dict]:
    prompt = f"""COURSE: {title}
TOPIC: {topic}
TAUGHT_POINTS: {json.dumps(taught_points[:25])}"""
    sys_prompt = QA_GEN_SYSTEM.replace("{count}", str(count))
    try:
        out = global_llm.complete_once(sys_prompt, prompt, temperature=0.4, max_tokens=800)
        obj = safe_json_load(out)
        qs = obj.get("questions") or []
        valid = []
        for q in qs:
            if not q.get("question"):
                continue
            q["type"]    = q.get("type", "simple")
            q["answer"]  = str(q.get("answer", ""))
            q["concept"] = str(q.get("concept", ""))
            if q["type"] == "mcq" and not q.get("options"):
                q["type"] = "simple"
            valid.append(q)
        return valid[:count]
    except Exception:
        return []


def evaluate_qa_answer(
    question: Dict, student_answer: str, taught_points: List[str]
) -> Dict[str, Any]:
    prompt = f"""QUESTION: {question.get('question', '')}
TYPE: {question.get('type', 'simple')}
OPTIONS: {json.dumps(question.get('options', {}))}
CORRECT_ANSWER: {question.get('answer', '')}
CONCEPT: {question.get('concept', '')}
TAUGHT_POINTS: {', '.join(taught_points[:15])}
STUDENT_ANSWER: {student_answer}"""
    try:
        out = timed_llm(QA_EVAL_SYSTEM, prompt, label="QA eval", temperature=0.15, max_tokens=120)
        obj = safe_json_load(out)
        return {
            "correct":  bool(obj.get("correct", False)),
            "feedback": str(obj.get("feedback", "Interesting. Let's continue.")),
            "partial":  bool(obj.get("partial", False)),
        }
    except Exception:
        return {"correct": False, "feedback": "Let me check that. Moving on.", "partial": False}


def generate_similar_question(wrong_question: Dict, taught_points: List[str]) -> Optional[Dict]:
    prompt = f"""WRONG_QUESTION: {json.dumps(wrong_question)}
TAUGHT_POINTS: {', '.join(taught_points[:15])}
Generate a similar question testing the same concept with different wording."""
    try:
        out = global_llm.complete_once(QA_SIMILAR_SYSTEM, prompt, temperature=0.45, max_tokens=250)
        obj = safe_json_load(out)
        if obj.get("question"):
            obj["type"]   = obj.get("type", "simple")
            obj["answer"] = str(obj.get("answer", ""))
            obj["concept"] = wrong_question.get("concept", "")
            return obj
    except Exception:
        pass
    return None


def reteach_concept_chunks(
    title: str, topic: str, concept: str, taught_points: List[str]
) -> List[Tuple[str, str, List[dict], dict]]:
    resume = build_resume_context(mode=LearningMode.COURSE, query=concept)
    prompt = f"""COURSE: {title}
TOPIC: {topic}
CONCEPT TO RE-EXPLAIN: {concept}
TAUGHT_POINTS (for context): {', '.join(taught_points[:15])}
CONTEXT: {resume}

Student got this wrong. Re-explain with a fresh angle, new example, or analogy.
Keep it brief: 1â€“3 CHUNKs max."""
    out = timed_llm(
        RETEACH_SYSTEM,
        prompt,
        label="Reteach",
        temperature=0.5,
        max_tokens=800,
        use_emotion_engine=True,
    )
    out = out + "\n" + CHUNK_MARK
    chunks_raw, _ = split_ready_chunks(out)
    results = []
    for c in chunks_raw:
        sp, bd, fx, meta = parse_chunk(c)
        results.append((sp, bd, fx, meta))
    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Board filtering
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BOARD_BAD_RE = re.compile(r"\b(hello|hi|welcome back|how are you|audible)\b", re.I)
BOARD_STATUS_LINE_RE = re.compile(r"^\s*(Active:\s*.+|Mode:\s*(?:Shallow|Course).*)\s*$", re.I | re.M)


def _strip_board_status_lines(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    cleaned_lines = [
        line for line in raw.splitlines()
        if not BOARD_STATUS_LINE_RE.match(line.strip())
    ]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_board_output(board_text: str) -> str:
    b = _strip_board_status_lines(normalize_board_text(postprocess_board_text(board_text or "")))
    if not b.strip():
        return ""
    if BOARD_BAD_RE.search(b):
        return ""
    return b


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FX sync: anchored to audio start
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def emit_fx_synced_from_audio_start(fx_actions: List[dict]):
    for a in fx_actions or []:
        delay    = int(a.get("delay_ms", 0))
        fx_type  = a.get("type", "fx")
        fx_tgt   = (a.get("target") or "")[:60]

        def _emit_one(action=a, t=fx_type, tgt=fx_tgt):
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            icon = "âœ¨" if t == "glow" else "ðŸ’¥"
            _file_logger.write(f"[{ts}] FX {t.upper()}: [{tgt}]")
            socketio.emit("fx_timing_log", {"type": t, "target": tgt, "ts": ts})
            emit_fx([action])

        if delay <= 0:
            _emit_one()
        else:
            threading.Timer(delay / 1000.0, _emit_one).start()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… speak_chunks â€” parallel TTS pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _micro_split_speech(speech: str, board: str, fx: list) -> List[Tuple[str, str, list]]:
    """
    Split a long speech string into sentence-level micro-chunks.
    Board text only goes on the FIRST micro-chunk (sync point).
    FX only goes on the first micro-chunk.
    Chunks â‰¤ MAX_CHARS are kept as-is.
    """
    MAX_CHARS = 160  # comfortable TTS chunk size â€” ~5-8 seconds
    speech = (speech or "").strip()
    if not speech or len(speech) <= MAX_CHARS:
        return [(speech, board, fx)]

    # Split at sentence boundaries preserving punctuation
    import re as _re
    raw_parts = _re.split(r'(?<=[.!?])\s+', speech)

    # Merge very short parts (< 40 chars) with the next one
    merged: List[str] = []
    buf = ""
    for part in raw_parts:
        if buf and len(buf) + 1 + len(part) <= MAX_CHARS:
            buf = buf + " " + part
        else:
            if buf:
                merged.append(buf.strip())
            buf = part
    if buf.strip():
        merged.append(buf.strip())

    if not merged:
        return [(speech, board, fx)]

    result = []
    for k, txt in enumerate(merged):
        b = board if k == 0 else ""   # board only on first chunk
        f = fx    if k == 0 else []   # fx only on first chunk
        result.append((txt, b, f))
    return result


def speak_chunks(chunks: List[Tuple[str, str, List[dict]]], instruct: str = None):
    """
    CLEAN PIPELINE ARCHITECTURE:
      1. Micro-split all chunks into â‰¤160-char sentence pieces
      2. ONE TTS worker: while chunk[N] plays, chunk[N+1] is already being fetched
      3. Playback loop: board emit (fire-and-forget) â†’ audio play in strict order
      4. Sentinel None in audio_q signals completion â€” zero race conditions
    """
    if not chunks:
        return

    # â”€â”€ Step 1: Flatten + micro-split into ordered list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_items: List[Tuple[int, str, str, list, bool, bool]] = []
    idx = 0
    for item in chunks:
        if len(item) == 4:
            sp, bd, fx, meta = item
        else:
            sp, bd, fx = item[0], item[1], item[2] if len(item) > 2 else []
            meta = {}
        clear_before = bool(meta.get("clear_before")) if isinstance(meta, dict) else False
        micros = _micro_split_speech(sp, bd, fx)
        for k, (msp, mbd, mfx) in enumerate(micros):
            # _log_first: only the first micro-chunk logs the speech text
            all_items.append((idx, msp, mbd, mfx, k == 0, clear_before and k == 0))
            idx += 1

    if not all_items:
        return

    total = len(all_items)

    # â”€â”€ Step 2: Audio queue â€” ONE TTS worker, prefetches N+1 while N plays â”€â”€
    audio_q: queue.Queue = queue.Queue(maxsize=50)

    def _tts_worker():
        for (i, sp, bd, fx, log_first, clear_before) in all_items:
            if interrupt_event.is_set():
                break
            wav = b""
            if sp:
                # speech logged at _on_start (playback time) â€” not here (gen time)
                clean = sanitize_for_speech(sp)
                with _voice_lock:
                    active_engine  = VOICE_ENGINE
                    active_speaker = VOICE_SPEAKER
                if active_engine == "humanised":
                    active_engine, active_speaker = _auto_fallback_voice(
                        "humanised_tts_unreachable",
                        emit=True,
                    )
                _tts_start = time.time()
                emit_timing("tts_start", chunk=i, preview=clean[:60])
                if active_engine == "bot":
                    try:
                        wav = bot_tts_to_wav_bytes(active_speaker, clean)
                    except Exception as e:
                        log_cli(f"❌ Bot TTS Error: {e}")
                elif active_engine == "piper":
                    try:
                        wav = piper_tts_to_wav_bytes(active_speaker, clean)
                    except Exception as e:
                        log_cli(f"❌ Piper TTS Error: {e}")
                else:
                    api_speaker = active_speaker.lower().replace(" ", "_")
                    for attempt in range(1):
                        if interrupt_event.is_set():
                            break
                        try:
                            _instruct = instruct if instruct else DEFAULT_INSTRUCT
                            wav, _ = tts_request_bytes(
                                clean, speaker=api_speaker, instruct=_instruct
                            )
                            break
                        except Exception as e:
                            log_cli(f"❌ TTS attempt {attempt+1}: {e}")
                            time.sleep(0.35 * (attempt + 1))
                    if not wav and not interrupt_event.is_set():
                        try:
                            active_engine, active_speaker = _auto_fallback_voice(
                                "humanised_tts_request_failed",
                                emit=True,
                            )
                            wav = piper_tts_to_wav_bytes(active_speaker, clean)
                            active_engine = "piper"
                        except Exception as e:
                            log_cli(f"❌ Piper fallback TTS Error: {e}")
                _tts_dur_ms = round((time.time() - _tts_start) * 1000)
                emit_timing("tts_gen_done", chunk=i, duration_ms=_tts_dur_ms,
                            preview=clean[:60], ok=bool(wav))
            audio_q.put((i, sp, bd, fx, wav, active_engine if sp else "", log_first, clear_before))
        audio_q.put(None)  # sentinel â€” signals completion

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    # â”€â”€ Step 3: Playback in order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    set_status("SPEAKING")
    if cam_monitor:
        cam_monitor.tutor_speaking = True

    while not interrupt_event.is_set():
        # Wait for next audio item; poll so interrupt_event is checked regularly
        item = None
        while not interrupt_event.is_set():
            try:
                item = audio_q.get(timeout=0.5)
                break
            except queue.Empty:
                # If TTS worker finished and queue empty â†’ done
                if not tts_thread.is_alive() and audio_q.empty():
                    break
        if item is None:
            break  # interrupted or truly done

        if item is None:
            break  # sentinel: all chunks done

        i, sp, bd, fx, wav, eng, log_first, clear_before = item
        _gain = 1.8 if eng == "humanised" else 1.0

        if clear_before:
            clear_board()

        # Dedup board text + extract only NEW delta (LLM often outputs cumulative board)
        bd = sanitize_board_output(bd)
        if bd:
            bd = extract_board_delta(bd)   # strip already-shown content
        if bd:
            # Fix 2: line-level dedup (catches partial repeats hash misses)
            bd = _filter_duplicate_lines(bd)
        if bd:
            # Fix 3: hold lone headings until body arrives with them
            bd = _apply_heading_buffer(bd)
        if bd:
            h = _board_hash(bd)
            state = get_visual_state()
            if h in state.last_board_hashes:
                bd = ""
            else:
                state.last_board_hashes.append(h)
                remember_board(bd)
                session_logger.log_board(bd)

        if wav and not interrupt_event.is_set():
            _speak_wall = [0.0]  # mutable container so closure can write it

            # Board fires the instant audio output begins (true sync)
            def _on_start(_b=bd, _f=fx, _i=i, _sp=sp, _lf=log_first):
                _speak_wall[0] = time.time()
                # Log speech only on the FIRST micro-chunk (prevents N-repeat in log)
                if _sp and _lf:
                    remember_speech(_sp)
                    log_analytics_tutor_speech(_sp, instruct)
                    if instruct:
                        log_analytics_empathy_event("tutor_tone", str(instruct)[:120])
                    log_cli(f"ðŸ—£ï¸ {_sp}")
                emit_timing("speak_start", chunk=_i)
                if _b:
                    # â”€â”€ Board timing event â€” visible in sidebar log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    emit_timing("board_text", chunk=_i,
                                preview=_b[:80].replace("\n", " "),
                                chars=len(_b))
                    # Fix 1: "\n" prefix ensures no bleed from previous board content
                    socketio.emit("board_text", {
                        "text": "\n" + _b + "\n\n",
                        "append": True,
                        "mode": "type",
                        "cps": 38,
                        "scroll": True,
                    })
                if _f:
                    emit_fx_synced_from_audio_start(_f)

            play_wav_bytes_interruptible(wav, interrupt_event, on_audio_start=_on_start, volume_gain=_gain)
            _speak_dur = round((time.time() - _speak_wall[0]) * 1000) if _speak_wall[0] else 0
            emit_timing("speak_end", chunk=i, duration_ms=_speak_dur)

        elif bd and not wav:
            # No audio â€” show board with small pause so student can read
            # Fix 1: "\n" prefix here too
            socketio.emit("board_text", {
                "text": "\n" + bd + "\n\n",
                "append": True,
                "mode": "type",
                "cps": 38,
                "scroll": True,
            })
            if fx:
                emit_fx(fx)
            time.sleep(0.15)

    # Fix 5: if loop exited because of interrupt (not natural completion), mark it
    if interrupt_event.is_set():
        with state_lock:
            if S.phase == Phase.TEACH_SUBTOPIC and not S.subtopic_interrupted:
                S.subtopic_interrupted = True
                log_cli(f"âš¡ Subtopic interrupted mid-teaching: {S.subtopics[S.subtopic_idx] if S.subtopics and S.subtopic_idx < len(S.subtopics) else 'unknown'}")

    if cam_monitor:
        cam_monitor.tutor_speaking = False




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Q&A DISPLAY HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def restore_visual_board(mode: Optional[Any] = None, force_clear_if_empty: bool = False) -> bool:
    target = normalize_learning_mode(mode) if mode is not None else get_active_learning_mode()
    state = get_visual_state(target)
    payload = dict(state.render_payload or {})

    if state.render_kind == "qa_result":
        question = payload.get("question") or {}
        result = payload.get("result") or {}
        if question:
            socketio.emit("qa_board", question)
        if result:
            socketio.emit("qa_board_result", result)
        return bool(question or result)

    if state.render_kind == "qa_board" and payload:
        socketio.emit("qa_board", payload)
        return True

    if state.render_kind == "qa_end" and payload:
        socketio.emit("qa_end", payload)
        return True

    if state.live_board_text:
        socketio.emit("board_text", {
            "text": sanitize_board_output(state.live_board_text),
            "append": False,
            "mode": "instant",
            "cps": 0,
            "scroll": True,
        })
        return True

    if force_clear_if_empty:
        socketio.emit("clear_board")
        return True

    return False


def emit_learning_mode_state():
    socketio.emit("learning_mode_state", {"mode": get_active_learning_mode().value})


def format_qa_for_board(idx: int, q: Dict) -> str:
    lines = [f"**Q{idx + 1}. {q['question']}**"]
    if q.get("type") == "mcq" and q.get("options"):
        for letter, text in q["options"].items():
            lines.append(f"  {letter}) {text}")
    return "\n".join(lines)


def emit_qa_card(idx: int, q: Dict, total: int):
    """Emit structured MCQ/simple question to frontend for GfG-style rendering."""
    socketio.emit("qa_question", {
        "idx": idx,
        "total": total,
        "type": q.get("type", "simple"),
        "question": q.get("question", ""),
        "options": q.get("options", {}),
        "concept": q.get("concept", ""),
    })


def emit_qa_result(idx: int, correct: bool, correct_answer: str, feedback: str):
    """Show GfG-style correct/wrong result on frontend."""
    socketio.emit("qa_result", {
        "idx": idx,
        "correct": correct,
        "correct_answer": correct_answer,
        "feedback": feedback,
    })


def emit_qa_board(idx: int, q: Dict, total: int):
    """Render MCQ question directly INSIDE the blackboard (replaces board content)."""
    payload = {
        "idx": idx,
        "total": total,
        "type": q.get("type", "simple"),
        "question": q.get("question", ""),
        "options": q.get("options", {}),
        "concept": q.get("concept", ""),
    }
    set_visual_render_payload("qa_board", payload)
    socketio.emit("qa_board", payload)


def emit_qa_board_result(correct: bool, correct_answer: str, feedback: str, selected_letter: str = ""):
    """Update the in-board MCQ to show correct/wrong highlights."""
    result_payload = {
        "correct": correct,
        "correct_answer": correct_answer,
        "feedback": feedback,
        "selected_letter": selected_letter,
    }
    current_payload = get_visual_state().render_payload if get_visual_state().render_kind == "qa_board" else {}
    set_visual_render_payload("qa_result", {
        "question": dict(current_payload or {}),
        "result": result_payload,
    })
    socketio.emit("qa_board_result", result_payload)


def emit_qa_end(result_log: list = None, topic: str = "", correct_count: int = 0, total: int = 0):
    """Tell frontend QA session is over â€” resume normal board rendering + show summary."""
    summary_items = []
    if result_log:
        for i, r in enumerate(result_log):
            summary_items.append({
                "idx": i,
                "question": r.get("question", ""),
                "correct": r.get("correct", False),
                "answer": r.get("answer", ""),
                "student": r.get("student", ""),
            })
    payload = {
        "topic": topic,
        "correct": correct_count,
        "total": total,
        "items": summary_items,
    }
    set_visual_render_payload("qa_end", payload)
    socketio.emit("qa_end", payload)


def format_qa_for_speech(q: Dict) -> str:
    parts = [q["question"]]
    if q.get("type") == "mcq" and q.get("options"):
        parts.append("Your options are:")
        for letter, text in q["options"].items():
            # Period after letter and after option text gives Piper a natural pause
            parts.append(f"{letter}. {text}.")
    return " ".join(parts)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… WORKER STATE MACHINE â€” v3 DEEP TUTOR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def llm_worker(stt_instance):
    global tutor_busy, S
    log_cli("âœ… DEEP TUTOR ENGINE v3.0 STARTED")

    side_stack = []
    SIDE_RE = re.compile(r"\b(what is|define|what's)\b.{0,40}\b(the term|this word|meaning|difference)\b", re.I)

    while True:
        turn_mode_token = None
        turn_web_token = None
        try:
            queued_input = text_q.get()
            if queued_input is None:
                continue
            camera_event = ""
            camera_attention = ""
            camera_details: Dict[str, Any] = {}
            camera_instruct = ""
            if isinstance(queued_input, dict):
                user_txt = str(queued_input.get("text") or "").strip()
                input_source = str(queued_input.get("source") or "typed")
                wants_web_search = bool(queued_input.get("web_search", False))
                camera_event = str(queued_input.get("camera_event") or "").strip()
                camera_attention = str(queued_input.get("camera_attention") or "").strip()
                camera_details = dict(queued_input.get("camera_details") or {})
                camera_instruct = str(queued_input.get("camera_instruct") or "").strip()
            else:
                user_txt = str(queued_input).strip()
                input_source = "legacy"
                wants_web_search = False
            if not user_txt:
                continue
            turn_learning_mode = get_active_learning_mode()
            turn_mode_token = TURN_LEARNING_MODE.set(turn_learning_mode)

            log_cli(f"ðŸ§µ Worker picked queued input ({len(str(user_txt))} chars)")
            interrupt_event.clear()
            tutor_busy = True
            log_cli("ðŸ§µ Worker pausing STT")
            stt_instance.pause()
            log_cli("ðŸ§µ Worker paused STT")

            try:
                while True:
                    text_q.get_nowait()
            except Exception:
                pass

            if input_source == "camera_monitor":
                log_cli(f"[Monitor] {camera_event or 'notice'}: {camera_attention or 'unknown'}")
                emit_timing("stt_received", text=(camera_attention or "camera")[:80])
                set_status("ANALYZING")
                line = user_txt or "Please come back to the lesson."
                instruct = camera_instruct or (EMPATHY_INSTRUCT_WELCOME if camera_event == "return" else EMPATHY_INSTRUCT_BREAK)
                if camera_event == "alert" and camera_attention in {"phone", "sleepy"}:
                    interrupt_event.set()
                speak_chunks([(line, "", [], {"clear_before": False})], instruct=instruct)
                if camera_event == "alert":
                    log_analytics_empathy_event("camera_alert", camera_attention or line[:80])
                elif camera_event == "return":
                    log_analytics_empathy_event("camera_return", camera_attention or line[:80])
                continue

            remember_user(user_txt)
            show_user_speech(user_txt)
            log_cli(f"ðŸ§  Analyzing... '{user_txt}'")
            emit_timing("stt_received", text=user_txt[:80])
            set_status("ANALYZING")
            if wants_web_search:
                try:
                    emit_timing("llm_start", label="Web search")
                    web_payload = build_trusted_web_context(user_txt, max_sources=3)
                    if web_payload.get("sources"):
                        remember_web_research_context(user_txt, web_payload)
                        turn_web_token = TURN_WEB_CONTEXT.set(web_payload)
                        source_domains = [
                            str(item.get("domain") or "").strip()
                            for item in (web_payload.get("sources") or [])
                            if item.get("domain")
                        ]
                        source_domains = [
                            domain for idx, domain in enumerate(source_domains)
                            if domain and domain not in source_domains[:idx]
                        ]
                        log_cli(f"Web search ready from: {', '.join(source_domains[:3])}")
                    else:
                        log_cli(f"Web search found no usable trusted sources: {web_payload.get('error') or 'no_results'}")
                        turn_web_token = TURN_WEB_CONTEXT.set(dict(web_payload))
                    emit_timing(
                        "llm_done",
                        duration_ms=0,
                        label="Web search",
                        tokens=len(str(web_payload.get("context") or "").split()),
                    )
                except Exception as exc:
                    log_cli(f"Web search skipped: {exc}")
                    turn_web_token = TURN_WEB_CONTEXT.set({
                        "query": user_txt,
                        "sources": [],
                        "context": "",
                        "error": str(exc),
                    })
            # â”€â”€ Session analytics: track turn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with state_lock:
                _cur_topic_for_log = S.last_topic or ""
            log_analytics_turn(user_txt, "pending", 0.0, _cur_topic_for_log)
            # â”€â”€ Architecture layer: Emotion Input active â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            emit_layer_update("emotion_input", "active",
                input_text=user_txt[:120],
                meta={"source": input_source, "web_search": wants_web_search})

            # â”€â”€ Extract student setup/tool context (universal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            import re as _rctx
            _SETUP_VER_RE = _rctx.compile(
                r"\b(?:version\s*|v)(\d+\.\d+(?:\.\d+)?)\b|"
                r"\b([\w+#]{2,15})\s+(?:version\s*)?(\d+\.\d+)\b", _rctx.I)
            _SETUP_TOOL_RE = _rctx.compile(
                r"\b(?:already\s+)?(?:installed|set\s*up|configured|using|have|got)\s+"
                r"([\w\s+#]{2,25}?)(?:\s+(?:version|v|already|\d)|\s*[,.]|\s*$)", _rctx.I)
            # Detect terminal/command-line output pasted by student (e.g. "Python 3.10.9")
            _TERMINAL_OUT_RE = _rctx.compile(
                r"(?:^|\n)\s*[A-Za-z]:\\.*?>|"  # Windows prompt like C:\Users\...>
                r"Python\s+(\d+\.\d+\.\d+)|"  # bare "Python 3.10.9"
                r"\$\s*python", _rctx.I | _rctx.M)
            _setup_notes = []
            _is_terminal_output = bool(_TERMINAL_OUT_RE.search(user_txt))
            for _m in _SETUP_VER_RE.finditer(user_txt):
                _setup_notes.append(user_txt[_m.start():_m.end()].strip())
            for _m in _SETUP_TOOL_RE.finditer(user_txt):
                _h = _m.group(1).strip()
                if _h and len(_h) > 1 and _h.lower() not in ("it","the","a","an","my","so","we","can","i"):
                    _setup_notes.append(f"has {_h}")
            _ctx_new_note = None
            if _setup_notes:
                _note = "; ".join(_setup_notes[:3])
                global shared_student_context
                with state_lock:
                    if _note not in ((shared_student_context or "") + " " + (S.student_context or "")):
                        shared_student_context = (_note + ". " + shared_student_context).strip(". ")
                        S.student_context = shared_student_context
                        _ctx_new_note = _note
                        schedule_runtime_state_save()
                        log_cli(f"ðŸ“Œ Student context: {S.student_context}")
                # Interactive version acknowledgment: student pasted terminal output or mentioned version
                if _ctx_new_note and _is_terminal_output:
                    _ver_match = _rctx.search(r"Python\s+(\d+\.\d+\.\d+)", user_txt, _rctx.I)
                    if _ver_match:
                        _ver = _ver_match.group(1)
                        try:
                            _ack_ctx = (
                                f"Student just showed their Python version: {_ver}. "
                                f"Course: {S.title or 'Python'}. Current topic: {S.last_topic or 'setup'}. "
                                f"Respond warmly in 1-2 sentences: confirm the version is good, "
                                f"maybe one practical note about it, then say ready to continue. "
                                f"Be conversational, not robotic."
                            )
                            _ack_msg = global_llm.complete_once(
                                "You are a warm tutor.", _ack_ctx, temperature=0.8, max_tokens=60
                            ).strip()
                        except Exception:
                            _ack_msg = f"Python {_ver} â€” perfect! That's a solid version for everything we'll do. Ready to continue!"
                        speak_chunks([(_ack_msg, f"âœ… Python {_ver} confirmed", [])])
                        log_cli(f"ðŸ Version ack: {_ack_msg}")
                        continue

            with state_lock:
                phase_before_route = S.phase

            qa_like_answer = (
                phase_before_route in (Phase.QA, Phase.QA_REVIEW)
                and _looks_like_qa_answer(user_txt)
            )

            # â”€â”€ PRE-ROUTER: catch obvious session end/pause before LLM â”€â”€
            _pre_intent = None
            if not qa_like_answer and SESSION_END_RE.search(user_txt):
                _pre_intent = "session_end"
            elif not qa_like_answer and SESSION_PAUSE_RE.search(user_txt):
                _pre_intent = "session_pause"

            log_cli("[Route] resolving intent")
            if qa_like_answer:
                route = {"intent": "answer_qa", "tone": "normal", "confidence": 0.99}
            else:
                route = llm_route(user_txt)
            intent = route.get("intent", "unknown")

            # Pre-router heuristic overrides LLM if confident
            if _pre_intent:
                intent = _pre_intent
                log_cli(f"âš¡ Pre-router override: intent={intent}")
            upload_turn = (turn_learning_mode == LearningMode.COURSE and _should_use_upload_context(user_txt))
            if upload_turn and intent not in ("session_pause", "session_end"):
                intent = "ask_question"
            tone   = route.get("tone", "normal")

            with state_lock:
                phase    = S.phase
                mode     = S.mode
                cur_topic = S.last_topic
                # "in_tutor" excludes WAIT_START and CONFIDENCE_CHECK:
                # student is answering questions there, not requesting new topics
                _safe_switch_phases = (Phase.IDLE, Phase.AGENDA, Phase.WAIT_START,
                                       Phase.CONFIDENCE_CHECK, Phase.QA, Phase.QA_REVIEW)
                in_tutor = (mode == Mode.TUTOR and phase not in _safe_switch_phases)

            if emotion_engine is not None:
                try:
                    log_cli("[Emotion] processing user turn")
                    emotion_engine.handle_user_turn(
                        user_txt,
                        intent=intent,
                        phase=getattr(phase, "name", str(phase)),
                        timestamp=time.time(),
                    )
                    log_cli("[Emotion] user turn processed")
                    # â”€â”€ Layer: emotion_input done â†’ state_tracker active â”€â”€
                    emit_layer_update("emotion_input", "done",
                        input_text=user_txt[:120],
                        output_text=f"intent={intent}",
                        meta={"intent": intent, "phase": getattr(phase, "name", "")})
                    emit_layer_update("state_tracker", "active",
                        input_text=f"intent={intent} | phase={getattr(phase,'name','')}",
                        meta={"intent": intent})
                    emit_layer_update("empathy_policy", "active",
                        input_text="evaluating emotional state",
                        meta={"intent": intent})
                except Exception as exc:
                    log_cli(f"Emotion engine turn skipped: {exc}")
                    emit_layer_update("emotion_input", "error", output_text=str(exc)[:80])
            else:
                # Engine disabled â€” mark layers done passively
                emit_layer_update("emotion_input", "done",
                    input_text=user_txt[:120], output_text=f"intent={intent}")
                emit_layer_update("state_tracker", "done",
                    input_text="engine disabled", output_text="passthrough")
                emit_layer_update("empathy_policy", "done",
                    input_text="engine disabled", output_text="no conditioning")
                emit_layer_update("pedagogy", "active",
                    input_text=f"intent={intent}", output_text="selecting strategy...")

            # â”€â”€ MID-SESSION TOPIC SWITCH (highest priority inside tutor) â”€â”€
            # STRICT: requires BOTH LLM intent AND heuristic keyword AND
            #         message must be â‰¥ 4 words AND not about current topic
            _words = user_txt.strip().split()
            _has_topic_keyword = is_tutor_request(user_txt)
            _intent_says_new   = (intent == "tutor_request")
            _long_enough       = len(_words) >= 4
            _mentions_cur      = cur_topic and any(
                w.lower() in user_txt.lower() for w in cur_topic.split()[:3]
            )
            _is_real_switch    = (in_tutor and _has_topic_keyword and _intent_says_new
                                  and _long_enough and not _mentions_cur)
            if _is_real_switch:
                # Check if already confirming a switch â€” if yes, this is their confirmation
                with state_lock:
                    _scm = S.switch_confirm_mode
                    _scg = S.switch_confirm_goal

                if _scm:
                    # Student confirmed the switch
                    _confirm_words = ("yes","yeah","yep","sure","okay","ok","go ahead",
                                      "do it","switch","let's","lets","please","proceed")
                    _deny_words    = ("no","nope","never mind","nevermind","stay","continue",
                                      "keep","back","python","same","cancel","forget it")
                    _low = user_txt.lower()
                    _deny  = any(w in _low for w in _deny_words)
                    _confirm = (not _deny) and (any(w in _low for w in _confirm_words) or intent == "confirm_start")

                    if _deny or ("no" in _low[:6]):
                        with state_lock:
                            S.switch_confirm_mode = False
                            S.switch_confirm_goal = ""
                        speak_chunks([("No problem â€” let's keep going with our current topic!", "", [])])
                        log_cli("â†©ï¸  Topic switch cancelled â€” continuing current lesson")
                        continue
                    elif _confirm:
                        with state_lock:
                            _saved_ctx = S.student_context
                        handle_end_session({"forced": False})
                        reset_session_runtime(get_active_learning_mode())
                        with state_lock:
                            S = TutorState(mode=Mode.TUTOR, phase=Phase.AGENDA)
                            S.last_user_goal = _scg
                            S.student_context = _saved_ctx
                        log_cli(f"Topic switch confirmed -> new session for {_scg}")
                    else:
                        # Ambiguous â€” ask again once
                        with state_lock:
                            S.switch_confirm_mode = False
                            S.switch_confirm_goal = ""
                        speak_chunks([("I wasn't sure what you meant â€” let's just continue with what we were doing.", "", [])])
                        continue
                else:
                    # First detection â€” ask user to confirm before switching
                    with state_lock:
                        S.switch_confirm_mode = True
                        S.switch_confirm_goal = user_txt
                        _cur = S.last_topic or "the current topic"
                        _title = S.title or "Python"
                    try:
                        _ask_switch = global_llm.complete_once(
                            "You are a warm tutor. Write ONE natural question (max 15 words).",
                            f"Student is learning '{_title}' (currently on '{_cur}'). "
                            f"They suddenly asked: '{user_txt}'. "
                            f"Ask if they want to start a fresh new session for that new subject, because the current lesson is already running. "
                            f"Be warm, slightly playful.",
                            temperature=0.8, max_tokens=50
                        ).strip()
                    except Exception:
                        _ask_switch = f"We're in {_title} {get_active_learning_mode().value} mode right now. Do you want a new session for {user_txt}?"
                    speak_chunks([(_ask_switch, "", [])])
                    log_cli(f"â“ Switch confirm question: {_ask_switch}")
                    continue  # wait for their answer

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # âœ… GLOBAL: session_pause / session_end
            # Highest priority â€” fires before ANY phase handler
            # LLM generates natural farewell with full empathy context
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if intent in ("session_pause", "session_end"):
                with emp_lock:
                    name        = EMP.student_name
                    study_secs  = EMP.total_study_secs
                    tz          = EMP.real_tz
                    weather     = EMP.weather_label
                    temp        = EMP.weather_temp

                # Current local time
                try:
                    import zoneinfo
                    local_now = datetime.datetime.now(zoneinfo.ZoneInfo(tz))
                    time_str  = local_now.strftime("%I:%M %p")
                    hour      = local_now.hour
                except Exception:
                    local_now = datetime.datetime.now()
                    time_str  = local_now.strftime("%I:%M %p")
                    hour      = local_now.hour

                study_min = int(study_secs // 60)
                is_late   = (hour >= 22 or hour < 4)

                with state_lock:
                    topic    = S.last_topic or S.title
                    covered  = list(S.subtopics_done)
                    remaining= [s for s in S.subtopics if s not in S.subtopics_done]

                _pause_or_end = "pause" if intent == "session_pause" else "end"

                # Build rich context for LLM â€” let it think naturally
                _late_tag    = "(very late night)" if is_late else ""
                _covered_str = ", ".join(covered) if covered else "just started"
                _remain_str  = ", ".join(remaining) if remaining else "none"
                _mood_type   = "farewell" if _pause_or_end == "end" else "pause acknowledgement"
                _time_note   = "Wish them goodnight warmly, mention the time if relevant, acknowledge their effort." if is_late else "Acknowledge their tiredness/need to pause with understanding."
                _resume_note = "Mention what topic they were on and that you will continue from there when they return." if _pause_or_end == "pause" else "Be warm and encouraging about their progress."
                _name_note   = "and mention it is late" if is_late else ""
                _study_line = (
                    f"- Study time today: {study_min} minutes\n"
                    if study_min > 0 else
                    "- Study time today: no meaningful study time tracked yet\n"
                )
                _progress_note = (
                    "Avoid praising long study time if it is zero or negligible."
                    if study_min <= 0 else
                    "You may acknowledge the effort already spent."
                )
                ctx = (
                    "You are a warm human professor. The student " + name + " just said: '" + user_txt + "'\n\n"
                    "Context you naturally know:\n"
                    "- Current time: " + time_str + " " + _late_tag + "\n"
                    "- Weather: " + weather + ", " + str(int(temp)) + "C\n"
                    + _study_line +
                    "- Topic in progress: " + topic + "\n"
                    "- Subtopics covered: " + _covered_str + "\n"
                    "- Subtopics remaining: " + _remain_str + "\n\n"
                    "Write a natural, warm " + _mood_type + " as a caring professor who genuinely knows all the above context. "
                    + _time_note + " " + _resume_note + " " + _progress_note + " "
                    "2-3 sentences max. Sound completely natural, NOT like reading a template. "
                    "Use their name " + _name_note + "."
                )

                try:
                    farewell = global_llm.complete_once(
                        "You are a warm, human expert professor. Speak naturally and empathetically.",
                        ctx, temperature=0.85, max_tokens=100
                    ).strip()
                except Exception:
                    if is_late:
                        farewell = f"Get some rest, {name} â€” it's {time_str} and you've earned it. We'll pick up right where we left off."
                    else:
                        farewell = f"Of course, {name}! Take your time. We'll continue from {topic} when you're back."

                # TTS tone â€” warm, human, concerned-if-late
                _tone = EMPATHY_INSTRUCT_LATE if is_late else EMPATHY_INSTRUCT_BREAK
                speak_chunks([(farewell, "", [])], instruct=_tone)

                # If session_end â€” reset phase to IDLE, keep course state for resume
                if intent == "session_end":
                    with state_lock:
                        S.phase = Phase.IDLE
                    log_cli(f"ðŸŒ™ Session ended â€” {name} said goodbye at {time_str}")
                else:
                    log_cli(f"â¸ï¸ Session paused â€” {name} taking a break")

                continue

            # â”€â”€ SIDE QUESTION HANDLER (inside tutor) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if upload_turn:
                with state_lock:
                    course_title = S.title or "Course"
                    topic = S.topics[S.topic_idx] if S.topics and S.topic_idx < len(S.topics) else (S.last_topic or S.title or "General")
                    subtopic = S.subtopics[S.subtopic_idx] if S.subtopics and S.subtopic_idx < len(S.subtopics) else topic
                    phase_name = getattr(S.phase, "value", getattr(S.phase, "name", str(S.phase)))
                    quiz_q = S.qa_questions[S.qa_idx] if S.phase == Phase.QA and S.qa_idx < len(S.qa_questions) else None
                    retry_q = S.qa_retry_questions[S.qa_retry_idx] if S.phase == Phase.QA_REVIEW and S.qa_retry_idx < len(S.qa_retry_questions) else None
                    S.interrupt_just_answered = True
                upload_chunks = answer_course_upload_question(user_txt, course_title, topic, subtopic, str(phase_name))
                if upload_chunks:
                    if quiz_q:
                        _total_upload_qa = len(S.qa_questions)
                        emit_qa_board(S.qa_idx, quiz_q, _total_upload_qa)
                        emit_qa_card(S.qa_idx, quiz_q, _total_upload_qa)
                        speak_chunks(upload_chunks + [(format_qa_for_speech(quiz_q), format_qa_for_board(S.qa_idx, quiz_q), [])])
                    elif retry_q:
                        speak_chunks(upload_chunks + [(format_qa_for_speech(retry_q), format_qa_for_board(S.qa_retry_idx, retry_q), [])])
                    else:
                        speak_chunks(upload_chunks)
                    log_cli("Uploaded file question handled from course context")
                    continue

            if turn_learning_mode == LearningMode.SHALLOW:
                with state_lock:
                    current_title = _normalize_session_title(S.title, "")
                inferred_title = current_title
                if not inferred_title or inferred_title.lower() == "session":
                    inferred_title = _infer_live_session_title(LearningMode.SHALLOW, user_txt)
                with state_lock:
                    if inferred_title:
                        S.title = inferred_title
                log_cli(f"[Route] intent={intent} tone={tone} mode=shallow")
                emit_session_meta(LearningMode.SHALLOW, user_txt, S.title)
                log_cli("[Shallow] generating reply")
                shallow_chunks = shallow_mode_chunks(user_txt)
                log_cli(f"[Shallow] generated {len(shallow_chunks)} chunk(s)")
                speak_chunks(shallow_chunks)
                log_cli("[Shallow] reply delivered")
                continue

            if in_tutor and intent == "ask_question" and SIDE_RE.search(user_txt):
                with state_lock:
                    side_stack.append(S)
                sys_quick = "You are a helpful teacher. Answer in 1 short natural sentence."
                ans = emotion_complete_once(sys_quick, f'Q: "{user_txt}"\nA:', temperature=0.2, max_tokens=70).strip()
                speak_chunks([(ans, "", [])])
                with state_lock:
                    S = side_stack.pop() if side_stack else S
                continue

            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # QUICK MODE
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if mode == Mode.QUICK:
                low = (user_txt or "").lower()

                if ("last lesson" in low) or ("previous lesson" in low):
                    with state_lock:
                        if S.title and S.last_topic:
                            speak_chunks([(f"Last time, we were on {S.title}. Current topic is {S.last_topic}.", "", [])])
                        else:
                            speak_chunks([("We haven't started a lesson yet. Tell me what you want to learn.", "", [])])
                    continue

                if intent == "continue_lesson":
                    with state_lock:
                        if S.last_user_goal and S.title and S.topics:
                            S.mode  = Mode.TUTOR
                            if S.phase == Phase.IDLE:
                                S.phase = Phase.WAIT_START
                            first = S.topics[S.topic_idx] if S.topics else "the current topic"
                            msg = route.get("one_line_reply") or f"Okay. We'll continue. Ready for {first}?"
                            speak_chunks([(msg, f"Proceed with **{first}**?", [])])
                        else:
                            speak_chunks([("We don't have an active lesson yet. Tell me what you want to learn.", "", [])])
                    continue

                if intent == "tutor_request" or is_tutor_request(user_txt):
                    with state_lock:
                        S = TutorState(mode=Mode.TUTOR, phase=Phase.AGENDA)
                        S.last_user_goal = user_txt
                    # fall through to AGENDA handler
                else:
                    if intent == "prior_knowledge_claim":
                        speak_chunks([("Tell me first what you'd like to learn, then mention what you already know!", "", [])])
                        continue
                    if tone == "tired":
                        reply = route.get("one_line_reply") or "Okay. We'll go easy. What would you like to learn?"
                        speak_chunks([(reply, "", [])])
                        continue
                    if intent == "smalltalk":
                        reply = route.get("one_line_reply") or "Yes, you're audible! Tell me what you want to learn today."
                        speak_chunks([(reply, "", [])])
                        continue
                    if intent == "meta_complaint":
                        reply = route.get("one_line_reply") or "Got it. Tell me what felt off and I'll adjust."
                        speak_chunks([(reply, "", [])])
                        continue
                    sys_quick = "You are a warm human tutor. Reply naturally in 1â€“2 short sentences. No bullets."
                    ans = emotion_complete_once(sys_quick, f'Student: "{user_txt}"\nTutor:', temperature=0.55, max_tokens=140).strip()
                    speak_chunks([(ans, "", [])])
                    continue

            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # TUTOR STATE MACHINE
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with state_lock:
                phase = S.phase

            # â”€â”€ AGENDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.AGENDA:
                # Mark session start for empathy timing
                with emp_lock:
                    if EMP.session_start_ts == 0.0:
                        EMP.session_start_ts = time.time()
                ag = agenda_json(user_txt)
                with state_lock:
                    S.title     = ag["title"]
                    S.topics    = ag["topics"]
                    S.topic_idx = 0
                    S.phase     = Phase.WAIT_START
                    S.all_taught = {}
                    S.taught_points = []
                    S.prior_known_indices = []
                    S.subtopics = []
                    S.subtopic_idx = 0
                    S.subtopics_done = []
                    S.qa_questions = []
                    S.qa_idx = 0
                    S.qa_correct = 0
                    S.qa_wrong_items = []
                    S.last_topic = ag["topics"][0] if ag["topics"] else ""
                emit_course_progress()
                emit_session_meta(LearningMode.COURSE, user_txt, ag["title"])

                # Only clear the visual board when this mode has no prior content yet.
                _was_idle = not bool(get_visual_state().live_board_text.strip())
                if _was_idle:
                    clear_board()

                agenda_board = f"**{ag['title']}**\n\n**Contents**\n"
                agenda_board += "\n".join([f"{i}) {t}" for i, t in enumerate(ag["topics"], 1)])
                update_board_sync(agenda_board, append=not _was_idle, mode="type", cps=40, scroll=True, timeout=12.0)

                chunks = [
                    (f"Alright! Here's our plan for {ag['title']}.", "", []),
                ]
                for i, tpc in enumerate(ag["topics"], 1):
                    target = f"{i}) {tpc}"
                    chunks.append((f"{tpc}.", "", [{"type": "pop", "target": target, "duration_ms": 650, "delay_ms": 0}]))

                first = ag["topics"][0] if ag["topics"] else "the first topic"
                # LLM-generated natural prior-knowledge question (not hardcoded)
                try:
                    _pk_prompt = (
                        f"A student just asked to learn: {ag['title']}. "
                        f"The course has these topics: {', '.join(ag['topics'])}. "
                        f"Write ONE natural, conversational sentence (not robotic) asking if they already know any of these topics. "
                        f"Vary phrasing each time. Keep it under 40 words. No quotes, no prefix."
                    )
                    _pk_q = global_llm.complete_once("You are a friendly tutor.", _pk_prompt, temperature=0.8, max_tokens=60).strip()
                    if not _pk_q or len(_pk_q) < 10:
                        raise ValueError("empty")
                except Exception:
                    _pk_q = f"Have you seen any of these topics before, or are we starting completely fresh?"
                chunks.append((
                    _pk_q,
                    f"Ready?\n-> **{first}**",
                    [{"type": "glow", "target": first, "duration_ms": 900, "delay_ms": 0}]
                ))
                speak_chunks(chunks)
                continue

            # â”€â”€ WAIT_START â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.WAIT_START:

                # â”€â”€ PRIOR CONFIRM MODE: student is responding to our clarification question â”€â”€
                with state_lock:
                    _pcm = S.prior_confirm_mode
                    _pci = list(S.prior_confirm_indices)
                    _pct = S.prior_confirm_topic

                if _pcm:
                    # Student has responded to "should I skip X?" â€” use LLM to understand intent
                    _CLARIFY_SYS = (
                        "You are an intelligent tutor assistant. "
                        "A student was asked whether to skip topics they claim to know. "
                        "Classify their response into ONE action:\n"
                        "- skip_all: skip all detected topics, start from next\n"
                        "- skip_but_partial: skip but teach a specific part first (extract what they want)\n"
                        "- dont_skip: student wants to learn normally\n"
                        "- needs_clarification: response is ambiguous\n"
                        "Return ONLY JSON: {\"action\": \"...\", \"partial_request\": \"...\", \"one_line_reply\": \"...\"}"
                    )
                    _clarify_prompt = (
                        f"Topics student claimed to know: {', '.join([S.topics[i] for i in _pci if i < len(S.topics)])}. "
                        f"Next topic I planned to jump to: {_pct}. "
                        f"Student's latest response: '{user_txt}'. "
                        f"Classify their intent."
                    )
                    try:
                        _cr = global_llm.complete_once(_CLARIFY_SYS, _clarify_prompt, temperature=0.2, max_tokens=120)
                        _cr = _cr.strip().lstrip("```json").rstrip("```").strip()
                        import json as _json
                        _cd = _json.loads(_cr)
                        _caction = _cd.get("action","skip_all")
                        _cpartial = _cd.get("partial_request","")
                        _creply   = _cd.get("one_line_reply","")
                    except Exception:
                        _caction = "skip_all"
                        _cpartial = ""
                        _creply   = ""

                    with state_lock:
                        S.prior_confirm_mode    = False
                        S.prior_confirm_indices = []
                        S.prior_confirm_topic   = ""

                    if _caction == "dont_skip":
                        # Student wants full teaching â€” reset and start normally
                        with state_lock:
                            S.prior_known_indices = []
                            S.topic_idx = 0
                            S.last_topic = S.topics[0] if S.topics else ""
                        _reply = _creply or "No problem, let's go through it properly then!"
                        speak_chunks([(_reply, "", [])])
                        continue

                    elif _caction == "skip_but_partial" and _cpartial:
                        # Student wants to skip but needs a specific sub-part first
                        # Teach the specific partial request inline, then jump to next topic
                        _partial_sys = (
                            "You are an expert tutor. Student knows most of this topic "
                            "but wants a specific brief explanation before moving on. "
                            "Provide a focused, practical explanation in 3-5 sentences. "
                            "Use board-style summary. End with: 'Shall I continue to the next topic now?'"
                        )
                        _partial_ctx = (
                            f"Course: {S.title}. Skipped topic: {', '.join([S.topics[i] for i in _pci if i < len(S.topics)])}. "
                            f"Student's specific request: '{_cpartial}'. "
                            f"Student context: {S.student_context or 'none'}."
                        )
                        try:
                            _partial_speech = emotion_complete_once(_partial_sys, _partial_ctx, temperature=0.45, max_tokens=200).strip()
                        except Exception:
                            _partial_speech = f"Sure! {_cpartial}. Shall I continue to the next topic now?"
                        # Simple board summary of the partial
                        try:
                            _pb_sys = "Summarize as chalkboard notes, max 4 lines. Use bullet format."
                            _partial_board = global_llm.complete_once(_pb_sys, _partial_speech, temperature=0.2, max_tokens=80).strip()
                        except Exception:
                            _partial_board = f"**Quick Note**\n- {_cpartial}"
                        # Set confirm mode again so next response continues to next topic
                        with state_lock:
                            S.prior_confirm_mode    = True
                            S.prior_confirm_indices = _pci
                            S.prior_confirm_topic   = _pct
                        _intro = _creply or f"Sure! Let me cover that quickly before we move on."
                        speak_chunks([(_intro, "", []), (_partial_speech, _partial_board, [])])
                        continue

                    else:
                        # skip_all or needs_clarification â†’ skip and proceed
                        with state_lock:
                            S.prior_known_indices = _pci
                            while S.topic_idx in _pci and S.topic_idx < len(S.topics) - 1:
                                S.topic_idx += 1
                            topic = S.topics[S.topic_idx] if S.topics else _pct
                            S.last_topic = topic
                            S.phase = Phase.SUBTOPIC_PLAN
                            S.topic_completed = False
                            S.taught_points = []
                            S.subtopics = []
                            S.subtopic_idx = 0
                            S.subtopics_done = []
                            all_topics = list(S.topics)
                            pref = S.last_user_goal or ""
                        _reply = _creply or f"Perfect, jumping straight to {topic}!"
                        board_msg = f"Skipping -> **{topic}**"
                        subj = guess_subject_type(S.title + " " + topic)
                        log_cli(f"â© Prior confirmed (skip_all) â†’ {topic}")
                        subs = plan_subtopics(S.title, topic, subj, all_topics=all_topics, student_pref=pref)
                        with state_lock:
                            S.subtopics = subs
                            S.subtopic_idx = 0
                            S.phase = Phase.TEACH_SUBTOPIC
                        subtopic = subs[0]
                        chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, f"Student context: {S.student_context or 'none'}. Start teaching.")
                        teach_chunks = []
                        with state_lock:
                            for (sp2, bd2, fx2, meta2) in chunks_llm:
                                if isinstance(meta2, dict):
                                    for tp in (meta2.get("taught_points") or []):
                                        tp = str(tp).strip()
                                        if tp and tp not in S.taught_points:
                                            S.taught_points.append(tp)
                                teach_chunks.append((sp2, bd2, fx2))
                            if len(subs) > 1:
                                S.phase = Phase.SUBTOPIC_WRAP
                                S.subtopics_done.append(subtopic)
                            else:
                                S.phase = Phase.CONFIDENCE_CHECK
                        subs_board = "\n".join([f"- {s}" for s in subs])
                        intro_chunks = [(
                            f"Let me start with {topic}.",
                            f"**{topic}**\n{subs_board}",
                            [{"type": "glow", "target": topic, "duration_ms": 900, "delay_ms": 0}]
                        )]
                        speak_chunks([(_reply, board_msg, [])] + intro_chunks + teach_chunks)
                        continue

                # â”€â”€ First-time prior knowledge claim: ask clarifying question â”€
                if intent == "prior_knowledge_claim":
                    known = detect_prior_knowledge(user_txt, S.topics)
                    if not known:
                        # No specific topics detected â€” treat as fresh start
                        speak_chunks([(route.get("one_line_reply") or "Got it! Tell me when you're ready to start.", "", [])])
                        continue

                    with state_lock:
                        _known_names = [S.topics[i] for i in known if i < len(S.topics)]
                        # Safety: find next unknown topic
                        _tmp_idx = 0
                        while _tmp_idx in known and _tmp_idx < len(S.topics) - 1:
                            _tmp_idx += 1
                        _next_topic = S.topics[_tmp_idx] if S.topics else "the first topic"
                        # Store for confirm flow
                        S.prior_confirm_mode    = True
                        S.prior_confirm_indices = known
                        S.prior_confirm_topic   = _next_topic

                    # Generate a warm, conversational clarification question
                    _known_str = ", ".join(_known_names) if _known_names else "some topics"
                    try:
                        _ask_sys = (
                            "You are a warm, intelligent tutor. Student just mentioned they already know "
                            "some topics. Ask a natural clarifying question to understand exactly what they "
                            "want â€” should you skip entirely, or do they need any specific part first? "
                            "Keep it conversational, 1-2 sentences max. Reference what they said and what "
                            "you were about to skip."
                        )
                        _ask_ctx = (
                            f"Course: {S.title}. "
                            f"Topics student claims to know: {_known_str}. "
                            f"I planned to jump to: {_next_topic}. "
                            f"Student said: '{user_txt}'. "
                            f"Student context: {S.student_context or 'none'}. "
                            f"Ask them to confirm or tell you what they need."
                        )
                        _ask_msg = emotion_complete_once(_ask_sys, _ask_ctx, temperature=0.8, max_tokens=60).strip()
                    except Exception:
                        _ask_msg = f"Sounds good! So should I skip {_known_str} entirely and jump to {_next_topic}, or is there anything specific from {_known_str} you'd like me to cover first?"

                    speak_chunks([(_ask_msg, "", [])])
                    log_cli(f"â“ Prior knowledge clarification: {_ask_msg}")
                    continue

                if intent in ("confirm_start", "confirm"):
                    with state_lock:
                        S.phase = Phase.SUBTOPIC_PLAN
                        S.topic_completed = False
                        S.taught_points = []
                        S.subtopics = []
                        S.subtopic_idx = 0
                        S.subtopics_done = []
                        topic = S.topics[S.topic_idx] if S.topics else "Topic 1"
                        S.last_topic = topic
                        all_topics = list(S.topics)
                        pref = S.last_user_goal or ""

                    subj = guess_subject_type(S.title + " " + topic)
                    log_cli(f"ðŸ“š Planning subtopics for: {topic}")
                    subs = plan_subtopics(S.title, topic, subj, all_topics=all_topics, student_pref=pref)
                    with state_lock:
                        S.subtopics = subs
                        S.subtopic_idx = 0
                        S.phase = Phase.TEACH_SUBTOPIC

                    subtopic = subs[0]
                    set_status("ANALYZING")

                    chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, "Start teaching now, please.")
                    chunks = []
                    with state_lock:
                        for (sp, bd, fx, meta) in chunks_llm:
                            if isinstance(meta, dict):
                                for tp in (meta.get("taught_points") or []):
                                    tp = str(tp).strip()
                                    if tp and tp not in S.taught_points:
                                        S.taught_points.append(tp)
                            chunks.append((sp, bd, fx))
                        if len(S.subtopics) > 1:
                            S.phase = Phase.SUBTOPIC_WRAP
                            S.subtopics_done.append(subtopic)
                        else:
                            S.phase = Phase.CONFIDENCE_CHECK

                    # intro before teaching
                    subs_board = "\n".join([f"- {s}" for s in subs])
                    intro_chunks = [(
                        f"Let's start with {topic}.",
                        f"**{topic}**\n{subs_board}",
                        [{"type": "glow", "target": topic, "duration_ms": 900, "delay_ms": 0}]
                    )]
                    speak_chunks(intro_chunks + chunks)
                    continue

                if intent == "ask_question":
                    speak_chunks([("Good question. Go ahead â€” then we'll get started.", "", [])])
                    continue

                # â”€â”€ WAIT_START CATCH-ALL: any unrecognized intent in this phase
                # is treated as "I'm a beginner, let's start" â€” never topic switch
                if intent not in ("prior_knowledge_claim", "confirm_start", "confirm", "ask_question"):
                    intent = "confirm_start"   # force-treat as start signal

                # Fix Issue 1: If confirm_start message ALSO contains topic-skip language
                # ("proceed from X", "skip to X", "start from topic N", "already know") â†’
                # treat it as prior_knowledge_claim so the student isn't re-taught stuff they know
                _PRIOR_HINT_RE = re.compile(
                    r"\b(proceed from|start from|skip to|begin from|i already|"
                    r"i know|i've done|already installed|already have|already covered|"
                    r"from topic\s*\d|from\s+\w+.*(?:syntax|variable|function|loop|class|module))\b",
                    re.I
                )
                if intent == "confirm_start" and _PRIOR_HINT_RE.search(user_txt):
                    _recheck_known = detect_prior_knowledge(user_txt, S.topics)
                    if _recheck_known:
                        log_cli(f"ðŸ” confirm_start re-classified as prior_knowledge_claim: {_recheck_known}")
                        intent = "prior_knowledge_claim"
                        with state_lock:
                            S.prior_known_indices = _recheck_known
                            while S.topic_idx in S.prior_known_indices and S.topic_idx < len(S.topics) - 1:
                                S.topic_idx += 1
                            if S.topic_idx in S.prior_known_indices:
                                S.topic_idx = len(S.topics) - 1
                            next_topic = S.topics[S.topic_idx] if S.topics else "the first topic"
                            known_names = [S.topics[i] for i in _recheck_known if i < len(S.topics)]
                            if known_names and next_topic in known_names:
                                known_names = [n for n in known_names if n != next_topic]
                            S.last_topic = next_topic

                        _known_count = len(known_names)
                        _skip_to = next_topic
                        try:
                            _pk_confirm_prompt = (
                                f"Student claims to know {_known_count} topic(s): {', '.join(known_names)}. "
                                f"The next topic to teach is: {_skip_to}. "
                                f"Write ONE short, natural confirmation (max 20 words). "
                                f"Never list all topic names. Brief and human."
                            )
                            _pk_msg = global_llm.complete_once(
                                "You are a friendly tutor. Reply in ONE short sentence only.",
                                _pk_confirm_prompt, temperature=0.8, max_tokens=40
                            ).strip() or f"Got it - jumping to {next_topic}!"
                        except Exception:
                            _pk_msg = f"Got it - starting from {next_topic}!"
                        _pk_board = f"Skipping {_known_count} topic(s) -> **{next_topic}**"
                        log_cli(f"â© Prior knowledge (re-detected): skipping {_known_count} â†’ {next_topic}")

                        # Continue as prior_knowledge_claim path below
                        with state_lock:
                            S.phase = Phase.SUBTOPIC_PLAN
                            S.topic_completed = False
                            S.taught_points = []
                            S.subtopics = []
                            S.subtopic_idx = 0
                            S.subtopics_done = []
                            topic = S.topics[S.topic_idx] if S.topics else "Topic 1"
                            S.last_topic = topic
                            all_topics_b = list(S.topics)
                            pref_b = S.last_user_goal or ""

                        subj = guess_subject_type(S.title + " " + topic)
                        subs = plan_subtopics(S.title, topic, subj, all_topics=all_topics_b, student_pref=pref_b)
                        with state_lock:
                            S.subtopics = subs
                            S.subtopic_idx = 0
                            S.phase = Phase.TEACH_SUBTOPIC

                        subtopic = subs[0]
                        chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, "Start teaching now.")
                        teach_chunks = []
                        with state_lock:
                            for (sp, bd, fx, meta) in chunks_llm:
                                if isinstance(meta, dict):
                                    for tp in (meta.get("taught_points") or []):
                                        tp = str(tp).strip()
                                        if tp and tp not in S.taught_points:
                                            S.taught_points.append(tp)
                                teach_chunks.append((sp, bd, fx))
                            if len(subs) > 1:
                                S.phase = Phase.SUBTOPIC_WRAP
                                S.subtopics_done.append(subtopic)
                            else:
                                S.phase = Phase.CONFIDENCE_CHECK

                        subs_board = "\n".join([f"- {s}" for s in subs])
                        intro_ch = [(
                            f"Let me start with {topic}.",
                            f"**{topic}**\n{subs_board}",
                            [{"type": "glow", "target": topic, "duration_ms": 900, "delay_ms": 0}]
                        )]
                        speak_chunks([(_pk_msg, _pk_board, [])] + intro_ch + teach_chunks)
                        continue

                # "I'm a beginner / I don't know anything / let's start" â†’ treat as confirm_start
                _beginner_signals = (
                    "beginner", "new to", "don't know", "no idea", "never", "fresh",
                    "zero", "nothing", "start", "go ahead", "let's go",
                    "ok", "okay", "yes", "sure", "yep", "yeah", "ready", "alright",
                    "basic", "completely", "absolute", "clueless", "haven't", "never heard",
                    "just started", "no background", "from scratch", "no knowledge",
                    "not familiar", "unfamiliar", "newbie", "rookie", "total beginner",
                )
                _txt_low = user_txt.lower()
                if intent == "confirm_start" or any(sig in _txt_low for sig in _beginner_signals):
                    with state_lock:
                        S.phase = Phase.SUBTOPIC_PLAN
                        S.topic_completed = False
                        S.taught_points = []
                        S.subtopics = []
                        S.subtopic_idx = 0
                        S.subtopics_done = []
                        topic = S.topics[S.topic_idx] if S.topics else "Topic 1"
                        S.last_topic = topic
                        all_topics_b = list(S.topics)
                        pref_b = S.last_user_goal or ""

                    subj = guess_subject_type(S.title + " " + topic)
                    log_cli(f"ðŸ“š Planning subtopics for: {topic}")
                    subs = plan_subtopics(S.title, topic, subj, all_topics=all_topics_b, student_pref=pref_b)
                    with state_lock:
                        S.subtopics = subs
                        S.subtopic_idx = 0
                        S.phase = Phase.TEACH_SUBTOPIC

                    subtopic = subs[0]
                    chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, "Start teaching now.")
                    teach_chunks = []
                    with state_lock:
                        for (sp, bd, fx, meta) in chunks_llm:
                            if isinstance(meta, dict):
                                for tp in (meta.get("taught_points") or []):
                                    tp = str(tp).strip()
                                    if tp and tp not in S.taught_points:
                                        S.taught_points.append(tp)
                            teach_chunks.append((sp, bd, fx))
                        if len(subs) > 1:
                            S.phase = Phase.SUBTOPIC_WRAP
                            S.subtopics_done.append(subtopic)
                        else:
                            S.phase = Phase.CONFIDENCE_CHECK

                    subs_board_b = "\n".join([f"- {s}" for s in subs])
                    intro_ch = [(
                        f"Alright, let's begin with {topic}!",
                        f"**{topic}**\n{subs_board_b}",
                        [{"type": "glow", "target": topic, "duration_ms": 900, "delay_ms": 0}]
                    )]
                    speak_chunks(intro_ch + teach_chunks)
                    continue

                first = S.topics[S.topic_idx] if S.topics else "the first topic"
                speak_chunks([(route.get("one_line_reply") or f"Alright. Say 'start' and we'll dive into {first}.", "", [])])
                continue

            # â”€â”€ TEACH_SUBTOPIC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.TEACH_SUBTOPIC:
                with state_lock:
                    topic    = S.topics[S.topic_idx] if S.topics else "Topic"
                    subtopic = S.subtopics[S.subtopic_idx] if S.subtopics else topic
                    S.last_topic = topic

                subj = guess_subject_type(S.title + " " + topic)

                # â”€â”€ INTENT HANDLERS IN TEACH_SUBTOPIC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if intent == "ask_question":
                    import re as _re

                    # â”€â”€ Off-topic detection: is the question completely unrelated to the current lesson? â”€â”€
                    _OFF_TOPIC_SYS = (
                        "You are a tutor assistant. Decide if a student's question is RELATED or UNRELATED "
                        "to the current course topic. Return ONLY JSON: "
                        '{"off_topic": true/false, "confidence": 0.0-1.0}'
                    )
                    _off_topic_prompt = (
                        f"Course: {S.title}. Current topic: {topic}. Subtopic: {subtopic}.\n"
                        f"Student question: '{user_txt}'"
                    )
                    _is_off_topic = False
                    try:
                        _otr = global_llm.complete_once(_OFF_TOPIC_SYS, _off_topic_prompt, temperature=0.1, max_tokens=40)
                        import json as _otj
                        _otd = _otj.loads(_otr.strip().lstrip("```json").rstrip("```").strip())
                        _is_off_topic = _otd.get("off_topic", False) and _otd.get("confidence", 0) >= 0.75
                    except Exception:
                        _is_off_topic = False

                    if _is_off_topic:
                        # Warm, witty response acknowledging the topic switch + redirect
                        _ot_sys = (
                            "You are a warm, witty human tutor. The student went completely off-topic "
                            "mid-lesson. Respond in 2-3 sentences: be genuinely warm and playful "
                            "(a light joke or surprised remark is fine), give a VERY brief answer "
                            "to their question (1 sentence), then naturally redirect back to the lesson. "
                            "Sound like a real person, not a robot. Never say 'off-topic'."
                        )
                        _ot_ctx = (
                            f"Course: {S.title}. Teaching: {topic} -> {subtopic}.\n"
                            f"Student's random question: '{user_txt}'.\n"
                            f"Student name: {S.student_name if hasattr(S, 'student_name') else 'there'}."
                        )
                        try:
                            _ot_reply = emotion_complete_once(_ot_sys, _ot_ctx, temperature=0.85, max_tokens=120).strip()
                        except Exception:
                            _ot_reply = (f"Ha! Quite the plot twist â€” from Python to that! "
                                        f"Briefly: I'll save you a deep dive on that one. "
                                        f"But let's get back to {subtopic} â€” we were just getting to the good part!")
                        speak_chunks([(_ot_reply, "", [])])
                        log_cli(f"ðŸŒ€ Off-topic detected: '{user_txt[:50]}' â€” redirected")
                        continue

                    # Detect: is this a FULL EXAMPLE/TEACH request, or a quick clarification?
                    _teach_signals = ("teach me", "show me", "give me an example", "can you demonstrate",
                                      "explain in detail", "real.?time example", "how does.*work",
                                      "walk me through", "illustrate", "step.?by.?step example")
                    _wants_example = any(
                        _re.search(sig, user_txt, _re.I) for sig in _teach_signals
                    )

                    if _wants_example:
                        # Student wants a real example/mini-teach â€” give proper board content + speech
                        ex_sys = (
                            "You are an expert tutor. The student interrupted to ask for a practical example. "
                            "Teach the concept with a real code/practical example. "
                            "Keep it focused to 3-5 sentences + one concrete example. "
                            "Do NOT re-teach everything â€” just the example they asked for. "
                            "End with: 'Does that example help? Shall we continue?'"
                        )
                        ex_ctx = (
                            f'Course: {S.title}. Topic: {topic}. Subtopic: {subtopic}.\n'
                            f'Student request: "{user_txt}"\n'
                            f'Already covered: {", ".join(S.taught_points[-5:]) if S.taught_points else "just started"}.'
                        )
                        try:
                            ex_speech = emotion_complete_once(ex_sys, ex_ctx, temperature=0.45, max_tokens=250).strip()
                        except Exception:
                            ex_speech = f"Great question! Here is a quick example for {subtopic}. Shall we continue?"

                        # Generate proper board note for this example
                        try:
                            brd_sys = ("Turn this example into a chalkboard-style summary. "
                                       "Use: 'ðŸ“Œ Example:\n[code/fact]'. Max 5 lines. No extra text.")
                            ex_board = global_llm.complete_once(brd_sys, ex_speech, temperature=0.2, max_tokens=80).strip()
                        except Exception:
                            ex_board = f"ðŸ“Œ Example: {subtopic}"

                        with state_lock:
                            S.interrupt_just_answered = True
                        # Show board note BEFORE speaking (immediate visual feedback)
                        socketio.emit("board_text", {
                            "text": f"\nðŸ“Œ Quick Example â€” {subtopic}\n",
                            "append": True, "mode": "type", "cps": 30, "scroll": True,
                        })
                        set_visual_render_text(f"Quick Example - {subtopic}", append=True)
                        speak_chunks([(ex_speech, ex_board, [])])
                        continue

                    else:
                        # Quick clarification â€” 2-3 sentences, compact board note
                        sys_q = (
                            "You are a warm expert tutor who was interrupted mid-lesson. "
                            "Answer ONLY the specific question asked. Be brief and direct (2-3 sentences max). "
                            "Do NOT restart from the beginning or recap what was already covered. "
                            "End with: 'Shall we continue where we left off?'"
                        )
                        ctx_q = (
                            f'Course: {S.title}. Topic: {topic}. Subtopic being taught: {subtopic}.\n'
                            f'Already covered: {", ".join(S.taught_points[-5:]) if S.taught_points else "just started"}.\n'
                            f"Student question: {user_txt}"
                        )
                        try:
                            ans_full = emotion_complete_once(sys_q, ctx_q, temperature=0.4, max_tokens=150).strip()
                        except Exception:
                            ans_full = "Great question! " + user_txt + " â€” let me explain that briefly. Shall we continue where we left off?"

                        # Board: key fact, always non-empty
                        try:
                            board_sys = "Extract the key fact from this answer as a 1-line board note. Max 60 chars. No prefix. Just the fact."
                            board_ans = global_llm.complete_once(board_sys, ans_full, temperature=0.2, max_tokens=40).strip()
                        except Exception:
                            board_ans = ""
                        if not board_ans:
                            board_ans = f"ðŸ’¡ {user_txt[:60]}"

                        with state_lock:
                            S.interrupt_just_answered = True

                        speak_chunks([(ans_full, board_ans, [])])
                        continue

                # âœ… FIX: confirm_start / agree â†’ advance state, don't re-teach!
                if intent in ("confirm_start", "reject_or_skip") or user_txt.strip().lower() in ("ok", "okay", "yes", "got it", "sure", "continue", "next", "yep", "yeah", "now continue"):
                    with state_lock:
                        current_phase = S.phase
                        _was_interrupted = S.subtopic_interrupted
                        _interrupted_subtopic = S.subtopics[S.subtopic_idx] if (S.subtopics and S.subtopic_idx < len(S.subtopics)) else None

                    # Fix 5: If hand-raise interrupted mid-subtopic, resume it first
                    # (ask the closing question again so student gets a chance to confirm)
                    if _was_interrupted and _interrupted_subtopic:
                        with state_lock:
                            S.subtopic_interrupted = False
                        _resume_msg = (
                            f"Alright, let's continue where we left off â€” {_interrupted_subtopic}. "
                            f"Does that all make sense, or shall we move to the next part?"
                        )
                        speak_chunks([(_resume_msg, "", [])])
                        log_cli(f"â†©ï¸  Resumed interrupted subtopic: {_interrupted_subtopic}")
                        continue

                    if current_phase == Phase.TEACH_SUBTOPIC:
                        # Student acknowledged â€” advance to wrap
                        with state_lock:
                            if S.subtopics and S.subtopics[S.subtopic_idx] not in S.subtopics_done:
                                S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                            if S.subtopic_idx + 1 < len(S.subtopics):
                                S.subtopic_idx += 1
                                S.phase = Phase.TEACH_SUBTOPIC
                                next_sub = S.subtopics[S.subtopic_idx]
                            else:
                                S.phase = Phase.CONFIDENCE_CHECK
                                next_sub = None
                        if next_sub:
                            subj2 = guess_subject_type(S.title + " " + topic)
                            chunks_llm2 = teach_subtopic_chunks(S.title, topic, next_sub, subj2, "Start this subtopic now.")
                            chunks_out2 = []
                            with state_lock:
                                for (sp2, bd2, fx2, meta2) in chunks_llm2:
                                    if isinstance(meta2, dict):
                                        for tp2 in (meta2.get("taught_points") or []):
                                            tp2 = str(tp2).strip()
                                            if tp2 and tp2 not in S.taught_points:
                                                S.taught_points.append(tp2)
                                        if meta2.get("subtopic_complete") is True:
                                            if S.subtopic_idx + 1 < len(S.subtopics):
                                                S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                                                S.phase = Phase.SUBTOPIC_WRAP
                                            else:
                                                S.phase = Phase.CONFIDENCE_CHECK
                                    chunks_out2.append((sp2, bd2, fx2))
                            speak_chunks(chunks_out2)
                        else:
                            speak_chunks([("Great, we've covered all the subtopics! Let me check your understanding.", "", [])])
                    continue

                if intent == "not_understood":
                    # re-explain current subtopic from a different angle
                    chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, "The student didn't understand. Re-explain from a different angle.")
                    chunks = []
                    with state_lock:
                        for (sp, bd, fx, meta) in chunks_llm:
                            if isinstance(meta, dict):
                                for tp in (meta.get("taught_points") or []):
                                    tp = str(tp).strip()
                                    if tp and tp not in S.taught_points:
                                        S.taught_points.append(tp)
                            chunks.append((sp, bd, fx))
                    speak_chunks(chunks)
                    continue

                # â”€â”€ Interrupt-answer recovery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # If we just answered a hand-raise question, the NEXT student
                # input (whatever it is, even garbled STT) means "continue" â€”
                # do NOT restart the subtopic from scratch.
                with state_lock:
                    _just_answered = S.interrupt_just_answered
                    if _just_answered:
                        S.interrupt_just_answered = False
                        S.subtopic_interrupted = False   # side-question handled; clean slate

                if _just_answered:
                    # Resume: advance to SUBTOPIC_WRAP so we don't re-teach
                    with state_lock:
                        if S.subtopics and S.subtopics[S.subtopic_idx] not in S.subtopics_done:
                            S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                        if S.subtopic_idx + 1 < len(S.subtopics):
                            S.subtopic_idx += 1
                            S.phase = Phase.TEACH_SUBTOPIC
                            _next_resume = S.subtopics[S.subtopic_idx]
                        else:
                            S.phase = Phase.CONFIDENCE_CHECK
                            _next_resume = None

                    if _next_resume:
                        subj_r = guess_subject_type(S.title + " " + topic)
                        chunks_r = teach_subtopic_chunks(S.title, topic, _next_resume, subj_r, "Start this subtopic now.")
                        chunks_out_r = []
                        with state_lock:
                            for (sp_r, bd_r, fx_r, meta_r) in chunks_r:
                                if isinstance(meta_r, dict):
                                    for tp_r in (meta_r.get("taught_points") or []):
                                        tp_r = str(tp_r).strip()
                                        if tp_r and tp_r not in S.taught_points:
                                            S.taught_points.append(tp_r)
                                    if meta_r.get("subtopic_complete") is True:
                                        if S.subtopic_idx + 1 < len(S.subtopics):
                                            S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                                            S.phase = Phase.SUBTOPIC_WRAP
                                        else:
                                            S.phase = Phase.CONFIDENCE_CHECK
                                chunks_out_r.append((sp_r, bd_r, fx_r))
                        transition_r = [(f"Alright, continuing with {_next_resume}.", f"\n**{_next_resume}**", [])]
                        speak_chunks(transition_r + chunks_out_r)
                    else:
                        speak_chunks([("Great, we've covered the subtopics. Let me check your understanding.", "", [])])
                    continue

                # â”€â”€ Normal fallthrough: student asked something mid-lesson â”€â”€
                chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic, subj, user_txt)
                chunks = []
                with state_lock:
                    for (sp, bd, fx, meta) in chunks_llm:
                        if isinstance(meta, dict):
                            for tp in (meta.get("taught_points") or []):
                                tp = str(tp).strip()
                                if tp and tp not in S.taught_points:
                                    S.taught_points.append(tp)
                            if meta.get("subtopic_complete") is True:
                                # auto-advance
                                if S.subtopic_idx + 1 < len(S.subtopics):
                                    S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                                    S.phase = Phase.SUBTOPIC_WRAP
                                else:
                                    S.phase = Phase.CONFIDENCE_CHECK
                        chunks.append((sp, bd, fx))
                speak_chunks(chunks)
                continue

            # â”€â”€ SUBTOPIC_WRAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.SUBTOPIC_WRAP:
                with state_lock:
                    done_list = list(S.subtopics_done)
                    topic     = S.topics[S.topic_idx] if S.topics else "Topic"
                    remaining = [s for s in S.subtopics if s not in S.subtopics_done]
                    subtopic_next = remaining[0] if remaining else None

                if intent == "ask_question" or intent == "not_understood":
                    with state_lock:
                        S.phase = Phase.TEACH_SUBTOPIC
                    sys_q = "You are a kind tutor. Answer in 1-2 sentences, naturally."
                    ans = emotion_complete_once(sys_q, f'Teaching {topic}. Student says: "{user_txt}"', temperature=0.4, max_tokens=120).strip()
                    speak_chunks([(ans, "", [])])
                    continue

                if intent == "reject_or_skip" and subtopic_next:
                    with state_lock:
                        S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                        S.subtopic_idx += 1
                        if S.subtopic_idx >= len(S.subtopics):
                            S.phase = Phase.CONFIDENCE_CHECK
                        else:
                            S.phase = Phase.TEACH_SUBTOPIC
                    speak_chunks([("Okay, moving on.", "", [])])
                    continue

                # default: student confirms or says something â†’ teach next subtopic
                if subtopic_next:
                    with state_lock:
                        # mark current as done and advance
                        if S.subtopics and S.subtopics[S.subtopic_idx] not in S.subtopics_done:
                            S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                        S.subtopic_idx += 1
                        S.phase = Phase.TEACH_SUBTOPIC

                    subj = guess_subject_type(S.title + " " + topic)
                    chunks_llm = teach_subtopic_chunks(S.title, topic, subtopic_next, subj, "Start this subtopic now.")
                    chunks_out = []
                    with state_lock:
                        for (sp, bd, fx, meta) in chunks_llm:
                            if isinstance(meta, dict):
                                for tp in (meta.get("taught_points") or []):
                                    tp = str(tp).strip()
                                    if tp and tp not in S.taught_points:
                                        S.taught_points.append(tp)
                                if meta.get("subtopic_complete") is True:
                                    if S.subtopic_idx + 1 < len(S.subtopics):
                                        S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                                        S.phase = Phase.SUBTOPIC_WRAP
                                    else:
                                        S.phase = Phase.CONFIDENCE_CHECK
                            chunks_out.append((sp, bd, fx))

                    transition = [(
                        f"Now let's get into {subtopic_next}.",
                        f"\n**{subtopic_next}**",
                        [{"type": "glow", "target": subtopic_next, "duration_ms": 900, "delay_ms": 0}]
                    )]
                    speak_chunks(transition + chunks_out)
                else:
                    with state_lock:
                        S.phase = Phase.CONFIDENCE_CHECK
                    speak_chunks([("We've covered all the subtopics. Let me check your understanding.", "", [])])
                continue

            # â”€â”€ CONFIDENCE_CHECK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.CONFIDENCE_CHECK:
                # Fix 3: If student is on break, don't start Q&A from random input
                with emp_lock:
                    _break_active = EMP.break_active
                if _break_active:
                    log_cli("â¸ï¸  Confidence check skipped â€” student is on break")
                    continue
                # If student explicitly asks a question â€” go back and answer it
                if intent in ("ask_question", "not_understood"):
                    with state_lock:
                        S.phase = Phase.TEACH_SUBTOPIC
                    speak_chunks([("Sure! Let me clear that up first.", "", [])])
                    continue

                with state_lock:
                    topic     = S.topics[S.topic_idx] if S.topics else "Topic"
                    subs_done = list(S.subtopics_done)
                    tp_list   = list(S.taught_points)
                    # Fix 6: strip audibility/smalltalk lines from history
                    _SMALLTALK_RE = re.compile(
                        r"\b(hello|hi|audible|can you hear|is my voice|testing|"
                        r"are you there|how are you|good morning|good evening|"
                        r"yes please|okay|ok|sure|proceed|continue|go ahead)\b",
                        re.I
                    )
                    _course_memory = get_mode_memory(LearningMode.COURSE)
                    u_hist = [
                        u for u in list(_course_memory.user_history)
                        if not _SMALLTALK_RE.search(u.strip())
                    ]
                    recap_attempts = getattr(S, "recap_attempts", 0)

                # Force-override: student explicitly says they're ready
                _ready_signals = ("ready", "q&a", "question", "quiz", "proceed", "go ahead",
                                   "yes", "ok", "okay", "sure", "yep", "yeah", "start quiz",
                                   "test me", "let's go", "bring it on", "fire away")
                # Exclude if session_pause/session_end intent detected â€” never force Q&A on those
                _no_qa = intent in ("session_pause", "session_end", "reject_or_skip")
                _force_ready = (not _no_qa) and any(sig in user_txt.lower() for sig in _ready_signals)

                # Also force after 1 recap attempt â€” never loop more than once
                if _force_ready or recap_attempts >= 1:
                    check = {"ready": True, "qa_count": max(3, min(5, len(tp_list) // 2 + 2)), "reason": "forced"}
                    log_cli(f"ðŸ“Š Confidence check FORCED ready (override={_force_ready}, recaps={recap_attempts})")
                else:
                    check = check_confidence_ready(S.title, topic, subs_done, tp_list, u_hist)
                    log_cli(f"ðŸ“Š Confidence check: ready={check['ready']}, count={check['qa_count']}, reason={check['reason']}")

                if not check["ready"]:
                    with state_lock:
                        S.phase = Phase.TEACH_SUBTOPIC
                        S.recap_attempts = recap_attempts + 1
                        subtopic_recap = subs_done[-1] if subs_done else (S.subtopics[0] if S.subtopics else topic)

                    recap_msg = (
                        f"Let me do a quick recap of {subtopic_recap} before we move to questions.",
                        f"**Recap: {subtopic_recap}**",
                        []
                    )
                    speak_chunks([recap_msg])
                    continue

                # Reset recap counter
                with state_lock:
                    S.recap_attempts = 0

                # Ready â†’ generate Q&A batch
                qa_count = check["qa_count"]
                log_cli(f"ðŸŽ¯ Generating {qa_count} Q&A questions for: {topic}")
                qs = generate_qa_questions(S.title, topic, tp_list, qa_count)

                if not qs:
                    # fallback: skip Q&A
                    with state_lock:
                        S.phase = Phase.NEXT
                    speak_chunks([("Great job covering this topic! Let's move forward.", "", [])])
                    continue

                with state_lock:
                    S.qa_questions = qs
                    S.qa_idx       = 0
                    S.qa_correct   = 0
                    S.qa_wrong_items = []
                    S.qa_result_log = []
                    S.last_question = ""
                    S.phase = Phase.QA

                # Q&A intro â€” board is emitted via speak_chunks intro chunk (no separate emit needed)
                prior_mention = ""
                if S.prior_known_indices:
                    known_names = [S.topics[i] for i in S.prior_known_indices if i < len(S.topics)]
                    if known_names:
                        prior_mention = f" I may also touch on {', '.join(known_names[:2])} as well."

                reset_board_tracker()  # fresh board for Q&A

                intro = (
                    f"Excellent work on {topic}! Let's have a quick Q&A â€” {qa_count} questions.{prior_mention} "
                    f"Mix of multiple choice and direct answer. Here we go!",
                    f"**Q&A â€” {topic}** ({qa_count} questions)",
                    [{"type": "glow", "target": f"Q&A â€” {topic}", "duration_ms": 900, "delay_ms": 0}]
                )
                speak_chunks([intro])
                # Ask first question immediately
                q0 = S.qa_questions[0]
                board_q = format_qa_for_board(0, q0)
                speech_q = format_qa_for_speech(q0)
                with state_lock:
                    S.last_question         = q0["question"]
                    S.last_question_answer  = q0["answer"]
                    S.last_question_concept = q0.get("concept", "")
                    S.last_question_type    = q0.get("type", "simple")
                    _total_qa = len(S.qa_questions)
                emit_qa_board(0, q0, _total_qa)   # render MCQ inside board
                emit_qa_card(0, q0, _total_qa)    # keep legacy card
                speak_chunks([(speech_q, board_q, [])])
                continue

            # â”€â”€ QA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.QA:
                if _looks_like_qa_answer(user_txt) and intent in ("reject_or_skip", "unknown", "smalltalk"):
                    intent = "answer_qa"   # force-classify as answer

                if _is_quiz_resume_signal(user_txt, intent) and not _looks_like_qa_answer(user_txt):
                    with state_lock:
                        cur_q = S.qa_questions[S.qa_idx] if S.qa_idx < len(S.qa_questions) else None
                        q_idx = S.qa_idx
                        total_q = len(S.qa_questions)
                    if cur_q:
                        board_q = format_qa_for_board(q_idx, cur_q)
                        speech_q = format_qa_for_speech(cur_q)
                        emit_qa_board(q_idx, cur_q, total_q)
                        emit_qa_card(q_idx, cur_q, total_q)
                        speak_chunks([("Alright, let's continue with this question.", "", []), (speech_q, board_q, [])])
                    else:
                        with state_lock:
                            S.phase = Phase.NEXT
                        speak_chunks([("That quiz item is already done. Let's continue.", "", [])])
                    continue

                if intent == "reject_or_skip":
                    # Fix 1: Skip THIS question only â€” not the whole quiz
                    with state_lock:
                        S.qa_wrong_items.append(S.qa_questions[S.qa_idx]) if S.qa_idx < len(S.qa_questions) else None
                        S.qa_result_log.append({
                            "question": S.qa_questions[S.qa_idx].get("question", "") if S.qa_idx < len(S.qa_questions) else "",
                            "correct": False,
                            "answer": S.qa_questions[S.qa_idx].get("answer", "") if S.qa_idx < len(S.qa_questions) else "",
                            "student": "[skipped]",
                        })
                        S.qa_idx += 1
                        _more = S.qa_idx < len(S.qa_questions)
                        _next_q = S.qa_questions[S.qa_idx] if _more else None
                        _total_qa_skip = len(S.qa_questions)

                    if _more and _next_q:
                        with state_lock:
                            S.last_question        = _next_q["question"]
                            S.last_question_answer = _next_q["answer"]
                            S.last_question_concept= _next_q.get("concept", "")
                            S.last_question_type   = _next_q.get("type", "simple")
                        board_q  = format_qa_for_board(S.qa_idx, _next_q)
                        speech_q = format_qa_for_speech(_next_q)
                        emit_qa_board(S.qa_idx, _next_q, _total_qa_skip)
                        emit_qa_card(S.qa_idx, _next_q, _total_qa_skip)
                        speak_chunks([("No problem, moving to the next one.", "", []), (speech_q, board_q, [])])
                    else:
                        # All questions done (some skipped) â€” go to scoring
                        with state_lock:
                            S.phase = Phase.NEXT
                            S.last_question = ""
                        speak_chunks([("Alright, that's the end of the quiz. Let's move on.", "", [])])
                    continue

                if intent == "ask_question":
                    # Save which question we're on so we re-ask it after answering
                    with state_lock:
                        S.qa_interrupted_idx = S.qa_idx
                        cur_interrupted_q = S.qa_questions[S.qa_idx] if S.qa_idx < len(S.qa_questions) else None

                    sys_q = "You are a kind tutor mid-quiz. Answer ONLY the specific question asked in 2 sentences. End with 'Now, back to the question.'"
                    ctx_q = f'Quiz question being asked: "{S.last_question}". Student interrupts and asks: "{user_txt}"'
                    ans = emotion_complete_once(sys_q, ctx_q, temperature=0.4, max_tokens=120).strip()

                    # Re-ask the same question after answering
                    if cur_interrupted_q:
                        board_q  = format_qa_for_board(S.qa_interrupted_idx, cur_interrupted_q)
                        speech_q = format_qa_for_speech(cur_interrupted_q)
                        with state_lock:
                            _total_reask = len(S.qa_questions)
                        emit_qa_board(S.qa_interrupted_idx, cur_interrupted_q, _total_reask)
                        speak_chunks([(ans, "", []), (speech_q, board_q, [])])
                    else:
                        speak_chunks([(ans, "", [])])
                    continue

                # Evaluate the answer
                with state_lock:
                    cur_q   = S.qa_questions[S.qa_idx] if S.qa_idx < len(S.qa_questions) else None
                    tp_list = list(S.taught_points)

                if not cur_q:
                    with state_lock:
                        S.phase = Phase.NEXT
                    speak_chunks([("That's all the questions! Well done.", "", [])])
                    continue

                ev = evaluate_qa_answer(cur_q, user_txt, tp_list)
                feedback = ev["feedback"]
                correct  = ev["correct"]
                if emotion_engine is not None:
                    try:
                        emotion_engine.record_qa_result(
                            correct=bool(correct),
                            partial=bool(ev.get("partial", False)),
                            timestamp=time.time(),
                        )
                    except Exception as exc:
                        log_cli(f"Emotion QA result skipped: {exc}")

                with state_lock:
                    if correct:
                        S.qa_correct += 1
                    else:
                        S.qa_wrong_items.append(cur_q)
                    S.qa_idx += 1
                    _cur_idx = S.qa_idx - 1
                    _total_qa = len(S.qa_questions)
                    # Log for summary card
                    S.qa_result_log.append({
                        "question": cur_q.get("question", ""),
                        "correct": correct,
                        "answer": str(cur_q.get("answer", "")),
                        "student": user_txt,
                    })

                # Show result: in-board highlight + feedback
                emit_qa_board_result(correct, str(cur_q.get("answer", "")), feedback)
                emit_qa_result(_cur_idx, correct, str(cur_q.get("answer", "")), feedback)
                log_cli(f"{'âœ…' if correct else 'âŒ'} Q{_cur_idx+1}: {feedback}")
                speak_chunks([(feedback, "", [])])

                # Check if more questions
                with state_lock:
                    more_qs = S.qa_idx < len(S.qa_questions)

                if more_qs:
                    with state_lock:
                        next_q = S.qa_questions[S.qa_idx]
                        S.last_question         = next_q["question"]
                        S.last_question_answer  = next_q["answer"]
                        S.last_question_concept = next_q.get("concept", "")
                        S.last_question_type    = next_q.get("type", "simple")
                        q_idx = S.qa_idx
                        _total_qa2 = len(S.qa_questions)

                    board_q  = format_qa_for_board(q_idx, next_q)
                    speech_q = format_qa_for_speech(next_q)
                    emit_qa_board(q_idx, next_q, _total_qa2)   # render inside board
                    emit_qa_card(q_idx, next_q, _total_qa2)    # keep legacy card too
                    speak_chunks([(speech_q, board_q, [])])
                    continue

                # All questions done â†’ score
                with state_lock:
                    total   = len(S.qa_questions)
                    correct_count = S.qa_correct
                    wrong_items   = list(S.qa_wrong_items)
                    topic   = S.topics[S.topic_idx] if S.topics else "Topic"

                score_pct = correct_count / total if total else 0
                log_cli(f"ðŸ“Š Q&A Score: {correct_count}/{total} = {score_pct:.0%}")

                if score_pct >= 0.85:
                    # PASS
                    with state_lock:
                        # Save this topic's taught points for cross-topic memory
                        S.all_taught[topic] = list(S.taught_points)
                        S.phase = Phase.NEXT
                        S.last_question = ""

                    pass_msg = (
                        f"Outstanding! You got {correct_count} out of {total} - that's {score_pct:.0%}! "
                        f"You've mastered {topic}. Let's move to the next one.",
                        f"**Score: {correct_count}/{total} - Excellent!**\n{topic} -> Complete",
                        [{"type": "glow", "target": f"Score: {correct_count}/{total} - Excellent!", "duration_ms": 1200, "delay_ms": 0}]
                    )
                    with state_lock:
                        _rlog = list(S.qa_result_log)
                    emit_qa_end(_rlog, topic, correct_count, total)
                    emit_course_progress(topic)
                    speak_chunks([pass_msg])
                else:
                    # REVIEW needed
                    with state_lock:
                        _rlog2 = list(S.qa_result_log)
                    emit_qa_end(_rlog2, topic, correct_count, total)
                    with state_lock:
                        S.qa_review_idx = 0
                        S.qa_retry_questions = []
                        S.qa_retry_idx = 0
                        S.qa_review_phase = "RETEACH"
                        S.phase = Phase.QA_REVIEW
                        S.last_question = ""
                        # Keep result_log â€” don't clear, used for final summary

                    wrong_concepts = [q.get("concept", "that concept") for q in wrong_items if q.get("concept")]
                    wrong_summary  = ", ".join(set(wrong_concepts)) if wrong_concepts else "a few concepts"

                    review_msg = (
                        f"You got {correct_count} out of {total}. Let me quickly revisit {wrong_summary} "
                        f"â€” we'll go over it differently, then I'll ask a similar question.",
                        f"**Score: {correct_count}/{total}**\nLet's revisit: {wrong_summary}",
                        []
                    )
                    speak_chunks([review_msg])
                    # Highlight on board if possible
                    if wrong_summary:
                        emit_board_highlight(wrong_concepts[0] if wrong_concepts else wrong_summary)
                continue

            # â”€â”€ QA_REVIEW â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # NEW FLOW: 2 phases controlled by S.qa_review_phase
            #   "RETEACH" â†’ batch re-teach ALL wrong concepts at once, then flip to "RETRY"
            #   "RETRY"   â†’ ask ALL retry questions in batch, then move to NEXT
            if phase == Phase.QA_REVIEW:
                with state_lock:
                    wrong_items   = list(S.qa_wrong_items)
                    rev_phase     = S.qa_review_phase       # "RETEACH" or "RETRY"
                    retry_qs      = list(S.qa_retry_questions)
                    retry_idx     = S.qa_retry_idx
                    topic         = S.topics[S.topic_idx] if S.topics else "Topic"
                    tp_list       = list(S.taught_points)

                if _is_quiz_resume_signal(user_txt, intent) and not _looks_like_qa_answer(user_txt):
                    if rev_phase == "RETRY":
                        cur_rq = retry_qs[retry_idx] if retry_idx < len(retry_qs) else None
                        if cur_rq:
                            board_q = format_qa_for_board(retry_idx, cur_rq)
                            speech_q = format_qa_for_speech(cur_rq)
                            emit_qa_board(retry_idx, cur_rq, len(retry_qs))
                            emit_qa_card(retry_idx, cur_rq, len(retry_qs))
                            speak_chunks([("Okay, let's continue with the review question.", "", []), (speech_q, board_q, [])])
                            continue
                    else:
                        speak_chunks([("I'm still revisiting the missed concept. Stay with me for a moment, then I'll ask the next similar question.", "", [])])
                        continue

                # â”€â”€ Handle ask_question / skip in review â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if intent == "ask_question":
                    sys_q = "You are a kind tutor in a review session. Answer briefly in 2 sentences."
                    ans = emotion_complete_once(sys_q, f'Student asks: "{user_txt}"', temperature=0.4, max_tokens=100).strip()
                    speak_chunks([(ans, "", [])])
                    continue

                # Issue 4: Student wants to skip review and move on
                _skip_signals = ("move on", "next topic", "skip", "continue", "next", "go ahead", "proceed")
                if any(sig in user_txt.lower() for sig in _skip_signals) and rev_phase == "RETEACH":
                    with state_lock:
                        S.all_taught[topic] = list(S.taught_points)
                        S.phase = Phase.NEXT
                        S.last_question = ""
                    wrong_concepts = [q.get("concept", "that concept") for q in wrong_items if q.get("concept")]
                    wrong_summary  = ", ".join(set(wrong_concepts)) if wrong_concepts else "those concepts"
                    speak_chunks([(
                        f"No problem! Just keep in mind the areas we flagged: {wrong_summary}. "
                        f"You can always revisit them. Let's move forward!",
                        f"**Review skipped**\nWatch out for: {wrong_summary}",
                        []
                    )])
                    continue

                # â”€â”€ RETEACH phase: teach ALL wrong concepts in one batch â”€â”€
                if rev_phase == "RETEACH":
                    if not wrong_items:
                        with state_lock:
                            S.phase = Phase.NEXT
                        continue

                    # Build all reteach chunks + generate all retry questions
                    all_reteach_chunks = []
                    new_retry_qs = []
                    for wq in wrong_items:
                        concept = wq.get("concept", "the concept")
                        log_cli(f"ðŸ“– Re-teaching concept: {concept}")
                        reteach_cks = reteach_concept_chunks(S.title, topic, concept, tp_list)
                        for sp, bd, fx, _ in reteach_cks:
                            all_reteach_chunks.append((sp, bd, fx))
                        # Generate retry question while teaching
                        retry_q = generate_similar_question(wq, tp_list)
                        if retry_q:
                            new_retry_qs.append(retry_q)

                    with state_lock:
                        S.qa_retry_questions = new_retry_qs
                        S.qa_retry_idx = 0
                        S.qa_review_phase = "RETRY"

                    # Flip phase announcement + all reteach content in one speak
                    wrong_concepts = [q.get("concept", "") for q in wrong_items if q.get("concept")]
                    wrong_summary  = ", ".join(set(wrong_concepts)) if wrong_concepts else "those concepts"
                    intro = (
                        f"Let me go over the concepts you missed: {wrong_summary}.",
                        f"**Review: {wrong_summary}**",
                        []
                    )
                    bridge = (
                        "Alright, that covers the review. Now let me ask you a couple of similar questions to make sure it clicked.",
                        "**Quick Check â€” Similar Questions**",
                        []
                    )
                    speak_chunks([intro] + all_reteach_chunks + [bridge])

                    # Immediately ask first retry question (no wait for student input)
                    if new_retry_qs:
                        first_rq = new_retry_qs[0]
                        with state_lock:
                            S.last_question         = first_rq["question"]
                            S.last_question_answer  = first_rq["answer"]
                            S.last_question_concept = first_rq.get("concept", "")
                            S.last_question_type    = first_rq.get("type", "simple")
                            S.qa_retry_idx          = 0

                        board_q  = format_qa_for_board(0, first_rq)
                        speech_q = format_qa_for_speech(first_rq)
                        emit_qa_board(0, first_rq, len(new_retry_qs))
                        emit_qa_card(0, first_rq, len(new_retry_qs))
                        speak_chunks([(speech_q, board_q, [])])
                    continue

                # â”€â”€ RETRY phase: evaluate retry answers in sequence â”€â”€
                if rev_phase == "RETRY":
                    cur_rq = retry_qs[retry_idx] if retry_idx < len(retry_qs) else None

                    if not cur_rq:
                        # All retry questions done â†’ move to NEXT
                        with state_lock:
                            S.all_taught[topic] = list(S.taught_points)
                            S.phase = Phase.NEXT
                            S.last_question = ""
                        emit_qa_end()
                        emit_course_progress(topic)
                        nxt = S.topics[S.topic_idx + 1] if (S.topic_idx + 1 < len(S.topics)) else None
                        if nxt:
                            speak_chunks([(
                                f"Great effort on the review! You're ready for {nxt}. Shall we continue?",
                                f"Next -> **{nxt}**",
                                [{"type": "glow", "target": nxt, "duration_ms": 900, "delay_ms": 0}]
                            )])
                        else:
                            speak_chunks([("Excellent work! You've completed the course. Well done!", "**Course Complete!**", [])])
                            with state_lock:
                                S = TutorState(mode=Mode.QUICK, phase=Phase.IDLE)
                        continue

                    # Evaluate this retry answer
                    ev_r      = evaluate_qa_answer(cur_rq, user_txt, tp_list)
                    correct_r = ev_r["correct"]
                    feedback_r = ev_r["feedback"]
                    if emotion_engine is not None:
                        try:
                            emotion_engine.record_qa_result(
                                correct=bool(correct_r),
                                partial=bool(ev_r.get("partial", False)),
                                timestamp=time.time(),
                            )
                        except Exception as exc:
                            log_cli(f"Emotion QA review skipped: {exc}")

                    emit_qa_board_result(correct_r, str(cur_rq.get("answer", "")), feedback_r)
                    emit_qa_result(retry_idx, correct_r, str(cur_rq.get("answer", "")), feedback_r)
                    log_cli(f"{'âœ…' if correct_r else 'âŒ'} Retry Q{retry_idx+1}: {feedback_r}")
                    speak_chunks([(feedback_r, "", [])])

                    with state_lock:
                        S.qa_retry_idx += 1
                        next_retry_idx = S.qa_retry_idx

                    # More retry questions?
                    if next_retry_idx < len(retry_qs):
                        next_rq = retry_qs[next_retry_idx]
                        with state_lock:
                            S.last_question        = next_rq["question"]
                            S.last_question_answer = next_rq["answer"]
                            S.last_question_concept = next_rq.get("concept", "")
                            S.last_question_type    = next_rq.get("type", "simple")

                        board_q  = format_qa_for_board(next_retry_idx, next_rq)
                        speech_q = format_qa_for_speech(next_rq)
                        emit_qa_board(next_retry_idx, next_rq, len(retry_qs))
                        emit_qa_card(next_retry_idx, next_rq, len(retry_qs))
                        speak_chunks([(speech_q, board_q, [])])
                    else:
                        # All retry done â†’ move to NEXT
                        with state_lock:
                            S.all_taught[topic] = list(S.taught_points)
                            S.phase = Phase.NEXT
                            S.last_question = ""
                        emit_qa_end()
                        emit_course_progress(topic)
                        nxt = S.topics[S.topic_idx + 1] if (S.topic_idx + 1 < len(S.topics)) else None
                        if nxt:
                            speak_chunks([(
                                f"Well done on the review! Ready for {nxt}? Let's go.",
                                f"Next -> **{nxt}**",
                                [{"type": "glow", "target": nxt, "duration_ms": 900, "delay_ms": 0}]
                            )])
                        else:
                            speak_chunks([("Excellent! You've finished the course. Outstanding work!", "**Course Complete!**", [])])
                            with state_lock:
                                S = TutorState(mode=Mode.QUICK, phase=Phase.IDLE)
                    continue

            # â”€â”€ NEXT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if phase == Phase.NEXT:
                if intent == "reject_or_skip":
                    speak_chunks([("Alright. Take your time and come back when ready.", "", [])])
                    continue

                if intent == "ask_question":
                    with state_lock:
                        S.phase = Phase.TEACH_SUBTOPIC
                    speak_chunks([("Sure. Let me answer that first.", "", [])])
                    continue

                # Confirm or any positive â†’ advance to next topic
                with state_lock:
                    if S.topic_idx + 1 < len(S.topics):
                        S.topic_idx += 1
                        # Skip prior-known topics
                        while S.topic_idx in S.prior_known_indices and S.topic_idx + 1 < len(S.topics):
                            log_cli(f"â© Skipping prior-known topic: {S.topics[S.topic_idx]}")
                            S.topic_idx += 1

                        topic = S.topics[S.topic_idx]
                        S.phase         = Phase.SUBTOPIC_PLAN
                        S.last_topic    = topic
                        S.taught_points = []
                        S.subtopics     = []
                        S.subtopic_idx  = 0
                        S.subtopics_done = []
                        S.qa_questions  = []
                        S.qa_idx        = 0
                        S.qa_correct    = 0
                        S.qa_wrong_items = []
                        S.topic_completed = False
                        S.subtopic_interrupted = False
                    else:
                        topic = None
                        S = TutorState(mode=Mode.QUICK, phase=Phase.IDLE)

                if not topic:
                    speak_chunks([(
                        "Congratulations! You've completed the entire course. Excellent work!",
                        "**Course Complete!**",
                        [{"type": "glow", "target": "Course Complete!", "duration_ms": 1500, "delay_ms": 0}]
                    )])
                    continue

                subj = guess_subject_type(S.title + " " + topic)
                with state_lock:
                    all_topics_next = list(S.topics)
                    pref_next = S.last_user_goal or ""
                subs = plan_subtopics(S.title, topic, subj, all_topics=all_topics_next, student_pref=pref_next)
                with state_lock:
                    S.subtopics    = subs
                    S.subtopic_idx = 0
                    S.phase        = Phase.TEACH_SUBTOPIC

                # Check if topic in prior-known (student said they know it but we still reference)
                prior_note = ""
                with state_lock:
                    if S.topic_idx in S.prior_known_indices:
                        prior_note = f" You mentioned you know {topic} already â€” I'll go quick but cover it properly."

                # â”€â”€ Show previous topic notes on board before starting new topic â”€â”€
                with state_lock:
                    prev_topic_name = S.topics[S.topic_idx - 1] if S.topic_idx > 0 else ""
                    prev_taught     = list(S.all_taught.get(prev_topic_name, []))

                if prev_topic_name and prev_taught:
                    # Build compact notes card for previous topic
                    notes_lines = [f"**{prev_topic_name} - Notes**", ""]
                    for pt in prev_taught[:8]:    # max 8 key points
                        notes_lines.append(f"  - {pt}")
                    notes_lines += ["", "-" * 38, ""]
                    notes_board = "\n".join(notes_lines)
                    reset_board_tracker()
                    socketio.emit("board_text", {
                        "text": notes_board,
                        "append": False,   # fresh board with notes
                        "mode": "type", "cps": 32, "scroll": False,
                    })
                    set_visual_render_text(notes_board, append=False)
                    time.sleep(0.8)   # brief pause so student sees notes

                subs_board_next = "\n".join([f"- {s}" for s in subs])
                intro_chunks = [(
                    f"Next up: {topic}!{prior_note}",
                    f"**{topic}**\n{subs_board_next}",
                    [{"type": "glow", "target": topic, "duration_ms": 900, "delay_ms": 0}]
                )]

                # Also reference prior topic knowledge if relevant
                with state_lock:
                    all_t = dict(S.all_taught)
                if all_t:
                    last_t_name = list(all_t.keys())[-1]
                    prior_ref = f" We'll build on what we learned in {last_t_name} as well."
                    intro_chunks[0] = (
                        intro_chunks[0][0] + prior_ref,
                        intro_chunks[0][1],
                        intro_chunks[0][2]
                    )

                chunks_llm = teach_subtopic_chunks(S.title, topic, subs[0], subj, "Start teaching this new topic now.")
                chunks_out = []
                with state_lock:
                    for (sp, bd, fx, meta) in chunks_llm:
                        if isinstance(meta, dict):
                            for tp in (meta.get("taught_points") or []):
                                tp = str(tp).strip()
                                if tp and tp not in S.taught_points:
                                    S.taught_points.append(tp)
                            if meta.get("subtopic_complete") is True:
                                if S.subtopic_idx + 1 < len(S.subtopics):
                                    S.subtopics_done.append(S.subtopics[S.subtopic_idx])
                                    S.phase = Phase.SUBTOPIC_WRAP
                                else:
                                    S.phase = Phase.CONFIDENCE_CHECK
                        chunks_out.append((sp, bd, fx))

                speak_chunks(intro_chunks + chunks_out)
                continue

        except Exception as e:
            log_cli(f"âŒ Pipeline Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if turn_web_token is not None:
                try:
                    TURN_WEB_CONTEXT.reset(turn_web_token)
                except Exception:
                    pass
            try:
                TURN_LEARNING_MODE.reset(turn_mode_token)
            except Exception:
                pass
            tutor_busy = False
            set_status("LISTENING")
            stt_instance.resume()
            log_cli("ðŸŽ¤ Listening...")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Routes + Socket Events (UNCHANGED)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/session-report/download")
def download_session_report():
    session_id = str(request.args.get("session_id") or (last_session_artifacts.get("session_id") if last_session_artifacts else current_session_id))
    fmt = str(request.args.get("format") or "pdf").strip().lower()
    if fmt == "image":
        fmt = "png"
    manifest = None
    if last_session_artifacts and last_session_artifacts.get("session_id") == session_id:
        manifest = dict(last_session_artifacts)
    if manifest is None:
        manifest = _load_session_manifest(session_id)
    if not manifest:
        return jsonify({"ok": False, "error": "Session report not found."}), 404
    file_map = dict(manifest.get("files") or {})
    target = Path(str(file_map.get(fmt) or ""))
    if not target.exists():
        return jsonify({"ok": False, "error": f"Report format '{fmt}' is unavailable."}), 404
    return send_file(target, as_attachment=True, download_name=target.name)


@app.route("/session-material/download")
def download_session_material():
    session_id = str(request.args.get("session_id") or (last_session_artifacts.get("session_id") if last_session_artifacts else current_session_id))
    manifest = None
    if last_session_artifacts and last_session_artifacts.get("session_id") == session_id:
        manifest = dict(last_session_artifacts)
    if manifest is None:
        manifest = _load_session_manifest(session_id)
    if not manifest:
        return jsonify({"ok": False, "error": "Session material not found."}), 404
    target = Path(str((manifest.get("files") or {}).get("material_pdf") or ""))
    if not target.exists():
        return jsonify({"ok": False, "error": "Material export is unavailable."}), 404
    return send_file(target, as_attachment=True, download_name=target.name)


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file uploaded."}), 400

        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"ok": False, "error": "Empty filename."}), 400

        raw = uploaded.read()
        mime = uploaded.content_type or "application/octet-stream"
        chunks = process_upload(raw, uploaded.filename, mime)

        stored = 0
        total_chars = 0
        usable_chunks = 0
        upload_issue = ""
        for content, label in chunks:
            if not (content or "").strip():
                continue
            issue = extract_upload_issue(content)
            if issue:
                upload_issue = upload_issue or issue
            else:
                usable_chunks += 1
            stored += 1
            total_chars += len(content)

        if stored == 0:
            return jsonify({"ok": False, "error": "No readable content found in file."}), 400

        indexed_doc = remember_upload_document(chunks, uploaded.filename, mime)
        if indexed_doc is None:
            return jsonify({"ok": False, "error": "No readable content found in file."}), 400

        log_cli(f"Uploaded file: {uploaded.filename} ({stored} chunk(s), {total_chars} chars)")
        response_payload = {
            "filename": uploaded.filename,
            "chunks": stored,
            "chars": total_chars,
            "indexed_sections": len(indexed_doc.get("chunks") or []),
        }
        response_body = {
            "ok": True,
            "filename": uploaded.filename,
            "chunks": stored,
            "indexed_sections": len(indexed_doc.get("chunks") or []),
        }

        if upload_issue and usable_chunks == 0:
            warning = f'Uploaded "{uploaded.filename}", but vision analysis is unavailable: {upload_issue}'
            response_payload["warning"] = warning
            response_body["warning"] = warning
            log_cli(f"Upload warning: {warning}")
        return jsonify(response_body)
    except Exception as exc:
        log_cli(f"Upload failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500


@socketio.on("connect")
def handle_connect():
    log_cli("âœ… System Ready. Speak or type!")
    _auto_fallback_voice("humanised_tts_unreachable", force_check=True, emit=True)
    emit_learning_mode_state()
    emit_course_progress()
    socketio.emit("llm_model_status", _current_llm_model_payload())
    if emotion_engine is not None:
        emotion_engine.emit_status()
        emotion_engine.emit_monitor_update()
    # Fix Issue 3: Restore break state to reconnected client
    with emp_lock:
        _brk = EMP.break_active
        _brk_start = EMP.last_break_start
    if _brk:
        elapsed = int(time.time() - _brk_start) if _brk_start else 0
        remaining = max(0, 300 - elapsed)
        socketio.emit("break_restore", {
            "active": True,
            "elapsed_secs": elapsed,
            "remaining_secs": remaining,
        })


@socketio.on("set_learning_mode")
def handle_set_learning_mode(data):
    global ACTIVE_LEARNING_MODE
    requested = normalize_learning_mode((data or {}).get("mode"))
    current = get_active_learning_mode()

    if tutor_busy and requested != current:
        socketio.emit("learning_mode_busy", {"mode": current.value})
        return

    with learning_mode_lock:
        ACTIVE_LEARNING_MODE = requested

    restore_visual_board(requested, force_clear_if_empty=True)
    emit_learning_mode_state()
    schedule_runtime_state_save()

    if requested != current:
        log_cli(f"Mode switched to {learning_mode_label(requested)}")


@socketio.on("start_new_session")
def handle_start_new_session(data=None):
    requested = normalize_learning_mode((data or {}).get("mode")) if isinstance(data, dict) and (data or {}).get("mode") else get_active_learning_mode()
    if tutor_busy:
        socketio.emit("learning_mode_busy", {"mode": get_active_learning_mode().value})
        return
    reset_session_runtime(requested)


@socketio.on("request_board_restore")
def handle_board_restore(data=None):
    """Replay the currently selected mode's board content to a reconnecting browser."""
    requested = normalize_learning_mode((data or {}).get("mode")) if isinstance(data, dict) else get_active_learning_mode()
    restored = restore_visual_board(requested, force_clear_if_empty=True)
    emit_course_progress(mode=requested)
    if restored:
        size = len(get_visual_state(requested).live_board_text or "")
        log_cli(f"Board restored to reconnected client ({size} chars) [{requested.value}]")
        return
        log_cli(f"ðŸ”„ Board restored to reconnected client ({len(_live_board_text)} chars)")


@socketio.on("request_student_state")
def handle_request_student_state(data=None):
    emit_student_state()


@socketio.on("student_login")
def handle_student_login(data=None):
    payload = dict(data or {})
    profile, created = create_student(
        payload.get("name") or "Student",
        payload.get("age") or 20,
        payload.get("email") or "",
        payload.get("student_id") or "",
    )
    activated = _set_active_student(profile, restore_runtime=not created, fresh_session=created)
    welcome_email_result = {"sent": False, "reason": "not_attempted"}
    if created and str(activated.get("email") or "").strip():
        welcome_email_result = send_welcome_email(
            activated.get("email") or "",
            activated.get("name") or "Student",
            activated.get("student_id") or "",
        )
    socketio.emit("student_login_result", {
        "ok": True,
        "created": created,
        "student": _build_student_state_payload(activated).get("active"),
        "welcome_email": welcome_email_result,
    })
    log_cli(f"Student login: {activated.get('name')} [{activated.get('student_id')}] ({'new' if created else 'existing'})")
    if created and str(activated.get("email") or "").strip():
        if welcome_email_result.get("sent"):
            log_cli(f"Welcome email sent to {activated.get('email')}")
        else:
            log_cli(f"Welcome email failed for {activated.get('email')}: {welcome_email_result.get('reason')}")


@socketio.on("switch_student")
def handle_switch_student(data=None):
    if tutor_busy:
        socketio.emit("student_switch_result", {"ok": False, "error": "Tutor is busy. Try switching after this turn."})
        return
    student_id = str((data or {}).get("student_id") or "").strip()
    profile = get_student(student_id)
    if not profile:
        socketio.emit("student_switch_result", {"ok": False, "error": "Student account not found."})
        return
    activated = _set_active_student(profile, restore_runtime=True, fresh_session=False)
    socketio.emit("student_switch_result", {"ok": True, "student": _build_student_state_payload(activated).get("active")})
    log_cli(f"Switched student: {activated.get('name')} [{activated.get('student_id')}]")


@socketio.on("student_logout")
def handle_student_logout(data=None):
    global ACTIVE_STUDENT, RUNTIME_STATE_PATH, last_session_artifacts, last_material_text
    guest = {
        "student_id": "guest_default",
        "name": "Student",
        "age": 20,
        "email": "",
        "folder_name": "guest_default",
    }
    clear_last_active_student()
    with active_student_lock:
        ACTIVE_STUDENT = dict(guest)
    RUNTIME_STATE_PATH = student_runtime_state_path(guest)
    last_session_artifacts = {}
    last_material_text = ""
    _sync_student_to_empathy(guest)
    reset_session_runtime(get_active_learning_mode())
    emit_student_state(guest)
    socketio.emit("student_switch_result", {"ok": True, "student": _build_student_state_payload(guest).get("active")})
    log_cli("Student logged out to guest profile")


@socketio.on("update_empathy_profile")
def handle_empathy_profile(data):
    """Frontend sends real GPS location + student profile every 5 min."""
    global _last_profile_log_signature
    data = dict(data or {})
    profile_changed = False
    profile = get_active_student_profile()
    with emp_lock:
        if data.get("name"):
            EMP.student_name = str(data["name"]).strip()
            if profile.get("name") != EMP.student_name:
                profile["name"] = EMP.student_name
                profile_changed = True
        if data.get("age"):
            try:
                EMP.student_age = int(data["age"])
                if int(profile.get("age") or 0) != EMP.student_age:
                    profile["age"] = EMP.student_age
                    profile_changed = True
            except Exception:
                pass
        if data.get("real_lat") is not None:
            EMP.real_lat = float(data["real_lat"])
        if data.get("real_lon") is not None:
            EMP.real_lon = float(data["real_lon"])
        if data.get("city"):
            EMP.real_city = str(data["city"])
            if profile.get("city") != EMP.real_city:
                profile["city"] = EMP.real_city
                profile_changed = True
        if data.get("tz"):
            EMP.real_tz = str(data["tz"])
            if profile.get("tz") != EMP.real_tz:
                profile["tz"] = EMP.real_tz
                profile_changed = True
        if data.get("temp") is not None:
            try:
                EMP.weather_temp = float(data["temp"])
            except Exception:
                pass
        if data.get("weather"):
            EMP.weather_label = str(data["weather"])
    if data.get("email") is not None:
        email = str(data.get("email") or "").strip()
        if profile.get("email", "") != email:
            profile["email"] = email
            profile_changed = True
    if profile_changed and profile.get("student_id") != "guest_default":
        updated = save_profile(profile, make_last_active=True)
        with active_student_lock:
            ACTIVE_STUDENT.update(updated)
        emit_student_state(updated)
    name_text = str(data.get("name") or profile.get("name") or "Student").strip() or "Student"
    city_text = str(data.get("city") or getattr(EMP, "real_city", "") or "?").strip() or "?"
    temp_value = data.get("temp")
    if temp_value is None:
        temp_value = getattr(EMP, "weather_temp", None)
    try:
        temp_text = f"{float(temp_value):.0f}C" if temp_value is not None else "?"
    except Exception:
        temp_text = "?"
    email_text = str(data.get("email") or profile.get("email", "") or "").strip()
    log_signature = (profile.get("student_id"), name_text, city_text, temp_text, email_text)
    if log_signature != _last_profile_log_signature:
        _last_profile_log_signature = log_signature
        log_cli(f"Profile updated: {name_text} | {city_text} | {temp_text}")


@socketio.on("session_started")
def handle_session_started():
    """Called when tutor first starts teaching (not just page load)."""
    with emp_lock:
        if EMP.session_start_ts == 0.0:
            EMP.session_start_ts = time.time()
    log_cli("â±ï¸ Study session timer started")


@socketio.on("break_started")
def handle_break_started():
    with emp_lock:
        EMP.break_active     = True
        EMP.last_break_start = time.time()
        EMP.break_pause_topic    = S.last_topic
        EMP.break_pause_subtopic = S.subtopics[S.subtopic_idx] if S.subtopics and S.subtopic_idx < len(S.subtopics) else ""
    log_cli("â˜• Break started")


@socketio.on("break_ended")
def handle_break_ended(data):
    """Frontend sends actual break duration when student stops timer."""
    actual_secs = float(data.get("actual_secs", 300))
    with emp_lock:
        EMP.break_active    = False
        EMP.last_break_secs = actual_secs
        EMP.breaks_today   += 1
        EMP.last_break_start = 0.0

    # Generate LLM comment based on actual duration (natural, not hard-coded)
    recommended = 300  # 5 mins
    extra_secs  = max(0, actual_secs - recommended)
    with emp_lock:
        name    = EMP.student_name
        topic   = EMP.break_pause_topic
        subtopic= EMP.break_pause_subtopic

    if extra_secs > 30:
        extra_str = f"{int(extra_secs//60)}m {int(extra_secs%60)}s"
        ctx = (f"Student {name} just returned from break. "
               f"They were supposed to take 5 minutes but took {int(actual_secs//60)}m {int(actual_secs%60)}s "
               f"({extra_str} extra). Gently acknowledge this â€” no lecture, keep it warm and brief. "
               f"Then say you're resuming from where we left off: {topic} â€” {subtopic}. "
               f"Max 2 sentences. Natural, human tutor voice.")
    else:
        ctx = (f"Student {name} just returned from their 5-minute break. "
               f"Welcome them back warmly in 1 sentence. "
               f"Then say you're continuing with {topic} â€” {subtopic}.")
    try:
        welcome_back = global_llm.complete_once(
            "You are a warm, human expert tutor.",
            ctx, temperature=0.75, max_tokens=80
        ).strip()
    except Exception:
        welcome_back = f"Welcome back, {name}! Let's continue from where we left off."

    if _stt_ref:
        try: _stt_ref.pause()
        except Exception: pass
    speak_chunks([(welcome_back, "", [])], instruct=EMPATHY_INSTRUCT_WELCOME)
    if _stt_ref:
        try: _stt_ref.resume()
        except Exception: pass
    log_cli(f"â˜• Break ended â€” {int(actual_secs)}s actual | resume: {topic} â†’ {subtopic}")


@socketio.on("qa_option_click")
def handle_qa_option_click(data):
    """Student clicked an MCQ option on the board â€” treat as their answer."""
    letter = (data.get("letter") or "").strip().upper()
    if not letter:
        return
    log_cli(f"ðŸ–±ï¸  MCQ click: {letter}")
    # Show as user speech in UI chip
    socketio.emit("user_speech", {"text": f"Option {letter}"})
    # Signal frontend to stop mic immediately (prevent ambient speech race)
    socketio.emit("pause_mic_for_click", {})
    # Push to global text_q â€” same queue STT uses
    enqueue_student_text(letter, source="qa_click", web_search=False)


@socketio.on("hand_raise")
def handle_hand_raise():
    global tutor_busy, _last_interrupt_ts
    now = time.time()
    if emotion_engine is not None:
        try:
            emotion_engine.handle_support_event("hand_raise", timestamp=now)
        except Exception as exc:
            log_cli(f"Emotion hand raise skipped: {exc}")
    if tutor_busy and (now - _last_interrupt_ts) > 2.5:
        _last_interrupt_ts = now
        log_cli("âœ‹ Hand Raised! Interrupting...")
        interrupt_event.set()
    elif tutor_busy:
        log_cli("âœ‹ Hand Raise ignored (debounce)")


@socketio.on("text_submit")
def handle_text_submit(data):
    global tutor_busy
    txt = (data.get("text") or "").strip()
    wants_web_search = bool((data or {}).get("web_search", False))
    if not txt:
        return
    if tutor_busy:
        interrupt_event.set()
        if enqueue_student_text(txt, source="typed", web_search=wants_web_search):
            suffix = " + web search" if wants_web_search else ""
            log_cli(f"âŒ¨ï¸ Queued typed input ({len(txt)} chars{suffix}) while tutor busy â€” interrupt requested.")
        else:
            log_cli("âŒ Typed input dropped because the queue is full.")
        return
    show_user_speech(txt)
    if enqueue_student_text(txt, source="typed", web_search=wants_web_search):
        suffix = " + web search" if wants_web_search else ""
        log_cli(f"âŒ¨ï¸ Queued typed input ({len(txt)} chars{suffix})")
    else:
        log_cli("âŒ Typed input dropped because the queue is full.")


@socketio.on("set_camera")
def handle_set_camera(data):
    enabled = bool(data.get("enabled", False))
    if cam_monitor is None:
        socketio.emit("camera_ack", {"enabled": False})
        return

    if enabled:
        cam_monitor.start()
        log_cli("Camera monitoring: ON")
    else:
        cam_monitor.stop()
        log_cli("Camera monitoring: OFF")

    socketio.emit("camera_ack", {"enabled": enabled})


@socketio.on("camera_frame")
def handle_camera_frame(data):
    if cam_monitor and data.get("frame"):
        cam_monitor.push_frame_from_client(data["frame"])
    if emotion_engine is not None and data.get("frame"):
        try:
            emotion_engine.handle_camera_frame(data["frame"], timestamp=time.time())
        except Exception as exc:
            log_cli(f"Emotion camera frame skipped: {exc}")


@socketio.on("set_emotion_settings")
def handle_set_emotion_settings(data):
    if emotion_engine is None:
        socketio.emit("emotion_engine_status", {"settings": {"enabled": False, "show_monitor": False, "face_mesh_overlay": False}})
        return
    settings = emotion_engine.update_settings(
        enabled=bool(data.get("enabled", emotion_engine.get_settings().enabled)),
        show_monitor=bool(data.get("show_monitor", emotion_engine.get_settings().show_monitor)),
        face_mesh_overlay=bool(data.get("face_mesh_overlay", emotion_engine.get_settings().face_mesh_overlay)),
    )
    socketio.emit("emotion_engine_status", {"settings": settings.to_dict(), "packet": emotion_engine.latest_packet.to_dict()})


@socketio.on("request_emotion_settings")
def handle_request_emotion_settings():
    if emotion_engine is None:
        socketio.emit("emotion_engine_status", {"settings": {"enabled": False, "show_monitor": False, "face_mesh_overlay": False}})
        return
    emotion_engine.emit_status()
    emotion_engine.emit_monitor_update()


@socketio.on("clear_excuse")
def handle_clear_excuse():
    if cam_monitor:
        cam_monitor.clear_excuse()
    log_cli("Camera monitoring excuse cleared")


@socketio.on("set_mic")
def handle_set_mic(data):
    global MIC_MUTED
    muted = bool(data.get("muted", False))
    MIC_MUTED = muted
    log_cli(f"ðŸŽ™ï¸ Mic: {'MUTED' if muted else 'ON'}")
    socketio.emit("mic_ack", {"muted": muted})


@socketio.on("set_speaker")
def handle_set_speaker(data):
    global SPK_MUTED
    muted = bool(data.get("muted", False))
    SPK_MUTED = muted
    log_cli(f"ðŸ”Š Speaker: {'MUTED' if muted else 'ON'}")
    socketio.emit("speaker_ack", {"muted": muted})


@socketio.on("set_voice")
def handle_set_voice(data):
    global VOICE_ENGINE, VOICE_SPEAKER
    engine  = (data.get("engine")  or "humanised").strip().lower()
    if engine not in VOICE_DEFAULTS:
        engine = "humanised"
    speaker = (data.get("speaker") or VOICE_DEFAULTS[engine]).strip() or VOICE_DEFAULTS[engine]
    reason = ""
    if engine == "humanised":
        try:
            if not tts_backend_available(force=True):
                engine = "piper"
                speaker = DEFAULT_PIPER_SPEAKER
                reason = "humanised_tts_unreachable"
        except Exception:
            engine = "piper"
            speaker = DEFAULT_PIPER_SPEAKER
            reason = "humanised_tts_unreachable"
    with _voice_lock:
        VOICE_ENGINE  = engine
        VOICE_SPEAKER = speaker
    if reason:
        log_cli(f"⚠️ Humanised TTS unavailable ({reason}). Switching to local Piper.")
        socketio.emit("voice_ack", _voice_ack_payload(engine, speaker, fallback=True, reason=reason))
        return
    log_cli(f"ðŸ”Š Voice: {engine} â†’ {speaker}")
    socketio.emit("voice_ack", _voice_ack_payload(engine, speaker))


@socketio.on("set_llm_model")
def handle_set_llm_model(data):
    requested = _normalize_llm_model_key((data or {}).get("model"))
    if tutor_busy and requested != _normalize_llm_model_key(current_llm_model_key):
        payload = _current_llm_model_payload()
        payload["busy"] = True
        payload["message"] = "Wait for the current response to finish before changing the LLM."
        socketio.emit("llm_model_status", payload)
        return
    changed = requested != _normalize_llm_model_key(current_llm_model_key)
    apply_llm_model(requested, emit=True, log_change=changed)


@socketio.on("request_llm_settings")
def handle_request_llm_settings():
    socketio.emit("llm_model_status", _current_llm_model_payload())


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Boot (UNCHANGED)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def start_background_threads():
    global global_llm, cam_monitor, emotion_engine, _stt_ref
    apply_llm_model(current_llm_model_key, emit=False, log_change=True)
    emotion_engine = EmotionEngine(
        EmotionEngineConfig(),
        socketio=socketio,
        log_callback=log_cli,
    )
    _orig_emit_monitor_update = emotion_engine.emit_monitor_update

    def _emit_monitor_update_with_layers():
        _orig_emit_monitor_update()
        try:
            _on_emotion_monitor_data(emotion_engine.latest_monitor.to_dict())
        except Exception:
            pass

    emotion_engine.emit_monitor_update = _emit_monitor_update_with_layers

    def on_text(txt):
        if MIC_MUTED:
            return
        enqueue_student_text(txt, source="stt", web_search=False)

    def on_status(msg):
        log_cli(msg)

    stt_cfg = STTConfig(
        api_url="http://172.16.13.91:8009/stt",
        fs=16000,
        frame_ms=30,
        vad_mode=2,
        rms_thresh=50,
        end_silence_ms=900,
        min_utter_sec=0.7,
        min_speech_ratio=0.35,
        preroll_frames=12,
        latency="high",
    )

    stt = STTClient(on_text=on_text, config=stt_cfg, on_status=on_status)
    _stt_ref = stt   # Fix 2: give empathy threads access to STT pause/resume
    def on_camera_attention(attention, gender, confidence, details=None):
        effective_attention = attention
        merged_details = dict(details or {})
        if emotion_engine is not None:
            emotion_engine.handle_camera_status(attention, gender, confidence, details=details)
            try:
                monitor_data = emotion_engine.latest_monitor.to_dict()
                effective_attention = str(monitor_data.get("attention_status") or attention or "unknown")
                extras = dict(monitor_data.get("extras") or {})
                merged_details.update({
                    "attention_source": extras.get("attention_source"),
                    "pose_attention": extras.get("pose_attention_status"),
                    "scene_attention": extras.get("scene_attention_status"),
                    "monitoring_active": True,
                    "blink_count": monitor_data.get("blink_count"),
                    "blink_rate": monitor_data.get("blink_rate"),
                    "yawn_count": monitor_data.get("yawn_count"),
                    "current_eye_closure_duration": extras.get("current_eye_closure_duration"),
                    "prolonged_eye_closure_count": extras.get("prolonged_eye_closure_count"),
                })
            except Exception:
                pass
            try:
                event_map = {
                    "away": "looking_away",
                    "distracted_side": "looking_away",
                    "phone": "phone_detected",
                    "sleepy": "sleepy",
                }
                mapped_event = event_map.get(str(effective_attention or "").strip().lower())
                if mapped_event:
                    emotion_engine.handle_support_event(mapped_event, timestamp=time.time(), details=merged_details)
            except Exception:
                pass

        _remember_camera_runtime(effective_attention, gender, merged_details)
        log_analytics_attention(effective_attention, float(confidence or 0), merged_details)
        emit_layer_update(
            "observe",
            "done" if effective_attention in ("focused", "looking_down", "text_active") else "active",
            input_text=f"camera={effective_attention} conf={round(float(confidence or 0) * 100)}%",
            output_text=f"gender={gender}",
            meta={"attention": effective_attention, "confidence": float(confidence or 0)},
        )
        return {
            "attention": effective_attention,
            "gender": gender,
            "confidence": float(confidence or 0),
            "details": merged_details,
        }

    def on_camera_alert(message, attention, gender, details=None):
        log_cli(f"Camera alert [{attention}]: {message}")
        details = dict(details or {})
        if attention in {"phone", "sleepy"} and tutor_busy:
            interrupt_event.set()
        if attention == "phone":
            warning = "Please put the phone aside for a moment. We are in the middle of the session, and I will continue as soon as you are back."
            _queue_monitor_intervention("alert", attention, warning, details, instruct=EMPATHY_INSTRUCT_BREAK)
        elif attention == "sleepy":
            yawn_count = int(details.get("yawn_count") or 0)
            closure_count = int(details.get("prolonged_eye_closure_count") or 0)
            if yawn_count >= 1 or closure_count >= 2:
                warning = "You seem tired now. Take a breath, open your eyes fully, and then we will continue together."
            else:
                warning = "Are you thinking for a moment, or are you feeling tired? Re-focus when you are ready, and we will continue."
            _queue_monitor_intervention("alert", attention, warning, details, instruct=EMPATHY_INSTRUCT_BREAK)
        elif attention == "away":
            _queue_monitor_intervention("alert", attention, "I cannot see you clearly right now. Come back to the screen when you are ready.", details, instruct=EMPATHY_INSTRUCT_BREAK)

    def on_camera_return(message, gender, payload=None):
        details = dict(payload or {})
        from_attention = str(details.get("from_attention") or "")
        if from_attention in {"phone", "sleepy", "away"}:
            resume = str(details.get("resume_message") or "Okay, it looks like you're back again. Let's continue our session.")
            _queue_monitor_intervention("return", from_attention, resume, details, instruct=EMPATHY_INSTRUCT_WELCOME)

    cam_monitor = CameraMonitor(
        socketio=socketio,
        on_alert_callback=on_camera_alert,
        on_return_callback=on_camera_return,
        on_attention_callback=on_camera_attention,
    )

    t_worker = threading.Thread(target=llm_worker, args=(stt,), daemon=True)
    t_worker.start()

    stt.start()
    log_cli("âœ… STT Started in Background")
    socketio.emit("service_status", {"stt": True, "llm": True, "tts": True})
    if emotion_engine is not None:
        emotion_engine.emit_status()
        emotion_engine.emit_monitor_update()
    if _runtime_restore_note:
        log_cli(_runtime_restore_note)


def _empathy_monitor():
    """Background thread: tracks study time, offers breaks, 50-min warning."""
    BREAK_INTERVAL_SECS = 20 * 60    # offer break every 20 mins
    DAILY_CAP_SECS      = 50 * 60    # 50-min soft cap
    OFFER_COOLDOWN_SECS = 5 * 60     # don't re-offer within 5 mins

    while True:
        time.sleep(30)

        with emp_lock:
            if EMP.session_start_ts == 0.0:
                continue   # session not started yet
            if EMP.break_active:
                # Fix Issue 3: Auto-expire break after 10 minutes (handles reconnect deadlock)
                _break_elapsed = time.time() - EMP.last_break_start if EMP.last_break_start else 0
                if _break_elapsed > 600:   # 10 minutes
                    EMP.break_active = False
                    EMP.last_break_secs = _break_elapsed
                    EMP.breaks_today += 1
                    EMP.last_break_start = 0.0
                    log_cli(f"â° Break auto-expired after {int(_break_elapsed)}s (no break_ended received)")
                    socketio.emit("break_auto_ended", {})
                else:
                    continue   # student is on break

        if not _has_break_worthy_study_activity():
            continue

        # Accumulate study time (only when tutor is running)
        with emp_lock:
            EMP.total_study_secs += 30   # 30s tick (we sleep 30s)

        with emp_lock:
            study_secs   = EMP.total_study_secs
            last_offer   = EMP.break_offered_at
            warned_50    = EMP.session_warned_50

        now = time.time()

        # â”€â”€ 50-min soft cap warning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if study_secs >= DAILY_CAP_SECS and not warned_50:
            with emp_lock:
                EMP.session_warned_50 = True
                name = EMP.student_name

            # Wait until tutor is not speaking, then suggest wrap-up
            if not tutor_busy:
                ctx = (f"Student {name} has been studying for 50 minutes today â€” the recommended daily cap. "
                       f"Gently suggest wrapping up for the day, but make it clear they can continue if they really want. "
                       f"Ask if they'd like to continue or call it a day. 2 sentences max. Natural tutor voice.")
                try:
                    msg = global_llm.complete_once(
                        "You are a caring human tutor.",
                        ctx, temperature=0.75, max_tokens=70
                    ).strip()
                except Exception:
                    msg = f"Hey {name}, you've been studying hard for 50 minutes! Want to call it a day, or shall we keep going?"
                socketio.emit("empathy_nudge", {
                    "type": "cap_warning", "message": msg, "study_secs": int(study_secs)
                })
                # Fix 2: pause STT so mic doesn't pick up tutor speech
                if _stt_ref:
                    try: _stt_ref.pause()
                    except Exception: pass
                speak_chunks([(msg, "", [])], instruct=EMPATHY_INSTRUCT_CAP)
                if _stt_ref:
                    try: _stt_ref.resume()
                    except Exception: pass
            continue

        # â”€â”€ 20-min break suggestion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if study_secs > 0 and (study_secs % BREAK_INTERVAL_SECS) < 30:
            # Only offer once per interval â€” check cooldown
            if (now - last_offer) < OFFER_COOLDOWN_SECS:
                continue

            with emp_lock:
                EMP.break_offered_at = now
                name = EMP.student_name

            # Only suggest break when tutor is NOT mid-speech
            # If tutor is busy, let current chunk finish â€” wait up to 60s
            waited = 0
            while tutor_busy and waited < 60:
                time.sleep(2)
                waited += 2
            if tutor_busy:
                continue

            ctx = (f"Student {name} has been studying for {int(study_secs//60)} minutes. "
                   f"It's time for a 5-minute break. Suggest it warmly. "
                   f"Ask if they'd like to take a break now or finish the current section first. "
                   f"2 sentences max. Natural, caring tutor voice.")
            try:
                msg = global_llm.complete_once(
                    "You are a caring human tutor.",
                    ctx, temperature=0.75, max_tokens=70
                ).strip()
            except Exception:
                msg = f"Hey {name}, great work! You've been going for {int(study_secs//60)} minutes â€” how about a quick 5-minute break?"

            socketio.emit("empathy_nudge", {
                "type": "break_offer", "message": msg, "study_secs": int(study_secs)
            })
            # Fix 2: pause STT before speaking empathy message
            if _stt_ref:
                try: _stt_ref.pause()
                except Exception: pass
            speak_chunks([(msg, "", [])], instruct=EMPATHY_INSTRUCT_BREAK)
            if _stt_ref:
                try: _stt_ref.resume()
                except Exception: pass


# â”€â”€ Late-night empathy check (fires once per session if student is late) â”€â”€
def _night_check():
    """Fires once when student is detected studying past 10 PM."""
    _checked = False
    while True:
        time.sleep(60)
        if _checked:
            time.sleep(3600)
            continue
        with emp_lock:
            tz  = EMP.real_tz
            name= EMP.student_name
            if EMP.session_start_ts == 0.0:
                continue
        try:
            import zoneinfo
            local_h = datetime.datetime.now(zoneinfo.ZoneInfo(tz)).hour
        except Exception:
            local_h = datetime.datetime.now().hour

        if local_h >= 22 or local_h < 4:
            with state_lock:
                active_phase = S.phase
            if active_phase in (Phase.QA, Phase.QA_REVIEW):
                continue
            _checked = True
            if tutor_busy:
                # Wait for chunk to finish
                while tutor_busy:
                    time.sleep(2)
            ctx = (f"Student {name} is studying at {local_h}:00 {'AM' if local_h < 12 else 'PM'} â€” very late. "
                   f"Gently check in: ask when they need to wake up tomorrow, mention health matters. "
                   f"Don't force them to stop. Max 2 sentences. Warm, human tone.")
            try:
                msg = global_llm.complete_once(
                    "You are a caring human tutor.", ctx, temperature=0.8, max_tokens=70
                ).strip()
            except Exception:
                msg = f"Hey {name}, it's getting late â€” are you sure you don't need to be up early tomorrow?"
            if _stt_ref:
                try: _stt_ref.pause()
                except Exception: pass
            speak_chunks([(msg, "", [])], instruct=EMPATHY_INSTRUCT_LATE)
            if _stt_ref:
                try: _stt_ref.resume()
                except Exception: pass



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Hook: Log emotion data when monitor updates
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _on_emotion_monitor_data(packet_dict: dict):
    """Called whenever emotion engine emits a full packet â€” log analytics + update layers."""
    try:
        valence    = float(packet_dict.get("valence", 0.0) or 0.0)
        arousal    = float(packet_dict.get("arousal", 0.0) or 0.0)
        dominant   = str(packet_dict.get("raw_face_emotion", "") or "neutral")
        engagement = str(packet_dict.get("engagement_label", "") or "")
        log_analytics_emotion(valence, arousal, dominant, engagement)
        # Update state_tracker layer with fused emotion data
        emit_layer_update("state_tracker", "done",
            input_text=f"face={dominant} | eng={engagement}",
            output_text=f"V={valence:+.2f} A={arousal:+.2f}",
            meta={"valence": valence, "arousal": arousal})
        # Update empathy_policy layer
        policy  = packet_dict.get("policy") or {}
        action  = policy.get("pedagogical_action", "normal_explain")
        empathy = policy.get("empathy_type", "none")
        emit_layer_update("empathy_policy", "done",
            input_text=f"V={valence:+.2f} A={arousal:+.2f}",
            output_text=f"action={action} | empathy={empathy}",
            meta={"action": action, "empathy": empathy})
        emit_layer_update("pedagogy", "done",
            input_text=f"action={action}",
            output_text=action.replace("_", " ").title(),
            meta={"action": action})
    except Exception:
        pass


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Session Report â€” end_session + disconnect
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@socketio.on("end_session")
def handle_end_session(data=None):
    """Assembles session analytics, generates exports, and emits session_report."""
    global last_session_artifacts, last_material_text

    forced = bool((data or {}).get("forced", False))
    suppress_emit = bool((data or {}).get("suppress_emit", False))
    with _analytics_lock:
        start_ts  = _session_analytics["start_time"] or time.time()
        end_ts    = time.time()
        turns     = list(_session_analytics["turns"])
        attn_log  = list(_session_analytics["attention_log"])
        emo_log   = list(_session_analytics["emotion_log"])
        topics    = list(_session_analytics["topics"])
        qa_res    = list(_session_analytics["qa_results"])
        tone_log  = list(_session_analytics["tutor_tone_log"])
        empathy_events = list(_session_analytics["empathy_events"])

    duration_secs = max(0, end_ts - start_ts)
    duration_min  = round(duration_secs / 60, 1)

    engagement_timeline = []
    if emo_log:
        bucket_size = max(30, duration_secs / 20)
        bucket_start = start_ts
        while bucket_start < end_ts:
            bucket_end = bucket_start + bucket_size
            bucket_entries = [e for e in emo_log if bucket_start <= e["ts"] < bucket_end]
            ts_label = round((bucket_start - start_ts) / 60, 1)
            if bucket_entries:
                avg_v = sum(e["valence"] for e in bucket_entries) / len(bucket_entries)
                avg_a = sum(e["arousal"] for e in bucket_entries) / len(bucket_entries)
                eng_score = round(min(1.0, max(0.0, (avg_v + 1) / 2 * 0.6 + avg_a * 0.4)), 2)
                confusion_score = round(min(1.0, max(0.0, max(0.0, -avg_v) * 0.45 + avg_a * 0.55)), 2)
            else:
                eng_score = None
                confusion_score = None
            engagement_timeline.append({"t": ts_label, "score": eng_score, "confusion": confusion_score})
            bucket_start = bucket_end

    attn_counts: Dict[str, int] = {}
    for entry in attn_log:
        s = entry.get("state", "unknown")
        attn_counts[s] = attn_counts.get(s, 0) + 1
    total_attn = sum(attn_counts.values()) or 1
    attention_dist = {k: round(v / total_attn * 100, 1) for k, v in attn_counts.items()}

    topic_turns: Dict[str, int] = {}
    for t in turns:
        tp = t.get("topic", "General") or "General"
        topic_turns[tp] = topic_turns.get(tp, 0) + 1

    emo_counts: Dict[str, int] = {}
    for e in emo_log:
        dom = e.get("dominant", "neutral") or "neutral"
        emo_counts[dom] = emo_counts.get(dom, 0) + 1

    correct_qa = sum(1 for q in qa_res if q.get("correct"))
    total_qa   = len(qa_res)
    qa_score   = round(correct_qa / total_qa * 100) if total_qa else None

    valid_ms   = [t["llm_ms"] for t in turns if t.get("llm_ms", 0) > 0]
    avg_llm_ms = round(sum(valid_ms) / len(valid_ms)) if valid_ms else 0

    focused_count = sum(v for k, v in attn_counts.items() if k in ("focused", "looking_down"))
    attn_score    = round(focused_count / (sum(attn_counts.values()) or 1) * 100)

    eng_label_map = {"high_engagement": 85, "engaged": 70, "neutral": 50,
                     "low_engagement": 30, "disengaged": 15, "distressed": 20}
    valid_emo    = [e for e in emo_log if e.get("engagement")]
    avg_engagement = 50
    if valid_emo:
        scores = [eng_label_map.get(e["engagement"], 50) for e in valid_emo]
        avg_engagement = round(sum(scores) / len(scores))

    start_str = datetime.datetime.fromtimestamp(start_ts).strftime("%H:%M")
    end_str   = datetime.datetime.fromtimestamp(end_ts).strftime("%H:%M")
    date_str  = datetime.datetime.fromtimestamp(start_ts).strftime("%d %b %Y")

    with state_lock:
        report_mode = get_active_learning_mode().value
        report_title = _infer_report_title(report_mode, S.title, S.last_topic, topics, turns)
        board_material = _current_board_material()
    active_student = get_active_student_profile()
    report_student_name = str(active_student.get("name") or "Student")
    report_student_email = str(active_student.get("email") or "")
    tone_timeline = _build_tone_timeline(start_ts, tone_log)
    tone_graph_timeline = _build_tone_graph_timeline(start_ts, tone_log, attn_log)
    empathy_enabled = bool(emotion_engine is not None and getattr(emotion_engine, "enabled", True))
    empathy_summary = _build_empathy_summary(engagement_timeline, tone_timeline, empathy_enabled, attn_score)
    uploaded_materials = retrieve_upload_context(query=report_title, max_chars=1800, max_chunks=4)
    if not _session_has_meaningful_activity(turns, qa_res, topics, board_material, uploaded_materials, duration_secs):
        last_session_artifacts = {}
        last_material_text = ""
        log_cli(f"Session end skipped: not enough study activity [{current_session_id}]")
        if not suppress_emit:
            socketio.emit("session_report_skipped", {
                "session_id": current_session_id,
                "mode": report_mode,
                "reason": "No meaningful study activity yet.",
            })
        _reset_analytics_only()
        save_runtime_state()
        return

    save_runtime_state()

    report = {
        "forced":              forced,
        "session_id":          current_session_id,
        "student_id":          active_student.get("student_id"),
        "mode":                report_mode,
        "title":               report_title,
        "student_name":        report_student_name,
        "student_email":       report_student_email,
        "date":                date_str,
        "start_time":          start_str,
        "end_time":            end_str,
        "duration_min":        duration_min,
        "turn_count":          len(turns),
        "topics":              topics or ["General"],
        "topic_turns":         topic_turns,
        "attention_dist":      attention_dist,
        "emotion_dist":        emo_counts,
        "engagement_timeline": engagement_timeline,
        "avg_engagement":      avg_engagement,
        "attn_score":          attn_score,
        "qa_score":            qa_score,
        "qa_correct":          correct_qa,
        "qa_total":            total_qa,
        "avg_llm_ms":          avg_llm_ms,
        "tone_timeline":       tone_timeline,
        "tone_graph_timeline": tone_graph_timeline,
        "empathy_events":      empathy_events[:16],
        "empathy_summary":     empathy_summary,
        "board_material":      board_material,
        "uploaded_materials":  uploaded_materials,
        "performance_radar": {
            "Engagement":  avg_engagement,
            "Attention":   attn_score,
            "Q&A Score":   qa_score or 0,
            "Consistency": min(100, max(0, 100 - len(
                [t for t in turns if t.get("intent") == "off_topic"]) * 10)),
            "Pace":        min(100, max(0, 100 - max(0, avg_llm_ms - 2000) // 20)),
        },
    }

    material_text = build_material_text(report)
    last_material_text = material_text
    artifact_manifest = {
        "session_id": current_session_id,
        "student_id": active_student.get("student_id"),
        "files": {},
        "created_at": time.time(),
    }
    email_result = {"sent": False, "reason": "missing_recipient" if not report_student_email else "not_attempted"}
    try:
        session_dir = _student_session_dir(current_session_id)
        artifacts = generate_session_artifacts(report, material_text, session_dir)
        artifact_manifest["files"] = artifacts
        _save_session_manifest(current_session_id, artifact_manifest, active_student)
        last_session_artifacts = dict(artifact_manifest)
        if report_student_email:
            email_result = send_session_email(report_student_email, report, artifacts.get("png", ""), artifacts.get("material_pdf", ""))
    except Exception as exc:
        log_cli(f"Session artifact generation failed: {exc}")
        last_session_artifacts = dict(artifact_manifest)
        email_result = {"sent": False, "reason": str(exc)}

    report["downloads"] = _report_download_urls(current_session_id)
    report["email_status"] = email_result

    if not suppress_emit:
        socketio.emit("session_report", report)
    log_cli(f"📊 Session report emitted: {len(turns)} turns, {duration_min}min, topics={topics}")

    _reset_analytics_only()


@socketio.on("open_recent_session")
def handle_open_recent_session(data=None):
    global last_session_artifacts, last_material_text, ACTIVE_LEARNING_MODE, current_session_id, current_session_started_at, session_logger
    if tutor_busy:
        socketio.emit("recent_session_error", {"error": "Tutor is busy. Open the recent after this turn."})
        return

    session_id = str((data or {}).get("session_id") or "").strip()
    if not session_id:
        socketio.emit("recent_session_error", {"error": "Missing session id."})
        return

    try:
        save_runtime_state()
    except Exception:
        pass

    profile = get_active_student_profile()
    snapshot_path = _runtime_snapshot_path(session_id, profile)
    manifest, report = _load_session_report_payload(session_id, profile)
    restored = False

    if snapshot_path.exists():
        try:
            snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            _apply_runtime_state_payload(snapshot_data)
            restored = True
        except Exception as exc:
            log_cli(f"Recent restore snapshot skipped for {session_id}: {exc}")

    if not restored and report:
        target_mode = normalize_learning_mode(report.get("mode"))
        with learning_mode_lock:
            ACTIVE_LEARNING_MODE = target_mode
        current_session_id = session_id
        current_session_started_at = time.time()
        session_logger = SessionLogger(str(_student_session_dir(session_id) / "live_notes"))
        with state_lock:
            S.title = str(report.get("title") or "")
            S.last_topic = str((report.get("topics") or [""])[0] or "")
            if target_mode == LearningMode.COURSE:
                S.topics = list(report.get("topics") or [])
                S.all_taught = list(report.get("topics") or [])
                S.topic_idx = min(len(S.topics), len(S.all_taught))
        set_visual_render_text(str(report.get("board_material") or ""), append=False, mode=target_mode)
        restored = True

    if not restored:
        socketio.emit("recent_session_error", {"error": "Could not restore that recent session."})
        return

    last_session_artifacts = dict(manifest or {"session_id": session_id, "files": {}})
    if report:
        with state_lock:
            if not S.title:
                S.title = str(report.get("title") or "")
            if not S.last_topic:
                S.last_topic = str((report.get("topics") or [""])[0] or "")
            if get_active_learning_mode() == LearningMode.COURSE and not S.topics:
                S.topics = list(report.get("topics") or [])
                S.all_taught = list(report.get("topics") or [])
        last_material_text = build_material_text(report)
    restore_visual_board(get_active_learning_mode(), force_clear_if_empty=True)
    emit_learning_mode_state()
    emit_course_progress()
    emit_student_state()
    schedule_runtime_state_save()

    restored_title = _normalize_session_title((report or {}).get("title") or S.title or "", "") or session_id
    payload = {
        "ok": True,
        "session_id": session_id,
        "mode": get_active_learning_mode().value,
        "title": restored_title,
        "has_report": bool(report),
    }
    socketio.emit("recent_session_opened", payload)
    emit_session_meta(get_active_learning_mode(), title_override=payload["title"])
    log_cli(f"Recent session restored: {payload['title']} [{session_id}]")


@socketio.on("disconnect")
def handle_disconnect():
    """Persist runtime on disconnect without ending the session."""
    log_cli("🔌 Client disconnected")
    try:
        save_runtime_state()
    except Exception:
        pass


_shutdown_report_flushed = False


def _flush_session_on_shutdown() -> None:
    global _shutdown_report_flushed
    if _shutdown_report_flushed:
        return
    _shutdown_report_flushed = True
    try:
        handle_end_session({"forced": True, "suppress_emit": True})
    except Exception:
        pass


atexit.register(_flush_session_on_shutdown)


if __name__ == "__main__":
    load_runtime_state()
    _mon_t = threading.Thread(target=_empathy_monitor, daemon=True, name="EmpathyMonitor")
    _mon_t.start()
    _night_t = threading.Thread(target=_night_check, daemon=True, name="NightCheck")
    _night_t.start()
    log_cli("🧠 Empathy monitor started")
    for startup_url in _startup_urls(5000):
        log_cli(f"Open tutor at {startup_url}")
    start_background_threads()
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)

