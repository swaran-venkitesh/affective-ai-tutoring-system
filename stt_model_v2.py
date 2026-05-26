"""
stt_model_v2.py
Laptop-side STT client:
Mic -> VAD endpointing -> send utterance WAV to STT server -> returns text

This file is meant to be IMPORTED by main.py (or any other app).
So: no "run forever" code at import time.

Requirements:
  pip install sounddevice numpy requests scipy webrtcvad-wheels
"""

from __future__ import annotations

import io
import time
import queue
import threading
import collections
from dataclasses import dataclass
from typing import Callable, Optional

import requests
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.io.wavfile import write as wav_write


# -----------------------------
# CONFIG DEFAULTS (edit here anytime)
# -----------------------------
DEFAULT_STT_API_URL = "http://172.16.13.91:8009/stt"
DEFAULT_FS = 16000


# -----------------------------
# Helper filters
# -----------------------------
def mostly_english_ascii(text: str) -> bool:
    """Reject outputs dominated by non-ASCII scripts."""
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii <= max(2, int(0.05 * len(text)))


@dataclass
class STTConfig:
    api_url: str = DEFAULT_STT_API_URL
    fs: int = DEFAULT_FS

    frame_ms: int = 30
    vad_mode: int = 2
    rms_thresh: int = 50

    # NOTE: breath_gap_ms not used explicitly in this version.
    # Endpointing is controlled by end_silence_ms.
    breath_gap_ms: int = 2
    end_silence_ms: int = 0

    min_utter_sec: float = 0.9
    min_speech_ratio: float = 0.40

    preroll_frames: int = 12
    latency: str = "low"   # sounddevice latency setting: "low"/"high"


class STTClient:
    """
    Threaded STT client (safe design):
      - Audio callback only enqueues audio frames
      - Worker thread does VAD + endpointing, then calls STT API
      - When text is ready, calls on_text(text)

    Half-duplex support:
      - pause(): stops accepting audio immediately
      - resume(): re-enables mic
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        config: Optional[STTConfig] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.on_text = on_text
        self.on_status = on_status or (lambda msg: None)
        self.cfg = config or STTConfig()

        self.frame_samples = int(self.cfg.fs * self.cfg.frame_ms / 1000)
        self.end_silence_frames = int(self.cfg.end_silence_ms / self.cfg.frame_ms)

        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=250)
        self._stop_event = threading.Event()

        # ✅ half-duplex controls
        self._paused = threading.Event()       # when set -> mic ignored, worker idle
        self._reset_state = threading.Event()  # ask worker to reset VAD state + drain

        self._worker_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._last_status_signature = ""
        self._last_status_at = 0.0

        self._sess = requests.Session()
        self._vad = webrtcvad.Vad(self.cfg.vad_mode)

    # -----------------------------
    # Half-duplex controls
    # -----------------------------
    def pause(self):
        """Stop accepting mic audio immediately (Tutor speaking)."""
        self._paused.set()
        self._reset_state.set()

        # drain queued frames so no stale audio triggers later
        try:
            while True:
                self._audio_q.get_nowait()
        except Exception:
            pass

        self.on_status("🔇 STT paused (Tutor speaking).")

    def resume(self):
        """Re-enable mic (User can speak)."""
        self._reset_state.set()   # reset once more on resume
        self._paused.clear()
        self.on_status("🎤 STT resumed (Your turn).")

    def is_paused(self) -> bool:
        return self._paused.is_set()

    # -----------------------------
    # Internals
    # -----------------------------
    def _rms_ok(self, x_i16: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(x_i16.astype(np.float32) ** 2))
        return rms > self.cfg.rms_thresh

    def _wav_bytes_from_int16(self, audio_i16: np.ndarray) -> io.BytesIO:
        memfile = io.BytesIO()
        wav_write(memfile, self.cfg.fs, audio_i16)
        memfile.seek(0)
        return memfile

    def _audio_callback(self, indata, frames, time_info, status):
        # ✅ hard mute during tutor response
        if self._paused.is_set():
            return

        if status:
            status_text = str(status)
            now = time.time()
            if status_text != self._last_status_signature or (now - self._last_status_at) >= 8.0:
                self._last_status_signature = status_text
                self._last_status_at = now
                self.on_status(f"⚠️ Audio Status: {status_text}")

        x = np.int16(indata[:, 0] * 32767)

        # callback must be FAST: just enqueue
        try:
            self._audio_q.put_nowait(x)
        except queue.Full:
            pass

    def _worker(self):
        ring = collections.deque(maxlen=self.cfg.preroll_frames)

        triggered = False
        silence_count = 0
        utter_frames = []

        speech_frames = 0
        total_frames = 0

        while not self._stop_event.is_set():

            # ✅ reset state when asked (pause/resume or external)
            if self._reset_state.is_set():
                ring.clear()
                triggered = False
                silence_count = 0
                utter_frames = []
                speech_frames = 0
                total_frames = 0

                # drain queue
                try:
                    while True:
                        self._audio_q.get_nowait()
                except Exception:
                    pass

                self._reset_state.clear()

            # ✅ if paused, do nothing
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            try:
                x = self._audio_q.get(timeout=0.25)
            except queue.Empty:
                continue

            # VAD decision (RMS gate + webrtcvad)
            is_speech = False
            if self._rms_ok(x):
                try:
                    is_speech = self._vad.is_speech(x.tobytes(), self.cfg.fs)
                except Exception:
                    is_speech = False

            if not triggered:
                ring.append(x)
                if is_speech:
                    triggered = True
                    utter_frames = list(ring)  # include preroll
                    ring.clear()

                    silence_count = 0
                    speech_frames = 1
                    total_frames = 1
            else:
                utter_frames.append(x)
                total_frames += 1

                if is_speech:
                    speech_frames += 1
                    silence_count = 0
                else:
                    silence_count += 1

                # End utterance only after END_SILENCE_MS
                if silence_count >= self.end_silence_frames:
                    audio = np.concatenate(utter_frames) if utter_frames else None
                    if audio is not None:
                        dur = len(audio) / self.cfg.fs
                        speech_ratio = speech_frames / max(1, total_frames)

                        if dur >= self.cfg.min_utter_sec and speech_ratio >= self.cfg.min_speech_ratio:
                            try:
                                memfile = self._wav_bytes_from_int16(audio)
                                r = self._sess.post(
                                    self.cfg.api_url,
                                    files={"file": ("audio.wav", memfile, "audio/wav")},
                                    timeout=15,
                                )
                                if r.ok:
                                    txt = (r.json().get("text", "") or "").strip()
                                    if txt and mostly_english_ascii(txt):
                                        self.on_text(txt)
                                else:
                                    self.on_status(f"❌ STT API Error: {r.status_code} {r.text[:200]}")
                            except Exception as e:
                                self.on_status(f"❌ STT API Request Failed: {e}")

                    # reset state after utterance
                    triggered = False
                    utter_frames = []
                    silence_count = 0
                    ring.clear()
                    speech_frames = 0
                    total_frames = 0

    def start(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._paused.clear()
        self._reset_state.set()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.cfg.fs,
            blocksize=self.frame_samples,
            dtype="float32",
            callback=self._audio_callback,
            latency=self.cfg.latency,
        )
        self._stream.start()
        self.on_status("🎤 STTClient started (VAD endpointing).")

    def stop(self):
        self._stop_event.set()
        self._paused.set()
        self._reset_state.set()

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

        try:
            if self._worker_thread:
                self._worker_thread.join(timeout=1.0)
        except Exception:
            pass
        self._worker_thread = None

        self.on_status("🛑 STTClient stopped.")
