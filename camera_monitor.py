from __future__ import annotations

import base64
import collections
import io
import json
import random
import re
import threading
import time
from typing import Callable, Deque, Optional

VLM_BASE = "http://172.16.13.91:8023/v1"
VLM_MODEL = "qwen2.5-vl-7b"

CHECK_INTERVAL_SEC = 2
EXCUSE_TIMEOUT_SEC = 120
JPEG_QUALITY = 70
MAX_WIDTH = 640
MIN_CONF = 0.45
ALERT_COOLDOWN_SEC = 18
WINDOW_SIZE = 3

ALERT_THRESHOLDS = {
    "phone": 3,
    "distracted_side": 4,
    "sleepy": 2,
    "away": 3,
}

NON_ALERT_STATES = {"focused", "looking_down", "unknown", "multiple_people"}

MONITOR_SYSTEM = (
    "You are an AI student attention monitor. "
    "You analyze webcam frames and return only valid JSON."
)

MONITOR_PROMPT = """\
Analyze this webcam image carefully.

Return only this JSON object:
{
  "student_gender": "male"|"female"|"unknown",
  "attention": "focused"|"looking_down"|"phone"|"distracted_side"|"sleepy"|"away"|"multiple_people"|"unknown",
  "confidence": <0.0 to 1.0>,
  "person_count": <integer>,
  "phone_present": true|false,
  "phone_activity": "none"|"on_call"|"typing"|"scrolling"|"browsing"|"video"|"holding"|"unknown",
  "object_label": "<short noun or empty string>",
  "observed_action": "<short visible action phrase or empty string>",
  "visible_text": "<short visible text or empty string>",
  "attention_comment": "<one short sentence grounded in the image>",
  "multiple_people": true|false,
  "alert": true|false,
  "alert_message": "<short message or empty string>"
}

Rules:
- "looking_down" is normal behavior. Never treat it as alertable.
- "phone" means the student is clearly looking at a phone.
- "away" means no student is visible.
- "multiple_people" is informational only.
- If a phone is not clearly visible, set phone_present=false and phone_activity="none".
- Use person_count=0 if nobody is visible, otherwise count the visible people.
- If the image is too dark or unclear, use "unknown" with low confidence.
"""

_MSG_POOLS = {
    "phone": {
        "male": [
            "If something is pulling your attention to the phone, please put it aside and come back to this step.",
            "Take a quick moment to finish with the phone, then rejoin the lesson here.",
        ],
        "female": [
            "If something is pulling your attention to the phone, please put it aside and come back to this step.",
            "Take a quick moment to finish with the phone, then rejoin the lesson here.",
        ],
        "unknown": [
            "If a phone is distracting you, please put it aside and return to the lesson.",
            "Finish that quick phone check, then come back and we will continue from here.",
        ],
    },
    "distracted_side": {
        "male": [
            "Let's come back to the lesson and focus on this next step.",
            "When you're ready, bring your attention back to the screen.",
        ],
        "female": [
            "Let's come back to the lesson and focus on this next step.",
            "When you're ready, bring your attention back to the screen.",
        ],
        "unknown": [
            "Please look back at the screen so we can continue.",
            "Let's refocus on the lesson and pick up the next step.",
        ],
    },
    "sleepy": {
        "male": [
            "You look a bit drowsy. Take a breath or a short reset, then come back to this step.",
            "If you're getting sleepy, pause for a moment and return when you're ready to focus.",
        ],
        "female": [
            "You look a bit drowsy. Take a breath or a short reset, then come back to this step.",
            "If you're getting sleepy, pause for a moment and return when you're ready to focus.",
        ],
        "unknown": [
            "You seem drowsy. Take a brief reset if needed, then come back to the lesson.",
            "If you're sleepy, pause for a short moment and return ready for the next step.",
        ],
    },
    "away": {
        "male": [
            "I can't see you right now. Come back to the screen when you're ready to continue.",
            "When you're back at the screen, we'll continue from this step.",
        ],
        "female": [
            "I can't see you right now. Come back to the screen when you're ready to continue.",
            "When you're back at the screen, we'll continue from this step.",
        ],
        "unknown": [
            "I can't see you right now. Please come back when you're ready.",
            "Come back to the screen and we'll continue from here.",
        ],
    },
}

_RETURN_MSGS = {
    "male": ["Welcome back. Let's continue.", "Good, let's pick up from this step."],
    "female": ["Welcome back. Let's continue.", "Good, let's pick up from this step."],
    "unknown": ["Welcome back. Let's continue.", "Good, let's pick up from this step."],
}

_DRINK_RE = re.compile(r"\b(drink|drinking|sip|sipping|water|bottle|glass|cup|mug)\b", re.I)
_PHONE_RE = re.compile(r"\b(phone|mobile|cell phone|smartphone)\b", re.I)


def _pick_message(attention: str, gender: str) -> str:
    group = gender if gender in ("male", "female") else "unknown"
    pool = _MSG_POOLS.get(attention, {}).get(group) or _MSG_POOLS.get(attention, {}).get("unknown", [])
    return random.choice(pool) if pool else "Please pay attention to the lesson."


def _pick_return(gender: str) -> str:
    group = gender if gender in ("male", "female") else "unknown"
    return random.choice(_RETURN_MSGS.get(group, _RETURN_MSGS["unknown"]))


def _details_text(details: Optional[dict]) -> str:
    info = dict(details or {})
    parts = [
        str(info.get("observed_action") or ""),
        str(info.get("object_label") or ""),
        str(info.get("attention_comment") or ""),
        str(info.get("visible_text") or ""),
    ]
    return " ".join(part.strip() for part in parts if str(part or "").strip()).lower()


def _looks_like_drink_break(details: Optional[dict]) -> bool:
    blob = _details_text(details)
    return bool(blob and _DRINK_RE.search(blob))


def _looks_like_real_phone(details: Optional[dict]) -> bool:
    info = dict(details or {})
    if bool(info.get("phone_present")):
        return True
    blob = _details_text(info)
    return bool(blob and _PHONE_RE.search(blob))


def _coerce_attention(attention: str, confidence: float, details: Optional[dict]) -> tuple[str, float, dict]:
    info = dict(details or {})
    person_count = int(info.get("person_count") or 0)
    if person_count > 0 and _looks_like_drink_break(info):
        info["activity_context"] = "drinking"
        if attention in {"away", "sleepy", "distracted_side"}:
            info["attention_comment"] = "Student briefly looks down while drinking water."
            return "looking_down", max(confidence, 0.72), info
    if attention == "phone" and not _looks_like_real_phone(info):
        return "unknown", min(confidence, 0.45), info
    return attention, confidence, info


class CameraMonitor:
    def __init__(
        self,
        socketio,
        tts_callback: Optional[Callable[[str], None]] = None,
        on_alert_callback: Optional[Callable[[str, str, str, Optional[dict]], None]] = None,
        on_return_callback: Optional[Callable[[str, str, Optional[dict]], None]] = None,
        on_attention_callback: Optional[Callable[[str, str, float, Optional[dict]], None]] = None,
    ):
        self.socketio = socketio
        self.tts_callback = tts_callback
        self.on_alert_callback = on_alert_callback
        self.on_return_callback = on_return_callback
        self.on_attention_callback = on_attention_callback

        self._running = False
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._window: Deque[tuple[str, str, float]] = collections.deque(maxlen=WINDOW_SIZE)
        self._bad_streak = 0
        self._current_bad_state = ""
        self._was_alerting = False
        self._last_alert_time: dict[str, float] = {}
        self._alert_counts: dict[str, int] = {}

        self._excused = False
        self._excuse_reason = ""
        self._excuse_expires = 0.0
        self._excuse_lock = threading.Lock()

        self.tutor_speaking = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._latest_frame = None
        self._window.clear()
        self._bad_streak = 0
        self._current_bad_state = ""
        threading.Thread(target=self._analyze_loop, daemon=True, name="CameraAnalyze").start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        with self._frame_lock:
            self._latest_frame = None
        self._window.clear()
        self._bad_streak = 0
        self._current_bad_state = ""
        self._was_alerting = False
        self._alert_counts.clear()

    def push_frame_from_client(self, jpeg_b64: str):
        if not self._running:
            return
        try:
            raw = base64.b64decode(jpeg_b64)
        except Exception:
            return
        with self._frame_lock:
            self._latest_frame = raw

    def set_excuse(self, reason: str):
        with self._excuse_lock:
            self._excused = True
            self._excuse_reason = reason
            self._excuse_expires = time.time() + EXCUSE_TIMEOUT_SEC
            self._window.clear()
            self._bad_streak = 0
            self._current_bad_state = ""
        self.socketio.emit("camera_excuse", {
            "active": True,
            "reason": reason,
            "seconds": EXCUSE_TIMEOUT_SEC,
        })

    def clear_excuse(self):
        with self._excuse_lock:
            self._excused = False
            self._excuse_reason = ""
        self.socketio.emit("camera_excuse", {"active": False, "reason": ""})

    def _is_excused(self) -> bool:
        with self._excuse_lock:
            if not self._excused:
                return False
            if time.time() > self._excuse_expires:
                self._excused = False
                self.socketio.emit("camera_excuse", {"active": False, "reason": ""})
                return False
            return True

    def _analyze_loop(self):
        last_check = 0.0
        while not self._stop_event.is_set():
            now = time.time()
            if now - last_check < CHECK_INTERVAL_SEC:
                time.sleep(0.25)
                continue
            last_check = now

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                continue

            result = self._query_vlm(frame)
            if result is None:
                continue

            attention = str(result.get("attention") or "unknown")
            gender = str(result.get("student_gender") or "unknown")
            confidence = float(result.get("confidence") or 0.0)
            phone_present = bool(result.get("phone_present", False))
            phone_activity = str(result.get("phone_activity") or "none")
            object_label = str(result.get("object_label") or "").strip().lower()
            if attention == "phone":
                if not phone_present:
                    attention = "unknown"
                elif phone_activity in {"holding", "unknown", "none"} and confidence < 0.78 and "phone" not in object_label:
                    attention = "unknown"
            multi = bool(result.get("multiple_people", False))
            details = {
                "person_count": int(result.get("person_count", 0) or 0),
                "phone_present": phone_present,
                "phone_activity": phone_activity,
                "object_label": str(result.get("object_label") or ""),
                "observed_action": str(result.get("observed_action") or ""),
                "visible_text": str(result.get("visible_text") or ""),
                "attention_comment": str(result.get("attention_comment") or ""),
                "multiple_people": multi,
                "vlm_runtime": f"{VLM_MODEL}@{VLM_BASE}",
            }

            if self.on_attention_callback:
                try:
                    override = self.on_attention_callback(attention, gender, confidence, details)
                    if isinstance(override, dict):
                        attention = str(override.get("attention") or attention or "unknown")
                        gender = str(override.get("gender") or gender or "unknown")
                        confidence = float(override.get("confidence") or confidence or 0.0)
                        merged_details = dict(details)
                        merged_details.update(dict(override.get("details") or {}))
                        details = merged_details
                        multi = bool(details.get("multiple_people", multi))
                except Exception:
                    pass

            attention, confidence, details = _coerce_attention(attention, confidence, details)

            self.socketio.emit("camera_status", {
                "attention": attention,
                "confidence": confidence,
                "gender": gender,
                "multi": multi,
                "details": details,
                "excused": self._is_excused(),
            })

            if self._is_excused():
                if attention == "phone" and confidence > 0.6:
                    self.clear_excuse()
                else:
                    self._window.clear()
                    self._bad_streak = 0
                    self._current_bad_state = ""
                    continue

            if attention == "unknown":
                continue

            self._window.append((attention, gender, confidence))
            if len(self._window) < max(2, WINDOW_SIZE // 2):
                continue

            dominant_state, dominant_gender = self._window_majority()

            if dominant_state == "multiple_people":
                self.socketio.emit("camera_alert", {
                    "message": "Multiple people detected near the camera.",
                    "attention": dominant_state,
                    "gender": dominant_gender,
                    "soft": True,
                })
                self._bad_streak = 0
                self._current_bad_state = ""
                continue

            if dominant_state in NON_ALERT_STATES:
                if self._was_alerting:
                    previous_state = self._current_bad_state
                    payload = {
                        "message": _pick_return(dominant_gender),
                        "from_attention": previous_state,
                        "resume_message": "Okay, it seems like you're back again. Let's continue our session.",
                    }
                    self.socketio.emit("camera_returned", payload)
                    if self.on_return_callback:
                        try:
                            self.on_return_callback(payload["resume_message"], dominant_gender, payload)
                        except Exception:
                            pass
                self._bad_streak = 0
                self._current_bad_state = ""
                self._was_alerting = False
                continue

            threshold = ALERT_THRESHOLDS.get(dominant_state, 3)
            if dominant_state != self._current_bad_state:
                self._bad_streak = 0
                self._current_bad_state = dominant_state
            self._bad_streak += 1

            now_ts = time.time()
            cooldown_ok = (now_ts - self._last_alert_time.get(dominant_state, 0.0)) >= ALERT_COOLDOWN_SEC
            if self._bad_streak < threshold or not cooldown_ok:
                continue

            self._bad_streak = 0
            self._last_alert_time[dominant_state] = now_ts
            self._alert_counts[dominant_state] = self._alert_counts.get(dominant_state, 0) + 1
            self._was_alerting = True
            message = _pick_message(dominant_state, dominant_gender)
            severity = "high" if dominant_state in {"away", "sleepy", "phone"} else "medium"
            continuous = dominant_state == "sleepy"
            pause_lesson = dominant_state in {"phone", "sleepy"}
            self.socketio.emit("camera_alert", {
                "message": message,
                "attention": dominant_state,
                "gender": dominant_gender,
                "severity": severity,
                "continuous": continuous,
                "pause_lesson": pause_lesson,
                "resume_hint": "I'll continue once you're back with me." if pause_lesson else "",
                "details": details,
                "soft": False,
            })

            if self.on_alert_callback:
                try:
                    self.on_alert_callback(message, dominant_state, dominant_gender, details)
                except Exception:
                    pass

            if self.tts_callback and not self.tutor_speaking:
                try:
                    self.tts_callback(message)
                except Exception:
                    pass

    def _window_majority(self) -> tuple[str, str]:
        valid = [(a, g, c) for a, g, c in self._window if c >= MIN_CONF]
        entries = valid if valid else list(self._window)

        attention_counts: dict[str, int] = {}
        gender_counts: dict[str, int] = {}
        for attention, gender, _confidence in entries:
            attention_counts[attention] = attention_counts.get(attention, 0) + 1
            if gender in ("male", "female"):
                gender_counts[gender] = gender_counts.get(gender, 0) + 1

        dominant_attention = max(attention_counts, key=attention_counts.__getitem__)
        dominant_gender = max(gender_counts, key=gender_counts.__getitem__) if gender_counts else "unknown"
        return dominant_attention, dominant_gender

    def _query_vlm(self, frame_bytes: bytes) -> Optional[dict]:
        try:
            from openai import OpenAI

            try:
                from PIL import Image, ImageEnhance, ImageStat

                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                mean_lum = sum(ImageStat.Stat(img).mean) / 3
                if mean_lum < 5:
                    return None
                if mean_lum < 60:
                    boost = min(100.0 / max(mean_lum, 1), 6.0)
                    img = ImageEnhance.Brightness(img).enhance(boost)
                    img = ImageEnhance.Contrast(img).enhance(1.4)
                if img.width > MAX_WIDTH:
                    ratio = MAX_WIDTH / img.width
                    img = img.resize((MAX_WIDTH, int(img.height * ratio)))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                frame_bytes = buf.getvalue()
            except Exception:
                pass

            client = OpenAI(api_key="EMPTY", base_url=VLM_BASE)
            b64 = base64.b64encode(frame_bytes).decode("ascii")
            resp = client.chat.completions.create(
                model=VLM_MODEL,
                messages=[
                    {"role": "system", "content": MONITOR_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": MONITOR_PROMPT},
                        ],
                    },
                ],
                max_tokens=200,
                temperature=0.1,
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group())
        except Exception:
            return None
