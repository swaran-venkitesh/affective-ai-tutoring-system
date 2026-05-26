from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent / "student_login"
INDEX_PATH = ROOT / "students_index.json"


def _ensure_root() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (text or "student").strip()).strip("-")
    return cleaned[:48] or "student"


def _default_index() -> Dict[str, Any]:
    return {"students": {}, "last_active_id": ""}


def load_registry() -> Dict[str, Any]:
    _ensure_root()
    if not INDEX_PATH.exists():
        return _default_index()
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_index()
        data.setdefault("students", {})
        data.setdefault("last_active_id", "")
        return data
    except Exception:
        return _default_index()


def save_registry(data: Dict[str, Any]) -> None:
    _ensure_root()
    tmp_path = INDEX_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(INDEX_PATH)


def _normalize_age(age: Any) -> int:
    try:
        value = int(age)
        return max(3, min(120, value))
    except Exception:
        return 20


def _normalize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    profile = {
        "student_id": str(data.get("student_id") or "").strip(),
        "name": str(data.get("name") or "Student").strip() or "Student",
        "age": _normalize_age(data.get("age")),
        "email": str(data.get("email") or "").strip(),
        "created_at": float(data.get("created_at") or now),
        "updated_at": float(data.get("updated_at") or now),
        "last_seen_at": float(data.get("last_seen_at") or now),
        "folder_name": str(data.get("folder_name") or "").strip(),
        "city": str(data.get("city") or "").strip(),
        "tz": str(data.get("tz") or "").strip(),
        "notes": str(data.get("notes") or "").strip(),
    }
    if not profile["student_id"]:
        profile["student_id"] = "stu_" + uuid.uuid4().hex[:8]
    if not profile["folder_name"]:
        profile["folder_name"] = f"{_slugify(profile['name'])}__{profile['student_id']}"
    return profile


def student_paths(profile: Dict[str, Any]) -> Dict[str, Path]:
    normalized = _normalize_profile(profile)
    root = ROOT / normalized["folder_name"]
    paths = {
        "root": root,
        "profile": root / "profile",
        "all_memory": root / "all_memory",
        "session_reports": root / "session_reports",
        "materials": root / "materials",
        "sessions": root / "sessions",
        "a_z_memory": root / "a_z_memory",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_profile(profile: Dict[str, Any], make_last_active: bool = True) -> Dict[str, Any]:
    normalized = _normalize_profile(profile)
    paths = student_paths(normalized)
    registry = load_registry()
    normalized["updated_at"] = time.time()
    normalized["last_seen_at"] = normalized["updated_at"]
    registry.setdefault("students", {})[normalized["student_id"]] = normalized
    if make_last_active:
        registry["last_active_id"] = normalized["student_id"]
    save_registry(registry)
    profile_path = paths["profile"] / "profile.json"
    profile_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def create_student(name: str, age: Any, email: str = "", student_id: str = "") -> Tuple[Dict[str, Any], bool]:
    registry = load_registry()
    students = registry.setdefault("students", {})
    if student_id and student_id in students:
        existing = dict(students[student_id])
        existing["name"] = (name or existing.get("name") or "Student").strip() or "Student"
        existing["age"] = _normalize_age(age if age is not None else existing.get("age"))
        if email is not None:
            existing["email"] = str(email or "").strip()
        return save_profile(existing, make_last_active=True), False

    name = (name or "Student").strip() or "Student"
    email = str(email or "").strip()
    if email:
        for candidate in students.values():
            if str(candidate.get("email") or "").strip().lower() == email.lower() and str(candidate.get("name") or "").strip().lower() == name.lower():
                existing = dict(candidate)
                existing["age"] = _normalize_age(age)
                return save_profile(existing, make_last_active=True), False

    profile = {
        "student_id": "stu_" + uuid.uuid4().hex[:8],
        "name": name,
        "age": _normalize_age(age),
        "email": email,
        "created_at": time.time(),
        "updated_at": time.time(),
        "last_seen_at": time.time(),
    }
    return save_profile(profile, make_last_active=True), True


def list_students() -> List[Dict[str, Any]]:
    registry = load_registry()
    students = [_normalize_profile(v) for v in (registry.get("students") or {}).values() if isinstance(v, dict)]
    return sorted(students, key=lambda item: (float(item.get("last_seen_at") or 0), float(item.get("updated_at") or 0)), reverse=True)


def get_student(student_id: str) -> Dict[str, Any] | None:
    registry = load_registry()
    raw = (registry.get("students") or {}).get(student_id)
    if not isinstance(raw, dict):
        return None
    profile = _normalize_profile(raw)
    profile_path = student_paths(profile)["profile"] / "profile.json"
    if profile_path.exists():
        try:
            profile = _normalize_profile(json.loads(profile_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return profile


def get_last_active_student() -> Dict[str, Any] | None:
    registry = load_registry()
    last_active = str(registry.get("last_active_id") or "").strip()
    if last_active:
        student = get_student(last_active)
        if student:
            return student
    return None


def set_last_active_student(student_id: str) -> Dict[str, Any] | None:
    profile = get_student(student_id)
    if not profile:
        return None
    save_profile(profile, make_last_active=True)
    return profile


def student_runtime_state_path(profile: Dict[str, Any]) -> Path:
    return student_paths(profile)["all_memory"] / "runtime_state.json"


def student_shared_profile_path(profile: Dict[str, Any]) -> Path:
    return student_paths(profile)["profile"] / "profile.json"


def student_session_dir(profile: Dict[str, Any], session_id: str) -> Path:
    path = student_paths(profile)["sessions"] / str(session_id or "session")
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_last_active_student() -> None:
    registry = load_registry()
    registry["last_active_id"] = ""
    save_registry(registry)
