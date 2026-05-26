from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import requests

VLM_BASE = os.getenv("TUTOR_VLM_BASE_URL", os.getenv("VLM_BASE_URL", "http://172.16.13.91:8023/v1")).rstrip("/")
VLM_MODEL = os.getenv("TUTOR_VLM_MODEL", os.getenv("VLM_MODEL", "qwen2.5-vl-7b")).strip() or "qwen2.5-vl-7b"
VLM_ROOT = VLM_BASE[:-3] if VLM_BASE.endswith("/v1") else VLM_BASE
VLM_HEALTH_URL = f"{VLM_ROOT}/health"
VLM_MODELS_URL = f"{VLM_BASE}/models"
VLM_CHECK_TIMEOUT_SEC = float(os.getenv("TUTOR_VLM_TIMEOUT_SEC", "3.0"))
VLM_INFERENCE_TIMEOUT_SEC = float(os.getenv("TUTOR_VLM_INFERENCE_TIMEOUT_SEC", os.getenv("TUTOR_UPLOAD_TIMEOUT_SEC", "45.0")))
VLM_STATUS_TTL_SEC = 8.0
_vlm_status_cache = {
    "checked_at": 0.0,
    "available": False,
    "reason": "not checked yet",
}


def _unavailable_marker(reason: str) -> str:
    reason = (reason or "unknown reason").strip()
    return f"[Vision server unavailable: {reason}]"


def _is_unavailable_description(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return normalized.startswith("[image content unavailable:") or normalized.startswith("[vision server unavailable:")


def _set_vlm_status(available: bool, reason: str) -> tuple[bool, str]:
    normalized_reason = (reason or "").strip() or ("ready" if available else "unknown reason")
    _vlm_status_cache["checked_at"] = time.time()
    _vlm_status_cache["available"] = bool(available)
    _vlm_status_cache["reason"] = normalized_reason
    return bool(available), normalized_reason


def get_vlm_status(force: bool = False) -> tuple[bool, str]:
    now = time.time()
    if not force and (now - float(_vlm_status_cache["checked_at"])) < VLM_STATUS_TTL_SEC:
        return bool(_vlm_status_cache["available"]), str(_vlm_status_cache["reason"])

    health_error = ""
    try:
        response = requests.get(VLM_HEALTH_URL, timeout=VLM_CHECK_TIMEOUT_SEC)
        if response.ok:
            health_ok = True
        else:
            health_ok = False
            health_error = f"health returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        health_ok = False
        health_error = str(exc)

    models_error = ""
    try:
        response = requests.get(VLM_MODELS_URL, timeout=VLM_CHECK_TIMEOUT_SEC)
        if response.ok:
            try:
                payload = response.json() or {}
            except ValueError:
                payload = {}
            model_ids = {
                str(item.get("id", "")).strip()
                for item in (payload.get("data") or [])
                if isinstance(item, dict)
            }
            if not model_ids or VLM_MODEL in model_ids:
                return _set_vlm_status(True, "ready")
            models_error = f"model '{VLM_MODEL}' is not listed at {VLM_MODELS_URL}"
        else:
            models_error = f"models returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        models_error = str(exc)

    if health_ok:
        return _set_vlm_status(True, "health endpoint reachable")

    reason = models_error or health_error or f"could not reach {VLM_ROOT}"
    return _set_vlm_status(False, reason)


def _describe_image_bytes(img_bytes: bytes, mime: str = "image/jpeg") -> str:
    available, reason = get_vlm_status()
    if not available:
        return _unavailable_marker(f"{reason} ({VLM_ROOT})")

    try:
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=VLM_BASE, timeout=VLM_INFERENCE_TIMEOUT_SEC)

        try:
            from PIL import Image as PILImage

            img = PILImage.open(io.BytesIO(img_bytes))
            if img.width > 1024:
                ratio = 1024 / img.width
                img = img.resize((1024, int(img.height * ratio)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            img_bytes = buf.getvalue()
            mime = "image/jpeg"
        except Exception:
            pass

        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. Extract all visible text, equations, "
                        "tables, diagrams, labels, and structure."
                    ),
                },
            ],
        }]
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
            stream=False,
            timeout=VLM_INFERENCE_TIMEOUT_SEC,
        )
        _set_vlm_status(True, "ready")
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _set_vlm_status(False, str(exc))
        return _unavailable_marker(f"{exc} ({VLM_ROOT})")


def _extract_pdf_visual_chunks(file_bytes: bytes, filename: str, page_indices: list[int]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for index in page_indices:
                if index < 0 or index >= len(doc):
                    continue
                page = doc[index]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                image_bytes = pix.tobytes("png")
                desc = _describe_image_bytes(image_bytes, "image/png").strip()
                if not desc or _is_unavailable_description(desc):
                    continue
                label = f"FROM PDF '{filename}' Page {index + 1} (visual)"
                results.append((desc, label))
        finally:
            doc.close()
    except Exception:
        return []
    return results


def _pdf_chunk_span(total_pages: int, default_span: int) -> int:
    if total_pages >= 120:
        return 1
    if total_pages >= 36:
        return 2
    return max(1, default_span)


def _pdf_chunk_label(filename: str, start: int, end: int) -> str:
    if start == end:
        return f"FROM PDF '{filename}' Page {start}"
    return f"FROM PDF '{filename}' Pages {start}-{end}"


def extract_pdf(file_bytes: bytes, filename: str, pages_per_chunk: int = 4) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    total_text_chars = 0
    mostly_empty_pages: list[int] = []

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total = len(pdf.pages)
            chunk_span = _pdf_chunk_span(total, pages_per_chunk)
            for start in range(0, total, chunk_span):
                end = min(start + chunk_span, total)
                chunk_lines = []
                for index in range(start, end):
                    page = pdf.pages[index]
                    text = (page.extract_text() or "").strip()
                    if text:
                        chunk_lines.append(f"[Page {index + 1}]\n{text}")
                        total_text_chars += len(text)
                    else:
                        mostly_empty_pages.append(index)
                if chunk_lines:
                    label = _pdf_chunk_label(filename, start + 1, end)
                    results.append(("\n\n".join(chunk_lines), label))
        if results and total_text_chars >= 500:
            return results
    except ImportError:
        pass
    except Exception as exc:
        print(f"pdfplumber error: {exc}", flush=True)

    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total = len(doc)
        chunk_span = _pdf_chunk_span(total, pages_per_chunk)
        for start in range(0, total, chunk_span):
            end = min(start + chunk_span, total)
            chunk_lines = []
            for index in range(start, end):
                page = doc[index]
                text = (page.get_text() or "").strip()
                if text:
                    chunk_lines.append(f"[Page {index + 1}]\n{text}")
                    total_text_chars += len(text)
                elif index not in mostly_empty_pages:
                    mostly_empty_pages.append(index)
            if chunk_lines:
                label = _pdf_chunk_label(filename, start + 1, end)
                results.append(("\n\n".join(chunk_lines), label))
        doc.close()
        if results and total_text_chars >= 500:
            return results
    except ImportError:
        pass
    except Exception as exc:
        print(f"PyMuPDF error: {exc}", flush=True)

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        total = len(reader.pages)
        chunk_span = _pdf_chunk_span(total, pages_per_chunk)
        for start in range(0, total, chunk_span):
            end = min(start + chunk_span, total)
            chunk_lines = []
            for index in range(start, end):
                text = (reader.pages[index].extract_text() or "").strip()
                if text:
                    chunk_lines.append(f"[Page {index + 1}]\n{text}")
                    total_text_chars += len(text)
                elif index not in mostly_empty_pages:
                    mostly_empty_pages.append(index)
            if chunk_lines:
                label = _pdf_chunk_label(filename, start + 1, end)
                results.append(("\n\n".join(chunk_lines), label))
        if results and total_text_chars >= 500:
            return results
    except ImportError:
        pass
    except Exception as exc:
        print(f"PyPDF2 error: {exc}", flush=True)

    if results:
        visual_pages = mostly_empty_pages[:2]
        if not visual_pages and total_text_chars < 900:
            visual_pages = list(range(min(2, len(results))))
        if visual_pages:
            results.extend(_extract_pdf_visual_chunks(file_bytes, filename, visual_pages))
        return results

    raise RuntimeError(f"Could not extract readable content from PDF '{filename}'.")


def extract_docx(file_bytes: bytes, filename: str) -> list[tuple[str, str]]:
    try:
        import docx as python_docx
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX uploads.") from exc

    doc = python_docx.Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if (p.text or "").strip()]
    full_text = "\n".join(paras)

    image_descs = []
    try:
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                img_data = rel.target_part.blob
                ext = rel.target_part.content_type.split("/")[-1]
                mime = f"image/{ext}" if ext else "image/jpeg"
                image_descs.append(f"[Embedded Image]: {_describe_image_bytes(img_data, mime)}")
            except Exception:
                pass
    except Exception:
        pass

    combined = full_text
    if image_descs:
        combined = (combined + "\n\n" if combined else "") + "\n".join(image_descs)

    label = f"FROM DOCUMENT '{filename}'"
    chunk_size = 3000
    chunks = []
    for start in range(0, len(combined), chunk_size):
        part = combined[start:start + chunk_size]
        part_label = f"{label} (Part {start // chunk_size + 1})" if len(combined) > chunk_size else label
        chunks.append((part, part_label))
    return chunks or [(combined, label)]


def extract_image(file_bytes: bytes, filename: str, mime_type: str) -> list[tuple[str, str]]:
    return [(_describe_image_bytes(file_bytes, mime_type), f"FROM IMAGE '{filename}'")]


def extract_text_file(file_bytes: bytes, filename: str) -> list[tuple[str, str]]:
    text = file_bytes.decode("utf-8", errors="replace")
    label = f"FROM FILE '{filename}'"
    chunk_size = 3000
    chunks = []
    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        part_label = f"{label} (Part {start // chunk_size + 1})" if len(text) > chunk_size else label
        chunks.append((chunk, part_label))
    return chunks or [(text, label)]


def process_upload(file_bytes: bytes, filename: str, mime_type: str) -> list[tuple[str, str]]:
    ext = Path(filename).suffix.lower()
    mime_type = (mime_type or "").lower()

    if ext == ".pdf" or "pdf" in mime_type:
        return extract_pdf(file_bytes, filename)

    if ext in (".docx", ".doc") or "word" in mime_type or "officedocument" in mime_type:
        return extract_docx(file_bytes, filename)

    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif") or mime_type.startswith("image/"):
        return extract_image(file_bytes, filename, mime_type or "image/jpeg")

    if ext in (".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv"):
        return extract_text_file(file_bytes, filename)

    try:
        decoded = file_bytes.decode("utf-8", errors="replace")
        if decoded.strip():
            return extract_text_file(file_bytes, filename)
    except Exception:
        pass

    return [(f"[Unsupported file type: {mime_type or 'unknown'}]", f"FROM FILE '{filename}'")]
