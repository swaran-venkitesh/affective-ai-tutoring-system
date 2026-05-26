import re


WHITESPACE_RE = re.compile(r"\s+")
STEP_LABEL_RE = re.compile(r"\b(step\s+\d+)\s*:", re.I)
FRACTION_RE = re.compile(r"(?<!\w)([A-Za-z0-9.]+)\s*/\s*([A-Za-z0-9.]+)(?!\w)")
POWER_TWO_RE = re.compile(r"([A-Za-z0-9)\]])\s*\^\s*2\b")
POWER_THREE_RE = re.compile(r"([A-Za-z0-9)\]])\s*\^\s*3\b")
POWER_N_RE = re.compile(r"([A-Za-z0-9)\]])\s*\^\s*([A-Za-z0-9]+)")
SQRT_RE = re.compile(r"(?:sqrt|√)\s*\(?\s*([A-Za-z0-9+\-*/ ]+?)\s*\)?(?=\s|$)", re.I)
COEFFICIENT_VAR_RE = re.compile(r"(?<![\w.])(\d+)\s*([A-Za-z])\b")
COEFFICIENT_PAREN_RE = re.compile(r"(?<![\w.])(\d+)\s*\(")
RATIO_RE = re.compile(r"(?<!\w)(\d+)\s*:\s*(\d+)(?!\w)")
DECIMAL_RE = re.compile(r"(?<!\w)(\d+)\.(\d+)(?!\w)")


UNICODE_REPLACEMENTS = {
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "≠": " not equal to ",
    "≈": " approximately equal to ",
    "≃": " approximately equal to ",
    "∞": " infinity ",
    "π": " pi ",
    "θ": " theta ",
    "Δ": " delta ",
    "δ": " delta ",
    "Σ": " sigma ",
    "∑": " summation ",
    "√": " square root of ",
    "×": " times ",
    "÷": " divided by ",
    "−": " minus ",
    "—": " minus ",
    "–": " minus ",
    "°": " degrees ",
    "→": " gives ",
    "⇒": " therefore ",
}


def _normalize_unicode(text: str) -> str:
    out = str(text or "")
    for old, new in UNICODE_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def _replace_powers(text: str) -> str:
    out = POWER_TWO_RE.sub(r"\1 squared", text)
    out = POWER_THREE_RE.sub(r"\1 cubed", out)
    out = POWER_N_RE.sub(r"\1 to the power of \2", out)
    return out


def _replace_sqrt(text: str) -> str:
    return SQRT_RE.sub(lambda m: f"square root of {m.group(1).strip()}", text)


def _replace_fraction(text: str) -> str:
    return FRACTION_RE.sub(r"\1 divided by \2", text)


def _replace_ratio(text: str) -> str:
    return RATIO_RE.sub(r"\1 to \2", text)


def _replace_coefficients(text: str) -> str:
    out = COEFFICIENT_VAR_RE.sub(r"\1 times \2", text)
    out = COEFFICIENT_PAREN_RE.sub(r"\1 times (", out)
    return out


def _replace_decimals(text: str) -> str:
    return DECIMAL_RE.sub(r"\1 point \2", text)


def _replace_basic_ops(text: str) -> str:
    out = text
    out = out.replace("**", " to the power of ")
    out = out.replace("//", " floor divided by ")
    out = out.replace("!=", " not equal to ")
    out = out.replace(">=", " greater than or equal to ")
    out = out.replace("<=", " less than or equal to ")
    out = out.replace("==", " equals ")
    out = re.sub(
        r"(^|[=(:,])\s*-\s*(?=\d|[A-Za-z])",
        lambda m: f"{m.group(1)} negative ",
        out,
    )
    out = out.replace("=>", " therefore ")
    out = out.replace("->", " gives ")
    out = out.replace("=", " equals ")
    out = out.replace("+", " plus ")
    out = out.replace("*", " times ")
    out = out.replace("%", " percent ")
    out = out.replace("-", " minus ")
    return out


def _replace_keywords(text: str) -> str:
    out = text
    out = STEP_LABEL_RE.sub(lambda m: f"{m.group(1)}. ", out)
    out = re.sub(r"\bprint\(", "print ", out)
    out = re.sub(r"\bdef\s+", "define function ", out)
    out = re.sub(r"\bint\(", "integer of ", out)
    out = re.sub(r"\bfloat\(", "float of ", out)
    out = re.sub(r"\brange\(", "range ", out)
    return out


def _cleanup(text: str) -> str:
    out = str(text or "")
    out = out.replace("(", " open bracket ")
    out = out.replace(")", " close bracket ")
    out = out.replace("[", " open bracket ")
    out = out.replace("]", " close bracket ")
    out = out.replace("{", " open brace ")
    out = out.replace("}", " close brace ")
    out = out.replace(":", ". ")
    out = out.replace(";", ". ")
    out = out.replace(",", ", ")
    out = out.replace("`", " ")
    out = out.replace('"', " ")
    out = re.sub(r"<[^>]+>", " ", out)
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def tts_friendly(text: str, **_kwargs) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _normalize_unicode(clean)
    clean = _replace_keywords(clean)
    clean = _replace_sqrt(clean)
    clean = _replace_powers(clean)
    clean = _replace_coefficients(clean)
    clean = _replace_fraction(clean)
    clean = _replace_ratio(clean)
    clean = _replace_decimals(clean)
    clean = _replace_basic_ops(clean)
    clean = _cleanup(clean)
    return clean
