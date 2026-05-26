import html
import re
import time
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests


SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

STUDY_DOMAINS = [
    "geeksforgeeks.org",
    "khanacademy.org",
    "mathsisfun.com",
    "ncert.nic.in",
    "wikipedia.org",
]

TECH_DOMAINS = [
    "docs.python.org",
    "developer.mozilla.org",
    "nodejs.org",
    "react.dev",
    "numpy.org",
    "pandas.pydata.org",
]

RESEARCH_DOMAINS = [
    "arxiv.org",
    "nature.com",
    "springer.com",
]

NEWS_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "thehindu.com",
]
EXAM_DOMAINS = [
    "ets.org",
]

LATEST_HINT_RE = re.compile(
    r"\b(latest|recent|today|current|news|headline|update|updated|new study|research)\b",
    re.I,
)
TECH_HINT_RE = re.compile(
    r"\b(python|javascript|js|node|react|html|css|numpy|pandas|programming|code|coding)\b",
    re.I,
)
MATH_HINT_RE = re.compile(
    r"\b(math|mathematics|algebra|geometry|trigonometry|probability|statistics|class\s*\d+)\b",
    re.I,
)
EXAM_HINT_RE = re.compile(
    r"\b(gre|toefl|ielts|sat|act|gmat)\b",
    re.I,
)
RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="[^"]*(?:result__a|result-link)[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.I | re.S,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)
BLOCK_TAG_RE = re.compile(r"</?(?:p|div|section|article|main|li|ul|ol|h[1-6]|br|tr|td|th)[^>]*>", re.I)
SCRIPT_STYLE_RE = re.compile(r"<(?:script|style|noscript|svg)[^>]*>.*?</(?:script|style|noscript|svg)>", re.I | re.S)

_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}
_CACHE_TTL_SECS = 900.0


def _normalize_domain(host: str) -> str:
    return host.lower().lstrip("www.")


def _is_allowed_url(url: str, domain: str) -> bool:
    parsed = urlparse(url)
    host = _normalize_domain(parsed.netloc)
    return bool(host) and (host == domain or host.endswith("." + domain))


def _resolve_result_url(raw_href: str) -> str:
    href = html.unescape(raw_href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(uddg or "").strip()
    return href


def _strip_tags(fragment: str) -> str:
    text = TAG_RE.sub(" ", fragment or "")
    return SPACE_RE.sub(" ", html.unescape(text)).strip()


def _visible_text(page_html: str) -> str:
    if not page_html:
        return ""
    cleaned = SCRIPT_STYLE_RE.sub(" ", page_html)
    cleaned = BLOCK_TAG_RE.sub("\n", cleaned)
    cleaned = TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [SPACE_RE.sub(" ", line).strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _pick_domains(query: str) -> List[str]:
    lowered = query.lower()
    chosen: List[str] = []
    if EXAM_HINT_RE.search(lowered):
        chosen.extend(EXAM_DOMAINS)
    if TECH_HINT_RE.search(lowered):
        chosen.extend(TECH_DOMAINS)
        chosen.extend(STUDY_DOMAINS[:2])
    elif MATH_HINT_RE.search(lowered):
        chosen.extend(["khanacademy.org", "mathsisfun.com", "ncert.nic.in"])
        chosen.extend(STUDY_DOMAINS[:2])
    else:
        chosen.extend(STUDY_DOMAINS)
    if LATEST_HINT_RE.search(lowered):
        chosen.extend(RESEARCH_DOMAINS)
        chosen.extend(NEWS_DOMAINS)
    deduped: List[str] = []
    for domain in chosen:
        if domain not in deduped:
            deduped.append(domain)
    return deduped[:8]


def _search_domain(query: str, domain: str, timeout: float = 12.0) -> List[Tuple[str, str]]:
    response = requests.get(
        SEARCH_ENDPOINT,
        params={"q": f"site:{domain} {query}"},
        headers=REQUEST_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    hits: List[Tuple[str, str]] = []
    seen_urls = set()
    for href, title_html in RESULT_LINK_RE.findall(response.text or ""):
        resolved = _resolve_result_url(href)
        if not resolved or not _is_allowed_url(resolved, domain) or resolved in seen_urls:
            continue
        title = _strip_tags(title_html)
        if not title:
            continue
        hits.append((title, resolved))
        seen_urls.add(resolved)
        if len(hits) >= 2:
            break
    return hits


def _best_excerpt(text: str, query: str, max_chars: int = 520) -> str:
    cleaned = SPACE_RE.sub(" ", text or "").strip()
    if not cleaned:
        return ""
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return cleaned[:max_chars]
    pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    best_piece = ""
    best_score = -1
    for idx, piece in enumerate(pieces):
        piece_tokens = set(_tokens(piece))
        score = len(query_tokens & piece_tokens)
        if idx < 2:
            score += 0.2
        if score > best_score and piece.strip():
            best_score = score
            best_piece = piece.strip()
    if len(best_piece) > max_chars:
        return best_piece[: max_chars - 3].rstrip() + "..."
    if best_piece:
        return best_piece
    return cleaned[:max_chars]


def _fetch_source_excerpt(url: str, query: str, timeout: float = 12.0) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    text = _visible_text(response.text or "")
    return _best_excerpt(text, query)


def build_trusted_web_context(query: str, max_sources: int = 3) -> Dict[str, object]:
    normalized_query = SPACE_RE.sub(" ", str(query or "").strip())
    if not normalized_query:
        return {"query": "", "sources": [], "context": "", "error": "empty_query"}

    cached = _CACHE.get(normalized_query)
    now = time.time()
    if cached and (now - cached[0]) < _CACHE_TTL_SECS:
        return dict(cached[1])

    candidate_domains = _pick_domains(normalized_query)
    sources: List[Dict[str, str]] = []
    errors: List[str] = []
    seen_urls = set()

    for domain in candidate_domains:
        try:
            hits = _search_domain(normalized_query, domain)
        except Exception as exc:
            errors.append(f"{domain}: {exc}")
            continue
        for title, url in hits:
            if url in seen_urls:
                continue
            try:
                snippet = _fetch_source_excerpt(url, normalized_query)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue
            if not snippet:
                continue
            sources.append({
                "title": title,
                "url": url,
                "domain": domain,
                "snippet": snippet,
            })
            seen_urls.add(url)
            if len(sources) >= max_sources:
                break
        if len(sources) >= max_sources:
            break

    context_blocks = []
    for idx, source in enumerate(sources, start=1):
        context_blocks.append(
            f"[{idx}] {source['title']} ({source['domain']})\n"
            f"URL: {source['url']}\n"
            f"Snippet: {source['snippet']}"
        )

    payload: Dict[str, object] = {
        "query": normalized_query,
        "sources": sources,
        "context": "\n\n".join(context_blocks),
        "error": "" if sources else ("; ".join(errors[:3]) or "no_results"),
    }
    _CACHE[normalized_query] = (now, dict(payload))
    return payload
