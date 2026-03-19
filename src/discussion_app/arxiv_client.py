from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from xml.etree import ElementTree

import requests


ARXIV_API_URLS = (
    "https://export.arxiv.org/api/query",
    "http://export.arxiv.org/api/query",
)
ARXIV_TIMEOUT_SECONDS = 30
ARXIV_USER_AGENT = "Cyber-Colloquium/0.1 (+https://github.com/)"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

ARXIV_CN_TERM_MAP: tuple[tuple[str, str], ...] = (
    ("遥感图像变化检测", "remote sensing change detection"),
    ("遥感影像变化检测", "remote sensing change detection"),
    ("图像变化检测", "image change detection"),
    ("变化检测", "change detection"),
    ("遥感图像", "remote sensing image"),
    ("遥感影像", "remote sensing image"),
    ("遥感", "remote sensing"),
    ("时空", "spatiotemporal"),
    ("时序", "time series"),
    ("语义分割", "semantic segmentation"),
    ("图像分割", "image segmentation"),
    ("目标检测", "object detection"),
    ("图像检索", "image retrieval"),
    ("多模态", "multimodal"),
    ("视觉", "vision"),
    ("综述", "survey"),
)
ARXIV_ASCII_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "using",
    "study",
    "research",
    "application",
    "applications",
    "model",
    "models",
    "analysis",
    "advantages",
    "limitations",
}


@dataclass(frozen=True)
class ArxivPaper:
    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_at: str
    updated_at: str
    entry_url: str
    pdf_url: str

    @property
    def safe_stem(self) -> str:
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", self.paper_id).strip("_")
        return stem or "arxiv_paper"


def search_arxiv(query: str, *, max_results: int = 5, timeout: int = ARXIV_TIMEOUT_SECONDS) -> list[ArxivPaper]:
    cleaned_query = " ".join(query.split())
    if not cleaned_query:
        return []

    query_candidates = build_arxiv_query_candidates(cleaned_query)
    last_error: Exception | None = None
    had_successful_response = False
    for candidate in query_candidates:
        params = {
            "search_query": f"all:{candidate}",
            "start": 0,
            "max_results": max(1, max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        for api_url in ARXIV_API_URLS:
            try:
                response = requests.get(
                    api_url,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": ARXIV_USER_AGENT},
                )
                response.raise_for_status()
                had_successful_response = True
                papers = parse_arxiv_feed(response.text)
                if papers:
                    return papers
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
    if had_successful_response:
        return []
    if last_error is not None:
        raise last_error
    return []


def build_arxiv_query_candidates(query: str, *, max_terms: int = 6) -> list[str]:
    cleaned_query = " ".join((query or "").split()).strip()
    if not cleaned_query:
        return []

    english_terms = _dedupe_preserve_order(_mapped_english_terms(cleaned_query))
    ascii_terms = _dedupe_preserve_order(_ascii_query_terms(cleaned_query))

    candidates: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        normalized = " ".join(candidate.split()).strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    if not _contains_cjk(cleaned_query):
        _push(cleaned_query)

    combined = _dedupe_preserve_order([*ascii_terms, *english_terms])[:max_terms]
    if combined:
        _push(" ".join(combined))
    if len(combined) >= 2:
        _push(" ".join(combined[: min(4, len(combined))]))
    if english_terms:
        _push(" ".join(english_terms[:max_terms]))
    if ascii_terms:
        _push(" ".join(ascii_terms[:max_terms]))

    if not candidates:
        _push(cleaned_query)
    return candidates


def parse_arxiv_feed(xml_text: str) -> list[ArxivPaper]:
    root = ElementTree.fromstring(xml_text)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_url = _entry_text(entry, "atom:id")
        title = _normalize_ws(_entry_text(entry, "atom:title"))
        abstract = _normalize_ws(_entry_text(entry, "atom:summary"))
        authors = tuple(
            _normalize_ws(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
            if _normalize_ws(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        )
        categories = tuple(
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term", "").strip()
        )
        published_at = _normalize_ws(_entry_text(entry, "atom:published"))
        updated_at = _normalize_ws(_entry_text(entry, "atom:updated"))
        pdf_url = _resolve_pdf_url(entry, entry_url)
        paper_id = _paper_id_from_entry(entry_url)
        if not paper_id or not title:
            continue
        papers.append(
            ArxivPaper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published_at=published_at,
                updated_at=updated_at,
                entry_url=entry_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def download_arxiv_pdf(paper: ArxivPaper, target_dir: Path, *, timeout: int = 60) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{paper.safe_stem}.pdf"
    pdf_urls = (paper.pdf_url, paper.pdf_url.replace("https://", "http://")) if paper.pdf_url.startswith("https://") else (paper.pdf_url,)
    last_error: Exception | None = None
    for pdf_url in pdf_urls:
        try:
            response = requests.get(
                pdf_url,
                timeout=timeout,
                headers={"User-Agent": ARXIV_USER_AGENT},
            )
            response.raise_for_status()
            target_path.write_bytes(response.content)
            return target_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return target_path


def render_bibtex_entry(paper: ArxivPaper) -> tuple[str, str]:
    key = _bibtex_key(paper)
    author_field = " and ".join(paper.authors) or "Unknown Author"
    year = paper.published_at[:4] if len(paper.published_at) >= 4 else "0000"
    category_field = ", ".join(paper.categories)
    body = (
        f"@article{{{key},\n"
        f"  title = {{{_bibtex_escape(paper.title)}}},\n"
        f"  author = {{{_bibtex_escape(author_field)}}},\n"
        f"  journal = {{arXiv preprint arXiv:{_bibtex_escape(paper.paper_id)}}},\n"
        f"  year = {{{year}}},\n"
        f"  eprint = {{{_bibtex_escape(paper.paper_id)}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass = {{{_bibtex_escape(paper.categories[0] if paper.categories else '')}}},\n"
        f"  keywords = {{{_bibtex_escape(category_field)}}},\n"
        f"  url = {{{_bibtex_escape(paper.entry_url)}}}\n"
        "}\n"
    )
    return key, body


def save_arxiv_metadata(papers: list[object], path: Path) -> Path:
    payload = [asdict(paper) if is_dataclass(paper) else paper for paper in papers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_bibtex_library(entries: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = "\n".join(entry.rstrip() for entry in entries if entry.strip()).strip() + "\n"
    path.write_text(normalized, encoding="utf-8")
    return path


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _mapped_english_terms(text: str) -> list[str]:
    ordered: list[str] = []
    for cn_term, en_term in ARXIV_CN_TERM_MAP:
        if cn_term in text:
            ordered.append(en_term)
    return ordered


def _ascii_query_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,}", text)
    return [
        token
        for token in tokens
        if token.lower() not in ARXIV_ASCII_STOPWORDS
    ]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(item.split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def _entry_text(entry: ElementTree.Element, key: str) -> str:
    return entry.findtext(key, default="", namespaces=ATOM_NS)


def _resolve_pdf_url(entry: ElementTree.Element, entry_url: str) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "").strip()
        if not href:
            continue
        title = link.attrib.get("title", "").strip().lower()
        if title == "pdf":
            return href
    paper_id = _paper_id_from_entry(entry_url)
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def _paper_id_from_entry(entry_url: str) -> str:
    candidate = entry_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return candidate.strip()


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _bibtex_key(paper: ArxivPaper) -> str:
    first_author = paper.authors[0] if paper.authors else "unknown"
    surname = re.sub(r"[^a-zA-Z0-9]+", "", first_author.split()[-1]).lower() or "unknown"
    year = paper.published_at[:4] if len(paper.published_at) >= 4 else "0000"
    first_word = re.sub(r"[^a-zA-Z0-9]+", "", paper.title.split()[0]).lower() if paper.title.split() else "paper"
    return f"{surname}{year}{first_word}"


def _bibtex_escape(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("&", "\\&")
