from __future__ import annotations

import base64
import re
from pathlib import Path

from pypdf import PdfReader

from .models import AttachmentPayload, AttachmentSnippet


TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LITERATURE_SECTION_HINTS = (
    ("abstract", 14),
    ("introduction", 10),
    ("related work", 7),
    ("method", 11),
    ("framework", 8),
    ("module", 6),
    ("experiment", 11),
    ("result", 9),
    ("ablation", 8),
    ("conclusion", 10),
    ("discussion", 6),
    ("dataset", 6),
)


def load_attachment(path_str: str) -> AttachmentPayload:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return AttachmentPayload(
            path=path,
            kind="pdf",
            content=extract_pdf_text(path),
            display_name=path.name,
        )
    if suffix in TEXT_EXTENSIONS:
        return AttachmentPayload(
            path=path,
            kind="text",
            content=path.read_text(encoding="utf-8", errors="ignore"),
            display_name=path.name,
        )
    if suffix in IMAGE_EXTENSIONS:
        return AttachmentPayload(
            path=path,
            kind="image",
            content=encode_image_as_data_url(path),
            display_name=path.name,
        )
    raise ValueError(f"Unsupported attachment type: {path.suffix}")


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {index}]\n{text.strip()}")
    return "\n\n".join(pages).strip()


def encode_image_as_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{path.suffix.lower().lstrip('.')}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_attachment_context(attachments: list[AttachmentPayload], max_chars: int = 18000) -> str:
    parts: list[str] = []
    used = 0
    for attachment in attachments:
        if attachment.kind == "image":
            parts.append(f"[Image Attachment] {attachment.display_name}")
            continue
        excerpt = attachment.content[: max(0, max_chars - used)]
        block = f"[Attachment: {attachment.display_name}]\n{excerpt}"
        used += len(excerpt)
        parts.append(block)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def build_attachment_index(
    attachments: list[AttachmentPayload],
    *,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    max_chunks_per_attachment: int = 32,
) -> list[AttachmentSnippet]:
    snippets: list[AttachmentSnippet] = []
    evidence_counter = 1

    for attachment in attachments:
        if attachment.kind == "image":
            continue

        text = _normalize_text(attachment.content)
        if not text:
            continue

        chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)[:max_chunks_per_attachment]
        for chunk_index, chunk in enumerate(chunks, start=1):
            snippets.append(
                AttachmentSnippet(
                    evidence_id=f"E{evidence_counter}",
                    attachment_name=attachment.display_name,
                    kind=attachment.kind,
                    chunk_index=chunk_index,
                    content=chunk,
                    page_hint=_page_hint(chunk),
                    keywords=_extract_keywords(f"{attachment.display_name} {chunk[:500]}"),
                )
            )
            evidence_counter += 1

    return snippets


def select_attachment_snippets(
    snippets: list[AttachmentSnippet],
    query: str,
    *,
    max_chars: int = 2400,
    max_snippets: int = 4,
) -> list[AttachmentSnippet]:
    if not snippets:
        return []

    query_terms = _extract_keywords(query)
    ranked = sorted(
        snippets,
        key=lambda item: (_score_snippet(item, query_terms), -item.chunk_index, item.attachment_name),
        reverse=True,
    )

    selected: list[AttachmentSnippet] = []
    used = 0
    for snippet in ranked:
        if len(selected) >= max_snippets:
            break
        snippet_cost = len(snippet.content) + len(snippet.attachment_name) + 32
        if selected and used + snippet_cost > max_chars:
            continue
        selected.append(snippet)
        used += snippet_cost

    if selected:
        return selected

    fallback: list[AttachmentSnippet] = []
    used = 0
    for snippet in snippets[:max_snippets]:
        snippet_cost = len(snippet.content) + len(snippet.attachment_name) + 32
        if fallback and used + snippet_cost > max_chars:
            continue
        fallback.append(snippet)
        used += snippet_cost
    return fallback


def select_literature_review_snippets(
    snippets: list[AttachmentSnippet],
    *,
    max_chars: int = 12000,
    max_snippets: int = 12,
) -> list[AttachmentSnippet]:
    if not snippets:
        return []

    selected: list[AttachmentSnippet] = []
    used = 0
    seen: set[tuple[str, int]] = set()

    grouped = _group_by_attachment(snippets)
    for attachment_name in sorted(grouped):
        ordered = sorted(grouped[attachment_name], key=lambda item: item.chunk_index)
        for snippet in _literature_coverage_candidates(ordered):
            key = (snippet.attachment_name, snippet.chunk_index)
            if key in seen:
                continue
            snippet_cost = len(snippet.content) + len(snippet.attachment_name) + 32
            if selected and used + snippet_cost > max_chars:
                continue
            selected.append(snippet)
            seen.add(key)
            used += snippet_cost
            if len(selected) >= max_snippets:
                return selected

    if selected:
        return selected
    return snippets[:max_snippets]


def render_attachment_snippets(snippets: list[AttachmentSnippet], max_chars: int = 2400) -> str:
    if not snippets:
        return ""

    parts: list[str] = []
    used = 0
    for snippet in snippets:
        block = f"[{describe_snippet(snippet)} | internal_id={snippet.evidence_id}]\n{snippet.content}"
        if parts and used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def render_literature_review_context(snippets: list[AttachmentSnippet], max_chars: int = 12000) -> str:
    if not snippets:
        return ""

    coverage = summarize_snippet_coverage(snippets)
    body_budget = max(2400, max_chars - len(coverage) - 64)
    body = render_attachment_snippets(snippets, max_chars=body_budget)
    return f"[Coverage Summary]\n{coverage}\n\n[Literature Packet]\n{body}"


def summarize_snippet_coverage(snippets: list[AttachmentSnippet]) -> str:
    grouped = _group_by_attachment(snippets)
    lines: list[str] = []
    for attachment_name in sorted(grouped):
        ordered = sorted(grouped[attachment_name], key=lambda item: item.chunk_index)
        pages = [page for page in (_page_hint(item.content) for item in ordered) if page is not None]
        page_text = _format_page_summary(pages)
        lines.append(
            f"- {attachment_name}: {len(ordered)} snippets, chunks {ordered[0].chunk_index}-{ordered[-1].chunk_index}, pages {page_text}"
        )
    return "\n".join(lines)


def render_evidence_catalog(snippets: list[AttachmentSnippet], max_items: int = 8) -> str:
    selected = snippets[:max_items]
    return "\n".join(
        f"- {describe_snippet(snippet)} | {snippet.content[:110].strip()}"
        for snippet in selected
    )


def split_literature_review_packets(
    snippets: list[AttachmentSnippet],
    *,
    max_chars_per_packet: int = 4200,
    max_packets: int = 3,
) -> list[list[AttachmentSnippet]]:
    if not snippets:
        return []

    grouped = _group_by_attachment(snippets)
    ordered: list[AttachmentSnippet] = []
    for attachment_name in sorted(grouped):
        ordered.extend(sorted(grouped[attachment_name], key=lambda item: item.chunk_index))

    total_chars = sum(len(snippet.content) + len(snippet.attachment_name) + 40 for snippet in ordered)
    packet_count = max(1, (len(ordered) + 3) // 4)
    packet_count = max(packet_count, (total_chars + max_chars_per_packet - 1) // max_chars_per_packet)
    packet_count = min(max_packets, max(1, packet_count))

    packets: list[list[AttachmentSnippet]] = []
    for packet_index in range(packet_count):
        start_index = round(packet_index * len(ordered) / packet_count)
        end_index = round((packet_index + 1) * len(ordered) / packet_count)
        packet = ordered[start_index:end_index]
        if packet:
            packets.append(packet)
    return packets


def describe_snippet(snippet: AttachmentSnippet) -> str:
    number = ''.join(char for char in snippet.evidence_id if char.isdigit()) or snippet.evidence_id
    parts = [snippet.attachment_name]
    if snippet.page_hint is not None:
        parts.append(f"page {snippet.page_hint}")
    parts.append(f"chunk {snippet.chunk_index}")
    return f"Evidence {number} ({", ".join(parts)})"


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _normalize_text(text: str) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text)
    compact = re.sub(r"[ \t]{2,}", " ", compact)
    return compact.strip()


def _extract_keywords(text: str) -> list[str]:
    normalized = text.lower()
    ascii_terms = re.findall(r"[a-z0-9_\-]{3,}", normalized)
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    ordered: list[str] = []
    seen: set[str] = set()
    for term in ascii_terms + cjk_terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered[:28]


def _score_snippet(snippet: AttachmentSnippet, query_terms: list[str]) -> int:
    if not query_terms:
        return 0

    score = 0
    haystack = f"{snippet.attachment_name.lower()}\n{snippet.content.lower()}"
    for term in query_terms:
        lowered = term.lower()
        if lowered in haystack:
            score += 8
        elif lowered in snippet.attachment_name.lower():
            score += 5
        elif lowered in snippet.keywords:
            score += 3
    return score


def _group_by_attachment(snippets: list[AttachmentSnippet]) -> dict[str, list[AttachmentSnippet]]:
    grouped: dict[str, list[AttachmentSnippet]] = {}
    for snippet in snippets:
        grouped.setdefault(snippet.attachment_name, []).append(snippet)
    return grouped


def _literature_coverage_candidates(snippets: list[AttachmentSnippet]) -> list[AttachmentSnippet]:
    if not snippets:
        return []
    ordered = sorted(snippets, key=lambda item: item.chunk_index)
    anchors = [ordered[index] for index in _anchor_positions(len(ordered))]
    focus = sorted(
        ordered,
        key=lambda item: (_literature_section_score(item), -abs(item.chunk_index - ((len(ordered) + 1) / 2))),
        reverse=True,
    )
    spread = [ordered[index] for index in _spread_positions(len(ordered), target=min(6, len(ordered)))]
    return _dedupe_snippets(anchors + spread + focus)


def _anchor_positions(length: int) -> list[int]:
    if length <= 1:
        return [0]
    anchors = {0, length - 1, length // 2}
    if length >= 4:
        anchors.add(length // 4)
        anchors.add((3 * length) // 4)
    return sorted(index for index in anchors if 0 <= index < length)


def _spread_positions(length: int, target: int) -> list[int]:
    if length <= target:
        return list(range(length))
    positions: list[int] = []
    for offset in range(target):
        position = round(offset * (length - 1) / max(target - 1, 1))
        if position not in positions:
            positions.append(position)
    return positions


def _literature_section_score(snippet: AttachmentSnippet) -> int:
    text = snippet.content.lower()
    score = 0
    for token, weight in LITERATURE_SECTION_HINTS:
        if token in text:
            score += weight
    if _page_hint(snippet.content) == 1:
        score += 5
    if "references" in text or "acknowledg" in text:
        score -= 4
    return score


def _dedupe_snippets(snippets: list[AttachmentSnippet]) -> list[AttachmentSnippet]:
    ordered: list[AttachmentSnippet] = []
    seen: set[tuple[str, int]] = set()
    for snippet in snippets:
        key = (snippet.attachment_name, snippet.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(snippet)
    return ordered


def _page_hint(text: str) -> int | None:
    match = re.search(r"\[Page\s+(\d+)\]", text)
    if match is None:
        return None
    return int(match.group(1))


def _format_page_summary(pages: list[int]) -> str:
    if not pages:
        return "unknown"
    unique = sorted(set(pages))
    if len(unique) <= 6:
        return ", ".join(str(page) for page in unique)
    head = ", ".join(str(page) for page in unique[:3])
    tail = ", ".join(str(page) for page in unique[-2:])
    return f"{head}, ..., {tail}"
