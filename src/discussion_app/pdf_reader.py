from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .attachments import encode_image_as_data_url
from .llm_client import LLMError, OpenAICompatibleClient
from .models import AttachmentPayload, ProviderConfig, ReaderReference


PDF_READER_DIR = Path("pdf_reader")
MAX_SECTION_COUNT = 24
MAX_SECTION_CHARS = 5200
MAX_FIGURE_COUNT = 8
MAX_FORMULA_COUNT = 18
MIN_FIGURE_WIDTH = 220
MIN_FIGURE_HEIGHT = 180
MIN_FIGURE_AREA = 90000
SECTION_SUMMARY_SECTIONS = [
    "## Section",
    "## Summary",
    "## Technical Details",
    "## Open Questions",
]
FIGURE_SUMMARY_SECTIONS = [
    "## Figure",
    "## What It Shows",
    "## Why It Matters",
    "## Reusable Notes",
]
OVERVIEW_SECTIONS = [
    "## Document Overview",
    "## Reusable Section Map",
    "## Reading Risks",
]
SECTION_HEADING_HINTS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "framework",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgment",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "index terms",
}


@dataclass
class PdfSection:
    section_id: str
    heading: str
    page_start: int
    page_end: int
    content: str


@dataclass
class PdfSectionDigest:
    section_id: str
    heading: str
    page_start: int
    page_end: int
    summary_markdown: str


@dataclass
class PdfFigure:
    figure_id: str
    page_number: int
    image_name: str
    image_path: str
    width: int
    height: int
    caption: str = ""
    summary_markdown: str = ""


@dataclass
class PdfReaderBuildResult:
    source_pdf: str
    index_path: str
    digest_markdown_path: str
    digest_json_path: str
    mode: str
    section_count: int


class PdfReaderBuilder:
    def __init__(self, provider: ProviderConfig | None) -> None:
        self.provider = provider

    def build_many(
        self,
        attachments: list[AttachmentPayload],
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> list[PdfReaderBuildResult]:
        results: list[PdfReaderBuildResult] = []
        for attachment in attachments:
            if attachment.kind != "pdf":
                continue
            if on_status is not None:
                on_status(f"Preparing PDF reader workspace for {attachment.display_name}")
            results.append(self.build_for_attachment(attachment, on_status=on_status))
        return results

    def build_for_attachment(
        self,
        attachment: AttachmentPayload,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> PdfReaderBuildResult:
        pages = _extract_pdf_pages(attachment.path)
        sections = _detect_pdf_sections(pages)
        figures = _extract_pdf_figures(attachment.path, pages)
        formula_candidates = _extract_formula_candidates(sections)
        index_payload = _build_index_payload(attachment.path, pages, sections)
        index_path = save_pdf_index_payload(attachment.path, index_payload)

        digests: list[PdfSectionDigest] = []
        mode = "index-only"
        overview_markdown = "No LLM-assisted overview was generated."

        if self.provider is not None and self.provider.api_key:
            mode = f"llm:{self.provider.name}"
            for position, section in enumerate(sections, start=1):
                if on_status is not None:
                    on_status(
                        f"Summarizing {attachment.display_name} section {position}/{len(sections)}: {section.heading}"
                    )
                summary_markdown = self._summarize_section(attachment.display_name, section)
                digests.append(
                    PdfSectionDigest(
                        section_id=section.section_id,
                        heading=section.heading,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        summary_markdown=summary_markdown,
                    )
                )
            if self.provider.supports_vision:
                for position, figure in enumerate(figures, start=1):
                    if on_status is not None:
                        on_status(
                            f"Summarizing {attachment.display_name} figure {position}/{len(figures)} from page {figure.page_number}"
                        )
                    figure.summary_markdown = self._summarize_figure(attachment.display_name, figure)
            else:
                for figure in figures:
                    figure.summary_markdown = _fallback_figure_summary(figure)
            if digests:
                overview_markdown = self._summarize_document(attachment.display_name, digests, figures)
        else:
            digests = [
                PdfSectionDigest(
                    section_id=section.section_id,
                    heading=section.heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    summary_markdown=(
                        "## Section\n"
                        f"{section.heading} (pages {section.page_start}-{section.page_end})\n\n"
                        "## Summary\n"
                        "No LLM summary was generated. Configure a literature-review provider to create section summaries.\n\n"
                        "## Technical Details\n"
                        "Index-only mode.\n\n"
                        "## Open Questions\n"
                        "Summary generation was skipped because no literature-review provider was available."
                    ),
                )
                for section in sections
            ]
            figures = [
                PdfFigure(
                    figure_id=figure.figure_id,
                    page_number=figure.page_number,
                    image_name=figure.image_name,
                    image_path=figure.image_path,
                    width=figure.width,
                    height=figure.height,
                    caption=figure.caption,
                    summary_markdown=_fallback_figure_summary(figure),
                )
                for figure in figures
            ]

        digest_payload = _build_digest_payload(
            attachment.path,
            provider=self.provider,
            mode=mode,
            pages=pages,
            sections=sections,
            digests=digests,
            figures=figures,
            formula_candidates=formula_candidates,
            overview_markdown=overview_markdown,
        )
        digest_json_path, digest_markdown_path = save_pdf_digest_payload(attachment.path, digest_payload)
        return PdfReaderBuildResult(
            source_pdf=str(attachment.path),
            index_path=str(index_path),
            digest_markdown_path=str(digest_markdown_path),
            digest_json_path=str(digest_json_path),
            mode=mode,
            section_count=len(sections),
        )

    def _summarize_section(self, pdf_name: str, section: PdfSection) -> str:
        prompt = (
            f"PDF: {pdf_name}\n"
            f"Section ID: {section.section_id}\n"
            f"Heading: {section.heading}\n"
            f"Pages: {section.page_start}-{section.page_end}\n\n"
            "Read the section text below and create a reusable section digest for later academic discussion.\n\n"
            f"Section text:\n{_truncate_text(section.content, MAX_SECTION_CHARS)}"
        )
        return self._chat_with_repair(
            system_prompt=(
                "You are building a reusable PDF reading cache for a multi-model academic discussion app. "
                "Summarize one section at a time. Stay faithful to the supplied text only."
            ),
            user_prompt=prompt,
            max_tokens=700,
            required_sections=SECTION_SUMMARY_SECTIONS,
        )

    def _summarize_document(self, pdf_name: str, digests: list[PdfSectionDigest], figures: list[PdfFigure]) -> str:
        packet = "\n\n".join(
            f"### {digest.section_id} | {digest.heading} | pages {digest.page_start}-{digest.page_end}\n\n{digest.summary_markdown}"
            for digest in digests
        )
        figure_packet = "\n".join(
            f"- {figure.figure_id} | page {figure.page_number} | caption: {figure.caption or 'No caption detected'} | summary: {_truncate_text(_collapse_whitespace(figure.summary_markdown), 320)}"
            for figure in figures
        )
        prompt = (
            f"PDF: {pdf_name}\n\n"
            "You will receive section digests for one PDF. Build a compact document-level overview for later retrieval.\n\n"
            f"Section digests:\n\n{_truncate_text(packet, 12000)}\n\n"
            f"Figure index:\n{figure_packet or 'No extracted figures were available.'}"
        )
        return self._chat_with_repair(
            system_prompt=(
                "You are writing a reusable PDF reading overview for an academic discussion app. "
                "Keep it factual, section-aware, and explicit about gaps."
            ),
            user_prompt=prompt,
            max_tokens=900,
            required_sections=OVERVIEW_SECTIONS,
        )

    def _summarize_figure(self, pdf_name: str, figure: PdfFigure) -> str:
        prompt = (
            f"PDF: {pdf_name}\n"
            f"Figure ID: {figure.figure_id}\n"
            f"Page: {figure.page_number}\n"
            f"Detected caption: {figure.caption or 'No caption detected'}\n"
            f"Image name: {figure.image_name}\n\n"
            "Describe the figure or flowchart. Focus on pipelines, modules, arrows, inputs/outputs, and what later experts can reuse."
        )
        image_attachment = AttachmentPayload(
            path=Path(figure.image_path),
            kind="image",
            content=encode_image_as_data_url(Path(figure.image_path)),
            display_name=Path(figure.image_path).name,
        )
        return self._chat_with_repair(
            system_prompt=(
                "You are building a reusable PDF figure cache for an academic discussion app. "
                "Describe only what is visible or explicitly captioned. Do not invent unreadable labels."
            ),
            user_prompt=prompt,
            max_tokens=420,
            required_sections=FIGURE_SUMMARY_SECTIONS,
            attachments=[image_attachment],
        )

    def _chat_with_repair(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        required_sections: list[str],
        attachments: list[AttachmentPayload] | None = None,
    ) -> str:
        content = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            attachments=attachments,
        )
        if not _response_needs_repair(content, required_sections):
            return content
        repair_prompt = (
            f"{user_prompt}\n\n"
            f"Your previous answer did not include all required sections: {', '.join(required_sections)}. "
            "Rewrite it fully and keep every required section header exactly.\n\n"
            f"Previous answer:\n{_truncate_text(content, 1200)}"
        )
        repaired = self._chat(
            system_prompt=system_prompt,
            user_prompt=repair_prompt,
            max_tokens=max_tokens,
            attachments=attachments,
        )
        if not _response_needs_repair(repaired, required_sections):
            return repaired
        return content

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        attachments: list[AttachmentPayload] | None = None,
    ) -> str:
        if self.provider is None or not self.provider.api_key:
            return "[Call Failed] No literature-review provider is configured for PDF digest generation."
        client = OpenAICompatibleClient(self.provider)
        try:
            return client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                attachments=attachments,
                max_tokens=max_tokens,
                max_continuations=1,
            )
        except LLMError as exc:
            return f"[Call Failed] {exc}"


def has_pdf_reader_cache(path: Path) -> bool:
    digest_json_path, digest_markdown_path = _digest_paths(path)
    return digest_json_path.exists() and digest_markdown_path.exists()


def pdf_reader_status(path: Path) -> str:
    index_path = _index_path(path)
    digest_json_path, digest_markdown_path = _digest_paths(path)
    if digest_json_path.exists() and digest_markdown_path.exists():
        try:
            payload = json.loads(digest_json_path.read_text(encoding="utf-8"))
        except Exception:
            return "reader-ready"
        mode = str(payload.get("mode", "")).lower()
        if mode == "index-only":
            return "index-only"
        return "reader-ready"
    if index_path.exists():
        return "indexed"
    return "missing"


def render_cached_pdf_reader_context(attachments: list[AttachmentPayload], max_chars: int = 9000) -> str:
    blocks: list[str] = []
    used = 0
    for attachment in attachments:
        if attachment.kind != "pdf":
            continue
        digest_json_path, _ = _digest_paths(attachment.path)
        if not digest_json_path.exists():
            continue
        payload = json.loads(digest_json_path.read_text(encoding="utf-8"))
        block = _render_cached_digest_block(payload)
        if blocks and used + len(block) > max_chars:
            continue
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def load_pdf_reader_references(attachments: list[AttachmentPayload]) -> list[ReaderReference]:
    references: list[ReaderReference] = []
    for attachment in attachments:
        if attachment.kind != "pdf":
            continue
        digest_json_path, _ = _digest_paths(attachment.path)
        index_path = _index_path(attachment.path)
        if not digest_json_path.exists() or not index_path.exists():
            continue
        try:
            digest_payload = json.loads(digest_json_path.read_text(encoding="utf-8"))
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        references.extend(_section_references_from_payload(attachment.display_name, digest_payload, index_payload))
        references.extend(_figure_references_from_payload(attachment.display_name, digest_payload))
        references.extend(_formula_references_from_payload(attachment.display_name, index_payload))
    return references


def select_pdf_reader_references(
    references: list[ReaderReference],
    query: str,
    *,
    max_chars: int = 1800,
    max_items: int = 5,
) -> list[ReaderReference]:
    if not references:
        return []
    query_terms = _extract_reference_keywords(query)
    ranked = sorted(
        references,
        key=lambda item: (_score_reference(item, query_terms), _reference_priority(item), -(item.page_hint or 9999)),
        reverse=True,
    )
    selected: list[ReaderReference] = []
    used = 0
    for reference in ranked:
        if len(selected) >= max_items:
            break
        block_cost = len(reference.title) + len(reference.content) + len(reference.attachment_name) + 40
        if selected and used + block_cost > max_chars:
            continue
        selected.append(reference)
        used += block_cost
    return selected or ranked[: max_items]


def render_pdf_reader_references(references: list[ReaderReference], max_chars: int = 1800) -> str:
    if not references:
        return ""
    blocks: list[str] = []
    used = 0
    for reference in references:
        page_text = f"page {reference.page_hint}" if reference.page_hint is not None else "page unknown"
        block = (
            f"[{reference.ref_id} | {reference.kind.title()} | {reference.attachment_name} | {page_text}]\n"
            f"Title: {reference.title}\n"
            f"{reference.content}"
        )
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_reader_reference_attachments(
    references: list[ReaderReference],
    *,
    max_images: int = 2,
) -> list[AttachmentPayload]:
    attachments: list[AttachmentPayload] = []
    for reference in references:
        if reference.kind != "figure" or not reference.image_path:
            continue
        image_path = Path(reference.image_path)
        if not image_path.exists():
            continue
        attachments.append(
            AttachmentPayload(
                path=image_path,
                kind="image",
                content=encode_image_as_data_url(image_path),
                display_name=image_path.name,
            )
        )
        if len(attachments) >= max_images:
            break
    return attachments


def save_pdf_index_payload(path: Path, payload: dict) -> Path:
    workspace = _workspace_dir(path)
    workspace.mkdir(parents=True, exist_ok=True)
    index_path = _index_path(path)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def save_pdf_digest_payload(path: Path, payload: dict) -> tuple[Path, Path]:
    workspace = _workspace_dir(path)
    workspace.mkdir(parents=True, exist_ok=True)
    digest_json_path, digest_markdown_path = _digest_paths(path)
    digest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest_markdown_path.write_text(_render_digest_markdown(payload), encoding="utf-8")
    return digest_json_path, digest_markdown_path


def _build_index_payload(path: Path, pages: list[str], sections: list[PdfSection]) -> dict:
    return {
        "source_pdf": str(path),
        "source_name": path.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "page_count": len(pages),
        "section_count": len(sections),
        "formula_candidates": _extract_formula_candidates(sections),
        "sections": [
            {
                "section_id": section.section_id,
                "heading": section.heading,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "char_count": len(section.content),
                "preview": _truncate_text(_collapse_whitespace(section.content), 220),
            }
            for section in sections
        ],
    }


def _build_digest_payload(
    path: Path,
    *,
    provider: ProviderConfig | None,
    mode: str,
    pages: list[str],
    sections: list[PdfSection],
    digests: list[PdfSectionDigest],
    figures: list[PdfFigure],
    formula_candidates: list[dict],
    overview_markdown: str,
) -> dict:
    return {
        "source_pdf": str(path),
        "source_name": path.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider_name": provider.name if provider is not None else "",
        "provider_model": provider.model if provider is not None else "",
        "mode": mode,
        "page_count": len(pages),
        "section_count": len(sections),
        "figure_count": len(figures),
        "formula_count": len(formula_candidates),
        "overview_markdown": overview_markdown,
        "sections": [
            {
                "section_id": digest.section_id,
                "heading": digest.heading,
                "page_start": digest.page_start,
                "page_end": digest.page_end,
                "summary_markdown": digest.summary_markdown,
            }
            for digest in digests
        ],
        "figures": [
            {
                "figure_id": figure.figure_id,
                "page_number": figure.page_number,
                "image_name": figure.image_name,
                "image_path": figure.image_path,
                "width": figure.width,
                "height": figure.height,
                "caption": figure.caption,
                "summary_markdown": figure.summary_markdown,
            }
            for figure in figures
        ],
        "formula_candidates": formula_candidates,
    }


def _render_digest_markdown(payload: dict) -> str:
    lines = [
        "# PDF Reader Digest",
        "",
        f"- Source PDF: `{payload.get('source_pdf', '')}`",
        f"- Generated at: {payload.get('generated_at', '')}",
        f"- Provider: {payload.get('provider_name', 'N/A')} | {payload.get('provider_model', 'N/A')}",
        f"- Mode: {payload.get('mode', 'unknown')}",
        f"- Page count: {payload.get('page_count', 0)}",
        f"- Section count: {payload.get('section_count', 0)}",
        f"- Figure count: {payload.get('figure_count', 0)}",
        f"- Formula count: {payload.get('formula_count', 0)}",
        "",
        "## Section Index",
        "",
        "| No. | Heading | Pages |",
        "| --- | --- | --- |",
    ]
    for index, section in enumerate(payload.get("sections", []), start=1):
        lines.append(
            f"| {index} | {section.get('heading', '')} | {section.get('page_start', '?')}-{section.get('page_end', '?')} |"
        )
    lines.extend(
        [
            "",
            "## Overview",
            "",
            payload.get("overview_markdown", "No overview was generated."),
            "",
            "## Section Digests",
            "",
        ]
    )
    for index, section in enumerate(payload.get("sections", []), start=1):
        lines.extend(
            [
                f"### {index}. {section.get('heading', '')}",
                "",
                f"Pages: {section.get('page_start', '?')}-{section.get('page_end', '?')}",
                "",
                section.get("summary_markdown", "No summary was generated."),
            "",
        ]
    )
    figures = payload.get("figures", [])
    if figures:
        lines.extend(
            [
                "## Figure Index",
                "",
                "| No. | Page | Figure | Image |",
                "| --- | --- | --- | --- |",
            ]
        )
        for index, figure in enumerate(figures, start=1):
            lines.append(
                f"| {index} | {figure.get('page_number', '?')} | {figure.get('caption', 'No caption detected')} | `{figure.get('image_name', '')}` |"
            )
        lines.append("")
        lines.append("## Figure Digests")
        lines.append("")
        for index, figure in enumerate(figures, start=1):
            lines.extend(
                [
                    f"### Figure {index}. {figure.get('figure_id', '')}",
                    "",
                    f"Page: {figure.get('page_number', '?')} | Image: `{figure.get('image_name', '')}`",
                    "",
                    f"Caption: {figure.get('caption', 'No caption detected')}",
                    "",
                    figure.get("summary_markdown", "No figure summary was generated."),
                    "",
                ]
            )
    formulas = payload.get("formula_candidates", [])
    if formulas:
        lines.extend(
            [
                "## Formula Index",
                "",
                "| No. | Page | Formula |",
                "| --- | --- | --- |",
            ]
        )
        for index, formula in enumerate(formulas, start=1):
            lines.append(
                f"| {index} | {formula.get('page_hint', '?')} | {formula.get('text', '')} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_cached_digest_block(payload: dict) -> str:
    lines = [
        f"[Cached PDF Reader Digest: {payload.get('source_name', 'PDF')}]",
        f"Mode: {payload.get('mode', 'unknown')}",
        f"Generated at: {payload.get('generated_at', '')}",
    ]
    overview = _truncate_text(_collapse_whitespace(payload.get("overview_markdown", "")), 900)
    if overview:
        lines.extend(["Overview:", overview])
    lines.append("Sections:")
    for section in payload.get("sections", [])[:12]:
        summary = _truncate_text(_collapse_whitespace(section.get("summary_markdown", "")), 520)
        lines.append(
            f"- {section.get('section_id', '')} | {section.get('heading', '')} | pages {section.get('page_start', '?')}-{section.get('page_end', '?')}: {summary}"
        )
    figures = payload.get("figures", [])
    if figures:
        lines.append("Figures:")
        for figure in figures[:4]:
            figure_summary = _truncate_text(_collapse_whitespace(figure.get("summary_markdown", "")), 360)
            lines.append(
                f"- {figure.get('figure_id', '')} | page {figure.get('page_number', '?')} | {figure.get('caption', 'No caption detected')}: {figure_summary}"
            )
    formula_candidates = payload.get("formula_candidates", [])
    if formula_candidates:
        lines.append("Formulas:")
        for item in formula_candidates[:4]:
            lines.append(
                f"- {item.get('formula_id', '')} | page {item.get('page_hint', '?')} | {item.get('text', '')}"
            )
    return "\n".join(lines)


def _section_references_from_payload(attachment_name: str, digest_payload: dict, index_payload: dict) -> list[ReaderReference]:
    section_lookup = {
        str(section.get("section_id", "")): section
        for section in index_payload.get("sections", [])
        if isinstance(section, dict)
    }
    references: list[ReaderReference] = []
    for section in digest_payload.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id", "")).strip()
        preview = ""
        source_section = section_lookup.get(section_id)
        if source_section is not None:
            preview = str(source_section.get("preview", "")).strip()
        summary = _collapse_whitespace(str(section.get("summary_markdown", "")).strip())
        content = _truncate_text(" ".join(part for part in [preview, summary] if part), 720)
        references.append(
            ReaderReference(
                ref_id=section_id or f"S{len(references) + 1}",
                attachment_name=attachment_name,
                kind="section",
                title=str(section.get("heading", "Section")).strip(),
                content=content,
                page_hint=_coerce_int(section.get("page_start")),
                keywords=_extract_reference_keywords(
                    f"{attachment_name} {section.get('heading', '')} {preview} {summary}"
                ),
            )
        )
    return references


def _figure_references_from_payload(attachment_name: str, digest_payload: dict) -> list[ReaderReference]:
    references: list[ReaderReference] = []
    for figure in digest_payload.get("figures", []):
        if not isinstance(figure, dict):
            continue
        caption = str(figure.get("caption", "")).strip() or str(figure.get("figure_id", "Figure")).strip()
        summary = _collapse_whitespace(str(figure.get("summary_markdown", "")).strip())
        references.append(
            ReaderReference(
                ref_id=str(figure.get("figure_id", f"F{len(references) + 1}")).strip(),
                attachment_name=attachment_name,
                kind="figure",
                title=caption,
                content=_truncate_text(summary, 640),
                page_hint=_coerce_int(figure.get("page_number")),
                image_path=str(figure.get("image_path", "")).strip(),
                keywords=_extract_reference_keywords(
                    f"{attachment_name} {caption} {figure.get('image_name', '')} {summary}"
                ),
            )
        )
    return references


def _formula_references_from_payload(attachment_name: str, index_payload: dict) -> list[ReaderReference]:
    references: list[ReaderReference] = []
    for item in index_payload.get("formula_candidates", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        formula_id = str(item.get("formula_id", f"M{len(references) + 1}")).strip()
        references.append(
            ReaderReference(
                ref_id=formula_id,
                attachment_name=attachment_name,
                kind="formula",
                title=f"Formula candidate {formula_id}",
                content=text,
                page_hint=_coerce_int(item.get("page_hint")),
                keywords=_extract_reference_keywords(f"{attachment_name} {text}"),
            )
        )
    return references


def _reference_priority(reference: ReaderReference) -> int:
    mapping = {"figure": 4, "formula": 3, "section": 2}
    return mapping.get(reference.kind, 1)


def _score_reference(reference: ReaderReference, query_terms: list[str]) -> int:
    if not query_terms:
        return 0
    haystack = f"{reference.attachment_name.lower()}\n{reference.title.lower()}\n{reference.content.lower()}"
    score = 0
    for term in query_terms:
        lowered = term.lower()
        if lowered in reference.title.lower():
            score += 10
        elif lowered in haystack:
            score += 7
        elif lowered in reference.keywords:
            score += 4
    if reference.kind == "figure" and any(token in haystack for token in ["flowchart", "architecture", "pipeline", "framework", "fig"]):
        score += 3
    if reference.kind == "formula" and any(token in query_terms for token in ["loss", "objective", "equation", "公式", "损失", "推导"]):
        score += 4
    return score


def _extract_pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(_normalize_pdf_page_text(text))
    return pages


def _extract_pdf_figures(path: Path, pages: list[str]) -> list[PdfFigure]:
    reader = PdfReader(str(path))
    figure_dir = _workspace_dir(path) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[PdfFigure] = []
    seen_hashes: set[str] = set()

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = pages[page_index - 1] if page_index - 1 < len(pages) else ""
        page_caption = _extract_page_figure_caption(page_text)
        for image_index, image in enumerate(list(page.images), start=1):
            pil_image = image.image
            width, height = pil_image.size
            if width < MIN_FIGURE_WIDTH or height < MIN_FIGURE_HEIGHT or width * height < MIN_FIGURE_AREA:
                continue
            image_hash = hashlib.sha1(image.data).hexdigest()
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)
            suffix = Path(image.name or f"figure_{image_index}.png").suffix or ".png"
            image_name = f"page_{page_index:02d}_figure_{image_index:02d}{suffix}"
            image_path = figure_dir / image_name
            try:
                pil_image.save(image_path)
            except Exception:
                image_path.write_bytes(image.data)
            figures.append(
                PdfFigure(
                    figure_id=f"F{len(figures) + 1}",
                    page_number=page_index,
                    image_name=image_name,
                    image_path=str(image_path),
                    width=width,
                    height=height,
                    caption=page_caption,
                )
            )
            if len(figures) >= MAX_FIGURE_COUNT:
                return figures
    return figures


def _detect_pdf_sections(pages: list[str]) -> list[PdfSection]:
    entries: list[tuple[int, str]] = []
    for page_number, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            cleaned = _collapse_whitespace(raw_line)
            if cleaned:
                entries.append((page_number, cleaned))

    if not entries:
        return [PdfSection(section_id="S1", heading="Document Overview", page_start=1, page_end=max(1, len(pages)), content="")]

    markers: list[tuple[int, int, str]] = []
    for index, (page_number, line) in enumerate(entries):
        if not _looks_like_section_heading(line):
            continue
        if markers and markers[-1][2].lower() == line.lower():
            continue
        markers.append((index, page_number, line))

    if not markers or markers[0][0] > 0:
        markers.insert(0, (0, entries[0][0], "Front Matter"))

    sections: list[PdfSection] = []
    for marker_index, (start_index, page_start, heading) in enumerate(markers):
        end_index = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(entries)
        if end_index <= start_index:
            continue
        section_lines = [line for _, line in entries[start_index:end_index]]
        content = "\n".join(section_lines).strip()
        if not content:
            continue
        page_end = entries[end_index - 1][0]
        sections.append(
            PdfSection(
                section_id=f"S{len(sections) + 1}",
                heading=heading,
                page_start=page_start,
                page_end=page_end,
                content=content,
            )
        )

    sections = _merge_small_sections(sections)
    if len(sections) > MAX_SECTION_COUNT:
        sections = sections[:MAX_SECTION_COUNT]
    return sections or [PdfSection(section_id="S1", heading="Document Overview", page_start=1, page_end=max(1, len(pages)), content="\n".join(page for page in pages if page))]


def _extract_page_figure_caption(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines:
        collapsed = _collapse_whitespace(line)
        if re.match(r"^(fig(?:ure)?\.?)\s*\d+", collapsed, re.IGNORECASE):
            return collapsed
    return ""


def _extract_formula_candidates(sections: list[PdfSection]) -> list[dict]:
    formulas: list[dict] = []
    seen: set[str] = set()
    for section in sections:
        for raw_line in section.content.splitlines():
            collapsed = _collapse_whitespace(raw_line)
            if not _looks_like_formula_line(collapsed):
                continue
            key = collapsed.lower()
            if key in seen:
                continue
            seen.add(key)
            formulas.append(
                {
                    "formula_id": f"M{len(formulas) + 1}",
                    "section_id": section.section_id,
                    "heading": section.heading,
                    "page_hint": section.page_start,
                    "text": _truncate_text(collapsed, 180),
                }
            )
            if len(formulas) >= MAX_FORMULA_COUNT:
                return formulas
    return formulas


def _looks_like_formula_line(text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 180:
        return False
    lowered = text.lower()
    if lowered.startswith(("fig", "figure", "table", "where ", "thus ", "because ")):
        return False
    operator_count = sum(text.count(symbol) for symbol in ("=", "+", "-", "/", "*", "^", "∑", "Σ", "λ", "μ"))
    token_matches = re.findall(r"\b[a-zA-Z]\w{0,3}\b", text)
    digit_count = sum(char.isdigit() for char in text)
    if operator_count >= 2 and token_matches and digit_count <= 18:
        return True
    if "=" in text and ("(" in text or ")" in text):
        return True
    return False


def _merge_small_sections(sections: list[PdfSection]) -> list[PdfSection]:
    if not sections:
        return sections
    merged: list[PdfSection] = []
    for section in sections:
        if merged and len(section.content) < 260 and section.page_end <= merged[-1].page_end + 1:
            previous = merged[-1]
            merged[-1] = PdfSection(
                section_id=previous.section_id,
                heading=f"{previous.heading} + {section.heading}",
                page_start=previous.page_start,
                page_end=section.page_end,
                content=f"{previous.content}\n\n{section.content}".strip(),
            )
            continue
        merged.append(section)
    for index, section in enumerate(merged, start=1):
        section.section_id = f"S{index}"
    return merged


def _looks_like_section_heading(line: str) -> bool:
    lowered = line.lower().strip()
    if len(line) < 3 or len(line) > 120:
        return False
    if lowered.startswith(("fig", "figure", "table", "http", "www.", "doi", "page ")):
        return False
    if "@" in lowered or lowered.count(",") > 3:
        return False
    if re.match(r"^\[?\d+\]?$", lowered):
        return False
    if lowered in SECTION_HEADING_HINTS:
        return True
    if re.match(r"^(?:\d+(?:\.\d+){0,3}|[ivxlcdm]+|[a-z])(?:[\.)])?\s+[a-z].*$", lowered, re.IGNORECASE):
        word_count = len(lowered.split())
        return 2 <= word_count <= 14
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio >= 0.72 and len(line.split()) <= 14


def _normalize_pdf_page_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fallback_figure_summary(figure: PdfFigure) -> str:
    return (
        "## Figure\n"
        f"{figure.figure_id} on page {figure.page_number}\n\n"
        "## What It Shows\n"
        f"Embedded figure extracted from the PDF. Caption: {figure.caption or 'No caption detected'}\n\n"
        "## Why It Matters\n"
        "This figure is now cached locally and can be referenced by the expert team when discussing pipelines, flowcharts, or module interactions.\n\n"
        "## Reusable Notes\n"
        "Enable a vision-capable literature-review model to generate a figure-level interpretation."
    )


def _response_needs_repair(content: str, required_sections: list[str]) -> bool:
    if not content or content.startswith("[Call Failed]"):
        return False
    if len(_collapse_whitespace(content)) < 80:
        return True
    return any(section not in content for section in required_sections)


def _workspace_dir(path: Path) -> Path:
    return PDF_READER_DIR / _slugify(path.stem)


def _index_path(path: Path) -> Path:
    return _workspace_dir(path) / "section_index.json"


def _digest_paths(path: Path) -> tuple[Path, Path]:
    workspace = _workspace_dir(path)
    return workspace / "section_digest.json", workspace / "section_digest.md"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "pdf"


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_reference_keywords(text: str) -> list[str]:
    normalized = (text or "").lower()
    ascii_terms = re.findall(r"[a-z0-9_\-]{3,}", normalized)
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
    ordered: list[str] = []
    seen: set[str] = set()
    for term in ascii_terms + cjk_terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered[:30]


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def pdf_reader_badge(path: Path) -> str:
    status = pdf_reader_status(path)
    mapping = {
        "reader-ready": "reader ready",
        "index-only": "index only",
        "indexed": "indexed",
    }
    return mapping.get(status, "")
