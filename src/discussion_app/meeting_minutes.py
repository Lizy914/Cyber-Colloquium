from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import DiscussionMessage, MeetingState, ProviderConfig


MINUTES_DIR = Path("meeting_minutes")


def save_discussion_outputs(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    messages: list[DiscussionMessage],
    literature_review_text: str,
    summary_text: str,
    minutes_text: str,
    cancelled: bool,
    meeting_state: MeetingState | None = None,
) -> tuple[Path | None, Path, Path]:
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    literature_path = MINUTES_DIR / f"literature_review_{timestamp}.md" if literature_review_text.strip() else None
    minutes_path = MINUTES_DIR / f"meeting_minutes_{timestamp}.md"
    report_path = MINUTES_DIR / f"research_report_{timestamp}.md"

    if literature_path is not None:
        literature_path.write_text(
            _render_literature_review_document(
                user_request=user_request,
                providers=providers,
                literature_review_text=literature_review_text,
            ),
            encoding="utf-8",
        )

    minutes_path.write_text(
        _render_minutes_document(
            user_request=user_request,
            providers=providers,
            messages=messages,
            minutes_text=minutes_text,
            cancelled=cancelled,
            meeting_state=meeting_state,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        _render_report_document(
            user_request=user_request,
            providers=providers,
            report_text=summary_text,
            cancelled=cancelled,
            meeting_state=meeting_state,
        ),
        encoding="utf-8",
    )
    return literature_path, minutes_path, report_path


def save_failure_snapshot(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    messages: list[DiscussionMessage],
    status_lines: list[str],
    error_text: str,
    literature_review_text: str = "",
    meeting_state: MeetingState | None = None,
) -> Path:
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_path = MINUTES_DIR / f"discussion_failure_{timestamp}.md"
    failure_path.write_text(
        _render_failure_document(
            user_request=user_request,
            providers=providers,
            messages=messages,
            status_lines=status_lines,
            error_text=error_text,
            literature_review_text=literature_review_text,
            meeting_state=meeting_state,
        ),
        encoding="utf-8",
    )
    return failure_path


def _active_roles(providers: list[ProviderConfig]) -> list[str]:
    return [
        f"- {provider.name} | {provider.duty} | Specialty: {provider.specialty or 'Not set'} | {provider.model}"
        for provider in providers
        if provider.enabled and provider.api_key
    ]


def _render_transcript(messages: list[DiscussionMessage]) -> str:
    return "\n\n".join(
        f"### {message.speaker} | {message.duty or 'Unlabeled duty'} | Task {message.round_index} | {message.stage or 'discussion'}\n\n{message.content}"
        for message in messages
    ) or "No discussion transcript was captured."


def _render_literature_review_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    literature_review_text: str,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers)
    return (
        "# Literature Review\n\n"
        f"- Created at: {created_at}\n\n"
        "## User Task\n\n"
        f"{user_request}\n\n"
        "## Active Roles\n\n"
        f"{chr(10).join(active_roles) or '- None'}\n\n"
        "## Review Text\n\n"
        f"{literature_review_text or 'No literature review was produced.'}\n"
    )


def _render_minutes_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    messages: list[DiscussionMessage],
    minutes_text: str,
    cancelled: bool,
    meeting_state: MeetingState | None,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers)
    status = "Cancelled" if cancelled else "Completed"
    return (
        "# Meeting Minutes\n\n"
        f"- Created at: {created_at}\n"
        f"- Status: {status}\n\n"
        "## User Task\n\n"
        f"{user_request}\n\n"
        "## Active Roles\n\n"
        f"{chr(10).join(active_roles) or '- None'}\n\n"
        f"{_render_state_sections(meeting_state)}"
        "## Minutes Body\n\n"
        f"{minutes_text}\n\n"
        "## Discussion Transcript\n\n"
        f"{_render_transcript(messages)}\n"
    )


def _render_report_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    report_text: str,
    cancelled: bool,
    meeting_state: MeetingState | None,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers)
    status = "Cancelled" if cancelled else "Completed"
    return (
        "# Research Report\n\n"
        f"- Created at: {created_at}\n"
        f"- Status: {status}\n\n"
        "## User Task\n\n"
        f"{user_request}\n\n"
        "## Active Roles\n\n"
        f"{chr(10).join(active_roles) or '- None'}\n\n"
        f"{_render_state_sections(meeting_state)}"
        "## Final Report\n\n"
        f"{report_text or 'No report was produced.'}\n"
    )


def _render_failure_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    messages: list[DiscussionMessage],
    status_lines: list[str],
    error_text: str,
    literature_review_text: str,
    meeting_state: MeetingState | None,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers)
    status_block = "\n".join(f"- {line}" for line in status_lines[-20:]) or "- No status messages were captured."
    error_block = error_text.strip() or "No exception text was captured."
    literature_block = literature_review_text.strip() or "No literature review output was available before failure."
    return (
        "# Discussion Failure Snapshot\n\n"
        f"- Created at: {created_at}\n"
        "- Status: Failed\n\n"
        "## User Task\n\n"
        f"{user_request or 'Not recorded'}\n\n"
        "## Active Roles\n\n"
        f"{chr(10).join(active_roles) or '- None'}\n\n"
        f"{_render_state_sections(meeting_state)}"
        "## Recent System Status\n\n"
        f"{status_block}\n\n"
        "## Literature Review Output Before Failure\n\n"
        f"{literature_block}\n\n"
        "## Exception\n\n"
        "```text\n"
        f"{error_block}\n"
        "```\n\n"
        "## Transcript Before Failure\n\n"
        f"{_render_transcript(messages)}\n"
    )


def _render_state_sections(meeting_state: MeetingState | None) -> str:
    if meeting_state is None:
        return ""

    consensus = _render_named_items(meeting_state.stable_consensus[-8:], "Consensus")
    conflicts = _render_named_items(meeting_state.conflicts[-8:], "Conflict")
    questions = _render_named_items(meeting_state.open_questions[-8:], "Open Question")
    checkpoints = "\n".join(
        f"- {_checkpoint_label(checkpoint.checkpoint_id)} | Task {checkpoint.workpackage_index} | {checkpoint.label} | {checkpoint.summary}"
        for checkpoint in meeting_state.checkpoints[-8:]
    ) or "- None"
    evidence = "\n".join(
        f"- {card.display_label or _fallback_evidence_label(card.evidence_id, card.source)} | Task {card.workpackage_index} | {card.summary}"
        for card in meeting_state.evidence_cards[-12:]
    ) or "- None"

    return (
        "## Structured Meeting State\n\n"
        f"- Topic: {meeting_state.topic or 'Not distilled'}\n"
        f"- Domain: {meeting_state.domain or 'Undetermined'}\n"
        f"- Goal: {meeting_state.goal or 'Not distilled'}\n"
        f"- Current Stage: {meeting_state.current_stage or 'Not recorded'}\n\n"
        "### Stable Consensus\n\n"
        f"{consensus}\n\n"
        "### Active Conflicts\n\n"
        f"{conflicts}\n\n"
        "### Open Questions\n\n"
        f"{questions}\n\n"
        "### Checkpoints\n\n"
        f"{checkpoints}\n\n"
        "### Evidence Ledger\n\n"
        f"{evidence}\n\n"
    )


def _render_named_items(items: list[str], label: str) -> str:
    trimmed = [item for item in items if item.strip()]
    if not trimmed:
        return "- None"
    return "\n".join(f"- {label} {index}: {item}" for index, item in enumerate(trimmed, start=1))


def _checkpoint_label(raw_id: str) -> str:
    digits = ''.join(char for char in raw_id if char.isdigit()) or raw_id
    return f"Checkpoint {digits}"


def _fallback_evidence_label(evidence_id: str, source: str) -> str:
    digits = ''.join(char for char in evidence_id if char.isdigit()) or evidence_id
    cleaned_source = source.replace('|', ',').strip()
    return f"Evidence {digits} ({cleaned_source})" if cleaned_source else f"Evidence {digits}"
