from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import DiscussionMessage, ProviderConfig
from .state import ApprovalRecord, DiscussionState, ExperimentRunRecord, PaperRecord, ProjectStateManager, WorkflowTask
from .workflow_config import WorkflowConfig, load_workflow_config
from .workflow_graph import build_workflow_graph, render_workflow_graph_mermaid, workflow_policy_snapshot


MINUTES_DIR = Path("meeting_minutes")
STATE_MANAGER = ProjectStateManager()


def save_discussion_outputs(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    messages: list[DiscussionMessage],
    literature_review_text: str,
    summary_text: str,
    minutes_text: str,
    cancelled: bool,
    meeting_state: DiscussionState | None = None,
    workflow_config: WorkflowConfig | None = None,
) -> tuple[Path | None, Path, Path]:
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    runtime_config = workflow_config or load_workflow_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    literature_path = MINUTES_DIR / f"literature_review_{timestamp}.md" if literature_review_text.strip() else None
    minutes_path = MINUTES_DIR / f"meeting_minutes_{timestamp}.md"
    report_path = MINUTES_DIR / f"research_report_{timestamp}.md"
    _persist_workflow_policy_artifacts(
        timestamp=timestamp,
        runtime_config=runtime_config,
        meeting_state=meeting_state,
    )
    if meeting_state is not None:
        STATE_MANAGER.update_summary(meeting_state, summary_text)
        if literature_path is not None:
            STATE_MANAGER.record_artifact(
                meeting_state,
                artifact_type="literature_review",
                title="Literature Review",
                path=str(literature_path),
                preview=_excerpt(literature_review_text),
            )
        STATE_MANAGER.record_artifact(
            meeting_state,
            artifact_type="meeting_minutes",
            title="Meeting Minutes",
            path=str(minutes_path),
            preview=_excerpt(minutes_text),
        )
        STATE_MANAGER.record_artifact(
            meeting_state,
            artifact_type="research_report",
            title="Research Report",
            path=str(report_path),
            preview=_excerpt(summary_text),
        )

    if literature_path is not None:
        literature_path.write_text(
            _render_literature_review_document(
                user_request=user_request,
                providers=providers,
                literature_review_text=literature_review_text,
                workflow_config=runtime_config,
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
            workflow_config=runtime_config,
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
            workflow_config=runtime_config,
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
    meeting_state: DiscussionState | None = None,
    workflow_config: WorkflowConfig | None = None,
) -> Path:
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    runtime_config = workflow_config or load_workflow_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_path = MINUTES_DIR / f"discussion_failure_{timestamp}.md"
    _persist_workflow_policy_artifacts(
        timestamp=timestamp,
        runtime_config=runtime_config,
        meeting_state=meeting_state,
    )
    if meeting_state is not None:
        STATE_MANAGER.record_artifact(
            meeting_state,
            artifact_type="failure_snapshot",
            title="Discussion Failure Snapshot",
            path=str(failure_path),
            preview=_excerpt(error_text, limit=240),
        )
    failure_path.write_text(
        _render_failure_document(
            user_request=user_request,
            providers=providers,
            messages=messages,
            status_lines=status_lines,
            error_text=error_text,
            literature_review_text=literature_review_text,
            meeting_state=meeting_state,
            workflow_config=runtime_config,
        ),
        encoding="utf-8",
    )
    return failure_path


def _active_roles(providers: list[ProviderConfig], workflow_config: WorkflowConfig | None = None) -> list[str]:
    lines: list[str] = []
    for provider in providers:
        if not provider.enabled or not provider.api_key:
            continue
        if workflow_config is not None and not workflow_config.is_role_enabled(provider.duty):
            continue
        role_config = workflow_config.role_config(provider.duty) if workflow_config is not None else None
        role_label = role_config.label if role_config is not None else provider.duty
        lines.append(
            f"- {provider.name} | Duty: {provider.duty} | Team role: {role_label} | Specialty: {provider.specialty or 'Not set'} | {provider.model}"
        )
    return lines


def _render_transcript(messages: list[DiscussionMessage], *, include_role_labels: bool) -> str:
    transcript = "\n\n".join(
        (
            f"### {_transcript_header(message, include_role_labels=include_role_labels)}\n\n"
            f"{message.content}"
        )
        for message in messages
    )
    return transcript or "No discussion transcript was captured."


def _render_literature_review_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    literature_review_text: str,
    workflow_config: WorkflowConfig,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers, workflow_config)
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
    meeting_state: DiscussionState | None,
    workflow_config: WorkflowConfig,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers, workflow_config)
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
        f"{_render_transcript(messages, include_role_labels=workflow_config.notes.include_role_labels)}\n"
    )


def _render_report_document(
    *,
    user_request: str,
    providers: list[ProviderConfig],
    report_text: str,
    cancelled: bool,
    meeting_state: DiscussionState | None,
    workflow_config: WorkflowConfig,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers, workflow_config)
    status = "Cancelled" if cancelled else "Completed"
    return (
        "# Research Report\n\n"
        f"- Created at: {created_at}\n"
        f"- Status: {status}\n\n"
        "## User Task\n\n"
        f"{user_request}\n\n"
        "## Active Roles\n\n"
        f"{chr(10).join(active_roles) or '- None'}\n\n"
        f"{_render_state_sections(meeting_state, include_consensus=workflow_config.report.include_consensus, include_open_questions=workflow_config.report.include_open_questions, include_action_items=workflow_config.report.include_action_items)}"
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
    meeting_state: DiscussionState | None,
    workflow_config: WorkflowConfig,
) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_roles = _active_roles(providers, workflow_config)
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
        f"{_render_transcript(messages, include_role_labels=workflow_config.notes.include_role_labels)}\n"
    )


def _render_state_sections(
    meeting_state: DiscussionState | None,
    *,
    include_consensus: bool = True,
    include_open_questions: bool = True,
    include_action_items: bool = True,
) -> str:
    if meeting_state is None:
        return ""

    consensus = _render_named_items(meeting_state.stable_consensus[-8:], "Consensus")
    conflicts = _render_named_items(meeting_state.conflicts[-8:], "Conflict")
    questions = _render_named_items(meeting_state.open_questions[-8:], "Open Question")
    action_items = _render_named_items(meeting_state.action_items[-8:], "Action Item")
    checkpoints = "\n".join(
        f"- {_checkpoint_label(checkpoint.checkpoint_id)} | Round {checkpoint.round_index} | {checkpoint.label} | {checkpoint.summary}"
        for checkpoint in meeting_state.checkpoints[-8:]
    ) or "- None"
    evidence = "\n".join(
        f"- {card.display_label or _fallback_evidence_label(card.evidence_id, card.source)} | Round {card.workpackage_index} | {card.summary}"
        for card in meeting_state.evidence_cards[-12:]
    ) or "- None"
    sources = "\n".join(f"- {item}" for item in meeting_state.uploaded_sources) or "- None"
    tasks = _render_workflow_tasks(meeting_state.workflow_tasks[-8:])
    stages = _render_workflow_stages(meeting_state)
    artifacts = _render_generated_artifacts(meeting_state)
    papers = _render_literature_library(meeting_state.literature_library[-8:])
    experiments = _render_experiment_runs(meeting_state.experiment_runs[-8:])
    approvals = _render_approvals(meeting_state.approval_records[-8:])

    return (
        "## Structured Meeting State\n\n"
        f"- Topic: {meeting_state.topic or 'Not distilled'}\n"
        f"- User Question: {meeting_state.user_question or 'Not recorded'}\n"
        f"- Domain: {meeting_state.domain or 'Undetermined'}\n"
        f"- Goal: {meeting_state.goal or 'Not distilled'}\n"
        f"- Summary: {meeting_state.summary or 'Not distilled'}\n"
        f"- Current Round: {meeting_state.current_round}\n"
        f"- Current Stage: {meeting_state.current_stage or 'Not recorded'}\n\n"
        "### Workflow Stages\n\n"
        f"{stages}\n\n"
        "### Uploaded Sources\n\n"
        f"{sources}\n\n"
        f"{'### Stable Consensus\\n\\n' + consensus + '\\n\\n' if include_consensus else ''}"
        "### Active Conflicts\n\n"
        f"{conflicts}\n\n"
        f"{'### Open Questions\\n\\n' + questions + '\\n\\n' if include_open_questions else ''}"
        f"{'### Action Items\\n\\n' + action_items + '\\n\\n' if include_action_items else ''}"
        "### Workflow Tasks\n\n"
        f"{tasks}\n\n"
        "### Literature Library\n\n"
        f"{papers}\n\n"
        "### Experiment Runs\n\n"
        f"{experiments}\n\n"
        "### Approval History\n\n"
        f"{approvals}\n\n"
        "### Checkpoints\n\n"
        f"{checkpoints}\n\n"
        "### Evidence Ledger\n\n"
        f"{evidence}\n\n"
        "### Generated Artifacts\n\n"
        f"{artifacts}\n\n"
    )


def _transcript_header(message: DiscussionMessage, *, include_role_labels: bool) -> str:
    header_parts = [message.speaker]
    if include_role_labels:
        header_parts.append(message.duty or "Unlabeled duty")
    header_parts.append(f"Round {message.round_index}")
    header_parts.append(message.stage or "discussion")
    return " | ".join(header_parts)


def _render_named_items(items: list[str], label: str) -> str:
    trimmed = [item for item in items if item.strip()]
    if not trimmed:
        return "- None"
    return "\n".join(f"- {label} {index}: {item}" for index, item in enumerate(trimmed, start=1))


def _checkpoint_label(raw_id: str) -> str:
    digits = "".join(char for char in raw_id if char.isdigit()) or raw_id
    return f"Checkpoint {digits}"


def _fallback_evidence_label(evidence_id: str, source: str) -> str:
    digits = "".join(char for char in evidence_id if char.isdigit()) or evidence_id
    cleaned_source = source.replace("|", ",").strip()
    return f"Evidence {digits} ({cleaned_source})" if cleaned_source else f"Evidence {digits}"


def _render_workflow_tasks(tasks: list[WorkflowTask]) -> str:
    if not tasks:
        return "- None"
    return "\n".join(
        f"- {_task_display_prefix(task)} | {task.title} | status={task.status} | owner={task.owner_name or 'TBD'} | reviewer={task.reviewer_name or 'None'}"
        for task in tasks
    )


def _task_display_prefix(task: WorkflowTask) -> str:
    if task.source_kind == "followup":
        return f"Closure Round {task.round_index}"
    return f"Main Round {task.round_index}"


def _render_workflow_stages(meeting_state: DiscussionState) -> str:
    if not meeting_state.workflow_stage_records:
        return "- None"
    return "\n".join(
        (
            f"- {record.stage_key} | {record.stage_label} | status={record.status}"
            + (f" | notes={_excerpt(record.notes, limit=160)}" if record.notes.strip() else "")
        )
        for record in meeting_state.workflow_stage_records[-8:]
    )


def _render_generated_artifacts(meeting_state: DiscussionState) -> str:
    if not meeting_state.generated_artifacts:
        return "- None"
    return "\n".join(
        f"- {artifact.artifact_type} | {artifact.title} | {artifact.path}"
        for artifact in meeting_state.generated_artifacts[-8:]
    )


def _render_literature_library(papers: list[PaperRecord]) -> str:
    if not papers:
        return "- None"
    return "\n".join(
        f"- {paper.title} | {paper.paper_id} | authors={', '.join(paper.authors[:3]) or 'Unknown'} | "
        f"pdf={paper.local_pdf_path or 'not downloaded'} | bibkey={paper.bibtex_key or 'n/a'}"
        for paper in papers
    )


def _render_experiment_runs(runs: list[ExperimentRunRecord]) -> str:
    if not runs:
        return "- None"
    return "\n".join(
        f"- {run.run_id} | mode={run.run_mode} | status={run.status} | script={run.script_path} | "
        f"python={run.interpreter_path or 'n/a'} | cwd={run.working_directory or 'n/a'} | "
        f"compile_rc={run.compile_returncode} | run_rc={run.runtime_returncode} | log={run.log_path or 'n/a'}"
        for run in runs
    )


def _render_approvals(approvals: list[ApprovalRecord]) -> str:
    if not approvals:
        return "- None"
    return "\n".join(
        f"- {approval.approval_type} | scope={approval.scope} | granted={'yes' if approval.granted else 'no'} | {approval.created_at}"
        for approval in approvals
    )


def _persist_workflow_policy_artifacts(
    *,
    timestamp: str,
    runtime_config: WorkflowConfig,
    meeting_state: DiscussionState | None,
) -> None:
    workflow_graph = build_workflow_graph(runtime_config.workflow_template)
    policy_snapshot = workflow_policy_snapshot(runtime_config)
    policy_path = MINUTES_DIR / f"workflow_policy_{timestamp}.json"
    graph_path = MINUTES_DIR / f"workflow_graph_{timestamp}.json"
    mermaid_path = MINUTES_DIR / f"workflow_graph_{timestamp}.mmd"

    policy_path.write_text(json.dumps(policy_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    graph_path.write_text(json.dumps(workflow_graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    mermaid_path.write_text(render_workflow_graph_mermaid(workflow_graph), encoding="utf-8")

    STATE_MANAGER.record_artifact(
        meeting_state,
        artifact_type="workflow_policy_snapshot",
        title="Workflow Policy Snapshot",
        path=str(policy_path),
        preview="Grouped workflow policy parameters (phi_disc, phi_ckpt, phi_ctx, phi_out, phi_ctrl).",
    )
    STATE_MANAGER.record_artifact(
        meeting_state,
        artifact_type="workflow_graph",
        title="Workflow Graph JSON",
        path=str(graph_path),
        preview=f"{len(workflow_graph.nodes)} nodes | {len(workflow_graph.edges)} edges",
    )
    STATE_MANAGER.record_artifact(
        meeting_state,
        artifact_type="workflow_graph_mermaid",
        title="Workflow Graph Mermaid",
        path=str(mermaid_path),
        preview=workflow_graph.name,
    )


def _excerpt(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."
