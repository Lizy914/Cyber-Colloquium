from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from .models import EvidenceCard, StructuredLogEntry


TASK_STATUS_PENDING = "pending"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    task_id: str
    title: str
    description: str = ""
    owner_name: str = ""
    reviewer_name: str = ""
    round_index: int = 0
    status: str = TASK_STATUS_PENDING
    source_kind: str = "assignment"
    notes: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def display_text(self) -> str:
        if self.description:
            return f"{self.title}: {self.description}"
        return self.title


@dataclass
class WorkflowStageRecord:
    stage_key: str
    stage_label: str
    status: str = TASK_STATUS_PENDING
    started_at: str = ""
    completed_at: str = ""
    notes: str = ""


@dataclass
class Checkpoint:
    checkpoint_id: str
    label: str
    workpackage_index: int
    round_index: int
    summary: str
    consensus_points: list[str] = field(default_factory=list)
    risks_or_disagreements: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    related_task_ids: list[str] = field(default_factory=list)

    @property
    def consensus(self) -> list[str]:
        return self.consensus_points

    @property
    def conflicts(self) -> list[str]:
        return self.risks_or_disagreements


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    title: str
    path: str
    created_at: str
    preview: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    published_at: str = ""
    updated_at: str = ""
    entry_url: str = ""
    pdf_url: str = ""
    local_pdf_path: str = ""
    bibtex_key: str = ""
    bibtex_entry: str = ""
    selection_reason: str = ""
    source_provider: str = "arXiv"


@dataclass
class ExperimentRunRecord:
    run_id: str
    script_path: str
    working_directory: str = ""
    interpreter_path: str = ""
    run_mode: str = "smoke"
    command: list[str] = field(default_factory=list)
    compile_returncode: int = 0
    runtime_returncode: int = 0
    log_path: str = ""
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    status: str = TASK_STATUS_PENDING
    authorized: bool = False
    created_at: str = ""


@dataclass
class ApprovalRecord:
    approval_id: str
    approval_type: str
    scope: str
    granted: bool
    details: str = ""
    created_at: str = ""


@dataclass
class DiscussionState:
    topic: str = ""
    user_question: str = ""
    uploaded_sources: list[str] = field(default_factory=list)
    summary: str = ""
    domain: str = ""
    goal: str = ""
    assignment_summary: str = ""
    coordination_summary: str = ""
    current_stage: str = ""
    current_question: str = ""
    current_round: int = 0
    rules: list[str] = field(default_factory=list)
    consensus_points: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    risks_or_disagreements: list[str] = field(default_factory=list)
    rejected_lines: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    log_entries: list[StructuredLogEntry] = field(default_factory=list)
    workflow_tasks: list[WorkflowTask] = field(default_factory=list)
    workflow_stage_records: list[WorkflowStageRecord] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    generated_artifacts: list[ArtifactRecord] = field(default_factory=list)
    literature_library: list[PaperRecord] = field(default_factory=list)
    experiment_runs: list[ExperimentRunRecord] = field(default_factory=list)
    approval_records: list[ApprovalRecord] = field(default_factory=list)
    execution_trace: list[dict[str, object]] = field(default_factory=list)
    token_usage: dict[str, object] = field(default_factory=dict)
    status: str = "initialized"

    @property
    def stable_consensus(self) -> list[str]:
        return self.consensus_points

    @stable_consensus.setter
    def stable_consensus(self, value: list[str]) -> None:
        self.consensus_points = value

    @property
    def conflicts(self) -> list[str]:
        return self.risks_or_disagreements

    @conflicts.setter
    def conflicts(self, value: list[str]) -> None:
        self.risks_or_disagreements = value


@dataclass
class ResearchProject:
    project_id: str
    topic: str
    user_question: str
    uploaded_sources: list[str]
    created_at: str
    language: str = "en"
    discussion_state: DiscussionState = field(default_factory=DiscussionState)


class ProjectStateStore(Protocol):
    def save(self, project: ResearchProject) -> None:
        ...

    def load(self, project_id: str) -> ResearchProject | None:
        ...


@dataclass
class InMemoryProjectStateStore:
    projects: dict[str, ResearchProject] = field(default_factory=dict)

    def save(self, project: ResearchProject) -> None:
        self.projects[project.project_id] = project

    def load(self, project_id: str) -> ResearchProject | None:
        return self.projects.get(project_id)


class ProjectStateManager:
    def __init__(self, store: ProjectStateStore | None = None) -> None:
        self.store = store or InMemoryProjectStateStore()

    def start_project(
        self,
        *,
        topic: str,
        user_question: str,
        uploaded_sources: list[str],
        language: str,
        rules: list[str],
        current_stage: str,
        current_question: str,
    ) -> ResearchProject:
        timestamp = _now_text()
        discussion_state = DiscussionState(
            topic=topic,
            user_question=user_question,
            uploaded_sources=list(uploaded_sources),
            goal=topic or user_question,
            rules=list(rules),
            current_stage=current_stage,
            current_question=current_question,
        )
        project = ResearchProject(
            project_id=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            topic=topic,
            user_question=user_question,
            uploaded_sources=list(uploaded_sources),
            created_at=timestamp,
            language=language,
            discussion_state=discussion_state,
        )
        self.store.save(project)
        return project

    def sync_workflow_tasks(
        self,
        state: DiscussionState,
        *,
        tasks: list[WorkflowTask],
        replace_for_source_kind: str = "assignment",
    ) -> None:
        preserved = [task for task in state.workflow_tasks if task.source_kind != replace_for_source_kind]
        preserved.extend(tasks)
        state.workflow_tasks = preserved

    def begin_stage(self, state: DiscussionState, *, stage_key: str, stage_label: str) -> WorkflowStageRecord:
        record = next((item for item in state.workflow_stage_records if item.stage_key == stage_key), None)
        if record is None:
            record = WorkflowStageRecord(stage_key=stage_key, stage_label=stage_label)
            state.workflow_stage_records.append(record)
        record.stage_label = stage_label
        record.status = TASK_STATUS_IN_PROGRESS
        record.started_at = record.started_at or _now_text()
        state.current_stage = stage_label
        state.status = TASK_STATUS_IN_PROGRESS
        return record

    def finish_stage(
        self,
        state: DiscussionState,
        *,
        stage_key: str,
        status: str = TASK_STATUS_COMPLETED,
        notes: str = "",
    ) -> WorkflowStageRecord | None:
        record = next((item for item in state.workflow_stage_records if item.stage_key == stage_key), None)
        if record is None:
            return None
        record.status = status
        record.completed_at = _now_text()
        if notes:
            record.notes = notes
        state.status = status
        return record

    def ensure_task(
        self,
        state: DiscussionState,
        *,
        task_id: str,
        title: str,
        description: str,
        owner_name: str,
        reviewer_name: str,
        round_index: int,
        source_kind: str,
    ) -> WorkflowTask:
        for task in state.workflow_tasks:
            if task.task_id == task_id:
                task.title = title
                task.description = description
                task.owner_name = owner_name
                task.reviewer_name = reviewer_name
                task.round_index = round_index
                task.source_kind = source_kind
                return task
        task = WorkflowTask(
            task_id=task_id,
            title=title,
            description=description,
            owner_name=owner_name,
            reviewer_name=reviewer_name,
            round_index=round_index,
            source_kind=source_kind,
        )
        state.workflow_tasks.append(task)
        return task

    def begin_task(
        self,
        state: DiscussionState,
        *,
        task_id: str,
        stage_label: str,
        question: str,
        round_index: int,
    ) -> WorkflowTask | None:
        task = self._task_by_id(state, task_id)
        if task is None:
            return None
        task.status = TASK_STATUS_IN_PROGRESS
        task.started_at = task.started_at or _now_text()
        state.current_stage = stage_label
        state.current_question = question
        state.current_round = round_index
        return task

    def complete_task(self, state: DiscussionState, *, task_id: str, notes: str = "") -> WorkflowTask | None:
        task = self._task_by_id(state, task_id)
        if task is None:
            return None
        task.status = TASK_STATUS_COMPLETED
        task.completed_at = _now_text()
        if notes:
            task.notes = notes
        return task

    def apply_log_entry(
        self,
        state: DiscussionState,
        *,
        entry: StructuredLogEntry,
        max_history_items: int,
        max_log_entries: int,
        max_evidence_cards: int,
    ) -> None:
        state.log_entries.append(entry)
        if len(state.log_entries) > max_log_entries:
            state.log_entries = state.log_entries[-max_log_entries:]

        _merge_unique(state.consensus_points, entry.consensus_add, limit=max_history_items)
        _merge_unique(state.risks_or_disagreements, entry.conflicts_add, limit=max_history_items)
        _merge_unique(state.open_questions, entry.open_questions_add, limit=max_history_items)
        _merge_unique(state.rejected_lines, entry.rejected_add, limit=max_history_items)
        _merge_unique(state.action_items, entry.action_items_add, limit=max_history_items)
        _remove_matching(state.risks_or_disagreements, entry.resolved_conflicts)
        _remove_matching(state.open_questions, entry.resolved_questions)

        evidence_by_id = {card.evidence_id: card for card in state.evidence_cards}
        for card in entry.evidence_add:
            if card.evidence_id in evidence_by_id:
                continue
            state.evidence_cards.append(card)
        if len(state.evidence_cards) > max_evidence_cards:
            state.evidence_cards = state.evidence_cards[-max_evidence_cards:]

    def create_checkpoint(
        self,
        state: DiscussionState,
        *,
        label: str,
        workpackage_index: int,
        max_checkpoints: int,
    ) -> Checkpoint:
        recent_entries = [entry for entry in state.log_entries if entry.workpackage_index == workpackage_index][-3:]
        summary_parts = [f"Completed stage: {label}"]
        if recent_entries:
            summary_parts.append("; ".join(entry.headline for entry in recent_entries if entry.headline))
        elif state.consensus_points:
            summary_parts.append("Recent consensus: " + "; ".join(state.consensus_points[-2:]))

        related_task_ids = [task.task_id for task in state.workflow_tasks if task.round_index == workpackage_index]
        checkpoint = Checkpoint(
            checkpoint_id=f"CP{len(state.checkpoints) + 1}",
            label=label,
            workpackage_index=workpackage_index,
            round_index=state.current_round,
            summary=_truncate_text(" ".join(part for part in summary_parts if part), 220),
            consensus_points=state.consensus_points[-4:],
            risks_or_disagreements=state.risks_or_disagreements[-3:],
            open_questions=state.open_questions[-3:],
            action_items=state.action_items[-4:],
            related_task_ids=related_task_ids,
        )
        state.checkpoints.append(checkpoint)
        if len(state.checkpoints) > max_checkpoints:
            state.checkpoints = state.checkpoints[-max_checkpoints:]
        return checkpoint

    def record_artifact(
        self,
        state: DiscussionState | None,
        *,
        artifact_type: str,
        title: str,
        path: str,
        preview: str = "",
        metadata: dict[str, str] | None = None,
    ) -> ArtifactRecord | None:
        if state is None:
            return None
        existing = next((artifact for artifact in state.generated_artifacts if artifact.path == path), None)
        if existing is not None:
            existing.preview = preview or existing.preview
            if metadata:
                existing.metadata.update(metadata)
            return existing
        artifact = ArtifactRecord(
            artifact_id=f"{artifact_type}_{len(state.generated_artifacts) + 1}",
            artifact_type=artifact_type,
            title=title,
            path=path,
            created_at=_now_text(),
            preview=preview,
            metadata=dict(metadata or {}),
        )
        state.generated_artifacts.append(artifact)
        return artifact

    def record_paper(self, state: DiscussionState | None, paper: PaperRecord) -> PaperRecord | None:
        if state is None:
            return None
        existing = next((item for item in state.literature_library if item.paper_id == paper.paper_id), None)
        if existing is not None:
            existing.title = paper.title
            existing.authors = list(paper.authors)
            existing.abstract = paper.abstract
            existing.categories = list(paper.categories)
            existing.published_at = paper.published_at
            existing.updated_at = paper.updated_at
            existing.entry_url = paper.entry_url
            existing.pdf_url = paper.pdf_url
            existing.local_pdf_path = paper.local_pdf_path
            existing.bibtex_key = paper.bibtex_key
            existing.bibtex_entry = paper.bibtex_entry
            existing.selection_reason = paper.selection_reason
            existing.source_provider = paper.source_provider
            return existing
        state.literature_library.append(paper)
        return paper

    def record_experiment_run(self, state: DiscussionState | None, run: ExperimentRunRecord) -> ExperimentRunRecord | None:
        if state is None:
            return None
        existing = next((item for item in state.experiment_runs if item.run_id == run.run_id), None)
        if existing is not None:
            existing.script_path = run.script_path
            existing.working_directory = run.working_directory
            existing.interpreter_path = run.interpreter_path
            existing.run_mode = run.run_mode
            existing.command = list(run.command)
            existing.compile_returncode = run.compile_returncode
            existing.runtime_returncode = run.runtime_returncode
            existing.log_path = run.log_path
            existing.stdout_excerpt = run.stdout_excerpt
            existing.stderr_excerpt = run.stderr_excerpt
            existing.status = run.status
            existing.authorized = run.authorized
            existing.created_at = run.created_at
            return existing
        state.experiment_runs.append(run)
        return run

    def record_approval(
        self,
        state: DiscussionState | None,
        *,
        approval_type: str,
        scope: str,
        granted: bool,
        details: str = "",
    ) -> ApprovalRecord | None:
        if state is None:
            return None
        approval = ApprovalRecord(
            approval_id=f"{approval_type}_{len(state.approval_records) + 1}",
            approval_type=approval_type,
            scope=scope,
            granted=granted,
            details=details,
            created_at=_now_text(),
        )
        state.approval_records.append(approval)
        return approval

    def record_execution_event(
        self,
        state: DiscussionState | None,
        *,
        event: dict[str, object],
        max_events: int = 256,
    ) -> None:
        if state is None:
            return
        state.execution_trace.append(dict(event))
        if len(state.execution_trace) > max_events:
            state.execution_trace = state.execution_trace[-max_events:]

    def update_token_usage(
        self,
        state: DiscussionState | None,
        *,
        usage: dict[str, object],
    ) -> None:
        if state is None:
            return
        state.token_usage.update(usage)

    def update_summary(self, state: DiscussionState, summary: str) -> None:
        state.summary = summary.strip()

    def _task_by_id(self, state: DiscussionState, task_id: str) -> WorkflowTask | None:
        return next((task for task in state.workflow_tasks if task.task_id == task_id), None)


def _merge_unique(target: list[str], incoming: list[str], *, limit: int) -> None:
    for item in incoming:
        if not item.strip():
            continue
        if item not in target:
            target.append(item)
    if len(target) > limit:
        del target[:-limit]


def _remove_matching(target: list[str], removals: list[str]) -> None:
    if not removals:
        return
    kept: list[str] = []
    for existing in target:
        lowered = existing.lower()
        if any(removal.lower() in lowered or lowered in removal.lower() for removal in removals):
            continue
        kept.append(existing)
    target[:] = kept


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
