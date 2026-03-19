from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import AttachmentPayload, AttachmentSnippet, DiscussionMessage, DiscussionResult, ProviderConfig
from .state import DiscussionState, ProjectStateManager, ResearchProject, TASK_STATUS_CANCELLED, TASK_STATUS_COMPLETED
from .workflow_config import WorkflowStageConfig, WorkflowTemplateConfig
from .workflow_graph import WorkflowGraph, build_workflow_graph


@dataclass
class DiscussionRunRecord:
    workpackage: Any
    owner: ProviderConfig
    reviewer: ProviderConfig | None
    primary_snippets: list[AttachmentSnippet]
    primary_message: DiscussionMessage
    review_snippets: list[AttachmentSnippet] = field(default_factory=list)
    review_message: DiscussionMessage | None = None


@dataclass
class WorkflowRuntimeContext:
    user_request: str
    attachments: list[AttachmentPayload]
    max_rounds: int
    generate_literature_review: bool
    local_execution_authorized: bool
    result: DiscussionResult
    project: ResearchProject
    state: DiscussionState
    state_manager: ProjectStateManager
    on_message: Callable[[DiscussionMessage], None] | None = None
    on_status: Callable[[str], None] | None = None
    should_cancel: Callable[[], bool] | None = None
    assignments_text: str = ""
    workpackages: list[Any] = field(default_factory=list)
    team_roster: str = ""
    literature_review_text: str = ""
    completed_rounds: int = 0
    successful_messages: list[DiscussionMessage] = field(default_factory=list)
    log_messages: list[DiscussionMessage] = field(default_factory=list)
    discussion_runs: list[DiscussionRunRecord] = field(default_factory=list)
    graph_trace: list[dict[str, Any]] = field(default_factory=list)
    workflow_graph: WorkflowGraph | None = None
    halted: bool = False


class ResearchDiscussionReviewWorkflow:
    def __init__(self, template: WorkflowTemplateConfig, graph: WorkflowGraph | None = None) -> None:
        self.template = template
        self.graph = graph or build_workflow_graph(template)

    def execute(self, runner: Any, context: WorkflowRuntimeContext) -> DiscussionResult:
        context.workflow_graph = self.graph
        previous_stage_key: str | None = None
        export_after_soft_cancel = False
        for stage in self.template.stages:
            if not stage.enabled or context.halted:
                continue
            if not self.graph.can_transition(previous_stage_key, stage.key):
                validation_error = {
                    "event": "graph_validation_error",
                    "source_stage_key": previous_stage_key or "",
                    "target_stage_key": stage.key,
                    "message": f"Workflow graph rejected transition from {previous_stage_key or '<start>'} to {stage.key}.",
                }
                context.graph_trace.append(validation_error)
                context.state_manager.record_execution_event(context.state, event=validation_error)
                raise ValueError(
                    f"Workflow graph rejected transition from {previous_stage_key or '<start>'} to {stage.key}."
                )
            start_event = {
                "event": "stage_start",
                "stage_key": stage.key,
                "stage_label": stage.label,
                "node_kind": self.graph.node_map.get(stage.key).node_kind if stage.key in self.graph.node_map else "stage",
                "previous_stage_key": previous_stage_key or "",
            }
            context.graph_trace.append(start_event)
            context.state_manager.record_execution_event(context.state, event=start_event)
            context.state_manager.begin_stage(context.state, stage_key=stage.key, stage_label=stage.label)
            if context.should_cancel is not None and context.should_cancel():
                if stage.key in {"run_experiment_cycle", "compile_latex_artifacts"}:
                    export_after_soft_cancel = True
                    context.state_manager.finish_stage(
                        context.state,
                        stage_key=stage.key,
                        status=TASK_STATUS_CANCELLED,
                        notes="User requested stop; skipped optional tool stage and continued to final exports.",
                    )
                    previous_stage_key = stage.key
                    continue
                if export_after_soft_cancel and stage.key in {"generate_meeting_notes", "generate_research_report"}:
                    pass
                else:
                    context.halted = True
                    context.state_manager.finish_stage(
                        context.state,
                        stage_key=stage.key,
                        status=TASK_STATUS_CANCELLED,
                        notes="Workflow cancelled before stage execution.",
                    )
                    break

            handler = getattr(runner, f"_workflow_stage_{stage.key}", None)
            if handler is None:
                raise AttributeError(f"Workflow stage handler is missing for '{stage.key}'.")

            notes = handler(context, stage) or ""
            status = TASK_STATUS_CANCELLED if context.halted else TASK_STATUS_COMPLETED
            context.state_manager.finish_stage(
                context.state,
                stage_key=stage.key,
                status=status,
                notes=notes,
            )

            if context.halted:
                break
            finish_event = {
                "event": "stage_finish",
                "stage_key": stage.key,
                "stage_label": stage.label,
                "status": status,
                "notes": notes,
            }
            context.graph_trace.append(finish_event)
            context.state_manager.record_execution_event(context.state, event=finish_event)
            previous_stage_key = stage.key
        return context.result
