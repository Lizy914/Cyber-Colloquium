from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .attachments import load_attachment
from .config import load_providers
from .models import AttachmentPayload, DiscussionMessage, DiscussionResult, ProviderConfig
from .orchestrator import DiscussionOrchestrator
from .workflow_config import WorkflowConfig, load_workflow_config
from .workflow_graph import build_workflow_graph, render_workflow_graph_mermaid, workflow_policy_snapshot


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_ID = "cyber_colloquium_benchmark_v1"
BENCHMARK_TASKS_DIR = Path("benchmarks") / "tasks"
BENCHMARK_RUNS_DIR = Path("benchmarks") / "runs"
REQUIRED_SLOT_ALIASES = {
    "summary": "summary",
    "consensus_points": "consensus_points",
    "consensus": "consensus_points",
    "disagreement_points": "risks_or_disagreements",
    "conflicts": "risks_or_disagreements",
    "risks_or_disagreements": "risks_or_disagreements",
    "open_questions": "open_questions",
    "action_items": "action_items",
}


@dataclass(frozen=True)
class BenchmarkTaskInputs:
    source_type: str
    topic: str
    pdf_paths: list[str] = field(default_factory=list)
    seed_summary: str | None = None
    user_question: str | None = None


@dataclass(frozen=True)
class BenchmarkExpectedOutputs:
    require_summary: bool = False
    require_meeting_notes: bool = True
    require_research_report: bool = True


@dataclass(frozen=True)
class BenchmarkScoringSpec:
    required_slots: list[str] = field(default_factory=list)
    reviewer_must_raise_critique: bool = False
    must_include_action_items: bool = False


@dataclass(frozen=True)
class BenchmarkMetadata:
    difficulty: str = "medium"
    domain: str = "general_research"
    split: str = "train"
    source_origin: str = "handcrafted"
    benchmark_level: str = "workflow_quality"
    category: str = ""


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_type: str
    title: str
    inputs: BenchmarkTaskInputs
    expected_outputs: BenchmarkExpectedOutputs
    scoring: BenchmarkScoringSpec
    metadata: BenchmarkMetadata
    schema_version: int = BENCHMARK_SCHEMA_VERSION
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload


@dataclass(frozen=True)
class BenchmarkMetrics:
    output_presence: dict[str, bool]
    required_slot_presence: dict[str, bool]
    checkpoint_count: int
    token_usage: dict[str, Any]
    api_calls: int
    node_steps: int
    round_count: int
    human_intervention_count: int
    graph_transition_valid: bool
    execution_event_count: int
    role_execution_trace: list[dict[str, Any]]
    workflow_stage_trace: list[dict[str, Any]]
    reviewer_comments: list[str]


@dataclass(frozen=True)
class WorkflowObjectiveWeights:
    quality_weight: float = 1.0
    cost_weight: float = 0.2
    latency_weight: float = 0.15
    human_intervention_weight: float = 0.1
    failure_weight: float = 0.8
    stability_weight: float = 0.2


@dataclass(frozen=True)
class BenchmarkRunResult:
    benchmark_id: str
    task_id: str
    policy_version: str
    benchmark_level: str
    benchmark_category: str
    config_snapshot: dict[str, Any]
    run_id: str
    success: bool
    completion_score: float
    consistency_score: float
    criticality_score: float
    usability_score: float
    structure_score: float
    overall_score: float
    objective_loss: float
    objective_breakdown: dict[str, float]
    token_usage: dict[str, Any]
    api_calls: int
    node_steps: int
    round_count: int
    checkpoint_count: int
    human_intervention_count: int
    graph_transition_valid: bool
    execution_event_count: int
    output_presence: dict[str, bool]
    required_slot_presence: dict[str, bool]
    failure_reason: str | None
    role_execution_trace: list[dict[str, Any]]
    workflow_stage_trace: list[dict[str, Any]]
    reviewer_comments: list[str]
    meeting_notes_path: str | None = None
    research_report_path: str | None = None
    literature_review_path: str | None = None
    execution_trace_path: str | None = None
    workflow_graph_path: str | None = None
    workflow_mermaid_path: str | None = None
    policy_snapshot_path: str | None = None
    result_path: str | None = None


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    benchmark_id: str
    policy_version: str
    suite_run_id: str
    task_count: int
    success_count: int
    success_rate: float
    average_completion_score: float
    average_consistency_score: float
    average_criticality_score: float
    average_usability_score: float
    average_structure_score: float
    average_overall_score: float
    average_objective_loss: float
    results_path: str
    task_results: list[BenchmarkRunResult]


EvaluationExecutor = Callable[
    [BenchmarkTask, WorkflowConfig, list[ProviderConfig], list[AttachmentPayload], str, bool],
    DiscussionResult,
]


def load_benchmark_task(path: Path) -> BenchmarkTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    inputs_raw = _coerce_mapping(raw.get("inputs"))
    expected_raw = _coerce_mapping(raw.get("expected_outputs"))
    scoring_raw = _coerce_mapping(raw.get("scoring"))
    metadata_raw = _coerce_mapping(raw.get("metadata"))
    return BenchmarkTask(
        schema_version=int(raw.get("schema_version") or BENCHMARK_SCHEMA_VERSION),
        task_id=str(raw.get("task_id") or path.stem).strip() or path.stem,
        task_type=str(raw.get("task_type") or "open_research_discussion").strip() or "open_research_discussion",
        title=str(raw.get("title") or path.stem).strip() or path.stem,
        inputs=BenchmarkTaskInputs(
            source_type=str(inputs_raw.get("source_type") or "topic_only").strip() or "topic_only",
            topic=str(inputs_raw.get("topic") or "").strip(),
            pdf_paths=[str(item).strip() for item in inputs_raw.get("pdf_paths", []) if str(item).strip()],
            seed_summary=_coerce_optional_text(inputs_raw.get("seed_summary")),
            user_question=_coerce_optional_text(inputs_raw.get("user_question")),
        ),
        expected_outputs=BenchmarkExpectedOutputs(
            require_summary=bool(expected_raw.get("require_summary", False)),
            require_meeting_notes=bool(expected_raw.get("require_meeting_notes", True)),
            require_research_report=bool(expected_raw.get("require_research_report", True)),
        ),
        scoring=BenchmarkScoringSpec(
            required_slots=[str(item).strip() for item in scoring_raw.get("required_slots", []) if str(item).strip()],
            reviewer_must_raise_critique=bool(scoring_raw.get("reviewer_must_raise_critique", False)),
            must_include_action_items=bool(scoring_raw.get("must_include_action_items", False)),
        ),
        metadata=BenchmarkMetadata(
            difficulty=str(metadata_raw.get("difficulty") or "medium").strip() or "medium",
            domain=str(metadata_raw.get("domain") or "general_research").strip() or "general_research",
            split=str(metadata_raw.get("split") or path.parent.name or "train").strip() or "train",
            source_origin=str(metadata_raw.get("source_origin") or "handcrafted").strip() or "handcrafted",
            benchmark_level=str(metadata_raw.get("benchmark_level") or "workflow_quality").strip() or "workflow_quality",
            category=str(metadata_raw.get("category") or "").strip(),
        ),
        source_path=str(path),
    )


def discover_benchmark_tasks(tasks_root: Path = BENCHMARK_TASKS_DIR, split: str | None = None) -> list[BenchmarkTask]:
    if not tasks_root.exists():
        return []
    pattern = "*.json"
    paths = sorted(tasks_root.rglob(pattern))
    tasks = [load_benchmark_task(path) for path in paths]
    if split is not None:
        tasks = [task for task in tasks if task.metadata.split == split]
    return tasks


class WorkflowEvaluationRunner:
    def __init__(
        self,
        *,
        workflow_config: WorkflowConfig | None = None,
        providers: list[ProviderConfig] | None = None,
        benchmark_id: str = DEFAULT_BENCHMARK_ID,
        policy_version: str = "",
        output_root: Path = BENCHMARK_RUNS_DIR,
        executor: EvaluationExecutor | None = None,
        objective_weights: WorkflowObjectiveWeights | None = None,
    ) -> None:
        self.workflow_config = workflow_config or load_workflow_config()
        self.providers = providers if providers is not None else load_providers()
        self.benchmark_id = benchmark_id
        self.policy_version = policy_version.strip() or _default_policy_version()
        self.output_root = output_root
        self.executor = executor or self._default_execute
        self.objective_weights = objective_weights or WorkflowObjectiveWeights()

    def run_task(self, task: BenchmarkTask) -> BenchmarkRunResult:
        run_id = _timestamp_id("task")
        task_dir = self.output_root / self.policy_version / task.metadata.split / task.task_id / run_id
        task_dir.mkdir(parents=True, exist_ok=True)

        user_request = self._build_user_request(task)
        attachments = self._build_attachments(task)
        generate_literature_review = task.inputs.source_type in {"pdf", "pdf_plus_summary"}
        failure_reason: str | None = None

        try:
            result = self.executor(
                task,
                self.workflow_config,
                self.providers,
                attachments,
                user_request,
                generate_literature_review,
            )
        except Exception as exc:  # noqa: BLE001
            result = DiscussionResult(cancelled=True)
            failure_reason = str(exc)

        metrics = _extract_metrics(task, result, user_request, attachments)
        if failure_reason is None and not _outputs_satisfy_expectations(task, metrics.output_presence):
            failure_reason = "Required outputs were not generated."
        scores = _score_task_run(
            task,
            result,
            metrics,
            objective_weights=self.objective_weights,
            failure_reason=failure_reason,
        )

        artifact_paths = self._persist_task_run(
            task_dir=task_dir,
            task=task,
            result=result,
            metrics=metrics,
            run_id=run_id,
        )
        success = failure_reason is None and not result.cancelled
        run_result = BenchmarkRunResult(
            benchmark_id=self.benchmark_id,
            task_id=task.task_id,
            policy_version=self.policy_version,
            benchmark_level=task.metadata.benchmark_level,
            benchmark_category=task.metadata.category or task.task_type,
            config_snapshot=asdict(self.workflow_config),
            run_id=run_id,
            success=success,
            completion_score=scores["completion_score"],
            consistency_score=scores["consistency_score"],
            criticality_score=scores["criticality_score"],
            usability_score=scores["usability_score"],
            structure_score=scores["structure_score"],
            overall_score=scores["overall_score"],
            objective_loss=scores["objective_loss"],
            objective_breakdown=scores["objective_breakdown"],
            token_usage=metrics.token_usage,
            api_calls=metrics.api_calls,
            node_steps=metrics.node_steps,
            round_count=metrics.round_count,
            checkpoint_count=metrics.checkpoint_count,
            human_intervention_count=metrics.human_intervention_count,
            graph_transition_valid=metrics.graph_transition_valid,
            execution_event_count=metrics.execution_event_count,
            output_presence=metrics.output_presence,
            required_slot_presence=metrics.required_slot_presence,
            failure_reason=failure_reason,
            role_execution_trace=metrics.role_execution_trace,
            workflow_stage_trace=metrics.workflow_stage_trace,
            reviewer_comments=metrics.reviewer_comments,
            meeting_notes_path=artifact_paths.get("meeting_notes_path"),
            research_report_path=artifact_paths.get("research_report_path"),
            literature_review_path=artifact_paths.get("literature_review_path"),
            execution_trace_path=artifact_paths.get("execution_trace_path"),
            workflow_graph_path=artifact_paths.get("workflow_graph_path"),
            workflow_mermaid_path=artifact_paths.get("workflow_mermaid_path"),
            policy_snapshot_path=artifact_paths.get("policy_snapshot_path"),
            result_path=artifact_paths.get("result_path"),
        )
        result_path = Path(run_result.result_path or task_dir / "benchmark_result.json")
        result_path.write_text(json.dumps(asdict(run_result), ensure_ascii=False, indent=2), encoding="utf-8")
        return run_result

    def run_suite(self, tasks: list[BenchmarkTask]) -> BenchmarkSuiteResult:
        suite_run_id = _timestamp_id("suite")
        task_results = [self.run_task(task) for task in tasks]
        task_count = len(task_results)
        success_count = sum(1 for item in task_results if item.success)
        suite_summary = BenchmarkSuiteResult(
            benchmark_id=self.benchmark_id,
            policy_version=self.policy_version,
            suite_run_id=suite_run_id,
            task_count=task_count,
            success_count=success_count,
            success_rate=_safe_average([1.0 if item.success else 0.0 for item in task_results]),
            average_completion_score=_safe_average([item.completion_score for item in task_results]),
            average_consistency_score=_safe_average([item.consistency_score for item in task_results]),
            average_criticality_score=_safe_average([item.criticality_score for item in task_results]),
            average_usability_score=_safe_average([item.usability_score for item in task_results]),
            average_structure_score=_safe_average([item.structure_score for item in task_results]),
            average_overall_score=_safe_average([item.overall_score for item in task_results]),
            average_objective_loss=_safe_average([item.objective_loss for item in task_results]),
            results_path=str(self.output_root / self.policy_version / f"{suite_run_id}_suite_results.json"),
            task_results=task_results,
        )
        results_path = Path(suite_summary.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(asdict(suite_summary), ensure_ascii=False, indent=2), encoding="utf-8")
        return suite_summary

    def _default_execute(
        self,
        task: BenchmarkTask,
        workflow_config: WorkflowConfig,
        providers: list[ProviderConfig],
        attachments: list[AttachmentPayload],
        user_request: str,
        generate_literature_review: bool,
    ) -> DiscussionResult:
        del task
        orchestrator = DiscussionOrchestrator(providers, workflow_config=workflow_config)
        return orchestrator.run_discussion(
            user_request=user_request,
            attachments=attachments,
            generate_literature_review=generate_literature_review,
        )

    def _build_user_request(self, task: BenchmarkTask) -> str:
        lines = []
        if task.inputs.topic:
            lines.append(task.inputs.topic)
        if task.inputs.user_question and task.inputs.user_question != task.inputs.topic:
            lines.append(task.inputs.user_question)
        if task.inputs.seed_summary:
            lines.append(f"[Seed Summary]\n{task.inputs.seed_summary.strip()}")
        return "\n\n".join(part for part in lines if part.strip()).strip() or task.title

    def _build_attachments(self, task: BenchmarkTask) -> list[AttachmentPayload]:
        attachments: list[AttachmentPayload] = []
        task_base = Path(task.source_path).parent if task.source_path else Path.cwd()
        for raw_path in task.inputs.pdf_paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = task_base / candidate
            attachments.append(load_attachment(str(candidate)))
        if task.inputs.seed_summary:
            attachments.append(
                AttachmentPayload(
                    path=task_base / f"{task.task_id}_seed_summary.md",
                    kind="text",
                    content=task.inputs.seed_summary.strip(),
                    display_name=f"{task.task_id}_seed_summary.md",
                )
            )
        return attachments

    def _persist_task_run(
        self,
        *,
        task_dir: Path,
        task: BenchmarkTask,
        result: DiscussionResult,
        metrics: BenchmarkMetrics,
        run_id: str,
    ) -> dict[str, str]:
        config_path = task_dir / "config_snapshot.json"
        config_path.write_text(json.dumps(asdict(self.workflow_config), ensure_ascii=False, indent=2), encoding="utf-8")
        policy_snapshot_path = task_dir / "policy_snapshot.json"
        policy_snapshot_path.write_text(json.dumps(workflow_policy_snapshot(self.workflow_config), ensure_ascii=False, indent=2), encoding="utf-8")

        task_snapshot_path = task_dir / "task_snapshot.json"
        task_snapshot_path.write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        workflow_graph = build_workflow_graph(self.workflow_config.workflow_template)
        workflow_graph_path = task_dir / "workflow_graph.json"
        workflow_graph_path.write_text(json.dumps(workflow_graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        workflow_mermaid_path = task_dir / "workflow_graph.mmd"
        workflow_mermaid_path.write_text(render_workflow_graph_mermaid(workflow_graph), encoding="utf-8")

        trace_payload = {
            "benchmark_id": self.benchmark_id,
            "task_id": task.task_id,
            "run_id": run_id,
            "policy_version": self.policy_version,
            "workflow_stage_trace": metrics.workflow_stage_trace,
            "role_execution_trace": metrics.role_execution_trace,
            "execution_trace": result.meeting_state.execution_trace if result.meeting_state is not None else [],
        }
        trace_path = task_dir / "execution_trace.json"
        trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        literature_path: Path | None = None
        if result.literature_review.strip():
            literature_path = task_dir / "literature_review.md"
            literature_path.write_text(result.literature_review, encoding="utf-8")

        meeting_notes_path = task_dir / "meeting_notes.md"
        meeting_notes_path.write_text(result.meeting_minutes or "", encoding="utf-8")
        report_path = task_dir / "research_report.md"
        report_path.write_text(result.final_summary or "", encoding="utf-8")

        return {
            "config_path": str(config_path),
            "policy_snapshot_path": str(policy_snapshot_path),
            "task_snapshot_path": str(task_snapshot_path),
            "execution_trace_path": str(trace_path),
            "workflow_graph_path": str(workflow_graph_path),
            "workflow_mermaid_path": str(workflow_mermaid_path),
            "literature_review_path": str(literature_path) if literature_path is not None else "",
            "meeting_notes_path": str(meeting_notes_path),
            "research_report_path": str(report_path),
            "result_path": str(task_dir / "benchmark_result.json"),
        }


def _extract_metrics(
    task: BenchmarkTask,
    result: DiscussionResult,
    user_request: str,
    attachments: list[AttachmentPayload],
) -> BenchmarkMetrics:
    state = result.meeting_state
    messages = result.messages or []
    output_presence = {
        "summary": bool(result.literature_review.strip() or (state.summary.strip() if state is not None else "")),
        "meeting_notes": bool(result.meeting_minutes.strip()),
        "research_report": bool(result.final_summary.strip()),
        "literature_review": bool(result.literature_review.strip()),
    }
    required_slot_presence = {
        slot: _slot_present(state, slot)
        for slot in task.scoring.required_slots
    }
    role_execution_trace = [
        {
            "speaker": message.speaker,
            "duty": message.duty,
            "stage": message.stage or "discussion",
            "round_index": message.round_index,
            "content_chars": len(message.content),
        }
        for message in messages
    ]
    workflow_stage_trace = [
        {
            "stage_key": record.stage_key,
            "stage_label": record.stage_label,
            "status": record.status,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
        }
        for record in (state.workflow_stage_records if state is not None else [])
    ]
    reviewer_comments = [
        _truncate_text(message.content, 420)
        for message in messages
        if "review" in (message.stage or "").lower() or "review" in (message.duty or "").lower()
    ]
    checkpoint_count = len(state.checkpoints) if state is not None else 0
    round_count = max((message.round_index for message in messages), default=(state.current_round if state is not None else 0))
    node_steps = len(workflow_stage_trace) + len(state.workflow_tasks if state is not None else [])
    human_intervention_count = len(state.approval_records) if state is not None else 0
    execution_trace = list(state.execution_trace) if state is not None else []
    graph_transition_valid = not any(
        isinstance(event, dict) and event.get("event") == "graph_validation_error"
        for event in execution_trace
    )
    token_usage = {
        **(state.token_usage if state is not None else {}),
        "available": False,
        "estimated_input_chars": len(user_request) + sum(len(item.content) for item in attachments if item.kind != "image"),
        "estimated_output_chars": sum(len(message.content) for message in messages) + len(result.meeting_minutes) + len(result.final_summary),
        "message_count": len(messages),
    }
    return BenchmarkMetrics(
        output_presence=output_presence,
        required_slot_presence=required_slot_presence,
        checkpoint_count=checkpoint_count,
        token_usage=token_usage,
        api_calls=len(messages),
        node_steps=node_steps,
        round_count=round_count,
        human_intervention_count=human_intervention_count,
        graph_transition_valid=graph_transition_valid,
        execution_event_count=len(execution_trace),
        role_execution_trace=role_execution_trace,
        workflow_stage_trace=workflow_stage_trace,
        reviewer_comments=reviewer_comments,
    )


def _score_task_run(
    task: BenchmarkTask,
    result: DiscussionResult,
    metrics: BenchmarkMetrics,
    *,
    objective_weights: WorkflowObjectiveWeights,
    failure_reason: str | None,
) -> dict[str, float | dict[str, float]]:
    state = result.meeting_state
    expected_checks: list[bool] = []
    if task.expected_outputs.require_summary:
        expected_checks.append(metrics.output_presence["summary"])
    if task.expected_outputs.require_meeting_notes:
        expected_checks.append(metrics.output_presence["meeting_notes"])
    if task.expected_outputs.require_research_report:
        expected_checks.append(metrics.output_presence["research_report"])
    expected_checks.extend(metrics.required_slot_presence.values())
    completion_score = _safe_average([1.0 if item else 0.0 for item in expected_checks]) if expected_checks else 1.0

    consistency_signals = [
        metrics.output_presence["meeting_notes"],
        metrics.output_presence["research_report"],
        bool(state is not None and state.summary.strip()),
        bool(metrics.workflow_stage_trace),
    ]
    consistency_score = _safe_average([1.0 if item else 0.0 for item in consistency_signals])

    critique_signals = [
        bool((state.risks_or_disagreements if state is not None else [])),
        bool((state.open_questions if state is not None else [])),
    ]
    if task.scoring.reviewer_must_raise_critique:
        critique_signals.append(bool(metrics.reviewer_comments))
    criticality_score = _safe_average([1.0 if item else 0.0 for item in critique_signals])

    usability_signals = [
        metrics.output_presence["meeting_notes"],
        metrics.output_presence["research_report"],
        bool((state.action_items if state is not None else [])),
        metrics.checkpoint_count > 0,
    ]
    if task.scoring.must_include_action_items:
        usability_signals.append(bool((state.action_items if state is not None else [])))
    usability_score = _safe_average([1.0 if item else 0.0 for item in usability_signals])

    structure_signals = [
        _looks_structured(result.meeting_minutes),
        _looks_structured(result.final_summary),
        bool(metrics.required_slot_presence) and all(metrics.required_slot_presence.values()) if metrics.required_slot_presence else True,
    ]
    structure_score = _safe_average([1.0 if item else 0.0 for item in structure_signals])

    overall_score = _safe_average(
        [
            completion_score,
            consistency_score,
            criticality_score,
            usability_score,
            structure_score,
        ]
    )
    objective_breakdown = _compute_objective_breakdown(
        quality_score=overall_score,
        metrics=metrics,
        failure_reason=failure_reason,
        objective_weights=objective_weights,
    )
    return {
        "completion_score": completion_score,
        "consistency_score": consistency_score,
        "criticality_score": criticality_score,
        "usability_score": usability_score,
        "structure_score": structure_score,
        "overall_score": overall_score,
        "objective_loss": objective_breakdown["objective_loss"],
        "objective_breakdown": objective_breakdown,
    }


def _compute_objective_breakdown(
    *,
    quality_score: float,
    metrics: BenchmarkMetrics,
    failure_reason: str | None,
    objective_weights: WorkflowObjectiveWeights,
) -> dict[str, float]:
    estimated_chars = int(metrics.token_usage.get("estimated_input_chars", 0)) + int(metrics.token_usage.get("estimated_output_chars", 0))
    cost_penalty = min(1.0, (estimated_chars / 25000.0) + (metrics.api_calls / 20.0)) / 2.0
    latency_penalty = min(1.0, metrics.node_steps / 24.0)
    human_penalty = min(1.0, metrics.human_intervention_count / 4.0)
    failure_penalty = 1.0 if failure_reason else 0.0
    stability_penalty = 0.0
    if not metrics.graph_transition_valid:
        stability_penalty += 0.5
    if metrics.required_slot_presence and not all(metrics.required_slot_presence.values()):
        stability_penalty += 0.25
    if metrics.execution_event_count == 0:
        stability_penalty += 0.25
    stability_penalty = min(1.0, stability_penalty)
    objective_loss = round(
        (-objective_weights.quality_weight * quality_score)
        + (objective_weights.cost_weight * cost_penalty)
        + (objective_weights.latency_weight * latency_penalty)
        + (objective_weights.human_intervention_weight * human_penalty)
        + (objective_weights.failure_weight * failure_penalty)
        + (objective_weights.stability_weight * stability_penalty),
        4,
    )
    return {
        "quality_score": round(quality_score, 4),
        "cost_penalty": round(cost_penalty, 4),
        "latency_penalty": round(latency_penalty, 4),
        "human_intervention_penalty": round(human_penalty, 4),
        "failure_penalty": round(failure_penalty, 4),
        "stability_penalty": round(stability_penalty, 4),
        "objective_loss": objective_loss,
    }


def _outputs_satisfy_expectations(task: BenchmarkTask, output_presence: dict[str, bool]) -> bool:
    checks = []
    if task.expected_outputs.require_summary:
        checks.append(output_presence.get("summary", False))
    if task.expected_outputs.require_meeting_notes:
        checks.append(output_presence.get("meeting_notes", False))
    if task.expected_outputs.require_research_report:
        checks.append(output_presence.get("research_report", False))
    return all(checks) if checks else True


def _slot_present(state: Any, slot_name: str) -> bool:
    if state is None:
        return False
    canonical = REQUIRED_SLOT_ALIASES.get(slot_name, slot_name)
    value = getattr(state, canonical, None)
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _looks_structured(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading_count = sum(1 for line in lines if line.startswith("#") or line.startswith("##"))
    return bool(lines) and (heading_count >= 1 or len(lines) >= 4)


def _default_policy_version() -> str:
    return f"workflow_policy_{datetime.now().strftime('%Y%m%d')}"


def _timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _safe_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _truncate_text(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."


def _coerce_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Cyber Colloquium benchmark tasks against a workflow config.")
    parser.add_argument("--tasks-root", type=Path, default=BENCHMARK_TASKS_DIR, help="Directory containing benchmark task JSON files.")
    parser.add_argument("--split", default="", help="Optional benchmark split to run, such as train, dev, or holdout.")
    parser.add_argument("--workflow-config", type=Path, default=None, help="Optional workflow config path.")
    parser.add_argument("--policy-version", default="", help="Label used to store benchmark outputs for this config.")
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_RUNS_DIR, help="Directory used to persist benchmark outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of tasks to run.")
    parser.add_argument("--quality-weight", type=float, default=1.0, help="Objective weight for output quality.")
    parser.add_argument("--cost-weight", type=float, default=0.2, help="Objective weight for workflow cost.")
    parser.add_argument("--latency-weight", type=float, default=0.15, help="Objective weight for latency / step count.")
    parser.add_argument("--human-weight", type=float, default=0.1, help="Objective weight for human intervention.")
    parser.add_argument("--failure-weight", type=float, default=0.8, help="Objective weight for failure penalty.")
    parser.add_argument("--stability-weight", type=float, default=0.2, help="Objective weight for stability penalty.")
    args = parser.parse_args(argv)

    workflow_config = load_workflow_config(args.workflow_config) if args.workflow_config is not None else load_workflow_config()
    tasks = discover_benchmark_tasks(args.tasks_root, split=args.split or None)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No benchmark tasks were found.")
        return 1

    runner = WorkflowEvaluationRunner(
        workflow_config=workflow_config,
        policy_version=args.policy_version,
        output_root=args.output_root,
        objective_weights=WorkflowObjectiveWeights(
            quality_weight=args.quality_weight,
            cost_weight=args.cost_weight,
            latency_weight=args.latency_weight,
            human_intervention_weight=args.human_weight,
            failure_weight=args.failure_weight,
            stability_weight=args.stability_weight,
        ),
    )
    suite_result = runner.run_suite(tasks)
    print(f"Suite run saved to: {suite_result.results_path}")
    print(f"Tasks: {suite_result.task_count} | Success rate: {suite_result.success_rate:.2f} | Avg overall score: {suite_result.average_overall_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
